"""Checkpoints, and getting them off the machine that made them.

The repository's rule is that a run checkpoints from the first step and resumes from
object storage. The reason is money: the GPU work is costed on spot instances, a spot
instance is reclaimed with about thirty seconds of warning, and a run that keeps its only
copy of the weights on the instance's own disk loses everything it has done. A run that
uploads from step one loses at most `save_steps` steps.

So this module is two small things. A store that knows where checkpoints go and which one
is newest, and a syncer that moves a directory there. The syncer is an interface with a
local implementation, because the object store is not chosen until the GPU provider is,
and because a test for "resume finds the newest checkpoint" should not need a network.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final, Protocol

#: What the trainer names a checkpoint directory. The number is the optimiser step.
CHECKPOINT = re.compile(r"^checkpoint-(\d+)$")

#: Written beside the weights when an upload finishes. A checkpoint without it was
#: interrupted mid-upload, and resuming from half a file is worse than resuming from the
#: step before it.
COMPLETE: Final = ".complete"

#: Seconds to wait before each retry of a store command that failed. An object store's API
#: fails now and then for reasons of its own: on 2026-09-25 Runpod's answered one upload
#: with AccessDenied, "failed to fetch user keys ... context deadline exceeded", its own
#: authentication timing out, and the run died at step 250 of 634 with two good hours on
#: the card. A command that still fails after these is a real failure and stops the run.
BACKOFF: Final = (10.0, 30.0, 90.0)


def step_of(path: Path) -> int | None:
    match = CHECKPOINT.match(path.name)
    return int(match.group(1)) if match else None


class Syncer(Protocol):
    """Moves a checkpoint directory to and from wherever checkpoints are kept."""

    def push(self, local: Path, name: str) -> None: ...
    def pull(self, name: str, local: Path) -> None: ...
    def names(self) -> list[str]: ...
    def remove(self, name: str) -> None: ...


class LocalSyncer:
    """Another directory. Used by the tests, and on a machine with a mounted volume."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def push(self, local: Path, name: str) -> None:
        target = self.root / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(local, target)
        (target / COMPLETE).write_text("", encoding="utf-8")

    def pull(self, name: str, local: Path) -> None:
        source = self.root / name
        if not (source / COMPLETE).is_file():
            raise FileNotFoundError(f"{name} has no {COMPLETE}: it was never finished uploading")
        if local.exists():
            shutil.rmtree(local)
        shutil.copytree(source, local)

    def names(self) -> list[str]:
        return sorted(
            d.name for d in self.root.iterdir() if d.is_dir() and (d / COMPLETE).is_file()
        )

    def remove(self, name: str) -> None:
        target = self.root / name
        (target / COMPLETE).unlink(missing_ok=True)
        shutil.rmtree(target, ignore_errors=True)


class CommandSyncer:
    """An object store, through whatever command line the provider gives us.

    `aws s3 sync`, `gcloud storage rsync`, `rclone sync`: all three take a source and a
    destination in that order, so the command is configuration rather than code, and the
    provider decision in docs/gpu-prices.md does not become a code change here.
    """

    def __init__(
        self,
        uri: str,
        *,
        argv: Sequence[str],
        list_argv: Sequence[str],
        remove_argv: Sequence[str] = (),
        empty_listing: int | None = None,
        retries: int = len(BACKOFF),
        pause: Callable[[float], None] = time.sleep,
    ) -> None:
        self.uri = uri.rstrip("/")
        self.argv = tuple(argv)
        self.list_argv = tuple(list_argv)
        #: Deletes a prefix, like `aws s3 rm --recursive`. Empty: this store keeps everything.
        self.remove_argv = tuple(remove_argv)
        #: The exit code the list command gives for "nothing here", when it is not zero.
        #: `aws s3 ls` exits 1, silently, which is every new run's first question.
        self.empty_listing = empty_listing
        #: How many times a failed command is tried again, waiting `BACKOFF` between.
        self.retries = retries
        self.pause = pause

    def _run(self, *args: str, empty_ok: bool = False) -> str:
        """Run a store command, retrying a failure. Every command here is safe to repeat:
        a sync sends only what is missing, and a listing or a removal repeated changes
        nothing more."""
        for attempt in range(self.retries + 1):
            # The argv is configuration from boundary.yaml's sibling, not user input.
            finished = subprocess.run([*args], capture_output=True, text=True, check=False)
            if (
                empty_ok
                and finished.returncode == self.empty_listing
                and not finished.stdout.strip()
                and not finished.stderr.strip()
            ):
                return ""
            if finished.returncode == 0:
                return finished.stdout
            if attempt < self.retries:
                self.pause(BACKOFF[min(attempt, len(BACKOFF) - 1)])
        raise RuntimeError(
            f"{' '.join(args[:2])} failed with {finished.returncode} after "
            f"{self.retries + 1} tries: {finished.stderr.strip()[:400]}"
        )

    def push(self, local: Path, name: str) -> None:
        """Upload, then mark. A sync copies files in no promised order, so a marker sent with
        the weights can arrive before them, and a machine lost mid-upload would leave a
        checkpoint that says it is complete and is not. The second sync sends only the
        marker, because everything else is already there."""
        marker = local / COMPLETE
        marker.unlink(missing_ok=True)
        self._run(*self.argv, str(local), f"{self.uri}/{name}")
        marker.write_text("", encoding="utf-8")
        self._run(*self.argv, str(local), f"{self.uri}/{name}")

    def pull(self, name: str, local: Path) -> None:
        local.mkdir(parents=True, exist_ok=True)
        self._run(*self.argv, f"{self.uri}/{name}", str(local))
        if not (local / COMPLETE).is_file():
            raise FileNotFoundError(f"{name} has no {COMPLETE}: it was never finished uploading")

    def names(self) -> list[str]:
        """The checkpoints whose upload finished: those with a marker.

        `list_argv` lists recursively, one object per line with its path last, which is
        what `aws s3 ls --recursive`, `gcloud storage ls -r` and `rclone lsf -R` all print.
        A checkpoint without its marker was cut off mid-upload and is not offered, so a
        resume falls back to the one before it rather than failing on it.
        """
        out = self._run(*self.list_argv, self.uri + "/", empty_ok=True)
        found = set()
        for line in out.splitlines():
            parts = line.strip().rsplit(" ", 1)[-1].strip("/").split("/")
            if len(parts) >= 2 and parts[-1] == COMPLETE and CHECKPOINT.match(parts[-2]):
                found.add(parts[-2])
        return sorted(found, key=lambda n: int(n.split("-")[1]))

    def remove(self, name: str) -> None:
        """The marker first, so a deletion cut off halfway leaves a checkpoint that no
        longer claims to be whole, rather than one that claims it and is not."""
        if not self.remove_argv:
            return
        self._run(*self.remove_argv, f"{self.uri}/{name}/{COMPLETE}")
        self._run(*self.remove_argv, f"{self.uri}/{name}")


def runpod_s3(uri: str, datacenter: str) -> CommandSyncer:
    """A Runpod network volume, reached over its S3-compatible API.

    Network volumes mount only on Secure Cloud pods, so a Community Cloud pod, the cheaper
    tier training runs on, reaches one this way instead. `uri` is `s3://<volume id>/<path>`,
    and the credentials are the account's S3 API key, in the environment as
    AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY, where the AWS CLI looks for them.
    """
    endpoint = (
        "--region",
        datacenter,
        "--endpoint-url",
        f"https://s3api-{datacenter.lower()}.runpod.io/",
    )
    return CommandSyncer(
        uri,
        argv=("aws", "s3", "sync", *endpoint),
        list_argv=("aws", "s3", "ls", "--recursive", *endpoint),
        remove_argv=("aws", "s3", "rm", "--recursive", *endpoint),
        empty_listing=1,
    )


class CheckpointStore:
    """Where a run's checkpoints live, locally and remotely."""

    def __init__(
        self, local: Path, syncer: Syncer | None = None, *, keep: int | None = None
    ) -> None:
        self.local = local
        self.local.mkdir(parents=True, exist_ok=True)
        self.syncer = syncer
        #: How many complete checkpoints to leave in the store; None keeps them all. A
        #: resume needs only the newest, and at 13 checkpoints a run and 30 runs, keeping
        #: every one is more than the 50 GB volume holds (docs/training.md).
        if keep is not None and keep < 1:
            raise ValueError("keep at least the newest checkpoint")
        self.keep = keep

    def save(self, directory: Path) -> None:
        """Called after the trainer writes `checkpoint-N`. Uploads it if there is anywhere to,
        and only once it is whole in the store are older ones removed from there."""
        if step_of(directory) is None:
            raise ValueError(f"{directory.name} is not a checkpoint directory")
        if self.syncer is None:
            return
        self.syncer.push(directory, directory.name)
        if self.keep is not None:
            names = self.syncer.names()
            for name in names[: -self.keep]:
                self.syncer.remove(name)

    def latest(self) -> Path | None:
        """The newest complete checkpoint, fetched from the store if it is only there.

        Local first, because an instance that is still alive has it on disk and a download
        is minutes of GPU rent. Remote second, which is the case that matters: a new
        instance after a reclaim has an empty disk.
        """
        local = max(
            (d for d in self.local.iterdir() if d.is_dir() and step_of(d) is not None),
            key=lambda d: step_of(d) or 0,
            default=None,
        )
        local_step = step_of(local) or 0 if local is not None else -1
        names = self.syncer.names() if self.syncer is not None else []
        remote_step = max((int(n.split("-")[1]) for n in names), default=-1)
        if self.syncer is not None and remote_step > local_step:
            target = self.local / f"checkpoint-{remote_step}"
            if not target.is_dir():
                self.syncer.pull(target.name, target)
            return target
        return local

    def resume_step(self) -> int:
        """The step a run would resume from. Zero means it starts at the beginning."""
        latest = self.latest()
        return step_of(latest) or 0 if latest is not None else 0
