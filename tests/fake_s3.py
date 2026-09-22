"""A stand-in for `aws s3`, over a local directory, for testing `CommandSyncer`.

`FAKE_S3_ROOT` is where the buckets live. `FAKE_S3_FAIL_AFTER=n` makes a sync copy n files
and then fail, the way an upload does when the machine under it is reclaimed. Files are
copied in name order, so the marker, `.complete`, goes before the weights if the caller
hands it over with them: an order a real sync does not rule out.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def local(uri: str) -> Path:
    if uri.startswith("s3://"):
        return Path(os.environ["FAKE_S3_ROOT"]) / uri.removeprefix("s3://")
    return Path(uri)


def main(argv: list[str]) -> int:
    # Drop the values of --region and --endpoint-url, which the real command needs.
    positional, skip = [], False
    for arg in argv:
        if skip:
            skip = False
            continue
        if arg in {"--region", "--endpoint-url"}:
            skip = True
            continue
        if arg.startswith("--") or arg == "s3":
            continue
        positional.append(arg)
    command, *paths = positional
    if command == "ls":
        root = local(paths[0])
        base = Path(os.environ["FAKE_S3_ROOT"]) / paths[0].removeprefix("s3://").split("/")[0]
        for file in sorted(root.rglob("*")) if root.exists() else []:
            if file.is_file():
                print(
                    f"2026-09-22 12:00:00 {file.stat().st_size:>9} {file.relative_to(base).as_posix()}"
                )
        return 0
    if command == "sync":
        source, target = local(paths[0]), local(paths[1])
        budget = int(os.environ.get("FAKE_S3_FAIL_AFTER", "-1"))
        for file in sorted(f for f in source.rglob("*") if f.is_file()):
            destination = target / file.relative_to(source)
            if destination.is_file() and destination.read_bytes() == file.read_bytes():
                continue
            if budget == 0:
                print("upload failed: connection reset", file=sys.stderr)
                return 1
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, destination)
            budget -= 1
        return 0
    print(f"unknown command {command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
