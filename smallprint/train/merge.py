"""Merging an adapter into its base: the bf16 weights every format is judged against.

A LoRA adapter is served and quantised as part of one set of weights, so each fine-tune is
merged once, into the base at the revision it was trained on, and saved in bf16. Those
weights are the reference: the bf16 row of the results is measured on them, and AWQ, GPTQ
and GGUF are each made from them and judged against them over the same filings.

**The merge is checked, not assumed.** Folding the adapter into the weights is arithmetic
that should change nothing but rounding, and a merge that went wrong, a wrong base
revision, an adapter saved without some of its layers, a model class that drops a module,
produces weights that load and answer and are subtly not the fine-tune. So before saving,
one validation filing's full chat, prompt and answer, is run through the adapter on the
base and through the merged model, and the next-token predictions are compared over its
last `CHECK_POSITIONS` positions, the answer among them. The share that agree is recorded in `merge.json`, and a merge below
`MIN_AGREEMENT` is refused. bf16 rounding can flip a near tie, so the bar is not one.

**Trained on 4-bit, merged into 16.** The adapter was trained against the NF4-rounded base
(QLoRA) and is merged into the bf16 base, the standard QLoRA practice and the only way to
publish weights that are not themselves NF4. The merged model is therefore not exactly the
model the training loss was measured on, which is one more reason its accuracy is measured
on the test set rather than carried over from the run.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict

from smallprint.train.dataset import Example
from smallprint.train.recipe import Base

#: The share of next-token predictions that must agree between the adapter on the base and
#: the merged model. See the module docstring.
MIN_AGREEMENT: Final = 0.99

#: Positions compared, counted back from the end of the check filing: the answer and the
#: end of the filing before it. The whole sequence would hold two copies of logits over a
#: 262,000-token vocabulary for 3,500 positions, several GB beside a 10 GB model on a 24 GB
#: card, and the answer is where the fine-tune differs from the base.
CHECK_POSITIONS: Final = 512

RECORD: Final = "merge.json"

#: Files a base ships beside its weights that a server needs and `save_pretrained` does not
#: write. The Gemma bases load as vision-language models, and vLLM refuses to serve one
#: without its processor configuration, even for text: the first merged 2B, 2026-09-24,
#: would not start for want of it. Copied from the base at its pinned revision.
PROCESSOR_FILES: Final = ("preprocessor_config.json", "processor_config.json")

INSTALL_HINT = (
    "merging needs the GPU extra, which is not installed on this machine: "
    "`uv sync --extra train` on the rented card"
)


class MergeRecord(BaseModel):
    """What a merged model was made from, and how closely it matches the adapter."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    size: str
    base: str
    base_revision: str
    #: Digest of the adapter directory as it was read, so a merged model can be traced to
    #: the exact adapter and not just to the run that wrote one.
    adapter_sha256: str
    #: Digest of every file written except this record. The formats cite it as their source.
    output_sha256: str
    check_item: str
    check_tokens: int
    agreement: float
    max_abs_logit_diff: float
    #: The token ids a server stops at, written to the merged generation_config.json.
    stop_token_ids: list[int] = []
    merged_at: dt.datetime
    versions: dict[str, str] = {}

    def write(self, out_dir: Path) -> Path:
        path = out_dir / RECORD
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8", newline="\n")
        return path


def read_record(out_dir: Path) -> MergeRecord:
    path = out_dir / RECORD
    if not path.is_file():
        raise FileNotFoundError(f"{out_dir} has no {RECORD}: it is not a merged model")
    return MergeRecord.model_validate_json(path.read_text(encoding="utf-8"))


def digest_dir(directory: Path, *, exclude: Sequence[str] = ()) -> str:
    """One SHA-256 over every file in a directory, names and contents, in name order.

    Names are in it so that a renamed shard is a different digest, and the order is fixed so
    the same files give the same digest on any machine.
    """
    digest = hashlib.sha256()
    files = sorted(
        p
        for p in directory.rglob("*")
        if p.is_file() and p.relative_to(directory).as_posix() not in exclude
    )
    if not files:
        raise ValueError(f"{directory} holds no files")
    for path in files:
        digest.update(path.relative_to(directory).as_posix().encode("utf-8") + b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def agreement(before: Sequence[int], after: Sequence[int]) -> float:
    """The share of positions at which two models predict the same next token."""
    if len(before) != len(after):
        raise ValueError(f"{len(before)} predictions against {len(after)}")
    if not before:
        raise ValueError("no positions to compare")
    return sum(1 for a, b in zip(before, after, strict=True) if a == b) / len(before)


def check_agreement(share: float, *, minimum: float = MIN_AGREEMENT) -> None:
    if share < minimum:
        raise RuntimeError(
            f"the merged model agrees with the adapter on {share:.1%} of next tokens, below "
            f"{minimum:.0%}: the merge changed more than rounding. Check the base revision "
            "and that the adapter loaded every layer"
        )


def processor_files(repo_files: Sequence[str]) -> list[str]:
    """Which of a base repository's files are processor configuration to carry over."""
    return sorted(f for f in repo_files if f in PROCESSOR_FILES)


def served_generation_config(
    base: Mapping[str, Any], instruct: Mapping[str, Any]
) -> dict[str, Any]:
    """The generation_config.json a merged model ships: its stop tokens, and no sampling.

    A fine-tune ends its answer with its chat template's end-of-turn token, `<turn|>` on
    Gemma and `<|im_end|>` on OLMo, and a base checkpoint names only its end-of-text token,
    so a server reading the base's configuration never stops: the first merged 2B,
    2026-09-24, wrote its answer, the end-of-turn token, and the answer again, to the
    2,048-token limit on every filing. The stop tokens are the instruction model's, from the
    revision the chat template came from. The base's sampling fields, temperature 1 and
    top-k 64, are left out: nobody here chose them.
    """
    eos = instruct.get("eos_token_id")
    ids = [eos] if isinstance(eos, int) else [int(t) for t in eos or []]
    if not ids:
        raise ValueError("the instruction model names no stop token")
    chosen: dict[str, Any] = {"eos_token_id": ids}
    for key in ("bos_token_id", "pad_token_id"):
        if base.get(key) is not None:
            chosen[key] = base[key]
    return chosen


def render_ids(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    """Token ids of a whole chat under the tokenizer's template.

    Some versions of transformers return the ids and some a mapping holding them, so both
    are taken rather than one assumed.
    """
    out = tokenizer.apply_chat_template(messages, tokenize=True)
    ids = out["input_ids"] if isinstance(out, Mapping) else out
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(t) for t in ids]


def merge(
    adapter_dir: Path,
    base: Base,
    out_dir: Path,
    *,
    run_id: str,
    size: str,
    check: Example,
) -> MergeRecord:
    """Merge the adapter into the base in bf16, check it, and save it with the tokenizer.

    The tokenizer comes from the adapter directory, where the trainer saved it with the
    chat template the run adopted, so the served model is prompted in the format it was
    trained on.
    """
    try:
        import torch
        from huggingface_hub import hf_hub_download, list_repo_files
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as missing:
        raise RuntimeError(f"{INSTALL_HINT} ({missing.name} is missing)") from missing

    adapter_digest = digest_dir(adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.chat_template is None:
        raise ValueError(f"{adapter_dir} has no chat template; it was not saved by a run")
    model = AutoModelForCausalLM.from_pretrained(
        base.repo, revision=base.revision, dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    ids = torch.tensor([render_ids(tokenizer, check.messages())], device=model.device)
    compared = min(CHECK_POSITIONS, int(ids.shape[1]))
    with torch.no_grad():
        before = model(input_ids=ids).logits[0, -compared:].float().cpu()
    merged = model.merge_and_unload()
    with torch.no_grad():
        after = merged(input_ids=ids).logits[0, -compared:].float().cpu()
    share = agreement(before.argmax(-1).tolist(), after.argmax(-1).tolist())
    difference = float((before - after).abs().max())
    check_agreement(share)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    merged.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    for name in processor_files(list_repo_files(base.repo, revision=base.revision)):
        shutil.copyfile(hf_hub_download(base.repo, name, revision=base.revision), out_dir / name)
    written = out_dir / "generation_config.json"
    base_config = json.loads(written.read_text(encoding="utf-8")) if written.is_file() else {}
    instruct_config = json.loads(
        Path(
            hf_hub_download(
                base.baseline_repo, "generation_config.json", revision=base.baseline_revision
            )
        ).read_text(encoding="utf-8")
    )
    served = served_generation_config(base_config, instruct_config)
    written.write_text(json.dumps(served, indent=2) + "\n", encoding="utf-8", newline="\n")
    record = MergeRecord(
        run_id=run_id,
        size=size,
        base=base.repo,
        base_revision=base.revision,
        adapter_sha256=adapter_digest,
        output_sha256=digest_dir(out_dir, exclude=(RECORD,)),
        check_item=check.item_id,
        check_tokens=compared,
        agreement=share,
        max_abs_logit_diff=difference,
        stop_token_ids=served["eos_token_id"],
        merged_at=dt.datetime.now(dt.UTC),
        versions=_versions(),
    )
    record.write(out_dir)
    return record


def _versions() -> dict[str, str]:
    found = {}
    for name in ("torch", "transformers", "peft"):
        try:
            module = __import__(name)
        except ImportError:
            continue
        found[name] = str(getattr(module, "__version__", "unknown"))
    return found
