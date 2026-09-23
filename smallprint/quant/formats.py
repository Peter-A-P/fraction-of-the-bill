"""Making the formats: AWQ and GPTQ with llm-compressor, GGUF with llama.cpp.

Every format starts from the same merged bf16 weights (`smallprint.train.merge`), which are
also the reference each one is judged against, so a delta measures the rounding and
nothing else. The heavy imports are inside the functions, as in `train.qlora`, and the parts
that decide what is measured are pure and tested: which layers are rounded, which filings
calibrate the rounding, and the llama.cpp command lines.

**AWQ and GPTQ both come from llm-compressor.** AutoAWQ, the library AWQ used to mean, was
archived in 2025 with its method moved into llm-compressor, and llm-compressor writes both
algorithms in one on-disk format, compressed-tensors, which vLLM serves with one flag. One
library for both means the two differ in the algorithm that chose the weights and in
nothing else. Both are 4-bit weights with 16-bit activations, in groups of 128: the scheme
vLLM runs fastest on the cards this project rents, and the one the published AWQ and GPTQ
results are usually quoted at.

**Calibration comes from the training pool, and only from it.** AWQ and GPTQ pick their
rounding by running real inputs through the model. Those inputs are data the format has
seen, so a calibration set drawn from the test pool would be a leak into every quantised
row of the results. `calibration` refuses anything but training filings, and draws them by
keyed hash so every size and both algorithms calibrate on the same filings.

**Rounded where the adapter went, and nowhere else.** The language model's linear layers.
Not `lm_head`, which every published scheme keeps at 16 bits because the output
distribution is where rounding shows first, and not the Gemma vision and audio towers,
which this task never feeds: rounding them costs nothing and measures nothing, and
calibrating them on text is meaningless.

**GGUF from llama.cpp's own tools.** The merged weights are converted once to a bf16 GGUF,
then `llama-quantize` writes Q8_0 and Q4_K_M from that. No importance matrix: it is the
llama.cpp default and the format most published GGUFs are, and an importance matrix is
calibration data by another name, which would be a second decision about what the model
has seen. It is a candidate ablation, not a default.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict

from smallprint.data.split import Split
from smallprint.quant.quality import Format
from smallprint.train.dataset import TRAINABLE, Example, take_examples
from smallprint.train.merge import digest_dir, render_ids

#: Filings the AWQ and GPTQ rounding is calibrated on. llm-compressor's examples use 256
#: to 512 sequences of 2,048 tokens; a filing here is about 3,100 tokens, so 256 of them
#: is about 800,000 tokens, more than either example.
CALIBRATION_FILINGS: Final = 256

#: A different seed from the volume subsets, so the calibration set is not simply the first
#: filings of the 1,000-filing training subset.
CALIBRATION_SEED: Final = 20260923

#: Weights in 4 bits, activations at 16, in groups of 128 columns.
SCHEME: Final = "W4A16"

#: What is never rounded. llm-compressor reads "re:" as a regular expression.
IGNORE: Final[tuple[str, ...]] = ("lm_head", r"re:.*vision.*", r"re:.*audio.*")

#: llama-quantize's name for each GGUF format.
GGUF_TYPES: Final[dict[Format, str]] = {Format.GGUF_Q8_0: "Q8_0", Format.GGUF_Q4_K_M: "Q4_K_M"}

RECORD: Final = "quant.json"

INSTALL_HINT = (
    "quantising needs the quant extra, which is not installed on this machine: "
    "`uv sync --extra quant` on the rented card"
)


class QuantRecord(BaseModel):
    """What a format was made from and how. Written beside the weights."""

    model_config = ConfigDict(frozen=True)

    format: Format
    #: The digest of the merged bf16 directory this was made from, as `merge.json` records
    #: it, so a format can be traced to the exact reference it is judged against.
    source_sha256: str
    scheme: str
    ignore: tuple[str, ...] = ()
    #: The filings the rounding was calibrated on, by id. Empty for GGUF, which has none.
    calibration_ids: tuple[str, ...] = ()
    max_seq_len: int | None = None
    output_sha256: str
    made_at: dt.datetime
    versions: dict[str, str] = {}

    def write(self, out_dir: Path) -> Path:
        path = out_dir / RECORD
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8", newline="\n")
        return path


def calibration(
    examples: Sequence[Example], n: int = CALIBRATION_FILINGS, *, seed: int = CALIBRATION_SEED
) -> list[Example]:
    """The filings AWQ and GPTQ calibrate on: training filings only, the same ones every time.

    The validation filings are left out as well as the test pools. They are what the
    checkpoints were scored on, and a format tuned on them would make that curve a
    little less honest for no gain.
    """
    leaked = sorted({e.split.value for e in examples} - {s.value for s in TRAINABLE})
    if leaked:
        raise ValueError(f"calibration may only see training filings; got {', '.join(leaked)}")
    train = [e for e in examples if e.split is Split.TRAIN]
    if n > len(train):
        raise ValueError(f"{n} calibration filings asked for, {len(train)} training filings")
    return take_examples(train, n, seed=seed)


def modifier(fmt: Format) -> dict[str, Any]:
    """The llm-compressor modifier for a format, as the arguments it is built from."""
    if fmt not in (Format.AWQ, Format.GPTQ):
        raise ValueError(f"{fmt.value} is not made by llm-compressor")
    return {"targets": ["Linear"], "scheme": SCHEME, "ignore": list(IGNORE)}


def quantise(
    fmt: Format,
    model_dir: Path,
    out_dir: Path,
    examples: Sequence[Example],
    *,
    max_seq_len: int = 8192,
) -> QuantRecord:
    """AWQ or GPTQ weights from the merged bf16 ones, calibrated on `examples`."""
    try:
        import torch
        from datasets import Dataset
        from llmcompressor import oneshot
        from llmcompressor.modifiers.awq import AWQModifier
        from llmcompressor.modifiers.quantization import GPTQModifier
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as missing:
        raise RuntimeError(f"{INSTALL_HINT} ({missing.name} is missing)") from missing

    chosen = calibration(examples)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    rows = []
    for e in chosen:
        ids = render_ids(tokenizer, e.messages())[:max_seq_len]
        rows.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="auto")
    arguments = modifier(fmt)
    recipe = AWQModifier(**arguments) if fmt is Format.AWQ else GPTQModifier(**arguments)
    oneshot(
        model=model,
        dataset=Dataset.from_list(rows),
        recipe=recipe,
        max_seq_length=max_seq_len,
        num_calibration_samples=len(rows),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, save_compressed=True)
    tokenizer.save_pretrained(out_dir)
    record = QuantRecord(
        format=fmt,
        source_sha256=_source_digest(model_dir),
        scheme=SCHEME,
        ignore=IGNORE,
        calibration_ids=tuple(e.item_id for e in chosen),
        max_seq_len=max_seq_len,
        output_sha256=digest_dir(out_dir, exclude=(RECORD,)),
        made_at=dt.datetime.now(dt.UTC),
        versions=_versions(("torch", "transformers", "llmcompressor", "compressed_tensors")),
    )
    record.write(out_dir)
    return record


def convert_argv(llama_cpp: Path, model_dir: Path, out_file: Path) -> list[str]:
    """llama.cpp's converter, merged safetensors to one bf16 GGUF.

    bf16 rather than f16: the weights are bf16 already, and f16 has the narrower exponent,
    so a conversion to it can overflow a large activation scale that bf16 held.
    """
    return [
        "python",
        str(llama_cpp / "convert_hf_to_gguf.py"),
        str(model_dir),
        "--outfile",
        str(out_file),
        "--outtype",
        "bf16",
    ]


def quantize_argv(binary: Path, source: Path, out_file: Path, fmt: Format) -> list[str]:
    """`llama-quantize` from the bf16 GGUF to one of the published types."""
    if fmt not in GGUF_TYPES:
        raise ValueError(f"{fmt.value} is not a GGUF format")
    return [str(binary), str(source), str(out_file), GGUF_TYPES[fmt]]


def gguf_file(out_dir: Path, name: str, fmt: Format) -> Path:
    """Where one GGUF format is written: one file, named for the run and the format."""
    if fmt not in GGUF_TYPES:
        raise ValueError(f"{fmt.value} is not a GGUF format")
    return out_dir / f"{name}-{GGUF_TYPES[fmt].lower()}.gguf"


def gguf_record(fmt: Format, model_dir: Path, out_file: Path) -> QuantRecord:
    """The record for a GGUF file, written beside it as `<file>.json`."""
    record = QuantRecord(
        format=fmt,
        source_sha256=_source_digest(model_dir),
        scheme=GGUF_TYPES[fmt],
        output_sha256=_file_digest(out_file),
        made_at=dt.datetime.now(dt.UTC),
    )
    out_file.with_suffix(".json").write_text(
        record.model_dump_json(indent=2), encoding="utf-8", newline="\n"
    )
    return record


def _source_digest(model_dir: Path) -> str:
    from smallprint.train.merge import read_record

    return read_record(model_dir).output_sha256


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _versions(names: Sequence[str]) -> dict[str, str]:
    found = {}
    for name in names:
        try:
            module = __import__(name)
        except ImportError:
            continue
        found[name] = str(getattr(module, "__version__", "unknown"))
    return found
