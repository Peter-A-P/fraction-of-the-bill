"""After the fine-tune: the merge, the formats, serving it through the gateway, its tables
and the files the release gate reads. None of it needs a GPU to be wrong, so none of it
waits for one to be tested."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from boundary.config import BoundaryConfig
from items import make_item, training_pool
from test_baseline import RIGHT, WRONG, StubGateway, items, reply, zero_shot
from test_cards import record
from test_serving import grades
from typer.testing import CliRunner

from smallprint import baseline, gate, report
from smallprint.cli import app
from smallprint.data.build import SplitItem
from smallprint.data.split import Split
from smallprint.grade import unpaired_delta_ci
from smallprint.quant import formats
from smallprint.quant.quality import Format
from smallprint.schema import FIELDS
from smallprint.serve import launch
from smallprint.train import dataset, merge
from smallprint.train.recipe import BASES

# Wide, so a refusal is not wrapped mid-phrase by the box the error is printed in.
runner = CliRunner(env={"COLUMNS": "240"})


# -- the served name ---------------------------------------------------------------------


def test_a_served_name_carries_the_run_and_the_format_and_reads_back() -> None:
    run = "2b-r16-lr1e-4-nall-s0-e1"
    name = launch.served_name(run, Format.GGUF_Q4_K_M)
    assert name == "2b-r16-lr1e-4-nall-s0-e1-gguf-q4_k_m"
    assert launch.parse_served_name(f"selfhosted/{name}") == ("2b", run, Format.GGUF_Q4_K_M)
    assert launch.parse_served_name(launch.served_name(run, Format.BF16))[2] is Format.BF16


def test_a_name_without_a_format_is_refused_rather_than_guessed() -> None:
    with pytest.raises(ValueError, match="does not end in a format"):
        launch.parse_served_name("selfhosted/2b-r16-lr1e-4")
    with pytest.raises(ValueError, match="does not end in a format"):
        launch.parse_served_name("openai/gpt-5.6-luna")
    with pytest.raises(ValueError, match="not a run name"):
        launch.served_name("a/b", Format.AWQ)


# -- the servers: the chat format they render ----------------------------------------------


def test_the_servers_render_the_chat_template_the_model_was_trained_on() -> None:
    """llama-server guesses a format unless told to render the embedded template; vLLM takes
    sampling defaults from the checkpoint unless told not to."""
    gguf = launch.llamacpp_argv("m.gguf", "m", Format.GGUF_Q8_0)
    assert "--jinja" in gguf
    vllm = launch.vllm_argv("merged/2b", "m", Format.BF16)
    assert vllm[vllm.index("--generation-config") + 1] == "vllm"


# -- the gateway configuration ------------------------------------------------------------


def project_config() -> dict[str, object]:
    loaded = yaml.safe_load(Path("boundary.yaml").read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_the_served_config_is_the_project_config_plus_one_self_hosted_provider() -> None:
    base = project_config()
    config = launch.served_config(base, "http://127.0.0.1:8000/v1")
    validated = BoundaryConfig.model_validate(config)
    assert validated.providers["selfhosted"].self_hosted
    assert validated.providers["selfhosted"].base_url == "http://127.0.0.1:8000/v1"
    # Everything else is the project's own: same ledger, same caps, same vendor prices.
    for key in ("prices", "caps", "ledger", "routes"):
        assert config[key] == base[key]
    assert set(validated.providers) == {*base["providers"], "selfhosted"}  # type: ignore[misc]
    assert "self_hosted_prices" not in config


def test_the_served_config_refuses_a_second_self_hosted_provider() -> None:
    config = launch.served_config(project_config(), "http://127.0.0.1:8000/v1")
    with pytest.raises(ValueError, match="already has"):
        launch.served_config(config, "http://127.0.0.1:8001/v1")


def test_serve_config_writes_beside_the_base_and_refuses_an_empty_overlay(tmp_path: Path) -> None:
    base = tmp_path / "boundary.yaml"
    base.write_text(Path("boundary.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    out = tmp_path / "boundary-served.yaml"
    args = ["serve", "config", "--base-url", "http://127.0.0.1:8000/v1", "--base", str(base)]
    result = runner.invoke(app, [*args, "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert yaml.safe_load(out.read_text(encoding="utf-8"))["providers"]["selfhosted"]
    empty = tmp_path / "prices"
    empty.mkdir()
    refused = runner.invoke(app, [*args, "--out", str(out), "--self-hosted-prices", str(empty)])
    assert refused.exit_code != 0
    assert "no price file" in refused.output


def test_serve_argv_prints_the_command_for_the_served_name() -> None:
    result = runner.invoke(
        app,
        ["serve", "argv", "--model", "models/x/awq", "--run", "4b-r16", "--format", "awq"],
    )
    assert result.exit_code == 0
    assert result.output.startswith("vllm serve models/x/awq --served-model-name 4b-r16-awq")


# -- calibration and the llm-compressor recipe --------------------------------------------


def examples() -> list[dataset.Example]:
    pool = training_pool(12) + [make_item(f"val-{i}", split=Split.VALIDATION) for i in range(3)]
    built, _ = dataset.build(pool)
    return built


def test_calibration_is_training_filings_only_and_the_same_every_time() -> None:
    chosen = formats.calibration(examples(), 5)
    assert len(chosen) == 5
    assert {e.split for e in chosen} == {Split.TRAIN}
    assert [e.item_id for e in chosen] == [e.item_id for e in formats.calibration(examples(), 5)]
    assert {e.item_id for e in formats.calibration(examples(), 3)} <= {e.item_id for e in chosen}


def test_calibration_refuses_a_test_filing_anywhere_in_its_input() -> None:
    """Calibration data is data the format has seen. A test filing in it would leak into
    every quantised row of the results."""
    leaked = examples()[0].model_copy(update={"split": Split.TEST_POST_CUTOFF})
    with pytest.raises(ValueError, match="only see training filings"):
        formats.calibration([*examples(), leaked], 2)
    with pytest.raises(ValueError, match="12 training filings"):
        formats.calibration(examples(), 13)


def test_the_rounding_skips_the_output_head_and_the_towers_the_task_never_feeds() -> None:
    for fmt in (Format.AWQ, Format.GPTQ):
        arguments = formats.modifier(fmt)
        assert arguments["scheme"] == "W4A16"
        assert "lm_head" in arguments["ignore"]
        assert any("vision" in i for i in arguments["ignore"])
        assert any("audio" in i for i in arguments["ignore"])
    with pytest.raises(ValueError, match="not made by llm-compressor"):
        formats.modifier(Format.GGUF_Q8_0)


# -- GGUF with llama.cpp's own tools ----------------------------------------------------------


def test_gguf_goes_through_one_bf16_file_to_both_published_types(tmp_path: Path) -> None:
    convert = formats.convert_argv(Path("llama.cpp"), Path("models/2b/bf16"), tmp_path / "x.gguf")
    assert convert[1].endswith("convert_hf_to_gguf.py")
    assert convert[convert.index("--outtype") + 1] == "bf16"
    q4 = formats.quantize_argv(
        Path("llama-quantize"), tmp_path / "x.gguf", tmp_path / "y", Format.GGUF_Q4_K_M
    )
    assert q4[-1] == "Q4_K_M"
    assert formats.gguf_file(tmp_path, "2b-r16", Format.GGUF_Q8_0).name == "2b-r16-q8_0.gguf"
    with pytest.raises(ValueError, match="not a GGUF format"):
        formats.quantize_argv(Path("q"), tmp_path, tmp_path, Format.AWQ)


def test_a_gguf_record_cites_the_merged_weights_it_was_made_from(tmp_path: Path) -> None:
    merged = tmp_path / "bf16"
    merged.mkdir()
    (merged / "model.safetensors").write_bytes(b"weights")
    source = merge.MergeRecord(
        run_id="r",
        size="2b",
        base=BASES["2b"].repo,
        base_revision=BASES["2b"].revision,
        adapter_sha256="a" * 64,
        output_sha256=merge.digest_dir(merged),
        check_item="val-0",
        check_tokens=512,
        agreement=1.0,
        max_abs_logit_diff=0.01,
        merged_at="2026-09-23T00:00:00Z",  # type: ignore[arg-type]
    )
    source.write(merged)
    target = tmp_path / "x-q8_0.gguf"
    target.write_bytes(b"gguf")
    written = formats.gguf_record(Format.GGUF_Q8_0, merged, target)
    assert written.source_sha256 == source.output_sha256
    assert json.loads(target.with_suffix(".json").read_text())["scheme"] == "Q8_0"


# -- the merge check ---------------------------------------------------------------------------


def test_the_directory_digest_sees_names_and_contents_and_skips_its_own_record(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.safetensors").write_bytes(b"one")
    (tmp_path / "config.json").write_text("{}")
    first = merge.digest_dir(tmp_path)
    (tmp_path / merge.RECORD).write_text("{}")
    assert merge.digest_dir(tmp_path, exclude=(merge.RECORD,)) == first
    (tmp_path / "a.safetensors").rename(tmp_path / "b.safetensors")
    assert merge.digest_dir(tmp_path, exclude=(merge.RECORD,)) != first
    with pytest.raises(ValueError, match="no files"):
        merge.digest_dir(tmp_path / "config.json")


def test_a_merge_that_changed_more_than_rounding_is_refused() -> None:
    assert merge.agreement([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0
    merge.check_agreement(merge.agreement([1] * 199 + [2], [1] * 200))  # one flip in 200
    with pytest.raises(RuntimeError, match="more than rounding"):
        merge.check_agreement(merge.agreement([1] * 90 + [2] * 10, [1] * 100))
    with pytest.raises(ValueError, match="against"):
        merge.agreement([1], [1, 2])


class Tokens:
    def __init__(self, out: object) -> None:
        self.out = out

    def apply_chat_template(self, messages: list[dict[str, str]], tokenize: bool) -> object:
        return self.out


def test_chat_ids_are_read_whether_the_tokenizer_returns_a_list_or_a_mapping() -> None:
    messages = [{"role": "user", "content": "x"}]
    assert merge.render_ids(Tokens([1, 2, 3]), messages) == [1, 2, 3]
    assert merge.render_ids(Tokens({"input_ids": [4, 5]}), messages) == [4, 5]
    assert merge.render_ids(Tokens({"input_ids": [[6, 7]]}), messages) == [6, 7]


def test_merge_refuses_a_run_trained_on_another_revision(tmp_path: Path) -> None:
    stale = record().model_copy(update={"base_revision": "0" * 40})
    stale.write(tmp_path)
    result = runner.invoke(app, ["quantise", "merge", "--run-dir", str(tmp_path), "--out", "x"])
    assert result.exit_code != 0
    assert "not this fine-tune" in result.output


# -- the fine-tuned tables ---------------------------------------------------------------------


def make_run(
    out: Path,
    answers: dict[int, str | None],
    *,
    model: str,
    run_items: list[SplitItem],
    cost: float,
) -> None:
    gateway = StubGateway(lambda i: reply(answers.get(i, RIGHT), cost=cost))
    baseline.run(
        gateway,
        model=model,
        items=run_items,
        prompt=zero_shot(),
        out_dir=out,
        build_dir=Path("data/build/full"),
        run_id=f"run-{out.name}",
    )


def pre_items(n: int = 4) -> list[SplitItem]:
    return [make_item(f"pre-{i}", split=Split.TEST_PRE_CUTOFF) for i in range(n)]


def test_the_finetuned_table_pairs_with_the_best_api_and_shows_the_cutoff_gap(
    tmp_path: Path,
) -> None:
    frontier, tuned = tmp_path / "frontier", tmp_path / "tuned"
    make_run(frontier / "cheap", {1: WRONG}, model="openai/cheap", run_items=items(), cost=0.001)
    make_run(frontier / "best", {}, model="openai/best", run_items=items(), cost=0.01)
    name = "selfhosted/2b-r16-bf16"
    make_run(tuned / "post", {0: WRONG}, model=name, run_items=items(), cost=0.0001)
    make_run(tuned / "pre", {}, model=name, run_items=pre_items(), cost=0.0001)

    post, _ = report.load(report.run_dirs(tuned, Split.TEST_POST_CUTOFF.value), items())
    pre, _ = report.load(report.run_dirs(tuned, Split.TEST_PRE_CUTOFF.value), pre_items())
    front, _ = report.load(report.run_dirs(frontier), items())
    assert report.best_frontier(front).summary.model == "openai/best"

    table = report.finetuned_table(post, pre, front)
    assert "Paired delta vs openai/best zero_shot" in table
    row = table.splitlines()[2]
    assert row.startswith("| `2b-r16` | 2b | bf16 |")
    # WRONG misses 2 of 15 fields on one filing of four: -3.3 points, paired.
    assert "-3.3% (-10.0% to +0.0%)" in row
    # Pre-cutoff all right, post-cutoff not: +3.3 points the other way, and unpaired.
    assert row.endswith("| +3.3% (+0.0% to +10.0%) |")


def test_a_split_directory_holds_only_its_own_runs(tmp_path: Path) -> None:
    make_run(tmp_path / "post", {}, model="selfhosted/a-bf16", run_items=items(), cost=0.0)
    make_run(tmp_path / "pre", {}, model="selfhosted/a-bf16", run_items=pre_items(), cost=0.0)
    assert [d.name for d in report.run_dirs(tmp_path, Split.TEST_PRE_CUTOFF.value)] == ["pre"]
    assert len(report.run_dirs(tmp_path)) == 2


def test_the_quantisation_table_judges_each_format_against_its_own_bf16(tmp_path: Path) -> None:
    make_run(tmp_path / "bf16", {}, model="selfhosted/2b-r16-bf16", run_items=items(), cost=0.0)
    make_run(tmp_path / "q8", {}, model="selfhosted/2b-r16-gguf-q8_0", run_items=items(), cost=0.0)
    make_run(
        tmp_path / "q4",
        {0: WRONG, 1: WRONG},
        model="selfhosted/2b-r16-gguf-q4_k_m",
        run_items=items(),
        cost=0.0,
    )
    make_run(tmp_path / "orphan", {}, model="selfhosted/7b-r16-awq", run_items=items(), cost=0.0)
    post, _ = report.load(report.run_dirs(tmp_path), items())

    table = report.quantisation_table(post)
    lines = table.splitlines()
    assert lines[2].startswith("| `2b-r16` | gguf-q4_k_m |") and lines[2].endswith("| no |")
    assert lines[3].startswith("| `2b-r16` | gguf-q8_0 | +0.0%") and lines[3].endswith("| yes |")
    assert "Measured without a bf16 run to judge them against: `7b-r16` awq." in table


def test_the_contamination_gap_is_unpaired_and_refuses_shared_filings() -> None:
    post = grades([RIGHT] * 10)
    pre = [g.model_copy(update={"item_id": f"pre-{g.item_id}"}) for g in grades([WRONG] * 10)]
    gap = unpaired_delta_ci(post, pre)
    assert gap.point == pytest.approx(2 / 15)
    assert gap.n == 20
    with pytest.raises(ValueError, match="share filings"):
        unpaired_delta_ci(post, post)


# -- the release gate's files ------------------------------------------------------------------


def test_each_field_is_a_suite_with_one_verdict_per_filing(tmp_path: Path) -> None:
    make_run(
        tmp_path, {0: WRONG, 2: None}, model="selfhosted/2b-r16-bf16", run_items=items(), cost=0.0
    )
    side = gate.side(
        "fine-tune",
        baseline.read_predictions(tmp_path),
        items(),
        source={"run_id": "r"},
    )
    assert set(side["suites"]) == set(FIELDS)
    revenue = side["suites"]["revenue"]
    assert revenue["outcomes"] == {"test-0": False, "test-1": True, "test-3": True}
    # The call that failed is not an answer, so it is ungradeable, not wrong.
    assert revenue["ungradeable_items"] == 1
    assert side["suites"]["period_end"]["outcomes"]["test-0"] is True


def test_the_spec_names_every_field_and_the_margin() -> None:
    spec = gate.spec()
    assert [s["key"] for s in spec["suites"]] == list(FIELDS)
    assert spec["delta_points"] == gate.DELTA_POINTS
