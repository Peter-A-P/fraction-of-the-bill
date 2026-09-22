"""The training code, on a machine with no GPU: the dataset, the recipe arithmetic and
the checkpoint store. What is tested here is everything that decides whether an hour of
rented card is spent correctly, which is all of it except the gradient step."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
from items import TRUTH, make_item

from smallprint.data.build import SplitItem
from smallprint.data.split import Split
from smallprint.grade import grade_item, parse_extraction
from smallprint.prompts import PromptStyle, build_prompt
from smallprint.train import checkpoint, dataset, recipe
from smallprint.train.qlora import (
    INSTALL_HINT,
    RunRecord,
    adopt_chat_template,
    loss_tokens,
    require_gpu_stack,
)

BASE = recipe.TrainConfig(base="google/gemma-4-E2B", size="2b")


def pool(n_train: int = 40, n_validation: int = 6) -> list[SplitItem]:
    train = [make_item(f"train-{i:03d}", split=Split.TRAIN) for i in range(n_train)]
    validation = [make_item(f"val-{i:03d}", split=Split.VALIDATION) for i in range(n_validation)]
    return [*train, *validation]


def test_the_target_is_exactly_what_the_grader_marks_right() -> None:
    """Train towards a string the grader would fail and the malformed rate is the recipe."""
    examples, _ = dataset.build(pool(2, 0))
    answer = examples[0].assistant
    parsed = parse_extraction(answer)
    assert parsed is not None
    assert grade_item("x", parsed, TRUTH).exact


def test_a_training_file_can_never_hold_a_test_filing() -> None:
    with pytest.raises(ValueError, match="train and validation"):
        dataset.build([*pool(2, 1), make_item("test-1", split=Split.TEST_POST_CUTOFF)])


def test_the_volume_subsets_are_nested_so_the_curve_varies_one_thing() -> None:
    items = pool(40, 0)
    small = {s.item.item_id for s in dataset.take(items, 10, seed=1)}
    larger = {s.item.item_id for s in dataset.take(items, 25, seed=1)}
    assert small < larger
    assert len(small) == 10 and len(larger) == 25
    assert dataset.take(items, None, seed=1) != items  # ordered by hash, not by file order


def test_the_manifest_carries_the_prompt_the_examples_were_written_with() -> None:
    prompt = build_prompt(PromptStyle.ZERO_SHOT)
    examples, manifest = dataset.build(pool(5, 2), prompt=prompt)
    assert manifest.prompt_fingerprint == prompt.fingerprint
    assert (manifest.train, manifest.validation) == (5, 2)
    assert len(examples) == 7


def test_examples_round_trip_through_the_file(tmp_path: Path) -> None:
    examples, manifest = dataset.build(pool(5, 1))
    dataset.write(tmp_path, examples, manifest, build_dir=Path("data/build/full"))
    again, read_manifest = dataset.read(tmp_path)
    assert [e.item_id for e in again] == [e.item_id for e in examples]
    assert Path(read_manifest.build_dir) == Path("data/build/full")
    assert again[0].messages()[0]["role"] == "system"


def test_a_sequence_length_that_would_cut_the_json_is_caught_before_the_gpu_is_paid_for() -> None:
    lengths = [100] * 99 + [5000]
    assert dataset.truncated(lengths, 8192) == (0, 0.0)
    cut, share = dataset.truncated(lengths, 1000)
    assert (cut, round(share, 2)) == (1, 0.01)


def test_the_step_arithmetic_counts_the_partial_batch() -> None:
    config = BASE.model_copy(update={"batch_size": 2, "grad_accum": 8, "epochs": 3})
    assert config.examples_per_step == 16
    assert config.steps(160) == 30  # 10 steps an epoch, exactly
    assert config.steps(161) == 33  # 11 an epoch: the last one is not full and still counts
    with pytest.raises(ValueError, match="no examples"):
        config.steps(0)


def test_a_run_checkpoints_often_enough_to_survive_a_reclaim() -> None:
    config = BASE.model_copy(update={"save_steps": 50, "epochs": 2})
    assert config.checkpoints(5_000) == 1 + config.steps(5_000) // 50


def test_alpha_below_rank_is_refused_because_it_confounds_the_rank_sweep() -> None:
    with pytest.raises(ValueError, match="scaling factor"):
        recipe.TrainConfig(base="google/gemma-4-E2B", size="2b", rank=64, alpha=32)


def test_the_run_id_is_the_recipe_so_the_same_recipe_is_the_same_run() -> None:
    assert BASE.run_id == BASE.model_copy().run_id
    assert BASE.run_id != BASE.model_copy(update={"rank": 64, "alpha": 128}).run_id
    assert BASE.run_id.startswith("2b-r16-lr0.0001-nall-s0-")


def test_the_sweep_is_one_factor_at_a_time_and_names_the_base_once() -> None:
    runs = recipe.sweep(BASE)
    # The base, plus two new ranks, two new learning rates and three volumes: the base
    # is rank 16 and 1e-4 already, and its volume is every filing rather than a number.
    assert len(runs) == 8
    assert len({r.run_id for r in runs}) == 8
    assert runs[0].run_id == BASE.run_id
    # Alpha follows rank, so a rank sweep does not also sweep the scaling factor.
    for run in runs:
        assert run.alpha == 2 * run.rank


def test_three_seeds_differ_in_the_seed_and_nothing_else() -> None:
    runs = recipe.seeds(BASE, 3)
    assert [r.seed for r in runs] == [0, 1, 2]
    assert len({r.model_dump_json(exclude={"seed"}) for r in runs}) == 1


def test_gpu_hours_needs_a_measured_rate_rather_than_a_guess() -> None:
    assert recipe.gpu_hours([BASE], 5_000, 3.6) == pytest.approx(BASE.steps(5_000) * 0.001)
    with pytest.raises(ValueError, match="measured"):
        recipe.gpu_hours([BASE], 5_000, 0.0)


def write_checkpoint(directory: Path, step: int) -> Path:
    out = directory / f"checkpoint-{step}"
    out.mkdir(parents=True)
    (out / "adapter_model.safetensors").write_text(f"weights at {step}", encoding="utf-8")
    return out


def test_a_checkpoint_is_uploaded_and_found_again_on_an_empty_machine(tmp_path: Path) -> None:
    """The case that matters: the instance was reclaimed and the new one has a bare disk."""
    remote = checkpoint.LocalSyncer(tmp_path / "bucket")
    first = checkpoint.CheckpointStore(tmp_path / "run", remote)
    first.save(write_checkpoint(tmp_path / "run", 1))
    first.save(write_checkpoint(tmp_path / "run", 50))

    fresh = checkpoint.CheckpointStore(tmp_path / "new-instance", remote)
    assert fresh.resume_step() == 50
    latest = fresh.latest()
    assert latest is not None
    assert (latest / "adapter_model.safetensors").read_text(encoding="utf-8") == "weights at 50"


def test_an_upload_that_never_finished_is_not_resumed_from(tmp_path: Path) -> None:
    remote = checkpoint.LocalSyncer(tmp_path / "bucket")
    store = checkpoint.CheckpointStore(tmp_path / "run", remote)
    store.save(write_checkpoint(tmp_path / "run", 10))
    half = tmp_path / "bucket" / "checkpoint-99"
    half.mkdir()
    (half / "adapter_model.safetensors").write_text("truncated", encoding="utf-8")

    fresh = checkpoint.CheckpointStore(tmp_path / "new-instance", remote)
    assert fresh.resume_step() == 10


def fake_s3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> checkpoint.CommandSyncer:
    """The Runpod syncer's own arguments, with `aws` swapped for a stand-in over a directory."""
    monkeypatch.setenv("FAKE_S3_ROOT", str(tmp_path / "s3"))
    real = checkpoint.runpod_s3("s3://vol123/checkpoints/run-a", "EU-RO-1")
    stand_in = (sys.executable, str(Path(__file__).parent / "fake_s3.py"))
    return checkpoint.CommandSyncer(
        real.uri,
        argv=(*stand_in, *real.argv[1:]),
        list_argv=(*stand_in, *real.list_argv[1:]),
        empty_listing=real.empty_listing,
    )


def test_a_new_run_on_an_empty_volume_starts_at_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`aws s3 ls` of an empty prefix exits 1 with nothing on stderr. The first real smoke
    run, 2026-09-22, stopped on it before its first step."""
    remote = fake_s3(tmp_path, monkeypatch)
    assert checkpoint.CheckpointStore(tmp_path / "run", remote).resume_step() == 0


def test_a_listing_that_fails_for_real_still_stops_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exit 1 with an error message is a failure, not an empty volume: bad credentials must
    not quietly start a run from zero that then cannot save anything."""
    remote = fake_s3(tmp_path, monkeypatch)
    broken = checkpoint.CommandSyncer(
        remote.uri,
        argv=remote.argv,
        list_argv=(sys.executable, "-c", "import sys; sys.exit('AccessDenied')"),
        empty_listing=1,
    )
    with pytest.raises(RuntimeError, match="AccessDenied"):
        broken.names()


def test_the_runpod_syncer_names_the_volume_s_endpoint_and_lists_recursively() -> None:
    syncer = checkpoint.runpod_s3("s3://vol123/checkpoints", "EU-RO-1")
    assert "https://s3api-eu-ro-1.runpod.io/" in syncer.argv
    assert syncer.list_argv[:4] == ("aws", "s3", "ls", "--recursive")


def test_a_checkpoint_goes_through_the_s3_api_and_comes_back_on_a_new_machine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = fake_s3(tmp_path, monkeypatch)
    first = checkpoint.CheckpointStore(tmp_path / "run", remote)
    first.save(write_checkpoint(tmp_path / "run", 5))
    first.save(write_checkpoint(tmp_path / "run", 10))

    fresh = checkpoint.CheckpointStore(tmp_path / "new-instance", remote)
    assert fresh.resume_step() == 10
    latest = fresh.latest()
    assert latest is not None
    assert (latest / "adapter_model.safetensors").read_text(encoding="utf-8") == "weights at 10"


def test_an_upload_cut_off_by_a_reclaim_is_never_marked_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stand-in copies in name order, so `.complete` would go first if it were handed
    over with the weights. The machine goes after the first file."""
    remote = fake_s3(tmp_path, monkeypatch)
    store = checkpoint.CheckpointStore(tmp_path / "run", remote)
    store.save(write_checkpoint(tmp_path / "run", 5))
    big = write_checkpoint(tmp_path / "run", 10)
    (big / "optimizer.pt").write_text("state", encoding="utf-8")
    monkeypatch.setenv("FAKE_S3_FAIL_AFTER", "1")
    with pytest.raises(RuntimeError, match="failed"):
        store.save(big)
    monkeypatch.delenv("FAKE_S3_FAIL_AFTER")

    assert remote.names() == ["checkpoint-5"]
    assert checkpoint.CheckpointStore(tmp_path / "new-instance", remote).resume_step() == 5


def test_a_fresh_run_with_nowhere_to_resume_from_starts_at_zero(tmp_path: Path) -> None:
    assert checkpoint.CheckpointStore(tmp_path / "run").resume_step() == 0


def test_a_directory_that_is_not_a_checkpoint_is_refused(tmp_path: Path) -> None:
    store = checkpoint.CheckpointStore(tmp_path / "run")
    (tmp_path / "run" / "adapter").mkdir(parents=True)
    with pytest.raises(ValueError, match="not a checkpoint"):
        store.save(tmp_path / "run" / "adapter")


def test_the_laptop_is_told_which_extra_to_install_rather_than_an_import_error() -> None:
    with pytest.raises(RuntimeError, match="GPU extra"):
        require_gpu_stack()
    assert "laptop" in INSTALL_HINT


def test_the_run_record_round_trips(tmp_path: Path) -> None:
    _, manifest = dataset.build(pool(3, 1))
    import datetime as dt

    record = RunRecord(
        run_id=BASE.run_id, config=BASE, dataset=manifest, started_at=dt.datetime.now(dt.UTC)
    )
    record.write(tmp_path)
    from smallprint.train.qlora import read_record

    assert read_record(tmp_path).config.run_id == BASE.run_id


def test_every_base_is_pinned_to_a_revision_not_a_branch() -> None:
    """A base named without its revision is not reproducible: a later push to the
    repository would change what a recorded run trained on."""
    import re

    assert set(recipe.BASES) == {"2b", "4b", "7b"}
    for base in recipe.BASES.values():
        assert re.fullmatch(r"[0-9a-f]{40}", base.revision), base.repo
        assert re.fullmatch(r"[0-9a-f]{40}", base.baseline_revision), base.baseline_repo
        assert base.revision != base.baseline_revision


def test_step_time_is_the_trainers_clock_over_this_sessions_steps() -> None:
    """Wall time since start would count loading the model, most of a twenty-step smoke
    run; the global step of a resumed run counts steps another machine did."""
    from smallprint.train.qlora import step_seconds

    assert step_seconds(100.0, 20, 0) == 5.0
    assert step_seconds(100.0, 70, 50) == 5.0  # resumed at 50, ran 20 here
    assert step_seconds(None, 20, 0) is None
    assert step_seconds(100.0, 50, 50) is None  # nothing ran in this session


class _Tokenizer:
    def __init__(self, chat_template: str | None) -> None:
        self.chat_template = chat_template


def test_a_base_without_a_chat_template_takes_its_instruction_model_s() -> None:
    """All three bases ship none; found on the first real smoke run, 2026-09-22."""
    base = _Tokenizer(None)
    asked: list[tuple[str, str]] = []

    def load(repo: str, revision: str) -> str | None:
        asked.append((repo, revision))
        return "{{ messages }}"

    origin, digest = adopt_chat_template(base, ("google/gemma-4-E2B-it", "3e22461f"), load)
    assert base.chat_template == "{{ messages }}"
    assert asked == [("google/gemma-4-E2B-it", "3e22461f")]
    assert origin == "google/gemma-4-E2B-it@3e22461f"
    assert len(digest) == 64


def test_a_base_with_its_own_template_keeps_it() -> None:
    base = _Tokenizer("own")
    origin, _ = adopt_chat_template(base, ("x", "y"), lambda r, v: "other")
    assert (origin, base.chat_template) == ("base", "own")


def test_a_base_with_no_template_and_no_source_is_refused() -> None:
    with pytest.raises(ValueError, match="no chat template"):
        adopt_chat_template(_Tokenizer(None), None, lambda r, v: "t")


def test_each_example_is_split_so_the_loss_falls_on_the_answer_alone() -> None:
    """With the whole chat in one field, TRL trains on the prompt too."""
    examples, _ = dataset.build(pool(2, 0))
    row = dataset.as_chat(examples)[0]
    assert set(row) == {"prompt", "completion"}
    assert row["prompt"] == examples[0].messages()[:2]
    assert row["completion"] == [{"role": "assistant", "content": examples[0].assistant}]


def ids(examples: list[dataset.Example]) -> list[str]:
    return [e.item_id for e in examples]


def test_a_run_takes_the_same_nested_subset_as_the_dataset_file_would() -> None:
    """The first real smoke run asked for 320 filings and was handed all 5,060."""
    items = pool(40, 0)
    examples, _ = dataset.build(items)
    assert ids(dataset.take_examples(examples, 10)) == [
        s.item.item_id for s in dataset.take(items, 10, seed=dataset.VOLUME_SEED)
    ]
    assert set(ids(dataset.take_examples(examples, 10))) < set(
        ids(dataset.take_examples(examples, 25))
    )
    assert len(dataset.take_examples(examples, None)) == 40


def test_a_volume_larger_than_the_pool_is_refused() -> None:
    examples, _ = dataset.build(pool(5, 0))
    with pytest.raises(ValueError, match="more than the 5"):
        dataset.take_examples(examples, 6)


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({"input_ids": [1, 2, 3, 4], "labels": [-100, -100, 3, 4]}, (2, 4)),
        ({"input_ids": [1, 2, 3], "completion_mask": [0, 1, 1]}, (2, 3)),
    ],
)
def test_the_loss_share_is_read_from_labels_or_a_mask(
    row: dict[str, list[int]], expected: tuple[int, int]
) -> None:
    assert loss_tokens(row) == expected


@pytest.mark.parametrize(
    "row",
    [
        {"input_ids": [1, 2, 3], "labels": [1, 2, 3]},  # every token counts: nothing split
        {"input_ids": [1, 2, 3], "prompt": [0]},  # no labels at all
    ],
)
def test_a_row_whose_answer_is_not_separated_from_the_prompt_stops_the_run(
    row: dict[str, list[int]],
) -> None:
    with pytest.raises(RuntimeError):
        loss_tokens(row)


@pytest.mark.parametrize(
    ("name", "adapted"),
    [
        ("model.language_model.layers.3.self_attn.q_proj", True),
        ("model.language_model.layers.3.mlp.down_proj", True),
        ("model.layers.31.mlp.gate_proj", True),  # OLMo
        ("model.vision_tower.encoder.layers.0.self_attn.q_proj", False),
        ("model.audio_tower.layers.2.self_attn.v_proj", False),
        ("lm_head", False),
        ("model.language_model.layers.3.per_layer_input_gate", False),
    ],
)
def test_adapters_go_on_every_language_model_projection_and_nowhere_else(
    name: str, adapted: bool
) -> None:
    """Names as printed by the three bases on the pod, 2026-09-22."""
    assert (re.fullmatch(BASE.target_modules, name) is not None) is adapted
