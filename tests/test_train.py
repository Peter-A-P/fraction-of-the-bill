"""The training code, on a machine with no GPU: the dataset, the recipe arithmetic and
the checkpoint store. What is tested here is everything that decides whether an hour of
rented card is spent correctly, which is all of it except the gradient step."""

from __future__ import annotations

from pathlib import Path

import pytest
from items import TRUTH, make_item

from smallprint.data.build import SplitItem
from smallprint.data.split import Split
from smallprint.grade import grade_item, parse_extraction
from smallprint.prompts import PromptStyle, build_prompt
from smallprint.train import checkpoint, dataset, recipe
from smallprint.train.qlora import INSTALL_HINT, RunRecord, require_gpu_stack

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
