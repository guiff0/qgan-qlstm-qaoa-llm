"""
Tests for run_pipeline.py. subprocess.run is monkeypatched throughout --
these test the orchestrator's OWN logic (state persistence, skip policy,
parallel batching, failure handling), not the real scripts it invokes
(those are tested by tests/test_data_ingestion.py, test_standalone_demo.py,
etc.). See the end-to-end runs described in the accompanying PR/README for
verification against the real subprocesses.
"""
from __future__ import annotations

import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run_pipeline as rp  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Every test gets its own state file path, never the real repo's."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rp, "STATE_PATH", os.path.join(str(tmp_path), "data", "pipeline_state.json"))
    yield


def fake_run(returncode=0):
    def _run(cmd, **kwargs):
        return types.SimpleNamespace(returncode=returncode)
    return _run


# ---------------------------------------------------------------- state persistence
def test_load_state_fresh_when_no_file():
    state = rp.load_state()
    assert state["steps"] == {} and state["schema_version"] == rp.STATE_SCHEMA_VERSION


def test_save_and_reload_roundtrip():
    state = rp.load_state()
    rp.mark_completed(state, "step_a", 12.3, "detail here")
    reloaded = rp.load_state()
    assert reloaded["steps"]["step_a"]["status"] == "completed"
    assert reloaded["steps"]["step_a"]["duration_seconds"] == 12.3
    assert reloaded["steps"]["step_a"]["detail"] == "detail here"


def test_mark_running_then_failed_leaves_correct_final_status():
    state = rp.load_state()
    rp.mark_running(state, "step_a")
    assert rp.load_state()["steps"]["step_a"]["status"] == "running"
    rp.mark_failed(state, "step_a", 5.0, returncode=1)
    reloaded = rp.load_state()
    assert reloaded["steps"]["step_a"]["status"] == "failed"
    assert reloaded["steps"]["step_a"]["returncode"] == 1


def test_state_file_is_valid_json_after_every_write(tmp_path):
    state = rp.load_state()
    for i in range(5):
        rp.mark_completed(state, f"step_{i}", 1.0, "x")
        with open(rp.STATE_PATH) as f:
            json.load(f)  # must never be partially written


# ---------------------------------------------------------------- skip policy (the "ask once" contract)
class Args:
    def __init__(self, yes=False, no_skip=False, reset=False):
        self.yes, self.no_skip, self.reset = yes, no_skip, reset


def _step(step_id, artifact_ok=True):
    return {"id": step_id, "desc": step_id, "cmd": ["true"], "artifact_ok": lambda: artifact_ok}


def test_no_prior_state_means_no_prompt_and_no_skip(monkeypatch):
    state = rp.load_state()
    steps = [_step("a")]
    # If input() were called, this would raise -- proves it never asks
    # when there is nothing to skip.
    monkeypatch.setattr("builtins.input", lambda *a: (_ for _ in ()).throw(AssertionError("should not prompt")))
    assert rp.decide_skip_policy(state, steps, Args()) is False


def test_yes_flag_skips_without_prompting(monkeypatch):
    state = rp.load_state()
    rp.mark_completed(state, "a", 1.0, "x")
    steps = [_step("a")]
    monkeypatch.setattr("builtins.input", lambda *a: (_ for _ in ()).throw(AssertionError("should not prompt")))
    assert rp.decide_skip_policy(state, steps, Args(yes=True)) is True


def test_no_skip_flag_redoes_without_prompting(monkeypatch):
    state = rp.load_state()
    rp.mark_completed(state, "a", 1.0, "x")
    steps = [_step("a")]
    monkeypatch.setattr("builtins.input", lambda *a: (_ for _ in ()).throw(AssertionError("should not prompt")))
    assert rp.decide_skip_policy(state, steps, Args(no_skip=True)) is False


def test_reset_removes_state_file_and_redoes_without_prompting(monkeypatch):
    state = rp.load_state()
    rp.mark_completed(state, "a", 1.0, "x")
    assert os.path.isfile(rp.STATE_PATH)
    steps = [_step("a")]
    monkeypatch.setattr("builtins.input", lambda *a: (_ for _ in ()).throw(AssertionError("should not prompt")))
    assert rp.decide_skip_policy(state, steps, Args(reset=True)) is False
    assert not os.path.isfile(rp.STATE_PATH)


def test_artifact_missing_means_not_skippable_even_if_state_says_completed(monkeypatch):
    """A stale 'completed' marker whose underlying file is gone must not
    be trusted -- this is the artifact_ok() second-opinion check."""
    state = rp.load_state()
    rp.mark_completed(state, "a", 1.0, "x")
    steps = [_step("a", artifact_ok=False)]  # artifact no longer exists
    monkeypatch.setattr("builtins.input", lambda *a: (_ for _ in ()).throw(AssertionError("should not prompt")))
    # Nothing genuinely skippable -> no prompt, and (moot) nothing to skip.
    assert rp.decide_skip_policy(state, steps, Args()) is False


def test_asks_exactly_once_and_respects_yes_answer(monkeypatch):
    state = rp.load_state()
    rp.mark_completed(state, "a", 1.0, "x")
    rp.mark_completed(state, "b", 1.0, "x")
    steps = [_step("a"), _step("b")]
    calls = []

    def fake_input(prompt):
        calls.append(prompt)
        return "y"
    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(rp.sys.stdin, "isatty", lambda: True)
    result = rp.decide_skip_policy(state, steps, Args())
    assert result is True
    assert len(calls) == 1  # asked exactly once, not once per step


def test_answering_no_means_redo_everything(monkeypatch):
    state = rp.load_state()
    rp.mark_completed(state, "a", 1.0, "x")
    steps = [_step("a")]
    monkeypatch.setattr("builtins.input", lambda p: "n")
    monkeypatch.setattr(rp.sys.stdin, "isatty", lambda: True)
    assert rp.decide_skip_policy(state, steps, Args()) is False


def test_non_tty_with_no_flags_defaults_to_skip(monkeypatch):
    state = rp.load_state()
    rp.mark_completed(state, "a", 1.0, "x")
    steps = [_step("a")]
    monkeypatch.setattr(rp.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr("builtins.input", lambda *a: (_ for _ in ()).throw(AssertionError("should not block on input")))
    assert rp.decide_skip_policy(state, steps, Args()) is True


# ---------------------------------------------------------------- run_step
def test_run_step_skips_when_completed_and_artifact_ok(monkeypatch):
    state = rp.load_state()
    rp.mark_completed(state, "a", 1.0, "x")
    called = []
    monkeypatch.setattr(rp.subprocess, "run", lambda *a, **k: called.append(1) or fake_run()(a))
    ok = rp.run_step(_step("a"), state, skip_completed=True)
    assert ok is True and called == []  # subprocess never invoked


def test_run_step_runs_and_marks_completed_on_success(monkeypatch):
    state = rp.load_state()
    monkeypatch.setattr(rp.subprocess, "run", fake_run(returncode=0))
    ok = rp.run_step(_step("a", artifact_ok=True), state, skip_completed=True)
    assert ok is True
    assert state["steps"]["a"]["status"] == "completed"


def test_run_step_success_but_missing_artifact_marks_failed_for_required_step(monkeypatch):
    state = rp.load_state()
    monkeypatch.setattr(rp.subprocess, "run", fake_run(returncode=0))
    step = _step("a", artifact_ok=False)
    ok = rp.run_step(step, state, skip_completed=True)
    assert ok is False
    assert state["steps"]["a"]["status"] == "failed"


def test_run_step_nonzero_returncode_marks_failed(monkeypatch):
    state = rp.load_state()
    monkeypatch.setattr(rp.subprocess, "run", fake_run(returncode=1))
    ok = rp.run_step(_step("a"), state, skip_completed=True)
    assert ok is False
    assert state["steps"]["a"]["status"] == "failed"
    assert state["steps"]["a"]["returncode"] == 1


def test_run_step_writes_to_log_file_when_parallel(monkeypatch, tmp_path):
    def fake_subprocess_run(cmd, stdout=None, stderr=None):
        if stdout is not None:
            stdout.write("hello from subprocess\n")
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(rp.subprocess, "run", fake_subprocess_run)
    log_path = str(tmp_path / "out.log")
    ok = rp.run_step(_step("a"), rp.load_state(), skip_completed=True, log_to_file=log_path)
    assert ok is True
    assert "hello from subprocess" in open(log_path).read()


# ---------------------------------------------------------------- parallel batching
def test_run_batch_parallel_runs_all_and_aggregates_results(monkeypatch, tmp_path):
    monkeypatch.setattr(rp.subprocess, "run", fake_run(returncode=0))
    state = rp.load_state()
    batch = [_step("a"), _step("b"), _step("c")]
    results = rp.run_batch_parallel(batch, state, skip_completed=True, max_workers=3,
                                    log_dir=str(tmp_path / "logs"))
    assert results == {"a": True, "b": True, "c": True}
    assert all(state["steps"][s]["status"] == "completed" for s in ("a", "b", "c"))


def test_run_batch_parallel_mixed_success_and_failure(monkeypatch, tmp_path):
    def selective_run(cmd, stdout=None, stderr=None):
        # cmd[0] encodes success/failure for this fake step
        return types.SimpleNamespace(returncode=0 if "ok" in cmd[0] else 1)
    monkeypatch.setattr(rp.subprocess, "run", selective_run)
    state = rp.load_state()
    batch = [
        {"id": "a", "desc": "a", "cmd": ["ok_a"], "artifact_ok": lambda: True},
        {"id": "b", "desc": "b", "cmd": ["fail_b"], "artifact_ok": lambda: True},
    ]
    results = rp.run_batch_parallel(batch, state, skip_completed=True, max_workers=2,
                                    log_dir=str(tmp_path / "logs"))
    assert results == {"a": True, "b": False}
    assert state["steps"]["a"]["status"] == "completed"
    assert state["steps"]["b"]["status"] == "failed"


def test_run_batch_parallel_creates_separate_log_files(monkeypatch, tmp_path):
    monkeypatch.setattr(rp.subprocess, "run", fake_run(returncode=0))
    state = rp.load_state()
    batch = [_step("train_eval:Model A"), _step("train_eval:Model B")]
    log_dir = str(tmp_path / "logs")
    rp.run_batch_parallel(batch, state, skip_completed=True, max_workers=2, log_dir=log_dir)
    files = sorted(os.listdir(log_dir))
    assert len(files) == 2
    assert files != [files[0], files[0]]  # distinct filenames, no clobbering


# ---------------------------------------------------------------- build_steps / artifact checks
def test_build_steps_tags_acquire_and_train_eval_as_parallel_groups(tmp_path):
    cfg = {
        "data": {"dukascopy_file": "d.csv", "forexsb_file": "f.csv", "fred_file": "fr.csv",
                 "processed_dir": str(tmp_path)},
        "logging": {"results_dir": str(tmp_path)},
    }
    steps = rp.build_steps(cfg, model_names=["Model A", "Model B"], only_models=None)
    groups = {s["id"]: s.get("parallel_group") for s in steps}
    assert groups["acquire_dukascopy"] == groups["acquire_xval"] == groups["acquire_fred"] == "acquire"
    assert groups["train_eval:Model A"] == groups["train_eval:Model B"] == "train_eval"
    assert groups["verify_environment"] is None
    assert groups["prepare_data"] is None
    assert groups["verify_chapter4"] is None


def test_build_steps_respects_only_models(tmp_path):
    cfg = {
        "data": {"dukascopy_file": "d.csv", "forexsb_file": "f.csv", "fred_file": "fr.csv",
                 "processed_dir": str(tmp_path)},
        "logging": {"results_dir": str(tmp_path)},
    }
    steps = rp.build_steps(cfg, model_names=["A", "B", "C"], only_models=["B"])
    train_eval_ids = [s["id"] for s in steps if s.get("group") == "train_eval"]
    assert train_eval_ids == ["train_eval:B"]


def test_prepare_data_artifact_ok_requires_all_six_files(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    cfg = {
        "data": {"dukascopy_file": "d.csv", "forexsb_file": "f.csv", "fred_file": "fr.csv",
                 "processed_dir": str(processed)},
        "logging": {"results_dir": str(tmp_path)},
    }
    steps = rp.build_steps(cfg, model_names=[], only_models=None)
    prepare_step = next(s for s in steps if s["id"] == "prepare_data")
    assert prepare_step["artifact_ok"]() is False
    for prefix in ("X", "y"):
        for split in ("train", "val", "test"):
            (processed / f"{prefix}_{split}.npy").write_bytes(b"x")
    assert prepare_step["artifact_ok"]() is True


# ---------------------------------------------------------------- monitoring / dashboard
def test_tail_last_line_reads_final_carriage_return_frame(tmp_path):
    p = tmp_path / "log.txt"
    # tqdm-style: many \r-separated frames on one physical line, final \n
    p.write_bytes(b"epoch 1: 10%\repoch 1: 50%\repoch 1: 99%\n")
    assert rp._tail_last_line(str(p)) == "epoch 1: 99%"


def test_tail_last_line_missing_file_returns_empty_string(tmp_path):
    assert rp._tail_last_line(str(tmp_path / "nope.log")) == ""


def test_tail_last_line_truncates_long_lines(tmp_path):
    p = tmp_path / "log.txt"
    p.write_text("x" * 500)
    assert len(rp._tail_last_line(str(p))) <= 88


def test_render_dashboard_shows_every_step_with_status(tmp_path):
    state = {"steps": {
        "a": {"status": "completed", "duration_seconds": 12.0},
        "b": {"status": "failed", "duration_seconds": 3.0},
    }}
    steps = [_step("a"), _step("b"), _step("c")]  # "c" never started: pending
    out = rp.render_dashboard(steps, state, str(tmp_path))
    assert "[DONE]" in out and "a" in out
    assert "[FAIL]" in out and "b" in out
    assert "[wait]" in out and "c" in out


def test_render_dashboard_running_step_shows_log_hint(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "a.log").write_text("epoch 2 batches: 75%\n")
    state = {"steps": {"a": {"status": "running",
                             "started_utc": rp._now()}}}
    out = rp.render_dashboard([_step("a")], state, str(log_dir))
    assert "[RUN ]" in out
    assert "75%" in out


def test_watch_pipeline_exits_once_all_steps_terminal(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(rp, "load_config", lambda cfg_path: {})
    monkeypatch.setattr(rp, "resolve_model_names", lambda cfg_path, only: (["A"], None))
    monkeypatch.setattr(rp, "build_steps", lambda cfg, model_names, only_models: [
        {"id": "a", "desc": "a", "cmd": [], "artifact_ok": lambda: True},
    ])
    state = {"steps": {"a": {"status": "completed", "duration_seconds": 1.0}}}
    monkeypatch.setattr(rp, "load_state", lambda: state)
    monkeypatch.setattr(rp.sys.stdout, "isatty", lambda: False)  # no ANSI clear in a test

    class WatchArgs:
        config = None
        only = None
        watch_interval = 0.01
    rc = rp.watch_pipeline(WatchArgs())
    assert rc == 0
    assert "final state" in capsys.readouterr().out


def test_run_batch_parallel_still_works_with_dashboard_thread_running(monkeypatch, tmp_path):
    """The live-dashboard thread added to run_batch_parallel must not
    change its correctness (results, state) even though it now also
    prints in the background while the batch runs."""
    monkeypatch.setattr(rp.subprocess, "run", fake_run(returncode=0))
    state = rp.load_state()
    batch = [_step("a"), _step("b")]
    results = rp.run_batch_parallel(batch, state, skip_completed=True, max_workers=2,
                                    log_dir=str(tmp_path / "logs"), dashboard_interval=0.01)
    assert results == {"a": True, "b": True}
    assert all(state["steps"][s]["status"] == "completed" for s in ("a", "b"))
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    import pandas as pd
    pd.DataFrame({"model_type": ["Classical LSTM"], "rmse": [0.1]}).to_csv(
        results_dir / "all_results.csv", index=False)
    cfg = {"logging": {"results_dir": str(results_dir)}}
    assert rp._model_has_result(cfg, "Classical LSTM") is True
    assert rp._model_has_result(cfg, "QGAN-LLM") is False


# ---------------------------------------------------------------- full run_pipeline() orchestration (mocked)
def test_run_pipeline_status_flag_prints_and_exits_without_running(monkeypatch, capsys):
    monkeypatch.setattr(rp.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("should not run anything under --status")))
    rc = rp.run_pipeline(["--status"])
    assert rc == 0
    out = capsys.readouterr().out
    assert '"schema_version"' in out


def test_run_pipeline_aborts_on_required_step_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(rp, "resolve_model_names", lambda cfg_path, only: (["Model A"], None))
    monkeypatch.setattr(rp, "load_config", lambda cfg_path: {
        "data": {"dukascopy_file": str(tmp_path / "d.csv"), "forexsb_file": str(tmp_path / "f.csv"),
                 "fred_file": str(tmp_path / "fr.csv"), "processed_dir": str(tmp_path)},
        "logging": {"results_dir": str(tmp_path)},
    })

    def failing_run(cmd, stdout=None, stderr=None):
        # verify_environment (advisory) "fails" but shouldn't abort;
        # acquire_dukascopy (required) fails and SHOULD abort.
        if "verify_environment" in cmd[-1]:
            return types.SimpleNamespace(returncode=1)
        if "dukascopy" in " ".join(cmd):
            return types.SimpleNamespace(returncode=1)
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(rp.subprocess, "run", failing_run)

    rc = rp.run_pipeline(["--yes"])
    assert rc == 1
    state = rp.load_state()
    assert state["steps"]["acquire_dukascopy"]["status"] == "failed"
    # prepare_data must never have been attempted after a required upstream failure
    assert "prepare_data" not in state["steps"]


def test_run_pipeline_continues_past_train_eval_failures_and_reports_them(monkeypatch, tmp_path):
    monkeypatch.setattr(rp, "resolve_model_names", lambda cfg_path, only: (["Good Model", "Bad Model"], None))
    monkeypatch.setattr(rp, "load_config", lambda cfg_path: {
        "data": {"dukascopy_file": str(tmp_path / "d.csv"), "forexsb_file": str(tmp_path / "f.csv"),
                 "fred_file": str(tmp_path / "fr.csv"), "processed_dir": str(tmp_path)},
        "logging": {"results_dir": str(tmp_path)},
    })
    for f in ("d.csv", "f.csv", "fr.csv"):
        (tmp_path / f).write_text("x")
    for prefix in ("X", "y"):
        for split in ("train", "val", "test"):
            (tmp_path / f"{prefix}_{split}.npy").write_bytes(b"x")

    def selective_run(cmd, stdout=None, stderr=None):
        # Simulate each subprocess's real side effect (a CSV row / a
        # chapter4_numbers.json file), since artifact_ok() checks disk,
        # not the mocked return code alone.
        joined = " ".join(cmd)
        if "Bad Model" in joined:
            return types.SimpleNamespace(returncode=1)
        if "Good Model" in joined:
            (tmp_path / "all_results.csv").write_text("model_type\nGood Model\n")
        if "verify_chapter4" in joined:
            (tmp_path / "chapter4_numbers.json").write_text("{}")
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(rp.subprocess, "run", selective_run)

    rc = rp.run_pipeline(["--yes"])
    assert rc == 1  # overall failure reported
    state = rp.load_state()
    assert state["steps"]["train_eval:Good Model"]["status"] == "completed"
    assert state["steps"]["train_eval:Bad Model"]["status"] == "failed"
    # verify_chapter4 still ran despite one model failing
    assert state["steps"]["verify_chapter4"]["status"] == "completed"
