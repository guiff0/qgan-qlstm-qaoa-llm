"""Tests for src/utils/step_tracer.py."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.step_tracer import StepTracer, _json_safe  # noqa: E402


def test_log_step_appends_jsonl(tmp_path):
    tracer = StepTracer("run_a", log_dir=str(tmp_path))
    tracer.log_step("PHASE_1", 1, 12.345, {"loss": 0.5}, "first step")
    tracer.log_step("PHASE_1", 2, 7.0, {"loss": 0.4}, "second step")
    lines = (tmp_path / "run_a.trace.jsonl").read_text().splitlines()
    assert len(lines) == 2
    e0, e1 = json.loads(lines[0]), json.loads(lines[1])
    assert e0["step_id"] == 1 and e1["step_id"] == 2
    assert e0["execution_time_ms"] == 12.345
    assert e0["run_id"] == "run_a"


def test_numpy_and_torch_like_scalars_are_json_safe(tmp_path):
    tracer = StepTracer("run_b", log_dir=str(tmp_path))
    entry = tracer.log_step("P", 1, 1.0, {"a": np.float32(1.5), "b": np.int64(3),
                                          "nested": {"c": np.float64(2.25)}}, "m")
    assert entry["metrics"] == {"a": 1.5, "b": 3, "nested": {"c": 2.25}}
    # round-trips through actual json.dumps without raising
    json.loads((tmp_path / "run_b.trace.jsonl").read_text())


def test_torch_tensor_scalar_is_coerced():
    class FakeTensor:  # avoid a hard torch dependency in this test
        def item(self):
            return 3.0
    assert _json_safe({"x": FakeTensor()}) == {"x": 3.0}


def test_read_all_returns_every_entry(tmp_path):
    tracer = StepTracer("run_c", log_dir=str(tmp_path))
    for i in range(5):
        tracer.log_step("P", i, 1.0, {"i": i}, "m")
    entries = tracer.read_all()
    assert [e["step_id"] for e in entries] == [0, 1, 2, 3, 4]


def test_read_all_on_missing_file_returns_empty_list(tmp_path):
    tracer = StepTracer("nope", log_dir=str(tmp_path))
    assert tracer.read_all() == []


def test_write_is_atomic_no_tmp_files_left_behind(tmp_path):
    tracer = StepTracer("run_d", log_dir=str(tmp_path))
    for i in range(10):
        tracer.log_step("P", i, 1.0, {}, "m")
    assert os.listdir(tmp_path) == ["run_d.trace.jsonl"]           # no leftover .tmp* files


def test_partial_crash_leaves_previous_lines_intact(tmp_path, monkeypatch):
    """Simulates a crash mid-write of the 3rd line: the file must still
    contain the first two COMPLETE lines, not a truncated 3rd one."""
    tracer = StepTracer("run_e", log_dir=str(tmp_path))
    tracer.log_step("P", 1, 1.0, {}, "m1")
    tracer.log_step("P", 2, 1.0, {}, "m2")

    import src.utils.step_tracer as st

    def boom(path, text):
        raise OSError("simulated crash mid-write")
    monkeypatch.setattr(st, "atomic_write_text", boom)
    with pytest.raises(OSError):
        tracer.log_step("P", 3, 1.0, {}, "m3")

    lines = (tmp_path / "run_e.trace.jsonl").read_text().splitlines()
    assert len(lines) == 2
    for line in lines:
        json.loads(line)   # every remaining line is still valid, complete JSON


def test_different_run_ids_write_separate_files(tmp_path):
    StepTracer("a", log_dir=str(tmp_path)).log_step("P", 1, 1.0, {}, "m")
    StepTracer("b", log_dir=str(tmp_path)).log_step("P", 1, 1.0, {}, "m")
    assert sorted(os.listdir(tmp_path)) == ["a.trace.jsonl", "b.trace.jsonl"]


def test_uses_provided_logger_instead_of_print(tmp_path, capsys):
    calls = []
    class FakeRunLogger:
        def info(self, msg):
            calls.append(("info", msg))
        def warning(self, msg):
            calls.append(("warning", msg))
    tracer = StepTracer("run_f", log_dir=str(tmp_path), logger=FakeRunLogger())
    tracer.log_step("P", 1, 1.0, {"x": 1}, "hello", severity="WARNING")
    assert calls and calls[0][0] == "warning" and "hello" in calls[0][1]
    assert capsys.readouterr().out == ""            # nothing printed directly when a logger is given


def test_no_logger_falls_back_to_print(tmp_path, capsys):
    tracer = StepTracer("run_g", log_dir=str(tmp_path), logger=None)
    tracer.log_step("P", 1, 1.0, {}, "hello")
    assert "hello" in capsys.readouterr().out
