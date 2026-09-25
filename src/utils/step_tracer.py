"""
Structured JSON step tracer, for logging inside a single phase of a run
(e.g. "load data", "fit PCA") at finer grain than RunLogger.log_epoch()'s
per-epoch entries.

This is adapted from a standalone `QGANLogger` class that was proposed for
this project. Only the tracing behaviour is kept here -- NOT that source's
`QuantumGeneratorCircuit`, `QGANTrainer`, `AdversarialAttackEngine`, or
`QuantumTomographyEngine.compute_ancova_pvalue`, which return numbers that
are fabricated rather than computed (see scripts/standalone_demo/ for a
fixed, clearly-separated version of that script). Nothing in this module
computes a metric; it only records values other code already computed.

CHANGES vs. the source `QGANLogger`:
  - Reuses this project's atomic_write_text() for the JSONL append, instead
    of a plain open(...).write(): a crash mid-write can otherwise leave a
    truncated last line that json.loads() can't parse when the trace is
    read back (see src/utils/io.py's docstring for why this matters on
    multi-hour runs).
  - Takes an explicit run_id and writes into RunLogger's log_dir, as
    "{run_id}.trace.jsonl", so a run's three artifacts (.log, .json,
    .trace.jsonl) sit together and share one id, matching the layout
    logging_utils.py already establishes -- rather than a separately
    timestamped file in its own logs/execution_traces/ directory that
    nothing else in the project points to.
  - Every metrics dict is passed through a JSON-safe coercion (numpy
    scalars, torch tensors -> plain float/int) before writing, since
    metrics computed by this project's models are frequently
    torch.Tensor / np.float32 and json.dumps() rejects those outright.
  - Uses this project's RunLogger console/file logger (passed in) instead
    of configuring its own logging.Logger with its own handlers, so trace
    messages land in the SAME {run_id}.log a person is already tailing,
    rather than a second, separately-configured stream.

Run standalone smoke test:  python -m src.utils.step_tracer
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

from .io import atomic_write_text


def _json_safe(value: Any) -> Any:
    """Best-effort coercion of numpy/torch scalars (and containers of
    them) to plain Python types, so json.dumps() never raises on a metric
    a model handed us as np.float32 / a 0-d torch.Tensor."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    return value


def _get_memory_mb() -> Optional[float]:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        return None


class StepTracer:
    """Appends one JSON object per line to logs/{run_id}.trace.jsonl.

    Each line is a genuinely atomic write (temp file + os.replace), so a
    trace file interrupted mid-run ends on the last COMPLETE line rather
    than a half-written one.
    """

    def __init__(self, run_id: str, log_dir: str = "logs", logger=None):
        self.run_id = run_id
        self.path = os.path.join(log_dir, f"{run_id}.trace.jsonl")
        os.makedirs(log_dir, exist_ok=True)
        self._logger = logger  # a RunLogger (or anything with .info/.warning), or None

    def log_step(
        self,
        phase: str,
        step_id: int | str,
        execution_time_ms: float,
        metrics: dict,
        message: str,
        severity: str = "INFO",
    ) -> dict:
        entry = {
            "run_id": self.run_id,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "phase": phase,
            "step_id": step_id,
            "execution_time_ms": round(execution_time_ms, 3),
            "memory_usage_mb": (lambda m: round(m, 2) if m is not None else None)(_get_memory_mb()),
            "metrics": _json_safe(metrics),
            "severity": severity,
            "message": message,
        }

        existing = ""
        if os.path.isfile(self.path):
            with open(self.path, "r") as f:
                existing = f.read()
        atomic_write_text(self.path, existing + json.dumps(entry, default=str) + "\n")

        console_msg = f"[{phase}] step {step_id} | {execution_time_ms:.1f}ms | {message} | {entry['metrics']}"
        if self._logger is not None:
            getattr(self._logger, severity.lower(), self._logger.info)(console_msg) \
                if hasattr(self._logger, severity.lower()) else self._logger.info(console_msg)
        else:
            print(console_msg)

        return entry

    def read_all(self) -> list[dict]:
        """Reads every entry written so far. Used by tests and by anything
        that wants to reconstruct the trace after the fact."""
        if not os.path.isfile(self.path):
            return []
        with open(self.path) as f:
            return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":
    # python -m src.utils.step_tracer -- writes a couple of demo lines to
    # logs/_step_tracer_demo.trace.jsonl and reads them back.
    import numpy as np

    tracer = StepTracer(run_id="_step_tracer_demo")
    t0 = time.time()
    tracer.log_step("DEMO_PHASE", 1, (time.time() - t0) * 1000,
                    {"a_numpy_float32": np.float32(1.5), "n": 3}, "demo step ran")
    print(f"wrote {len(tracer.read_all())} entries to {tracer.path}")
