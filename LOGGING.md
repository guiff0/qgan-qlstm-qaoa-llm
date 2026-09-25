# LOGGING.md — Structured Run Logs and Step Traces

Every run of `src/experiments/run_all.py` writes three files, all keyed by
the same `run_id` (see `src/utils/logging_utils.py`), plus a fourth added
here:

| File | Written by | Contains |
|---|---|---|
| `logs/{run_id}.log` | `RunLogger` | Human-readable log lines |
| `logs/{run_id}.json` | `RunLogger.save_json()` | Config used, per-epoch training history, final metrics, environment fingerprint |
| `results/all_results.csv` | `RunLogger.append_to_results_csv()` | One row per run, for `results_collector.py` / `verify_chapter4.py` |
| `logs/{run_id}.trace.jsonl` | `StepTracer` (new) | One JSON object per phase (train, evaluate, ...), with wall-clock time and memory at that point |

## Why a separate trace file

`RunLogger.log_epoch()` records one row per *training epoch*. It doesn't
tell you, after the fact, how long data loading took versus training
versus evaluation versus latency measurement, or what the process's
memory looked like at each of those boundaries. `StepTracer` fills that
gap at the phase level. It does not replace `RunLogger`, and it does not
compute anything itself — it only timestamps and records values other
code already computed.

## Trace line format

Each line in `{run_id}.trace.jsonl` is one JSON object:

```json
{
  "run_id": "classical_lstm_20260921t014119z",
  "timestamp_utc": "2026-09-21T01:41:19Z",
  "phase": "TRAIN",
  "step_id": 1,
  "execution_time_ms": 172440000.0,
  "memory_usage_mb": 1822.4,
  "metrics": {"reused_checkpoint": false, "n_train_rows": 3057984},
  "severity": "INFO",
  "message": "Trained Classical LSTM from scratch"
}
```

`memory_usage_mb` is `null` if `psutil` isn't installed (it's already a
required package per `SETUP.md`/`verify_environment.py`, so this should
not normally happen). Numpy/PyTorch scalar metrics (`np.float32`,
0-dimensional tensors, etc.) are coerced to plain Python numbers before
writing, since `json.dumps` rejects them directly.

## Reading a trace back

```python
from src.utils.step_tracer import StepTracer
entries = StepTracer(run_id="classical_lstm_20260921t014119z").read_all()
for e in entries:
    print(e["phase"], e["execution_time_ms"], e["metrics"])
```

## What is intentionally NOT here

An earlier proposal for this file included specific runtime numbers for
`lightning.gpu` vs. `default.qubit` (a claimed "320.5 minutes down to
49.8 minutes, an 84.5% speedup") and a sub-3.2ms live-inference latency
figure. Those are not included here because they were not measured on
this codebase or this hardware — they were asserted, not benchmarked. If
you want that comparison documented, run both device backends with
`StepTracer` active and this file can be updated with the real numbers
and the hardware/config they came from. See `RUNNING.md` for the caution
already given there about `default.qubit` being CPU-only and about not
trusting any specific timing figure that isn't tied to an actual run.
