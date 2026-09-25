"""
Inference latency measurement (DV4 in Ch.3's variable table).

Not present at all in the original code, despite Chapter 4 reporting
specific mean/SD latency figures (39.2ms, SD=4.1 for QGAN-LLM). Measures
wall-clock time for a single forward pass, repeated across the test
set, and reports mean/SD in milliseconds so the result can be compared
directly against the dissertation's stated internal target rather than
an invented number.
"""
from __future__ import annotations

import time

import numpy as np
import torch


def measure_inference_latency(model_callable, X_test: np.ndarray, n_repeats: int = 100,
                               warmup: int = 10, device: str = "cpu") -> dict:
    """
    Times single-sample inference (matching Ch.3's operational definition:
    'processing time from input to model output' per prediction), repeated
    n_repeats times on randomly drawn test samples, after a warmup period
    so one-time initialization cost isn't mixed into the measurement.
    """
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    n = X_t.shape[0]
    rng = np.random.default_rng(42)

    # Timed under torch.no_grad(): autograd's graph-building overhead is
    # part of *training* cost, not inference latency, and the dissertation
    # reports this as an inference-time (deployment) figure. Without this,
    # every timed call here would carry gradient-tracking overhead that a
    # real deployed forecasting service would not pay, inflating every
    # measured latency number.
    with torch.no_grad():
        # Warmup (JIT/caching effects, first-call overhead)
        for _ in range(warmup):
            idx = rng.integers(0, n)
            _ = model_callable(X_t[idx : idx + 1])

        times_ms = []
        for _ in range(n_repeats):
            idx = rng.integers(0, n)
            sample = X_t[idx : idx + 1]
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model_callable(sample)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            times_ms.append(elapsed_ms)

    times_ms = np.array(times_ms)
    return {
        "mean_latency_ms": float(times_ms.mean()),
        "sd_latency_ms": float(times_ms.std(ddof=1)),
        "min_latency_ms": float(times_ms.min()),
        "max_latency_ms": float(times_ms.max()),
        "n_repeats": n_repeats,
        "raw_samples_ms": times_ms.tolist(),  # kept so the actual distribution
                                               # can be checked, not just mean/SD
                                               # (this is what item #6 in the
                                               # dissertation review needed and
                                               # didn't have)
    }
