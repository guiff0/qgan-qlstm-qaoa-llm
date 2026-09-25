"""
Shared fixtures for the RQ1-RQ5 test suite.

======================================================================
WHAT THESE FILES DO AND DO NOT PROVE -- READ THIS FIRST
======================================================================
Each tests/research_questions/test_rqN_*.py file checks that the
PIPELINE for that research question's specific claim is mathematically
correct and runs end-to-end: the right metric or statistical test is
being computed, on correctly-shaped data, producing values in valid
ranges, with sane behavior on cases where the right answer is known in
advance (e.g. FID of a distribution against itself is ~0).

None of these tests can confirm or deny the dissertation's actual
reported numbers (the 26% RMSE reduction, 8.7% ASR, r=0.67 correlation,
etc.) -- that requires the real 5.5M-row EUR/USD dataset and GPU-scale
training this sandbox doesn't have (see README.md). What breaks here is
a genuinely different kind of bug: "the ANCOVA function has a sign
error" or "the FID calculation doesn't actually decrease for more
similar distributions" -- bugs that would produce a WRONG number no
matter how much real data and compute you throw at it. Catching those
here, on data small enough to run in seconds, is what makes it safe to
trust the same functions once pointed at real data.
"""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rq_synthetic_data():
    """Same shape/scale as tests/test_smoke.py's fixture, reused here so
    every RQ file can train a real (tiny) model rather than only testing
    metric functions in isolation."""
    rng = np.random.default_rng(42)
    n_rows, n_features = 400, 8
    X = rng.normal(size=(n_rows, n_features)).astype(np.float32)
    y = (X[:, 0] + 0.1 * rng.normal(size=n_rows)).astype(np.float32)
    split = n_rows // 2
    return {
        "X_train": X[:split], "y_train": y[:split],
        "X_val": X[split: split + n_rows // 4], "y_val": y[split: split + n_rows // 4],
        "X_test": X[split + n_rows // 4:], "y_test": y[split + n_rows // 4:],
        "n_features": n_features,
    }


def tiny_qgan_config(**overrides):
    """A QGANLLM config small enough to train in seconds on CPU. Real
    runs use config/default_config.yaml's qgan_llm section (20 qubits,
    32 features, 30+ epochs) -- this is deliberately much smaller."""
    cfg = {
        "n_qubits": 4, "n_layers": 1, "n_features": 8,
        "entanglement": "ring", "noise_strength": 0.01,
        "discriminator_hidden": 8, "epochs": 3, "batch_size": 16,
    }
    cfg.update(overrides)
    return cfg


def tiny_attack_cfg(**overrides):
    cfg = {
        "fgsm_epsilon": 0.1, "pgd_epsilon": 0.1, "pgd_alpha": 0.01,
        "pgd_steps": 2, "cw_c": 1.0, "cw_steps": 2, "attacks": ["fgsm"],
    }
    cfg.update(overrides)
    return cfg


@pytest.fixture
def qgan_config_factory():
    """Fixture wrapper around tiny_qgan_config so test files can request
    it by name (`def test_x(qgan_config_factory)`) without a fragile
    plain-module import across the tests/research_questions/ package."""
    return tiny_qgan_config


@pytest.fixture
def attack_config_factory():
    return tiny_attack_cfg
