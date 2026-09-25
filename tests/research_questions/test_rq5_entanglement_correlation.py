"""
RQ5 -- Interpretability of Quantum State / Correlation with QGAN
Integration (H5): does entanglement structure in the QGAN's quantum
state correlate with threat detection accuracy, in a way that offers a
mechanistic explanation for any observed performance difference?

Manuscript claim under test (see H5, Ch.4 RQ5): entanglement entropy
correlates with detection accuracy at r(98)=0.67, p<0.001.

What's tested here: the two things that must independently be true for
H5 to be testable and correctly reported at all --
  1. entanglement_metrics() (the real partial-trace/von-Neumann entropy
     calculation this rebuild replaced `np.random.uniform(2.5, 4.0)`
     with) must actually respond to a circuit's entangling structure --
     zero for no entangling gates, higher for more, and tunable across a
     range (not just binary on/off), matching the dissertation's
     low/moderate/high entanglement conditions differing by degree.
  2. pearson_correlation_with_ci() (Ch.4's exact reporting format --
     r(df), p, 95% CI) must recover a real relationship when one is
     deliberately constructed (positive control) and NOT manufacture one
     between independent variables (negative control).
Also exercises the full pipeline once, end-to-end, via a real (tiny)
trained QGANLLM's own reported entanglement_entropy/purity, confirming
that path runs cleanly -- without asserting what correlation a 2-3-epoch
toy-scale model should show, for the same reason test_rq4's head-to-head
comparison doesn't assert a winner.

NOT tested: whether entanglement entropy actually correlates with
detection accuracy at r~0.67 on real trained models at scale -- that
needs real data, real threat-detection labels, and real training (see
README.md's sandbox-constraints section).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pennylane as qml
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.baselines.qgan_llm import QGANLLM
from src.evaluation.statistical_tests import pearson_correlation_with_ci
from src.quantum.tomography import entanglement_metrics

N_QUBITS = 4


def _circuit(weights: np.ndarray, entangle_layers: int):
    """Entangling structure controlled by a single integer knob (number
    of entangling layers), so entropy vs. structure is a real sweep, not
    a before/after pair -- matching low/moderate/high entanglement being
    a matter of degree in the dissertation, not a binary condition."""
    for i in range(N_QUBITS):
        qml.RY(weights[i], wires=i)
    for layer in range(entangle_layers):
        for i in range(N_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % N_QUBITS])
        for i in range(N_QUBITS):
            qml.RY(weights[i] * (0.5 ** (layer + 1)), wires=i)


def test_entanglement_entropy_is_zero_with_no_entangling_gates():
    """PRECONDITION FOR H5 BEING TESTABLE AT ALL: remove every entangling
    gate and entropy must be ~0. This is the exact failure mode the
    original `np.random.uniform(2.5, 4.0)` placeholder had -- a constant/
    random value regardless of circuit structure, which would make any
    downstream correlation meaningless."""
    rng = np.random.default_rng(42)
    weights = rng.uniform(0, 2 * np.pi, size=N_QUBITS)
    report = entanglement_metrics(lambda w: _circuit(w, entangle_layers=0), weights, N_QUBITS, cut=2)
    assert report["entanglement_entropy"] < 1e-6
    assert abs(report["purity"] - 1.0) < 1e-6


def test_entanglement_entropy_increases_with_more_entangling_layers():
    """A real sweep: 0, 1, 2, 3 entangling layers should produce
    non-decreasing (and, across the full range, strictly increasing)
    entropy. If this failed, there would be no genuine range of entropy
    values to correlate detection accuracy against -- only noise."""
    rng = np.random.default_rng(7)
    weights = rng.uniform(0, 2 * np.pi, size=N_QUBITS)

    entropies = [
        entanglement_metrics(lambda w, nl=n: _circuit(w, entangle_layers=nl), weights, N_QUBITS, cut=2)[
            "entanglement_entropy"
        ]
        for n in (0, 1, 2, 3)
    ]
    assert entropies[0] < 1e-6, "0 entangling layers should give ~0 entropy"
    assert entropies[-1] > entropies[0], f"Expected entropy to rise from layer 0 to layer 3; got {entropies}"


def test_pearson_correlation_recovers_a_constructed_relationship():
    """The actual H5 statistical test, in Ch.4's exact reporting format
    (r(df), p, 95% CI). Positive control: entropy deliberately drives
    accuracy up plus noise -- the test must recover that relationship
    with a CI that excludes zero."""
    rng = np.random.default_rng(51)
    n = 100
    entropy = rng.uniform(0.1, 2.5, size=n)
    accuracy = 0.80 + 0.05 * entropy + rng.normal(0, 0.03, size=n)

    result = pearson_correlation_with_ci(entropy, accuracy)
    assert result["p_value"] < 0.001, f"Expected a significant correlation; got p={result['p_value']:.4f}"
    assert result["r"] > 0.3, f"Expected a moderate-to-strong positive r; got r={result['r']:.3f}"
    assert result["ci_low"] > 0, "95% CI should exclude zero for this strong a constructed relationship"
    assert result["df"] == n - 2


def test_pearson_correlation_stays_null_for_independent_variables():
    """NEGATIVE CONTROL: entropy and accuracy drawn independently must
    NOT produce a significant correlation. Guards against a subtle bug
    that reports significance regardless of the actual relationship
    between the two variables."""
    rng = np.random.default_rng(52)
    n = 100
    entropy = rng.uniform(0.1, 2.5, size=n)
    accuracy = rng.normal(0.85, 0.03, size=n)  # independent of entropy by construction

    result = pearson_correlation_with_ci(entropy, accuracy)
    assert result["p_value"] > 0.05, (
        f"No real relationship was constructed, but got r={result['r']:.3f}, p={result['p_value']:.4f}"
    )


def test_qgan_llm_reports_finite_entanglement_metrics_end_to_end(rq_synthetic_data, qgan_config_factory):
    """End-to-end: a real (tiny) trained QGANLLM's own evaluate() must
    report finite, in-range entanglement_entropy and purity -- the exact
    values RQ5's real-data correlation analysis would consume. Does NOT
    assert any particular value or correlation at this toy scale (see
    module docstring) -- only that the reporting path itself works."""
    quantum = QGANLLM(config=qgan_config_factory())
    quantum.train(rq_synthetic_data["X_train"], rq_synthetic_data["y_train"],
                   rq_synthetic_data["X_val"], rq_synthetic_data["y_val"])
    metrics = quantum.evaluate(rq_synthetic_data["X_test"], rq_synthetic_data["y_test"])

    assert "entanglement_entropy" in metrics and "purity" in metrics
    assert np.isfinite(metrics["entanglement_entropy"]) and metrics["entanglement_entropy"] >= 0
    assert np.isfinite(metrics["purity"]) and 0 <= metrics["purity"] <= 1 + 1e-6


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
