"""
Tests for src/attacks/quantum_attacks.py -- the Quantum Noise Injection
Attack (-> QSFR) and Entanglement Disruption Attack (-> EER) from the
paper's Table 6 companion "Quantum-Specific Attack Vectors" table.

Same philosophy as tests/research_questions/: a passing test here
confirms the attack/metric math is correct on a real (tiny) circuit --
not that a real trained 20-qubit generator has any particular QSFR/EER
value under attack.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.attacks.quantum_attacks import quantum_noise_injection_attack, entanglement_disruption_attack

N_QUBITS = 4
N_LAYERS = 2


@pytest.fixture
def circuit_inputs():
    rng = np.random.default_rng(0)
    inputs = rng.normal(size=N_QUBITS)
    weights = rng.normal(size=N_LAYERS * N_QUBITS * 3) * 0.5
    return inputs, weights


def test_qsfr_is_near_one_for_negligible_attack_strength(circuit_inputs):
    """NEGATIVE CONTROL: an attack with epsilon ~ 0 should leave the
    state almost exactly as it was -- QSFR (a fidelity, bounded [0,1])
    must be ~1."""
    inputs, weights = circuit_inputs
    report = quantum_noise_injection_attack(inputs, weights, N_QUBITS, N_LAYERS, "ring", epsilon=1e-8)
    assert report["qsfr"] > 0.9999


def test_qsfr_decreases_as_attack_strength_increases(circuit_inputs):
    """POSITIVE CONTROL: a real sweep. Larger epsilon (more parameter
    noise injected) must produce lower QSFR (less fidelity retained) --
    if this failed, QSFR would not actually be responding to attack
    strength, the same failure mode as the original _measure_entanglement
    stub this whole rebuild started from."""
    inputs, weights = circuit_inputs
    qsfrs = [
        quantum_noise_injection_attack(inputs, weights, N_QUBITS, N_LAYERS, "ring", epsilon=eps, seed=1)["qsfr"]
        for eps in (0.0, 0.2, 0.5, 1.0)
    ]
    assert qsfrs[0] > 0.999999, "epsilon=0 must give QSFR essentially 1"
    assert qsfrs[-1] < qsfrs[0], f"Expected QSFR to decrease as epsilon grows; got {qsfrs}"
    assert 0 <= min(qsfrs) and max(qsfrs) <= 1.0001


def test_eer_is_near_one_for_negligible_attack_strength(circuit_inputs):
    """NEGATIVE CONTROL: negligible disruption angles should barely
    change the entanglement entropy -- EER (a ratio of two entropies)
    must be ~1."""
    inputs, weights = circuit_inputs
    report = entanglement_disruption_attack(inputs, weights, N_QUBITS, N_LAYERS, "ring", epsilon=1e-8)
    assert abs(report["eer"] - 1.0) < 1e-4


def test_eer_decreases_as_disruption_strength_increases(circuit_inputs):
    """POSITIVE CONTROL: larger disruption-rotation angles should degrade
    (move away from 1.0) the entanglement-entropy ratio more. Checked as
    absolute deviation from 1, since disruption can in principle push
    entropy up or down depending on the random rotation angles drawn."""
    inputs, weights = circuit_inputs
    deviations = []
    for eps in (0.0, 0.3, 0.8, 2.0):
        report = entanglement_disruption_attack(inputs, weights, N_QUBITS, N_LAYERS, "ring", epsilon=eps, seed=2)
        deviations.append(abs(report["eer"] - 1.0))
    assert deviations[0] < 1e-4
    assert deviations[-1] > deviations[0], f"Expected larger disruption at higher epsilon; got deviations {deviations}"


def test_qsfr_and_eer_are_topology_sensitive(circuit_inputs):
    """A basic sanity check that these attacks actually depend on the
    entangling topology passed in, not just the parameter values --
    ring and linear topologies should generally produce different
    clean-state entropy (and therefore potentially different EER under
    the same disruption), confirming the topology argument does
    something rather than being silently ignored."""
    inputs, weights = circuit_inputs
    ring_report = entanglement_disruption_attack(inputs, weights, N_QUBITS, N_LAYERS, "ring", epsilon=0.5, seed=3)
    linear_report = entanglement_disruption_attack(inputs, weights, N_QUBITS, N_LAYERS, "linear", epsilon=0.5, seed=3)
    # Not asserting a specific direction (that's an empirical RQ2/RQ5 question) --
    # only that topology actually changes the clean-state entropy computed.
    assert ring_report["entropy_clean"] != linear_report["entropy_clean"]


def test_invalid_topology_raises():
    rng = np.random.default_rng(0)
    inputs = rng.normal(size=N_QUBITS)
    weights = rng.normal(size=N_LAYERS * N_QUBITS * 3)
    with pytest.raises(ValueError):
        quantum_noise_injection_attack(inputs, weights, N_QUBITS, N_LAYERS, "star", epsilon=0.3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
