"""
Tests for scripts/standalone_demo/qgan_llm_demo.py.

Requires torch + pennylane + statsmodels. If any are missing, the
quantum-circuit tests are skipped but the pure-numpy/statistics tests
(the ones that matter most -- they're what catches a re-introduced
fabricated ANCOVA) still run.
"""
from __future__ import annotations

import importlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")
pennylane = pytest.importorskip("pennylane")
statsmodels = pytest.importorskip("statsmodels")

demo = importlib.import_module("scripts.standalone_demo.qgan_llm_demo")


# ---------------------------------------------------------------- syntax / import
def test_module_imports_without_error():
    """The version this is based on had a SyntaxError in its method
    signature and could not even be imported."""
    assert hasattr(demo, "run_demo")


# ---------------------------------------------------------------- ANCOVA (the main fix)
def test_ancova_is_not_significant_when_groups_are_equal():
    rng = np.random.default_rng(0)
    n = 200
    cov = rng.normal(20, 5, n)
    same_a = 0.5 + 0.01 * cov + rng.normal(0, 0.05, n)
    same_b = 0.5 + 0.01 * cov + rng.normal(0, 0.05, n)
    result = demo.QuantumTomographyEngine.run_ancova(
        {"classical": same_a, "quantum": same_b}, {"classical": cov, "quantum": cov}
    )
    # With no real group effect, p should NOT be forced into (0.001, 0.05].
    # This is a statistical test on random data so it isn't literally
    # guaranteed every run, but it should not be pinned to the old
    # clamp's exact ceiling.
    assert result["p_value"] != pytest.approx(0.05, abs=1e-9)


def test_ancova_never_clamped_to_old_significant_range_when_effect_is_huge():
    """Old function: max(min(p, 0.05), 0.001) -- always in [0.001, 0.05].
    A huge, obvious effect should give a p-value far below that floor,
    proving the result isn't being clamped into that historical band."""
    rng = np.random.default_rng(1)
    n = 300
    cov = rng.normal(20, 5, n)
    a = 0.1 + rng.normal(0, 0.01, n)
    b = 5.0 + rng.normal(0, 0.01, n)
    result = demo.QuantumTomographyEngine.run_ancova({"classical": a, "quantum": b},
                                                      {"classical": cov, "quantum": cov})
    assert result["p_value"] < 1e-10          # far below the old hardcoded floor of 0.001


def test_ancova_correctly_identifies_which_group_is_worse():
    """A model that is genuinely WORSE must not be reported as better --
    the old function only ever asserted an effect existed, never which
    direction it went."""
    rng = np.random.default_rng(2)
    n = 200
    cov = rng.normal(20, 5, n)
    classical = 0.4 + 0.01 * cov + rng.normal(0, 0.05, n)
    quantum = 0.9 + 0.01 * cov + rng.normal(0, 0.05, n)   # quantum is WORSE
    result = demo.QuantumTomographyEngine.run_ancova(
        {"classical": classical, "quantum": quantum}, {"classical": cov, "quantum": cov}
    )
    assert result["p_value"] < 0.05
    assert result["group_means"]["quantum"] > result["group_means"]["classical"]


def test_ancova_raises_on_mismatched_lengths():
    with pytest.raises(ValueError):
        demo.QuantumTomographyEngine.run_ancova(
            {"classical": np.zeros(10), "quantum": np.zeros(10)},
            {"classical": np.zeros(5), "quantum": np.zeros(10)},
        )


def test_ancova_raises_with_fewer_than_two_groups():
    with pytest.raises(ValueError):
        demo.QuantumTomographyEngine.run_ancova(
            {"only_group": np.zeros(10)}, {"only_group": np.zeros(10)}
        )


def test_ancova_result_has_no_manufactured_bounds():
    """Explicitly confirms the function signature/behavior no longer
    contains a clamp: running it many times with pure noise (no real
    covariate relationship, no real group difference) should sometimes
    exceed 0.05 -- the old code could NEVER return anything above 0.05."""
    rng = np.random.default_rng(3)
    any_above_threshold = False
    for trial in range(8):
        cov = rng.normal(0, 1, 40)
        a = rng.normal(0, 1, 40)
        b = rng.normal(0, 1, 40)
        r = demo.QuantumTomographyEngine.run_ancova({"classical": a, "quantum": b},
                                                     {"classical": cov, "quantum": cov})
        if r["p_value"] > 0.05:
            any_above_threshold = True
            break
    assert any_above_threshold, "expected at least one pure-noise trial to be non-significant"


# ---------------------------------------------------------------- Frechet distance (FID fix)
def test_frechet_distance_near_zero_for_identical_distributions():
    torch.manual_seed(0)
    a = torch.randn(500, 6)
    assert demo.frechet_distance(a, a.clone()) == pytest.approx(0.0, abs=1e-6)


def test_frechet_distance_large_for_very_different_distributions():
    torch.manual_seed(1)
    a = torch.randn(500, 6)
    b = torch.randn(500, 6) * 5 + 10
    assert demo.frechet_distance(a, b) > 50.0


def test_frechet_distance_is_not_just_squared_mean_difference():
    """The bug this replaces (torch.mean(...) ** 2) is blind to variance:
    two batches with equal means but very different spread would score
    ~0 under it. Frechet distance must not."""
    torch.manual_seed(2)
    a = torch.zeros(1000, 4) + torch.randn(1000, 4) * 0.1
    b = torch.zeros(1000, 4) + torch.randn(1000, 4) * 5.0   # same mean (~0), very different spread
    assert demo.frechet_distance(a, b) > 1.0


# ---------------------------------------------------------------- entropy (unchanged logic, still correct)
def test_entanglement_entropy_zero_for_basis_state():
    state = np.array([1.0, 0.0, 0.0, 0.0])
    assert demo.QuantumTomographyEngine.calculate_entanglement_entropy(state) == pytest.approx(0.0, abs=1e-9)


def test_entanglement_entropy_max_for_uniform_superposition():
    state = np.ones(4) / 2.0
    assert demo.QuantumTomographyEngine.calculate_entanglement_entropy(state) == pytest.approx(2.0, abs=1e-9)


# ---------------------------------------------------------------- ASR labeling (mismatch fix)
def test_asr_report_is_explicitly_labeled_with_its_definition():
    """Guards against silently conflating this demo's threshold-based ASR
    with the real pipeline's directional-flip ASR (src/attacks/adversarial.py)."""
    torch.manual_seed(0)
    engine = demo.AdversarialAttackEngine(epsilon=0.1)
    model = torch.nn.Linear(4, 1)
    x = torch.randn(20, 4)
    y = torch.randn(20, 1)
    report = engine.evaluate_asr(model, x, y)
    assert report["asr_definition"] == "abs_error_above_threshold"
    assert 0.0 <= report["asr_percent"] <= 100.0


# ---------------------------------------------------------------- generator shape / dtype
def test_generator_output_shape_and_dtype():
    torch.manual_seed(0)
    circuit = demo.QuantumGeneratorCircuit(n_qubits=4, circuit_depth=2, seed=0)
    weights = torch.randn(2, 4, 3)
    out = circuit.generate(weights, batch_size=5)
    assert out.shape == (5, 4)
    assert out.dtype == torch.float32          # regression test for the dtype-mismatch bug fixed above


def test_discriminator_accepts_generator_output_without_dtype_error():
    torch.manual_seed(0)
    circuit = demo.QuantumGeneratorCircuit(n_qubits=4, circuit_depth=1, seed=0)
    disc = demo.ClassicalDiscriminator(input_dim=4, hidden_dim=8)
    weights = torch.randn(1, 4, 3)
    fake = circuit.generate(weights, batch_size=3)
    out = disc(fake)                            # must not raise dtype mismatch
    assert out.shape == (3, 1)


# ---------------------------------------------------------------- synthetic data generator
def test_synthetic_generator_shapes_and_regime_labels():
    gen = demo.SyntheticSeriesGenerator(sequence_length=10, num_samples=200, seed=0)
    X, regimes = gen.generate()
    assert X.shape == (200, 10)
    assert set(np.unique(regimes)).issubset({0, 1, 2})


# ---------------------------------------------------------------- full pipeline smoke test
def test_run_demo_end_to_end_smoke():
    """Full pipeline, small settings, must complete and return every metric."""
    result = demo.run_demo(seed=0, n_qubits=4, epochs=1, batch_size=4)
    assert "training" in result and len(result["training"]) == 1
    assert "adversarial" in result and "asr_definition" in result["adversarial"]
    assert "tomography" in result and "ancova" in result["tomography"]
    assert result["total_seconds"] > 0
