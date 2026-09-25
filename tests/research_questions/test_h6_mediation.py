"""
H6 mediation pathways (M2->DV2, M4->DV4) -- see src/evaluation/mediation.py
and mediation_pipelines.py.

Same philosophy as tests/research_questions/: positive control (a real
indirect effect must be detected) and negative control (no real indirect
effect must NOT be reported as significant). A passing test here means
"the mediation math and the FPR confusion-matrix math are both correct,"
NOT "this study's model configuration actually has these effects" --
that still requires real data and real trained models.

M1->DV1 and M5->DV5 are deliberately absent -- their operational
definitions (Encoding Fidelity Index, Quantum State Feature Index) are
still pending confirmation; testing undefined constructs would just be
testing whatever this file guessed they meant.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.evaluation.mediation import mediation_analysis
from src.evaluation.mediation_pipelines import (
    run_m2_dv2_mediation,
    run_m4_dv4_mediation,
    encode_entanglement_level,
    encode_asr_category,
)
from src.evaluation.metrics import classification_metrics


# ---------------------------------------------------------------------------
# Mediation engine: does the Baron & Kenny + bootstrap math work at all?
# ---------------------------------------------------------------------------

def test_mediation_detects_a_real_full_mediation_effect():
    """POSITIVE CONTROL: construct IV -> M -> DV by hand (IV has NO direct
    effect on DV except through M) and confirm the engine recovers a
    significant indirect effect whose CI excludes zero, with the direct
    effect (c') near zero -- i.e. it should classify this as full mediation."""
    rng = np.random.default_rng(1)
    n = 300
    iv = rng.integers(0, 2, size=n).astype(float)          # binary IV
    mediator = 2.0 * iv + rng.normal(0, 0.5, size=n)         # IV strongly drives M
    dv = 3.0 * mediator + rng.normal(0, 0.5, size=n)         # M drives DV; IV has NO direct term

    result = mediation_analysis(iv, mediator, dv, n_bootstrap=2000, seed=1)

    assert result["significant"] is True
    assert result["ci_low"] > 0, "Indirect effect should be reliably positive"
    assert result["mediation_type"] == "full", (
        f"Expected full mediation (no direct IV->DV term was constructed); "
        f"got path_c={result['path_c']:.3f}, path_c_prime={result['path_c_prime']:.3f}"
    )
    assert result["proportion_mediated"] > 0.8


def test_mediation_stays_null_when_iv_does_not_affect_mediator():
    """NEGATIVE CONTROL: IV affects DV directly, but IV has NO effect on
    the mediator (mediator is pure noise w.r.t. IV). The indirect effect
    must NOT be reported as significant."""
    rng = np.random.default_rng(2)
    n = 300
    iv = rng.integers(0, 2, size=n).astype(float)
    mediator = rng.normal(0, 1, size=n)                      # independent of IV
    dv = 2.0 * iv + rng.normal(0, 0.5, size=n)                # direct effect only

    result = mediation_analysis(iv, mediator, dv, n_bootstrap=2000, seed=2)

    assert result["significant"] is False, (
        f"Mediator was constructed independent of IV; got a 'significant' indirect "
        f"effect CI=({result['ci_low']:.3f}, {result['ci_high']:.3f})"
    )
    assert result["mediation_type"] == "none"


def test_mediation_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        mediation_analysis(np.zeros(10), np.zeros(9), np.zeros(10))


def test_mediation_rejects_too_few_observations():
    with pytest.raises(ValueError):
        mediation_analysis(np.zeros(5), np.zeros(5), np.zeros(5))


# ---------------------------------------------------------------------------
# Real FPR / classification metrics
# ---------------------------------------------------------------------------

def test_classification_metrics_matches_hand_computed_confusion_matrix():
    """Hand-constructed confusion matrix: 10 threat, 10 benign samples,
    known TP/FP/TN/FN by construction -- confirms the formulas, not just
    that the function runs."""
    y_true = np.array([1] * 10 + [0] * 10)
    # 8/10 threats correctly flagged (TP=8, FN=2); 3/10 benign wrongly flagged (FP=3, TN=7)
    y_pred_proba = np.array([0.9] * 8 + [0.3] * 2 + [0.8] * 3 + [0.2] * 7)

    result = classification_metrics(y_true, y_pred_proba, threshold=0.5)

    assert result["tp"] == 8 and result["fn"] == 2
    assert result["fp"] == 3 and result["tn"] == 7
    assert abs(result["fpr"] - 3 / 10) < 1e-9
    assert abs(result["recall"] - 8 / 10) < 1e-9
    assert abs(result["precision"] - 8 / 11) < 1e-9


def test_classification_metrics_perfect_classifier_has_zero_fpr():
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred_proba = np.array([0.99, 0.95, 0.90, 0.10, 0.05, 0.01])
    result = classification_metrics(y_true, y_pred_proba, threshold=0.5)
    assert result["fpr"] == 0.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert abs(result["auprc"] - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Pathway-specific wiring: M2->DV2 and M4->DV4
# ---------------------------------------------------------------------------

def test_encode_entanglement_level_matches_table_18_definition():
    codes = encode_entanglement_level(["ring", "full", "linear", "ring"])
    assert list(codes) == [1.0, 1.0, 0.0, 1.0]


def test_encode_asr_category_uses_documented_thresholds():
    codes = encode_asr_category([5.0, 15.0, 45.0, "Low", "High"])
    assert list(codes) == [0.0, 1.0, 2.0, 0.0, 2.0]


def test_m2_dv2_pathway_detects_a_constructed_entanglement_effect():
    """POSITIVE CONTROL for the wired pathway: quantum configs with
    ring/full topology constructed to have systematically lower ASR
    category than quantum configs with linear topology, mediating an
    IV effect that has no other direct path to DV2."""
    rng = np.random.default_rng(3)
    n = 200
    model_config = rng.integers(0, 2, size=n)
    # Quantum (IV=1) configs get ring/full more often; classical (IV=0) -> linear
    topology = np.where(
        model_config == 1,
        rng.choice(["ring", "full"], size=n),
        "linear",
    )
    m2 = encode_entanglement_level(topology)
    # Higher entanglement (M2=1) drives ASR down (better resilience)
    asr_pct = 35 - 25 * m2 + rng.normal(0, 3, size=n)
    asr_pct = np.clip(asr_pct, 0, 100)

    result = run_m2_dv2_mediation(model_config, topology, asr_pct, n_bootstrap=1500, seed=3)

    assert result["pathway"] == "M2 -> DV2 (decomposes H2)"
    assert result["significant"] is True
    assert result["indirect_effect"] < 0, "Higher entanglement should drive ASR category DOWN in this construction"


def test_m4_dv4_pathway_detects_a_constructed_overhead_effect():
    """POSITIVE CONTROL: quantum configuration (IV=1) adds computational
    overhead (M4), and that overhead is what drives up total latency
    (DV4) -- IV has no other direct effect on latency in this
    construction, so this should read as full mediation."""
    rng = np.random.default_rng(4)
    n = 200
    model_config = rng.integers(0, 2, size=n).astype(float)
    overhead_ms = 4.5 * model_config + rng.normal(0, 0.3, size=n)  # quantum adds real overhead
    baseline_latency_ms = 0.05 + rng.normal(0, 0.01, size=n)
    total_latency_ms = baseline_latency_ms + overhead_ms

    result = run_m4_dv4_mediation(model_config, overhead_ms, total_latency_ms, n_bootstrap=1500, seed=4)

    assert result["pathway"] == "M4 -> DV4 (decomposes H4)"
    assert result["significant"] is True
    assert result["indirect_effect"] > 0
    assert result["mediation_type"] == "full"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


# ---------------------------------------------------------------------------
# threat_labels.py (Option C) and the M3 -> DV3 pathway
# ---------------------------------------------------------------------------

def test_threat_benign_dataset_has_correct_sizes_and_labels():
    import torch
    from src.attacks.threat_labels import build_threat_benign_dataset

    torch.manual_seed(0)
    model = torch.nn.Linear(8, 1)
    X = torch.randn(50, 8)
    y = torch.randn(50)

    dataset = build_threat_benign_dataset(
        model, X, y, fgsm_epsilon=0.1, pgd_epsilon=0.1, pgd_alpha=0.02, pgd_steps=3,
        n_clean=20, n_fgsm=10, n_pgd=10, seed=0,
    )
    assert dataset["X"].shape[0] == 40
    assert int(dataset["y_threat"].sum().item()) == 20  # 10 fgsm + 10 pgd labeled threat
    assert dataset["source"].count("clean") == 20
    assert dataset["source"].count("fgsm") == 10
    assert dataset["source"].count("pgd") == 10


def test_threat_benign_dataset_raises_when_pool_too_small():
    import torch
    from src.attacks.threat_labels import build_threat_benign_dataset

    model = torch.nn.Linear(8, 1)
    X = torch.randn(10, 8)
    y = torch.randn(10)
    with pytest.raises(ValueError):
        build_threat_benign_dataset(model, X, y, fgsm_epsilon=0.1, pgd_epsilon=0.1,
                                     pgd_alpha=0.02, pgd_steps=3, n_clean=20, n_fgsm=10, n_pgd=10)


def test_m3_dv3_pathway_detects_a_constructed_noise_effect():
    """POSITIVE CONTROL: quantum configs get a range of noise_strength
    values constructed to drive FPR down (noise as defensive feature,
    per RQ3's own framing), mediating an IV effect with no other direct
    path to DV3."""
    rng = np.random.default_rng(5)
    n = 150
    model_config = rng.integers(0, 2, size=n).astype(float)
    noise_strength = np.where(model_config == 1, rng.uniform(0.0, 0.05, size=n), 0.0)
    fpr_per_run = 0.15 - 2.0 * noise_strength + rng.normal(0, 0.01, size=n)
    fpr_per_run = np.clip(fpr_per_run, 0, 1)

    from src.evaluation.mediation_pipelines import run_m3_dv3_mediation
    result = run_m3_dv3_mediation(model_config, noise_strength, fpr_per_run, n_bootstrap=1500, seed=5)

    assert result["pathway"] == "M3 -> DV3 (decomposes H3)"
    assert result["significant"] is True
    assert result["indirect_effect"] < 0, "Higher noise should drive FPR DOWN in this construction"


def test_compute_fpr_per_run_matches_classification_metrics():
    from src.evaluation.mediation_pipelines import compute_fpr_per_run
    y_true = np.array([1] * 10 + [0] * 10)
    y_pred_proba = np.array([0.9] * 8 + [0.3] * 2 + [0.8] * 3 + [0.2] * 7)
    fpr = compute_fpr_per_run(y_true, y_pred_proba, threshold=0.5)
    assert abs(fpr - 3 / 10) < 1e-9
