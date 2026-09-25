"""
RQ2 -- Adversarial Resilience by Entanglement Level (H2): does higher
qubit entanglement in the QGAN generator correlate with lower Attack
Success Rate (ASR) against the downstream forecaster?

Manuscript claim under test (see H2): QGANs configured with highly
entangled generator circuits produce synthetic data that leads to a
statistically significant reduction in ASR against gradient-based
adversarial attacks (FGSM/PGD/CW), compared to low-entanglement
configurations.

What's tested here: the attack pipeline's basic correctness (ASR in
[0,100], attacks behave monotonically with perturbation budget), that
different entanglement topologies (ring/full/linear) all run without
error end-to-end, and that the chi-square/t-test machinery used to
compare "high entanglement" vs "low entanglement" ASR groups is
correctly computing a valid test statistic and p-value. NOT tested:
whether entanglement actually reduces ASR on real data at this toy
scale/epoch count -- with 2-3 epochs of training on random data, there
is no real signal to detect, and this file does not assert one exists.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.attacks.adversarial import compute_attack_success_rate
from src.baselines.qgan_llm import QGANLLM
from src.evaluation.statistical_tests import chi_square_test, independent_ttest


def test_asr_is_a_valid_percentage(rq_synthetic_data, qgan_config_factory, attack_config_factory):
    model = QGANLLM(config=qgan_config_factory())
    model.train(rq_synthetic_data["X_train"], rq_synthetic_data["y_train"],
                rq_synthetic_data["X_val"], rq_synthetic_data["y_val"])
    last_prices = rq_synthetic_data["X_test"][:, 0]
    metrics = model.evaluate(rq_synthetic_data["X_test"], rq_synthetic_data["y_test"],
                              attack_cfg=attack_config_factory(), last_input_prices=last_prices)
    assert 0.0 <= metrics["asr"] <= 100.0
    assert "asr_breakdown" in metrics
    assert set(metrics["asr_breakdown"].keys()) >= {"overall_asr"}


def test_asr_pipeline_runs_across_entanglement_topologies(rq_synthetic_data, qgan_config_factory, attack_config_factory):
    """H2 specifically compares topologies (ring/full/linear -- 'high' vs
    'low' entanglement structure). Confirms all three actually train and
    evaluate without error, since a topology that silently crashed would
    be a much bigger problem than one that trains to a different number."""
    last_prices = rq_synthetic_data["X_test"][:, 0]
    results = {}
    for topology in ("ring", "full", "linear"):
        model = QGANLLM(config=qgan_config_factory(entanglement=topology))
        model.train(rq_synthetic_data["X_train"], rq_synthetic_data["y_train"],
                    rq_synthetic_data["X_val"], rq_synthetic_data["y_val"])
        metrics = model.evaluate(rq_synthetic_data["X_test"], rq_synthetic_data["y_test"],
                                  attack_cfg=attack_config_factory(), last_input_prices=last_prices)
        results[topology] = metrics
        assert 0.0 <= metrics["asr"] <= 100.0
        assert metrics["entanglement_entropy"] >= 0

    # Full entanglement (all-to-all CNOTs) should, by construction, reach at
    # least as much entanglement entropy as ring or linear on an untrained
    # (or lightly trained) circuit with the same random init scale -- this
    # is a structural property of the circuit, not a claim about ASR.
    assert results["full"]["entanglement_entropy"] >= 0  # sanity: didn't error/NaN out


def test_chi_square_comparison_of_two_asr_groups_is_valid():
    """H2's statistical test for comparing high- vs low-entanglement ASR
    is a chi-square test on a 2x2 contingency table (attack succeeded /
    failed, by entanglement group) per the dissertation's Table 42. This
    checks that comparison machinery directly, independent of any actual
    trained model."""
    # Contingency table: rows = [high entanglement, low entanglement],
    # columns = [attack succeeded, attack failed]. Constructed so high
    # entanglement has a visibly lower success rate, to confirm the test
    # correctly detects an association when one is present.
    table = np.array([
        [9, 91],    # high entanglement: 9% ASR
        [31, 69],   # low entanglement: 31% ASR
    ])
    result = chi_square_test(table)
    assert 0 <= result["p_value"] <= 1
    assert result["p_value"] < 0.05, (
        "This table was constructed with an obvious 9% vs 31% gap -- if "
        "chi_square_test doesn't call this significant, something in the "
        "test itself is wrong, independent of any real experiment."
    )


def test_independent_ttest_effect_size_direction():
    """Cross-check with a t-test (used elsewhere in Ch.4 for continuous
    DVs) on a case with a known, large effect size, to confirm Cohen's d
    comes out with the correct sign and magnitude."""
    rng = np.random.default_rng(0)
    high_entanglement_asr = rng.normal(loc=9.0, scale=2.0, size=50)
    low_entanglement_asr = rng.normal(loc=31.0, scale=4.0, size=50)
    result = independent_ttest(high_entanglement_asr, low_entanglement_asr)
    assert result["cohens_d"] < 0, "high < low ASR should give a negative Cohen's d for (high - low)"
    assert abs(result["cohens_d"]) > 0.8, "a 9 vs 31 mean gap at this scale should be a large effect size"
    assert result["p_value"] < 0.05


def test_fgsm_perturbation_budget_is_monotonic_in_epsilon(rq_synthetic_data, qgan_config_factory, attack_config_factory):
    """Basic attack-strength sanity: a larger perturbation budget (epsilon)
    should never make FGSM strictly less successful on average, for the
    same model. Not a statistical guarantee for any single run (FGSM is
    a single-step attack and can behave non-monotonically point-by-point),
    but checked here across a full test set average, where it should hold."""
    model = QGANLLM(config=qgan_config_factory())
    model.train(rq_synthetic_data["X_train"], rq_synthetic_data["y_train"],
                rq_synthetic_data["X_val"], rq_synthetic_data["y_val"])
    model.forecast_head.eval()

    X_test_t = torch.tensor(rq_synthetic_data["X_test"], dtype=torch.float32)
    y_test_t = torch.tensor(rq_synthetic_data["y_test"], dtype=torch.float32)
    last_prices_t = torch.tensor(rq_synthetic_data["X_test"][:, 0], dtype=torch.float32)

    small_eps_report = compute_attack_success_rate(
        model._forecast_model, X_test_t, y_test_t, last_prices_t,
        attack_config_factory(fgsm_epsilon=0.01), attacks=["fgsm"],
    )
    large_eps_report = compute_attack_success_rate(
        model._forecast_model, X_test_t, y_test_t, last_prices_t,
        attack_config_factory(fgsm_epsilon=0.5), attacks=["fgsm"],
    )
    assert large_eps_report["overall_asr"] >= small_eps_report["overall_asr"], (
        f"A larger perturbation budget produced a LOWER attack success rate "
        f"({large_eps_report['overall_asr']} < {small_eps_report['overall_asr']}) -- "
        f"that would indicate a sign error in the FGSM implementation."
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
