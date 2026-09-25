"""
RQ4 -- Quantum Advantage & Speedup (H4): does the QGAN-LLM framework
achieve a statistically significant improvement in forecast accuracy
(RMSE) and adversarial robustness (ASR) over the Classical GAN-LLM
baseline, at a measurable but tolerable latency cost?

Manuscript claim under test (see H4, Table 46/47): ANCOVA (VIX
covariate) shows QGAN-LLM's RMSE improvement over Classical GAN-LLM is
statistically significant; a separate comparison shows ASR reduction.

What's tested here: that Classical GAN-LLM and QGAN-LLM can both be
trained and evaluated under identical conditions (same data, same
seed, same config shape) so a head-to-head comparison is even
meaningful, that the ANCOVA machinery correctly isolates a group
effect while controlling for a covariate, and that results_collector's
cross-model comparison pipeline (the thing that would actually produce
Table 47's numbers from real runs) works end-to-end. NOT tested:
whether QGAN-LLM actually beats Classical GAN-LLM at this toy scale --
with random initialization and 2-3 epochs, there's no reason to expect
either model to "win," and this file does not assert one does.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.baselines.classical_gan_llm import ClassicalGANLLM
from src.baselines.qgan_llm import QGANLLM
from src.evaluation.statistical_tests import ancova


def test_both_models_train_under_identical_conditions(rq_synthetic_data, qgan_config_factory):
    """A head-to-head RQ4 comparison is only meaningful if both models
    saw the same data and the same seed -- confirms that setup, not any
    particular outcome."""
    n_features = rq_synthetic_data["n_features"]
    seed = 123

    classical = ClassicalGANLLM(config={
        "latent_dim": n_features, "output_dim": n_features,
        "generator_hidden": 8, "discriminator_hidden": 8,
        "epochs": 2, "batch_size": 16,
    }, seed=seed)
    quantum = QGANLLM(config=qgan_config_factory(), seed=seed)

    for model in (classical, quantum):
        model.train(rq_synthetic_data["X_train"], rq_synthetic_data["y_train"],
                     rq_synthetic_data["X_val"], rq_synthetic_data["y_val"])
        metrics = model.evaluate(rq_synthetic_data["X_test"], rq_synthetic_data["y_test"])
        assert np.isfinite(metrics["rmse"]) and metrics["rmse"] >= 0


def test_ancova_isolates_group_effect_from_covariate():
    """H4's actual statistical test: does the model-type group effect
    (QGAN-LLM vs. Classical GAN-LLM) remain significant after
    controlling for a covariate (VIX)? Constructed here with a KNOWN
    group effect confounded with a covariate correlation, to confirm
    ANCOVA correctly separates the two rather than attributing the
    covariate's effect to the group (a classic confound ANCOVA exists
    to guard against)."""
    rng = np.random.default_rng(0)
    n_per_group = 100
    # Covariate (VIX-like) correlated with the group assignment on purpose
    # -- a naive t-test ignoring this would risk attributing the
    # covariate's effect to the group; ANCOVA should not.
    group = np.array([0] * n_per_group + [1] * n_per_group)
    covariate = 0.3 * group + rng.normal(0, 1, size=2 * n_per_group)

    # RMSE-like dv: group 1 (QGAN-LLM) genuinely better (lower), plus a
    # covariate effect and noise.
    dv = 0.492 - 0.128 * group + 0.02 * covariate + rng.normal(0, 0.01, size=2 * n_per_group)

    result = ancova(dv, group, covariate)
    assert result["p_value"] < 0.05, (
        "This data was constructed with an obvious 0.492 vs 0.364 RMSE-"
        "scale group gap even after accounting for the covariate -- "
        "ANCOVA should detect it."
    )
    assert result["partial_eta_squared"] > 0


def test_ancova_does_not_falsely_attribute_covariate_effect_to_group():
    """The mirror-image check: construct data where covariate alone
    explains the DV and there is NO real group effect, confirming ANCOVA
    does NOT report a spuriously significant group effect just because
    the covariate happens to differ slightly by group in a finite
    sample. Guards against a subtle implementation bug where the group
    effect is computed without properly controlling for the covariate."""
    rng = np.random.default_rng(1)
    n_per_group = 200  # larger n to keep the null case well-powered against false positives
    group = np.array([0] * n_per_group + [1] * n_per_group)
    covariate = rng.normal(0, 1, size=2 * n_per_group)
    # dv depends ONLY on the covariate, not on group at all.
    dv = 0.4 + 0.05 * covariate + rng.normal(0, 0.005, size=2 * n_per_group)

    result = ancova(dv, group, covariate)
    assert result["p_value"] > 0.05, (
        f"No real group effect was constructed, but ANCOVA reported "
        f"p={result['p_value']:.4f} < 0.05 -- possible false-positive "
        f"bug in the group-effect calculation."
    )


def test_results_collector_summary_feeds_the_verification_script(tmp_path):
    """Exercises the exact machinery scripts/verify_chapter4.py uses:
    ResultsCollector.summary_table() reshapes a results CSV into the
    per-model-per-metric table that verify_chapter4.py's compare_value()
    then diffs against manually-entered dissertation claims. This is
    the RQ4-relevant half of the "verify Chapter 4" pipeline -- the
    RMSE/ASR comparison table itself -- checked end-to-end here on
    known, synthetic numbers rather than waiting for a real run."""
    import sys as _sys
    import pandas as pd
    from src.experiments.results_collector import ResultsCollector

    # scripts/ isn't a package (no __init__.py, by design -- see RUNNING.md),
    # so import its module directly by path rather than `from scripts...`.
    scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts")
    _sys.path.insert(0, scripts_dir)
    import verify_chapter4  # noqa: E402

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    df = pd.DataFrame([
        {"run_id": "r1", "model_type": "Classical GAN-LLM", "rmse": 0.492, "asr": 31.0},
        {"run_id": "r2", "model_type": "QGAN-LLM", "rmse": 0.364, "asr": 8.7},
    ])
    df.to_csv(results_dir / "all_results.csv", index=False)

    collector = ResultsCollector(results_dir=str(results_dir))
    summary = collector.summary_table().set_index("model_type")

    # Exact match -> compare_value should NOT flag a >1% difference.
    exact_match_line = verify_chapter4.compare_value(
        "QGAN-LLM", "rmse", claimed=0.364, actual=float(summary.loc["QGAN-LLM", "rmse"]),
    )
    assert "<--" not in exact_match_line

    # Claimed value deliberately off by more than 1% -> SHOULD be flagged.
    mismatch_line = verify_chapter4.compare_value(
        "Classical GAN-LLM", "rmse", claimed=0.30, actual=float(summary.loc["Classical GAN-LLM", "rmse"]),
    )
    assert "<--" in mismatch_line, "A claimed 0.30 vs. actual 0.492 (>1% off) should be flagged"

    # claimed=None (metric not asserted in the manuscript) must not crash
    # and must say so rather than silently treating it as a match.
    none_claim_line = verify_chapter4.compare_value("QGAN-LLM", "asr", claimed=None, actual=8.7)
    assert "none given" in none_claim_line


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
