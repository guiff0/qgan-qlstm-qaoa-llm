"""
RQ1 -- Synthetic Data Fidelity (H1): does quantum encoding produce
higher-fidelity synthetic data than classical encoding?

Manuscript claim under test (see H1, Ch.1/Ch.4): QGAN-generated
synthetic EUR/USD data achieves a lower FID score against real data
than Classical GAN-generated data, indicating higher fidelity.

What's tested here: the FID/MMD/Wasserstein machinery itself (does it
correctly rank "more similar" above "less similar"?), and that a
QGANLLM's generate_synthetic_data() output can actually be scored
against real data end-to-end. NOT tested: whether QGAN-LLM actually
beats Classical GAN-LLM on real EUR/USD data -- an untrained, 3-epoch,
4-qubit toy circuit has no reason to beat anything; this only confirms
the comparison PIPELINE is sound.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.baselines.classical_gan_llm import ClassicalGANLLM
from src.baselines.qgan_llm import QGANLLM
from src.evaluation.metrics import frechet_distance, maximum_mean_discrepancy, mean_wasserstein_distance


def test_fid_is_near_zero_for_identical_distributions():
    """Sanity check on the metric itself: FID(X, X) must be ~0. If this
    fails, the FID implementation is wrong regardless of any model."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 8))
    assert frechet_distance(X, X) < 1e-8


def test_fid_increases_with_distributional_shift():
    """The core sanity a fidelity metric must satisfy for H1 to be
    testable at all: a distribution further from real data must score
    WORSE (higher FID) than one closer to it. If this direction were
    ever flipped, "QGAN has lower FID than Classical GAN" would mean
    the OPPOSITE of what H1 claims it means."""
    rng = np.random.default_rng(0)
    real = rng.normal(loc=0, scale=1, size=(300, 8))
    close = rng.normal(loc=0.1, scale=1.05, size=(300, 8))   # small shift
    far = rng.normal(loc=3.0, scale=2.5, size=(300, 8))      # large shift

    fid_close = frechet_distance(real, close)
    fid_far = frechet_distance(real, far)
    assert fid_close < fid_far, (
        f"Expected the smaller distributional shift to score lower FID; "
        f"got fid_close={fid_close:.4f}, fid_far={fid_far:.4f}"
    )


def test_mmd_and_wasserstein_agree_on_direction():
    """Cross-check: MMD and per-feature Wasserstein distance should agree
    with FID's ranking on the same close-vs-far setup, since all three
    are being used together in Ch.4's fidelity tables (H1's table
    reports FID; RQ1's results table separately reports MMD by
    volatility regime) -- they should not contradict each other on an
    unambiguous case."""
    rng = np.random.default_rng(1)
    real = rng.normal(loc=0, scale=1, size=(300, 8))
    close = rng.normal(loc=0.1, scale=1.05, size=(300, 8))
    far = rng.normal(loc=3.0, scale=2.5, size=(300, 8))

    assert maximum_mean_discrepancy(real, close) < maximum_mean_discrepancy(real, far)
    assert mean_wasserstein_distance(real, close) < mean_wasserstein_distance(real, far)


def test_qgan_vs_classical_synthetic_fidelity_pipeline_runs(rq_synthetic_data, qgan_config_factory):
    """End-to-end: train both generators briefly, generate synthetic
    data from each, score both against real held-out data with FID, and
    confirm the comparison itself (not its winner) runs cleanly and
    produces two finite, comparable numbers -- this is the exact
    comparison H1's results table reports, just at a scale that fits in
    a test suite."""
    n_features = rq_synthetic_data["n_features"]

    classical = ClassicalGANLLM(config={
        "latent_dim": n_features, "output_dim": n_features,
        "generator_hidden": 8, "discriminator_hidden": 8,
        "epochs": 2, "batch_size": 16,
    })
    classical.train(rq_synthetic_data["X_train"], rq_synthetic_data["y_train"],
                     rq_synthetic_data["X_val"], rq_synthetic_data["y_val"])

    quantum = QGANLLM(config=qgan_config_factory())
    quantum.train(rq_synthetic_data["X_train"], rq_synthetic_data["y_train"],
                   rq_synthetic_data["X_val"], rq_synthetic_data["y_val"])

    real_holdout = rq_synthetic_data["X_test"]
    classical_synthetic = classical.generate_synthetic_data(n_samples=len(real_holdout))
    quantum_synthetic = quantum.generate_synthetic_data(n_samples=len(real_holdout))

    fid_classical = frechet_distance(real_holdout, classical_synthetic)
    fid_quantum = frechet_distance(real_holdout, quantum_synthetic)

    assert np.isfinite(fid_classical) and fid_classical >= 0
    assert np.isfinite(fid_quantum) and fid_quantum >= 0
    # Deliberately NOT asserting fid_quantum < fid_classical here -- with
    # 2-3 epochs on random toy data there is no reason to expect that
    # direction, and asserting it would be exactly the kind of
    # self-confirming test this whole rebuild exists to avoid.


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
