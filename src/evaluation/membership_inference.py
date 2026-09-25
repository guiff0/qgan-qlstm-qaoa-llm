"""
Membership inference attack: trains a classifier to distinguish real
training samples from synthetic samples, using cross-validated accuracy
as the "membership inference success rate."

WHY THIS WAS MISSING: Table 45 (Ch. 4) reports a "Membership Inference
Success" metric with real values, but nothing in this codebase computed
it -- there was no membership-inference implementation anywhere prior
to this file. Ported and adapted from an external repository audit
(this was one of the genuinely real files found there): it trains an
actual RandomForestClassifier and reports real cross-validated accuracy,
not a fabricated placeholder.

INTERPRETATION: a HIGHER score here means the attacker's classifier can
tell real and synthetic samples apart MORE easily -- i.e., worse
privacy protection (the synthetic data "leaks" distinguishable
structure). A score near 0.5 (chance level, for a balanced real/
synthetic split) means the attacker cannot do better than guessing --
i.e., better privacy protection. This is the OPPOSITE direction from
most of this study's other resilience-style metrics (where higher is
better) and should be labeled accordingly wherever it's reported.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def membership_inference_success(real: np.ndarray, synthetic: np.ndarray,
                                  n_samples: int = 5000, cv_folds: int = 3,
                                  n_estimators: int = 100, seed: int = 42) -> dict:
    """
    real, synthetic: (n, n_features) arrays of real and synthetic
    feature vectors (same feature space -- e.g. the same PCA-reduced
    representation used elsewhere in this study).

    Returns cross-validated accuracy of a RandomForest classifier
    trained to distinguish real (label 0) from synthetic (label 1)
    samples, plus the class balance actually used (relevant if real or
    synthetic has fewer than n_samples available).
    """
    n_real = min(n_samples, len(real))
    n_synth = min(n_samples, len(synthetic))
    if n_real < cv_folds or n_synth < cv_folds:
        raise ValueError(
            f"Need at least {cv_folds} samples per class for {cv_folds}-fold CV; "
            f"got {n_real} real and {n_synth} synthetic."
        )

    X = np.vstack([real[:n_real], synthetic[:n_synth]])
    y = np.array([0] * n_real + [1] * n_synth)

    classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
    scores = cross_val_score(classifier, X, y, cv=cv_folds)

    return {
        "membership_inference_accuracy": float(scores.mean()),
        "membership_inference_accuracy_sd": float(scores.std()),
        "n_real_used": n_real,
        "n_synthetic_used": n_synth,
        "cv_folds": cv_folds,
        "chance_level": max(n_real, n_synth) / (n_real + n_synth),
        "interpretation": "higher = easier to distinguish real from synthetic = WORSE privacy protection",
    }
