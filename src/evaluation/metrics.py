"""
Point-forecast and distributional-fidelity metrics.

The original code base never actually implemented FID, MMD, or
Wasserstein distance — Chapter 4's Tables 38-42 report specific FID/MMD
numbers (18.7 for QGAN, 42.3 for classical GAN, etc.) but nothing in
the provided code computes them. Implemented here from their standard
definitions so those tables can be regenerated from real output rather
than typed in by hand.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg
from scipy.stats import wasserstein_distance


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true.flatten() - y_pred.flatten()) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true.flatten() - y_pred.flatten())))


def frechet_distance(real_features: np.ndarray, synthetic_features: np.ndarray) -> float:
    """
    Frechet Inception Distance, adapted for tabular financial features
    rather than image-net embeddings (no Inception network involved —
    the "features" here are the model's own real vs. synthetic
    feature vectors, consistent with how FID is applied to non-image
    GANs in the financial-ML literature this dissertation cites).

    FID = ||mu_r - mu_s||^2 + Tr(Sigma_r + Sigma_s - 2*sqrt(Sigma_r @ Sigma_s))
    """
    mu_r, mu_s = real_features.mean(axis=0), synthetic_features.mean(axis=0)
    sigma_r = np.cov(real_features, rowvar=False)
    sigma_s = np.cov(synthetic_features, rowvar=False)

    diff = mu_r - mu_s
    covmean, _ = linalg.sqrtm(sigma_r @ sigma_s, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_r + sigma_s - 2 * covmean)
    return float(fid)


def maximum_mean_discrepancy(real: np.ndarray, synthetic: np.ndarray, gamma: float = 1.0) -> float:
    """
    MMD^2 with an RBF kernel: unbiased estimator.
    Lower = distributions are more similar.
    """
    def rbf_kernel(a, b):
        a2 = np.sum(a ** 2, axis=1, keepdims=True)
        b2 = np.sum(b ** 2, axis=1, keepdims=True)
        sq_dists = a2 + b2.T - 2 * a @ b.T
        return np.exp(-gamma * sq_dists)

    k_rr = rbf_kernel(real, real)
    k_ss = rbf_kernel(synthetic, synthetic)
    k_rs = rbf_kernel(real, synthetic)

    n, m = real.shape[0], synthetic.shape[0]
    term_rr = (k_rr.sum() - np.trace(k_rr)) / (n * (n - 1))
    term_ss = (k_ss.sum() - np.trace(k_ss)) / (m * (m - 1))
    term_rs = k_rs.sum() / (n * m)

    return float(term_rr + term_ss - 2 * term_rs)


def mean_wasserstein_distance(real: np.ndarray, synthetic: np.ndarray) -> float:
    """
    Mean 1D Wasserstein distance across all feature dimensions
    (a common tabular-data extension of Wasserstein/Earth-Mover distance,
    since the true multivariate optimal-transport distance is
    computationally expensive at this scale).
    """
    distances = [
        wasserstein_distance(real[:, i], synthetic[:, i])
        for i in range(real.shape[1])
    ]
    return float(np.mean(distances))


def classification_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Real confusion-matrix-based classification metrics -- FPR, precision,
    recall, F1, and AUPRC (area under the precision-recall curve).

    ADDED BECAUSE: False Positive Rate (FPR) is a named dependent
    variable (DV3, Table 16) with a formal hypothesis (H3) attached to
    it, but nothing in the original codebase computed it anywhere --
    confirmed by search, zero hits for "false_positive"/"FPR" prior to
    this function. This also backs the M3->DV3 mediation pathway
    (mediation.py), which cannot run at all without a real FPR value to
    use as its dependent variable.

    y_true: binary ground-truth labels (1 = threat, 0 = benign).
    y_pred_proba: predicted threat probability in [0, 1].
    threshold: probability cutoff for classifying as "threat" (the
    dissertation's QACL table specifies 0.7 for the threat-detection
    decision; pass that value explicitly rather than relying on this
    function's 0.5 default, which is a generic placeholder, not a
    methodology-derived choice).
    """
    y_true = np.asarray(y_true).flatten().astype(int)
    y_pred_proba = np.asarray(y_pred_proba).flatten()
    y_pred = (y_pred_proba >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")  # a.k.a. true positive rate / sensitivity
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else float("nan")
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else float("nan")

    auprc = _average_precision(y_true, y_pred_proba)

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "fpr": fpr, "precision": precision, "recall": recall, "f1": f1,
        "accuracy": accuracy, "auprc": auprc, "threshold": threshold,
    }


def _average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the precision-recall curve via the standard step-function
    (rectangle) approximation: sum over score thresholds of
    precision[k] * (recall[k] - recall[k-1]). No sklearn dependency --
    implemented directly so this module has no hidden dependency beyond
    numpy/scipy, matching the rest of this file."""
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]
    tp_cum = np.cumsum(y_true_sorted)
    fp_cum = np.cumsum(1 - y_true_sorted)
    n_positive = y_true.sum()
    if n_positive == 0:
        return float("nan")

    precision = tp_cum / (tp_cum + fp_cum)
    recall = tp_cum / n_positive

    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[precision[0] if len(precision) else 1.0], precision])

    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def synthetic_data_fidelity_report(real: np.ndarray, synthetic: np.ndarray) -> dict:
    """One-call report matching the columns of Ch.4 Table 38
    (Comparison of Synthetic Data Fidelity Metrics)."""
    return {
        "fid": frechet_distance(real, synthetic),
        "mmd": maximum_mean_discrepancy(real, synthetic),
        "wasserstein": mean_wasserstein_distance(real, synthetic),
    }
