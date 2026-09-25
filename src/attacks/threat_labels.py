"""
threat_labels.py: builds a labeled threat/benign dataset for DV3 (FPR)
and DV5 (Threat Detection Accuracy), using "Option C" from the M3->DV3
design discussion -- the option chosen specifically because it avoids
the circularity risk of the alternatives:

  Option A (rejected): generate attacks with the QGAN itself and label
    them "threat." Circular -- the QGAN could learn to generate trivial,
    easily-flagged attacks, inflating detection accuracy for reasons
    that have nothing to do with genuine threat-detection capability.

  Option B (rejected): adapt an external labeled intrusion-detection
    dataset (CIC-IDS2017, NSL-KDD, etc.) to the EUR/USD context. Avoids
    circularity but introduces a domain mismatch this study's threat
    model doesn't otherwise involve (network intrusion features, not
    financial time-series features).

  Option C (used here): label clean market samples "benign" and
  FGSM/PGD-perturbed samples "threat," using the SAME attack code
  already implemented and tested in adversarial.py. No new labeled
  corpus, no domain mismatch, and no circularity because the perturbing
  attacker and the QGAN are unrelated -- the QGAN never sees these
  labels or this data.

IMPORTANT SCOPE NOTE: this defines "threat" as "adversarially perturbed
input to the forecasting model," which is the threat model H2/H4's ASR
already uses. It is NOT a general-purpose cybersecurity intrusion
dataset, and detection accuracy measured this way should be described
in the paper as accuracy at detecting adversarially perturbed FORECAST
INPUTS specifically -- not threat detection in the broader sense the
"FIN-ATT&CK framework" language elsewhere in the manuscript implies.
That broader framing and this operational dataset are not the same
scope, and the paper should not claim they are without saying so.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .adversarial import fgsm_perturbation, pgd_perturbation


def build_threat_benign_dataset(model: nn.Module, X_clean: torch.Tensor, y_clean: torch.Tensor,
                                 fgsm_epsilon: float, pgd_epsilon: float, pgd_alpha: float,
                                 pgd_steps: int, n_clean: int = 1000, n_fgsm: int = 500,
                                 n_pgd: int = 500, seed: int = 0) -> dict:
    """
    Builds a labeled dataset: n_clean benign samples (label 0) plus
    n_fgsm FGSM-perturbed + n_pgd PGD-perturbed threat samples (label 1),
    matching the recommended 1000 clean / 500 FGSM / 500 PGD split.

    model: the forecasting model to attack (same interface as
        adversarial.py's attack functions -- callable as model(X)).
    X_clean, y_clean: pool of real, unperturbed samples to draw both the
        benign set and the attack-source set from (sampled without
        replacement per label; the same underlying rows are never used
        for both a benign and a threat example in the same call).

    Returns a dict with X (concatenated features), y_threat (0/1 labels),
    and source (string per row: "clean", "fgsm", or "pgd") so results can
    be broken down by attack type if needed.
    """
    n_available = X_clean.shape[0]
    n_needed = n_clean + n_fgsm + n_pgd
    if n_available < n_needed:
        raise ValueError(
            f"Requested {n_needed} total samples (n_clean={n_clean} + n_fgsm={n_fgsm} + "
            f"n_pgd={n_pgd}) but only {n_available} clean samples are available. Reduce the "
            f"requested counts or supply a larger X_clean/y_clean pool."
        )

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_available)
    idx_benign = perm[:n_clean]
    idx_fgsm_source = perm[n_clean: n_clean + n_fgsm]
    idx_pgd_source = perm[n_clean + n_fgsm: n_clean + n_fgsm + n_pgd]

    X_benign = X_clean[idx_benign]

    X_fgsm_source = X_clean[idx_fgsm_source]
    y_fgsm_source = y_clean[idx_fgsm_source]
    X_fgsm_threat = fgsm_perturbation(model, X_fgsm_source, y_fgsm_source, fgsm_epsilon) if n_fgsm > 0 \
        else X_fgsm_source

    X_pgd_source = X_clean[idx_pgd_source]
    y_pgd_source = y_clean[idx_pgd_source]
    X_pgd_threat = pgd_perturbation(model, X_pgd_source, y_pgd_source, pgd_epsilon, pgd_alpha, pgd_steps) \
        if n_pgd > 0 else X_pgd_source

    X_all = torch.cat([X_benign, X_fgsm_threat, X_pgd_threat], dim=0)
    y_threat = torch.cat([
        torch.zeros(n_clean),
        torch.ones(n_fgsm),
        torch.ones(n_pgd),
    ])
    source = ["clean"] * n_clean + ["fgsm"] * n_fgsm + ["pgd"] * n_pgd

    return {
        "X": X_all,
        "y_threat": y_threat,
        "source": source,
        "n_clean": n_clean,
        "n_fgsm": n_fgsm,
        "n_pgd": n_pgd,
    }
