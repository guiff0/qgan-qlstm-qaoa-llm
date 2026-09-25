"""
Mediation analysis engine: Baron & Kenny (1986) four-step procedure with
bootstrapped indirect-effect confidence intervals.

This is new infrastructure -- nothing in the codebase before this file
tested whether an effect runs THROUGH an intermediate variable (a
mediator). ancova()/chi_square_test()/independent_ttest()/
pearson_correlation_with_ci() (statistical_tests.py) all test DIRECT
associations; none of them decompose an effect into a path through M.

THE FOUR STEPS (all as ordinary least squares regressions; see the
"CATEGORICAL VARIABLES" note below for how non-continuous IV/M/DV are
handled):
  1. Total effect:      DV ~ IV                  -> path c
  2. IV -> mediator:     M  ~ IV                  -> path a
  3. Direct + mediator:  DV ~ IV + M              -> path b (M's coefficient)
                                                     and path c' (IV's coefficient)
  4. Indirect effect  = a * b  (equivalently c - c')
     Proportion mediated = (c - c') / c

INDIRECT EFFECT'S SAMPLING DISTRIBUTION IS NOT NORMAL (the product of
two approximately-normal estimates is not itself normal), so a Sobel
test's normal-theory p-value is a poor approximation -- bootstrapping
is used instead: resample cases with replacement, recompute a*b each
time, and report a percentile-based confidence interval (the standard,
widely-cited alternative to Sobel; see Preacher & Hayes, 2004/2008 for
why bootstrapping is preferred).

CATEGORICAL VARIABLES: this project's IV is always binary (classical vs.
quantum configuration); some mediators/outcomes named in the paper's
pathway table are ordinal categories (e.g. entanglement level High/Low;
ASR category Low/Moderate/High) rather than continuous measurements.
This engine numerically encodes ordinal categories (e.g. Low=0,
Moderate=1, High=2) and runs the same OLS-based procedure on the coded
values -- a widely used approximation (a "linear probability model"
style treatment), NOT the textbook-correct method for a strictly
categorical mediator or outcome (which would use logistic or ordinal
regression and a different indirect-effect formula). This is flagged
here, in the docstring of every result this engine returns, and belongs
in the paper's methodology write-up -- do not present bootstrapped OLS
mediation on ordinal data as identical in rigor to continuous-variable
mediation without this caveat.
"""
from __future__ import annotations

import numpy as np


def _ols_coef(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Coefficients (including intercept) for y = X @ beta via least squares.
    X must already include a column of ones for the intercept."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def _add_intercept(*columns: np.ndarray) -> np.ndarray:
    n = len(columns[0])
    return np.column_stack([np.ones(n), *columns])


def _path_coefficient(y: np.ndarray, *predictors: np.ndarray, coef_index: int) -> float:
    X = _add_intercept(*predictors)
    beta = _ols_coef(X, y)
    return float(beta[coef_index])


def mediation_analysis(iv: np.ndarray, mediator: np.ndarray, dv: np.ndarray,
                        n_bootstrap: int = 5000, alpha: float = 0.05,
                        seed: int = 42) -> dict:
    """
    Baron & Kenny four-step mediation with a bootstrapped indirect-effect CI.

    iv, mediator, dv: 1D arrays, one row per observation, already numerically
    encoded (binary IV as 0/1; ordinal M or DV as ordered integers -- see
    module docstring's CATEGORICAL VARIABLES note).

    Returns a dict with path_a, path_b, path_c, path_c_prime, indirect_effect,
    ci_low, ci_high (percentile bootstrap CI on the indirect effect),
    proportion_mediated, and a boolean `significant` (True if the CI excludes
    zero, the standard bootstrap-mediation significance criterion -- NOT a
    p-value from a normal-theory test).
    """
    iv = np.asarray(iv, dtype=float)
    mediator = np.asarray(mediator, dtype=float)
    dv = np.asarray(dv, dtype=float)
    n = len(iv)
    if not (len(mediator) == n and len(dv) == n):
        raise ValueError(f"iv, mediator, dv must be the same length; got {n}, {len(mediator)}, {len(dv)}")
    if n < 10:
        raise ValueError(f"Mediation analysis needs a reasonable sample size; got only {n} observations")

    # Step 1: total effect (path c)
    path_c = _path_coefficient(dv, iv, coef_index=1)
    # Step 2: IV -> mediator (path a)
    path_a = _path_coefficient(mediator, iv, coef_index=1)
    # Step 3: DV ~ IV + M (path b = M's coefficient, path c' = IV's coefficient)
    X3 = _add_intercept(iv, mediator)
    beta3 = _ols_coef(X3, dv)
    path_c_prime = float(beta3[1])
    path_b = float(beta3[2])

    indirect_effect = path_a * path_b
    proportion_mediated = (path_c - path_c_prime) / path_c if abs(path_c) > 1e-9 else float("nan")

    # Bootstrap CI on the indirect effect
    rng = np.random.default_rng(seed)
    boot_indirect = np.empty(n_bootstrap)
    idx_range = np.arange(n)
    for b in range(n_bootstrap):
        sample_idx = rng.choice(idx_range, size=n, replace=True)
        iv_b, m_b, dv_b = iv[sample_idx], mediator[sample_idx], dv[sample_idx]
        a_b = _path_coefficient(m_b, iv_b, coef_index=1)
        X3_b = _add_intercept(iv_b, m_b)
        beta3_b = _ols_coef(X3_b, dv_b)
        boot_indirect[b] = a_b * float(beta3_b[2])

    ci_low = float(np.percentile(boot_indirect, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_indirect, 100 * (1 - alpha / 2)))
    significant = not (ci_low <= 0 <= ci_high)

    return {
        "path_a": path_a,
        "path_b": path_b,
        "path_c": path_c,
        "path_c_prime": path_c_prime,
        "indirect_effect": indirect_effect,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "proportion_mediated": proportion_mediated,
        "significant": significant,
        "n_bootstrap": n_bootstrap,
        "n_observations": n,
        "mediation_type": (
            "full" if significant and abs(path_c_prime) < abs(path_c) * 0.1
            else "partial" if significant
            else "none"
        ),
    }
