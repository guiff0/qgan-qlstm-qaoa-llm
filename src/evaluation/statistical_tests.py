"""
Statistical hypothesis tests: ANCOVA, chi-square, independent t-test,
and Pearson correlation with confidence interval.

The original code never implemented any of these — every p-value and
F-statistic in Chapter 4 (e.g. "F(1,197)=9.64, p=0.002, eta^2=0.047")
has no corresponding computation anywhere in the provided files. This
module implements each test from its standard definition using
statsmodels/scipy, matching the specific test each hypothesis (H1-H5)
requires per Ch. 3's "Test" column (Table 16).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


def ancova(dv: np.ndarray, group: np.ndarray, covariate: np.ndarray) -> dict:
    """
    One-way ANCOVA: DV ~ group + covariate.
    `group` should be a 0/1 (or categorical) array identifying the two
    conditions being compared (e.g. QGAN-LLM vs Classical GAN-LLM).
    `covariate` is the control variable (VIX / market volatility, per
    the dissertation's stated covariate).

    Returns the F-statistic and p-value for the GROUP effect (i.e. the
    effect of interest, after controlling for the covariate), plus
    partial eta-squared as the effect size — matching the dissertation's
    reported format exactly (df, SS, MS, F, p, eta^2).
    """
    df = pd.DataFrame({"dv": dv, "group": group, "covariate": covariate})
    model = smf.ols("dv ~ C(group) + covariate", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    group_row = anova_table.loc["C(group)"]
    residual_row = anova_table.loc["Residual"]

    ss_group = group_row["sum_sq"]
    ss_residual = residual_row["sum_sq"]
    partial_eta_sq = ss_group / (ss_group + ss_residual)

    return {
        "df_group": int(group_row["df"]),
        "df_residual": int(residual_row["df"]),
        "ss_group": float(ss_group),
        "ms_group": float(ss_group / group_row["df"]),
        "f_statistic": float(group_row["F"]),
        "p_value": float(group_row["PR(>F)"]),
        "partial_eta_squared": float(partial_eta_sq),
    }


def chi_square_test(contingency_table: np.ndarray) -> dict:
    """Chi-square test of independence (e.g. ASR category x entanglement level).
    Also returns Cramer's V as the effect size, matching Ch.4's H2/H4 summary."""
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    n = contingency_table.sum()
    min_dim = min(contingency_table.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else float("nan")

    return {
        "chi2_statistic": float(chi2),
        "degrees_of_freedom": int(dof),
        "p_value": float(p),
        "cramers_v": float(cramers_v),
    }


def independent_ttest(group_a: np.ndarray, group_b: np.ndarray) -> dict:
    """Welch's t-test (does not assume equal variances — safer default
    than Student's t for comparing two model conditions with potentially
    different result variances)."""
    t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)
    n_a, n_b = len(group_a), len(group_b)
    pooled_std = np.sqrt(((n_a - 1) * group_a.var(ddof=1) + (n_b - 1) * group_b.var(ddof=1)) / (n_a + n_b - 2))
    cohens_d = (group_a.mean() - group_b.mean()) / pooled_std if pooled_std > 0 else float("nan")

    return {
        "t_statistic": float(t_stat),
        "df": n_a + n_b - 2,
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "mean_a": float(group_a.mean()),
        "mean_b": float(group_b.mean()),
        "sd_a": float(group_a.std(ddof=1)),
        "sd_b": float(group_b.std(ddof=1)),
    }


def pearson_correlation_with_ci(x: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> dict:
    """Pearson r with a Fisher-z confidence interval, matching the
    dissertation's reported format: r(df) = ..., p < ..., 95% CI [.., ..]."""
    r, p = stats.pearsonr(x, y)
    n = len(x)
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_low, ci_high = np.tanh(z - z_crit * se), np.tanh(z + z_crit * se)

    return {
        "r": float(r),
        "df": n - 2,
        "p_value": float(p),
        "r_squared": float(r ** 2),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }
