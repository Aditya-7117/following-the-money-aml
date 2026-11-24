"""
src/analysis/stats.py
Small statistics helpers for the EDA notebooks.

rate_table()       laundering rate per category, with 95% Wilson intervals
chi_square_test()  chi-square test of independence plus Cramer's V effect size
"""

import numpy as np
import pandas as pd
from scipy import stats

LABEL_COLUMN = "is_laundering"


def wilson_interval(successes, totals, confidence=0.95):
    """95% Wilson confidence interval for a proportion (works well for rare events)."""
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    successes = np.asarray(successes, dtype=float)
    totals = np.asarray(totals, dtype=float)
    p = successes / totals
    centre = (p + z**2 / (2 * totals)) / (1 + z**2 / totals)
    half = (z / (1 + z**2 / totals)) * np.sqrt(p * (1 - p) / totals + z**2 / (4 * totals**2))
    return np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


def rate_table(df, column, label=LABEL_COLUMN):
    """One row per category: counts, laundering rate, its 95% CI, share of all laundering and lift."""
    table = df.groupby(column, observed=True)[label].agg(transactions="count", laundering="sum")
    overall_rate = table["laundering"].sum() / table["transactions"].sum()

    low, high = wilson_interval(table["laundering"], table["transactions"])
    table["rate_%"] = 100 * table["laundering"] / table["transactions"]
    table["ci_low_%"] = 100 * low
    table["ci_high_%"] = 100 * high
    table["share_of_laundering_%"] = 100 * table["laundering"] / table["laundering"].sum()
    table["lift"] = (table["rate_%"] / 100) / overall_rate
    return table.sort_values("rate_%", ascending=False).round(4)


def chi_square_test(df, column, label=LABEL_COLUMN):
    """Chi-square test of independence between a category column and the label."""
    observed = pd.crosstab(df[column], df[label])
    chi2, p_value, dof, expected = stats.chi2_contingency(observed)
    n = observed.to_numpy().sum()
    cramers_v = np.sqrt(chi2 / (n * (min(observed.shape) - 1)))
    return {
        "column": column,
        "chi2": round(float(chi2), 1),
        "dof": int(dof),
        "p_value": float(p_value),
        "cramers_v": round(float(cramers_v), 4),
        "min_expected_count": round(float(expected.min()), 1),
        "n": int(n),
    }