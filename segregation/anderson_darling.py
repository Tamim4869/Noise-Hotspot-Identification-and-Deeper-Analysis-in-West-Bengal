import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import os 
import matplotlib.pyplot as plt

os.chdir("D:/Stuff3/Sem6/Stat Comprehensive/Group Project/Noise Data/dummy plots/References for segregation")

# =========================================================
# LOAD
# =========================================================

station1 = pd.read_csv("Ballygunge_Campus,_C.U.csv")
ts_here = pd.to_datetime(station1.iloc[:, 0], errors='coerce').dropna()

mask = ts_here.ne(ts_here.shift())
df = station1.loc[mask].copy()
ts_here= ts_here.loc[mask]

df['ts']= ts_here

df['timestamp'] = pd.to_datetime(
    df['timestamp']
)

#==================================
# FEATURES
#==================================

df['impulsiveness']= df['lapeakt']- df['laeqt']
df['low_freq']= df['lceqt']- df['laeqt']


# =========================================================
# DEFINE PERIODS
# =========================================================

festive_start = '2025-09-25'
festive_end   = '2026-01-10'


nonfestive_start= '2025-02-01'
nonfestive_end= '2025-09-24'


# festive
festive = df[
    (
        df['timestamp'] >= festive_start
    ) &
    (
        df['timestamp'] <= festive_end
    )
]['impulsiveness'].dropna()

# non-festive
nonfestive = df[
    (
        df['timestamp'] >= nonfestive_start
    ) &
    (
        df['timestamp'] <= nonfestive_end
    )
]['impulsiveness'].dropna()



alpha = 0.05



"""
One-Sided Two-Sample Anderson-Darling Test
===========================================
Tests H0: F = G  vs  H1: F(x) < G(x) for sufficiently large x
(i.e., X tends to be stochastically larger than Y)

Reference:
  Scholz, F.W. and Stephens, M.A. (1987). K-Sample Anderson-Darling Tests.
  Journal of the American Statistical Association, 82(399), 918-924.

  Pettitt, A.N. (1976). A two-sample Anderson-Darling rank statistic.
  Biometrika, 63(1), 161-168.
"""

from typing import Literal



def ad_onesided_test(
    x: np.ndarray,
    y: np.ndarray,
    alternative: Literal["greater", "less"] = "greater",
    n_permutations: int = 9999,
    trim: int = 0,
    random_state: int | None = None,
) -> dict:
    """
    One-sided two-sample Anderson-Darling test.

    Parameters
    ----------
    x : array-like
        Sample from distribution F (size m).
    y : array-like
        Sample from distribution G (size n).
    alternative : {'greater', 'less'}
        'greater' : H1: F(x) < G(x) for large x  (X stochastically larger than Y)
        'less'    : H1: F(x) > G(x) for large x  (Y stochastically larger than X)
    n_permutations : int
        Number of permutations for the null distribution. Default 9999.
    trim : int
        Number of extreme order statistics to exclude from each end.
        trim=0 uses all observations. trim>0 improves robustness to outliers.
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    dict with keys:
        statistic       : float  — observed A^- test statistic
        p_value         : float  — permutation p-value
        n_permutations  : int    — number of permutations used
        m               : int    — size of x
        n               : int    — size of y
        N               : int    — combined sample size
        trim            : int    — trim value used
        alternative     : str    — alternative hypothesis used
        reject_5pct     : bool   — whether H0 is rejected at 5% level
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m, n = len(x), len(y)
    N = m + n

    if m < 2 or n < 2:
        raise ValueError("Both samples must have at least 2 observations.")
    if trim < 0:
        raise ValueError("trim must be non-negative.")
    if 2 * trim >= N - 1:
        raise ValueError("trim is too large relative to the combined sample size.")

    observed_stat = _compute_statistic(x, y, trim, alternative)
    perm_stats = _permutation_null(x, y, n_permutations, trim, alternative, random_state)

    # p-value: proportion of permutation stats >= observed (one-sided, upper tail)
    p_value = (np.sum(perm_stats >= observed_stat) + 1) / (n_permutations + 1)

    return {
        "statistic": observed_stat,
        "p_value": p_value,
        "n_permutations": n_permutations,
        "m": m,
        "n": n,
        "N": N,
        "trim": trim,
        "alternative": alternative,
        "reject_5pct": p_value < 0.05,
    }


def _compute_statistic(
    x: np.ndarray,
    y: np.ndarray,
    trim: int,
    alternative: str,
) -> float:
    """
    Compute the one-sided AD statistic for given samples.

    Statistic:
        A^- = (1/N) * sum_{k=1}^{N-1} w_k * [G_n(Z_(k)) - F_m(Z_(k))]

    where w_k = N^2 / (k * (N - k))  and Z_(1) < ... < Z_(N) are the
    order statistics of the pooled sample.

    The sign is chosen so that large positive values support H1.
    """
    m, n = len(x), len(y)
    N = m + n

    # Pool and sort
    pooled = np.concatenate([x, y])
    order = np.argsort(pooled, kind="mergesort")
    labels = np.concatenate([np.ones(m), np.zeros(n)])  # 1 = X, 0 = Y

    # Compute empirical CDFs at each order statistic
    # After seeing k observations, F_m(Z_(k)) = #{X_i <= Z_(k)} / m
    # G_n(Z_(k)) = #{Y_j <= Z_(k)} / n
    sorted_labels = labels[order]

    cum_x = np.cumsum(sorted_labels)        # count of X's seen so far
    cum_y = np.cumsum(1 - sorted_labels)    # count of Y's seen so far

    Fm = cum_x / m   # F_m(Z_(k)) for k=1,...,N
    Gn = cum_y / n   # G_n(Z_(k)) for k=1,...,N

    # We sum over k = 1, ..., N-1 (exclude k=N since weight is undefined)
    # With trim, we sum over k = trim+1, ..., N-1-trim
    k = np.arange(1, N, dtype=float)   # k = 1, ..., N-1
    weights = (N ** 2) / (k * (N - k))

    diff = Gn[:-1] - Fm[:-1]  # G_n - F_m at Z_(k), k=1,...,N-1

    if alternative == "less":
        diff = -diff  # flip sign: test whether Y > X in upper tail

    # Apply trim
    if trim > 0:
        weights[:trim] = 0.0
        weights[-trim:] = 0.0

    stat = np.sum(weights * diff) / N
    return float(stat)


def _permutation_null(
    x: np.ndarray,
    y: np.ndarray,
    n_permutations: int,
    trim: int,
    alternative: str,
    random_state: int | None,
) -> np.ndarray:
    """
    Generate the permutation null distribution of the AD statistic.
    """
    rng = np.random.default_rng(random_state)
    m = len(x)
    pooled = np.concatenate([x, y])
    N = len(pooled)

    stats = np.empty(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(N)
        x_perm = pooled[perm[:m]]
        y_perm = pooled[perm[m:]]
        stats[i] = _compute_statistic(x_perm, y_perm, trim, alternative)

    return stats


def print_results(result: dict) -> None:
    """Pretty-print the test results."""
    alt_str = {
        "greater": "H1: F(x) < G(x) for large x  [X stochastically larger than Y]",
        "less":    "H1: F(x) > G(x) for large x  [Y stochastically larger than X]",
    }[result["alternative"]]

    print("=" * 60)
    print("  One-Sided Two-Sample Anderson-Darling Test")
    print("=" * 60)
    print(f"  H0 : F = G")
    print(f"  {alt_str}")
    print("-" * 60)
    print(f"  Sample sizes         : m = {result['m']},  n = {result['n']}")
    print(f"  Combined size N      : {result['N']}")
    print(f"  Trim                 : {result['trim']}")
    print(f"  Permutations         : {result['n_permutations']}")
    print("-" * 60)
    print(f"  Test statistic A^-   : {result['statistic']:>10.4f}")
    print(f"  p-value              : {result['p_value']:>10.4f}")
    print("-" * 60)
    verdict = "REJECT H0" if result["reject_5pct"] else "FAIL TO REJECT H0"
    print(f"  Decision (α = 0.05)  : {verdict}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

    
result = ad_onesided_test(
    x=festive.values,
    y=nonfestive.values,
    alternative="greater",   # H1: festive stochastically larger than nonfestive
    n_permutations=9999,
    trim=0,                  # set trim=3 or 5 if you suspect outliers
    random_state=42
)

print_results(result)
    
print(result["statistic"])   # A^- test statistic
print(result["p_value"])     # permutation p-value
print(result["reject_5pct"]) # True/False at 5% level   
    
    
    
    
    
    
    
    