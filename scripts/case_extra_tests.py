#!/usr/bin/env python3
"""
case_extra_tests.py -- additional statistics for RQ4 reported in the article.

Input : recheck_by_acq.csv written by case_recheck.py
Output: OUT/case_extra_tests.csv and OUT/case_year_bootstrap.csv

Computes
  1. one-sample Wilcoxon test of LCI against zero in the later period;
  2. Theil-Sen slope of LCI over years within the later period;
  3. Mann-Whitney test of 2026 against 2021-2025 (LCI, later period);
  4. Spearman correlation of LCI with the LST of the surrounding ring;
  5. Holm correction over the six p-values reported in Section III-E;
  6. before/after changes bootstrapped by year instead of by acquisition.

Usage:
  python scripts/case_extra_tests.py results/case_recheck/recheck_by_acq.csv \
         results/case_recheck --boot 5000 --seed 42
"""
import argparse, os
import numpy as np
import pandas as pd
from scipy import stats

PERIODS = {"botanic": ("construction", "mature"), "greenline": ("before", "after")}


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, p[i] * (m - k))); adj[i] = run
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("by_acq"); ap.add_argument("out")
    ap.add_argument("--boot", type=int, default=5000); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    A = pd.read_csv(a.by_acq)

    rows = []
    for site, (b, aft) in PERIODS.items():
        g = A[(A.site_name == site) & (A.period == aft)].dropna(subset=["LCI"])
        w = stats.wilcoxon(g.LCI)
        ts = stats.theilslopes(g.LCI, g.year)
        rho = stats.spearmanr(g.LST_local, g.LCI)
        mw = stats.mannwhitneyu(g[g.year == 2026].LCI, g[g.year < 2026].LCI, alternative="two-sided")
        rows += [dict(site=site, test="wilcoxon_LCI_vs_0_after", n=len(g), stat=w.statistic, p=w.pvalue),
                 dict(site=site, test="theil_sen_LCI_per_year_after", n=len(g), stat=ts[0],
                      ci_lo=ts[2], ci_hi=ts[3]),
                 dict(site=site, test="spearman_LCI_vs_ring_LST_after", n=len(g), stat=rho.statistic, p=rho.pvalue),
                 dict(site=site, test="mannwhitney_LCI_2026_vs_2021_25", n=len(g),
                      stat=mw.statistic, p=mw.pvalue,
                      mean_2026=g[g.year == 2026].LCI.mean(), mean_2021_25=g[g.year < 2026].LCI.mean())]
        for col in ("LCI", "dLST_ctrl"):
            x = A[(A.site_name == site) & A[col].notna()]
            B, C = x[x.period == b][col], x[x.period == aft][col]
            p = stats.mannwhitneyu(C, B, alternative="two-sided").pvalue
            rows.append(dict(site=site, test=f"mannwhitney_{col}_after_vs_before", n=len(B) + len(C),
                             stat=C.mean() - B.mean(), p=p))
    T = pd.DataFrame(rows)

    # Holm over the six p-values of Section III-E
    six = [("botanic", "mannwhitney_LCI_after_vs_before"), ("botanic", "mannwhitney_dLST_ctrl_after_vs_before"),
           ("greenline", "mannwhitney_LCI_after_vs_before"), ("greenline", "mannwhitney_dLST_ctrl_after_vs_before"),
           ("greenline", "wilcoxon_LCI_vs_0_after"), ("greenline", "mannwhitney_LCI_2026_vs_2021_25")]
    idx = [T[(T.site == s) & (T.test == t)].index[0] for s, t in six]
    T.loc[idx, "p_holm_six"] = holm(T.loc[idx, "p"].values)
    T.to_csv(f"{a.out}/case_extra_tests.csv", index=False)

    # bootstrap by year
    yb = []
    for site, (b, aft) in PERIODS.items():
        for col in ("LCI", "dLST_ctrl"):
            x = A[(A.site_name == site) & A[col].notna()]
            B, C = x[x.period == b], x[x.period == aft]
            yrsB, yrsC = B.year.unique(), C.year.unique()
            gB = {y: B[B.year == y][col].values for y in yrsB}
            gC = {y: C[C.year == y][col].values for y in yrsC}
            d = np.empty(a.boot)
            for k in range(a.boot):
                sb = np.concatenate([gB[y] for y in rng.choice(yrsB, len(yrsB))])
                sc = np.concatenate([gC[y] for y in rng.choice(yrsC, len(yrsC))])
                d[k] = sc.mean() - sb.mean()
            yb.append(dict(site=site, metric=col, change=C[col].mean() - B[col].mean(),
                           ci_lo=np.percentile(d, 2.5), ci_hi=np.percentile(d, 97.5),
                           years_before=len(yrsB), years_after=len(yrsC)))
    Y = pd.DataFrame(yb); Y.to_csv(f"{a.out}/case_year_bootstrap.csv", index=False)
    pd.set_option("display.width", 200)
    print(T.round(6).to_string(index=False)); print(); print(Y.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
