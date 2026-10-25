#!/usr/bin/env python3
"""
case_recheck.py  —  повторная проверка кейса по выходу 10_case_recheck.js.

Запуск:
  python scripts/case_recheck.py data/astana_case_recheck_lst.csv data/astana_case_recheck_s2.csv out/case_recheck \
         --min-valid 0.8 --boot 5000 --seed 42

Три взгляда на один вопрос:
  A) DiD к дальнему контролю (как в case_analysis.py, но 2013-2026 и облачность по окну):
     dLST = LST(объект) - LST(дальний контроль); сравнение периодов до/после.
  B) Локальная интенсивность охлаждения (LCI, аналог park cooling intensity):
     LCI = LST(кольца 0-240 м вокруг объекта) - LST(объект). LCI > 0: объект прохладнее соседей.
     Считается по каждому снимку и сравнивается по периодам.
  C) Профиль: LST(кольцо) - LST(объект) для колец 0-60 ... 480-960 м, по периодам.
  Плюс NDVI объекта по годам: Landsat 30 м (из снимков) и Sentinel-2 10 м (летняя медиана).
"""
import argparse, os, sys, json
import numpy as np
import pandas as pd
from scipy import stats

PERIODS = {"botanic": {"construction": (2013, 2017), "mature": (2021, 2026)},
           "greenline": {"before": (2013, 2019), "after": (2021, 2026)}}
LABELS = {"botanic": "Botanical Garden", "greenline": "GreenLine boulevard"}
LOCAL = ["ring_0_60", "ring_60_120", "ring_120_240"]
RING_ORDER = ["ring_0_60", "ring_60_120", "ring_120_240", "ring_240_480", "ring_480_960"]
RING_MID = {"ring_0_60": 30, "ring_60_120": 90, "ring_120_240": 180,
            "ring_240_480": 360, "ring_480_960": 720}


def boot_ci(x, n, rng):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan, np.nan
    m = rng.choice(x, (n, len(x))).mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def boot_diff(a, b, n, rng):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    d = rng.choice(b, (n, len(b))).mean(1) - rng.choice(a, (n, len(a))).mean(1)
    return np.percentile(d, 2.5), np.percentile(d, 97.5)


def period_of(site, year):
    for lab, (y0, y1) in PERIODS[site].items():
        if y0 <= year <= y1:
            return lab
    return "transition"


def compare(df, col, site, n, rng):
    """сводка по периодам и тест до/после для колонки col"""
    rows = []
    labs = list(PERIODS[site])
    for lab in labs + ["transition"]:
        g = df[df.period == lab][col].dropna()
        lo, hi = boot_ci(g, n, rng)
        rows.append(dict(site=site, metric=col, period=lab, n_acq=len(g), mean=g.mean(),
                         sd=g.std(), ci_lo=lo, ci_hi=hi, median=g.median()))
    A = df[df.period == labs[0]][col].dropna(); B = df[df.period == labs[1]][col].dropna()
    lo, hi = boot_diff(A, B, n, rng)
    p = stats.mannwhitneyu(B, A, alternative="two-sided").pvalue if len(A) > 1 and len(B) > 1 else np.nan
    sub = df[[col, "year"]].dropna()
    ts = stats.theilslopes(sub[col], sub.year) if len(sub) > 2 else (np.nan,) * 4
    test = dict(site=site, metric=col, before=labs[0], after=labs[1], n_before=len(A), n_after=len(B),
                diff_after_minus_before=B.mean() - A.mean(), ci_lo=lo, ci_hi=hi, mannwhitney_p=p,
                trend_per_yr=ts[0], trend_lo=ts[2], trend_hi=ts[3])
    return rows, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lst_csv"); ap.add_argument("s2_csv"); ap.add_argument("out")
    ap.add_argument("--min-valid", type=float, default=0.8)
    ap.add_argument("--stat", default="lst_median", choices=["lst_median", "lst_mean"])
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    d = pd.read_csv(a.lst_csv)
    need = {"acq_key", "year", "site", "zone", a.stat, "lst_mean", "ndvi_mean", "n_valid", "n_total"}
    if need - set(d.columns):
        sys.exit(f"Нет колонок: {need - set(d.columns)}")
    d = d.rename(columns={"site": "obj"})
    d["frac"] = d.n_valid / d.n_total.replace(0, np.nan)
    print(f"Строк {len(d)}, снимков {d.acq_key.nunique()}, годы {d.year.min()}-{d.year.max()}")
    print("Снимков по годам:", d.drop_duplicates("acq_key").groupby("year").size().to_dict())

    ok = d[(d.frac >= a.min_valid) & d[a.stat].notna()]
    W = ok.pivot_table(index=["acq_key", "year", "obj"], columns="zone", values=a.stat).reset_index()
    N = ok.pivot_table(index=["acq_key", "year", "obj"], columns="zone", values="n_valid").reset_index()
    V = ok.pivot_table(index=["acq_key", "year", "obj"], columns="zone", values="ndvi_mean").reset_index()
    LM = ok.pivot_table(index=["acq_key", "year", "obj"], columns="zone", values="lst_mean").reset_index()

    acq_rows, summ, tests, prof = [], [], [], []
    notes = []
    for site in PERIODS:
        w = W[W.obj == site].copy()
        if w.empty or "site" not in w.columns:
            notes.append(f"{site}: нет данных"); continue
        n = N[N.obj == site].set_index("acq_key"); lm = LM[LM.obj == site].set_index("acq_key")
        w = w.set_index("acq_key")
        w["period"] = [period_of(site, y) for y in w.year]
        # A) DiD к дальнему контролю
        w["dLST_ctrl"] = w["site"] - w.get("ctrl_auto", np.nan)
        # B) LCI: средняя LST колец 0-240 м (взвешенная по числу пикселей) минус объект
        have = [z for z in LOCAL if z in lm.columns]
        num = sum(lm[z].fillna(0) * n[z].fillna(0) for z in have)
        den = sum(n[z].fillna(0) for z in have)
        allok = np.all([lm[z].notna() for z in have], axis=0)
        local = (num / den.replace(0, np.nan)).where(allok)
        w["LST_local"] = local.reindex(w.index)
        site_mean = lm["site"].reindex(w.index)
        w["LCI"] = w["LST_local"] - site_mean
        v = V[V.obj == site].set_index("acq_key")
        w["ndvi_site"] = v["site"].reindex(w.index)
        w["ndvi_local"] = v[[z for z in LOCAL if z in v]].mean(axis=1).reindex(w.index)
        for col in ["dLST_ctrl", "LCI"]:
            r, t = compare(w.reset_index(), col, site, a.boot, rng)
            summ += r; tests.append(t)
        # C) профиль по кольцам
        for z in RING_ORDER:
            if z not in lm:
                continue
            delta = (lm[z] - lm["site"]).reindex(w.index)
            for lab in list(PERIODS[site]) + ["transition"]:
                x = delta[w.period == lab].dropna()
                lo, hi = boot_ci(x, a.boot, rng)
                prof.append(dict(site=site, period=lab, zone=z, dist_m=RING_MID[z], n_acq=len(x),
                                 dLST_ring_minus_site=x.mean(), ci_lo=lo, ci_hi=hi))
        w["site_name"] = site
        acq_rows.append(w.reset_index())
        notes.append(f"{site}: снимков с объектом {w['site'].notna().sum()}, "
                     f"с дальним контролем {w['dLST_ctrl'].notna().sum()}, с LCI {w['LCI'].notna().sum()}")

    A = pd.concat(acq_rows, ignore_index=True)
    S = pd.DataFrame(summ); T = pd.DataFrame(tests); P = pd.DataFrame(prof)
    A.to_csv(f"{a.out}/recheck_by_acq.csv", index=False)
    S.to_csv(f"{a.out}/recheck_period_summary.csv", index=False)
    T.to_csv(f"{a.out}/recheck_tests.csv", index=False)
    P.to_csv(f"{a.out}/recheck_profile.csv", index=False)

    s2 = pd.read_csv(a.s2_csv)
    s2 = s2.rename(columns={"site": "obj"})
    s2y = s2[s2.zone.isin(["site", "ring_0_60", "ring_120_240"])].pivot_table(
        index=["obj", "year"], columns="zone", values="s2_ndvi_median").reset_index()
    ly = A.groupby(["site_name", "year"])[["ndvi_site", "ndvi_local", "LCI", "dLST_ctrl"]].mean().reset_index()
    s2y.to_csv(f"{a.out}/recheck_s2_ndvi_by_year.csv", index=False)
    ly.to_csv(f"{a.out}/recheck_landsat_by_year.csv", index=False)

    # --- рисунок: строки = объекты; столбцы = LCI по годам, профиль, NDVI ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    C = {"botanic": "#2a7f62", "greenline": "#3b6fb6"}
    PC = {0: "#b35806", 1: "#1b7837"}
    fig, ax = plt.subplots(2, 3, figsize=(7.16, 4.6))
    for i, site in enumerate(PERIODS):
        g = A[A.site_name == site]
        a0, a1, a2 = ax[i]
        a0.axhline(0, color="#888", lw=0.8)
        a0.plot(g.year + rng.uniform(-0.15, 0.15, len(g)), g.LCI, "o", ms=3, color=C[site], alpha=0.6, mec="none")
        yr = g.groupby("year").LCI.mean()
        a0.plot(yr.index, yr.values, "-", color="#222", lw=1.2)
        a0.set_ylabel("LCI, °C")
        a0.set_title(LABELS[site], fontsize=9, loc="left")
        a1.axhline(0, color="#888", lw=0.8)
        for k, lab in enumerate(PERIODS[site]):
            p = P[(P.site == site) & (P.period == lab)].sort_values("dist_m")
            a1.errorbar(p.dist_m, p.dLST_ring_minus_site, yerr=[p.dLST_ring_minus_site - p.ci_lo,
                        p.ci_hi - p.dLST_ring_minus_site], fmt="o-", ms=3.5, lw=1.2, capsize=2,
                        color=PC[k], label=lab)
        a1.set_xscale("log"); a1.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); a1.set_xticks([30, 90, 180, 360, 720]); a1.set_xticklabels(["30", "90", "180", "360", "720"])
        a1.set_ylabel("LST(ring) − LST(site), °C"); a1.legend(frameon=False, fontsize=7)
        s = s2y[s2y.obj == site]
        if "site" in s:
            a2.plot(s.year, s["site"], "s-", ms=3.5, color=C[site], label="S2 10 m, site")
        if "ring_0_60" in s:
            a2.plot(s.year, s["ring_0_60"], "s--", ms=3.5, color="#999", label="S2 10 m, ring 0–60 m")
        l = ly[ly.site_name == site]
        a2.plot(l.year, l.ndvi_site, "o:", ms=3, color=C[site], alpha=0.7, label="Landsat 30 m, site")
        a2.set_ylabel("Summer NDVI")
        if i == 1:
            a0.set_xlabel("Year"); a1.set_xlabel("Distance from edge, m"); a2.set_xlabel("Year")
    from matplotlib.lines import Line2D
    hl = [Line2D([], [], color="#444", marker="s", ms=3.5, ls="-", label="Sentinel-2 10 m, site"),
          Line2D([], [], color="#999", marker="s", ms=3.5, ls="--", label="Sentinel-2 10 m, ring 0–60 m"),
          Line2D([], [], color="#444", marker="o", ms=3, ls=":", label="Landsat 30 m, site")]
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.legend(handles=hl, loc="lower right", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.99, 0.0))
    fig.savefig(f"{a.out}/fig_case_recheck.pdf"); fig.savefig(f"{a.out}/fig_case_recheck.png", dpi=200)

    with open(f"{a.out}/notes.txt", "w") as f:
        f.write("\n".join(notes) + "\n" + json.dumps(dict(min_valid=a.min_valid, stat=a.stat,
                boot=a.boot, seed=a.seed, periods=PERIODS), indent=1) + "\n")
    pd.set_option("display.width", 200)
    print("\n".join(notes))
    print("\nСводка по периодам:\n", S.round(3).to_string(index=False))
    print("\nДо/после:\n", T.round(4).to_string(index=False))
    print("\nNDVI Sentinel-2 по годам:\n", s2y.round(3).to_string(index=False))
    print("\nLandsat по годам:\n", ly.round(2).to_string(index=False))


if __name__ == "__main__":
    main()