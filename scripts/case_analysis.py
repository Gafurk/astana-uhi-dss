#!/usr/bin/env python3
"""
case_analysis.py  —  RQ4: охлаждающий эффект озеленения (до/после).

Вход:  astana_case_sites.csv из 09_case_sites.js
       (acq_key, sensor, date, year, site, lst_mean, lst_median, ndvi_mean, n_valid, n_total)
Выход: папка out/ с таблицами, рисунком PDF и notes.txt

Запуск:
  python case_analysis.py data/astana_case_sites.csv out/case --min-valid 0.8 --boot 5000 --seed 42

Метод:
  Для каждой пары (объект, контроль) берём только снимки, где у ОБОИХ доля валидных
  пикселей >= min-valid. Разность dLST = LST_объект - LST_контроль (медиана по пикселям)
  в одном и том же снимке убирает погоду и сезон (общие для объекта и контроля).
  Затем сравниваем периоды "до" и "после" (difference-in-differences по снимкам):
  bootstrap 95% CI для среднего dLST в каждом периоде и для разности периодов,
  Mann-Whitney U (двусторонний) и тренд Theil-Sen dLST по году.
  Дополнительно: связь dLST с dNDVI по всем снимкам (Theil-Sen + Spearman).
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
from scipy import stats

PAIRS = {
    # имя_пары: (объект, контроль, периоды {метка: (год_с, год_по)})
    "botanic": ("botanic", "botanic_ctrl_auto",
                {"construction": (2013, 2017), "mature": (2021, 2025)}),
    "greenline": ("greenline", "greenline_ctrl_auto",
                  {"before": (2013, 2019), "after": (2021, 2025)}),
    "greenline_zone": ("greenline_zone", "greenline_ctrl_auto",
                       {"before": (2013, 2019), "after": (2021, 2025)}),
}
LABELS = {"botanic": "Botanical Garden", "greenline": "GreenLine boulevard",
          "greenline_zone": "GreenLine + 100 m"}


def boot_mean_ci(x, n, rng):
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.nan, np.nan
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def boot_diff_ci(a, b, n, rng):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    d = (rng.choice(b, (n, len(b))).mean(1) - rng.choice(a, (n, len(a))).mean(1))
    return np.percentile(d, 2.5), np.percentile(d, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("out")
    ap.add_argument("--min-valid", type=float, default=0.8)
    ap.add_argument("--stat", default="lst_median", choices=["lst_median", "lst_mean"])
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    df = pd.read_csv(a.csv)
    need = {"acq_key", "date", "year", "site", a.stat, "ndvi_mean", "n_valid", "n_total"}
    miss = need - set(df.columns)
    if miss:
        sys.exit(f"Нет колонок: {miss}")
    df["frac"] = df["n_valid"] / df["n_total"].replace(0, np.nan)
    print(f"Строк: {len(df)}, снимков: {df.acq_key.nunique()}, объекты: {sorted(df.site.unique())}")

    pairs_rows, summ_rows, test_rows, notes = [], [], [], []
    for pname, (site, ctrl, periods) in PAIRS.items():
        s = df[df.site == site].set_index("acq_key")
        c = df[df.site == ctrl].set_index("acq_key")
        if s.empty or c.empty:
            notes.append(f"{pname}: нет строк для {site} или {ctrl}, пропуск")
            continue
        j = s.join(c, lsuffix="_s", rsuffix="_c", how="inner")
        ok = (j.frac_s >= a.min_valid) & (j.frac_c >= a.min_valid) \
            & j[f"{a.stat}_s"].notna() & j[f"{a.stat}_c"].notna()
        notes.append(f"{pname}: снимков всего {len(j)}, прошли фильтр {int(ok.sum())}")
        j = j[ok].copy()
        j["pair"] = pname
        j["year"] = j["year_s"]
        j["date"] = pd.to_datetime(j["date_s"])
        j["sensor"] = j["sensor_s"]
        j["lst_site"] = j[f"{a.stat}_s"]
        j["lst_ctrl"] = j[f"{a.stat}_c"]
        j["dLST"] = j.lst_site - j.lst_ctrl
        j["ndvi_site"] = j.ndvi_mean_s
        j["ndvi_ctrl"] = j.ndvi_mean_c
        j["dNDVI"] = j.ndvi_site - j.ndvi_ctrl
        j["period"] = "transition"
        for lab, (y0, y1) in periods.items():
            j.loc[j.year.between(y0, y1), "period"] = lab
        cols = ["pair", "date", "year", "sensor", "period", "lst_site", "lst_ctrl", "dLST",
                "ndvi_site", "ndvi_ctrl", "dNDVI", "frac_s", "frac_c"]
        pairs_rows.append(j.reset_index()[["acq_key"] + cols])

        labs = list(periods)
        for lab in labs + ["transition"]:
            g = j[j.period == lab]
            lo, hi = boot_mean_ci(g.dLST, a.boot, rng)
            summ_rows.append(dict(pair=pname, period=lab, n_acq=len(g),
                                  years=f"{g.year.min()}-{g.year.max()}" if len(g) else "",
                                  dLST_mean=g.dLST.mean(), dLST_sd=g.dLST.std(),
                                  dLST_ci_lo=lo, dLST_ci_hi=hi, dLST_median=g.dLST.median(),
                                  ndvi_site=g.ndvi_site.mean(), ndvi_ctrl=g.ndvi_ctrl.mean(),
                                  dNDVI_mean=g.dNDVI.mean(), lst_ctrl_mean=g.lst_ctrl.mean()))
        A = j[j.period == labs[0]].dLST
        B = j[j.period == labs[1]].dLST
        lo, hi = boot_diff_ci(A, B, a.boot, rng)
        if len(A) >= 2 and len(B) >= 2:
            U, p = stats.mannwhitneyu(B, A, alternative="two-sided")
            t, pw = stats.ttest_ind(B, A, equal_var=False)
        else:
            U = p = pw = np.nan
        ts = stats.theilslopes(j.dLST, j.year) if len(j) >= 3 else (np.nan,) * 4
        tau, ptau = stats.kendalltau(j.year, j.dLST) if len(j) >= 3 else (np.nan, np.nan)
        rho, prho = stats.spearmanr(j.dNDVI, j.dLST) if len(j) >= 3 else (np.nan, np.nan)
        tsn = stats.theilslopes(j.dLST, j.dNDVI) if len(j) >= 3 else (np.nan,) * 4
        test_rows.append(dict(pair=pname, before=labs[0], after=labs[1],
                              n_before=len(A), n_after=len(B),
                              did_mean=B.mean() - A.mean(), did_ci_lo=lo, did_ci_hi=hi,
                              mannwhitney_U=U, mannwhitney_p=p, welch_p=pw,
                              trend_C_per_yr=ts[0], trend_lo=ts[2], trend_hi=ts[3],
                              kendall_tau=tau, kendall_p=ptau,
                              slope_dLST_per_dNDVI=tsn[0], slope_lo=tsn[2], slope_hi=tsn[3],
                              spearman_rho=rho, spearman_p=prho))

    if not pairs_rows:
        sys.exit("Ни одна пара не прошла фильтр. Проверь имена объектов в CSV.")
    P = pd.concat(pairs_rows, ignore_index=True)
    S = pd.DataFrame(summ_rows)
    T = pd.DataFrame(test_rows)
    P.to_csv(f"{a.out}/case_pairs_by_acq.csv", index=False)
    S.to_csv(f"{a.out}/case_period_summary.csv", index=False)
    T.to_csv(f"{a.out}/case_tests.csv", index=False)

    # чувствительность к порогу покрытия
    sens = []
    for thr in (0.5, 0.7, 0.8, 0.9, 1.0):
        for pname, (site, ctrl, periods) in PAIRS.items():
            s = df[df.site == site].set_index("acq_key"); c = df[df.site == ctrl].set_index("acq_key")
            j = s.join(c, lsuffix="_s", rsuffix="_c", how="inner")
            j = j[(j.frac_s >= thr) & (j.frac_c >= thr) & j[f"{a.stat}_s"].notna()]
            d = j[f"{a.stat}_s"] - j[f"{a.stat}_c"]
            labs = list(periods)
            m = [d[j.year_s.between(*periods[l])].mean() for l in labs]
            n = [int(j.year_s.between(*periods[l]).sum()) for l in labs]
            sens.append(dict(min_valid=thr, pair=pname, n_before=n[0], n_after=n[1],
                             dLST_before=m[0], dLST_after=m[1], did=m[1] - m[0]))
    pd.DataFrame(sens).to_csv(f"{a.out}/case_sensitivity_min_valid.csv", index=False)

    # рисунок: 2 строки (dLST, dNDVI) x 2 столбца (сад, GreenLine); без двойной оси
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    show = ["botanic", "greenline"]
    colors = {"botanic": "#2a7f62", "greenline": "#3b6fb6"}
    fig, ax = plt.subplots(2, 2, figsize=(7.16, 3.6), sharex=True,
                           gridspec_kw={"height_ratios": [1.6, 1]})
    for k, pn in enumerate(show):
        g = P[P.pair == pn].sort_values("date")
        yrs = g.date.dt.year + (g.date.dt.dayofyear - 1) / 365.0
        a0, a1 = ax[0, k], ax[1, k]
        a0.axhline(0, color="#888", lw=0.8)
        a0.plot(yrs, g.dLST, "o", ms=4, color=colors[pn], mec="white", mew=0.6)
        for _, r in S[(S.pair == pn) & (S.period != "transition")].iterrows():
            y0, y1 = map(int, r.years.split("-")) if r.years else (None, None)
            if y0 is None:
                continue
            a0.hlines(r.dLST_mean, y0, y1 + 1, color="#222", lw=1.6)
            a0.fill_between([y0, y1 + 1], r.dLST_ci_lo, r.dLST_ci_hi, color="#222",
                            alpha=0.12, lw=0)
            a0.text(y0 + 0.1, r.dLST_ci_hi + 0.3, f"{r.period}: {r.dLST_mean:+.1f} °C".replace("-", "\u2212"),
                    fontsize=7, color="#222")
        a0.set_title(LABELS[pn], fontsize=9)
        a1.axhline(0, color="#888", lw=0.8)
        a1.plot(yrs, g.dNDVI, "s", ms=3.5, color=colors[pn], mec="white", mew=0.6)
        a1.set_xlabel("Year")
        if k == 0:
            a0.set_ylabel("ΔLST site − control (°C)")
            a1.set_ylabel("ΔNDVI site − control")
    fig.tight_layout()
    fig.savefig(f"{a.out}/fig_case_dlst.pdf")
    fig.savefig(f"{a.out}/fig_case_dlst.png", dpi=200)

    info = dict(min_valid=a.min_valid, stat=a.stat, boot=a.boot, seed=a.seed,
                numpy=np.__version__, pandas=pd.__version__, scipy=__import__("scipy").__version__)
    with open(f"{a.out}/notes.txt", "w") as f:
        f.write("\n".join(notes) + "\n" + json.dumps(info, indent=1) + "\n")
    pd.set_option("display.width", 200)
    print("\n".join(notes))
    print(S.round(3).to_string())
    print(T.round(4).T.to_string())


if __name__ == "__main__":
    main()
