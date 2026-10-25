"""
train_models.py  -  модели LST для Results (RQ1-RQ3) по astana_dataset.parquet.

Запуск (из папки astana_uhi, в .venv):
  python scripts/train_models.py out/astana_dataset.parquet out/feature_groups.json out/results

Быстрый пробный прогон (минуты):
  python scripts/train_models.py out/astana_dataset.parquet out/feature_groups.json out/results_quick --quick

ЦЕЛЬ. LST_ANOM = LST_C минус медиана LST_C этой же акквизиции (по всем точкам снимка).
  Около 2/3 дисперсии LST приходится на различия между снимками (погода, дата), а признаки погоду не
  описывают; поэтому модель предсказывает пространственную аномалию внутри снимка. Это отклонение от
  Methods, его нужно описать в тексте.

ПРОВЕРКА.
  spatial  : 2013-2023, GroupKFold по block_id (блоки 4 x 4 км), k фолдов.
  temporal : обучение на 2013-2023 (все блоки), тест на 2024-2025.
  Для s2_check годы 2019-2023 / 2024-2025 (Sentinel-2 есть только с 2019).
  Обучающая выборка в фолде ограничена --max-train строк (SVR: --max-train-svr), тестовая --max-test
  строк; одна и та же тестовая подвыборка используется для всех моделей и конфигураций фолда, поэтому
  сравнения парные.

МОДЕЛИ. scene_mean (базовая: средняя аномалия снимка по обучающим блокам), linear (линейная
  регрессия), rf (Random Forest), xgb (XGBoost), svr (SVR, RBF).

НЕОПРЕДЕЛЁННОСТЬ И ТЕСТЫ.
  - метрики по фолдам: среднее и SD;
  - 95% бутстрэп-ДИ по блокам (пересэмплинг блоков, --boot раз) для объединённых предсказаний spatial CV;
  - парный тест Уилкоксона по MAE на блоках (одни и те же блоки и строки), поправка Холма:
      лучшая модель против каждой другой (конфигурация main);
      same_scene против main для каждой модели (завышение);
  - завышение: dR2 = R2(same_scene) - R2(main) по фолдам.

ВАЖНОСТЬ ПРИЗНАКОВ. SHAP для XGBoost (встроенный pred_contribs, без пакета shap) на конфигурации main,
  плюс permutation importance для лучшей модели.

ВЫХОД (в папке out):
  dataset_summary.csv, split_sizes.csv, metrics_folds.csv, metrics_summary.csv, tests.csv,
  inflation.csv, shap_importance.csv, permutation_importance.csv, errors_by_group.csv,
  residuals_by_block.csv, run_info.json, figures/*.pdf, tables/*.tex
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SEED_DEFAULT = 42


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ------------------------------------------------------------------ метрики
def metrics(y, p):
    e = p - y
    ss = np.sum((y - y.mean()) ** 2)
    return {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "R2": float(1 - np.sum(e ** 2) / ss) if ss > 0 else np.nan, "bias": float(np.mean(e))}


def block_boot_ci(df, rng, n_boot):
    """95% ДИ метрик: пересэмплинг блоков с возвращением. df: y, p, block_id."""
    blocks = df["block_id"].to_numpy()
    ub, inv = np.unique(blocks, return_inverse=True)
    y, p = df["y"].to_numpy(), df["p"].to_numpy()
    # суммы по блокам для быстрого пересчёта
    nb = len(ub)
    cnt = np.bincount(inv, minlength=nb).astype(float)
    sae = np.bincount(inv, np.abs(p - y), nb)
    sse = np.bincount(inv, (p - y) ** 2, nb)
    se = np.bincount(inv, p - y, nb)
    sy = np.bincount(inv, y, nb)
    syy = np.bincount(inv, y ** 2, nb)
    out = {k: [] for k in ("MAE", "RMSE", "R2", "bias")}
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, nb, nb), minlength=nb).astype(float)
        n = (w * cnt).sum()
        mae = (w * sae).sum() / n
        mse = (w * sse).sum() / n
        my = (w * sy).sum() / n
        vy = (w * syy).sum() / n - my ** 2
        out["MAE"].append(mae); out["RMSE"].append(np.sqrt(mse))
        out["R2"].append(1 - mse / vy if vy > 0 else np.nan); out["bias"].append((w * se).sum() / n)
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in out.items()}


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


# ------------------------------------------------------------------ модели
def make_model(name, seed, quick):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if name == "linear":
        from sklearn.linear_model import LinearRegression
        return make_pipeline(StandardScaler(), LinearRegression())
    if name == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=50 if quick else 300, min_samples_leaf=5,
                                     max_features=0.5, n_jobs=-1, random_state=seed)
    if name == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=150 if quick else 800, learning_rate=0.05, max_depth=8,
                            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                            tree_method="hist", n_jobs=-1, random_state=seed)
    if name == "svr":
        from sklearn.svm import SVR
        return make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale"))
    raise ValueError(name)


def fit_predict(name, feats, tr, te, seed, quick):
    if name == "scene_mean":
        m = tr.groupby("acq_key", observed=True)["y"].mean()
        return te["acq_key"].map(m).fillna(tr["y"].mean()).to_numpy(), None
    mdl = make_model(name, seed, quick)
    # на macOS (Accelerate BLAS) matmul выдаёт ложные RuntimeWarning; на результат не влияет
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        mdl.fit(tr[feats].to_numpy(np.float32), tr["y"].to_numpy(np.float32))
        p = mdl.predict(te[feats].to_numpy(np.float32))
    if not np.all(np.isfinite(p)):
        sys.exit(f"{name}: в предсказаниях есть NaN/inf, остановка")
    return p, mdl


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("groups")
    ap.add_argument("out")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--test-years", type=int, nargs="+", default=[2024, 2025])
    ap.add_argument("--models", nargs="+", default=["scene_mean", "linear", "rf", "xgb", "svr"])
    ap.add_argument("--configs", nargs="+", default=["main", "s2_check", "same_scene"])
    ap.add_argument("--max-train", type=int, default=300_000)
    ap.add_argument("--max-train-svr", type=int, default=20_000)
    ap.add_argument("--max-test", type=int, default=150_000)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--quick", action="store_true", help="мало деревьев и строк, для проверки")
    a = ap.parse_args()
    if a.quick:
        a.max_train, a.max_train_svr, a.max_test, a.boot = 30_000, 3_000, 15_000, 200

    t0 = time.time()
    out = Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)
    rng = np.random.default_rng(a.seed)
    groups = json.loads(Path(a.groups).read_text(encoding="utf-8"))
    configs = {k: v for k, v in groups["configs"].items() if k in a.configs}
    # main_2019: те же признаки, что main, но только 2019+ (честное сравнение с s2_check)
    if "main_2019" in a.configs:
        configs["main_2019"] = groups["configs"]["main"]

    log("читаю таблицу ...")
    need = set(["pixel_id", "block_id", "urban_stratum", "acq_key", "year", "LST_C", "LCZ_REF", "LOO_WIDE"])
    for v in configs.values():
        need |= set(v)
    d = pd.read_parquet(a.parquet, columns=sorted(need))
    d["acq_key"] = d["acq_key"].astype("category")
    for c in d.columns:
        if d[c].dtype == np.float64:
            d[c] = d[c].astype(np.float32)
    n_all = len(d)

    # --- цель: аномалия внутри снимка
    d["LST_ANOM"] = d["LST_C"] - d.groupby("acq_key", observed=True)["LST_C"].transform("median")
    acq_mean = d.groupby("acq_key", observed=True)["LST_C"].transform("mean")
    share_between = float(((acq_mean - d["LST_C"].mean()) ** 2).mean() / d["LST_C"].var())
    d["y"] = d["LST_ANOM"].astype(np.float32)   # цель модели

    # --- сводка данных
    ds = (d.groupby("year").agg(acquisitions=("acq_key", "nunique"), rows=("y", "size"),
                                points=("pixel_id", "nunique"), lst_mean=("LST_C", "mean"),
                                lst_sd=("LST_C", "std"), anom_sd=("LST_ANOM", "std"))
            .round(3))
    ds.to_csv(out / "dataset_summary.csv")
    removed = {c: int(d[v].isna().any(axis=1).sum()) for c, v in configs.items()}
    d = d.drop(columns=["LST_ANOM"])
    log(f"строк {n_all}, доля дисперсии LST между снимками {share_between:.3f}")

    test_years = set(a.test_years)
    rows_folds, pooled, block_rows = [], {}, []
    split_rows = []
    from sklearn.model_selection import GroupKFold

    for cname, feats in configs.items():
        dc = d.dropna(subset=feats)
        if cname in ("s2_check", "main_2019"):
            dc = dc[dc["year"] >= 2019]
        yrs = dc["year"].to_numpy()
        blk = dc["block_id"].to_numpy()
        dev_pos = np.flatnonzero(~np.isin(yrs, list(test_years)))
        hold_pos = np.flatnonzero(np.isin(yrs, list(test_years)))
        log(f"[{cname}] признаки {feats}; разработка {len(dev_pos)}, отложенные годы {len(hold_pos)}")

        # храним только номера строк, а не копии таблицы (иначе не хватает памяти)
        splits = []
        gkf = GroupKFold(n_splits=a.folds)
        for k, (itr, ite) in enumerate(gkf.split(dev_pos, groups=blk[dev_pos])):
            splits.append((f"spatial_f{k + 1}", dev_pos[itr], dev_pos[ite]))
        splits.append(("temporal", dev_pos, hold_pos))

        for sname, ptr, pte in splits:
            split_rows.append({"config": cname, "split": sname, "train_rows": len(ptr),
                               "test_rows": len(pte), "train_blocks": len(np.unique(blk[ptr])),
                               "test_blocks": len(np.unique(blk[pte]))})
            srng = np.random.default_rng(a.seed + len(split_rows))
            te = dc.iloc[np.sort(srng.choice(pte, min(a.max_test, len(pte)), replace=False))]
            tr = dc.iloc[np.sort(srng.choice(ptr, min(a.max_train, len(ptr)), replace=False))]
            tr_svr = tr.sample(n=min(a.max_train_svr, len(tr)), random_state=a.seed)
            for mname in a.models:
                t1 = time.time()
                p, mdl = fit_predict(mname, feats, tr_svr if mname == "svr" else tr, te, a.seed, a.quick)
                m = metrics(te["y"].to_numpy(), p)
                m.update({"config": cname, "split": sname, "model": mname,
                          "n_train": len(tr_svr if mname == "svr" else tr), "n_test": len(te),
                          "seconds": round(time.time() - t1, 1)})
                rows_folds.append(m)
                kind = "temporal" if sname == "temporal" else "spatial"
                key = (cname, mname, kind)
                part = pd.DataFrame({"y": te["y"].to_numpy(), "p": p, "block_id": te["block_id"].to_numpy(),
                                     "pixel_id": te["pixel_id"].to_numpy(), "acq_key": te["acq_key"].to_numpy(),
                                     "year": te["year"].to_numpy(), "urban_stratum": te["urban_stratum"].to_numpy(),
                                     "LCZ_REF": te["LCZ_REF"].to_numpy(), "LOO_WIDE": te["LOO_WIDE"].to_numpy(),
                                     "split": sname})
                pooled.setdefault(key, []).append(part)
                log(f"  {sname:11s} {mname:10s} MAE={m['MAE']:.3f} R2={m['R2']:.3f} ({m['seconds']} c)")

    pd.DataFrame(split_rows).to_csv(out / "split_sizes.csv", index=False)
    mf = pd.DataFrame(rows_folds)
    mf.to_csv(out / "metrics_folds.csv", index=False)

    # --- сводка: spatial = среднее и SD по фолдам + бутстрэп ДИ; temporal = одно значение + ДИ
    summ = []
    pooled = {k: pd.concat(v, ignore_index=True) for k, v in pooled.items()}
    for (cname, mname, kind), pdf in pooled.items():
        sub = mf[(mf.config == cname) & (mf.model == mname) &
                 (mf.split.str.startswith("spatial") if kind == "spatial" else mf.split.eq("temporal"))]
        ci = block_boot_ci(pdf, rng, a.boot)
        row = {"config": cname, "model": mname, "validation": kind, "n_test_rows": len(pdf),
               "n_test_blocks": pdf["block_id"].nunique()}
        for k in ("MAE", "RMSE", "R2", "bias"):
            row[f"{k}_mean"] = sub[k].mean()
            row[f"{k}_sd"] = sub[k].std(ddof=1) if len(sub) > 1 else np.nan
            row[f"{k}_ci_lo"], row[f"{k}_ci_hi"] = ci[k]
        summ.append(row)
    ms = pd.DataFrame(summ).sort_values(["validation", "config", "MAE_mean"])
    ms.to_csv(out / "metrics_summary.csv", index=False)

    base = "main" if "main" in configs else list(configs)[0]
    # --- выбор лучшей модели: наименьший средний MAE на spatial CV, конфигурация main
    cand = ms[(ms.validation == "spatial") & (ms.config.eq(base))
              & (ms.model != "scene_mean")]
    best = cand.sort_values("MAE_mean").iloc[0]["model"] if len(cand) else None
    log(f"лучшая модель ({base}, spatial CV, MAE): {best}")

    # --- тесты Уилкоксона по блочным MAE
    from scipy.stats import wilcoxon

    def block_mae(pdf):
        return pdf.assign(ae=(pdf.p - pdf.y).abs()).groupby("block_id")["ae"].mean()

    tests = []
    if best is not None:
        b = block_mae(pooled[(base, best, "spatial")])
        for m in a.models:
            if m == best or (base, m, "spatial") not in pooled:
                continue
            o = block_mae(pooled[(base, m, "spatial")])
            j = b.index.intersection(o.index)
            st = wilcoxon(b[j], o[j])
            tests.append({"comparison": f"{best} vs {m}", "family": f"models ({base}, spatial)",
                          "n_blocks": len(j), "median_diff_MAE": float(np.median(b[j] - o[j])),
                          "W": float(st.statistic), "p": float(st.pvalue)})
    for m in a.models:
        if ("same_scene", m, "spatial") in pooled and ("main", m, "spatial") in pooled and m != "scene_mean":
            s_ = block_mae(pooled[("same_scene", m, "spatial")])
            mm = block_mae(pooled[("main", m, "spatial")])
            j = s_.index.intersection(mm.index)
            st = wilcoxon(s_[j], mm[j])
            tests.append({"comparison": f"same_scene vs main ({m})", "family": "inflation",
                          "n_blocks": len(j), "median_diff_MAE": float(np.median(s_[j] - mm[j])),
                          "W": float(st.statistic), "p": float(st.pvalue)})
    for m in a.models:
        if m != "scene_mean" and ("s2_check", m, "spatial") in pooled and ("main_2019", m, "spatial") in pooled:
            s2m = block_mae(pooled[("s2_check", m, "spatial")])
            cm = block_mae(pooled[("main_2019", m, "spatial")])
            j = s2m.index.intersection(cm.index)
            st = wilcoxon(s2m[j], cm[j])
            tests.append({"comparison": f"s2_check vs main_2019 ({m})", "family": "sentinel2 vs composite (2019+)",
                          "n_blocks": len(j), "median_diff_MAE": float(np.median(s2m[j] - cm[j])),
                          "W": float(st.statistic), "p": float(st.pvalue)})
    tt = pd.DataFrame(tests)
    if len(tt):
        for fam in tt.family.unique():
            msk = tt.family == fam
            tt.loc[msk, "p_holm"] = holm(tt.loc[msk, "p"])
    tt.to_csv(out / "tests.csv", index=False)

    # --- завышение R2 по фолдам
    infl = []
    for m in a.models:
        if m == "scene_mean":
            continue
        s_ = mf[(mf.config == "same_scene") & (mf.model == m)].set_index("split")["R2"]
        mm = mf[(mf.config == "main") & (mf.model == m)].set_index("split")["R2"]
        j = s_.index.intersection(mm.index)
        for sp in j:
            infl.append({"model": m, "split": sp, "R2_same_scene": s_[sp], "R2_main": mm[sp], "dR2": s_[sp] - mm[sp]})
    pd.DataFrame(infl).to_csv(out / "inflation.csv", index=False)

    # --- важность признаков на main: обучение на всех годах разработки (подвыборка)
    feats = configs.get("main")
    if feats is not None:
        ok = d[feats].notna().all(axis=1).to_numpy() & ~d["year"].isin(test_years).to_numpy()
        pos = np.flatnonzero(ok)
        irng = np.random.default_rng(a.seed)
        sel = irng.choice(pos, min(a.max_train + 20_000, len(pos)), replace=False)
        tr = d.iloc[np.sort(sel[:min(a.max_train, len(sel) - 1)])]
        ev = d.iloc[np.sort(sel[len(tr):])]
        if "xgb" in a.models:
            import xgboost as xgb
            mdl = make_model("xgb", a.seed, a.quick).fit(tr[feats].to_numpy(np.float32), tr["y"].to_numpy(np.float32))
            contrib = mdl.get_booster().predict(xgb.DMatrix(ev[feats].to_numpy(np.float32)), pred_contribs=True)
            shap_df = pd.DataFrame({"feature": feats, "mean_abs_shap": np.abs(contrib[:, :-1]).mean(0)})
            shap_df.sort_values("mean_abs_shap", ascending=False).to_csv(out / "shap_importance.csv", index=False)
            np.save(out / "shap_values_sample.npy", contrib[:, :-1])
            ev[feats].to_parquet(out / "shap_features_sample.parquet")
        if best is not None:
            from sklearn.inspection import permutation_importance
            mdl_b = make_model(best, a.seed, a.quick)
            trb = tr.sample(n=min(a.max_train_svr, len(tr)), random_state=a.seed) if best == "svr" else tr
            mdl_b.fit(trb[feats].to_numpy(np.float32), trb["y"].to_numpy(np.float32))
            evp = ev.sample(n=min(5_000, len(ev)), random_state=a.seed)
            pi = permutation_importance(mdl_b, evp[feats].to_numpy(np.float32), evp["y"].to_numpy(np.float32),
                                        n_repeats=5, random_state=a.seed, scoring="neg_mean_absolute_error")
            pd.DataFrame({"feature": feats, "model": best, "mae_increase_mean": pi.importances_mean,
                          "mae_increase_sd": pi.importances_std}).sort_values(
                "mae_increase_mean", ascending=False).to_csv(out / "permutation_importance.csv", index=False)

    # --- анализ ошибок для лучшей модели (main, spatial)
    if best is not None:
        pb = pooled[(base, best, "spatial")].copy()
        pb["err"] = pb.p - pb.y
        pb["urban"] = np.where(pb.urban_stratum == 1, "built-up", "non-built")
        pb["period"] = np.where(pb.year >= 2021, "2021-2023", "2013-2020")
        eg = []
        for col in ("urban", "LCZ_REF", "period", "LOO_WIDE"):
            g = pb.groupby(col).agg(n=("err", "size"), MAE=("err", lambda e: e.abs().mean()),
                                    bias=("err", "mean")).reset_index().rename(columns={col: "group"})
            g.insert(0, "grouping", col)
            eg.append(g)
        pd.concat(eg).round(4).to_csv(out / "errors_by_group.csv", index=False)
        rb = pb.groupby("block_id").agg(n=("err", "size"), MAE=("err", lambda e: e.abs().mean()),
                                        bias=("err", "mean")).reset_index()
        rb["bx"] = rb.block_id // 1000
        rb["by"] = rb.block_id % 1000
        rb.to_csv(out / "residuals_by_block.csv", index=False)
        pb.sample(n=min(200_000, len(pb)), random_state=a.seed)[["y", "p"]].to_parquet(out / "pred_obs_sample.parquet")

    # --- рисунки и таблицы
    make_figures(out, ms, best)
    make_tables(out, ms, tt, ds, share_between, removed, n_all)

    info = {"python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor(), "seed": a.seed, "args": vars(a),
            "runtime_min": round((time.time() - t0) / 60, 1), "best_model_main_spatial": best,
            "share_variance_between_acquisitions": share_between, "rows_dropped_nan_features": removed}
    for mod in ("numpy", "pandas", "sklearn", "xgboost", "scipy", "matplotlib", "pyarrow"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            pass
    (out / "run_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"готово за {info['runtime_min']} мин, результаты в {out}")


# ------------------------------------------------------------------ рисунки
NAMES = {"scene_mean": "Scene\nmean", "linear": "Linear", "rf": "RF", "xgb": "XGBoost", "svr": "SVR"}
CNAMES = {"main": "Composite (main)", "s2_check": "Sentinel-2", "same_scene": "Same scene",
          "main_2019": "Composite, 2019+"}


def make_figures(out, ms, best):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False,
                         "axes.spines.right": False})
    fig_dir = out / "figures"
    colors = {"main": "#1f5f8b", "s2_check": "#4a9d5b", "same_scene": "#c8553d", "main_2019": "#7a6fb0"}

    # 1. MAE по моделям и конфигурациям (spatial CV), 95% ДИ
    sp = ms[ms.validation == "spatial"]
    models = [m for m in NAMES if m in sp.model.unique()]
    cfgs = [c for c in CNAMES if c in sp.config.unique()]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    w = 0.8 / max(len(cfgs), 1)
    for i, c in enumerate(cfgs):
        s = sp[sp.config == c].set_index("model").reindex(models)
        xs = np.arange(len(models)) + (i - (len(cfgs) - 1) / 2) * w
        ax.bar(xs, s["MAE_mean"], w, color=colors[c], label=CNAMES[c])
        ax.errorbar(xs, s["MAE_mean"], yerr=[s["MAE_mean"] - s["MAE_ci_lo"], s["MAE_ci_hi"] - s["MAE_mean"]],
                    fmt="none", ecolor="black", lw=0.7, capsize=1.5)
    ax.set_xticks(np.arange(len(models)), [NAMES[m] for m in models])
    ax.set_ylabel("MAE of LST anomaly (°C)")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(fig_dir / "fig_mae_models.pdf"); plt.close(fig)

    # 2. предсказание против наблюдения
    po = out / "pred_obs_sample.parquet"
    if po.exists():
        p = pd.read_parquet(po)
        fig, ax = plt.subplots(figsize=(3.3, 3.0))
        hb = ax.hexbin(p.y, p.p, gridsize=60, bins="log", cmap="viridis", mincnt=1)
        lim = np.nanpercentile(np.r_[p.y, p.p], [0.5, 99.5])
        ax.plot(lim, lim, color="black", lw=0.7)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("Observed LST anomaly (°C)"); ax.set_ylabel("Predicted LST anomaly (°C)")
        fig.colorbar(hb, ax=ax, label="Count (log)")
        fig.tight_layout(); fig.savefig(fig_dir / "fig_pred_obs.pdf"); plt.close(fig)

    # 3. SHAP
    sh = out / "shap_importance.csv"
    if sh.exists():
        s = pd.read_csv(sh).sort_values("mean_abs_shap")
        fig, ax = plt.subplots(figsize=(3.3, 2.2))
        ax.barh(s.feature, s.mean_abs_shap, color="#1f5f8b")
        ax.set_xlabel("Mean |SHAP| (°C)")
        fig.tight_layout(); fig.savefig(fig_dir / "fig_shap.pdf"); plt.close(fig)

    # 4. карта средней ошибки по блокам
    rb = out / "residuals_by_block.csv"
    if rb.exists():
        r = pd.read_csv(rb)
        grid = np.full((r.by.max() + 1, r.bx.max() + 1), np.nan)
        grid[r.by, r.bx] = r.bias
        v = np.nanpercentile(np.abs(grid), 98)
        fig, ax = plt.subplots(figsize=(3.0, 3.4))
        im = ax.imshow(grid, cmap="RdBu_r", vmin=-v, vmax=v, extent=[0, grid.shape[1] * 4, grid.shape[0] * 4, 0])
        ax.set_xlabel("Easting from window edge (km)"); ax.set_ylabel("Northing from window top (km)")
        fig.colorbar(im, ax=ax, label="Mean error (°C)", shrink=0.8)
        fig.tight_layout(); fig.savefig(fig_dir / "fig_block_bias.pdf"); plt.close(fig)


def fmt(m, lo, hi, nd=2):
    return f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def make_tables(out, ms, tt, ds, share_between, removed, n_all):
    tab = out / "tables"
    rows = []
    for _, r in ms.sort_values(["validation", "config", "MAE_mean"], ascending=[False, True, True]).iterrows():
        rows.append(f"{r.validation} & {CNAMES.get(r.config, r.config)} & {NAMES.get(r.model, r.model)} & "
                    f"{fmt(r.MAE_mean, r.MAE_ci_lo, r.MAE_ci_hi)} & {fmt(r.RMSE_mean, r.RMSE_ci_lo, r.RMSE_ci_hi)} & "
                    f"{fmt(r.R2_mean, r.R2_ci_lo, r.R2_ci_hi)} & {r.bias_mean:+.2f} \\\\")
    (tab / "main_results_rows.tex").write_text("\n".join(rows), encoding="utf-8")
    ds.to_csv(tab / "dataset_by_year.csv")
    (tab / "notes.txt").write_text(
        f"rows total {n_all}\nshare of LST variance between acquisitions {share_between:.3f}\n"
        f"rows dropped (NaN features) per config {removed}\n", encoding="utf-8")


if __name__ == "__main__":
    main()