#!/usr/bin/env python3
"""Redraws the article figures (Fig. 1 confusion matrix + distributions, Fig. 2, Fig. 3) from result tables (Times-like STIX font).
Usage: python scripts/05_make_figures.py RESULTS_DIR OUT_DIR [pred_obs_sample.parquet]"""
import sys, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyproj import Transformer
plt.rcParams.update({"font.size": 9.5, "axes.labelsize": 9.5, "xtick.labelsize": 9, "ytick.labelsize": 9, "font.family": "STIXGeneral", "mathtext.fontset": "stix",
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
res, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
LAB = {"LOO_NDBI": "NDBI (LOO)", "POP_DENS": "Population density", "DIST_WATER_M": "Distance to water",
       "LOO_ALBEDO": "Albedo (LOO)", "ELEV": "Elevation", "DIST_GREEN_M": "Distance to green space",
       "LOO_NDVI": "NDVI (LOO)"}

# Fig. 2: SHAP (XGBoost) and permutation importance (RF), same predictor order
sh = pd.read_csv(f"{res}/shap_importance.csv").set_index("feature")["mean_abs_shap"]
pm = pd.read_csv(f"{res}/permutation_importance.csv").set_index("feature")
order = sh.sort_values().index
fig, ax = plt.subplots(1, 2, figsize=(3.5, 2.9), sharey=True)
ax[0].barh([LAB[f] for f in order], sh[order], color="#1f5f8b")
ax[0].set_xlabel("Mean |SHAP| (°C)"); ax[0].set_title("(a) SHAP (XGB)", fontsize=9.5)
ax[1].barh([LAB[f] for f in order], pm.loc[order, "mae_increase_mean"], xerr=pm.loc[order, "mae_increase_sd"],
           color="#7a9cc6", error_kw={"lw": 0.6})
ax[1].set_xlabel("ΔMAE (°C)"); ax[1].set_title("(b) Permutation (RF)", fontsize=9.5)
for a in ax: a.tick_params(axis="y", length=0)
fig.tight_layout(); fig.subplots_adjust(wspace=0.15); fig.savefig(f"{out}/fig_shap.pdf"); fig.savefig(f"{out}/fig_shap.png", dpi=200); plt.close(fig)

# Fig. 3: mean error per 4x4 km block; empty blocks grey; colour scale clipped at 98th percentile
r = pd.read_csv(f"{res}/residuals_by_block.csv")
nby, nbx = 18, 15                       # window 60 x 72 km = 15 x 18 blocks of 4 km
grid = np.full((nby, nbx), np.nan); grid[r.by, r.bx] = r.bias
v = float(np.nanpercentile(np.abs(grid), 98))
cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("#d9d9d9")
fig, ax = plt.subplots(figsize=(3.4, 3.7))
im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=-v, vmax=v, extent=[0, nbx * 4, nby * 4, 0])
t = Transformer.from_crs(4326, 32642, always_xy=True)
cx, cy = t.transform(71.4305, 51.1283)  # Bayterek, city centre
ax.plot((cx - 642000) / 1000, (5706000 - cy) / 1000, marker="*", ms=8, mfc="black", mec="white", mew=0.6)
ax.annotate("city centre", ((cx - 642000) / 1000, (5706000 - cy) / 1000), xytext=(4, -6),
            textcoords="offset points", fontsize=9)
ax.set_xlabel("Distance from western edge (km)"); ax.set_ylabel("Distance from northern edge (km)")
fig.colorbar(im, ax=ax, label="Mean error (°C)", shrink=0.8, extend="both")
fig.tight_layout(); fig.savefig(f"{out}/fig_block_bias.pdf"); fig.savefig(f"{out}/fig_block_bias.png", dpi=200); plt.close(fig)

# Fig. 1: (a) three-class confusion matrix, (b) distributions of observed and predicted anomaly
# (needs pred_obs_sample.parquet: 200,000 random RF test records, spatial CV; not in the repository)
if len(sys.argv) > 3:
    p = pd.read_parquet(sys.argv[3])
    edges = [-np.inf, -2.0, 2.0, np.inf]; names = ["Cooler\n(< −2 °C)", "Near median\n(±2 °C)", "Warmer\n(> +2 °C)"]
    o = np.digitize(p.y, edges[1:-1]); q = np.digitize(p.p, edges[1:-1])
    cm = np.zeros((3, 3), int)
    for i, j in zip(o, q): cm[i, j] += 1
    rowpct = cm / cm.sum(1, keepdims=True) * 100
    pd.DataFrame(cm, index=["obs_cool", "obs_near", "obs_warm"], columns=["pred_cool", "pred_near", "pred_warm"]).to_csv(f"{out}/confusion_matrix_counts.csv")
    fig, ax = plt.subplots(2, 1, figsize=(3.5, 5.4), gridspec_kw={"height_ratios": [1.15, 1]})
    a = ax[0]
    a.imshow(rowpct, cmap="Blues", vmin=0, vmax=100)
    for i in range(3):
        for j in range(3):
            a.text(j, i, f"{rowpct[i, j]:.0f}%\n({cm[i, j]:,})", ha="center", va="center", fontsize=9,
                   color="white" if rowpct[i, j] > 55 else "black")
    short = ["Cooler", "Near", "Warmer"]
    a.set_xticks(range(3), short); a.set_yticks(range(3), short)
    a.set_xlabel("Predicted class"); a.set_ylabel("Observed class")
    a.set_title("(a) Classes: < −2, −2 to +2, > +2 °C", fontsize=9.5)
    for sp in a.spines.values(): sp.set_visible(False)
    a.tick_params(length=0)
    b = ax[1]
    bins = np.arange(-14, 10.5, 0.5)
    hy, _ = np.histogram(p.y, bins); hp, _ = np.histogram(p.p, bins)
    mid = (bins[:-1] + bins[1:]) / 2
    b.plot(mid, hy / 1000, color="#444444", lw=1.6, label="Observed")
    b.plot(mid, hp / 1000, color="#1f5f8b", lw=1.6, label="Predicted (RF)")
    for x in (-2, 2): b.axvline(x, color="#999999", lw=0.8, ls="--")
    b.set_xlabel("LST anomaly (°C)"); b.set_ylabel("Records (thousands)")
    b.legend(frameon=False, fontsize=9)
    b.set_title(f"(b) SD observed {p.y.std():.2f} °C, predicted {p.p.std():.2f} °C", fontsize=9.5)
    fig.tight_layout(); fig.savefig(f"{out}/fig_anomaly_classes.pdf"); fig.savefig(f"{out}/fig_anomaly_classes.png", dpi=200); plt.close(fig)
print("v (colour limit) =", round(v, 2))
