#!/usr/bin/env python3
"""Redraws the article figures from result tables (Times-like STIX font).
Usage: python make_figures.py RESULTS_MAIN_DIR OUT_DIR [pred_obs_sample.parquet]"""
import sys, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyproj import Transformer
plt.rcParams.update({"font.size": 8, "font.family": "STIXGeneral", "mathtext.fontset": "stix",
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
fig, ax = plt.subplots(1, 2, figsize=(3.5, 2.5), sharey=True)
ax[0].barh([LAB[f] for f in order], sh[order], color="#1f5f8b")
ax[0].set_xlabel("Mean |SHAP| (°C)"); ax[0].set_title("(a) XGBoost, SHAP", fontsize=8)
ax[1].barh([LAB[f] for f in order], pm.loc[order, "mae_increase_mean"], xerr=pm.loc[order, "mae_increase_sd"],
           color="#7a9cc6", error_kw={"lw": 0.6})
ax[1].set_xlabel("Increase in MAE (°C)"); ax[1].set_title("(b) RF, permutation", fontsize=8)
for a in ax: a.tick_params(axis="y", length=0)
fig.tight_layout(); fig.savefig(f"{out}/fig_shap.pdf"); fig.savefig(f"{out}/fig_shap.png", dpi=200); plt.close(fig)

# Fig. 3: mean error per 4x4 km block; empty blocks grey; colour scale clipped at 98th percentile
r = pd.read_csv(f"{res}/residuals_by_block.csv")
nby, nbx = 18, 15                       # window 60 x 72 km = 15 x 18 blocks of 4 km
grid = np.full((nby, nbx), np.nan); grid[r.by, r.bx] = r.bias
v = float(np.nanpercentile(np.abs(grid), 98))
cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("#d9d9d9")
fig, ax = plt.subplots(figsize=(3.0, 3.4))
im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=-v, vmax=v, extent=[0, nbx * 4, nby * 4, 0])
t = Transformer.from_crs(4326, 32642, always_xy=True)
cx, cy = t.transform(71.4305, 51.1283)  # Bayterek, city centre
ax.plot((cx - 642000) / 1000, (5706000 - cy) / 1000, marker="*", ms=8, mfc="black", mec="white", mew=0.6)
ax.annotate("city centre", ((cx - 642000) / 1000, (5706000 - cy) / 1000), xytext=(4, -6),
            textcoords="offset points", fontsize=7)
ax.set_xlabel("Distance from western edge (km)"); ax.set_ylabel("Distance from northern edge (km)")
fig.colorbar(im, ax=ax, label="Mean error (°C)", shrink=0.8, extend="both")
fig.tight_layout(); fig.savefig(f"{out}/fig_block_bias.pdf"); fig.savefig(f"{out}/fig_block_bias.png", dpi=200); plt.close(fig)

# Fig. 1: predicted vs observed (needs pred_obs_sample.parquet, not in the repository)
if len(sys.argv) > 3:
    p = pd.read_parquet(sys.argv[3])
    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    hb = ax.hexbin(p.y, p.p, gridsize=60, bins="log", cmap="viridis", mincnt=1)
    lim = np.nanpercentile(np.r_[p.y, p.p], [0.5, 99.5])
    ax.plot(lim, lim, color="black", lw=0.7); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Observed LST anomaly (°C)"); ax.set_ylabel("Predicted LST anomaly (°C)")
    fig.colorbar(hb, ax=ax, label="Count (log)")
    fig.tight_layout(); fig.savefig(f"{out}/fig_pred_obs.pdf"); plt.close(fig)
print("v (colour limit) =", round(v, 2))
