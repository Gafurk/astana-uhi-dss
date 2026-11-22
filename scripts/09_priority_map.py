"""
09_priority_map.py  -  priority map of 1 km cells and a decision example (Section VI-A of the
conference version of the article).

Usage (from the repository root, inside .venv):
  python scripts/09_priority_map.py out/astana_dataset.parquet out/priority \
         --groups out/feature_groups.json --districts data/astana_districts.geojson --name-field name_std

District boundaries: GeoJSON of the city districts of Astana in any CRS, from OpenStreetMap via
overpass-turbo.eu (Run, then Export -> GeoJSON). Since August 2024 the city has six districts
(Almaty, Baikonur, Nura, Saraishyk, Saryarka, Yesil). The name:en tags of OpenStreetMap are not
uniform, so data/astana_districts.geojson adds a field name_std with Latin names
(use --name-field name_std).
    [out:json][timeout:120];
    area["boundary"="administrative"]["name:en"="Astana"]->.a;
    rel(area.a)["boundary"="administrative"]["admin_level"~"^(6|7|8)$"];
    out body;
    >;
    out skel qt;

Steps
  1. Target as in 04_train_models.py: LST anomaly within each acquisition (pixel LST minus the
     acquisition median over all sample points).
  2. Built-up sample pixels (urban_stratum == 1) are averaged over the recent period (default
     2021-2025) and aggregated to cells of --cell-km (default 1 km). The urban area is the group of
     cells with at least --min-pixels such pixels that is 4-connected to the cell of the city
     centre (Bayterek); separate settlements around the city drop out. District polygons
     (optional) only label the cells; cells outside all polygons are labelled "outside".
  3. Indicators, scaled to 0-1 by percentile rank among cells (higher = higher priority):
       E  exposure                    mean LST anomaly of the recent period
       S  sensitivity                 GHSL population density
       A  lack of adaptive capacity   mean of the ranks of the median distances to green space
                                      and to a health facility (OpenStreetMap)
     PI = (E + S + A) / 3. Monte Carlo (--mc runs): each weight multiplied by U(0.8, 1.2) and
     renormalised. Robust priority cell = top decile of PI in at least 90 % of the runs.
  4. Candidate measure per cell (fixed rule):
       NDVI fell by more than 0.1 between 2013-2016 and the recent period -> restore vegetation;
       else median distance to green space > 500 m and NDVI below the median of all cells
                                                                         -> new consolidated park;
       else                                                              -> cool roofs and pavements.
  5. Decision example: random forest with the hyperparameters of 04 (main predictors, 300 trees;
     about 1.3 GB of memory), trained on --max-train records of all years, run on the built-up
     pixels of the robust cells with
       cool  : LOO albedo + --albedo-delta (default 0.05);
       green : LOO NDVI, NDBI and albedo set to the median of vegetated built-up pixels
               (recent-period LOO NDVI >= 0.5).
     The predicted change is correlational: the anomaly that went with such a surface in the data.

Redraw only the figure (e.g. after adding --districts):  ... --plot-only
Output (in the output folder): cells.csv, districts.csv, scenario_cells.csv, run_info.json,
fig_priority_map.pdf and priority_values.tex (LaTeX macros read by main.tex; upload it to
Overleaf next to main.tex together with fig_priority_map.pdf).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

X0, Y0 = 642000.0, 5706000.0                 # upper-left corner of the study window, EPSG:32642
CRS = "EPSG:32642"
CENTRE_LONLAT = (71.4305, 51.1283)           # Bayterek, used only as the city centre
SITES = {                                    # polygons of gee/09_case_rings.js (lon, lat)
    "BG": [(71.41143, 51.11128), (71.40851, 51.10306), (71.42169, 51.10109), (71.42469, 51.10939)],
    "GL": [(71.39764, 51.13375), (71.36966, 51.13801), (71.36957, 51.13771), (71.39751, 51.13324)],
}
MAIN_FEATS = ["ELEV", "POP_DENS", "DIST_WATER_M", "DIST_GREEN_M", "LOO_NDVI", "LOO_NDBI", "LOO_ALBEDO"]
MEASURE_TEXT = {"cool": "cool roofs and pavements",
                "park": "a new consolidated park",
                "restore": "protecting or restoring vegetation"}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def pct_rank(s):
    return s.rank(pct=True, method="average")


def to_utm(lonlat):
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    return [tr.transform(lon, lat) for lon, lat in lonlat]


def spearman_cols(base, mat):
    """Spearman correlation between one vector and every column of a matrix."""
    rb = pd.Series(base).rank().to_numpy()
    rm = pd.DataFrame(mat).rank().to_numpy()
    rb = (rb - rb.mean()) / rb.std()
    rm = (rm - rm.mean(0)) / rm.std(0)
    return (rb[:, None] * rm).mean(0)


def plot_map(cells, districts, out, cell_km, cxy):
    """Fig. of the article: priority index of the cells, robust priority cells with their measure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection
    from shapely.geometry import box
    from shapely.ops import unary_union
    from shapely.affinity import affine_transform
    plt.rcParams.update({"font.size": 8.5, "font.family": "STIXGeneral", "mathtext.fontset": "stix",
                         "pdf.fonttype": 42})
    cell_m = cell_km * 1000
    km = lambda X, Y: ((np.asarray(X) - cxy[0]) / 1000, (np.asarray(Y) - cxy[1]) / 1000)
    r = cells[cells["robust"]]
    top = r.sort_values("PI", ascending=False).iloc[0]
    fig, ax = plt.subplots(figsize=(3.5, 3.9))
    gx, gy = km(cells["X"] - cell_m / 2, cells["Y"] - cell_m / 2)
    pc = PatchCollection([Rectangle((x, y), cell_km, cell_km) for x, y in zip(gx, gy)],
                         cmap="YlOrRd", linewidths=0)
    pc.set_array(cells["PI"].to_numpy()); pc.set_clim(0, 1)
    ax.add_collection(pc)
    # outline of the urban area, built on the integer cell grid so that neighbours share edges
    area = unary_union([box(cx, -cy - 1, cx + 1, -cy) for cx, cy in zip(cells["cx"], cells["cy"])])
    area = affine_transform(area, [cell_km, 0, 0, cell_km, (X0 - cxy[0]) / 1000, (Y0 - cxy[1]) / 1000])
    for g in getattr(area, "geoms", [area]):
        ax.plot(*g.exterior.xy, color="0.3", lw=0.7, zorder=2)
        for ring in g.interiors:
            ax.plot(*ring.xy, color="0.3", lw=0.5, zorder=2)
    allx, ally = km(cells["X"], cells["Y"])
    lim = (allx.min() - 1.5, allx.max() + 1.5, ally.min() - 1.5, ally.max() + 1.5)
    if districts is not None:
        axbox = box(lim[0] + 1, lim[2] + 1, lim[1] - 1, lim[3] - 1)
        for _, row in districts.iterrows():
            gk = affine_transform(row.geometry, [1e-3, 0, 0, 1e-3, -cxy[0] / 1000, -cxy[1] / 1000])
            for g in getattr(gk, "geoms", [gk]):
                ax.plot(*g.exterior.xy, color="0.45", lw=0.5, ls="--", zorder=2)
            # label in the part of the district that lies outside the urban area, if there is one
            inside = gk.intersection(axbox)
            free = inside.difference(area.buffer(0.3))
            parts = [q for q in getattr(free, "geoms", [free]) if q.geom_type == "Polygon" and q.area > 2]
            spot = max(parts, key=lambda q: q.area) if parts else inside
            if spot.is_empty:
                continue
            lp = spot.representative_point()
            ax.text(lp.x, lp.y, row["dname"], fontsize=7, ha="center", va="center", color="0.15", clip_on=True,
                    zorder=4, bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6))
    rx, ry = km(r["X"], r["Y"])
    style = {"cool": ("^", "Cool roofs and pavements"), "park": ("o", "New consolidated park"),
             "restore": ("s", "Restore vegetation")}
    for m, (mk, lab) in style.items():
        sel = r["measure"].to_numpy() == m
        if sel.any():
            ax.scatter(rx[sel], ry[sel], marker=mk, s=9, facecolor="none", edgecolor="black",
                       linewidth=0.6, label=f"{lab} ({int(sel.sum())})", zorder=3)
    ax.scatter(0, 0, marker="+", s=45, color="black", lw=0.9, zorder=5)
    ax.annotate("centre", (0, 0), xytext=(4, 2), textcoords="offset points", fontsize=7, zorder=5)
    offsets = {"BG": (-13, -7), "GL": (-12, 4)}
    for lab, poly in SITES.items():
        sx, sy = km(*np.array(to_utm(poly)).mean(0))
        ax.scatter(sx, sy, marker="*", s=40, color="tab:green", edgecolor="black", lw=0.4, zorder=5)
        ax.annotate(lab, (sx, sy), xytext=offsets.get(lab, (3, 3)), textcoords="offset points",
                    fontsize=7, zorder=5)
    tx, ty = km(top["X"], top["Y"])
    ax.annotate("example", (tx, ty), xytext=(6, -14), textcoords="offset points", fontsize=7,
                arrowprops=dict(arrowstyle="-", lw=0.6), zorder=5)
    ax.set_xlim(lim[0], lim[1]); ax.set_ylim(lim[2], lim[3])
    ax.set_aspect("equal")
    ax.set_xlabel("Easting from city centre (km)"); ax.set_ylabel("Northing from city centre (km)")
    cb = fig.colorbar(pc, ax=ax, fraction=0.045, pad=0.02); cb.set_label("Priority index")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=6.5, frameon=False,
              title="Robust priority cells and proposed measure", title_fontsize=6.5,
              handletextpad=0.3, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(out / "fig_priority_map.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_priority_map.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def read_districts(path, name_field, strip_suffix):
    import geopandas as gpd
    districts = gpd.read_file(path)
    districts = districts[districts.geometry.type.isin(["Polygon", "MultiPolygon"])].to_crs(CRS)
    if name_field not in districts.columns:
        raise SystemExit(f"field {name_field!r} not in {list(districts.columns)}")
    districts["dname"] = districts[name_field].astype(str).str.replace(strip_suffix, "", regex=False)
    return districts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("out")
    ap.add_argument("--groups", default=None, help="feature_groups.json (default: main predictors of 04)")
    ap.add_argument("--districts", default=None, help="GeoJSON of the city districts")
    ap.add_argument("--name-field", default="name:en")
    ap.add_argument("--strip-suffix", default=" District", help="removed from district names in the text")
    ap.add_argument("--recent", type=int, nargs=2, default=[2021, 2025])
    ap.add_argument("--early", type=int, nargs=2, default=[2013, 2016])
    ap.add_argument("--cell-km", type=float, default=1.0)
    ap.add_argument("--min-pixels", type=int, default=15)
    ap.add_argument("--mc", type=int, default=1000)
    ap.add_argument("--albedo-delta", type=float, default=0.05)
    ap.add_argument("--max-train", type=int, default=300_000)
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plot-only", action="store_true",
                    help="only redraw the figure from cells.csv in the output folder (parquet is ignored)")
    a = ap.parse_args()
    if a.plot_only:
        out = Path(a.out)
        cells = pd.read_csv(out / "cells.csv")
        districts = read_districts(a.districts, a.name_field, a.strip_suffix) if a.districts else None
        plot_map(cells, districts, out, a.cell_km, to_utm([CENTRE_LONLAT])[0])
        log(f"figure redrawn in {out}")
        return

    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    feats = MAIN_FEATS
    if a.groups:
        feats = json.loads(Path(a.groups).read_text(encoding="utf-8"))["configs"]["main"]

    # ---------------------------------------------------------------- 1. data and target
    cols = sorted(set(["pixel_id", "x", "y", "urban_stratum", "acq_key", "year", "LST_C",
                       "DIST_HEALTH_M"] + feats))
    log("reading the table ...")
    d = pd.read_parquet(a.parquet, columns=cols)
    d["acq_key"] = d["acq_key"].astype("category")
    for c in d.columns:
        if d[c].dtype == np.float64:
            d[c] = d[c].astype(np.float32)
    d["anom"] = d["LST_C"] - d.groupby("acq_key", observed=True)["LST_C"].transform("median")
    log(f"rows {len(d)}, pixels {d['pixel_id'].nunique()}")

    # ---------------------------------------------------------------- 2. pixels and cells
    built = d[d["urban_stratum"] == 1]
    rec = built[built["year"].between(*a.recent)]
    agg = {"x": "first", "y": "first", "anom": "mean", "acq_key": "size", "DIST_HEALTH_M": "first"}
    agg.update({f: "mean" for f in feats})
    pix = rec.groupby("pixel_id").agg(agg).rename(columns={"acq_key": "n_acq"})
    early = (built[built["year"].between(*a.early)].groupby("pixel_id")["LOO_NDVI"].mean()
             .rename("ndvi_early"))
    pix = pix.join(early)
    cell_m = a.cell_km * 1000
    pix["cx"] = np.floor((pix["x"] - X0) / cell_m).astype(int)
    pix["cy"] = np.floor((Y0 - pix["y"]) / cell_m).astype(int)
    pix["ndvi_change"] = pix["LOO_NDVI"] - pix["ndvi_early"]

    cells = pix.groupby(["cx", "cy"]).agg(
        n_pix=("anom", "size"), anom=("anom", "mean"), pop=("POP_DENS", "mean"),
        green_m=("DIST_GREEN_M", "median"), health_m=("DIST_HEALTH_M", "median"),
        ndvi=("LOO_NDVI", "mean"), ndbi=("LOO_NDBI", "mean"), albedo=("LOO_ALBEDO", "mean"),
        ndvi_change=("ndvi_change", "mean")).reset_index()
    cells = cells[cells["n_pix"] >= a.min_pixels].copy()
    log(f"cells with >= {a.min_pixels} built-up pixels: {len(cells)}")
    from scipy import ndimage
    cxy = to_utm([CENTRE_LONLAT])[0]
    ccx, ccy = int((cxy[0] - X0) // cell_m), int((Y0 - cxy[1]) // cell_m)
    grid = np.zeros((cells["cy"].max() + 2, cells["cx"].max() + 2), dtype=int)
    grid[cells["cy"], cells["cx"]] = 1
    lab, _ = ndimage.label(grid)          # 4-connectivity
    if lab[ccy, ccx] == 0:
        raise SystemExit("the cell of the city centre has too few built-up pixels: lower --min-pixels")
    cells = cells[lab[cells["cy"], cells["cx"]] == lab[ccy, ccx]].copy()
    cells["X"] = X0 + (cells["cx"] + 0.5) * cell_m
    cells["Y"] = Y0 - (cells["cy"] + 0.5) * cell_m
    log(f"cells of the contiguous urban area: {len(cells)}")

    # ---------------------------------------------------------------- districts
    districts = None
    cells["district"] = ""
    if a.districts:
        import geopandas as gpd
        districts = read_districts(a.districts, a.name_field, a.strip_suffix)
        log(f"districts: {sorted(districts['dname'])}")
        pts = gpd.GeoDataFrame(cells, geometry=gpd.points_from_xy(cells["X"], cells["Y"]), crs=CRS)
        j = gpd.sjoin(pts, districts[["dname", "geometry"]], how="left", predicate="within")
        j = j[~j.index.duplicated()]
        cells["district"] = j["dname"].fillna("outside")
        log(f"cells outside all district polygons: {int((cells['district'] == 'outside').sum())}")
    else:
        log("! no --districts: districts are not reported")

    # ---------------------------------------------------------------- 3. priority index
    cells["E"] = pct_rank(cells["anom"])
    cells["S"] = pct_rank(cells["pop"])
    cells["A"] = (pct_rank(cells["green_m"]) + pct_rank(cells["health_m"])) / 2
    M = cells[["E", "S", "A"]].to_numpy()
    cells["PI"] = M.mean(1)
    cells["top"] = cells["PI"] >= cells["PI"].quantile(0.9)
    W = rng.uniform(0.8, 1.2, size=(a.mc, 3))
    W /= W.sum(1, keepdims=True)
    P = M @ W.T
    top_mc = P >= np.quantile(P, 0.9, axis=0)
    cells["top_freq"] = top_mc.mean(1)
    cells["robust"] = cells["top_freq"] >= 0.9
    rho = spearman_cols(cells["PI"].to_numpy(), P)
    log(f"top decile {int(cells['top'].sum())}, robust {int(cells['robust'].sum())}, "
        f"median Spearman {np.median(rho):.3f}")

    # ---------------------------------------------------------------- 4. measure rule
    ndvi_med = cells["ndvi"].median()
    cells["measure"] = np.where(cells["ndvi_change"] < -0.1, "restore",
                                np.where((cells["green_m"] > 500) & (cells["ndvi"] < ndvi_med), "park", "cool"))

    # ---------------------------------------------------------------- 5. scenarios
    from sklearn.ensemble import RandomForestRegressor
    tr = d.dropna(subset=feats + ["anom"])
    if len(tr) > a.max_train:
        tr = tr.sample(a.max_train, random_state=a.seed)
    tr = tr[feats + ["anom"]].copy()
    del d, built, rec
    log(f"training RF on {len(tr)} records ...")
    rf = RandomForestRegressor(n_estimators=a.trees, min_samples_leaf=5, max_features=0.5,
                               n_jobs=-1, random_state=a.seed).fit(tr[feats].to_numpy(), tr["anom"].to_numpy())
    vsel = pix["LOO_NDVI"] >= 0.5
    if vsel.sum() < 50:
        log(f"! only {int(vsel.sum())} built-up pixels with NDVI >= 0.5: the top decile of NDVI is used instead")
        vsel = pix["LOO_NDVI"] >= pix["LOO_NDVI"].quantile(0.9)
    veg = pix[vsel][["LOO_NDVI", "LOO_NDBI", "LOO_ALBEDO"]].median()
    log(f"vegetated profile from {int(vsel.sum())} pixels: {veg.round(3).to_dict()}")
    rob = cells[cells["robust"]][["cx", "cy"]]
    sp = pix.reset_index().merge(rob, on=["cx", "cy"]).dropna(subset=feats)
    X = sp[feats].copy()
    base = rf.predict(X.to_numpy())
    Xc = X.copy(); Xc["LOO_ALBEDO"] = Xc["LOO_ALBEDO"] + a.albedo_delta
    Xg = X.copy()
    for f in ["LOO_NDVI", "LOO_NDBI", "LOO_ALBEDO"]:
        Xg[f] = veg[f]
    sp["d_cool"] = rf.predict(Xc.to_numpy()) - base
    sp["d_green"] = rf.predict(Xg.to_numpy()) - base
    scen = sp.groupby(["cx", "cy"]).agg(n=("d_cool", "size"), d_cool=("d_cool", "mean"),
                                        d_green=("d_green", "mean")).reset_index()
    scen.to_csv(out / "scenario_cells.csv", index=False)

    # ---------------------------------------------------------------- top cell
    top = cells[cells["robust"]].sort_values("PI", ascending=False).iloc[0]
    ts = sp[(sp["cx"] == top["cx"]) & (sp["cy"] == top["cy"])]
    top_km = np.hypot(top["X"] - cxy[0], top["Y"] - cxy[1]) / 1000

    # ---------------------------------------------------------------- tables
    cells.to_csv(out / "cells.csv", index=False)
    if districts is not None:
        ds = (cells.groupby("district").agg(cells=("PI", "size"), top_decile=("top", "sum"),
                                            robust=("robust", "sum"), anom_mean=("anom", "mean"),
                                            pop_median=("pop", "median"))
              .sort_values(["robust", "top_decile"], ascending=False))
        ds["robust_share"] = ds["robust"] / ds["cells"]
        ds.to_csv(out / "districts.csv")
        log("\n" + ds.round(3).to_string())
    else:
        ds = pd.DataFrame()

    # ---------------------------------------------------------------- LaTeX macros
    r, o = cells[cells["robust"]], cells[~cells["robust"]]
    f1 = lambda v: f"{v:.1f}"
    f2 = lambda v: f"{v:.2f}"
    sgn = lambda v: f"${v:+.2f}$"
    rng_txt = lambda s: f"${np.percentile(s, 10):+.2f}$ to ${np.percentile(s, 90):+.2f}$"
    mc = r["measure"].value_counts()
    vals = {
        "PYears": f"{a.recent[0]}--{a.recent[1]}",
        "PCells": f"{len(cells)}",
        "PTop": f"{int(cells['top'].sum())}",
        "PRobust": f"{len(r)}",
        "PRho": f2(np.median(rho)),
        "PAnomRobust": sgn(r["anom"].mean()),
        "PAnomOther": sgn(o["anom"].mean()),
        "PPopRobust": f"{r['pop'].median():,.0f}".replace(",", "{,}"),
        "PPopOther": f"{o['pop'].median():,.0f}".replace(",", "{,}"),
        "PMeasCool": f"{int(mc.get('cool', 0))}",
        "PMeasPark": f"{int(mc.get('park', 0))}",
        "PMeasRestore": f"{int(mc.get('restore', 0))}",
        "PVegNDVI": f2(veg["LOO_NDVI"]), "PVegNDBI": f2(veg["LOO_NDBI"]), "PVegAlb": f2(veg["LOO_ALBEDO"]),
        "TopDist": str(top["district"]) or "--",
        "TopKm": f1(top_km),
        "TopAnom": sgn(top["anom"]),
        "TopPop": f"{top['pop']:,.0f}".replace(",", "{,}"),
        "TopGreen": f"{top['green_m']:.0f}",
        "TopHealth": f"{top['health_m']:.0f}",
        "TopNDVI": f2(top["ndvi"]), "TopNDBI": f2(top["ndbi"]), "TopAlb": f2(top["albedo"]),
        "TopMeasure": MEASURE_TEXT[top["measure"]],
        "ScenCool": sgn(ts["d_cool"].mean()), "ScenCoolRange": rng_txt(ts["d_cool"]),
        "ScenGreen": sgn(ts["d_green"].mean()), "ScenGreenRange": rng_txt(ts["d_green"]),
        "ScenCoolAll": sgn(scen["d_cool"].mean()), "ScenGreenAll": sgn(scen["d_green"].mean()),
        "AlbDelta": f"{a.albedo_delta:.2f}",
    }
    dsi = ds.drop(index="outside", errors="ignore") if len(ds) else ds
    if len(dsi):
        vals.update({"PDistA": dsi.index[0], "PDistAN": f"{int(dsi.iloc[0]['robust'])}",
                     "PDistB": dsi.index[1] if len(dsi) > 1 else "--",
                     "PDistBN": f"{int(dsi.iloc[1]['robust'])}" if len(dsi) > 1 else "0",
                     "PRobustOutside": f"{int(ds.loc['outside', 'robust']) if 'outside' in ds.index else 0}"})
    for k, v in vals.items():
        if not str(v).isascii():
            log(f"! {k} = {v!r} is not ASCII: use a field with Latin names (--name-field)")
    tex = ["% written by scripts/09_priority_map.py; do not edit by hand"]
    tex += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in vals.items()]
    (out / "priority_values.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    plot_map(cells, districts, out, a.cell_km, cxy)

    info = {k: getattr(a, k) for k in vars(a)}
    info.update({"features": feats, "n_cells": int(len(cells)), "n_robust": int(len(r)),
                 "median_spearman": float(np.median(rho)), "veg_profile": veg.round(4).to_dict(),
                 "runtime_min": round((time.time() - t0) / 60, 1)})
    (out / "run_info.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    log(f"done in {info['runtime_min']} min, results in {out}")


if __name__ == "__main__":
    main()
