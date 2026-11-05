"""
03_build_dataset.py  -  assembles the modelling table (Parquet) described in the Methods of the article.

One row = one sample point (30 m pixel) x one Landsat acquisition. TARGET = LST of that acquisition.

Usage:
  python scripts/03_build_dataset.py DATA_FOLDER  astana_dataset.parquet  [--block-km 4] [--modis-sat TERRA]

Files searched for in the folder (and its subfolders):
  astana_scene_samples_<year>.csv  from gee/06_scene_samples.js  (required, at least one year)
  astana_static_30m.tif            from gee/04_static_layers.js (13 bands, required)
  astana_osm_30m.tif               from scripts/01_osm_to_rasters.py (5 bands, optional)
  astana_S2_<year>_JJA.tif         from gee/02_sentinel2_composites.js (4 bands, optional; second check)
  astana_modis_daily_city.csv      from gee/07_modis_daily_city.js (optional; flags unrepresentative scenes)
The composites astana_L8/L9_<year>_JJA.tif from gee/01 are NOT used here: they serve maps, UHII and trends.

Predictor configurations (feature_groups.json), all as described in the Methods:
  main        structural + LOO_NDVI, LOO_NDBI, LOO_ALBEDO   (composite without the target scene; model selection)
  s2_check    structural + S2_NDVI, S2_NDBI + LOO_ALBEDO     (NDVI/NDBI from Sentinel-2 of the same season)
  same_scene  structural + SS_NDVI, SS_NDBI, SS_ALBEDO       (same scene; only for estimating R2 inflation)
Structural predictors (named in the Methods): ELEV, POP_DENS, DIST_WATER_M, DIST_GREEN_M.
  LCZ will be added as a column once the local LCZ map exists.
The remaining columns stay in the table but are not used by the models:
  not_in_methods      SLOPE_DEG, WC_MODE, FRAC_*, BUILT_H, FRAC_BLDG, DIST_ROAD_M, S2_MNDWI
  vulnerability_only  DIST_HEALTH_M
  reference_not_feature LCZ_REF (check of the LCZ labels)
MODIS scene flag: modis_dev_sd = (MODIS LST over the city on the scene date - June-August mean
  of that year) / SD; scene_flag_modis = |modis_dev_sd| > 1. Empty if MODIS has no value on that date.
No L8/L9 harmonisation is done here: the column sensor distinguishes them.
If the same file is found in two folders, the script stops and prints both paths.
"""
import argparse
import json
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio

NODATA = -9999.0
NCOL, NROW = 2000, 2400
SAMPLE_COLS = ["pixel_id", "x", "y", "urban_stratum", "acq_key", "sensor", "date", "time_utc",
               "cloud_frac_aoi", "cover_frac_aoi", "LST_C", "SS_NDVI", "SS_NDBI", "SS_ALBEDO",
               "LOO_NDVI", "LOO_NDBI", "LOO_ALBEDO", "LOO_N", "LOO_WIDE"]
S2_BANDS = ["S2_NDVI", "S2_NDBI", "S2_MNDWI", "S2_N_OBS"]
STATIC_BANDS = ["ELEV", "SLOPE_DEG", "WC_MODE", "FRAC_TREE", "FRAC_SHRUB", "FRAC_GRASS", "FRAC_CROP",
                "FRAC_BUILT", "FRAC_BARE", "FRAC_WATER", "POP_DENS", "BUILT_H", "LCZ_REF"]
OSM_BANDS = ["DIST_WATER_M", "DIST_GREEN_M", "DIST_HEALTH_M", "DIST_ROAD_M", "FRAC_BLDG"]

STRUCTURAL = ["ELEV", "POP_DENS", "DIST_WATER_M", "DIST_GREEN_M"]
LOO = ["LOO_NDVI", "LOO_NDBI", "LOO_ALBEDO"]
SAME = ["SS_NDVI", "SS_NDBI", "SS_ALBEDO"]
NOT_IN_METHODS = (["SLOPE_DEG", "WC_MODE", "FRAC_TREE", "FRAC_SHRUB", "FRAC_GRASS", "FRAC_CROP",
                   "FRAC_BUILT", "FRAC_BARE", "FRAC_WATER", "BUILT_H", "FRAC_BLDG", "DIST_ROAD_M",
                   "S2_MNDWI"])


def read_all(path, names):
    with rasterio.open(path) as ds:
        if ds.count != len(names):
            sys.exit(f"{path.name}: {ds.count} bands, expected {len(names)}. Run 02_check_grid.py")
        if (ds.width, ds.height) != (NCOL, NROW):
            sys.exit(f"{path.name}: size {ds.width}x{ds.height}, expected {NCOL}x{NROW}. Run 02_check_grid.py")
        arr = ds.read().astype("float32")
        nd = ds.nodata
        if nd is not None and not np.isnan(nd):
            arr[arr == nd] = np.nan
        return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("out")
    ap.add_argument("--block-km", type=float, default=4.0)
    ap.add_argument("--modis-sat", default="TERRA", choices=["TERRA", "AQUA"],
                    help="which MODIS satellite to use for flagging scenes")
    a = ap.parse_args()
    folder = Path(a.folder)

    def no_duplicates(paths):
        seen = {}
        for p in paths:
            seen.setdefault(p.name, []).append(str(p))
        dup = {k: v for k, v in seen.items() if len(v) > 1}
        if dup:
            sys.exit(f"Identical files in different folders, keep one of each: {dup}")
        return paths

    def find_one(name):
        hits = sorted(folder.rglob(name))
        if len(hits) > 1:
            sys.exit(f"{name} found {len(hits)} times: {[str(h) for h in hits]}. Keep one.")
        return hits[0] if hits else None

    # --- per-scene sample tables from gee/06 ---
    csvs = no_duplicates(sorted(folder.rglob("astana_scene_samples_*.csv")))
    if not csvs:
        sys.exit("No astana_scene_samples_<year>.csv found (gee/06_scene_samples.js)")
    parts = []
    for f in csvs:
        d = pd.read_csv(f)
        miss = [c for c in SAMPLE_COLS if c not in d.columns]
        if miss:
            sys.exit(f"{f.name}: missing columns {miss}")
        d = d[SAMPLE_COLS]
        num = [c for c in SAMPLE_COLS if c not in ("acq_key", "sensor", "date", "time_utc")]
        d[num] = d[num].apply(pd.to_numeric, errors="coerce").replace(NODATA, np.nan)
        parts.append(d)
        print(f"{f.name}: rows {len(d)}, acquisitions {d['acq_key'].nunique()}")
    df = pd.concat(parts, ignore_index=True)
    dup = df.duplicated(["pixel_id", "acq_key"]).sum()
    if dup:
        sys.exit(f"{dup} duplicate (pixel_id, acq_key) pairs in the tables: was one acquisition exported twice?")
    df = df[np.isfinite(df["LST_C"])]
    df["pixel_id"] = df["pixel_id"].astype("int64")
    df["year"] = df["date"].str[:4].astype(int)
    df["urban_stratum"] = df["urban_stratum"].astype("int8")

    pid = df["pixel_id"].to_numpy()
    rows, cols = pid // NCOL, pid % NCOL
    if (rows >= NROW).any() or (rows < 0).any():
        sys.exit("pixel_id outside the 2000 x 2400 grid: points from another window?")
    b = a.block_km * 1000
    df["block_id"] = (np.floor((df["x"] - 642000) / b).astype(int) * 1000
                      + np.floor((5706000 - df["y"]) / b).astype(int))

    # --- static and OSM layers, read by pixel_id ---
    static_path = find_one("astana_static_30m.tif")
    if static_path is None:
        sys.exit("astana_static_30m.tif not found (gee/04_static_layers.js)")
    arr = read_all(static_path, STATIC_BANDS)
    for i, n in enumerate(STATIC_BANDS):
        df[n] = arr[i][rows, cols]
    del arr
    osm_path = find_one("astana_osm_30m.tif")
    if osm_path is not None:
        arr = read_all(osm_path, OSM_BANDS)
        for i, n in enumerate(OSM_BANDS):
            df[n] = arr[i][rows, cols]
        del arr
    else:
        print("! astana_osm_30m.tif not found: no OSM predictors")

    # --- Sentinel-2 by year ---
    for n in S2_BANDS:
        df[n] = np.nan
    for f in no_duplicates(sorted(folder.rglob("astana_S2_*_JJA.tif"))):
        year = int(re.search(r"_S2_(\d{4})_", f.name).group(1))
        arr = read_all(f, S2_BANDS)
        m = (df["year"] == year).to_numpy()
        for i, n in enumerate(S2_BANDS):
            df.loc[m, n] = arr[i][rows[m], cols[m]]
        del arr
        print(f"{f.name}: filled into {m.sum()} rows")

    # --- MODIS scene flags ---
    df["modis_dev_sd"] = np.nan
    df["scene_flag_modis"] = pd.array([pd.NA] * len(df), dtype="boolean")
    mpath = find_one("astana_modis_daily_city.csv")
    if mpath is not None:
        md = pd.read_csv(mpath)
        md = md[(md["sat"] == a.modis_sat) & np.isfinite(pd.to_numeric(md["lst_city_c"], errors="coerce"))].copy()
        md["lst_city_c"] = md["lst_city_c"].astype(float)
        md["year"] = md["date"].str[:4].astype(int)
        stats = md.groupby("year")["lst_city_c"].agg(["mean", "std"])
        day = md.set_index("date")["lst_city_c"]
        acq = df[["acq_key", "date", "year"]].drop_duplicates("acq_key").copy()
        acq["lst"] = acq["date"].map(day)
        acq = acq.join(stats, on="year")
        acq["dev"] = (acq["lst"] - acq["mean"]) / acq["std"]
        dev = acq.set_index("acq_key")["dev"]
        df["modis_dev_sd"] = df["acq_key"].map(dev)
        df["scene_flag_modis"] = (df["modis_dev_sd"].abs() > 1).astype("boolean").mask(df["modis_dev_sd"].isna())
        print(f"MODIS ({a.modis_sat}): acquisitions flagged > 1 SD: "
              f"{int((acq['dev'].abs() > 1).sum())} of {len(acq)}, without MODIS on that date: {int(acq['dev'].isna().sum())}")
    else:
        print("! astana_modis_daily_city.csv not found: no MODIS scene flags")

    lead = ["pixel_id", "x", "y", "block_id", "urban_stratum", "acq_key", "sensor", "date", "year", "LST_C"]
    df = df[lead + [c for c in df.columns if c not in lead]]
    df.to_parquet(a.out, index=False)

    have = lambda cols: [c for c in cols if c in df.columns]
    groups = {
        "target": "LST_C",
        "structural": have(STRUCTURAL),
        "configs": {
            "main": have(STRUCTURAL) + LOO,
            "s2_check": have(STRUCTURAL) + ["S2_NDVI", "S2_NDBI", "LOO_ALBEDO"],
            "same_scene": have(STRUCTURAL) + SAME,
        },
        "not_in_methods": have(NOT_IN_METHODS),
        "vulnerability_only": have(["DIST_HEALTH_M"]),
        "reference_not_feature": ["LCZ_REF"],
        "categorical": [],
        "cv_keys": ["block_id", "year", "sensor", "acq_key"],
        "quality": ["LOO_N", "LOO_WIDE", "cloud_frac_aoi", "cover_frac_aoi", "S2_N_OBS",
                    "modis_dev_sd", "scene_flag_modis"],
    }
    gpath = Path(a.out).with_name("feature_groups.json")
    gpath.write_text(json.dumps(groups, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDone: {a.out}  rows {len(df)}, columns {df.shape[1]}, points {df['pixel_id'].nunique()}")
    print(f"Feature groups: {gpath}")
    print(df.groupby(["year", "sensor"])["acq_key"].nunique().rename("acquisitions").to_string())


if __name__ == "__main__":
    main()