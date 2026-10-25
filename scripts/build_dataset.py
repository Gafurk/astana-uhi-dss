"""
build_dataset.py  -  собирает таблицу для моделей (Parquet) по Methods 2.4.

Строка = точка (пиксель 30 м) x акквизиция Landsat. ЦЕЛЬ = LST этой акквизиции.

Запуск:
  python build_dataset.py ПАПКА_С_ДАННЫМИ  astana_dataset.parquet  [--block-km 4] [--modis-sat TERRA]

Что ищется в папке (и в подпапках):
  astana_scene_samples_<год>.csv   из 07_scene_samples.js  (обязательно, хотя бы один год)
  astana_static_30m.tif            из 06 (13 каналов, обязательно)
  astana_osm_30m.tif               из osm_to_rasters.py (5 каналов, необязательно)
  astana_S2_<год>_JJA.tif          из 04 (4 канала, необязательно; вторая проверка)
  astana_modis_daily_city.csv      из 08 (необязательно; пометка нерепрезентативных сцен)
Композиты astana_L8/L9_<год>_JJA.tif из 01 здесь НЕ используются: они для карт, UHII и трендов.

Конфигурации предикторов (feature_groups.json), все по тексту Methods:
  main        структурные + LOO_NDVI, LOO_NDBI, LOO_ALBEDO   (композит без целевой сцены; выбор модели)
  s2_check    структурные + S2_NDVI, S2_NDBI + LOO_ALBEDO     (NDVI/NDBI из Sentinel-2 того же сезона)
  same_scene  структурные + SS_NDVI, SS_NDBI, SS_ALBEDO       (та же сцена; только оценка завышения R2)
Структурные (названы в Methods): ELEV, POP_DENS, DIST_WATER_M, DIST_GREEN_M.
  LCZ добавится колонкой, когда будет своя карта LCZ.
Остальные колонки остаются в таблице, но в модель не идут:
  not_in_methods      SLOPE_DEG, WC_MODE, FRAC_*, BUILT_H, FRAC_BLDG, DIST_ROAD_M, S2_MNDWI
  vulnerability_only  DIST_HEALTH_M
  reference_not_feature LCZ_REF (проверка меток LCZ)
Пометка сцен по MODIS: modis_dev_sd = (LST MODIS над городом в день сцены - среднее за июнь-август
  этого года) / SD; scene_flag_modis = |modis_dev_sd| > 1. Если в день сцены MODIS нет, значения пустые.
Гармонизации L8/L9 здесь НЕТ: колонка sensor различает их.
Если один и тот же файл лежит в двух папках, скрипт останавливается и показывает пути.
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
            sys.exit(f"{path.name}: каналов {ds.count}, ожидалось {len(names)}. Запусти check_grid.py")
        if (ds.width, ds.height) != (NCOL, NROW):
            sys.exit(f"{path.name}: размер {ds.width}x{ds.height}, ожидалось {NCOL}x{NROW}. Запусти check_grid.py")
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
                    help="какой спутник MODIS брать для пометки сцен")
    a = ap.parse_args()
    folder = Path(a.folder)

    def no_duplicates(paths):
        seen = {}
        for p in paths:
            seen.setdefault(p.name, []).append(str(p))
        dup = {k: v for k, v in seen.items() if len(v) > 1}
        if dup:
            sys.exit(f"Одинаковые файлы в разных папках, оставь по одному: {dup}")
        return paths

    def find_one(name):
        hits = sorted(folder.rglob(name))
        if len(hits) > 1:
            sys.exit(f"{name} найден {len(hits)} раз: {[str(h) for h in hits]}. Оставь один.")
        return hits[0] if hits else None

    # --- таблицы по сценам из 07 ---
    csvs = no_duplicates(sorted(folder.rglob("astana_scene_samples_*.csv")))
    if not csvs:
        sys.exit("Не найдено ни одного astana_scene_samples_<год>.csv (скрипт 07)")
    parts = []
    for f in csvs:
        d = pd.read_csv(f)
        miss = [c for c in SAMPLE_COLS if c not in d.columns]
        if miss:
            sys.exit(f"{f.name}: нет колонок {miss}")
        d = d[SAMPLE_COLS]
        num = [c for c in SAMPLE_COLS if c not in ("acq_key", "sensor", "date", "time_utc")]
        d[num] = d[num].apply(pd.to_numeric, errors="coerce").replace(NODATA, np.nan)
        parts.append(d)
        print(f"{f.name}: строк {len(d)}, акквизиций {d['acq_key'].nunique()}")
    df = pd.concat(parts, ignore_index=True)
    dup = df.duplicated(["pixel_id", "acq_key"]).sum()
    if dup:
        sys.exit(f"В таблицах {dup} повторов (pixel_id, acq_key): одна акквизиция выгружена дважды?")
    df = df[np.isfinite(df["LST_C"])]
    df["pixel_id"] = df["pixel_id"].astype("int64")
    df["year"] = df["date"].str[:4].astype(int)
    df["urban_stratum"] = df["urban_stratum"].astype("int8")

    pid = df["pixel_id"].to_numpy()
    rows, cols = pid // NCOL, pid % NCOL
    if (rows >= NROW).any() or (rows < 0).any():
        sys.exit("pixel_id вне сетки 2000 x 2400: точки из другого окна?")
    b = a.block_km * 1000
    df["block_id"] = (np.floor((df["x"] - 642000) / b).astype(int) * 1000
                      + np.floor((5706000 - df["y"]) / b).astype(int))

    # --- статические слои и OSM: по pixel_id ---
    static_path = find_one("astana_static_30m.tif")
    if static_path is None:
        sys.exit("Нет astana_static_30m.tif (скрипт 06)")
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
        print("! astana_osm_30m.tif не найден: OSM-признаков не будет")

    # --- Sentinel-2 по годам ---
    for n in S2_BANDS:
        df[n] = np.nan
    for f in no_duplicates(sorted(folder.rglob("astana_S2_*_JJA.tif"))):
        year = int(re.search(r"_S2_(\d{4})_", f.name).group(1))
        arr = read_all(f, S2_BANDS)
        m = (df["year"] == year).to_numpy()
        for i, n in enumerate(S2_BANDS):
            df.loc[m, n] = arr[i][rows[m], cols[m]]
        del arr
        print(f"{f.name}: подставлен в {m.sum()} строк")

    # --- пометка сцен по MODIS ---
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
        print(f"MODIS ({a.modis_sat}): акквизиций с пометкой > 1 SD: "
              f"{int((acq['dev'].abs() > 1).sum())} из {len(acq)}, без MODIS в этот день: {int(acq['dev'].isna().sum())}")
    else:
        print("! astana_modis_daily_city.csv не найден: пометки сцен по MODIS не будет")

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
    print(f"\nГотово: {a.out}  строк {len(df)}, колонок {df.shape[1]}, точек {df['pixel_id'].nunique()}")
    print(f"Группы признаков: {gpath}")
    print(df.groupby(["year", "sensor"])["acq_key"].nunique().rename("акквизиций").to_string())


if __name__ == "__main__":
    main()