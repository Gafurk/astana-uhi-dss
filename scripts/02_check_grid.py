"""
02_check_grid.py  -  checks that all GeoTIFFs exported from Google Earth Engine share one grid.

Usage:  python scripts/02_check_grid.py  TIF_FOLDER
Checks: size, resolution, origin, coordinate system and number of bands.
30 m files must be exactly 2000 x 2400; MODIS files: 1 km, 60 x 72, 4 bands per year
(TERRA_DAY, TERRA_NIGHT, AQUA_DAY, AQUA_NIGHT); the number of years is read from the file name.
"""
import re
import sys
from pathlib import Path
import rasterio

EXPECTED_30 = {  # name pattern -> number of bands
    "astana_L8_": 5, "astana_L9_": 5, "astana_L7_": 5,
    "astana_scene_": 4, "astana_S2_": 4, "astana_static_30m": 13, "astana_osm_30m": 5,
}
XMIN, YMAX = 642000.0, 5706000.0
SIZE_30 = (2000, 2400)  # 60 x 72 km window at 30 m (width x height)
SIZE_1K = (60, 72)      # the same window at 1 km
MODIS_PER_YEAR = 4


def main(folder):
    files = sorted(Path(folder).rglob("*.tif"))
    if not files:
        sys.exit(f"No .tif files in {folder}")
    bad = 0
    for f in files:
        with rasterio.open(f) as ds:
            t = ds.transform
            res = round(t.a, 3)
            info = f"{f.name:48s} {ds.width:5d}x{ds.height:<5d} res={res:>7} bands={ds.count:<3d} crs={ds.crs}"
            problems = []
            if ds.crs is None or ds.crs.to_epsg() != 32642:  # GEE may write the CRS as WKT
                problems.append("CRS is not EPSG:32642")
            if abs(t.c - XMIN) > 1e-6 or abs(t.f - YMAX) > 1e-6:
                problems.append(f"origin ({t.c}, {t.f}) instead of ({XMIN}, {YMAX})")
            if res == 30.0 and (ds.width, ds.height) != SIZE_30:
                problems.append(f"size {ds.width}x{ds.height}, expected {SIZE_30[0]}x{SIZE_30[1]} (60 x 72 km at 30 m)")
            for key, nb in EXPECTED_30.items():
                if f.name.startswith(key) and ds.count != nb:
                    problems.append(f"{ds.count} bands, expected {nb}")
            if f.name.startswith("astana_MODIS"):
                if res != 1000.0:
                    problems.append(f"MODIS: pixel {res} m, expected 1000 m")
                if (ds.width, ds.height) != SIZE_1K:
                    problems.append(f"MODIS: size {ds.width}x{ds.height}, expected {SIZE_1K[0]}x{SIZE_1K[1]}")
                m = re.search(r"_(\d{4})_(\d{4})", f.name)
                if m:
                    need = MODIS_PER_YEAR * (int(m.group(2)) - int(m.group(1)) + 1)
                    if ds.count != need:
                        problems.append(f"MODIS: {ds.count} bands, expected {need} (4 per year)")
                elif ds.count % MODIS_PER_YEAR:
                    problems.append(f"MODIS: {ds.count} bands, not a multiple of 4")
            print(("OK   " if not problems else "FAIL ") + info)
            for p in problems:
                print("       -> " + p)
            bad += bool(problems)
    print(f"\nFiles: {len(files)}, with problems: {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])