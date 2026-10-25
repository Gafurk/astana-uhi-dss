"""
check_grid.py  -  проверка, что все GeoTIFF из Google Drive лежат на одной сетке.

Запуск:  python check_grid.py  ПАПКА_С_TIF
Что проверяет: размер, разрешение, привязку (origin), систему координат и число каналов.
Файлы 30 м должны быть ровно 2000 x 2400; файлы MODIS: 1 км, 60 x 72, по 4 канала на год
(TERRA_DAY, TERRA_NIGHT, AQUA_DAY, AQUA_NIGHT), число лет берётся из имени файла.
"""
import re
import sys
from pathlib import Path
import rasterio

EXPECTED_30 = {  # маска имени -> число каналов
    "astana_L8_": 5, "astana_L9_": 5, "astana_L7_": 5,
    "astana_scene_": 4, "astana_S2_": 4, "astana_static_30m": 13, "astana_osm_30m": 5,
}
XMIN, YMAX = 642000.0, 5706000.0
SIZE_30 = (2000, 2400)  # окно 60 x 72 км при пикселе 30 м (ширина x высота)
SIZE_1K = (60, 72)      # то же окно при 1 км
MODIS_PER_YEAR = 4


def main(folder):
    files = sorted(Path(folder).rglob("*.tif"))
    if not files:
        sys.exit(f"В {folder} нет .tif файлов")
    bad = 0
    for f in files:
        with rasterio.open(f) as ds:
            t = ds.transform
            res = round(t.a, 3)
            info = f"{f.name:48s} {ds.width:5d}x{ds.height:<5d} res={res:>7} bands={ds.count:<3d} crs={ds.crs}"
            problems = []
            if ds.crs is None or ds.crs.to_epsg() != 32642:  # GEE может записать CRS как WKT
                problems.append("CRS не EPSG:32642")
            if abs(t.c - XMIN) > 1e-6 or abs(t.f - YMAX) > 1e-6:
                problems.append(f"origin ({t.c}, {t.f}) вместо ({XMIN}, {YMAX})")
            if res == 30.0 and (ds.width, ds.height) != SIZE_30:
                problems.append(f"размер {ds.width}x{ds.height}, ожидалось {SIZE_30[0]}x{SIZE_30[1]} (60 x 72 км при 30 м)")
            for key, nb in EXPECTED_30.items():
                if f.name.startswith(key) and ds.count != nb:
                    problems.append(f"каналов {ds.count}, ожидалось {nb}")
            if f.name.startswith("astana_MODIS"):
                if res != 1000.0:
                    problems.append(f"MODIS: пиксель {res} м, ожидался 1000 м")
                if (ds.width, ds.height) != SIZE_1K:
                    problems.append(f"MODIS: размер {ds.width}x{ds.height}, ожидалось {SIZE_1K[0]}x{SIZE_1K[1]}")
                m = re.search(r"_(\d{4})_(\d{4})", f.name)
                if m:
                    need = MODIS_PER_YEAR * (int(m.group(2)) - int(m.group(1)) + 1)
                    if ds.count != need:
                        problems.append(f"MODIS: каналов {ds.count}, ожидалось {need} (4 на год)")
                elif ds.count % MODIS_PER_YEAR:
                    problems.append(f"MODIS: каналов {ds.count}, не кратно 4")
            print(("OK   " if not problems else "FAIL ") + info)
            for p in problems:
                print("       -> " + p)
            bad += bool(problems)
    print(f"\nФайлов: {len(files)}, с проблемами: {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])