"""
osm_to_rasters.py  -  из выгрузки OpenStreetMap (Geofabrik GeoPackage) делает растр на той же
сетке 30 м, что и слои из Google Earth Engine.

Запуск:
  python osm_to_rasters.py --list  kazakhstan-latest-free.gpkg        # показать слои файла
  python osm_to_rasters.py kazakhstan-latest-free.gpkg astana_osm_30m.tif

Каналы результата (порядок важен для build_dataset.py):
  1 DIST_WATER_M    расстояние до воды (полигоны воды), м
  2 DIST_GREEN_M    расстояние до зелёной зоны (парки, лес, луг, ...), м
  3 DIST_HEALTH_M   расстояние до ближайшей медицинской организации (hospital, clinic, doctors), м
  4 DIST_ROAD_M     расстояние до крупной дороги (motorway/trunk/primary/secondary), м
  5 FRAC_BLDG       доля площади пикселя 30 м, занятая контурами зданий (0..1)

Важно: полнота зданий и учреждений в OSM для Астаны не проверена, посмотри результат в QGIS.
Имена слоёв и fclass Geofabrik: если у тебя другие, поправь словарь LAYERS ниже.
"""
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt
from pyproj import Transformer

XMIN, XMAX, YMIN, YMAX = 642000, 702000, 5634000, 5706000
RES = 30
W, H = (XMAX - XMIN) // RES, (YMAX - YMIN) // RES
TRANSFORM = from_origin(XMIN, YMAX, RES, RES)
CRS = "EPSG:32642"

# (слой, колонка, допустимые значения или None = все)
LAYERS = {
    "water":  [("gis_osm_water_a_free", None, None)],
    "green":  [("gis_osm_landuse_a_free", "fclass",
                {"park", "forest", "grass", "meadow", "nature_reserve", "recreation_ground", "orchard", "vineyard"})],
    "health": [("gis_osm_pois_free", "fclass", {"hospital", "clinic", "doctors"}),
               ("gis_osm_pois_a_free", "fclass", {"hospital", "clinic", "doctors"})],
    "roads":  [("gis_osm_roads_free", "fclass",
                {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
                 "secondary", "secondary_link"})],
    "bldg":   [("gis_osm_buildings_a_free", None, None)],
}
BANDS = ["DIST_WATER_M", "DIST_GREEN_M", "DIST_HEALTH_M", "DIST_ROAD_M", "FRAC_BLDG"]


def window_wgs84():
    t = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    xs, ys = zip(*[t.transform(x, y) for x, y in [(XMIN, YMIN), (XMAX, YMIN), (XMIN, YMAX), (XMAX, YMAX)]])
    return (min(xs), min(ys), max(xs), max(ys))


def load(path, spec, bbox):
    """Читает слои из спецификации, фильтрует по fclass, переводит в EPSG:32642."""
    parts = []
    for layer, col, vals in spec:
        try:
            g = gpd.read_file(path, layer=layer, bbox=bbox)
        except Exception as e:  # слоя нет в файле
            print(f"  ! слой {layer} не прочитан: {str(e).splitlines()[0]}")
            continue
        if col and vals is not None:
            if col not in g.columns:
                print(f"  ! в слое {layer} нет колонки {col}")
                continue
            g = g[g[col].isin(vals)]
        if len(g):
            parts.append(g.to_crs(CRS)[["geometry"]])
    if not parts:
        return None
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=CRS)


def distance_band(gdf):
    if gdf is None or gdf.empty:
        return np.full((H, W), np.nan, dtype="float32")
    mask = rasterize([(g, 1) for g in gdf.geometry if g is not None and not g.is_empty],
                     out_shape=(H, W), transform=TRANSFORM, fill=0, dtype="uint8", all_touched=True)
    if mask.sum() == 0:
        return np.full((H, W), np.nan, dtype="float32")
    return (distance_transform_edt(mask == 0) * RES).astype("float32")


def fraction_band(gdf, k=3):
    if gdf is None or gdf.empty:
        return np.full((H, W), np.nan, dtype="float32")
    fine = from_origin(XMIN, YMAX, RES / k, RES / k)
    m = rasterize([(g, 1) for g in gdf.geometry if g is not None and not g.is_empty],
                  out_shape=(H * k, W * k), transform=fine, fill=0, dtype="uint8")
    return m.reshape(H, k, W, k).mean(axis=(1, 3)).astype("float32")


def main(path, out):
    bbox = window_wgs84()
    data = {}
    for key in LAYERS:
        print(f"читаю {key} ...")
        data[key] = load(path, LAYERS[key], bbox)
        print(f"  объектов: {0 if data[key] is None else len(data[key])}")
    bands = [distance_band(data["water"]), distance_band(data["green"]),
             distance_band(data["health"]), distance_band(data["roads"]), fraction_band(data["bldg"])]
    with rasterio.open(out, "w", driver="GTiff", height=H, width=W, count=len(bands), dtype="float32",
                       crs=CRS, transform=TRANSFORM, nodata=np.nan, compress="deflate") as dst:
        for i, (b, name) in enumerate(zip(bands, BANDS), 1):
            dst.write(b, i)
            dst.set_band_description(i, name)
            print(f"канал {i} {name}: min={np.nanmin(b) if np.isfinite(b).any() else 'нет данных'}")
    print("готово:", out)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--list":
        import pyogrio
        for name, geom in pyogrio.list_layers(sys.argv[2]):
            print(f"{name:40s} {geom}")
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        sys.exit(__doc__)
