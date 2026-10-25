// =====================================================================================
// 07a_sample_points.js   (Google Earth Engine Code Editor)   ЗАПУСКАЕТСЯ ОДИН РАЗ
// Фиксированный набор пикселей для модели: одни и те же места во всех сценах и годах.
// Половина точек в застройке (доля WorldCover "built" в пикселе 30 м > 0.1), половина вне её.
// Точки стоят в центрах пикселей той же сетки 30 м, что и все растры (EPSG:32642, 2000 x 2400).
//
// Выход (два экспорта):
//   1) asset  ASSET_ID              -> его читает 07_scene_samples.js
//   2) CSV    astana_points.csv     -> для проверки / для Python (pixel_id, x, y, urban_stratum)
// pixel_id = row * 2000 + col (row сверху, col слева), как в build_dataset.py.
// =====================================================================================

// ============ ОБЩИЕ НАСТРОЙКИ (одинаковые во всех скриптах, не меняй) ============
var CRS = 'EPSG:32642';
var XMIN = 642000, XMAX = 702000, YMIN = 5634000, YMAX = 5706000;
var FOLDER = 'astana_uhi';
var AOI = ee.Geometry.Rectangle([XMIN, YMIN, XMAX, YMAX], CRS, false);
function grid(res) { return [res, 0, XMIN, 0, -res, YMAX]; }
// ==================================================================================

var START_EXPORT = false;
// Куда сохранить asset. Проверь корень во вкладке Assets (слева) и поправь, если он другой.
var ASSET_ID = 'projects/dataset-collection-511715/assets/astana_points_v1';
var N_URBAN = 50000, N_RURAL = 50000, SEED = 42;
var NCOL = 2000;   // ширина сетки 30 м

var P30 = ee.Projection(CRS).atScale(30);   // сетка выровнена: 642000 и 5706000 делятся на 30
var wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map');
var fracBuilt = wc.eq(50).reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 1024})
                  .reproject({crs: CRS, crsTransform: grid(30)});
var stratum = fracBuilt.gt(0.1).rename('urban_stratum').toInt().clip(AOI);

var raw = stratum.stratifiedSample({
  numPoints: 0, classBand: 'urban_stratum', region: AOI, projection: P30,
  classValues: [0, 1], classPoints: [N_RURAL, N_URBAN], seed: SEED, geometries: true
});

var points = raw.map(function (f) {
  var c = f.geometry().transform(CRS, 0.01).coordinates();
  var col = ee.Number(c.get(0)).subtract(XMIN).divide(30).floor();
  var row = ee.Number(YMAX).subtract(ee.Number(c.get(1))).divide(30).floor();
  var x = col.add(0.5).multiply(30).add(XMIN);
  var y = ee.Number(YMAX).subtract(row.add(0.5).multiply(30));
  return ee.Feature(ee.Geometry.Point([x, y], CRS), {
    pixel_id: row.multiply(NCOL).add(col), x: x, y: y, urban_stratum: f.get('urban_stratum')
  });
});

print('Точек всего (ожидается ' + (N_URBAN + N_RURAL) + '):', points.size());
print('Из них в застройке:', points.filter(ee.Filter.eq('urban_stratum', 1)).size());
print('Первые 5:', points.limit(5));
Map.setCenter(71.43, 51.13, 10);
Map.addLayer(points.filter(ee.Filter.eq('urban_stratum', 1)).limit(5000), {color: 'red'}, 'точки: застройка (5000)');
Map.addLayer(points.filter(ee.Filter.eq('urban_stratum', 0)).limit(5000), {color: 'blue'}, 'точки: вне застройки (5000)');

if (START_EXPORT) {
  Export.table.toAsset({collection: points, description: 'astana_points_v1_asset', assetId: ASSET_ID});
  Export.table.toDrive({collection: points, description: 'astana_points', folder: FOLDER,
                        fileFormat: 'CSV', selectors: ['pixel_id', 'x', 'y', 'urban_stratum']});
}
