// =====================================================================================
// 04_sentinel2_composites.js   (Google Earth Engine Code Editor)
// Летние композиты Sentinel-2 L2A: индексы ДРУГОГО сенсора, чем температура (Landsat).
// Нужны, чтобы признаки не совпадали с целевой переменной по сцене (фидбэк про завышенный R2).
//
// Файл на выходе: astana_S2_<год>_JJA.tif, 4 канала на сетке 30 м:
//   1 S2_NDVI    (B8 - B4) / (B8 + B4)
//   2 S2_NDBI    (B11 - B8) / (B11 + B8)
//   3 S2_MNDWI   (B3 - B11) / (B3 + B11)   вода, хорошо отделяет пруды и Ишим
//   4 S2_N_OBS   среднее число ясных наблюдений (по 10 м внутри пикселя 30 м)
// Индексы считаются на 10 м и усредняются до 30 м (reduceResolution), а не берутся
// "из центра" пикселя.
//
// L2A в Казахстане надёжно доступна с 2019 года, поэтому годы 2019-2025.
// Скрипт тяжёлый: если задача падает с ошибкой памяти/времени, сократи окно или экспортируй
// по одному году (поменяй YEARS).
// =====================================================================================

// ============ ОБЩИЕ НАСТРОЙКИ (одинаковые во всех скриптах, не меняй) ============
var CRS = 'EPSG:32642';
var XMIN = 642000, XMAX = 702000, YMIN = 5634000, YMAX = 5706000;
var FOLDER = 'astana_uhi';
var AOI = ee.Geometry.Rectangle([XMIN, YMIN, XMAX, YMAX], CRS, false);
function grid(res) { return [res, 0, XMIN, 0, -res, YMAX]; }
// ==================================================================================

// Регион экспорта на 10 м внутри окна, чтобы файл был ровно по сетке без лишних строки и столбца
// (как в 01). Сетку задаёт crsTransform, начало координат остаётся (XMIN, YMAX).
var EXPORT_REGION = ee.Geometry.Rectangle([XMIN + 10, YMIN + 10, XMAX - 10, YMAX - 10], CRS, false);

var START_EXPORT = false;
var YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025];
var MAX_TILE_CLOUD = 20; // CLOUDY_PIXEL_PERCENTAGE

function prepS2(img) {
  var scl = img.select('SCL');
  // 4 растительность, 5 голая почва, 6 вода. Остальное (облака, тени, снег, cirrus) выбрасываем.
  var ok = scl.eq(4).or(scl.eq(5)).or(scl.eq(6));
  var b = img.select(['B3', 'B4', 'B8', 'B11']).multiply(0.0001);
  var b3 = b.select('B3'), b4 = b.select('B4'), b8 = b.select('B8'), b11 = b.select('B11');
  var ndvi = b8.subtract(b4).divide(b8.add(b4)).rename('S2_NDVI');
  var ndbi = b11.subtract(b8).divide(b11.add(b8)).rename('S2_NDBI');
  var mndwi = b3.subtract(b11).divide(b3.add(b11)).rename('S2_MNDWI');
  // Как в 01 и 03: индексы вне [-1, 1] маскируем (при неотрицательной отражательной способности
  // их быть не должно, маска здесь страховка)
  ndvi = ndvi.updateMask(ndvi.abs().lte(1));
  ndbi = ndbi.updateMask(ndbi.abs().lte(1));
  mndwi = mndwi.updateMask(mndwi.abs().lte(1));
  return ndvi.addBands(ndbi).addBands(mndwi).updateMask(ok);
}

YEARS.forEach(function (year) {
  var raw = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(AOI)
    .filterDate(year + '-06-01', year + '-09-01')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_TILE_CLOUD));
  var prepped = raw.map(prepS2);
  var nObs = prepped.select('S2_NDVI').count().rename('S2_N_OBS');
  var comp = prepped.median().addBands(nObs)
    .select(['S2_NDVI', 'S2_NDBI', 'S2_MNDWI', 'S2_N_OBS']).toFloat().clip(AOI);

  // 10 м -> среднее по 30 м
  var agg = comp.reproject({crs: CRS, scale: 10})
    .reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 64})
    .reproject({crs: CRS, crsTransform: grid(30)});

  var name = 'astana_S2_' + year + '_JJA';
  raw.size().evaluate(function (n) {
    print(name + ': снимков = ' + n);
    if (n === 0) { print('  -> пропущено'); return; }
    if (!START_EXPORT) { return; }
    Export.image.toDrive({
      image: agg,
      description: name,
      folder: FOLDER,
      fileNamePrefix: name,
      region: EXPORT_REGION,
      crs: CRS,
      crsTransform: grid(30),
      maxPixels: 1e9,
      fileFormat: 'GeoTIFF'
    });
  });
});