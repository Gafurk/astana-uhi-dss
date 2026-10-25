// =====================================================================================
// 09_case_sites.js   (Google Earth Engine Code Editor)
// Кейс научрука: изменение температуры поверхности на озеленённых участках по сравнению с
// контрольными участками. Для каждого ясного снимка Landsat (те же 88 акквизиций, что в 07)
// считает по каждому участку среднюю и медианную LST, средний NDVI и число ясных пикселей 30 м.
// Анализ (разность "участок минус контроль" по каждому снимку, до/после, тренд) делается в Python.
//
// Полигоны рисуются на карте этого скрипта (Geometry Imports) и называются:
//   botanic, botanic_ctrl, greenline, greenline_ctrl   (можно добавить *_ctrl2)
// Контроль: участок рядом, примерно того же размера, где с 2013 года ничего не менялось.
//
// Отступ от края (INNER_BUFFER_M) убирает смешанные пиксели на границе. Для узких участков
// (GreenLine около 55 м шириной) отступ не делается: иначе от полигона ничего не останется.
// Ширина GreenLine меньше теплового пикселя Landsat (около 100 м), поэтому LST там смешанная;
// для сравнения добавлена зона greenline_zone = бульвар плюс 100 м с каждой стороны.
//
// Выход: astana_case_sites.csv
//   acq_key, sensor, date, year, site, lst_mean, lst_median, ndvi_mean, n_valid, n_total
// =====================================================================================

// ============ ОБЩИЕ НАСТРОЙКИ (одинаковые во всех скриптах, не меняй) ============
var CRS = 'EPSG:32642';
var XMIN = 642000, XMAX = 702000, YMIN = 5634000, YMAX = 5706000;
var FOLDER = 'astana_uhi';
var AOI = ee.Geometry.Rectangle([XMIN, YMIN, XMAX, YMAX], CRS, false);
function grid(res) { return [res, 0, XMIN, 0, -res, YMAX]; }
// ==================================================================================

var START_EXPORT = false;
var ACQ_TABLE_ID = 'projects/dataset-collection-511715/assets/astana_acq_list_v1';  // из 07, шаг 1
var MAX_CLOUD_AOI = 0.10;
var INNER_BUFFER_M = 60;

// Участки: имя, геометрия из Imports, отступ внутрь (м). Закомментируй строку, если участка нет.
// Участок, который ещё не нарисован, просто пропускается (можно запускать скрипт и до того,
// как нарисованы контроли, например чтобы посмотреть снимки прошлых лет).
// Объекты заданы координатами прямо в коде (нарисованы 25.10.2026), чтобы не зависеть от Imports.
var BOTANIC = ee.Geometry.Polygon([[[71.41143054465597, 51.11127944560988],
                                    [71.40851230124777, 51.10306119901771],
                                    [71.42168731192892, 51.10109399211938],
                                    [71.4246913860256, 51.10939341985413]]]);
var GREENLINE = ee.Geometry.Polygon([[[71.397636885105, 51.13375237309245],
                                      [71.36965608066164, 51.13800704933611],
                                      [71.36957024997317, 51.1377108503962],
                                      [71.39750813907229, 51.133240708402084]]]);
var SITES = [];
function addSite(name, geomFn, inner) {
  try { SITES.push({name: name, geom: geomFn(), inner: inner}); }
  catch (e) { print('участок ' + name + ' не нарисован, пропущен'); }
}
addSite('botanic',        function () { return BOTANIC; },               INNER_BUFFER_M);
addSite('greenline',      function () { return GREENLINE; },             0);
addSite('greenline_zone', function () { return GREENLINE.buffer(100); }, 0);
// контроли, нарисованные вручную (необязательно; основные контроли строятся автоматически ниже)
addSite('botanic_ctrl',   function () { return botanic_ctrl; },          INNER_BUFFER_M);
addSite('greenline_ctrl', function () { return greenline_ctrl; },        0);

// --- Контроли по правилу (не рисуются руками) ---
// В кольце вокруг объекта берутся пиксели, где летний NDVI почти не изменился между 2013-2014 и
// 2024-2025 (|dNDVI| < DNDVI_MAX), то есть участок не застраивали и не озеленяли.
//   botanic_ctrl_auto   : кольцо 1-4 км вокруг сада; открытая земля (не застройка и не вода по
//                         WorldCover 2021) с NDVI 2013-2014 < 0.5, как степь на месте сада до посадки.
//   greenline_ctrl_auto : кольцо 0.5-3 км вокруг бульвара; застройка по WorldCover 2021, которая уже
//                         в 2013-2014 была без растительности (NDVI < 0.3), то есть старый квартал.
// Пороги можно менять; они пишутся в Methods как правило выбора контроля.
var DNDVI_MAX = 0.05;
function summerNdvi(y0, y1) {
  return ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(AOI)
    .filterDate(y0 + '-06-01', y1 + '-09-01').filter(ee.Filter.calendarRange(6, 8, 'month'))
    .filter(ee.Filter.lt('CLOUD_COVER', 20))
    .map(function (img) {
      var sr = img.select(['SR_B4', 'SR_B5']).multiply(0.0000275).add(-0.2);
      return sr.normalizedDifference(['SR_B5', 'SR_B4'])
               .updateMask(img.select('QA_PIXEL').bitwiseAnd(0x3E).eq(0));
    }).median();
}
var NDVI_EARLY = summerNdvi(2013, 2014);
var NDVI_LATE = summerNdvi(2024, 2025);
var STABLE = NDVI_LATE.subtract(NDVI_EARLY).abs().lt(DNDVI_MAX);
var WC = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map');
var AUTO = [];
if (true) {
  AUTO.push({name: 'botanic_ctrl_auto',
             ring: BOTANIC.buffer(4000).difference(BOTANIC.buffer(1000)),
             mask: STABLE.and(WC.neq(50)).and(WC.neq(80)).and(NDVI_EARLY.lt(0.5))});
}
if (true) {
  AUTO.push({name: 'greenline_ctrl_auto',
             ring: GREENLINE.buffer(3000).difference(GREENLINE.buffer(500)),
             mask: STABLE.and(WC.eq(50)).and(NDVI_EARLY.lt(0.3))});
}

var sites = ee.FeatureCollection(SITES.map(function (s) {
  var g = s.inner > 0 ? s.geom.buffer(-s.inner) : s.geom;
  return ee.Feature(g, {site: s.name});
}));

// --- те же снимки и та же подготовка, что в 07 ---
function withSensor(id, sensor) {
  return ee.ImageCollection(id).filterBounds(AOI)
    .filter(ee.Filter.calendarRange(6, 8, 'month'))
    .filter(ee.Filter.calendarRange(2013, 2025, 'year'))
    .map(function (img) {
      return img.set('sensor', sensor,
                     'acq_key', ee.String(sensor).cat('_').cat(img.get('DATE_ACQUIRED')));
    });
}
var RAW = withSensor('LANDSAT/LC08/C02/T1_L2', 'L8').merge(withSensor('LANDSAT/LC09/C02/T1_L2', 'L9'));

function prep(img) {
  var clear = img.select('QA_PIXEL').bitwiseAnd(0x3E).eq(0);
  var sr = img.select(['SR_B4', 'SR_B5']).multiply(0.0000275).add(-0.2);
  var r = sr.select('SR_B4'), n = sr.select('SR_B5');
  var lst = img.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15).rename('LST_C');
  var ndvi = n.subtract(r).divide(n.add(r)).rename('NDVI');
  ndvi = ndvi.updateMask(ndvi.abs().lte(1));
  return lst.addBands(ndvi).updateMask(clear.and(lst.mask()));
}

var KEPT = ee.FeatureCollection(ACQ_TABLE_ID).filter(ee.Filter.lt('cloud_frac_aoi', MAX_CLOUD_AOI));

var reducer = ee.Reducer.mean().combine(ee.Reducer.median(), null, true)
                               .combine(ee.Reducer.count(), null, true);
var ones = ee.Image.constant(1).rename('px');

var table = KEPT.map(function (f) {
  var key = f.get('acq_key');
  var img = RAW.filter(ee.Filter.eq('acq_key', key)).map(prep).mosaic();
  var stats = img.addBands(ones.updateMask(1)).reduceRegions({
    collection: sites, reducer: reducer, crs: CRS, crsTransform: grid(30), tileScale: 2
  });
  var poly = stats.map(function (s) {
    return ee.Feature(null, {
      acq_key: key, sensor: f.get('sensor'), date: f.get('date'), year: f.get('year'),
      site: s.get('site'),
      lst_mean: s.get('LST_C_mean'), lst_median: s.get('LST_C_median'),
      ndvi_mean: s.get('NDVI_mean'), n_valid: s.get('LST_C_count'), n_total: s.get('px_count')
    });
  });
  var auto = AUTO.map(function (c) {
    var r = img.updateMask(c.mask).addBands(ones.updateMask(c.mask)).reduceRegion({
      reducer: reducer, geometry: c.ring, crs: CRS, crsTransform: grid(30), maxPixels: 1e9, tileScale: 2
    });
    return ee.Feature(null, {
      acq_key: key, sensor: f.get('sensor'), date: f.get('date'), year: f.get('year'),
      site: c.name,
      lst_mean: r.get('LST_C_mean'), lst_median: r.get('LST_C_median'),
      ndvi_mean: r.get('NDVI_mean'), n_valid: r.get('LST_C_count'), n_total: r.get('px_count')
    });
  });
  return poly.merge(ee.FeatureCollection(auto));
}).flatten();

// --- проверка в Console и на карте ---
sites.evaluate(function (fc) {
  fc.features.forEach(function (ft) { print('участок', ft.properties.site); });
});
print('Площадь участков после отступа, га:', sites.map(function (s) {
  return s.set('ha', s.geometry().area(1).divide(1e4));
}).aggregate_array('ha'));
print('Снимков (облачность < 10% над окном):', KEPT.size());
AUTO.forEach(function (c) {
  print(c.name + ': площадь контроля, га', ones.updateMask(c.mask).multiply(ee.Image.pixelArea()).divide(1e4)
    .reduceRegion({reducer: ee.Reducer.sum(), geometry: c.ring, crs: CRS, crsTransform: grid(30),
                   maxPixels: 1e9}).get('px'));
  Map.addLayer(ones.updateMask(c.mask).clip(c.ring),
               {palette: [c.name.indexOf('botanic') === 0 ? 'cyan' : 'magenta']}, c.name, true);
});
print('Первые 10 строк:', table.limit(10));

// --- снимки прошлых лет, чтобы выбрать контроль (слои включаются в панели Layers справа) ---
// Подложка GEE показывает только современный снимок. Здесь летние медианы по годам:
// Landsat 8 (30 м, с 2013) и Sentinel-2 (10 м, берём с 2019). Плюс изменение NDVI 2013 -> 2025:
// для контроля нужен участок, где изменение около нуля (белый цвет).
var VIEW = ee.Geometry.Point([71.40, 51.12]).buffer(8000);
function l8(year) {
  return ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(VIEW)
    .filterDate(year + '-06-01', year + '-09-01').filter(ee.Filter.lt('CLOUD_COVER', 20))
    .median().select(['SR_B4', 'SR_B3', 'SR_B2', 'SR_B5']).multiply(0.0000275).add(-0.2).clip(VIEW);
}
function s2(year) {
  return ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(VIEW)
    .filterDate(year + '-06-01', year + '-09-01').filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
    .median().select(['B4', 'B3', 'B2']).multiply(0.0001).clip(VIEW);
}
var RGB = {bands: ['SR_B4', 'SR_B3', 'SR_B2'], min: 0, max: 0.25};
[2013, 2016, 2019, 2022, 2025].forEach(function (y) {
  Map.addLayer(l8(y), RGB, 'Landsat ' + y + ' (30 м)', false);
});
[2019, 2022, 2025].forEach(function (y) {
  Map.addLayer(s2(y), {bands: ['B4', 'B3', 'B2'], min: 0, max: 0.25}, 'Sentinel-2 ' + y + ' (10 м)', false);
});
function ndvi(img) { return img.normalizedDifference(['SR_B5', 'SR_B4']); }
Map.addLayer(ndvi(l8(2025)).subtract(ndvi(l8(2013))),
             {min: -0.4, max: 0.4, palette: ['brown', 'white', 'green']}, 'изменение NDVI 2013 -> 2025', false);

Map.centerObject(sites, 13);
Map.addLayer(sites, {color: 'yellow'}, 'участки (после отступа)');

if (START_EXPORT) {
  Export.table.toDrive({collection: table, description: 'astana_case_sites', folder: FOLDER,
                        fileFormat: 'CSV',
                        selectors: ['acq_key', 'sensor', 'date', 'year', 'site', 'lst_mean',
                                    'lst_median', 'ndvi_mean', 'n_valid', 'n_total']});
}