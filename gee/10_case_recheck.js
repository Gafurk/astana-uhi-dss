// =====================================================================================
// 10_case_recheck.js   (Google Earth Engine Code Editor)
// Повторная проверка кейса (ботанический сад, GreenLine). Отличия от 09:
//  1) Годы 2013-2026 (добавлено лето 2026). Список снимков строится здесь, без asset из 07.
//  2) Облачность считается над окном вокруг объектов (WIN), а не над всей Астаной:
//     снимок, облачный на окраине, но ясный над садом, теперь не выбрасывается.
//  3) Кроме дальних контролей (как в 09) считаются кольца вокруг каждого объекта:
//     0-60, 60-120, 120-240, 240-480, 480-960 м от края. Это профиль "объект -> окрестности".
//  4) Вода (WorldCover 80) исключена из всех зон, чтобы пруд сада не выдавался за эффект деревьев.
//  5) Sentinel-2 (10 м): летняя медиана NDVI по годам 2019-2026 для тех же зон. Узкий бульвар
//     (около 55 м) на 10 м виден, на 30 м Landsat почти нет.
//  6) Можно перерисовать объекты: нарисуй полигон greenline_v2 и/или botanic_v2 в Imports,
//     скрипт возьмёт его вместо координат в коде. Не нарисовал -> используется старый полигон.
//
// Выход (Drive, папка astana_uhi):
//   astana_case_recheck_lst.csv : acq_key, sensor, date, year, site, zone, lst_mean, lst_median,
//                                 ndvi_mean, n_valid, n_total, win_clear
//   astana_case_recheck_s2.csv  : year, site, zone, s2_ndvi_mean, s2_ndvi_median, n_valid
// =====================================================================================

// ============ ОБЩИЕ НАСТРОЙКИ (одинаковые во всех скриптах, не меняй) ============
var CRS = 'EPSG:32642';
var XMIN = 642000, XMAX = 702000, YMIN = 5634000, YMAX = 5706000;
var FOLDER = 'astana_uhi';
var AOI = ee.Geometry.Rectangle([XMIN, YMIN, XMAX, YMAX], CRS, false);
function grid(res) { return [res, 0, XMIN, 0, -res, YMAX]; }
// ==================================================================================

var START_EXPORT = false;
var YEAR0 = 2013, YEAR1 = 2026;
var MIN_WIN_CLEAR = 0.5;      // доля ясных пикселей в окне; строже фильтр в Python (по каждой зоне)
var INNER_BUFFER_M = 60;      // отступ внутрь для сада (как в 09)
var RINGS = [0, 60, 120, 240, 480, 960];   // границы колец, м от края объекта

// --- объекты: старые координаты (09) или перерисованные полигоны *_v2 из Imports ---
var BOTANIC_OLD = ee.Geometry.Polygon([[[71.41143054465597, 51.11127944560988],
                                        [71.40851230124777, 51.10306119901771],
                                        [71.42168731192892, 51.10109399211938],
                                        [71.4246913860256, 51.10939341985413]]]);
var GREENLINE_OLD = ee.Geometry.Polygon([[[71.397636885105, 51.13375237309245],
                                          [71.36965608066164, 51.13800704933611],
                                          [71.36957024997317, 51.1377108503962],
                                          [71.39750813907229, 51.133240708402084]]]);
function pick(getter, fallback, label) {
  try { var g = getter(); print(label + ': взят перерисованный полигон'); return g; }
  catch (e) { print(label + ': взят полигон из 09'); return fallback; }
}
var BOTANIC = pick(function () { return botanic_v2; }, BOTANIC_OLD, 'botanic');
var GREENLINE = pick(function () { return greenline_v2; }, GREENLINE_OLD, 'greenline');

var OBJECTS = [
  {name: 'botanic', geom: BOTANIC, inner: INNER_BUFFER_M},
  {name: 'greenline', geom: GREENLINE, inner: 0}
];

// окно для облачности: оба объекта плюс 1 км
var WIN = BOTANIC.union(GREENLINE, 1).bounds(1).buffer(1000);

var WC = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map');
var LAND = WC.neq(80);

// зоны: сам объект (с отступом) и кольца снаружи от края
var zoneList = [];
OBJECTS.forEach(function (o) {
  var core = o.inner > 0 ? o.geom.buffer(-o.inner, 1) : o.geom;
  zoneList.push(ee.Feature(core, {site: o.name, zone: 'site'}));
  for (var i = 0; i < RINGS.length - 1; i++) {
    var outer = o.geom.buffer(RINGS[i + 1], 1);
    var ring = RINGS[i] === 0 ? outer.difference(o.geom, 1)
                              : outer.difference(o.geom.buffer(RINGS[i], 1), 1);
    zoneList.push(ee.Feature(ring, {site: o.name,
                                    zone: 'ring_' + RINGS[i] + '_' + RINGS[i + 1]}));
  }
});
var ZONES = ee.FeatureCollection(zoneList);

// --- дальние контроли по правилу, как в 09 (стабильный NDVI 2013-14 -> 2024-25) ---
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
var STABLE = summerNdvi(2024, 2025).subtract(NDVI_EARLY).abs().lt(DNDVI_MAX);
var AUTO = [
  {site: 'botanic', zone: 'ctrl_auto',
   ring: BOTANIC.buffer(4000, 1).difference(BOTANIC.buffer(1000, 1), 1),
   mask: STABLE.and(WC.neq(50)).and(LAND).and(NDVI_EARLY.lt(0.5))},
  {site: 'greenline', zone: 'ctrl_auto',
   ring: GREENLINE.buffer(3000, 1).difference(GREENLINE.buffer(500, 1), 1),
   mask: STABLE.and(WC.eq(50)).and(NDVI_EARLY.lt(0.3))}
];

// --- снимки Landsat 8/9, июнь-август, акквизиция = сенсор + дата ---
function withSensor(id, sensor) {
  return ee.ImageCollection(id).filterBounds(WIN)
    .filter(ee.Filter.calendarRange(6, 8, 'month'))
    .filter(ee.Filter.calendarRange(YEAR0, YEAR1, 'year'))
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
var ones = ee.Image.constant(1).rename('px');

var KEYS = RAW.aggregate_array('acq_key').distinct().sort();
var ACQ = ee.FeatureCollection(KEYS.map(function (key) {
  var imgs = RAW.filter(ee.Filter.eq('acq_key', key));
  var first = ee.Image(imgs.first());
  var img = imgs.map(prep).mosaic();
  var cnt = img.select('LST_C').mask().unmask(0).rename('ok').addBands(ones)
    .reduceRegion({reducer: ee.Reducer.sum(), geometry: WIN, crs: CRS, crsTransform: grid(30),
                   maxPixels: 1e8});
  var date = ee.String(first.get('DATE_ACQUIRED'));
  return ee.Feature(null, {acq_key: key, sensor: first.get('sensor'), date: date,
                           year: ee.Number.parse(date.slice(0, 4)),
                           win_clear: ee.Number(cnt.get('ok')).divide(cnt.get('px'))});
}));
var KEPT = ACQ.filter(ee.Filter.gte('win_clear', MIN_WIN_CLEAR));

var reducer = ee.Reducer.mean().combine(ee.Reducer.median(), null, true)
                               .combine(ee.Reducer.count(), null, true);

var LST_TABLE = KEPT.map(function (f) {
  var key = f.get('acq_key');
  var img = RAW.filter(ee.Filter.eq('acq_key', key)).map(prep).mosaic().updateMask(LAND);
  var props = {acq_key: key, sensor: f.get('sensor'), date: f.get('date'), year: f.get('year'),
               win_clear: f.get('win_clear')};
  var z = img.addBands(ones.updateMask(LAND)).reduceRegions({
    collection: ZONES, reducer: reducer, crs: CRS, crsTransform: grid(30), tileScale: 2
  }).map(function (s) {
    return ee.Feature(null, props).set({
      site: s.get('site'), zone: s.get('zone'),
      lst_mean: s.get('LST_C_mean'), lst_median: s.get('LST_C_median'),
      ndvi_mean: s.get('NDVI_mean'), n_valid: s.get('LST_C_count'), n_total: s.get('px_count')});
  });
  var auto = AUTO.map(function (c) {
    var r = img.updateMask(c.mask).addBands(ones.updateMask(c.mask)).reduceRegion({
      reducer: reducer, geometry: c.ring, crs: CRS, crsTransform: grid(30), maxPixels: 1e9, tileScale: 2
    });
    return ee.Feature(null, props).set({
      site: c.site, zone: c.zone,
      lst_mean: r.get('LST_C_mean'), lst_median: r.get('LST_C_median'),
      ndvi_mean: r.get('NDVI_mean'), n_valid: r.get('LST_C_count'), n_total: r.get('px_count')});
  });
  return z.merge(ee.FeatureCollection(auto));
}).flatten();

// --- Sentinel-2: летняя медиана NDVI по годам, 10 м, маска SCL 4/5/6 как в 04 ---
function s2Ndvi(year) {   // year: число или ee.Number
  return ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(WIN)
    .filterDate(ee.Date.fromYMD(year, 6, 1), ee.Date.fromYMD(year, 9, 1))
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .map(function (img) {
      var scl = img.select('SCL');
      var ok = scl.eq(4).or(scl.eq(5)).or(scl.eq(6));
      var b = img.select(['B4', 'B8']).multiply(0.0001);
      var nd = b.select('B8').subtract(b.select('B4')).divide(b.select('B8').add(b.select('B4')));
      return nd.updateMask(ok.and(nd.abs().lte(1))).rename('S2_NDVI');
    }).median().updateMask(LAND);
}
var S2_YEARS = ee.List.sequence(2019, YEAR1);
var S2_TABLE = ee.FeatureCollection(S2_YEARS.map(function (y) {
  y = ee.Number(y).int();
  var img = s2Ndvi(y);
    return img.addBands(ones).reduceRegions({collection: ZONES, reducer: reducer, scale: 10, crs: CRS, tileScale: 2})
    .map(function (s) {
      return ee.Feature(null, {year: y, site: s.get('site'), zone: s.get('zone'),
                               s2_ndvi_mean: s.get('S2_NDVI_mean'),
                               s2_ndvi_median: s.get('S2_NDVI_median'),
                               n_valid: s.get('S2_NDVI_count')});
    });
})).flatten();

// --- проверка в Console ---
print('Акквизиций всего 2013-' + YEAR1 + ':', ACQ.size());
print('Прошли фильтр по окну (win_clear >= ' + MIN_WIN_CLEAR + '):', KEPT.size());
print('Из них 2026:', KEPT.filter(ee.Filter.eq('year', 2026)).aggregate_array('acq_key'));
print('Площадь зон, га:', ZONES.map(function (z) {
  return z.set('ha', z.geometry().area(1).divide(1e4));
}).reduceColumns(ee.Reducer.toList(3), ['site', 'zone', 'ha']).get('list'));
print('S2 NDVI по годам, только объекты:', S2_TABLE.filter(ee.Filter.eq('zone', 'site')));
print('Первые 5 строк LST:', LST_TABLE.limit(5));

// --- карта: проверь, что контур лежит на зелёной полосе (S2 2025 и 2026, 10 м) ---
function s2Rgb(year) {
  return ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(WIN)
    .filterDate(year + '-06-01', year + '-09-01').filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
    .median().select(['B4', 'B3', 'B2']).multiply(0.0001).clip(WIN);
}
var NDVI_VIS = {min: 0, max: 0.7, palette: ['#8c510a', '#f6e8c3', '#c7eae5', '#01665e']};
Map.addLayer(s2Rgb(2019), {min: 0, max: 0.25}, 'Sentinel-2 2019 RGB', false);
Map.addLayer(s2Rgb(2025), {min: 0, max: 0.25}, 'Sentinel-2 2025 RGB', false);
Map.addLayer(s2Rgb(2026), {min: 0, max: 0.25}, 'Sentinel-2 2026 RGB', true);
Map.addLayer(s2Ndvi(2019).clip(WIN), NDVI_VIS, 'S2 NDVI 2019', false);
Map.addLayer(s2Ndvi(2026).clip(WIN), NDVI_VIS, 'S2 NDVI 2026', false);
Map.addLayer(ZONES.filter(ee.Filter.neq('zone', 'site')).style({color: 'cyan', fillColor: '00000000', width: 1}),
             {}, 'кольца', false);
Map.addLayer(ee.FeatureCollection([ee.Feature(BOTANIC), ee.Feature(GREENLINE)])
               .style({color: 'yellow', fillColor: '00000000', width: 2}), {}, 'объекты (контур)');
Map.centerObject(GREENLINE, 15);

if (START_EXPORT) {
  Export.table.toDrive({collection: LST_TABLE, description: 'astana_case_recheck_lst', folder: FOLDER,
    fileFormat: 'CSV',
    selectors: ['acq_key', 'sensor', 'date', 'year', 'site', 'zone', 'lst_mean', 'lst_median',
                'ndvi_mean', 'n_valid', 'n_total', 'win_clear']});
  Export.table.toDrive({collection: S2_TABLE, description: 'astana_case_recheck_s2', folder: FOLDER,
    fileFormat: 'CSV',
    selectors: ['year', 'site', 'zone', 's2_ndvi_mean', 's2_ndvi_median', 'n_valid']});
}