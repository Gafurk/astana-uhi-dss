// =====================================================================================
// 09_case_rings.js   (Google Earth Engine Code Editor)
// Greening case study, main export (Botanical Garden, GreenLine). Differences from 08:
//  1) Years 2013-2026 (summer 2026 added). The scene list is built here, without the asset from 06.
//  2) Cloud is computed over a window around the sites (WIN), not over the whole of Astana:
//     a scene that is cloudy on the outskirts but clear over the garden is no longer discarded.
//  3) Besides the distant controls (as in 08), rings around each site are computed:
//     0-60, 60-120, 120-240, 240-480, 480-960 m from the edge. This gives the "site -> surroundings" profile.
//  4) Water (WorldCover 80) is excluded from all zones, so that the garden pond is not taken for a tree effect.
//  5) Sentinel-2 (10 m): summer median NDVI by year, 2019-2026, for the same zones. The narrow boulevard
//     (about 55 m) is visible at 10 m but hardly at Landsat 30 m.
//  6) The sites can be redrawn: draw a polygon greenline_v2 and/or botanic_v2 in Imports,
//     and the script uses it instead of the coordinates in the code. If none is drawn, the old polygon is used.
//
// Output (Google Drive, folder astana_uhi):
//   astana_case_rings_lst.csv : acq_key, sensor, date, year, site, zone, lst_mean, lst_median,
//                                 ndvi_mean, n_valid, n_total, win_clear
//   astana_case_rings_s2.csv  : year, site, zone, s2_ndvi_mean, s2_ndvi_median, n_valid
// =====================================================================================

// ============ COMMON SETTINGS (identical in all scripts, do not change) ============
var CRS = 'EPSG:32642';
var XMIN = 642000, XMAX = 702000, YMIN = 5634000, YMAX = 5706000;
var FOLDER = 'astana_uhi';
var AOI = ee.Geometry.Rectangle([XMIN, YMIN, XMAX, YMAX], CRS, false);
function grid(res) { return [res, 0, XMIN, 0, -res, YMAX]; }
// ==================================================================================

var START_EXPORT = false;
var YEAR0 = 2013, YEAR1 = 2026;
var MIN_WIN_CLEAR = 0.5;      // share of clear pixels in the window; the stricter filter is applied in Python (per zone)
var INNER_BUFFER_M = 60;      // inner buffer for the garden (as in 08)
var RINGS = [0, 60, 120, 240, 480, 960];   // ring limits, m from the site edge

// --- sites: original coordinates (08) or redrawn *_v2 polygons from Imports ---
var BOTANIC_OLD = ee.Geometry.Polygon([[[71.41143054465597, 51.11127944560988],
                                        [71.40851230124777, 51.10306119901771],
                                        [71.42168731192892, 51.10109399211938],
                                        [71.4246913860256, 51.10939341985413]]]);
var GREENLINE_OLD = ee.Geometry.Polygon([[[71.397636885105, 51.13375237309245],
                                          [71.36965608066164, 51.13800704933611],
                                          [71.36957024997317, 51.1377108503962],
                                          [71.39750813907229, 51.133240708402084]]]);
function pick(getter, fallback, label) {
  try { var g = getter(); print(label + ': redrawn polygon used'); return g; }
  catch (e) { print(label + ': polygon from 08 used'); return fallback; }
}
var BOTANIC = pick(function () { return botanic_v2; }, BOTANIC_OLD, 'botanic');
var GREENLINE = pick(function () { return greenline_v2; }, GREENLINE_OLD, 'greenline');

var OBJECTS = [
  {name: 'botanic', geom: BOTANIC, inner: INNER_BUFFER_M},
  {name: 'greenline', geom: GREENLINE, inner: 0}
];

// cloud window: both sites plus 1 km
var WIN = BOTANIC.union(GREENLINE, 1).bounds(1).buffer(1000);

var WC = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map');
var LAND = WC.neq(80);

// zones: the site itself (with inner buffer) and rings outside its edge
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

// --- rule-based distant controls, as in 08 (stable NDVI 2013-14 -> 2024-25) ---
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

// --- Landsat 8/9 scenes, June-August, acquisition = sensor + date ---
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

// --- Sentinel-2: summer median NDVI by year, 10 m, SCL mask 4/5/6 as in 02 ---
function s2Ndvi(year) {   // year: a number or ee.Number
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

// --- checks in the Console ---
print('Acquisitions in total 2013-' + YEAR1 + ':', ACQ.size());
print('Passed the window filter (win_clear >= ' + MIN_WIN_CLEAR + '):', KEPT.size());
print('Of these in 2026:', KEPT.filter(ee.Filter.eq('year', 2026)).aggregate_array('acq_key'));
print('Zone area, ha:', ZONES.map(function (z) {
  return z.set('ha', z.geometry().area(1).divide(1e4));
}).reduceColumns(ee.Reducer.toList(3), ['site', 'zone', 'ha']).get('list'));
print('S2 NDVI by year, sites only:', S2_TABLE.filter(ee.Filter.eq('zone', 'site')));
print('First 5 LST rows:', LST_TABLE.limit(5));

// --- map: check that the outline lies on the green strip (S2 2025 and 2026, 10 m) ---
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
             {}, 'rings', false);
Map.addLayer(ee.FeatureCollection([ee.Feature(BOTANIC), ee.Feature(GREENLINE)])
               .style({color: 'yellow', fillColor: '00000000', width: 2}), {}, 'sites (outline)');
Map.centerObject(GREENLINE, 15);

if (START_EXPORT) {
  Export.table.toDrive({collection: LST_TABLE, description: 'astana_case_rings_lst', folder: FOLDER,
    fileFormat: 'CSV',
    selectors: ['acq_key', 'sensor', 'date', 'year', 'site', 'zone', 'lst_mean', 'lst_median',
                'ndvi_mean', 'n_valid', 'n_total', 'win_clear']});
  Export.table.toDrive({collection: S2_TABLE, description: 'astana_case_rings_s2', folder: FOLDER,
    fileFormat: 'CSV',
    selectors: ['year', 'site', 'zone', 's2_ndvi_mean', 's2_ndvi_median', 'n_valid']});
}