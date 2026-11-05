// =====================================================================================
// 07_modis_daily_city.js   (Google Earth Engine Code Editor)
// Daily daytime MODIS LST over the city, June-August 2013-2025 (Methods, second role of MODIS):
// for each Landsat scene the MODIS value of the same day is compared with the June-August mean
// of that year; scenes deviating by more than 1 SD are flagged (done in 03_build_dataset.py).
// Terra and Aqua are written as separate rows (Terra passes closer to the Landsat overpass time).
// Quality rule as in 03: QC bits 0-1 <= 1 and bits 6-7 <= 1; empty QC where LST exists = 0.
//
// TEMPORARY CITY MASK: 1 km pixels with WorldCover built-up share > 0.5 within 15 km of the centre.
// Once the Astana boundary is available, replace CITY_MASK (see below) and re-export.
//
// Output: astana_modis_daily_city.csv: date, sat, lst_city_c, n_px, n_px_city
//   n_px      number of city pixels with a value on that day
//   n_px_city total number of city pixels (for the coverage share)
// =====================================================================================

// ============ COMMON SETTINGS (identical in all scripts, do not change) ============
var CRS = 'EPSG:32642';
var XMIN = 642000, XMAX = 702000, YMIN = 5634000, YMAX = 5706000;
var FOLDER = 'astana_uhi';
var AOI = ee.Geometry.Rectangle([XMIN, YMIN, XMAX, YMAX], CRS, false);
function grid(res) { return [res, 0, XMIN, 0, -res, YMAX]; }
// ==================================================================================

var START_EXPORT = true;
var Y0 = 2013, Y1 = 2025;
var CENTER = ee.Geometry.Point([71.43, 51.13]);

var wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map');
// Built-up share of a 1 km pixel, in two steps (10 m -> 1 km directly gives "Reprojection output too large"):
//   1) 10 m -> 30 m, as in 04 and 05 (about 9 input pixels per output pixel);
//   2) 30 m -> 1 km (about 1100 input pixels per output pixel).
var built30 = wc.eq(50).reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 1024})
                .reproject({crs: CRS, crsTransform: grid(30)});
var built1k = built30.reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 2048})
                     .reproject({crs: CRS, crsTransform: grid(1000)});
var CITY_MASK = built1k.gt(0.5).clip(CENTER.buffer(15000)).selfMask().rename('city');
var nCity = CITY_MASK.reduceRegion({reducer: ee.Reducer.count(), geometry: AOI,
                                    crs: CRS, crsTransform: grid(1000), maxPixels: 1e9,
                                    tileScale: 4}).get('city');

function qualityOk(img, qcBand) {
  var qc = img.select(qcBand).unmask(0);
  return qc.bitwiseAnd(3).lte(1).and(qc.rightShift(6).bitwiseAnd(3).lte(1));
}

function daily(collId, sat) {
  return ee.ImageCollection(collId).filterBounds(AOI)
    .filter(ee.Filter.calendarRange(6, 8, 'month'))
    .filter(ee.Filter.calendarRange(Y0, Y1, 'year'))
    .map(function (img) {
      var lst = img.select('LST_Day_1km').multiply(0.02).subtract(273.15)
                   .updateMask(qualityOk(img, 'QC_Day')).updateMask(CITY_MASK).rename('v');
      var st = lst.reduceRegion({reducer: ee.Reducer.mean().combine(ee.Reducer.count(), null, true),
                                 geometry: AOI, crs: CRS, crsTransform: grid(1000), maxPixels: 1e9,
                                 tileScale: 4});
      return ee.Feature(null, {date: img.date().format('YYYY-MM-dd'), sat: sat,
                               lst_city_c: st.get('v_mean'), n_px: st.get('v_count'), n_px_city: nCity});
    });
}
var table = daily('MODIS/061/MOD11A1', 'TERRA').merge(daily('MODIS/061/MYD11A1', 'AQUA')).sort('date');

print('City pixels at 1 km (temporary mask):', nCity);
print('Rows (expected about 2 x 13 x 92 = 2392):', table.size());
print('First 10 rows:', table.limit(10));
Map.setCenter(71.43, 51.13, 10);
Map.addLayer(CITY_MASK, {palette: ['red']}, 'temporary city mask (1 km)');

if (START_EXPORT) {
  Export.table.toDrive({collection: table, description: 'astana_modis_daily_city', folder: FOLDER,
                        fileFormat: 'CSV', selectors: ['date', 'sat', 'lst_city_c', 'n_px', 'n_px_city']});
}