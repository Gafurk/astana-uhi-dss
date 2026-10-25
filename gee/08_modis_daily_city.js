// =====================================================================================
// 08_modis_daily_city.js   (Google Earth Engine Code Editor)
// Ежедневная дневная LST MODIS над городом, июнь-август 2013-2025 (Methods 2.4, вторая роль MODIS):
// для каждой сцены Landsat берётся MODIS того же дня и сравнивается со средним за июнь-август
// этого года; сцены с отклонением больше 1 SD помечаются (это делает build_dataset.py).
// Terra и Aqua пишутся отдельными строками (Terra пролетает ближе ко времени Landsat).
// Правило качества то же, что в 05: биты QC 0-1 <= 1 и биты 6-7 <= 1; пустой QC при наличии LST = 0.
//
// ГОРОД ВРЕМЕННО: пиксели 1 км с долей застройки WorldCover > 0.5 в радиусе 15 км от центра.
// Когда будет граница Астаны, замени CITY_MASK (см. ниже) и перевыгрузи.
//
// Выход: astana_modis_daily_city.csv: date, sat, lst_city_c, n_px, n_px_city
//   n_px      сколько городских пикселей дали значение в этот день
//   n_px_city сколько городских пикселей всего (для доли покрытия)
// =====================================================================================

// ============ ОБЩИЕ НАСТРОЙКИ (одинаковые во всех скриптах, не меняй) ============
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
// Доля застройки в пикселе 1 км, в два шага (сразу 10 м -> 1 км даёт "Reprojection output too large"):
//   1) 10 м -> 30 м, так же как в 06 и 07a (около 9 входных пикселей на выходной);
//   2) 30 м -> 1 км (около 1100 входных на выходной).
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

print('Городских пикселей 1 км (временная маска):', nCity);
print('Строк (ожидается около 2 x 13 x 92 = 2392):', table.size());
print('Первые 10 строк:', table.limit(10));
Map.setCenter(71.43, 51.13, 10);
Map.addLayer(CITY_MASK, {palette: ['red']}, 'временная маска города (1 км)');

if (START_EXPORT) {
  Export.table.toDrive({collection: table, description: 'astana_modis_daily_city', folder: FOLDER,
                        fileFormat: 'CSV', selectors: ['date', 'sat', 'lst_city_c', 'n_px', 'n_px_city']});
}