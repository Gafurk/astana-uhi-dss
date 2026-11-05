// =====================================================================================
// 03_modis_jja.js   (Google Earth Engine Code Editor)
// MODIS Terra + Aqua, mean summer (June-August) land surface temperature, day and night, 1 km.
//
// Role of this file (Methods): night-time heat island at district scale and comparison with Yu et al.
// on the MODIS series (2013-2020). The representativeness check of Landsat scenes is done by 07_modis_daily_city.js.
// MODIS does NOT enter the 30 m model.
//
// Terra (MOD11A1) and Aqua (MYD11A1) are exported SEPARATELY: the overpass times of the satellites drift
// over the years, so they are not merged in advance. How to combine them is decided later from the data.
//
// Output file: astana_MODIS_JJA_<y0>_<y1>.tif on the 1 km grid (60 x 72 pixels).
// 4 bands per year, in this order:
//   TERRA_DAY_<year>, TERRA_NIGHT_<year>, AQUA_DAY_<year>, AQUA_NIGHT_<year>
// 2003-2025: 23 years x 4 = 92 bands. Values: mean summer LST, °C.
// Quality rule (the same for day and night), QC bits as in the GEE catalogue:
//   bits 0-1 <= 1  (LST produced: 0 = good, 1 = other quality)  AND  bits 6-7 <= 1 (LST error <= 2 K).
// Why not "good" only (bits 0-1 = 0): in central Astana in summer 2024 there is NOT ONE observation flagged
// good, by day or by night (diagnostic of 16.10.2026), so that filter removed the city itself.
// Where LST exists but the QC band is empty (masked), which is frequent at night (raw getRegion values,
// June 2024), QC = 0 is assumed, i.e. a good observation. This is inferred from the data, not documented in the catalogue.
// =====================================================================================

// ============ COMMON SETTINGS (identical in all scripts, do not change) ============
var CRS = 'EPSG:32642';
var XMIN = 642000, XMAX = 702000, YMIN = 5634000, YMAX = 5706000;
var FOLDER = 'astana_uhi';
var AOI = ee.Geometry.Rectangle([XMIN, YMIN, XMAX, YMAX], CRS, false);
function grid(res) { return [res, 0, XMIN, 0, -res, YMAX]; }
// ==================================================================================

// Export region 10 m inside the window, so that the file matches the grid without an extra row and column
// (as in 01). The grid is set by crsTransform; the origin stays at (XMIN, YMAX).
var EXPORT_REGION = ee.Geometry.Rectangle([XMIN + 10, YMIN + 10, XMAX - 10, YMAX - 10], CRS, false);

var START_EXPORT = false;
var Y0 = 2003, Y1 = 2025;   // 2003: first full year with both Terra and Aqua

var terra = ee.ImageCollection('MODIS/061/MOD11A1');
var aqua = ee.ImageCollection('MODIS/061/MYD11A1');

function qualityOk(img, qcBand) {
  var qc = img.select(qcBand).unmask(0);        // empty QC where LST exists -> 0 (see header)
  var flag = qc.bitwiseAnd(3);                  // bits 0-1
  var err = qc.rightShift(6).bitwiseAnd(3);     // bits 6-7
  return flag.lte(1).and(err.lte(1));
}
// The mask of the LST band itself is kept: where LST was not produced, the pixel stays empty.
function dayImg(img) {
  return img.select('LST_Day_1km').multiply(0.02).subtract(273.15)
            .updateMask(qualityOk(img, 'QC_Day')).rename('v');
}
function nightImg(img) {
  return img.select('LST_Night_1km').multiply(0.02).subtract(273.15)
            .updateMask(qualityOk(img, 'QC_Night')).rename('v');
}

var SATS = [{id: 'TERRA', coll: terra}, {id: 'AQUA', coll: aqua}];
var bands = [];
for (var y = Y0; y <= Y1; y++) {
  SATS.forEach(function (sat) {
    var col = sat.coll.filterDate(y + '-06-01', y + '-09-01').filterBounds(AOI);
    bands.push(col.map(dayImg).mean().rename(sat.id + '_DAY_' + y));
    bands.push(col.map(nightImg).mean().rename(sat.id + '_NIGHT_' + y));
  });
}
var stack = ee.Image.cat(bands).toFloat().clip(AOI);

print('Bands in the file (expected ' + 4 * (Y1 - Y0 + 1) + '):', stack.bandNames().size());
print('First 8 bands:', stack.bandNames().slice(0, 8));
Map.setCenter(71.43, 51.13, 9);
var PAL = ['blue', 'cyan', 'yellow', 'orange', 'red'];
Map.addLayer(stack.select('AQUA_NIGHT_2024'), {min: 8, max: 22, palette: PAL}, 'Aqua night 2024 °C');
Map.addLayer(stack.select('TERRA_NIGHT_2024'), {min: 8, max: 22, palette: PAL}, 'Terra night 2024 °C', false);
Map.addLayer(stack.select('TERRA_DAY_2024'), {min: 20, max: 45, palette: PAL}, 'Terra day 2024 °C', false);

if (START_EXPORT) {
  Export.image.toDrive({
    image: stack,
    description: 'astana_MODIS_JJA_' + Y0 + '_' + Y1,
    folder: FOLDER,
    fileNamePrefix: 'astana_MODIS_JJA_' + Y0 + '_' + Y1,
    region: EXPORT_REGION,
    crs: CRS,
    crsTransform: grid(1000),
    maxPixels: 1e9,
    fileFormat: 'GeoTIFF'
  });
}