// =====================================================================================
// 02_sentinel2_composites.js   (Google Earth Engine Code Editor)
// Summer Sentinel-2 L2A composites: indices from a DIFFERENT sensor than the temperature (Landsat).
// They ensure that predictors and target do not share a scene (check of the R2 inflation).
//
// Output file: astana_S2_<year>_JJA.tif, 4 bands on the 30 m grid:
//   1 S2_NDVI    (B8 - B4) / (B8 + B4)
//   2 S2_NDBI    (B11 - B8) / (B11 + B8)
//   3 S2_MNDWI   (B3 - B11) / (B3 + B11)   water; separates ponds and the Ishim river well
//   4 S2_N_OBS   mean number of clear observations (over the 10 m pixels inside each 30 m pixel)
// Indices are computed at 10 m and averaged to 30 m (reduceResolution) instead of being taken
// from the pixel "centre".
//
// L2A is reliably available over Kazakhstan from 2019, hence the years 2019-2025.
// The script is heavy: if a task fails with a memory or time error, shrink the window or export
// one year at a time (change YEARS).
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
var YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025];
var MAX_TILE_CLOUD = 20; // CLOUDY_PIXEL_PERCENTAGE

function prepS2(img) {
  var scl = img.select('SCL');
  // 4 vegetation, 5 bare soil, 6 water. Everything else (cloud, shadow, snow, cirrus) is discarded.
  var ok = scl.eq(4).or(scl.eq(5)).or(scl.eq(6));
  var b = img.select(['B3', 'B4', 'B8', 'B11']).multiply(0.0001);
  var b3 = b.select('B3'), b4 = b.select('B4'), b8 = b.select('B8'), b11 = b.select('B11');
  var ndvi = b8.subtract(b4).divide(b8.add(b4)).rename('S2_NDVI');
  var ndbi = b11.subtract(b8).divide(b11.add(b8)).rename('S2_NDBI');
  var mndwi = b3.subtract(b11).divide(b3.add(b11)).rename('S2_MNDWI');
  // As in 01 and 06: indices outside [-1, 1] are masked (with non-negative reflectance
  // they should not occur; the mask is a safeguard)
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

  // 10 m -> mean over 30 m
  var agg = comp.reproject({crs: CRS, scale: 10})
    .reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 64})
    .reproject({crs: CRS, crsTransform: grid(30)});

  var name = 'astana_S2_' + year + '_JJA';
  raw.size().evaluate(function (n) {
    print(name + ': images = ' + n);
    if (n === 0) { print('  -> skipped'); return; }
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