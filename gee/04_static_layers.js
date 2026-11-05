// =====================================================================================
// 04_static_layers.js   (Google Earth Engine Code Editor)
// Static layers on the 30 m grid: elevation, land cover, population, building height, and the reference
// LCZ map (Demuzere et al., 2022) for an independent check of the local LCZ labels.
//
// Output file: astana_static_30m.tif, 13 bands in this order (the order matters for
// scripts/03_build_dataset.py):
//   1  ELEV         elevation, m (Copernicus DEM GLO-30)
//   2  SLOPE_DEG    slope, degrees
//   3  WC_MODE      dominant ESA WorldCover class (10 m -> 30 m, mode)
//   4  FRAC_TREE    share of class 10 inside the 30 m pixel (from 10 m)
//   5  FRAC_SHRUB   class 20
//   6  FRAC_GRASS   class 30
//   7  FRAC_CROP    class 40
//   8  FRAC_BUILT   class 50 (built-up)
//   9  FRAC_BARE    class 60
//   10 FRAC_WATER   class 80
//   11 POP_DENS     population 2020, people/km2 (GHSL GHS-POP, 100 m)
//   12 BUILT_H      mean building height, m (GHSL GHS-BUILT-H, 2018, 100 m)
//   13 LCZ_REF      LCZ of Demuzere et al. 2022 (LCZ_Filter, 100 m; 1-10 built types, 11-17 land cover types)
//
// Predictor group "A" (structural, not from the same image as the temperature): bands 1-12.
// Band 13 is used only as an independent reference for checking the local LCZ labels.
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

// --- terrain ---
var demCol = ee.ImageCollection('COPERNICUS/DEM/GLO30').filterBounds(AOI).select('DEM');
var demProj = demCol.first().select(0).projection();
var dem = demCol.mosaic().setDefaultProjection(demProj);
var elev = dem.rename('ELEV');
var slope = ee.Terrain.slope(dem).rename('SLOPE_DEG');

// --- ESA WorldCover 2021, 10 m ---
var wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map');
var to30 = function (img, reducer) {
  return img.reduceResolution({reducer: reducer, maxPixels: 1024})
            .reproject({crs: CRS, crsTransform: grid(30)});
};
var wcMode = to30(wc, ee.Reducer.mode()).rename('WC_MODE');
function frac(code, name) { return to30(wc.eq(code), ee.Reducer.mean()).rename(name); }
var fracs = ee.Image.cat([
  frac(10, 'FRAC_TREE'), frac(20, 'FRAC_SHRUB'), frac(30, 'FRAC_GRASS'), frac(40, 'FRAC_CROP'),
  frac(50, 'FRAC_BUILT'), frac(60, 'FRAC_BARE'), frac(80, 'FRAC_WATER')
]);

// --- GHSL ---
// GHS-POP stores the number of people per 100 m cell (0.01 km2): converted to density, people/km2.
var pop = ee.Image('JRC/GHSL/P2023A/GHS_POP/2020').select('population_count')
  .divide(0.01).rename('POP_DENS');
var builtH = ee.Image('JRC/GHSL/P2023A/GHS_BUILT_H/2018').select('built_height').rename('BUILT_H');

// --- reference LCZ map ---
var lczRef = ee.ImageCollection('RUB/RUBCLIM/LCZ/global_lcz_map/latest').mosaic()
  .select('LCZ_Filter').rename('LCZ_REF');

var stack = ee.Image.cat([elev, slope, wcMode, fracs, pop, builtH, lczRef]).toFloat().clip(AOI);
print('Bands:', stack.bandNames());

Map.setCenter(71.43, 51.13, 10);
Map.addLayer(stack.select('FRAC_BUILT'), {min: 0, max: 1, palette: ['white', 'black']}, 'Built-up share');
Map.addLayer(stack.select('LCZ_REF'), {min: 1, max: 17}, 'LCZ (Demuzere)', false);
Map.addLayer(stack.select('POP_DENS'), {min: 0, max: 20000, palette: ['white', 'orange', 'red']}, 'Population density', false);

if (START_EXPORT) {
  Export.image.toDrive({
    image: stack,
    description: 'astana_static_30m',
    folder: FOLDER,
    fileNamePrefix: 'astana_static_30m',
    region: EXPORT_REGION,
    crs: CRS,
    crsTransform: grid(30),
    maxPixels: 1e9,
    fileFormat: 'GeoTIFF'
  });
}
