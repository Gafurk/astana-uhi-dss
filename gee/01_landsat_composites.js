// =====================================================================================
// 01_landsat_composites.js   (Google Earth Engine Code Editor)
// Summer (June-August) Landsat 8 and 9 composites for Astana, one file per sensor and year.
//
// Output file: astana_L8_2019_JJA.tif etc., 5 bands in this order:
//   1 LST_C   land surface temperature, °C
//   2 NDVI    (NIR - Red) / (NIR + Red)
//   3 NDBI    (SWIR1 - NIR) / (SWIR1 + NIR)
//   4 ALBEDO  broadband albedo, Liang (2001) coefficients as applied to Landsat 9 by Barletta et al. (2023)
//   5 N_OBS   number of clear observations behind the pixel value
//
// NOTE: NDVI, NDBI and ALBEDO here come from the SAME scene as LST.
// In the model this is predictor group "C" (same scene); it must not be mixed with the others without
// a separate estimate of the R2 inflation (see the article, predictor source).
// Landsat 7 is off by default: it is not needed for mapping.
//
// How to run: paste into the Code Editor -> Run -> Tasks tab on the right -> Run for each task.
//
// Changes after checking the test file (L8 2024):
//   - NDVI/NDBI outside [-1, 1] and albedo outside [0, 1] are masked (dark surfaces with negative SR
//     gave values up to 112). LST is not affected by this mask. The band order is unchanged.
//   - The export region is shrunk by 10 m inside the window: otherwise GEE added an empty row and column (2001x2401).
//     The grid does not change: it is set by crsTransform. Expected file size 2000 x 2400.
//   - Indices are masked where the same scene has no LST (in the 2024 test: 144,543 pixels with indices
//     but no LST in the south-west of the window). N_OBS now describes all 4 bands.
// =====================================================================================

// ============ COMMON SETTINGS (identical in all scripts, do not change) ============
var CRS = 'EPSG:32642';
var XMIN = 642000, XMAX = 702000, YMIN = 5634000, YMAX = 5706000;
var FOLDER = 'astana_uhi';
var AOI = ee.Geometry.Rectangle([XMIN, YMIN, XMAX, YMAX], CRS, false);
function grid(res) { return [res, 0, XMIN, 0, -res, YMAX]; }
// ==================================================================================

var INCLUDE_L7 = false;   // true only if the role of L7 is justified in the Methods
var MAX_SCENE_CLOUD = 10; // per cent, scene-level filter (as in the Methods)
var START_EXPORT = false; // false: only print the scene counts to the console; true: create export tasks

// Export region 10 m inside the window, so that the file is exactly 2000 x 2400 (the grid is set by grid(30))
var EXPORT_REGION = ee.Geometry.Rectangle([XMIN + 10, YMIN + 10, XMAX - 10, YMAX - 10], CRS, false);

var SENSORS = [
  {id: 'L8', coll: 'LANDSAT/LC08/C02/T1_L2', y0: 2013, y1: 2025,
   blue: 'SR_B2', red: 'SR_B4', nir: 'SR_B5', swir1: 'SR_B6', swir2: 'SR_B7', tir: 'ST_B10'},
  {id: 'L9', coll: 'LANDSAT/LC09/C02/T1_L2', y0: 2022, y1: 2025,
   blue: 'SR_B2', red: 'SR_B4', nir: 'SR_B5', swir1: 'SR_B6', swir2: 'SR_B7', tir: 'ST_B10'}
];
if (INCLUDE_L7) {
  SENSORS.push({id: 'L7', coll: 'LANDSAT/LE07/C02/T1_L2', y0: 2013, y1: 2021,
   blue: 'SR_B1', red: 'SR_B3', nir: 'SR_B4', swir1: 'SR_B5', swir2: 'SR_B7', tir: 'ST_B6'});
}

function makePrep(s) {
  return function (img) {
    // QA_PIXEL: bits 1 (dilated cloud), 2 (cirrus), 3 (cloud), 4 (shadow), 5 (snow) must be 0
    var clear = img.select('QA_PIXEL').bitwiseAnd(0x3E).eq(0);
    // Collection 2 Level-2: SR = 0.0000275*DN - 0.2 ; ST(K) = 0.00341802*DN + 149.0
    var sr = img.select([s.blue, s.red, s.nir, s.swir1, s.swir2]).multiply(0.0000275).add(-0.2);
    var b = sr.select(s.blue), r = sr.select(s.red), n = sr.select(s.nir),
        s1 = sr.select(s.swir1), s2 = sr.select(s.swir2);
    var lst = img.select(s.tir).multiply(0.00341802).add(149.0).subtract(273.15).rename('LST_C');
    var ndvi = n.subtract(r).divide(n.add(r)).rename('NDVI');
    var ndbi = s1.subtract(n).divide(s1.add(n)).rename('NDBI');
    // Liang (2001): 0.356*B + 0.130*R + 0.373*NIR + 0.085*SWIR1 + 0.072*SWIR2 - 0.0018
    var albedo = b.multiply(0.356).add(r.multiply(0.130)).add(n.multiply(0.373))
                  .add(s1.multiply(0.085)).add(s2.multiply(0.072)).add(-0.0018).rename('ALBEDO');
    // Indices outside the physical range (dark surfaces with negative SR) are removed; LST is left as is
    ndvi = ndvi.updateMask(ndvi.abs().lte(1));
    ndbi = ndbi.updateMask(ndbi.abs().lte(1));
    albedo = albedo.updateMask(albedo.gte(0).and(albedo.lte(1)));
    // Indices are kept only where the same scene has LST: the thermal band (ST) has its own coverage,
    // and without this the index median used more scenes than LST (group C would not be the "same scene").
    var valid = clear.and(lst.mask());
    return lst.addBands(ndvi).addBands(ndbi).addBands(albedo).updateMask(valid)
              .copyProperties(img, ['system:time_start']);
  };
}

function collectionFor(s, year) {
  return ee.ImageCollection(s.coll)
    .filterBounds(AOI)
    .filterDate(year + '-06-01', year + '-09-01')
    .filter(ee.Filter.lt('CLOUD_COVER', MAX_SCENE_CLOUD));
}

SENSORS.forEach(function (s) {
  for (var year = s.y0; year <= s.y1; year++) {
    (function (year) {
      var raw = collectionFor(s, year);
      var prepped = raw.map(makePrep(s));
      var nObs = prepped.select('LST_C').count().rename('N_OBS');
      var composite = prepped.median().addBands(nObs)
        .select(['LST_C', 'NDVI', 'NDBI', 'ALBEDO', 'N_OBS']).toFloat().clip(AOI);
      var name = 'astana_' + s.id + '_' + year + '_JJA';

      raw.size().evaluate(function (n) {
        print(name + ': scenes = ' + n);
        if (n === 0) { print('  -> skipped, no scenes'); return; }
        if (!START_EXPORT) { return; }
        Export.image.toDrive({
          image: composite,
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
    })(year);
  }
});

// For a quick look on the map (last L8 year):
var preview = collectionFor(SENSORS[0], 2024).map(makePrep(SENSORS[0])).median().clip(AOI);
Map.setCenter(71.43, 51.13, 10);
Map.addLayer(preview.select('NDVI'), {min: 0, max: 0.8, palette: ['brown', 'yellow', 'green']}, 'NDVI 2024 L8', false);
Map.addLayer(preview.select('ALBEDO'), {min: 0.05, max: 0.35, palette: ['black', 'white']}, 'Albedo 2024 L8', false);
Map.addLayer(preview.select('LST_C'), {min: 20, max: 45, palette: ['blue', 'cyan', 'yellow', 'orange', 'red']}, 'LST °C 2024 L8');