// =====================================================================================
// 05_sample_points.js   (Google Earth Engine Code Editor)   RUN ONCE
// Fixed set of pixels for the model: the same locations in all scenes and years.
// Half of the points in built-up land (WorldCover "built" share of the 30 m pixel > 0.1), half outside it.
// Points sit at the pixel centres of the same 30 m grid as all rasters (EPSG:32642, 2000 x 2400).
//
// Output (two exports):
//   1) asset  ASSET_ID              -> read by 06_scene_samples.js
//   2) CSV    astana_points.csv     -> for checking / for Python (pixel_id, x, y, urban_stratum)
// pixel_id = row * 2000 + col (row from the top, col from the left), as in 03_build_dataset.py.
// =====================================================================================

// ============ COMMON SETTINGS (identical in all scripts, do not change) ============
var CRS = 'EPSG:32642';
var XMIN = 642000, XMAX = 702000, YMIN = 5634000, YMAX = 5706000;
var FOLDER = 'astana_uhi';
var AOI = ee.Geometry.Rectangle([XMIN, YMIN, XMAX, YMAX], CRS, false);
function grid(res) { return [res, 0, XMIN, 0, -res, YMAX]; }
// ==================================================================================

var START_EXPORT = false;
// Where to save the asset. Check the root in the Assets tab (left) and edit it if yours differs.
var ASSET_ID = 'projects/dataset-collection-511715/assets/astana_points_v1';
var N_URBAN = 50000, N_RURAL = 50000, SEED = 42;
var NCOL = 2000;   // width of the 30 m grid

var P30 = ee.Projection(CRS).atScale(30);   // aligned grid: 642000 and 5706000 are divisible by 30
var wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map');
var fracBuilt = wc.eq(50).reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 1024})
                  .reproject({crs: CRS, crsTransform: grid(30)});
var stratum = fracBuilt.gt(0.1).rename('urban_stratum').toInt().clip(AOI);

var raw = stratum.stratifiedSample({
  numPoints: 0, classBand: 'urban_stratum', region: AOI, projection: P30,
  classValues: [0, 1], classPoints: [N_RURAL, N_URBAN], seed: SEED, geometries: true
});

var points = raw.map(function (f) {
  var c = f.geometry().transform(CRS, 0.01).coordinates();
  var col = ee.Number(c.get(0)).subtract(XMIN).divide(30).floor();
  var row = ee.Number(YMAX).subtract(ee.Number(c.get(1))).divide(30).floor();
  var x = col.add(0.5).multiply(30).add(XMIN);
  var y = ee.Number(YMAX).subtract(row.add(0.5).multiply(30));
  return ee.Feature(ee.Geometry.Point([x, y], CRS), {
    pixel_id: row.multiply(NCOL).add(col), x: x, y: y, urban_stratum: f.get('urban_stratum')
  });
});

print('Points in total (expected ' + (N_URBAN + N_RURAL) + '):', points.size());
print('Of these in built-up land:', points.filter(ee.Filter.eq('urban_stratum', 1)).size());
print('First 5:', points.limit(5));
Map.setCenter(71.43, 51.13, 10);
Map.addLayer(points.filter(ee.Filter.eq('urban_stratum', 1)).limit(5000), {color: 'red'}, 'points: built-up (5000)');
Map.addLayer(points.filter(ee.Filter.eq('urban_stratum', 0)).limit(5000), {color: 'blue'}, 'points: outside built-up (5000)');

if (START_EXPORT) {
  Export.table.toAsset({collection: points, description: 'astana_points_v1_asset', assetId: ASSET_ID});
  Export.table.toDrive({collection: points, description: 'astana_points', folder: FOLDER,
                        fileFormat: 'CSV', selectors: ['pixel_id', 'x', 'y', 'urban_stratum']});
}
