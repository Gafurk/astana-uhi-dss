// =====================================================================================
// 06_static_layers.js   (Google Earth Engine Code Editor)
// Статические слои на сетке 30 м: высота, покрытие, население, высота зданий, и опорная
// карта LCZ (Demuzere et al., 2022) для независимой проверки твоих меток LCZ.
//
// Файл на выходе: astana_static_30m.tif, 13 каналов в таком порядке (порядок важен для
// build_dataset.py):
//   1  ELEV         высота, м (Copernicus DEM GLO-30)
//   2  SLOPE_DEG    уклон, градусы
//   3  WC_MODE      доминирующий класс ESA WorldCover (10 м -> 30 м, мода)
//   4  FRAC_TREE    доля класса 10 внутри пикселя 30 м (из 10 м)
//   5  FRAC_SHRUB   класс 20
//   6  FRAC_GRASS   класс 30
//   7  FRAC_CROP    класс 40
//   8  FRAC_BUILT   класс 50 (застройка)
//   9  FRAC_BARE    класс 60
//   10 FRAC_WATER   класс 80
//   11 POP_DENS     население 2020, чел/км2 (GHSL GHS-POP, 100 м)
//   12 BUILT_H      средняя высота зданий, м (GHSL GHS-BUILT-H, 2018, 100 м)
//   13 LCZ_REF      LCZ по Demuzere et al. 2022 (LCZ_Filter, 100 м; 1-10 застройка, 11-17 природа)
//
// Группа признаков "A" (структурные, не из того же снимка, что температура): каналы 1-12.
// Канал 13 используется только как независимый ориентир для проверки твоих LCZ.
// =====================================================================================

// ============ ОБЩИЕ НАСТРОЙКИ (одинаковые во всех скриптах, не меняй) ============
var CRS = 'EPSG:32642';
var XMIN = 642000, XMAX = 702000, YMIN = 5634000, YMAX = 5706000;
var FOLDER = 'astana_uhi';
var AOI = ee.Geometry.Rectangle([XMIN, YMIN, XMAX, YMAX], CRS, false);
function grid(res) { return [res, 0, XMIN, 0, -res, YMAX]; }
// ==================================================================================

// Регион экспорта на 10 м внутри окна, чтобы файл был ровно по сетке без лишних строки и столбца
// (как в 01). Сетку задаёт crsTransform, начало координат остаётся (XMIN, YMAX).
var EXPORT_REGION = ee.Geometry.Rectangle([XMIN + 10, YMIN + 10, XMAX - 10, YMAX - 10], CRS, false);

var START_EXPORT = false;

// --- рельеф ---
var demCol = ee.ImageCollection('COPERNICUS/DEM/GLO30').filterBounds(AOI).select('DEM');
var demProj = demCol.first().select(0).projection();
var dem = demCol.mosaic().setDefaultProjection(demProj);
var elev = dem.rename('ELEV');
var slope = ee.Terrain.slope(dem).rename('SLOPE_DEG');

// --- ESA WorldCover 2021, 10 м ---
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
// GHS-POP хранит число людей в клетке 100 м (0.01 км2): переводим в плотность, чел/км2.
var pop = ee.Image('JRC/GHSL/P2023A/GHS_POP/2020').select('population_count')
  .divide(0.01).rename('POP_DENS');
var builtH = ee.Image('JRC/GHSL/P2023A/GHS_BUILT_H/2018').select('built_height').rename('BUILT_H');

// --- опорная карта LCZ ---
var lczRef = ee.ImageCollection('RUB/RUBCLIM/LCZ/global_lcz_map/latest').mosaic()
  .select('LCZ_Filter').rename('LCZ_REF');

var stack = ee.Image.cat([elev, slope, wcMode, fracs, pop, builtH, lczRef]).toFloat().clip(AOI);
print('Каналы:', stack.bandNames());

Map.setCenter(71.43, 51.13, 10);
Map.addLayer(stack.select('FRAC_BUILT'), {min: 0, max: 1, palette: ['white', 'black']}, 'Доля застройки');
Map.addLayer(stack.select('LCZ_REF'), {min: 1, max: 17}, 'LCZ (Demuzere)', false);
Map.addLayer(stack.select('POP_DENS'), {min: 0, max: 20000, palette: ['white', 'orange', 'red']}, 'Плотность населения', false);

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
