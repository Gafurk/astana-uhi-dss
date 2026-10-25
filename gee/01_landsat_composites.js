// =====================================================================================
// 01_landsat_composites.js   (Google Earth Engine Code Editor)
// Летние (июнь-август) композиты Landsat 8 и 9 для Астаны, по одному файлу на сенсор и год.
//
// Файл на выходе: astana_L8_2019_JJA.tif и т.д., 5 каналов в таком порядке:
//   1 LST_C   температура поверхности, °C
//   2 NDVI    (NIR - Red) / (NIR + Red)
//   3 NDBI    (SWIR1 - NIR) / (SWIR1 + NIR)
//   4 ALBEDO  широкополосное альбедо по Liang (2001)
//   5 N_OBS   сколько ясных наблюдений дало значение пикселя
//
// ВАЖНО по фидбэку: NDVI, NDBI и ALBEDO здесь взяты из ТОЙ ЖЕ сцены, что и LST.
// В модели это группа признаков "C" (та же сцена), её нельзя смешивать с остальными без
// отдельной оценки R2 (см. гид, раздел про утечку).
// Landsat 7 по умолчанию выключен: для картографии он не нужен.
//
// Как запускать: вставить в Code Editor -> Run -> справа вкладка Tasks -> у каждой задачи Run.
//
// Правки после проверки тестового файла (L8 2024):
//   - NDVI/NDBI вне [-1, 1] и альбедо вне [0, 1] маскируются (тёмные поверхности с отрицательной SR
//     давали значения до 112). LST эта маска не трогает. Порядок каналов прежний.
//   - Регион экспорта сдвинут внутрь окна на 10 м: иначе GEE добавлял пустые строку и столбец (2001x2401).
//     Сетка не меняется: её задаёт crsTransform. Ожидаемый размер файла 2000 x 2400.
//   - Индексы маскируются там, где нет LST той же сцены (в тесте 2024: 144 543 пикселя с индексами
//     без LST в юго-западной части окна). Теперь N_OBS описывает все 4 канала.
// =====================================================================================

// ============ ОБЩИЕ НАСТРОЙКИ (одинаковые во всех скриптах, не меняй) ============
var CRS = 'EPSG:32642';
var XMIN = 642000, XMAX = 702000, YMIN = 5634000, YMAX = 5706000;
var FOLDER = 'astana_uhi';
var AOI = ee.Geometry.Rectangle([XMIN, YMIN, XMAX, YMAX], CRS, false);
function grid(res) { return [res, 0, XMIN, 0, -res, YMAX]; }
// ==================================================================================

var INCLUDE_L7 = false;   // true только если ты обосновал роль L7 в Methods
var MAX_SCENE_CLOUD = 10; // процентов, фильтр по сцене (как в Methods)
var START_EXPORT = false; // false: только таблица числа сцен в консоль; true: создать задачи экспорта

// Регион экспорта на 10 м внутри окна, чтобы файл был ровно 2000 x 2400 (сетку задаёт grid(30))
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
    // QA_PIXEL: биты 1 (dilated cloud), 2 (cirrus), 3 (cloud), 4 (shadow), 5 (snow) должны быть 0
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
    // Индексы вне физического диапазона (тёмные поверхности с отрицательной SR) убираем, LST не трогаем
    ndvi = ndvi.updateMask(ndvi.abs().lte(1));
    ndbi = ndbi.updateMask(ndbi.abs().lte(1));
    albedo = albedo.updateMask(albedo.gte(0).and(albedo.lte(1)));
    // Индексы оставляем только там, где есть LST этой же сцены: у теплового канала (ST) своё покрытие,
    // и без этого медиана индексов шла по большему числу сцен, чем LST (группа C была бы не «та же сцена»).
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
        print(name + ': сцен = ' + n);
        if (n === 0) { print('  -> пропущено, нет сцен'); return; }
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

// Для быстрого просмотра на карте (последний год L8):
var preview = collectionFor(SENSORS[0], 2024).map(makePrep(SENSORS[0])).median().clip(AOI);
Map.setCenter(71.43, 51.13, 10);
Map.addLayer(preview.select('NDVI'), {min: 0, max: 0.8, palette: ['brown', 'yellow', 'green']}, 'NDVI 2024 L8', false);
Map.addLayer(preview.select('ALBEDO'), {min: 0.05, max: 0.35, palette: ['black', 'white']}, 'Albedo 2024 L8', false);
Map.addLayer(preview.select('LST_C'), {min: 20, max: 45, palette: ['blue', 'cyan', 'yellow', 'orange', 'red']}, 'LST °C 2024 L8');