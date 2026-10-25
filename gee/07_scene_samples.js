// =====================================================================================
// 07_scene_samples.js   (Google Earth Engine Code Editor)
// Таблица для модели по Methods 2.4: ЦЕЛЬ = LST ОДНОЙ акквизиции Landsat,
// предикторы NDVI/NDBI/альбедо = медиана ВСЕХ ДРУГИХ ясных акквизиций того же лета
// (целевая исключена); если в пикселе других меньше двух, композит расширяется на соседние лета
// (y-1 .. y+1, целевая всё равно исключена). Для оценки завышения R2 рядом пишутся индексы
// ТОЙ ЖЕ сцены (конфигурация same-scene).
//
// Акквизиция = один сенсор (L8/L9) + одна дата: снимки соседних рядов WRS одной даты склеиваются.
// Отбор по Methods 2.2: облачность < 10% НАД ОКНОМ (а не по всей сцене). Облако = биты QA 1-3
// (dilated cloud, cirrus, cloud); доля считается по той части окна, которую снимок покрывает.
// Маски пикселей как в 01: облака, тени, снег; индексы только там, где есть LST той же сцены;
// NDVI/NDBI вне [-1, 1] и альбедо вне [0, 1] маскируются. LST по диапазону не маскируется.
//
// Нужен asset с точками из 07a_sample_points.js (POINTS_ID).
// Выход: astana_scene_samples_<год>.csv, одна строка = точка x акквизиция. Пустые значения = -9999.
// Колонки: pixel_id, x, y, urban_stratum, acq_key, sensor, date, time_utc, cloud_frac_aoi,
//          cover_frac_aoi, LST_C, SS_NDVI, SS_NDBI, SS_ALBEDO, LOO_NDVI, LOO_NDBI, LOO_ALBEDO,
//          LOO_N, LOO_WIDE
//   SS_*   индексы той же сцены (только для оценки завышения R2)
//   LOO_*  индексы композита без целевой сцены (основная конфигурация)
//   LOO_N  сколько акквизиций вошло в композит в этом пикселе
//   LOO_WIDE 1, если композит расширен на соседние лета
// =====================================================================================

// ============ ОБЩИЕ НАСТРОЙКИ (одинаковые во всех скриптах, не меняй) ============
var CRS = 'EPSG:32642';
var XMIN = 642000, XMAX = 702000, YMIN = 5634000, YMAX = 5706000;
var FOLDER = 'astana_uhi';
var AOI = ee.Geometry.Rectangle([XMIN, YMIN, XMAX, YMAX], CRS, false);
function grid(res) { return [res, 0, XMIN, 0, -res, YMAX]; }
// ==================================================================================

var START_EXPORT = false;
var POINTS_ID = 'projects/dataset-collection-511715/assets/astana_points_v1';  // как ASSET_ID в 07a
var YEARS = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025];
var MAX_CLOUD_AOI = 0.10;   // Methods 2.2
var NODATA = -9999;
var IDX = ['NDVI', 'NDBI', 'ALBEDO'];
var P30 = ee.Projection(CRS).atScale(30);

function withSensor(id, sensor) {
  return ee.ImageCollection(id).filterBounds(AOI)
    .filter(ee.Filter.calendarRange(6, 8, 'month'))
    .filter(ee.Filter.calendarRange(2013, 2025, 'year'))
    .map(function (img) {
      return img.set('sensor', sensor,
                     'acq_key', ee.String(sensor).cat('_').cat(img.get('DATE_ACQUIRED')));
    });
}
var RAW = withSensor('LANDSAT/LC08/C02/T1_L2', 'L8').merge(withSensor('LANDSAT/LC09/C02/T1_L2', 'L9'));

// Подготовка снимка: так же, как в 01
function prep(img) {
  var clear = img.select('QA_PIXEL').bitwiseAnd(0x3E).eq(0);
  var sr = img.select(['SR_B2', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']).multiply(0.0000275).add(-0.2);
  var b = sr.select('SR_B2'), r = sr.select('SR_B4'), n = sr.select('SR_B5'),
      s1 = sr.select('SR_B6'), s2 = sr.select('SR_B7');
  var lst = img.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15).rename('LST_C');
  var ndvi = n.subtract(r).divide(n.add(r)).rename('NDVI');
  var ndbi = s1.subtract(n).divide(s1.add(n)).rename('NDBI');
  var albedo = b.multiply(0.356).add(r.multiply(0.130)).add(n.multiply(0.373))
                .add(s1.multiply(0.085)).add(s2.multiply(0.072)).add(-0.0018).rename('ALBEDO');
  ndvi = ndvi.updateMask(ndvi.abs().lte(1));
  ndbi = ndbi.updateMask(ndbi.abs().lte(1));
  albedo = albedo.updateMask(albedo.gte(0).and(albedo.lte(1)));
  var valid = clear.and(lst.mask());
  return lst.addBands(ndvi).addBands(ndbi).addBands(albedo).updateMask(valid);
}
function cloudBand(img) {   // 1 = облако (биты 1-3), 0 = нет; вне снимка пусто
  return img.select('QA_PIXEL').bitwiseAnd(0x0E).neq(0).rename('cloud');
}

// Одна акквизиция = мозаика всех снимков одного сенсора за одну дату
function makeAcq(key) {
  var imgs = RAW.filter(ee.Filter.eq('acq_key', key));
  var first = ee.Image(imgs.first());
  var cl = imgs.map(cloudBand).mosaic();
  var cloud = cl.reduceRegion({reducer: ee.Reducer.mean(), geometry: AOI, scale: 120, crs: CRS,
                               maxPixels: 1e9, bestEffort: true}).get('cloud');
  var cover = cl.mask().unmask(0).rename('cover').reduceRegion({reducer: ee.Reducer.mean(),
                               geometry: AOI, scale: 120, crs: CRS, maxPixels: 1e9, bestEffort: true}).get('cover');
  var date = ee.String(first.get('DATE_ACQUIRED'));
  return imgs.map(prep).mosaic().set({
    acq_key: key, sensor: first.get('sensor'), date: date,
    time_utc: first.get('SCENE_CENTER_TIME'),
    year: ee.Number.parse(date.slice(0, 4)),
    cloud_frac_aoi: cloud, cover_frac_aoi: cover
  });
}
var KEYS = RAW.aggregate_array('acq_key').distinct().sort();
var ALL_ACQ = ee.ImageCollection.fromImages(KEYS.map(makeAcq));
var ACQ = ALL_ACQ.filter(ee.Filter.lt('cloud_frac_aoi', MAX_CLOUD_AOI));

// Медиана / число наблюдений с защитой от пустой коллекции
function medianOrEmpty(col) {
  return ee.Image(ee.Algorithms.If(col.size().gt(0), col.median(),
                  ee.Image.constant([0, 0, 0]).rename(IDX).updateMask(0)));
}
function countOrZero(col) {
  return ee.Image(ee.Algorithms.If(col.size().gt(0), col.select('NDVI').count(),
                  ee.Image.constant(0))).unmask(0).rename('n');
}

function sampleOne(t) {
  var key = t.get('acq_key');
  var y = ee.Number(t.get('year'));
  var others = ACQ.filter(ee.Filter.neq('acq_key', key)).select(IDX);
  var same = others.filter(ee.Filter.eq('year', y));
  var wide = others.filter(ee.Filter.rangeContains('year', y.subtract(1), y.add(1)));
  var nSame = countOrZero(same), nWide = countOrZero(wide);
  var useWide = nSame.lt(2);
  var loo = ee.ImageCollection([medianOrEmpty(same).updateMask(useWide.not()),
                                medianOrEmpty(wide).updateMask(useWide)]).mosaic()
              .rename(['LOO_NDVI', 'LOO_NDBI', 'LOO_ALBEDO']);
  var stack = t.select(['LST_C', 'NDVI', 'NDBI', 'ALBEDO'], ['LST_C', 'SS_NDVI', 'SS_NDBI', 'SS_ALBEDO'])
    .addBands(loo)
    .addBands(nSame.where(useWide, nWide).rename('LOO_N'))
    .addBands(useWide.rename('LOO_WIDE'))
    .unmask(NODATA).toFloat();
  return stack.sampleRegions({collection: ee.FeatureCollection(POINTS_ID),
                              properties: ['pixel_id', 'x', 'y', 'urban_stratum'],
                              projection: P30, tileScale: 4})
    .filter(ee.Filter.gt('LST_C', NODATA + 1))
    .map(function (f) {
      return f.set({acq_key: key, sensor: t.get('sensor'), date: t.get('date'),
                    time_utc: t.get('time_utc'), cloud_frac_aoi: t.get('cloud_frac_aoi'),
                    cover_frac_aoi: t.get('cover_frac_aoi')});
    });
}

var COLUMNS = ['pixel_id', 'x', 'y', 'urban_stratum', 'acq_key', 'sensor', 'date', 'time_utc',
               'cloud_frac_aoi', 'cover_frac_aoi', 'LST_C', 'SS_NDVI', 'SS_NDBI', 'SS_ALBEDO',
               'LOO_NDVI', 'LOO_NDBI', 'LOO_ALBEDO', 'LOO_N', 'LOO_WIDE'];

// Сводка в консоль: какие акквизиции прошли фильтр облачности над окном
var summary = ALL_ACQ.map(function (img) {
  return ee.Feature(null, img.toDictionary(['acq_key', 'year', 'cloud_frac_aoi', 'cover_frac_aoi']));
});
print('Акквизиций всего:', ALL_ACQ.size(), 'прошли фильтр облачности (< 10% над окном):', ACQ.size());
print('По годам (прошли фильтр):', ACQ.aggregate_histogram('year'));
print('Все акквизиции (облачность и покрытие окна, доли):', summary);

YEARS.forEach(function (year) {
  var yearAcq = ACQ.filter(ee.Filter.eq('year', year));
  yearAcq.size().evaluate(function (n) {
    print(year + ': акквизиций = ' + n);
    if (n === 0 || !START_EXPORT) { return; }
    var fc = ee.FeatureCollection(yearAcq.toList(200).map(function (t) {
      return sampleOne(ee.Image(t));
    })).flatten();
    Export.table.toDrive({collection: fc, description: 'astana_scene_samples_' + year,
                          folder: FOLDER, fileFormat: 'CSV', selectors: COLUMNS});
  });
});
