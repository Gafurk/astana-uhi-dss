// =====================================================================================
// 05_modis_jja.js   (Google Earth Engine Code Editor)
// MODIS Terra + Aqua, средняя летняя (июнь-август) температура поверхности, день и ночь, 1 км.
//
// Роль этого файла (Methods 2.4): ночной остров тепла на уровне районов и сравнение с Yu et al.
// на ряде MODIS (2013-2020). Проверку репрезентативности сцен Landsat делает 08_modis_daily_city.js.
// В 30-метровую модель MODIS НЕ входит.
//
// Terra (MOD11A1) и Aqua (MYD11A1) выгружаются РАЗДЕЛЬНО: время пролёта спутников со временем
// смещается, поэтому их не смешиваем заранее. Как объединять, решается потом по данным.
//
// Файл на выходе: astana_MODIS_JJA_<y0>_<y1>.tif на сетке 1 км (60 x 72 пикселя).
// По 4 канала на год, в таком порядке:
//   TERRA_DAY_<год>, TERRA_NIGHT_<год>, AQUA_DAY_<год>, AQUA_NIGHT_<год>
// 2003-2025: 23 года x 4 = 92 канала. Значения: средняя летняя LST, °C.
// Правило качества (одинаковое для дня и ночи), биты QC по каталогу GEE:
//   биты 0-1 <= 1  (LST посчитана: 0 = good, 1 = other quality)  И  биты 6-7 <= 1 (ошибка LST <= 2 K).
// Почему не только "good" (биты 0-1 = 0): в центре Астаны летом 2024 НИ ОДНОГО наблюдения с меткой
// good ни днём, ни ночью (диагностика 16.10.2026), фильтр выкидывал сам город.
// Если LST есть, а QC-канал пустой (masked): такие ночи встречаются часто (сырые значения getRegion,
// июнь 2024), считаем QC = 0, т.е. хорошее наблюдение. Это вывод из данных, в каталоге не описано.
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
var Y0 = 2003, Y1 = 2025;   // 2003: первый полный год с Terra и Aqua

var terra = ee.ImageCollection('MODIS/061/MOD11A1');
var aqua = ee.ImageCollection('MODIS/061/MYD11A1');

function qualityOk(img, qcBand) {
  var qc = img.select(qcBand).unmask(0);        // пустой QC там, где LST есть -> 0 (см. шапку)
  var flag = qc.bitwiseAnd(3);                  // биты 0-1
  var err = qc.rightShift(6).bitwiseAnd(3);     // биты 6-7
  return flag.lte(1).and(err.lte(1));
}
// Маска самого LST-канала сохраняется: где LST не посчитана, пиксель остаётся пустым.
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

print('Каналов в файле (ожидается ' + 4 * (Y1 - Y0 + 1) + '):', stack.bandNames().size());
print('Первые 8 каналов:', stack.bandNames().slice(0, 8));
Map.setCenter(71.43, 51.13, 9);
var PAL = ['blue', 'cyan', 'yellow', 'orange', 'red'];
Map.addLayer(stack.select('AQUA_NIGHT_2024'), {min: 8, max: 22, palette: PAL}, 'Aqua ночь 2024 °C');
Map.addLayer(stack.select('TERRA_NIGHT_2024'), {min: 8, max: 22, palette: PAL}, 'Terra ночь 2024 °C', false);
Map.addLayer(stack.select('TERRA_DAY_2024'), {min: 20, max: 45, palette: PAL}, 'Terra день 2024 °C', false);

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