# Astana UHI-DSS

Code and results for the article *Designing a GIS-Based Decision Support System for Urban Heat Island Analysis Using Big Data* (Gafur Khussanbayev, Astana IT University).

The study models summer land surface temperature (LST) over Astana, Kazakhstan, from open satellite data and tests the assumption behind the scenario layer of the planned decision support system with a before-and-after analysis of two greening projects. This repository holds:

- the Google Earth Engine scripts that export the satellite and ancillary layers;
- the Python scripts that build the dataset, train and evaluate the models, and run the greening analysis;
- the result tables and figures reported in the article.

## Study design in brief

| | |
|---|---|
| Area | 60 × 72 km window around Astana, 30 m grid, EPSG:32642 (origin 642000 E, 5706000 N; 2000 × 2400 pixels) |
| Period | June–August 2013–2025 (greening analysis: 2013–2026) |
| Target | LST anomaly within each Landsat 8/9 acquisition (pixel LST minus the acquisition median) |
| Predictors | NDVI, NDBI and albedo from leave-one-out summer composites that exclude the target scene; elevation; population density; distance to water and to green space |
| Models | random forest, XGBoost, SVR, plus two baselines (acquisition mean, linear regression) |
| Validation | spatial block cross-validation (4 × 4 km blocks, 5 folds, 2013–2023) and a temporal hold-out (2024–2025) |
| Greening sites | Botanical Garden and GreenLine boulevard, compared with 0–960 m rings and distant controls |

## Repository layout

```
gee/                    Google Earth Engine Code Editor scripts (JavaScript), run in order 01-09
scripts/                Python scripts, run in order 01-08
  tools/                helper for inspecting the dataset
data/case_study/        input tables of the greening analysis (exported by gee/08 and gee/09)
results/
  modelling/            main model run: metrics, tests, importance, errors (RQ1-RQ3)
  sentinel2_check/      Sentinel-2 vs Landsat predictors, 2019-2023 records (RQ2)
  case_study/           greening analysis with rings and distant controls (RQ4)
  case_sensitivity/     greening sensitivity analysis: distant controls only, stricter cloud filter
figures/                the four figures of the article
  supplementary/        additional diagnostic figures
requirements.txt        package list
requirements-lock.txt   exact package versions of the run reported in the article
CITATION.cff            citation metadata
```

## Where the article's tables and figures come from

| Article | File(s) |
|---|---|
| Table I (dataset) | `results/modelling/dataset_summary.csv`, `split_sizes.csv`, `dataset_by_year.csv`, `notes.txt` |
| Table II (model performance) | `results/modelling/metrics_summary.csv`, `metrics_folds.csv`, `tests.csv` |
| Table III, top (same scene vs composite) | `results/modelling/tests.csv`, `inflation.csv` |
| Table III, bottom (Sentinel-2 vs Landsat) | `results/sentinel2_check/metrics_summary.csv`, `tests.csv` |
| Table IV (greening sites) | `results/case_study/case_tests.csv`, `case_period_summary.csv`, `case_extra_tests.csv`, `case_year_bootstrap.csv` |
| Fig. 1 (anomaly classes) | `figures/fig_anomaly_classes.pdf`; counts in `results/modelling/confusion_matrix_counts.csv` |
| Fig. 2 (predictor importance) | `figures/fig_shap.pdf`; `results/modelling/shap_importance.csv`, `permutation_importance.csv` |
| Fig. 3 (block bias) | `figures/fig_block_bias.pdf`; `results/modelling/residuals_by_block.csv` |
| Fig. 4 (greening sites) | `figures/fig_case_study.pdf`; `results/case_study/case_by_acq.csv`, `case_profile.csv`, `case_s2_ndvi_by_year.csv` |
| Error analysis by LCZ and pixel group | `results/modelling/errors_by_group.csv` |
| Greening sensitivity analysis | `results/case_sensitivity/` |

## Running the pipeline

### 1. Google Earth Engine

Paste each script into the [Earth Engine Code Editor](https://code.earthengine.google.com/). Check the settings block at the top, run it once to preview, and set `START_EXPORT = true` only when the preview looks right. Exports go to Google Drive. An Earth Engine account is required, and the asset paths in the settings blocks must point to your own project.

| Script | Output |
|---|---|
| `01_landsat_composites.js` | seasonal Landsat composites (LST, NDVI, NDBI, albedo) for maps |
| `02_sentinel2_composites.js` | Sentinel-2 NDVI/NDBI composites, 2019–2025 |
| `03_modis_jja.js` | MODIS Terra/Aqua day and night summer LST, 4 bands per year |
| `04_static_layers.js` | WorldCover, DEM, GHSL population and building height, global LCZ map on the 30 m grid |
| `05_sample_points.js` | 100,000 stratified sample points (seed 42), saved as an asset; run once |
| `06_scene_samples.js` | step 1: acquisition table asset; step 2: per-acquisition samples, one CSV per year |
| `07_modis_daily_city.js` | daily city-mean MODIS LST, used to flag unrepresentative scenes |
| `08_case_sites.js` | first greening export: sites and distant controls, 2013–2025 (sensitivity analysis) |
| `09_case_rings.js` | main greening export: sites, 0–960 m rings and distant controls, 2013–2026, plus yearly Sentinel-2 NDVI |

### 2. Python

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt        # or requirements.txt for current versions

# modelling dataset and models (needs the Earth Engine exports in data/ and an OSM extract)
python scripts/01_osm_to_rasters.py kazakhstan-latest-free.gpkg data/astana_osm_30m.tif
python scripts/02_check_grid.py data/
python scripts/03_build_dataset.py data/ out/astana_dataset.parquet
python scripts/04_train_models.py out/astana_dataset.parquet out/feature_groups.json out/modelling \
       --configs main s2_check same_scene --seed 42
python scripts/04_train_models.py out/astana_dataset.parquet out/feature_groups.json out/sentinel2_check \
       --configs main_2019 s2_check --seed 42
python scripts/05_make_figures.py out/modelling out/figures out/modelling/pred_obs_sample.parquet

# greening analysis (runs from this repository alone)
python scripts/06_case_study.py data/case_study/astana_case_rings_lst.csv data/case_study/astana_case_rings_s2.csv \
       out/case_study --min-valid 0.8 --boot 5000 --seed 42
python scripts/07_case_extra_tests.py out/case_study/case_by_acq.csv out/case_study --boot 5000 --seed 42
python scripts/08_case_sensitivity.py data/case_study/astana_case_sites.csv out/case_sensitivity \
       --min-valid 0.8 --boot 5000 --seed 42
```

Each script prints its options with `-h`. `scripts/tools/peek_dataset.py` writes small CSV and text summaries of the parquet file for inspection.

The greening analysis can be reproduced from this repository alone, because its input tables are in `data/case_study/`. Rerunning steps 06–08 reproduces the tables in `results/case_study/` and `results/case_sensitivity/` to floating-point precision.

## Reproducibility notes

The reported run used Python 3.9.6 on an Apple MacBook Pro (M5 Pro, 24 GB RAM, macOS arm64, no GPU) with the package versions in `requirements-lock.txt`. All random steps use seed 42. Hyperparameters were fixed in advance and not tuned. The main model run (three configurations, five models, two validation schemes) took 26.2 minutes, the Sentinel-2 run 28.5 minutes and the greening analysis less than a minute.

The Earth Engine exports used in the article were made in October 2026. Because the Earth Engine catalogue and OpenStreetMap change over time, a rebuilt dataset may differ slightly from the one used in the article.

## Data

The modelling dataset (`astana_dataset.parquet`, 4,449,617 pixel-acquisition records, with `feature_groups.json`) is too large for this repository and is deposited separately in Zenodo: https://doi.org/10.5281/zenodo.XXXXXXX (ODbL 1.0, because two predictors are derived from OpenStreetMap).

Source data: USGS Landsat 8/9 Collection 2 Level 2, Copernicus Sentinel-2 L2A (harmonized), NASA MODIS MOD11A1/MYD11A1, ESA WorldCover v200, Copernicus DEM GLO-30, JRC GHSL, the global LCZ map (Demuzere et al., 2022) and OpenStreetMap contributors. Each keeps the licence of its provider.

## Licence

The code is released under the MIT licence (see `LICENSE`). The deposited dataset is released under ODbL 1.0, and the source data keep the licences of their providers.

## Citation

If you use this code, please cite the archived release (see `CITATION.cff`):

> G. Khussanbayev, *Astana UHI-DSS: code for urban heat island analysis in Astana*, release v1.0, Zenodo, 2026. https://doi.org/10.5281/zenodo.YYYYYYY

## Author

Gafur Khussanbayev, School of Artificial Intelligence and Data Science, Astana IT University, Astana, Kazakhstan.
