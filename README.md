# Astana UHI-DSS

Code for the diploma project "Designing a GIS-Based Decision Support System for Urban Heat Island Analysis Using Big Data" (Astana IT University). The repository contains the Google Earth Engine scripts that build the dataset, the Python scripts that assemble it and train the models, and the analysis of two greening sites in Astana.

## Study design in short

The study area is Astana on a 30 m grid in EPSG:32642 (origin 642000 E, 5706000 N, 2000 × 2400 pixels). MODIS products use the matching 1 km grid (60 × 72). The modelling target is the land surface temperature anomaly within each Landsat 8/9 acquisition (pixel LST minus the acquisition median), June to August 2013–2025. Spectral predictors (NDVI, NDBI, albedo) come from leave-one-out composites built from the other acquisitions of the same summer, so the predictors never share a scene with the target. Models are evaluated with spatial block cross-validation (4 × 4 km blocks, 5 folds, 2013–2023) and a temporal hold-out (2024–2025).

## Repository layout

```
gee/        Google Earth Engine Code Editor scripts (JavaScript)
scripts/    Python scripts, run in the order listed below
results/    small result tables and figures used in the article
requirements.txt
requirements-lock.txt   exact package versions of the run reported in the article
```

Raw rasters, the sampled CSVs and the final parquet file are not stored here because of their size. See "Data" below. The small case-study input tables are included in `results/case_inputs/`, so the case-study analysis can be rerun from this repository alone.

## Pipeline

Earth Engine (paste each script into the Code Editor, check the settings block at the top, set `START_EXPORT = true` only when the preview looks right):

| Script | Output |
|---|---|
| `01_landsat_composites.js` | seasonal Landsat composites (LST, NDVI, NDBI, MNDWI, albedo) for maps |
| `04_sentinel2_composites.js` | Sentinel-2 NDVI/NDBI composites, 2019–2025 |
| `05_modis_jja.js` | MODIS Terra/Aqua day/night summer LST, 4 bands per year |
| `06_static_layers.js` | WorldCover, DEM, GHSL, LCZ on the 30 m grid |
| `07a_sample_points.js` | 100,000 stratified sample points (seed 42), saved as an asset |
| `07_scene_samples.js` | step 1: acquisition table asset; step 2: per-acquisition samples, one CSV per year |
| `08_modis_daily_city.js` | daily city-mean MODIS LST |
| `09_case_sites.js` | first case-study export: per-acquisition LST and NDVI for the Botanical Garden, GreenLine and rule-based control areas (2013–2025) |
| `10_case_recheck.js` | main case-study export (2013–2026): site, rings 0–960 m around each site and distant controls per acquisition, plus yearly Sentinel-2 NDVI |

Python (from the repository root, inside a virtual environment):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/osm_to_rasters.py          # OSM distance layers on the 30 m grid
python scripts/check_grid.py              # checks size, CRS and origin of every exported raster
python scripts/build_dataset.py           # joins samples, static, OSM and S2 layers -> parquet + feature_groups.json
python scripts/peek_dataset.py            # quick summaries of the parquet
python scripts/train_models.py data/astana_dataset.parquet data/feature_groups.json out/results \
       --configs main s2_check same_scene --seed 42
python scripts/train_models.py data/astana_dataset.parquet data/feature_groups.json out/results_s2 \
       --configs main_2019 s2_check --seed 42
python scripts/case_analysis.py data/astana_case_sites.csv out/case --min-valid 0.8 --boot 5000 --seed 42
python scripts/case_recheck.py data/astana_case_recheck_lst.csv data/astana_case_recheck_s2.csv out/case_recheck \
       --min-valid 0.8 --boot 5000 --seed 42
```

Each script prints its options with `-h`.

## Reproducibility

The reported run used Python 3.9.6 on an Apple M-series MacBook Pro (macOS, arm64) with the package versions in `requirements-lock.txt`. All random steps use seed 42. The full model run (`train_models.py`, three configurations, five models) took 26.2 minutes on that machine. Earth Engine results can change slightly if the data catalogue is reprocessed; the exports used in the article were made in October 2026. The case study uses summer 2026 scenes, one year beyond the 2013–2025 modelling period.

## Data

The dataset (parquet, feature groups, case CSV) is available at: DATA_LINK_HERE

Sources: USGS Landsat 8/9 Collection 2 Level 2, Copernicus Sentinel-2 L2A (harmonized), NASA MODIS MOD11A1/MYD11A1, ESA WorldCover v200, Copernicus DEM GLO-30, JRC GHSL, global LCZ map (Demuzere et al., 2022), OpenStreetMap contributors.

## Author

Gafur Khussanbayev, Astana IT University.
