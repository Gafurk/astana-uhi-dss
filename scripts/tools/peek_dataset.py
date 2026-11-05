"""
peek_dataset.py  -  writes files from astana_dataset.parquet that can be opened in Excel / Numbers / VS Code.

Usage:  python scripts/tools/peek_dataset.py out/astana_dataset.parquet
Creates a folder peek/ next to the parquet file:
  summary.txt          summary: rows, points, acquisitions, missing values and ranges of all columns
  sample_1000.csv      1000 random full rows (all 45 columns)
  acquisitions.csv     one row per scene: date, sensor, cloud, coverage, rows, median LST, MODIS flag
  by_year.csv          by year: scenes, rows, median LST, share of LOO_WIDE, Sentinel-2 available
"""
import sys
from pathlib import Path
import pandas as pd


def main(path):
    p = Path(path)
    d = pd.read_parquet(p)
    out = p.parent / "peek"
    out.mkdir(exist_ok=True)

    d.sample(n=min(1000, len(d)), random_state=42).to_csv(out / "sample_1000.csv", index=False)

    acq = (d.groupby("acq_key")
             .agg(sensor=("sensor", "first"), date=("date", "first"), year=("year", "first"),
                  cloud_frac_aoi=("cloud_frac_aoi", "first"), cover_frac_aoi=("cover_frac_aoi", "first"),
                  rows=("pixel_id", "size"), urban_share=("urban_stratum", "mean"),
                  lst_median=("LST_C", "median"), loo_wide_share=("LOO_WIDE", "mean"),
                  modis_dev_sd=("modis_dev_sd", "first"), scene_flag_modis=("scene_flag_modis", "first"))
             .sort_values("date").round(3))
    acq.to_csv(out / "acquisitions.csv")

    yr = (d.groupby("year")
            .agg(acquisitions=("acq_key", "nunique"), rows=("pixel_id", "size"),
                 lst_median=("LST_C", "median"), loo_wide_share=("LOO_WIDE", "mean"),
                 s2_share=("S2_NDVI", lambda s: s.notna().mean()))
            .round(3))
    yr.to_csv(out / "by_year.csv")

    lines = [f"File: {p}", f"Rows: {len(d)}, columns: {d.shape[1]}",
             f"Points: {d['pixel_id'].nunique()}, scenes (acquisitions): {d['acq_key'].nunique()}",
             f"Years: {d['year'].min()}-{d['year'].max()}", "",
             f"{'column':16s} {'type':8s} {'empty %':>8s} {'min':>12s} {'median':>12s} {'max':>12s}"]
    for c in d.columns:
        s = d[c]
        na = 100 * s.isna().mean()
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            f = s.dropna()
            mn, md, mx = (f"{f.min():.3f}", f"{f.median():.3f}", f"{f.max():.3f}") if len(f) else ("-", "-", "-")
        else:
            mn = md = mx = "-"
        lines.append(f"{c:16s} {str(s.dtype)[:8]:8s} {na:8.2f} {mn:>12s} {md:>12s} {mx:>12s}")
    (out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:4]))
    print(f"\nDone, files in {out}/: summary.txt, sample_1000.csv, acquisitions.csv, by_year.csv")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
