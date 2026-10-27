# Data

The raw data is not included in this repository. It is public and free to download from the
**Medical Expenditure Panel Survey (MEPS)**, run by the Agency for Healthcare Research and Quality (AHRQ):
https://meps.ahrq.gov/mepsweb/data_stats/download_data_files.jsp

For each file below, download the **ASCII data file** (`.dat`) and its **SAS programming statements** (`...su.txt`),
and place them in these folders. Keep the file names exactly as listed, since the pipeline reads them by name.

| Folder | Files (2001–2006 / Panels 6–10) |
|---|---|
| `data/raw/longitudinal/` | `h71`, `h80`, `h86`, `h98`, `h106` |
| `data/raw/conditions/` | `H61`, `H69`, `H78`, `H87`, `H96`, `H104` |
| `data/raw/prescribed_meds/` | `H59A`, `H67A`, `H77A`, `H85A`, `H94A`, `H102A` |
| `data/raw/inpatient/` | `H59D`, `H67D`, `H77D`, `H85D`, `H94D`, `H102D` |
| `data/raw/er_visits/` | `H59E`, `H67E`, `H77E`, `H85E`, `H94E`, `H102E` |

Running `01_data_pipeline.Rmd` then creates `data/processed/final_for_modeling.csv`, which the notebooks and the dashboard use.
