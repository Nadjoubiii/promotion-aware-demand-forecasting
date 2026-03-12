# Data Ingestion and External Enrichment

## 1) Favorita source link
Kaggle competition data page:
- https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data

## 2) Download base dataset (manual)
Download Kaggle files and place these CSVs in `data/raw/favorita/`:
- `train.csv`
- `stores.csv`
- `items.csv`
- `holidays_events.csv`
- `oil.csv`
- `transactions.csv`

## 3) Download base dataset (automatic via Kaggle API)
Install dependencies and configure Kaggle API credentials:

```bash
pip install -r requirements.txt
```

Then place `kaggle.json` in your Kaggle config folder:
- Windows: `%USERPROFILE%\\.kaggle\\kaggle.json`
- Linux/macOS: `~/.kaggle/kaggle.json`

Run downloader inside `src/data/ingest_favorita.py`:

```bash
python -m src.data.ingest_favorita --raw-dir data/raw/favorita
```

## 4) Optional external sources
Create optional files in `data/external/`:
- `store_city_coords.csv` (for weather)
- `local_events.csv` (for custom event signals)

Use template files:
- `data/external/store_city_coords.sample.csv`
- `data/external/local_events.sample.csv`

## 5) Run pipeline
From repository root:

```bash
python -m src.data.build_training_table
```

Or auto-download from Kaggle first if files are missing:

```bash
python -m src.data.build_training_table --download-if-missing
```

Or specify custom paths:

```bash
python -m src.data.build_training_table --raw-dir data/raw/favorita --external-dir data/external --output-path data/processed/training_table.parquet
```

## Output
- Main dataset: `data/processed/training_table.parquet`
- Console validation summary:
  - total rows
  - negative units
  - null IDs
  - stockout candidate rows
  - missing segment dates
