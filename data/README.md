# Data

Dataset: **Give Me Some Credit** (Kaggle competition, [`GiveMeSomeCredit`](https://www.kaggle.com/c/GiveMeSomeCredit/data)),
included here so the project runs out of the box. The data belongs to the competition; see its rules on Kaggle.

What is committed and what is re-created:

```
data/
├── cs-training.csv        # committed. 150,000 rows with target: the only file used for modelling
├── cs-test.csv            # committed. 101,503 rows, empty target: only loaded into scoring.db
└── processed/
    ├── scoring.db                 # committed. SQLite read by the app (see below)
    ├── data_quality_report.xlsx   # committed. Data quality report written by prepare_data.py
    ├── raw_dev.csv                # not committed, re-created by prepare_data.py (or notebook 02)
    ├── raw_holdout.csv            # not committed, re-created by prepare_data.py (or notebook 02)
    ├── dev.csv                    # not committed, re-created by prepare_data.py (or notebook 03)
    └── holdout.csv                # not committed, re-created by prepare_data.py (or notebook 03)
```

`scoring.db` has one table, `customers` (146,516 rows), with the **raw** records the app scores:

| `source` | `customer_id` | Rows | Target |
|---|---|---|---|
| `holdout` | `HO_<id>` | 45,013 | known, so the app can validate the model |
| `test` | `TS_<id>` | 101,503 | empty (from `cs-test.csv`), scores only |

The app never reads `cs-test.csv` directly: it reads these customers from `scoring.db`. `prepare_data.py`
rebuilds the database from the two CSV files; without `cs-test.csv` it contains hold-out customers only.

Only `cs-training.csv` is used for modelling (10 predictors, binary target `SeriousDlqin2yrs`).
`cs-test.csv` is never used for training or validation. `Data Dictionary.xls` from Kaggle can sit in this
folder as a reference; no code reads it.
