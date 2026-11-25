# Feature Engineering Report

## Summary

Feature engineering is complete for the cleaned Rossmann Store Sales data. The
pipeline creates numeric, inference-time predictors, retains the training target
and prediction identifiers, and writes a chronological validation split without
training a model.

## Starting state and data inspection

- The repository already contained calendar feature logic, store/competition/
  Promo2 transformations, feature configuration, and cleaned-data notes.
- The package initializer referenced a `pipeline` module that did not exist;
  there was no runnable end-to-end feature pipeline or processed dataset yet.
- `train_cleaned.parquet` contains 1,017,209 rows and 9 columns, with sales
  dates from 2013-01-01 through 2015-07-31. It includes `Sales` and
  `Customers`.
- `test_cleaned.parquet` contains 41,088 rows and 8 columns, covering
  2015-08-01 through 2015-09-17. It has 856 stores and no `Sales` or
  `Customers`.
- `store_cleaned.parquet` contains 1,115 unique stores and 12 columns.
- The cleaned inputs have no missing values or duplicate rows; train and test
  have unique `Store`/`Date` keys and their store IDs resolve to metadata.
  Store histories are not uniformly dense, so historical features use exact
  calendar-date lookups rather than positional row shifts.
- The notebooks document the cleaning decisions and EDA findings. The
  configured chronological holdout is the final 48 days of training, starting
  2015-06-14.

## Final feature strategy

The pipeline produces 61 numeric predictors. `Store` and `Date` remain as
identifiers, and `Sales` remains only in the processed training and validation
datasets as the target. `Id` remains in processed test data for prediction
alignment. `Customers` is removed from training features and is never used.

### Feature groups

- **Existing daily inputs (4):** `DayOfWeek`, `Open`, `Promo`, and
  `SchoolHoliday`.
- **Calendar (20):** year, month, day-of-month, day-of-year, ISO week, quarter,
  days to month-end, weekend/Sunday and pay-window flags, and sine/cosine
  encodings for weekday, month, day-of-month, and day-of-year.
- **Store, competition, promotion, and holiday (28):** competition distance
  and log distance; distance/date-known indicators; competition-open status and
  duration; Promo2 status, start, active duration, refresh month and interval;
  fixed-vocabulary one-hot encodings for store type, assortment, state holiday,
  and Promo2 interval.
- **Causal sales history (9):** sales lags at 49, 56, and 70 calendar days,
  one missing-history flag per lag, and 28-day rolling mean, sample standard
  deviation, and observed-row count.

Calendar and store features are derived from prediction-date inputs and store
metadata. Fixed category vocabularies keep the train and test schemas aligned.

## Leakage prevention and chronological validation

- Historical sales features use only `train_cleaned.parquet` as their source;
  test rows never receive fabricated labels or customer counts.
- Every lag is joined by `(Store, exact calendar date)`. Missing source dates
  do not cause a different date's sales to be used.
- The minimum sales-history offset is 49 days. The rolling statistics end at
  that same offset, so their 28-day windows also precede the prediction date by
  at least 49 days. This keeps validation-window and test-window sales out of
  their own predictors for the configured 48-day horizon.
- Training and validation are split chronologically at 2015-06-14. The
  validation window ends on 2015-07-31. Because its historical features are
  causally offset, validation targets cannot enter validation predictors.
- `Sales` is preserved as the target but excluded from the predictor schema.
  `Customers` is excluded entirely. No target encoding, random split, or
  learned scaler/encoder is applied.

The lag offset is intentionally conservative and tied to the configured
48-day horizon. If a later forecasting setup uses a longer horizon or a
different validation boundary, review the offset and validation assumptions
before reusing these features.

## Missing values and rejected features

Input tables contain no missing values. If an exact historical lag is
unavailable for a store/date, its missing flag is set and its numeric value is
filled with zero. Rolling statistics use only observed historical rows; the
count feature communicates coverage, while undefined mean/std values are
filled with zero. Competition-date and distance missingness is retained through
the cleaning-stage indicators.

`Customers` was rejected because it is not available at inference time.
Shorter sales lags and rolling windows ending closer to the prediction date
were rejected because a multi-day forecast cannot know the intervening actual
sales. Target encodings and data-fitted transformations were not added; they
would require fold-aware fitting during chronological model validation.
Interactions and expanding statistics were not added without validation
evidence that they improve the eventual model.

## Outputs and checks

| Dataset | Rows | Columns | Description |
|---|---:|---:|---|
| `train_processed.parquet` | 1,017,209 | 64 | Full historical data with target |
| `test_processed.parquet` | 41,088 | 64 | Forecast period with `Id`, no target |
| `train_split_processed.parquet` | 963,689 | 64 | Training rows before 2015-06-14 |
| `val_split_processed.parquet` | 53,520 | 64 | Validation rows, 2015-06-14–2015-07-31 |

The pipeline and a separate read-back check verified:

- Processed row counts match the cleaned inputs; no `Store`/`Date` duplicates
  were introduced, and rows are chronologically ordered.
- Training `Sales` values are preserved exactly. Test `Id`, `Store`, and `Date`
  are preserved, and test contains neither `Sales` nor `Customers`.
- Train and test have the same ordered 61 predictor columns and matching
  dtypes; all predictors are numeric.
- There are no missing or infinite values in either processed dataset.
- The train/validation partitions are non-empty, chronological, and account
  for all 1,017,209 training rows.
- Regression tests cover exact-date causal lags, future-value independence,
  unavailable history, and rolling windows when the cutoff date itself is
  missing. A separate audit regenerated all validation history features using
  sales only through 2015-06-12, and all test history features using sales only
  through 2015-07-30; both matched the saved outputs exactly.
- Cleaned input files are read-only to this pipeline and their size and
  modification times are checked before and after writing outputs.
- Outputs are written to temporary Parquet files before replacement, allowing
  safe reruns without leaving partially written final datasets.
- Two consecutive executions of the documented command succeeded and produced
  identical SHA-256 hashes for all four Parquet outputs.

## Limitations

- Historical predictors are designed for the current 48-day forecast horizon;
  changing the horizon requires reassessing the causal offset and validation
  scheme.
- A zero-filled lag with a missing flag represents unavailable store history,
  not observed zero sales. Models should retain the missing flags.
- This stage does not fit a model, tune features, or assess feature importance.
  Category encoding, scaling, and any model-specific transformations belong in
  the later training workflow and must be fit using each chronological
  training fold only.

## Rerun

From the repository root, run:

```powershell
.\.venv\Scripts\python.exe -m src.feature_engineering
```

The pipeline reads only `data/cleaned/` and writes the four Parquet datasets to
`data/processed/`.
