# Data Cleaning Notes

## Project
**Retail Demand Forecasting & Sales Prediction System**

Dataset: Rossmann Store Sales (Kaggle)

## Original Dataset

Source: [Rossmann Store Sales — Kaggle](https://www.kaggle.com/c/rossmann-store-sales/data)

Original files: `train.csv`, `test.csv`, and `store.csv`.

## 1. Cleaned Datasets

The cleaned data is stored in three Parquet files in `data/cleaned/`.

| File | Shape | Description |
|---|---|---|
| `train_cleaned.parquet` | 1,017,209 × 9 | Historical daily sales data |
| `test_cleaned.parquet` | 41,088 × 8 | Data for the forecast period |
| `store_cleaned.parquet` | 1,115 × 12 | Store information |

**Note:** The test dataset has no `Sales` or `Customers` columns.

Load the files using pandas:

```python
import pandas as pd

train = pd.read_parquet("data/cleaned/train_cleaned.parquet")
test = pd.read_parquet("data/cleaned/test_cleaned.parquet")
store = pd.read_parquet("data/cleaned/store_cleaned.parquet")
```

Parquet was used to preserve data types and reduce file size. The `pyarrow` package is required.

## 2. Main Cleaning Steps

- Converted `Date` columns to datetime.
- Standardized `StateHoliday` values and data types.
- Handled missing values in `Open` and store information.
- Added two flags:
  - `CompetitionDistance_WasMissing`
  - `CompetitionDateKnown`
- Optimized data types to reduce memory usage.
- Checked for missing values, duplicates, invalid dates, and inconsistent store IDs.

## 3. Important Decisions

- **Closed days:** Kept in the dataset because they are part of the daily time series. Closed days have zero sales and customers.
- **Outliers:** Kept because extreme sales values may represent genuine high-volume stores.
- **Missing Promo2 information:** Filled with appropriate default values for stores that do not participate in Promo2.
- **Missing competition dates:** Used zero placeholders and a flag to identify unknown dates.
- **Missing `Open` values:** The 11 missing values in the test dataset were filled using store-specific historical opening patterns.
- **Zero sales on open days:** Kept during cleaning for further investigation during modeling.

## 4. Notes for the Team

### Ahmed — EDA & Visualization

- Filter `Open == 1` when analyzing sales patterns to avoid closed days distorting averages.
- Do not remove large stores or sales outliers without justification.
- Document which findings can be used as model features.

### Mohammed — Feature Engineering

- Merge store information into train and test using `Store`.
- Do not use `Customers` as a prediction feature because it is unavailable in the test dataset.
- Build the project's train/validation/test splits from `train_cleaned.parquet` using a time-based split.
- Use `test_cleaned.parquet` only for final forecasting.
- Handle zero placeholders carefully when calculating competition and Promo2 durations.
- Fit encoders and scalers on the training data only.

### Elsaeed — Model Training

- Closed days can be handled by predicting zero sales whenever `Open == 0`.
- Do not delete genuine sales outliers.
- If using a log transformation on sales, convert predictions back to the original scale before evaluation.

### Ali — Evaluation & Reporting

- Calculate MAPE only on rows where actual sales are greater than zero.
- Clearly document whether evaluation metrics are calculated on all rows or open days only.
