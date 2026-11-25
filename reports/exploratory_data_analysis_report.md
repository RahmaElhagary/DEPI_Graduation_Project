# Exploratory Data Analysis Report

## Purpose

The EDA explores the cleaned Rossmann store-sales data to understand sales
distribution and time patterns, compare sales across promotions and store types,
and identify candidate predictors for later modeling. It uses the cleaned
training, test, and store metadata datasets.

## Data loading and preparation

- Loaded `train_cleaned.parquet`, `test_cleaned.parquet`, and
  `store_cleaned.parquet`.
- The observed shapes were 1,017,209 × 9 (train), 41,088 × 8 (test), and
  1,115 × 12 (store metadata).
- Joined store metadata to train and test using `Store` with a left,
  many-to-one merge. Assertions confirmed that these joins did not change the
  number of daily records.
- For sales comparisons, filtered the training data to `Open == 1`. This keeps
  closed days, whose zero sales would otherwise affect comparisons, out of the
  sales distribution and average-sales analyses.

## Initial data inspection

The notebook inspected sample rows, column types, descriptive statistics, and
missing-value counts for the training data. It found 1,017,209 non-null values
in each of the nine columns and no missing values. `Date` was a datetime;
`StateHoliday` was categorical; the remaining columns used compact integer
types.

The training dates span 2013-01-01 through 2015-07-31. Across all training
rows, mean sales were 5,773.82 and mean customers were 633.15. These summaries
include closed days, so they are not the same as the open-day sales comparisons
below.

## Analyses and findings

### Sales distribution

The notebook plotted a histogram with a KDE and a boxplot of sales for open
stores. The observed distribution is right-skewed, with some stores and days
recording much higher sales than the average. The notebook keeps these
high-value observations: the cleaning review determined that many represent
genuine sales rather than errors.

### Sales over time

The notebook aggregated open-store sales by date and plotted daily total sales
from the beginning of 2013 to July 2015. It also plotted average sales by
calendar month, separated by year, to inspect possible trend and seasonal
patterns. The notebook does not record numerical summaries or explicit
conclusions from these two charts, so no specific trend or seasonal effect is
asserted here.

### Day of week

Average open-store sales were grouped by `DayOfWeek`. Monday averaged 8,216.07,
Saturday averaged 5,874.84, and Sunday averaged 8,224.72. The notebook then
checked the number of open stores and records for each day to see whether the
averages were comparable:

- Monday through Saturday each had 1,115 open stores.
- Sunday had only 33 open stores (3,593 records), compared with roughly
  134,644–144,058 records for each other weekday.

Therefore, Sunday’s high average is not directly comparable with the other
days. Among days with broad store coverage, Monday had the highest average and
Saturday the lowest.

### Promotion

For open stores, mean sales were 5,929.41 without a promotion and 8,228.28
with a promotion. The notebook describes the promotional-period average as
about 39% higher. This is a descriptive comparison; it does not isolate a
causal promotion effect or adjust for other differences between promotional
and non-promotional dates.

### Store type

After merging store metadata into the open-store records, the notebook
compared mean sales by `StoreType`:

| Store type | Mean sales |
|---|---:|
| A | 6,925.17 |
| B | 10,231.41 |
| C | 6,932.51 |
| D | 6,822.14 |

Store type B had the highest observed average sales in this comparison.

## Candidate modeling features

The notebook identifies the following as potentially useful features:

- `DayOfWeek`
- Month (derived from `Date`)
- `Promo`
- `StoreType`
- Year (derived from `Date`)

These are exploratory candidates, not a validated feature-importance ranking.
`Customers` is present in the training data but is unavailable in the test
data, so it should not be used as an inference-time predictor.

## Figures

When run, the notebook saves these plots under `reports/figures/`:

- `sales_distribution.png`
- `daily_sales_trend.png`
- `sales_by_weekday.png`
- `monthly_sales.png`
- `promo_analysis.png`
- `store_type_analysis.png`

## Scope and interpretation

The notebook performs descriptive analysis and visual comparisons. It does not
run statistical significance tests, establish causal effects, or fit and
evaluate a predictive model. Differences in averages should therefore be
treated as observations to investigate during validation, not proof that a
feature causes higher sales.
