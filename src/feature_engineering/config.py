"""Configuration constants and schema definitions for feature engineering."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEANED_DATA_DIR = PROJECT_ROOT / "data" / "cleaned"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_CLEANED_PATH = CLEANED_DATA_DIR / "train_cleaned.parquet"
TEST_CLEANED_PATH = CLEANED_DATA_DIR / "test_cleaned.parquet"
STORE_CLEANED_PATH = CLEANED_DATA_DIR / "store_cleaned.parquet"

TRAIN_PROCESSED_PATH = PROCESSED_DATA_DIR / "train_processed.parquet"
TEST_PROCESSED_PATH = PROCESSED_DATA_DIR / "test_processed.parquet"
TRAIN_SPLIT_PROCESSED_PATH = PROCESSED_DATA_DIR / "train_split_processed.parquet"
VAL_SPLIT_PROCESSED_PATH = PROCESSED_DATA_DIR / "val_split_processed.parquet"

# Forecasting horizon and strictly causal lag buffer.
# Test spans 2015-08-01 to 2015-09-17 (48 calendar days).
# Using a 49-day (7-week) minimum offset ensures every historical target-derived
# feature at date t only accesses observations <= t - 49 days, which is strictly
# before the start of any 48-day forecast or validation window.
FORECAST_HORIZON_DAYS = 48
CAUSAL_LAG_DAYS = 49
VAL_START_DATE = "2015-06-14"  # Final 48 days of train (2015-06-14 .. 2015-07-31)

# Fixed categorical vocabularies matching data/cleaned/ schema
STORE_TYPE_CATEGORIES = ("a", "b", "c", "d")
ASSORTMENT_CATEGORIES = ("a", "b", "c")
STATE_HOLIDAY_CATEGORIES = ("0", "a", "b", "c")
PROMO_INTERVAL_MAP = {
    "NoPromo2": (),
    "Jan,Apr,Jul,Oct": (1, 4, 7, 10),
    "Feb,May,Aug,Nov": (2, 5, 8, 11),
    "Mar,Jun,Sept,Dec": (3, 6, 9, 12),
}
PROMO_INTERVAL_CATEGORIES = (
    ("NoPromo2", "nopromo2"),
    ("Jan,Apr,Jul,Oct", "jan_apr_jul_oct"),
    ("Feb,May,Aug,Nov", "feb_may_aug_nov"),
    ("Mar,Jun,Sept,Dec", "mar_jun_sept_dec"),
)

# Exact calendar lags (multiples of 14 align with both DayOfWeek and biweekly Promo cycle)
CALENDAR_LAG_DAYS = (56, 70)
