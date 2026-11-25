"""End-to-end feature engineering pipeline for Rossmann sales forecasting."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_engineering.calendar_features import add_calendar_features
from src.feature_engineering.config import (
    PROCESSED_DATA_DIR,
    TEST_CLEANED_PATH,
    TEST_PROCESSED_PATH,
    TRAIN_CLEANED_PATH,
    TRAIN_PROCESSED_PATH,
    TRAIN_SPLIT_PROCESSED_PATH,
    VAL_SPLIT_PROCESSED_PATH,
    VAL_START_DATE,
)
from src.feature_engineering.historical_features import (
    add_historical_sales_features,
)
from src.feature_engineering.store_features import add_store_features


def _validate_inputs(
    train: pd.DataFrame, test: pd.DataFrame, stores: pd.DataFrame
) -> None:
    """Fail early when cleaned input tables do not match the documented schema."""
    required_train = {"Store", "Date", "Sales", "Customers", "DayOfWeek"}
    required_test = {"Id", "Store", "Date", "DayOfWeek"}
    required_stores = {"Store"}
    if not required_train.issubset(train.columns):
        raise ValueError(f"Training data is missing columns: {required_train - set(train.columns)}")
    if not required_test.issubset(test.columns):
        raise ValueError(f"Test data is missing columns: {required_test - set(test.columns)}")
    if not required_stores.issubset(stores.columns):
        raise ValueError("Store metadata is missing the Store key")
    if "Sales" in test.columns or "Customers" in test.columns:
        raise ValueError("Test data must not contain Sales or Customers")
    if train[["Store", "Date", "Sales"]].isna().any().any():
        raise ValueError("Training data has missing Store, Date, or Sales values")
    if test[["Id", "Store", "Date"]].isna().any().any():
        raise ValueError("Test data has missing Id, Store, or Date values")
    if stores["Store"].isna().any():
        raise ValueError("Store metadata has missing Store keys")
    if train.duplicated(["Store", "Date"]).any():
        raise ValueError("Training data has duplicate Store/Date rows")
    if test.duplicated(["Store", "Date"]).any():
        raise ValueError("Test data has duplicate Store/Date rows")
    if not stores["Store"].is_unique:
        raise ValueError("Store metadata must have unique Store keys")
    if not test["Id"].is_unique:
        raise ValueError("Test Id values must be unique")
    known_stores = set(stores["Store"])
    if not set(train["Store"]).issubset(known_stores):
        raise ValueError("Training data references stores without metadata")
    if not set(test["Store"]).issubset(known_stores):
        raise ValueError("Test data references stores without metadata")
    if train["Date"].max() >= test["Date"].min():
        raise ValueError("Training and test date ranges overlap or are out of order")


def _assert_finite_and_complete(frame: pd.DataFrame, name: str) -> None:
    if frame.isna().any().any():
        missing = frame.columns[frame.isna().any()].tolist()
        raise ValueError(f"{name} contains missing values: {missing}")
    for column in frame.select_dtypes(include="number").columns:
        if not np.isfinite(frame[column].to_numpy()).all():
            raise ValueError(f"{name}.{column} contains infinite values")


def _write_parquet(frame: pd.DataFrame, destination: Path) -> None:
    """Replace a parquet output only after its complete temporary write succeeds."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_pipeline() -> dict[str, object]:
    """Build, validate, and save full train/test data and chronological splits."""
    train_source = pd.read_parquet(TRAIN_CLEANED_PATH)
    test_source = pd.read_parquet(TEST_CLEANED_PATH)
    stores = pd.read_parquet(
        TRAIN_CLEANED_PATH.parent / "store_cleaned.parquet"
    )
    _validate_inputs(train_source, test_source, stores)
    cleaned_signatures = {
        path: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in (TRAIN_CLEANED_PATH, TEST_CLEANED_PATH, TRAIN_CLEANED_PATH.parent / "store_cleaned.parquet")
    }

    train_history = train_source.loc[:, ["Store", "Date", "Sales"]].copy()
    train = train_source.drop(columns=["Customers"])
    test = test_source.copy()

    train = add_calendar_features(train)
    test = add_calendar_features(test)
    train = add_store_features(train, stores)
    test = add_store_features(test, stores)
    train = add_historical_sales_features(train, train_history)
    test = add_historical_sales_features(test, train_history)

    train = train.sort_values(["Date", "Store"], kind="mergesort").reset_index(drop=True)
    test = test.sort_values(["Date", "Store"], kind="mergesort").reset_index(drop=True)

    train_predictors = [
        column for column in train.columns if column not in {"Sales", "Store", "Date"}
    ]
    test_predictors = [
        column for column in test.columns if column not in {"Id", "Store", "Date"}
    ]
    if train_predictors != test_predictors:
        raise ValueError("Training and test predictor columns are not identical and ordered")
    if "Customers" in train.columns or "Customers" in test.columns:
        raise ValueError("Customers must not be used as a predictive feature")
    if not train["Date"].is_monotonic_increasing or not test["Date"].is_monotonic_increasing:
        raise ValueError("Processed rows are not sorted chronologically")
    if train.duplicated(["Store", "Date"]).any() or test.duplicated(["Store", "Date"]).any():
        raise ValueError("Feature engineering introduced duplicate Store/Date rows")

    expected_target = (
        train_source.loc[:, ["Store", "Date", "Sales"]]
        .sort_values(["Date", "Store"], kind="mergesort")
        .reset_index(drop=True)
    )
    actual_target = train.loc[:, ["Store", "Date", "Sales"]]
    pd.testing.assert_frame_equal(actual_target, expected_target)
    expected_test_ids = (
        test_source.loc[:, ["Id", "Store", "Date"]]
        .sort_values(["Date", "Store"], kind="mergesort")
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        test.loc[:, ["Id", "Store", "Date"]], expected_test_ids
    )
    _assert_finite_and_complete(train, "processed train")
    _assert_finite_and_complete(test, "processed test")
    for column in ["Store", "Date", *train_predictors]:
        if train[column].dtype != test[column].dtype:
            raise ValueError(
                f"Train/test dtype mismatch for {column}: "
                f"{train[column].dtype} != {test[column].dtype}"
            )

    validation_start = pd.Timestamp(VAL_START_DATE)
    train_split = train.loc[train["Date"] < validation_start].copy()
    validation_split = train.loc[train["Date"] >= validation_start].copy()
    if train_split.empty or validation_split.empty:
        raise ValueError("Chronological train/validation split produced an empty partition")

    outputs = {
        TRAIN_PROCESSED_PATH: train,
        TEST_PROCESSED_PATH: test,
        TRAIN_SPLIT_PROCESSED_PATH: train_split,
        VAL_SPLIT_PROCESSED_PATH: validation_split,
    }
    for destination, frame in outputs.items():
        _write_parquet(frame, destination)

    for path, signature in cleaned_signatures.items():
        if (path.stat().st_size, path.stat().st_mtime_ns) != signature:
            raise RuntimeError(f"Cleaned input was unexpectedly modified: {path}")

    result = {
        "train_shape": train.shape,
        "test_shape": test.shape,
        "train_split_shape": train_split.shape,
        "validation_split_shape": validation_split.shape,
        "predictor_count": len(train_predictors),
        "predictors": train_predictors,
        "outputs": [str(path) for path in outputs],
    }
    print("Feature engineering completed and validation passed.")
    for name in ("train_shape", "test_shape", "train_split_shape", "validation_split_shape"):
        print(f"{name}: {result[name]}")
    print(f"predictor_count: {result['predictor_count']}")
    print(f"outputs: {', '.join(result['outputs'])}")
    return result


if __name__ == "__main__":
    run_pipeline()
