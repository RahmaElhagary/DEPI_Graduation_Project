"""Causal sales-history features for fixed-horizon retail forecasting."""

import pandas as pd

from src.feature_engineering.config import (
    CALENDAR_LAG_DAYS,
    CAUSAL_LAG_DAYS,
)


def add_historical_sales_features(
    frame: pd.DataFrame,
    history: pd.DataFrame,
    rolling_window_days: int = 28,
) -> pd.DataFrame:
    """Add per-store sales lags and rolling statistics using historical rows only.

    Every value is looked up at an exact calendar date. The causal offset keeps
    the complete validation and test forecast windows out of the source period.
    """
    required = {"Store", "Date"}
    if not required.issubset(frame.columns) or not {"Store", "Date", "Sales"}.issubset(
        history.columns
    ):
        raise ValueError("frame needs Store and Date; history needs Store, Date, and Sales")
    if frame[["Store", "Date"]].isna().any().any():
        raise ValueError("Prediction rows cannot have missing Store or Date values")
    if history[["Store", "Date", "Sales"]].isna().any().any():
        raise ValueError("Sales history cannot contain missing Store, Date, or Sales values")
    if history.duplicated(["Store", "Date"]).any():
        raise ValueError("Sales history must have at most one row per Store and Date")
    if rolling_window_days < 1:
        raise ValueError("rolling_window_days must be a positive integer")

    offsets = (CAUSAL_LAG_DAYS, *CALENDAR_LAG_DAYS)
    if any(offset <= 0 for offset in offsets):
        raise ValueError("Historical sales offsets must be positive")

    out = frame.copy()
    hist = history.loc[:, ["Store", "Date", "Sales"]].sort_values(
        ["Store", "Date"], kind="mergesort"
    )
    sales_by_date = hist.set_index(["Store", "Date"])["Sales"]

    for offset in offsets:
        feature_name = f"sales_lag_{offset}d"
        lookup = pd.MultiIndex.from_arrays(
            [
                out["Store"].to_numpy(),
                (out["Date"] - pd.Timedelta(days=offset)).to_numpy(),
            ],
            names=["Store", "Date"],
        )
        values = sales_by_date.reindex(lookup)
        out[f"{feature_name}_missing"] = values.isna().to_numpy(dtype="int8")
        out[feature_name] = values.fillna(0).to_numpy(dtype="float32")

    daily_index = pd.MultiIndex.from_product(
        [
            hist["Store"].drop_duplicates().sort_values().to_numpy(),
            pd.date_range(hist["Date"].min(), hist["Date"].max(), freq="D"),
        ],
        names=["Store", "Date"],
    )
    daily_history = (
        hist.set_index(["Store", "Date"])["Sales"]
        .reindex(daily_index)
        .rename("Sales")
        .reset_index()
    )
    rolling = (
        daily_history.groupby("Store", sort=False)
        .rolling(f"{rolling_window_days}D", on="Date", min_periods=1)["Sales"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    rolling_by_date = rolling.set_index(["Store", "Date"])
    cutoff_lookup = pd.MultiIndex.from_arrays(
        [
            out["Store"].to_numpy(),
            (out["Date"] - pd.Timedelta(days=CAUSAL_LAG_DAYS)).to_numpy(),
        ],
        names=["Store", "Date"],
    )
    rolling_values = rolling_by_date.reindex(cutoff_lookup)
    prefix = f"sales_rolling_{rolling_window_days}d"
    out[f"{prefix}_mean"] = rolling_values["mean"].fillna(0).to_numpy(
        dtype="float32"
    )
    out[f"{prefix}_std"] = rolling_values["std"].fillna(0).to_numpy(
        dtype="float32"
    )
    out[f"{prefix}_count"] = rolling_values["count"].fillna(0).to_numpy(
        dtype="int16"
    )
    return out
