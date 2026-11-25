"""Deterministic calendar, payday-cycle, and cyclical time features."""

import numpy as np
import pandas as pd


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar, German payday-cycle, and cyclical trigonometric features.

    All features are computed purely from ``Date`` and ``DayOfWeek`` with zero
    target or cross-row dependencies.
    """
    out = df.copy()
    dt = out["Date"].dt

    year = dt.year.astype("int16")
    month = dt.month.astype("int8")
    day_of_month = dt.day.astype("int8")
    day_of_year = dt.dayofyear.astype("int16")
    week_of_year = dt.isocalendar().week.astype("int8")
    quarter = dt.quarter.astype("int8")
    days_in_month = dt.days_in_month.astype("int8")
    days_to_month_end = (days_in_month - day_of_month).astype("int8")

    out["year"] = year
    out["month"] = month
    out["day_of_month"] = day_of_month
    out["day_of_year"] = day_of_year
    out["week_of_year"] = week_of_year
    out["quarter"] = quarter
    out["days_to_month_end"] = days_to_month_end

    # Operational and payday flags observed in German retail (Rossmann) sales
    dow = out["DayOfWeek"].astype("int8")
    out["is_weekend"] = (dow >= 6).astype("int8")
    out["is_sunday"] = (dow == 7).astype("int8")
    out["is_month_start_window"] = (day_of_month <= 5).astype("int8")
    out["is_month_end_window"] = (days_to_month_end <= 2).astype("int8")
    out["is_mid_month_pay_window"] = day_of_month.between(15, 18).astype("int8")

    # Smooth cyclical encodings
    dow_angle = 2.0 * np.pi * (dow.astype("float32") - 1.0) / 7.0
    month_angle = 2.0 * np.pi * (month.astype("float32") - 1.0) / 12.0
    dom_angle = (
        2.0
        * np.pi
        * (day_of_month.astype("float32") - 1.0)
        / days_in_month.astype("float32")
    )
    doy_angle = 2.0 * np.pi * (day_of_year.astype("float32") - 1.0) / 365.25

    out["dow_sin"] = np.sin(dow_angle).astype("float32")
    out["dow_cos"] = np.cos(dow_angle).astype("float32")
    out["month_sin"] = np.sin(month_angle).astype("float32")
    out["month_cos"] = np.cos(month_angle).astype("float32")
    out["dom_sin"] = np.sin(dom_angle).astype("float32")
    out["dom_cos"] = np.cos(dom_angle).astype("float32")
    out["doy_sin"] = np.sin(doy_angle).astype("float32")
    out["doy_cos"] = np.cos(doy_angle).astype("float32")

    return out
