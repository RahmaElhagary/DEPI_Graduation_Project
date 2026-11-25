"""Store metadata, competition dynamics, Promo2 schedule, and fixed-schema encodings."""

from datetime import date
import numpy as np
import pandas as pd

from src.feature_engineering.config import (
    ASSORTMENT_CATEGORIES,
    PROMO_INTERVAL_CATEGORIES,
    PROMO_INTERVAL_MAP,
    STATE_HOLIDAY_CATEGORIES,
    STORE_TYPE_CATEGORIES,
)


def _compute_promo2_start_dates(store_df: pd.DataFrame) -> pd.Series:
    """Convert valid (Promo2SinceYear, Promo2SinceWeek) pairs to Monday timestamps."""
    start_dates = pd.Series(pd.NaT, index=store_df.index, dtype="datetime64[ns]")
    active = (store_df["Promo2"] == 1) & (store_df["Promo2SinceYear"] > 0) & (store_df["Promo2SinceWeek"] > 0)
    if active.any():
        parsed = [
            pd.Timestamp(date.fromisocalendar(int(y), int(w), 1))
            for y, w in zip(
                store_df.loc[active, "Promo2SinceYear"],
                store_df.loc[active, "Promo2SinceWeek"],
            )
        ]
        start_dates.loc[active] = pd.DatetimeIndex(parsed)
    return start_dates


def add_store_features(df: pd.DataFrame, store_df: pd.DataFrame) -> pd.DataFrame:
    """Merge store metadata and compute competition, Promo2, and categorical encodings."""
    store_prep = store_df.copy()
    store_prep["promo2_start_date"] = _compute_promo2_start_dates(store_prep)

    # Precompute competition start date (day=1 placeholder since only month/year exist)
    comp_known = store_prep["CompetitionDateKnown"] == 1
    comp_dates = pd.Series(pd.NaT, index=store_prep.index, dtype="datetime64[ns]")
    if comp_known.any():
        comp_dates.loc[comp_known] = pd.to_datetime(
            {
                "year": store_prep.loc[comp_known, "CompetitionOpenSinceYear"].astype(int),
                "month": store_prep.loc[comp_known, "CompetitionOpenSinceMonth"].astype(int),
                "day": 1,
            }
        )
    store_prep["comp_start_date"] = comp_dates

    out = df.merge(
        store_prep,
        on="Store",
        how="left",
        validate="many_to_one",
    )

    # Competition features (masking zero placeholders via CompetitionDateKnown)
    out["CompetitionDistance"] = out["CompetitionDistance"].astype("float32")
    out["log_competition_distance"] = np.log1p(out["CompetitionDistance"]).astype("float32")
    out["CompetitionDistance_WasMissing"] = out["CompetitionDistance_WasMissing"].astype("int8")
    out["CompetitionDateKnown"] = out["CompetitionDateKnown"].astype("int8")

    known_mask = out["CompetitionDateKnown"] == 1
    raw_comp_months = (
        (out["Date"].dt.year - out["CompetitionOpenSinceYear"].astype("int32")) * 12
        + (out["Date"].dt.month - out["CompetitionOpenSinceMonth"].astype("int32"))
    )
    out["competition_is_open"] = (known_mask & (raw_comp_months >= 0)).astype("int8")
    comp_months_clean = np.where(known_mask, np.maximum(raw_comp_months, 0), 0).astype("float32")
    out["competition_months_open"] = comp_months_clean
    out["log_competition_months_open"] = np.log1p(comp_months_clean).astype("float32")

    # Promo2 features (time-varying relative to Promo2SinceWeek/Year & PromoInterval)
    out["Promo2"] = out["Promo2"].astype("int8")
    p2_started = (out["Promo2"] == 1) & out["promo2_start_date"].notna() & (out["Date"] >= out["promo2_start_date"])
    out["promo2_is_started"] = p2_started.astype("int8")

    days_since_p2 = (out["Date"] - out["promo2_start_date"]).dt.days.astype("float32")
    out["promo2_weeks_active"] = np.where(p2_started, np.maximum(days_since_p2 / 7.0, 0.0), 0.0).astype("float32")

    # PromoInterval refresh month matching (note 'Sept' spelling in cleaned data)
    month_series = out["Date"].dt.month.astype("int8")
    is_refresh = pd.Series(0, index=out.index, dtype="int8")
    months_since_refresh = pd.Series(-1, index=out.index, dtype="int8")

    p2_interval_str = out["PromoInterval"].astype("string")
    for interval_str, months_tuple in PROMO_INTERVAL_MAP.items():
        if not months_tuple:
            continue
        mask = p2_started & (p2_interval_str == interval_str)
        if not mask.any():
            continue
        m_vals = month_series.loc[mask]
        is_refresh.loc[mask] = m_vals.isin(months_tuple).astype("int8")
        # Minimum non-negative distance mod 12 to preceding refresh month -> {0, 1, 2}
        dist = np.min([(m_vals.to_numpy() - rm) % 12 for rm in months_tuple], axis=0)
        months_since_refresh.loc[mask] = dist.astype("int8")

    out["promo2_is_refresh_month"] = is_refresh
    out["promo2_months_since_refresh"] = months_since_refresh

    # Fixed-vocabulary one-hot encodings (schema-consistent across train & test)
    st_str = out["StoreType"].astype("string")
    for cat in STORE_TYPE_CATEGORIES:
        out[f"store_type_{cat}"] = (st_str == cat).astype("int8")

    as_str = out["Assortment"].astype("string")
    for cat in ASSORTMENT_CATEGORIES:
        out[f"assortment_{cat}"] = (as_str == cat).astype("int8")

    sh_str = out["StateHoliday"].astype("string")
    for cat in STATE_HOLIDAY_CATEGORIES:
        out[f"state_holiday_{cat}"] = (sh_str == cat).astype("int8")
    out["is_state_holiday"] = (sh_str != "0").astype("int8")

    for raw_cat, slug in PROMO_INTERVAL_CATEGORIES:
        out[f"promo_interval_{slug}"] = (p2_interval_str == raw_cat).astype("int8")

    # Drop raw string/categorical and intermediate helper columns so all features are numeric
    drop_cols = [
        "StoreType",
        "Assortment",
        "StateHoliday",
        "PromoInterval",
        "CompetitionOpenSinceMonth",
        "CompetitionOpenSinceYear",
        "Promo2SinceWeek",
        "Promo2SinceYear",
        "promo2_start_date",
        "comp_start_date",
    ]
    out = out.drop(columns=drop_cols)
    return out
