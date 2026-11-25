import unittest

import numpy as np
import pandas as pd

from src.feature_engineering.historical_features import (
    add_historical_sales_features,
)


class HistoricalSalesFeaturesTests(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2020-01-01", periods=150, freq="D")
        self.history = pd.DataFrame(
            {
                "Store": np.ones(len(dates), dtype="int16"),
                "Date": dates,
                "Sales": np.arange(len(dates), dtype="int32"),
            }
        )

    def test_features_use_exact_dates_and_only_the_causal_past(self):
        target_date = self.history.loc[100, "Date"]
        predictions = pd.DataFrame(
            {"Store": np.array([1], dtype="int16"), "Date": [target_date]}
        )
        features = add_historical_sales_features(predictions, self.history)

        self.assertEqual(features.loc[0, "sales_lag_49d"], 51)
        self.assertEqual(features.loc[0, "sales_lag_56d"], 44)
        self.assertEqual(features.loc[0, "sales_lag_70d"], 30)
        expected_window = self.history.loc[24:51, "Sales"]
        self.assertEqual(features.loc[0, "sales_rolling_28d_count"], 28)
        self.assertAlmostEqual(
            features.loc[0, "sales_rolling_28d_mean"], expected_window.mean()
        )

        changed_future = self.history.copy()
        changed_future.loc[52:, "Sales"] = -1000
        recomputed = add_historical_sales_features(predictions, changed_future)
        pd.testing.assert_series_equal(
            features.iloc[0].drop(labels=["Store", "Date"]),
            recomputed.iloc[0].drop(labels=["Store", "Date"]),
            check_names=False,
        )

    def test_missing_history_is_flagged_and_filled(self):
        predictions = pd.DataFrame(
            {"Store": np.array([1], dtype="int16"), "Date": [pd.Timestamp("2020-01-06")]}
        )
        features = add_historical_sales_features(predictions, self.history)

        self.assertEqual(features.loc[0, "sales_lag_49d"], 0)
        self.assertEqual(features.loc[0, "sales_lag_49d_missing"], 1)
        self.assertEqual(features.loc[0, "sales_rolling_28d_count"], 0)
        self.assertEqual(features.loc[0, "sales_rolling_28d_mean"], 0)

    def test_rolling_window_uses_available_rows_when_cutoff_date_is_missing(self):
        history = self.history.drop(index=51).reset_index(drop=True)
        predictions = pd.DataFrame(
            {"Store": np.array([1], dtype="int16"), "Date": [self.history.loc[100, "Date"]]}
        )
        features = add_historical_sales_features(predictions, history)

        self.assertEqual(features.loc[0, "sales_lag_49d_missing"], 1)
        self.assertEqual(features.loc[0, "sales_rolling_28d_count"], 27)
        self.assertAlmostEqual(
            features.loc[0, "sales_rolling_28d_mean"],
            self.history.loc[24:50, "Sales"].mean(),
        )


if __name__ == "__main__":
    unittest.main()
