import unittest

import pandas as pd


df = pd.read_csv("data/historico_equipos.csv", parse_dates=["Date"])
prices = ["Price_X", "Price_Y", "Price_Z", "Price_Equipo1", "Price_Equipo2"]


class TestDataQuality(unittest.TestCase):
    """Checks consistency of the historical equipment data."""

    def test_nulls(self):
        """Checks that no values are missing.

        Parameters
        ----------
        self : TestDataQuality
            Test case.

        Returns
        -------
        None
            Reports whether the null check passed.
        """
        passed = not df.isna().any().any()
        print(f"Null values: {'PASSED' if passed else 'FAILED'}")
        self.assertTrue(passed)

    def test_data_types(self):
        """Checks that dates are dates and prices are numeric.

        Parameters
        ----------
        self : TestDataQuality
            Test case.

        Returns
        -------
        None
            Reports whether the data type check passed.
        """
        passed = pd.api.types.is_datetime64_any_dtype(df["Date"]) and all(pd.api.types.is_numeric_dtype(df[column]) for column in prices)
        print(f"Data types: {'PASSED' if passed else 'FAILED'}")
        self.assertTrue(passed)

    def test_duplicates(self):
        """Checks for repeated rows and dates.

        Parameters
        ----------
        self : TestDataQuality
            Test case.

        Returns
        -------
        None
            Reports whether the duplicate check passed.
        """
        passed = not df.duplicated().any() and df["Date"].is_unique
        print(f"Duplicate rows and dates: {'PASSED' if passed else 'FAILED'}")
        self.assertTrue(passed)

    def test_dates(self):
        """Checks that dates are sorted in chronological order.

        Parameters
        ----------
        self : TestDataQuality
            Test case.

        Returns
        -------
        None
            Reports whether the date order check passed.
        """
        passed = df["Date"].is_monotonic_increasing
        print(f"Date order: {'PASSED' if passed else 'FAILED'}")
        self.assertTrue(passed)


if __name__ == "__main__":
    unittest.main()
