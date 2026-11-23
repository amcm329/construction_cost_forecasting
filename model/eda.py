import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from statsmodels.tsa.api import VAR
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, kpss


predictors = ["Price_X", "Price_Y", "Price_Z"]
targets = ["Price_Equipo1", "Price_Equipo2"]
columns = predictors + targets
data_path = "data/"


def separated_files_common_dates():
    """Finds and saves dates shared by X, Y, and Z.

    Parameters
    ----------
    None

    Returns
    -------
    set
        Dates shared by the three CSV files.
    """
    # We count and save the dates shared by the three separate price files.
    x = pd.read_csv(data_path + "X.csv")
    y = pd.read_csv(data_path + "Y.csv", sep=";", decimal=",")
    z = pd.read_csv(data_path + "Z.csv")

    x["Date"] = pd.to_datetime(x["Date"])
    y["Date"] = pd.to_datetime(y["Date"], format="%d/%m/%Y")
    z["Date"] = pd.to_datetime(z["Date"])

    common_dates = set(x["Date"]) & set(y["Date"]) & set(z["Date"])
    pd.DataFrame({"Date": sorted(common_dates)}).to_csv("results/eda/separated_files_common_dates.csv", index=False)
    print(f"Dates common to X, Y, and Z: {len(common_dates)}", flush=True)
    return common_dates



def null_report(df):
    """Counts missing values in the equipment dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Historical equipment prices indexed by date.

    Returns
    -------
    pandas.DataFrame
        Missing values and percentages for each column.
    """
    nulls = df.isna().sum()
    return pd.DataFrame({"column": ["Date"] + df.columns.tolist(), "nulls": [df.index.isna().sum()] + nulls.tolist(), "null_pct": [df.index.isna().mean() * 100] + (df.isna().mean() * 100).tolist()})


def dtype_report(df):
    """Reports the data type of each column.

    Parameters
    ----------
    df : pandas.DataFrame
        Historical equipment prices indexed by date.

    Returns
    -------
    pandas.DataFrame
        Column names and data types.
    """
    return pd.DataFrame({"column": ["Date"] + df.columns.tolist(), "dtype": [str(df.index.dtype)] + df.dtypes.astype(str).tolist()})


def stationarity_report(df):
    """Computes ADF and KPSS tests on original prices.

    Parameters
    ----------
    df : pandas.DataFrame
        Historical equipment prices indexed by date.

    Returns
    -------
    pandas.DataFrame
        P-values and stationarity decisions for all price series.
    """
    records = []
    for column in columns:
        series = df[column]
        adf_p = adfuller(series, autolag="AIC", result_object=False)[1]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The test statistic is outside of the range of p-values available in the look-up table.*")
            kpss_p = kpss(series, nlags="auto", result_object=False)[1]
        records.append({"column": column, "adf_p": adf_p, "kpss_p": kpss_p, "adf_stationary": adf_p < 0.05, "kpss_stationary": kpss_p >= 0.05})
    return pd.DataFrame(records)


def stl_report(df):
    """Separates equipment prices into trends and seasonal components.

    Parameters
    ----------
    df : pandas.DataFrame
        Historical equipment prices indexed by date.

    Returns
    -------
    pandas.DataFrame
        Seasonal periods and trend and seasonal strengths.
    """
    records = []
    for target in targets:
        period = 5
        decomposition = STL(df[target], period=period).fit()
        residual_variance = decomposition.resid.var()
        trend_strength = max(0, 1 - residual_variance / (decomposition.resid + decomposition.trend).var())
        seasonal_strength = max(0, 1 - residual_variance / (decomposition.resid + decomposition.seasonal).var())
        records.append({"target": target, "period_days": period, "trend_strength": trend_strength, "seasonal_strength": seasonal_strength})
        figure = decomposition.plot()
        figure.suptitle(f"{target} (period={period} recorded days)")
        figure.tight_layout()
        figure.savefig(f"results/eda/stl_{target}.png", dpi=150)
        plt.close(figure)
    return pd.DataFrame(records)


def correlation_report(df):
    """Computes absolute correlations for every available daily lag.

    Parameters
    ----------
    df : pandas.DataFrame
        Historical equipment prices indexed by date.

    Returns
    -------
    pandas.DataFrame
        Correlations sorted by absolute value and then lag.
    """
    records = []
    for target in targets:
        y = df[target].to_numpy()
        for predictor in predictors:
            x = df[predictor].to_numpy()
            for lag in range(-len(df) + 2, len(df) - 1):
                # We compare values that remain aligned after shifting dates.
                if lag < 0:
                    actual, previous = y[:lag], x[-lag:]
                elif lag > 0:
                    actual, previous = y[lag:], x[:-lag]
                else:
                    actual, previous = y, x
                # We use NaN if either series has no variation, avoiding division by zero.
                correlation = np.nan if np.ptp(actual) == 0 or np.ptp(previous) == 0 else np.corrcoef(actual, previous)[0, 1]
                records.append({"target": target, "predictor": predictor, "lag_days": lag, "abs_correlation": abs(correlation)})
    return pd.DataFrame(records).sort_values(["abs_correlation", "lag_days"], ascending=[False, True])


def granger_report(df):
    """Tests Granger causality using the lag selected by BIC.

    Parameters
    ----------
    df : pandas.DataFrame
        Historical equipment prices indexed by date.

    Returns
    -------
    pandas.DataFrame
        Predictors, targets, selected daily lags, p-values, and significance decisions.
    """
    records = []
    for target in targets:
        for predictor in predictors:
            # We use a numeric index because missing dates prevent VAR from finding a regular frequency.
            pair = df[[target, predictor]].reset_index(drop=True)
            # We select the daily lag that minimizes BIC.
            lag = int(VAR(pair).select_order().selected_orders["bic"])
            print(f"Granger: {predictor} -> {target}: BIC selected {lag} days", flush=True)

            # We run Granger only when BIC selects at least one lag.
            if lag == 0:
                p_value = np.nan
            else:
                result = grangercausalitytests(pair, maxlag=[lag])
                p_value = result[lag][0]["ssr_ftest"][1]

            # We save only the requested output columns.
            records.append({"target": target, "predictor": predictor, "lag_days": lag, "p_value": p_value, "significant_0_05": bool(p_value < 0.05)})
    return pd.DataFrame(records).sort_values(["significant_0_05", "lag_days"], ascending=[False, True])


def date_range_report(df, common_dates):
    """Reports the date range of equipment prices and shared X, Y, Z dates.

    Parameters
    ----------
    df : pandas.DataFrame
        Equipment history indexed by date.
    common_dates : set
        Dates shared by X, Y, and Z.

    Returns
    -------
    pandas.DataFrame
        Minimum and maximum dates for each dataset.
    """
    # We compare the full date ranges without splitting the observations into folds.
    return pd.DataFrame([
        {"dataset": "historico_equipos.csv", "min_date": df.index.min().strftime("%Y-%m-%d"), "max_date": df.index.max().strftime("%Y-%m-%d")},
        {"dataset": "X_Y_Z_common_dates", "min_date": min(common_dates).strftime("%Y-%m-%d"), "max_date": max(common_dates).strftime("%Y-%m-%d")}
    ])


def main():
    """Runs all EDA checks and saves their reports.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Saves EDA results to the results folder.
    """
    print("Loading original equipment prices...", flush=True)
    df = pd.read_csv(data_path + "historico_equipos.csv", parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")
    print(f"Loaded {len(df)} rows.", flush=True)

    print("1/6: Checking null values...", flush=True)
    null_report(df).to_csv("results/eda/nulls.csv", index=False)
    print("1/6: Done.", flush=True)
    print(" ")

    print("2/6: Checking data types...", flush=True)
    dtype_report(df).to_csv("results/eda/types.csv", index=False)
    print("2/6: Done.", flush=True)
    print(" ")

    common_dates = separated_files_common_dates()
    print(" ")

    print("3/6: Running ADF and KPSS...", flush=True)
    stationarity_report(df).to_csv("results/eda/stationarity.csv", index=False)
    print("3/6: Done.", flush=True)
    print(" ")

    print("4/6: Running STL...", flush=True)
    stl_report(df).to_csv("results/eda/stl.csv", index=False)
    print("4/6: Done.", flush=True)
    print(" ")

    print("5/6: Running CCF across all daily lags...", flush=True)
    correlation_report(df).to_csv("results/eda/correlation.csv", index=False)
    print("5/6: Done.", flush=True)
    print(" ")

    print("6/6: Running standard Granger using BIC-selected daily lags...", flush=True)
    granger_report(df).to_csv("results/eda/granger.csv", index=False)
    print("6/6: Done.", flush=True)
    print(" ")

    print("Last test: Checking complete date ranges...", flush=True)
    ranges = date_range_report(df, common_dates)
    ranges.to_csv("results/eda/date_ranges.csv", index=False)
    print(ranges.to_string(index=False), flush=True)
    print("Last test: Done.", flush=True)
    print(" ")

    print("EDA completed. Reports saved in results/eda/.", flush=True)


if __name__ == "__main__":
    main()
