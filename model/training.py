
import pickle

import numpy as np
import pandas as pd

from sklearn.cross_decomposition import PLSRegression


predictors = ["Price_X", "Price_Y", "Price_Z"]
targets = ["Price_Equipo1", "Price_Equipo2"]
train_size = 3480
forecast_days = 5
folds = 8

data_path = "data/"
model_path = "model/trained_model/"
results_path = "results/training/"

# We set the number of PLS components independently for each equipment model.
n_components_equipo_1 = 11
n_components_equipo_2 = 11

# We standardize predictors and targets.
scale_equipo_1 = True
scale_equipo_2 = False

# We set the maximum number of iterations.
max_iter_equipo_1 = 500
max_iter_equipo_2 = 500

# We set the convergence tolerance.
tol_equipo_1 = 1e-4
tol_equipo_2 = 1e-4

# We preserve the original input data.
# Don't modify. Static. 
copy_equipo_1 = True
copy_equipo_2 = True


def baseline(train, validation):
    """Computes naive forecast RMSE using the last observed equipment prices.

    Parameters
    ----------
    train : pandas.DataFrame
        Historical equipment prices.
    validation : pandas.DataFrame
        Actual equipment prices for the next five observations.

    Returns
    -------
    numpy.ndarray
        Baseline RMSE for Equipo 1 and Equipo 2.
    """
    # We repeat the last observed prices across the five forecast days.
    predicted = np.tile(train[targets].iloc[-1].to_numpy(), (len(validation), 1))
    actual = validation[targets].to_numpy()

    # We calculate RMSE for both equipment models.
    rmse = np.sqrt(np.mean((actual - predicted) ** 2, axis=0))
    return rmse


def forecast_prices(train, validation):
    """Forecasts five equipment prices using only historical X, Y, and Z values.

    Parameters
    ----------
    train : pandas.DataFrame
        Historical prices for the current training window.
    validation : pandas.DataFrame
        Actual equipment prices for the next five observations.

    Returns
    -------
    tuple
        Five-day predictions and the two fitted PLS models.
    """
    # We construct 15-input windows using five consecutive X, Y, and Z observations.
    x = np.array([
        train[predictors].iloc[i:i + forecast_days].to_numpy().reshape(-1)
        for i in range(len(train) - 2 * forecast_days + 1)
    ])

    # We associate each input window with the following five Equipo 1 prices.
    y_1 = np.array([
        train[targets[0]].iloc[i + forecast_days:i + 2 * forecast_days].to_numpy()
        for i in range(len(train) - 2 * forecast_days + 1)
    ])

    # We associate each input window with the following five Equipo 2 prices.
    y_2 = np.array([
        train[targets[1]].iloc[i + forecast_days:i + 2 * forecast_days].to_numpy()
        for i in range(len(train) - 2 * forecast_days + 1)
    ])

    # We fit two separate PLS models using only X, Y, and Z as predictors.
    model_1 = PLSRegression(n_components=n_components_equipo_1, scale=scale_equipo_1, max_iter=max_iter_equipo_1, tol=tol_equipo_1, copy=copy_equipo_1).fit(x, y_1)
    model_2 = PLSRegression(n_components=n_components_equipo_2, scale=scale_equipo_2, max_iter=max_iter_equipo_2, tol=tol_equipo_2, copy=copy_equipo_2).fit(x, y_2)

    # We use the latest five historical X, Y, and Z observations for forecasting.
    future = train[predictors].iloc[-forecast_days:].to_numpy().reshape(1, -1)

    # We predict five future prices independently for both equipment models.
    predicted_1 = model_1.predict(future).reshape(-1)
    predicted_2 = model_2.predict(future).reshape(-1)

    predicted = np.column_stack((predicted_1, predicted_2))
    models = {targets[0]: model_1, targets[1]: model_2}
    return predicted, models


def main():
    """Trains two PLS models and evaluates a naive baseline using nine rolling folds.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Writes PLS and baseline RMSE reports and saves both PLS models.
    """
    # We load historical prices in chronological order.
    df = pd.read_csv(data_path + "historico_equipos.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    print(f"PLS components Equipo 1: {n_components_equipo_1}")
    print(f"PLS components Equipo 2: {n_components_equipo_2}")

    # We move a fixed 3480-row training window forward five rows per fold.
    records = []
    baseline_records = []

    for fold in range(folds):
        start = fold * forecast_days
        end = start + train_size
        train = df.iloc[start:end].reset_index(drop=True)
        validation = df.iloc[end:end + forecast_days]
        actual = validation[targets].to_numpy()
        predicted, models = forecast_prices(train, validation)

        rmse = np.sqrt(np.mean((actual - predicted) ** 2, axis=0))
        records.append({
            "fold": fold + 1,
            "training_observations": len(train),
            "train_start": df["Date"].iloc[start].strftime("%Y-%m-%d"),
            "train_end": df["Date"].iloc[end - 1].strftime("%Y-%m-%d"),
            "forecast_start": df["Date"].iloc[end].strftime("%Y-%m-%d"),
            "forecast_end": df["Date"].iloc[end + forecast_days - 1].strftime("%Y-%m-%d"),
            "rmse_equipo1": rmse[0],
            "rmse_equipo2": rmse[1]
        })
        print(f"Fold {fold + 1}: Equipo 1 RMSE = {rmse[0]:.4f}, Equipo 2 RMSE = {rmse[1]:.4f}")

        # We evaluate the naive baseline on the same fold.
        baseline_rmse = baseline(train, validation)
        baseline_record = records[-1].copy()
        baseline_record["rmse_equipo1"] = baseline_rmse[0]
        baseline_record["rmse_equipo2"] = baseline_rmse[1]
        baseline_records.append(baseline_record)
        print(f"Fold {fold + 1} Naive Baseline: Equipo 1 RMSE = {baseline_rmse[0]:.4f}, Equipo 2 RMSE = {baseline_rmse[1]:.4f}")

    # We save the existing three RMSE reports.
    report = pd.DataFrame(records)
    report.to_csv(results_path + "rmse_cross_validation.csv", index=False)
    summary = report[["rmse_equipo1", "rmse_equipo2"]]
    print(f"Mean RMSE: Equipo 1 = {summary['rmse_equipo1'].mean():.4f}, Equipo 2 = {summary['rmse_equipo2'].mean():.4f}")
    np.sqrt((summary ** 2).mean()).to_frame().T.to_csv(results_path + "rmse_total.csv", index=False)
    pd.DataFrame({
        "variance_equipo1": [summary["rmse_equipo1"].var(ddof=0)],
        "variance_equipo2": [summary["rmse_equipo2"].var(ddof=0)]
    }).to_csv(results_path + "rmse_variance.csv", index=False)

    # We save the three baseline RMSE reports.
    baseline_report = pd.DataFrame(baseline_records)
    baseline_report.to_csv(results_path + "rmse_baseline_cross_validation.csv", index=False)

    baseline_summary = baseline_report[["rmse_equipo1", "rmse_equipo2"]]
    print(f"Baseline Mean RMSE: Equipo 1 = {baseline_summary['rmse_equipo1'].mean():.4f}, Equipo 2 = {baseline_summary['rmse_equipo2'].mean():.4f}")

    np.sqrt((baseline_summary ** 2).mean()).to_frame().T.to_csv(results_path + "rmse_baseline_total.csv", index=False)

    pd.DataFrame({
        "variance_equipo1": [baseline_summary["rmse_equipo1"].var(ddof=0)],
        "variance_equipo2": [baseline_summary["rmse_equipo2"].var(ddof=0)]
    }).to_csv(results_path + "rmse_baseline_variance.csv", index=False)

    # We fit both PLS models on the last rolling window before the five reserved observations.
    final_end = len(df) - 2 * forecast_days
    train = df.iloc[final_end - train_size:final_end].reset_index(drop=True)
    _, models = forecast_prices(train, df.iloc[final_end:final_end + forecast_days])

    # We save only the fitted PLS models in one pickle.
    with open(model_path + "best_model.pickle", "wb") as file:
        pickle.dump(models, file)

    print("Final models saved: " + model_path + "best_model.pickle")


if __name__ == "__main__":
    main()
