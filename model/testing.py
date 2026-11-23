import shap
import pickle

from scipy.stats import t

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


predictors = ["Price_X", "Price_Y", "Price_Z"]
targets = ["Price_Equipo1", "Price_Equipo2"]
train_size = 3480
forecast_days = 5

data_path = "data/"
model_path = "model/trained_model/"
results_path = "results/testing/"


def read_model():
    """Reads the two fitted PLS models from the saved pickle.

    Parameters
    ----------
    None

    Returns
    -------
    dict
        Fitted models for Equipo 1 and Equipo 2.
    """
    with open(model_path + "best_model.pickle", "rb") as file:
        return pickle.load(file)


def testing_set(df):
    """Selects the final ten observations reserved for testing.

    Parameters
    ----------
    df : pandas.DataFrame
        Historical material and equipment prices.

    Returns
    -------
    pandas.DataFrame
        Five input observations and five target observations.
    """
    return df.tail(2 * forecast_days).reset_index(drop=True)


def predict(model, values):
    """Predicts five equipment prices using the fitted PLS model.

    Parameters
    ----------
    model : PLSRegression
        Fitted equipment model.
    values : numpy.ndarray
        Rows containing fifteen material prices.

    Returns
    -------
    numpy.ndarray
        Five predictions for each input row.
    """
    return np.asarray(model.predict(values))


def feature_importance(models, df, test, output_path="agent/outputs/"):
    """Computes signed SHAP values using the supplied material inputs.

    Parameters
    ----------
    models : dict
        Fitted PLS models.
    df : pandas.DataFrame
        Historical data retained for function compatibility.
    test : pandas.DataFrame
        Five material input observations followed by five target observations.
    output_path : str
        Directory where the image is saved.

    Returns
    -------
    None
        Saves the SHAP feature importance image.
    """
    # We select the five material input observations.
    inputs = test.iloc[:forecast_days]
    values = inputs[predictors].to_numpy().reshape(1, -1)

    # We construct the SHAP reference from the same five inputs.
    background = np.tile(inputs[predictors].mean().to_numpy(), forecast_days).reshape(1, -1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for i, target in enumerate(targets):
        model = models[target]

        # We calculate signed SHAP contributions using the saved PLS model.
        explainer = shap.Explainer(lambda x: predict(model, x), background, algorithm="permutation")
        shap_values = explainer(values, max_evals=2 * forecast_days * len(predictors) + 1)

        # We aggregate contributions by material across input and forecast days.
        importance = shap_values.values[0].reshape(forecast_days, len(predictors), forecast_days).sum(axis=(0, 2)) / forecast_days

        axes[i].bar(predictors, importance)
        axes[i].axhline(0, color="black", linewidth=0.8)
        axes[i].set_title(target)
        axes[i].set_ylabel("SHAP value")

    fig.tight_layout()
    fig.savefig(output_path + "feature_importance.png", dpi=150)
    plt.close(fig)


def prediction_intervals(models, test, output_path="agent/outputs/", show_actual=True):
    """Plots five PLS forecasts with approximate 95% confidence intervals.

    Parameters
    ----------
    models : dict
        Fitted PLS equipment models.
    test : pandas.DataFrame
        Five material input observations followed by five target dates.
    output_path : str
        Directory where the image is saved.
    show_actual : bool
        Whether actual equipment prices are displayed.

    Returns
    -------
    None
        Saves the confidence interval image.
    """
    # We separate the five input records from the five target dates.
    inputs = test.iloc[:forecast_days]
    actual = test.iloc[forecast_days:].reset_index(drop=True)

    values = inputs[predictors].to_numpy().reshape(1, -1)
    dates = actual["Date"].dt.strftime("%Y-%m-%d")

    # We load the original training data to estimate uncertainty.
    df = pd.read_csv(data_path + "historico_equipos.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # We select the same training window used by the saved PLS models.
    train = df.iloc[-train_size - 2 * forecast_days:-2 * forecast_days].reset_index(drop=True)

    # We reconstruct the historical fifteen-input windows.
    x = np.array([
        train[predictors].iloc[i:i + forecast_days].to_numpy().reshape(-1)
        for i in range(len(train) - 2 * forecast_days + 1)
    ])

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    for i, target in enumerate(targets):
        model = models[target]

        # We reconstruct the historical targets from the following five days.
        y = np.array([
            train[target].iloc[j + forecast_days:j + 2 * forecast_days].to_numpy()
            for j in range(len(train) - 2 * forecast_days + 1)
        ])

        # We calculate predictions using the saved model without retraining.
        predicted = predict(model, values)[0]
        fitted = predict(model, x)

        # We transform the predictors into the fitted PLS component space.
        scores = model.transform(x)
        test_scores = model.transform(values)

        # We construct the matrices for approximate confidence interval estimation.
        design = np.column_stack((np.ones(len(scores)), scores))
        test_design = np.column_stack((np.ones(len(test_scores)), test_scores))

        # We calculate residual variance for each forecast day.
        residuals = y - fitted
        degrees = len(design) - np.linalg.matrix_rank(design)
        variance = np.sum(residuals ** 2, axis=0) / degrees

        # We calculate the standard error of the estimated conditional mean.
        covariance = np.linalg.pinv(design.T @ design)
        leverage = np.einsum("ij,jk,ik->i", test_design, covariance, test_design)[0]

        critical = t.ppf(0.975, degrees)
        margin = critical * np.sqrt(variance * leverage)

        lower = predicted - margin
        upper = predicted + margin

        if show_actual:
            axes[i].plot(dates, actual[target], marker="o", label="Actual")

        axes[i].plot(dates, predicted, marker="o", label="Predicted")
        axes[i].fill_between(dates, lower, upper, alpha=0.25, label="Approximate 95% confidence interval")
        axes[i].set_title(target)
        axes[i].set_ylabel("Price")
        axes[i].legend()

    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(output_path + "prediction_intervals.png", dpi=150)
    plt.close(fig)


def psi(actual, predicted):
    """Computes PSI between actual and predicted price distributions.

    Parameters
    ----------
    actual : numpy.ndarray
        Actual equipment prices.
    predicted : numpy.ndarray
        Predicted equipment prices.

    Returns
    -------
    float
        Population Stability Index.
    """
    # We use three shared bins and smoothing to avoid zero probabilities.
    bins = np.histogram_bin_edges(np.concatenate((actual, predicted)), bins=3)
    actual_counts = np.histogram(actual, bins=bins)[0]
    predicted_counts = np.histogram(predicted, bins=bins)[0]

    actual_pct = (actual_counts + 0.5) / (len(actual) + 1.5)
    predicted_pct = (predicted_counts + 0.5) / (len(predicted) + 1.5)

    return float(np.sum((actual_pct - predicted_pct) * np.log(actual_pct / predicted_pct)))


def main():
    """Evaluates the saved PLS models using ten reserved observations.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Saves SHAP, confidence interval, PSI, and RMSE results.
    """
    # We load the fitted PLS models.
    print("1. Reading trained models...")
    models = read_model()

    # We load historical observations in chronological order.
    print("2. Loading historical data...")
    df = pd.read_csv(data_path + "historico_equipos.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # We select the ten reserved observations.
    print("3. Selecting the last ten observations for testing...")
    test = testing_set(df)

    # We use the first five reserved observations as material price inputs.
    inputs = test.iloc[:forecast_days]
    values = inputs[predictors].to_numpy().reshape(1, -1)

    # We use the last five reserved observations as actual equipment prices.
    actual = test.iloc[forecast_days:]

    # We save SHAP to the testing results folder.
    print("4. Computing SHAP feature importance...")
    feature_importance(models, df, test, output_path=results_path)

    # We save confidence intervals to the testing results folder.
    print("5. Computing 95% confidence intervals...")
    prediction_intervals(models, test, output_path=results_path)

    # We calculate PSI and RMSE using the five actual and predicted equipment prices.
    print("6. Computing PSI and RMSE...")
    records = []
    rmse_results = {}

    for target in targets:
        actual_prices = actual[target].to_numpy()
        predicted = predict(models[target], values)[0]

        value = psi(actual_prices, predicted)

        if value < 0.1:
            grade = "No significant drift"
        elif value < 0.25:
            grade = "Moderate drift"
        else:
            grade = "Significant drift"

        records.append({
            "Equipo": target,
            "PSI": round(value, 4),
            "Drift Grade": grade
        })

        # We calculate RMSE between actual and predicted equipment prices.
        rmse = np.sqrt(np.mean((actual_prices - predicted) ** 2))
        column = "rmse_equipo1" if target == targets[0] else "rmse_equipo2"
        rmse_results[column] = rmse

    psi_df = pd.DataFrame(records)
    print(psi_df.to_string(index=False))
    psi_df.to_csv(results_path + "psi_drift.csv", index=False)

    # We display and save the testing RMSE table.
    print("7. Testing RMSE...")
    rmse_df = pd.DataFrame([rmse_results])
    print(rmse_df.round(4).to_string(index=False))
    rmse_df.to_csv(results_path + "rmse_testing.csv", index=False)

    print("8. Testing completed.")


if __name__ == "__main__":
    main()
