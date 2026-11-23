import sys

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


model_path = "model/"
sys.path.insert(0, model_path)

from testing import read_model, psi, predictors, targets, forecast_days, train_size, data_path


def main():
    """Displays PLS forecasts, SHAP, prediction ranges, and historical PSI.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Displays five equipment price predictions, SHAP, prediction ranges,
        and historical PSI results.
    """
    st.title("Construction Equipment Forecasting Agent DEMO")
    st.write("Hi. Please enter this week's X, Y, Z prices from Monday to Friday.")
    st.write("Note: The model forecasts equipment prices for Monday to Friday of the following week.")
    st.write("Note: Since this is a demo, if you want to predict other values, please fill the values again and click Forecast or refresh the website.")

    # We load the two fitted PLS models.
    models = read_model()

    # We create five rows containing only the material price inputs.
    values = pd.DataFrame({"Day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]})

    for column in predictors:
        values[column] = float("nan")

    st.subheader("Enter X, Y, Z Prices")
    values = st.data_editor(values, hide_index=True, num_rows="fixed", disabled=["Day"])

    if st.button("Forecast"):
        # We prepare the 15 material prices used by both PLS models.
        x = values[predictors].to_numpy().reshape(1, -1)

        # We calculate the dates for Monday to Friday of next week.
        today = pd.Timestamp.now(tz="America/Mexico_City").normalize().tz_localize(None)
        monday = today - pd.Timedelta(days=today.weekday())
        next_monday = monday + pd.Timedelta(days=7)
        dates = pd.bdate_range(start=next_monday, periods=forecast_days)

        # We predict the five prices for each equipment model.
        predictions = pd.DataFrame({"Date": dates})

        for target in targets:
            predictions[target] = models[target].predict(x).reshape(-1)

        st.subheader("Next Week Equipment Price Predictions")
        st.dataframe(predictions.round(4), hide_index=True)

        # We load historical data for SHAP and historical PSI evaluation.
        df = pd.read_csv(data_path + "historico_equipos.csv")
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        # We select the historical training window before the reserved observations.
        train = df.iloc[-train_size - forecast_days:-forecast_days]

        # We load the RMSE obtained from historical validation.
        rmse_report = pd.read_csv("results/training/rmse_total.csv")

        # We plot approximate prediction ranges using validation RMSE.
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

        for i, target in enumerate(targets):
            predicted = predictions[target].to_numpy()
            column = "rmse_equipo1" if i == 0 else "rmse_equipo2"
            rmse = rmse_report[column].iloc[0]
            margin = 1.96 * rmse

            axes[i].plot(dates, predicted, marker="o", label="Predicted")
            axes[i].fill_between(dates, predicted - margin, predicted + margin, alpha=0.25, label="Approximate 95% range")
            axes[i].set_title(target)
            axes[i].set_ylabel("Price")
            axes[i].legend()

        axes[-1].set_xlabel("Date")
        fig.tight_layout()

        st.subheader("Approximate 95% Prediction Ranges")
        st.pyplot(fig)
        plt.close(fig)

        # We create the historical reference values for SHAP.
        background = np.tile(train[predictors].mean().to_numpy(), forecast_days).reshape(1, -1)

        # We calculate signed SHAP contributions for both PLS models.
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        for i, target in enumerate(targets):
            model = models[target]

            explainer = shap.Explainer(lambda x: model.predict(x), background, algorithm="permutation")
            shap_values = explainer(x, max_evals=2 * forecast_days * len(predictors) + 1)

            importance = shap_values.values[0].reshape(forecast_days, len(predictors), forecast_days).sum(axis=(0, 2)) / forecast_days

            axes[i].bar(predictors, importance)
            axes[i].axhline(0, color="black", linewidth=0.8)
            axes[i].set_title(target)
            axes[i].set_ylabel("SHAP value")

        fig.tight_layout()

        st.subheader("SHAP Feature Importance")
        st.write("Note: Negative values reduce predicted prices; positive values increase them.")
        st.pyplot(fig)
        plt.close(fig)

        # We evaluate PSI using the five reserved historical observations.
        test = df.tail(forecast_days)
        historical_x = df[predictors].iloc[-2 * forecast_days:-forecast_days].to_numpy().reshape(1, -1)

        records = []

        for target in targets:
            actual = test[target].to_numpy()
            predicted = models[target].predict(historical_x).reshape(-1)

            value = psi(actual, predicted)
            grade = "No significant drift" if value < 0.1 else "Moderate drift" if value < 0.25 else "Significant drift"

            records.append({
                "Equipo": target,
                "PSI": round(value, 4),
                "Drift Grade": grade
            })

        st.subheader("Historical PSI Drift Evaluation")

        st.write(
            "Note: interpretation:\n\n"
            "- **PSI < 0.10:** No significant drift\n"
            "- **0.10 ≤ PSI < 0.25:** Moderate drift\n"
            "- **PSI ≥ 0.25:** Significant drift"
        )

        st.dataframe(pd.DataFrame(records), hide_index=True)
        st.caption("PSI uses the five reserved historical observations, not the new forecasts. Results are preliminary.")


if __name__ == "__main__":
    main()