# Construction Equipment Cost Forecasting — Project Report

**Author:** Aarón Martín Castillo Medina  
**Method:** Partial Least Squares (PLS) Regression  
**Implementation:** Python, scikit-learn, statsmodels, SHAP and Streamlit  
**Execution:** Local Windows computer  

## 1. Project motivation and case explanation

A construction company needs to plan spending on two types of equipment. Its equipment prices change over time, making future costs difficult to estimate and causing budget differences. The company suspects that prices of three materials, **X, Y and Z**, can help predict the equipment costs.

The project builds a method to estimate future prices of **Equipo 1** and **Equipo 2** using historical material prices. The model helps the company anticipate price changes, explain which inputs matter and support later financial planning.

The source data contains **3,530 aligned records** from **4 January 2010 through 31 August 2023**. The dates are weekdays. The equipment file already contains the three input prices and both equipment prices.

The prototype currently forecasts **the following working week**, not several months. This is intentional: the weekly approach must first be tested over more periods before extending the horizon.

### Supplied files and scope

| Source | What it provides |
|---|---|
| `historico_equipos.csv` | Date, X/Y/Z prices and the recorded prices of both equipment types. The model uses this file. |
| `X.csv` | Separate X price history: 9,144 rows; 27 June 1988–4 April 2024. |
| `Y.csv` | Separate Y price history: 4,485 rows; 11 July 2006–12 September 2023. |
| `Z.csv` | Separate Z price history: 3,565 rows; 1 January 2010–31 August 2023. |

All four files share the **3,530 equipment dates**. Beyond the equipment file's last date, X contains **154 further rows**, Y contains **8**, and Z contains **none**. The later X and Y observations therefore do not provide a complete set of X, Y and Z inputs for further aligned equipment targets. The project keeps `historico_equipos.csv` as the modeling source and preserves the original files for reference.

## 2. Project structure and purpose of each file

The following is the actual structure:

```text
construction_cost_forecasting/
├── agent/
│   ├── app.py
│   └── outputs/
│       ├── feature_importance.png
│       └── prediction_intervals.png
├── data/
│   ├── historico_equipos.csv
│   ├── X.csv
│   ├── Y.csv
│   └── Z.csv
├── model/
│   ├── eda.py
│   ├── training.py
│   ├── testing.py
│   └── trained_model/
│       └── best_model.pickle
├── results/
│   ├── eda/
│   │   ├── correlation.csv
│   │   ├── date_ranges.csv
│   │   ├── granger.csv
│   │   ├── nulls.csv
│   │   ├── separated_files_after_date.csv
│   │   ├── separated_files_common_dates.csv
│   │   ├── stationarity.csv
│   │   ├── stl.csv
│   │   ├── stl_Price_Equipo1.png
│   │   ├── stl_Price_Equipo2.png
│   │   └── types.csv
│   ├── training/
│   │   ├── rmse_cross_validation.csv
│   │   ├── rmse_total.csv
│   │   ├── rmse_variance.csv
│   │   ├── rmse_baseline_cross_validation.csv
│   │   ├── rmse_baseline_total.csv
│   │   └── rmse_baseline_variance.csv
│   └── testing/
│       ├── feature_importance.png
│       ├── prediction_intervals.png
│       ├── psi_drift.csv
│       └── rmse_testing.csv
├── tests/
│   └── test_data_consistency.py
├── Architecture_Diagram_AWS_Proposal.pdf
├── README.md
├── Presentation_Construction_Cost_Forecasting.pptx
└── requirements.txt
```

### Main folders and files

| File / folder | Purpose |
|---|---|
| `data/` | Keeps the four original CSV price files. |
| `model/eda.py` | Runs data checks, stationarity tests, STL decomposition, lag correlations, Granger tests and date comparisons. |
| `model/training.py` | Fits two PLS models, runs eight rolling validation folds, compares a naive baseline and saves the models. |
| `model/testing.py` | Loads the saved models, evaluates the reserved final test and produces SHAP, approximate intervals, PSI and RMSE results. |
| `model/trained_model/best_model.pickle` | Contains the saved fitted PLS model for each equipment target. |
| `tests/test_data_consistency.py` | Checks missing values, types, duplicate rows/dates and date ordering. |
| `agent/app.py` | Agent (Local Streamlit) interface: enter material prices, forecast next week's equipment prices, display SHAP and approximate intervals. |
| `agent/outputs/` | Stores the SHAP and confidence interval images produced by the agent. |
| `results/` | Stores generated EDA, model training and final test reports. |
| `requirements.txt` | Lists the eight Python package names; this file does not contain version pins in the ZIP. |
| `README.md` | Idem. |
| `Architecture_Diagram_AWS_Proposal.pdf` | Proposed cloud architecture and workflow, not a deployed service. |
| `Presentation_Construction_Cost_Forecasting.pptx` | Business context, EDA, model design, charts, results, cloud proposal and next steps. |

### Meaning of the result files

These are the purposes of the saved outputs; no changes are required to these files.

**`results/eda/`**

| File | Meaning |
|---|---|
| `nulls.csv` | Missing value count and percentage per column. |
| `types.csv` | Date and price column data types. |
| `stationarity.csv` | ADF and KPSS p-values and decisions for each price series. |
| `stl.csv` | Estimated trend and seasonal strength for both equipment prices. |
| `stl_Price_Equipo1.png` | STL decomposition plot of Equipo 1. |
| `stl_Price_Equipo2.png` | STL decomposition plot of Equipo 2. |
| `correlation.csv` | Absolute lag correlations between each material price and equipment price. |
| `granger.csv` | Selected lag, p-value and significance for each material–equipment pair. |
| `date_ranges.csv` | First and last dates in the equipment file and common material dates. |
| `separated_files_common_dates.csv` | Dates shared by X, Y and Z in the separate CSVs. |
| `separated_files_after_date.csv` | Count of shared material dates after the cutoff used in the EDA check (24 August 2023). |

**`results/training/`**

| File | Meaning |
|---|---|
| `rmse_cross_validation.csv` | PLS RMSE per fold and equipment type, with training and forecast dates. |
| `rmse_total.csv` | Square root of the mean squared fold RMSE, computed by the code. This is **not** the arithmetic mean of fold RMSE. |
| `rmse_variance.csv` | Variance of PLS RMSE across the eight folds. |
| `rmse_baseline_cross_validation.csv` | Naive baseline RMSE for the same folds. |
| `rmse_baseline_total.csv` | Square root of the mean squared baseline fold RMSE. |
| `rmse_baseline_variance.csv` | Variance of naive baseline RMSE across folds. |

**`results/testing/`**

| File | Meaning |
|---|---|
| `feature_importance.png` | SHAP contributions of X, Y and Z for the final historical test. |
| `prediction_intervals.png` | Actual and predicted equipment prices with approximate 95% confidence bands. |
| `psi_drift.csv` | PSI results from the reserved historical test data. |
| `rmse_testing.csv` | Final test RMSE for both equipment models. |

**`agent/outputs/`** holds the app's versions of `feature_importance.png` and `prediction_intervals.png`, separately from the testing results.

## 3. Dependencies and exact package versions

The following versions were supplied from the Python environment used to run this project. They are included **exactly as provided**.

| Package | Installed version | Main purpose |
|---|---|---|
| `numpy` | **2.4.6** | Numerical arrays and RMSE calculations. |
| `pandas` | **3.1.1** | CSV loading, dates and tables. |
| `scipy` | **1.17.1** | Student t critical values for approximate intervals. |
| `statsmodels` | **0.15.0** | ADF, KPSS, STL, VAR and Granger analysis. |
| `matplotlib` | **3.11.2** | Plots for EDA, SHAP and intervals. |
| `scikit-learn` | **1.9.1** | `PLSRegression`. |
| `shap` | **0.51.0** | SHAP explanations of model predictions. |
| `streamlit` | **1.67.0** | Local web interface. |

## 4. Python environment with pyenv — Windows

Use **PowerShell**. Install `pyenv-win` once, then open a **new PowerShell window** before entering the remaining commands. If you are using Linux-based systems, use their equivalences.

**Install pyenv-win (once):**

```powershell
Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"
.\install-pyenv-win.ps1
```

Official instructions: https://github.com/pyenv-win/pyenv-win/blob/master/docs/installation.md

**Open PowerShell in the extracted `construction_cost_forecasting` root and run:**

```powershell
# We check pyenv.
pyenv --version

# We install and select Python 3.11.9.
pyenv install 3.11.9
pyenv local 3.11.9
python --version

# We create and activate the project environment.
python -m venv .venv
.\.venv\Scripts\activate

# We install the exact package versions used in this project.
python -m pip install requirements.txt

# We list the versions installed in the environment.
python -m pip list
```

To activate the environment on a later day, run `.\.venv\Scripts\activate` from the project root. Use `deactivate` when finished. If PowerShell prevents script activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal before activation.

## 5. Run the project locally

All commands below run from **the project root**, not from the `model/`, `tests/` or `agent/` folder. The project scripts use paths relative to this root.

```powershell
# We activate the virtual environment.
.\.venv\Scripts\activate

# We check the original equipment data.
python -m unittest discover -s tests -v

# We create the EDA reports.
python model/eda.py

# We train and save the two PLS models and compare the baseline.
python model/training.py

# We run the reserved final test.
python model/testing.py

# We start the local Streamlit interface.
streamlit run agent/app.py
```

Streamlit opens a browser page, usually at **`http://localhost:8501`**. Enter five weekdays of X, Y and Z prices, click **Forecast**, and review the following week's equipment price predictions, SHAP and approximate confidence intervals. The app does not request actual future equipment prices, so it does not compute future RMSE or PSI.

The app reads the trained pickle file and uses the analysis functions from `model/testing.py`. Its plots are saved in `agent/outputs/`, while running `model/testing.py` writes to `results/testing/`.

To stop the local app, press **Ctrl+C** in its terminal. Then run:

```powershell
deactivate
```

Running training or EDA again replaces the reports at their documented output paths. The **final test was run once for model selection independence**; repeated checks of the holdout should not be used to tune the model.

## 6. Computer used to run the project

The characteristics of the local machine used for this exercise are:

| Component | Reported specification |
|---|---|
| Operating system | Windows version `10.0.26100.1`. |
| Processor | AMD Ryzen 7 4800H with Radeon Graphics. |
| Physical cores | 8. |
| Logical processors | 16. |
| Installed RAM | 32.0 GB (31.4 GB total usable). |