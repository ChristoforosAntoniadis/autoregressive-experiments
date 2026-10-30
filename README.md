# Autoregressive Experiments

Experiments on autoregressive approaches, from traditional econometrics (SARIMA, VAR / VECM, ARCH / GARCH) to neural networks, benchmarked on the same dataset: the Federal Reserve's total assets and the monetary base, monthly from December 2002.

![Python](https://img.shields.io/badge/python-3.12-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

---

## Plan

- **Univariate (SARIMA, then ARCH / GARCH):** stationarity tests (ADF / KPSS on levels and first differences), ACF / PACF, and SARIMA order selection by AIC / AICc / BIC, residual checks of the selected models (Ljung-Box, Breusch-Godfrey), volatility checks (Engle's ARCH-LM, with and without a variance break), GARCH(1,1), EGARCH and GJR-GARCH on the residuals are done on the training split. Every model's settings are then tuned by time-aware random search on the training split (expanding cross-validation folds in time order), and its rolling-origin forecasts (expanding window, one fold per monthly origin) are scored on validation and then once on test, against naive, drift and seasonal-naive baselines: MAE, MSE, RMSE, MAPE, MASE and Diebold-Mariano tests at 1–12 months, plus quantile forecast intervals (80% / 95% coverage, pinball loss) with and without GARCH; see [`reports/`](reports/).
- **Multivariate (VAR / VECM):** lag selection and the Johansen cointegration test are done on the training split; no cointegration was found, so the system is a VAR on the first differences. VAR(5) and a rank-1 VECM(5) are estimated, stable and pass the residual checks; Granger causality, impulse responses and the variance decomposition show a one-way link, from total assets to the monetary base. The tuned VAR and VECM are evaluated with the same harness, on validation and once on test, including a Diebold-Mariano comparison against SARIMA, as is SARIMAX (the tuned SARIMA plus the other series' lagged monthly changes).
- **Lag-feature ML:** Ridge, Lasso, random forest, XGBoost and k-nearest neighbours on the last monthly changes of both series, every setting (the lag count included) tuned the same way, forecasting recursively through the same harness.
- **Smoothing baselines and a combination:** a moving average and exponential smoothing (trend, damping, season, Box-Cox and initialization all tuned), and a weighted average of SARIMA and the naive forecast (weight tuned), all through the same harness.
- **Neural networks:** a multilayer perceptron (scikit-learn's MLPRegressor) and two networks written by hand in NumPy (A1: one sigmoid hidden layer; A2: three ReLU hidden layers; SGD, momentum, AdaBelief or Muon), and a TensorFlow (Keras) and a PyTorch multilayer perceptron (A1 or A2, tuned with Optuna), on the same lag features and evaluated the same way; next, models on lookback windows (MLP, N-BEATS, DLinear) and recurrent networks (LSTM, GRU), evaluated on the same chronological 60 / 20 / 20 split.

## Quickstart

The project follows the [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/) layout, and a `Makefile` wraps the common commands:

```bash
make create_environment           # python -m venv .venv
make requirements                 # pinned dependencies + the package + dev tools (pytest, ruff, jupyter)

make data                         # download the Kaggle dataset into data/raw/ (already committed)
make features                     # clean -> data/interim/, first-difference -> data/processed/
make plots                        # the cleaned series -> reports/figures/fed_balance_sheet.png
make diagnostics                  # ADF / KPSS, ACF / PACF and SARIMA information criteria -> reports/
make residuals                    # Ljung-Box / Breusch-Godfrey on the selected models -> reports/
make volatility                   # ARCH-LM and a variance-break check on those models -> reports/
make garch                        # GARCH(1,1) vs constant variance on those models, bootstrap checks (~3 min) -> reports/
make asymmetric-garch             # EGARCH and GJR-GARCH vs that GARCH, bootstrap checks (~12 min) -> reports/
make tune                         # tune every model by time-aware random search on train (~50 min) -> models/, reports/
make evaluate                     # rolling-origin forecasts of every tuned model on validation and test (~60 min)
make var                          # VAR lag order and Johansen cointegration test -> reports/
make var-checks                   # fit VAR / VECM candidates; stability and Portmanteau checks -> reports/
make var-analysis                 # Granger causality, impulse responses, variance decomposition -> reports/
make all                          # every step above, in order (~2.25 h)

make test                         # unit tests
make lint                         # check style and lint with ruff (changes nothing)
make typecheck                    # check the type hints with mypy (changes nothing)
make format                       # apply ruff's fixes and formatting
make help                         # list every command
```

`make` isn't installed on Windows by default. Install it with `winget install ezwinports.make`, then open a new terminal. Without `make`, run the same steps directly:

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
pip install -e ".[dev]"

python -m autoregressive_experiments.features
python -m autoregressive_experiments.plots
python -m autoregressive_experiments.diagnostics
python -m autoregressive_experiments.residual_checks
python -m autoregressive_experiments.volatility_checks
python -m autoregressive_experiments.volatility_models
python -m autoregressive_experiments.asymmetric_garch
python -m autoregressive_experiments.forecast_evaluation
python -m autoregressive_experiments.var_diagnostics
python -m autoregressive_experiments.var_models
python -m autoregressive_experiments.var_analysis
python -m pytest
```

When run directly (not through `make`), set `MPLBACKEND=Agg` to stop plot windows from opening.

## Project structure

```
autoregressive-experiments/
├── Makefile                     # make data / features / plots / diagnostics / test / lint / ...
├── pyproject.toml               # package metadata, dev tools, pytest and ruff settings
├── requirements.txt             # pinned runtime dependencies
├── data/
│   ├── external/                # (empty) data from third-party sources
│   ├── interim/                 # cleaned data, regenerated by make features (not committed)
│   ├── processed/               # first-differenced data, regenerated by make features (not committed)
│   └── raw/                     # the original Kaggle CSVs, never modified
├── docs/                        # (empty) detailed documentation
├── models/                      # tuned_hyperparameters.json: every model's tuned settings
├── notebooks/                   # (empty) Jupyter notebooks
├── references/                  # (empty) data dictionaries, manuals, papers
├── reports/                     # stationarity, ACF / PACF and information-criteria reports
│   └── figures/                 # the series and its ACF / PACF
├── autoregressive_experiments/  # the source package
│   ├── config.py                # project paths
│   ├── configs/
│   │   ├── configurables.py     # every setting's type and allowed values, one dataclass per step
│   │   ├── config_loader.py     # reads and validates hyperparameters.json
│   │   └── hyperparameters.json # every setting's value and its reasoning
│   ├── runlog.py                # run logs in logs/, phase timings, progress lines
│   ├── dataset.py               # downloads the raw dataset
│   ├── features.py              # cleaning, first-differencing, chronological 60 / 20 / 20 split
│   ├── plots.py                 # figure of the cleaned series
│   ├── diagnostics.py           # ADF / KPSS, ACF / PACF, SARIMA information criteria
│   ├── residual_checks.py       # Ljung-Box and Breusch-Godfrey on the selected models' residuals
│   ├── volatility_checks.py     # ARCH-LM, with and without an estimated variance break
│   ├── volatility_models.py     # GARCH(1,1) (normal and Student-t) vs constant variance, by BIC; residual checks by parametric bootstrap
│   ├── asymmetric_garch.py      # EGARCH and GJR-GARCH (Student-t), plain or break-adjusted, by BIC
│   ├── tuning.py                # time-aware random search of every model's settings on train
│   ├── forecast_evaluation.py   # rolling-origin forecasts of every tuned model vs baselines, validation and test
│   ├── ml_forecasting.py        # lag-feature ML: Ridge, Lasso, random forest, XGBoost, KNN
│   ├── nn_forecasting.py        # neural networks: scikit-learn's MLP, NumPy-by-hand A1 / A2, Keras and PyTorch MLPs
│   ├── var_diagnostics.py       # VAR lag order (AIC / BIC / HQIC) and Johansen cointegration
│   ├── var_models.py            # VAR / VECM candidates: estimation, stability, residual checks
│   ├── var_analysis.py          # Granger causality, impulse responses, variance decomposition
│   └── modeling/
│       ├── train.py             # (template stub) model training
│       └── utilities/           # the networks: NumPy by hand (activations, optimizers, network), Keras, PyTorch
├── tests/                       # pytest: the split, each diagnostic step, GARCH, forecast metrics and baselines
└── tools/                       # make help
```

Every setting (split, test lags, bootstrap size, forecast horizons, every model's search space and so on) lives in `autoregressive_experiments/configs/hyperparameters.json`, each with its `reasoning`; `config_loader.py` rejects a missing, unknown, mistyped or out-of-range setting, and any setting without reasoning, before an analysis runs.

Every make step writes a timestamped log of everything it prints to `logs/run_<date>_<time>_<step>.log` (not committed), with progress lines for long loops and a table of phase timings at the end.

The empty folders hold a `.gitkeep` file so they exist for future use, as in the template. `configs/`, `runlog.py`, `diagnostics.py`, `residual_checks.py`, `volatility_checks.py`, `volatility_models.py`, `asymmetric_garch.py`, `tuning.py`, `forecast_evaluation.py`, `ml_forecasting.py`, `nn_forecasting.py`, `var_diagnostics.py`, `var_models.py`, `var_analysis.py` and `tools/` are this project's additions to the template.

## Datasets & attribution

[Fed Balance Sheet & M2 Money Supply 1950 - Present](https://www.kaggle.com/datasets/belbino/fed-balance-sheet-and-m2-money-supply-1950-present) by Belbin Beno R M on Kaggle, aggregated from the Federal Reserve Bank of St. Louis (FRED). Released under CC0: Public Domain.

## License

MIT, see [`LICENSE`](LICENSE).
