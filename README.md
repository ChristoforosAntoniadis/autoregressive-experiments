# Autoregressive Experiments

Experiments on autoregressive approaches, from traditional econometrics (SARIMA, VAR / VECM, ARCH / GARCH) to neural networks, benchmarked on the same dataset and split. Two dataset profiles share every step: the Federal Reserve's total assets and the monetary base, monthly from December 2002 (the original benchmark), and the US Treasury's daily deposits and withdrawals of operating cash, business days from October 2005 (the daily-cashflow rehearsal).

![Python](https://img.shields.io/badge/python-3.12-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

---

## Plan

- **Univariate (SARIMA, then ARCH / GARCH):** stationarity tests (ADF / KPSS on levels and first differences), ACF / PACF, and SARIMA order selection by AIC / AICc / BIC, residual checks of the selected models (Ljung-Box, Breusch-Godfrey), volatility checks (Engle's ARCH-LM, with and without a variance break), GARCH(1,1), EGARCH and GJR-GARCH on the residuals are done on the training split.
- **Multivariate (VAR / VECM):** lag selection and the Johansen cointegration test are done on the training split; no cointegration was found, so the system is a VAR on the first differences. VAR(5) and a rank-1 VECM(5) are estimated, stable and pass the residual checks; Granger causality, impulse responses and the variance decomposition show a one-way link, from total assets to the monetary base. VAR, VECM and SARIMAX (SARIMA plus the other series' lagged monthly changes) join the forecast comparison below.
- **Lag-feature ML:** Ridge, Lasso, random forest, XGBoost and k-nearest neighbours on the last monthly changes of both series, forecasting recursively.
- **Smoothing baselines and a combination:** a moving average and exponential smoothing (trend, damping, season, Box-Cox and initialization all tuned), and a weighted average of SARIMA and the naive forecast (weight tuned).
- **Neural networks:** a multilayer perceptron (scikit-learn's MLPRegressor) and two networks written by hand in NumPy (A1: one sigmoid hidden layer; A2: three ReLU hidden layers; SGD, momentum, AdaBelief or Muon), a TensorFlow (Keras) and a PyTorch multilayer perceptron (A1 or A2), and PyTorch sequence networks (LSTM, GRU, TCN, N-BEATS, DLinear) reading the same lags as a window.
- **Prophet** on each series' levels, counted with the econometric models.
- **Forecast comparison (train / validation / test, 60 / 20 / 20 in time order):** every model is fitted on an expanding window from the start of the data and forecasts 1, 3, 6, 9 and 12 months ahead. Its settings are tuned by Optuna on 14 validation origins, one every 4 months, scored by its MAE relative to the naive forecast's at the same horizon, averaged over the five horizons (so every horizon counts the same); the best model of each family (econometric, machine learning, neural network) is its champion, and the family's models within one bootstrap standard error of it are reported as tied, not beaten. The champions, their equal-weight mean, and two benchmarks (naive, and the random walk with drift) are then refitted at each of the 56 test months and scored once: MAE, MAPE, MASE, MSE and RMSE per horizon, MAE relative to naive, the bias (mean error and the share of forecasts above the actual), Diebold-Mariano tests against naive (with the number of tests stated, since about 5% come out significant by chance), a Ljung-Box test of the one-step errors, and 80% / 95% bands from the quantiles of each model's own past errors at that horizon (coverage, width, pinball loss); see [`reports/`](reports/).
- **Dataset profiles:** every step runs on one profile, chosen with `make DATASET=<name> ...` (default `fed_balance_sheet`); a profile is one config file, `autoregressive_experiments/configs/<name>.json`, holding the dataset's source, series, frequency, seasonal period, horizons, benchmarks and every search space, each setting with its reasoning, and its data, models and reports live in `<name>/` folders. The code names no frequency: a month, a business day or any other spacing comes from the config. The `daily_treasury` profile models the running totals of deposits and of withdrawals (billions USD) on business days, so the first differences are the daily flows and a level forecast 21 business days ahead is the 30-calendar-day total; holidays are zero-flow days, so every fifth row is the same weekday; the seasonal period is 5, the seasonal naive (last week's flows repeated) is the third benchmark and the reference the scores are relative to, the Hampel filter judges each day against the same weekday's median in its window, and origins are spaced every 40 business days on validation (23 origins) and every 20 on test (46).
- **Transform and outliers, tuned per model:** every model (benchmarks aside) chooses on validation whether it works on the levels or their logarithm (growth rates once differenced, so the early, small years weigh as much as the recent, large ones). A Hampel filter flags unusual monthly changes (more than 3 robust standard deviations from the median of their 13-month window), using only the history up to each forecast date; each model's tuned settings choose whether flagged changes are kept, clipped to the window's bounds, or (SARIMA, SARIMAX) given a pulse regressor, and the gradient-trained models choose between squared-error and Huber loss. Forecasts are always scored on every month; the test report adds a labelled sensitivity table without the flagged months.

## Limitations

- **Three regimes, one test.** Training (2002-12 to 2016-10) is mostly the quantitative-easing era, validation (2016-11 to 2021-06) holds the 2017-19 runoff and the COVID expansion, and test (2021-07 to 2026-02) holds quantitative tightening. The test results measure how models tuned on the first two regimes fare in the third; they are 56 monthly forecasts per horizon from one regime, not a general ranking of the methods.
- **Policy-driven series.** Both series move on Federal Reserve decisions that their own past does not predict (Meese and Rogoff 1983 found the same for exchange rates), so the naive forecast is a hard benchmark and the gains over it are modest beyond a few months. Known future information (announced runoff caps) is outside the autoregressive scope of this project.
- **Selection noise.** About 1,300 settings per series are scored on 14 validation origins, so a champion's validation score is optimistic and the margins between champions are often inside their standard errors; the reports mark such ties.
- **The volatility chapter is descriptive.** GARCH, EGARCH and GJR-GARCH are fitted and checked on the training residuals, but every test band is empirical (quantiles of each model's own past errors), so the variance models inform no forecast.

## Quickstart

The project follows the [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/) layout, and a `Makefile` wraps the common commands:

```bash
make create_environment           # python -m venv .venv
make requirements                 # pinned dependencies + the package + dev tools (pytest, ruff, jupyter)

make data                         # download the profile's Kaggle dataset into data/raw/<dataset>/ (already committed)
make features                     # build the levels -> data/interim/<dataset>/, first-difference -> data/processed/<dataset>/
make plots                        # the cleaned series -> reports/<dataset>/figures/series.png
make diagnostics                  # ADF / KPSS, ACF / PACF and SARIMA information criteria -> reports/
make residuals                    # Ljung-Box / Breusch-Godfrey on the selected models -> reports/
make volatility                   # ARCH-LM and a variance-break check on those models -> reports/
make garch                        # GARCH(1,1) vs constant variance on those models, bootstrap checks (~3 min) -> reports/
make asymmetric-garch             # EGARCH and GJR-GARCH vs that GARCH, bootstrap checks (~12 min) -> reports/
make outliers                     # flag unusual changes in the train and validation history -> reports/
make tune                         # tune every model on validation, choose each family's champion -> models/, reports/
make evaluate                     # test the champions, their mean and the benchmarks once: metrics, bands, figures -> reports/
make var                          # VAR lag order and Johansen cointegration test -> reports/
make var-checks                   # fit VAR / VECM candidates; stability and Portmanteau checks -> reports/
make var-analysis                 # Granger causality, impulse responses, variance decomposition -> reports/
make all                          # every step above, in order (~2.5 h for the Fed profile)
make DATASET=daily_treasury all   # the same on the daily Treasury profile; every target takes DATASET=<name>

make test                         # unit tests and an end-to-end smoke test of the forecast pipeline (~3.5 min)
make test-fast                    # the unit tests alone (~1 min)
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
python -m autoregressive_experiments.outliers
python -m autoregressive_experiments.tuning
python -m autoregressive_experiments.forecast_evaluation
python -m autoregressive_experiments.var_diagnostics
python -m autoregressive_experiments.var_models
python -m autoregressive_experiments.var_analysis
python -m pytest
```

When run directly (not through `make`), set `MPLBACKEND=Agg` to stop plot windows from opening, and `DATASET=<name>` to pick a profile other than `fed_balance_sheet`.

## Project structure

```
autoregressive-experiments/
├── Makefile                     # make data / features / plots / diagnostics / test / lint / ...
├── pyproject.toml               # package metadata, dev tools, pytest and ruff settings
├── requirements.txt             # pinned runtime dependencies
├── data/
│   ├── external/                # (empty) data from third-party sources
│   ├── interim/<dataset>/       # the cleaned levels, regenerated by make features (not committed)
│   ├── processed/<dataset>/     # their first differences, regenerated by make features (not committed)
│   └── raw/<dataset>/           # the Kaggle CSVs as downloaded (the Treasury's reduced to the table used)
├── docs/                        # (empty) detailed documentation
├── models/<dataset>/            # every model's tuned settings and each family's champion
├── notebooks/                   # (empty) Jupyter notebooks
├── references/                  # (empty) data dictionaries, manuals, papers
├── reports/<dataset>/           # every report and table of the profile
│   └── figures/                 # every figure of the profile
├── autoregressive_experiments/  # the source package
│   ├── config.py                # project paths; the DATASET profile they follow
│   ├── configs/
│   │   ├── configurables.py     # every setting's type and allowed values, one dataclass per step
│   │   ├── config_loader.py     # reads and validates the profile's config file
│   │   ├── fed_balance_sheet.json   # the monthly profile: every setting's value and its reasoning
│   │   └── daily_treasury.json  # the daily profile, same layout
│   ├── runlog.py                # run logs in logs/, phase timings, progress lines
│   ├── dataset.py               # downloads the profile's raw dataset (and reduces the Treasury's)
│   ├── features.py              # builds the levels per profile, first-differencing, chronological 60 / 20 / 20 split
│   ├── plots.py                 # figure of the cleaned series
│   ├── diagnostics.py           # ADF / KPSS, ACF / PACF, SARIMA information criteria
│   ├── residual_checks.py       # Ljung-Box and Breusch-Godfrey on the selected models' residuals
│   ├── volatility_checks.py     # ARCH-LM, with and without an estimated variance break
│   ├── volatility_models.py     # GARCH(1,1) (normal and Student-t) vs constant variance, by BIC; residual checks by parametric bootstrap
│   ├── asymmetric_garch.py      # EGARCH and GJR-GARCH (Student-t), plain or break-adjusted, by BIC
│   ├── model_forecasts.py       # every model's fit and forecast from the periods up to an origin; the benchmarks
│   ├── metrics.py               # MAE / MAPE / MASE / MSE / RMSE, Diebold-Mariano, error-quantile bands
│   ├── outliers.py              # Hampel detection of unusual changes; clipping and pulse regressors
│   ├── tuning.py                # Optuna search of every model on validation; each family's champion
│   ├── forecast_evaluation.py   # the champions, their mean and the benchmarks on test, once: metrics, bands, figures
│   ├── ml_forecasting.py        # lag-feature ML: Ridge, Lasso, random forest, XGBoost, KNN
│   ├── nn_forecasting.py        # neural networks: MLPs (scikit-learn, NumPy by hand, Keras, PyTorch), LSTM, GRU, TCN, N-BEATS, DLinear
│   ├── prophet_forecasting.py   # Prophet
│   ├── var_diagnostics.py       # VAR lag order (AIC / BIC / HQIC) and Johansen cointegration
│   ├── var_models.py            # VAR / VECM candidates: estimation, stability, residual checks
│   ├── var_analysis.py          # Granger causality, impulse responses, variance decomposition
│   └── modeling/
│       └── utilities/           # the networks: NumPy by hand (activations, optimizers, network), Keras, PyTorch, sequence networks
├── tests/                       # pytest: the split, each diagnostic step, GARCH, forecast metrics and baselines, an end-to-end smoke test
└── tools/                       # make help
```

Every setting (the dataset, its frequency, split, test lags, bootstrap size, forecast horizons, every model's search space and so on) lives in the profile's `autoregressive_experiments/configs/<dataset>.json`, each with its `reasoning`; `config_loader.py` rejects a missing, unknown, mistyped or out-of-range setting, and any setting without reasoning, before an analysis runs.

Every make step writes a timestamped log of everything it prints to `logs/run_<date>_<time>_<step>.log` (not committed), with progress lines for long loops and a table of phase timings at the end.

The empty folders hold a `.gitkeep` file so they exist for future use, as in the template. `configs/`, `runlog.py`, `diagnostics.py`, `residual_checks.py`, `volatility_checks.py`, `volatility_models.py`, `asymmetric_garch.py`, `model_forecasts.py`, `metrics.py`, `outliers.py`, `tuning.py`, `forecast_evaluation.py`, `ml_forecasting.py`, `nn_forecasting.py`, `prophet_forecasting.py`, `var_diagnostics.py`, `var_models.py`, `var_analysis.py` and `tools/` are this project's additions to the template.

## Datasets & attribution

- `fed_balance_sheet`: [Fed Balance Sheet & M2 Money Supply 1950 - Present](https://www.kaggle.com/datasets/belbino/fed-balance-sheet-and-m2-money-supply-1950-present) by Belbin Beno R M on Kaggle, aggregated from the Federal Reserve Bank of St. Louis (FRED). Released under CC0: Public Domain.
- `daily_treasury`: [Daily cash and debt operations of U.S. Treasury](https://www.kaggle.com/datasets/nakulshahdadpuri314/daily-treasury-statement) on Kaggle, a mirror of the [Daily Treasury Statement](https://fiscaldata.treasury.gov/datasets/daily-treasury-statement/) published by the Bureau of the Fiscal Service (US government work, public domain), 2005-10-03 to 2023-04-03. Only the deposits and withdrawals of the Treasury's main account are kept, summed per day and transaction type.

## License

MIT, see [`LICENSE`](LICENSE).
