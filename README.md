# Autoregressive Experiments

Experiments on autoregressive approaches, from traditional econometrics (SARIMA, VAR / VECM, ARCH / GARCH) to neural networks, benchmarked on the same dataset: the Federal Reserve's total assets and the monetary base, monthly from December 2002.

![Python](https://img.shields.io/badge/python-3.12-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

---

## Plan

- **Univariate (SARIMA, then ARCH / GARCH):** stationarity tests (ADF / KPSS on levels and first differences), ACF / PACF, and SARIMA order selection by AIC / AICc / BIC, residual checks of the selected models (Ljung-Box, Breusch-Godfrey), volatility checks (Engle's ARCH-LM, with and without a variance break), GARCH(1,1), EGARCH and GJR-GARCH on the residuals are done on the training split. Rolling-origin forecasts (expanding window, one fold per monthly origin) are scored on validation and then once on test, against naive, drift and seasonal-naive baselines: MAE, MSE, RMSE, MAPE, MASE and Diebold-Mariano tests at 1–12 months, plus quantile forecast intervals (80% / 95% coverage, pinball loss) with and without GARCH; see [`reports/`](reports/).
- **Multivariate (VAR / VECM):** lag selection and the Johansen cointegration test are done on the training split; no cointegration was found, so the system is a VAR on the first differences. VAR(5) and a rank-1 VECM(5) are estimated, stable and pass the residual checks; Granger causality, impulse responses and the variance decomposition show a one-way link, from total assets to the monetary base. Both systems are evaluated with the same harness, on validation and once on test, including a Diebold-Mariano comparison against the selected SARIMA: joint modelling doesn't beat univariate modelling for either series. SARIMAX (the selected SARIMA plus 1, 3 or 5 lagged monthly changes of the other series, chosen on validation) gives the same answer: no gain over SARIMA on test.
- **Smoothing baselines and a combination:** moving averages (3, 6, 12 months), single, double (Holt, plain and damped) and triple (Holt-Winters) exponential smoothing, and a 50 / 50 average of SARIMA and the naive forecast, all through the same harness. On test, damped Holt is the best total-assets model so far, significantly better than naive and SARIMA at every horizon.
- **Machine learning and neural networks:** models on lag features (gradient boosting, regularized regression), on lookback windows (MLP, N-BEATS, DLinear), and recurrent networks (LSTM, GRU), evaluated on the same chronological 60 / 20 / 20 split.

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
make evaluate                     # rolling-origin forecasts of every model on validation and test (~12 min)
make var                          # VAR lag order and Johansen cointegration test -> reports/
make var-checks                   # fit VAR / VECM candidates; stability and Portmanteau checks -> reports/
make var-analysis                 # Granger causality, impulse responses, variance decomposition -> reports/
make all                          # every step above, in order (~25 min)

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
├── configs/
│   └── hyperparameters.json     # every setting's value and its justification
├── docs/                        # (empty) detailed documentation
├── models/                      # (empty) trained and serialized models
├── notebooks/                   # (empty) Jupyter notebooks
├── references/                  # (empty) data dictionaries, manuals, papers
├── reports/                     # stationarity, ACF / PACF and information-criteria reports
│   └── figures/                 # the series and its ACF / PACF
├── autoregressive_experiments/  # the source package
│   ├── config.py                # project paths
│   ├── configurables.py         # every setting's type and allowed values, one dataclass per step
│   ├── config_loader.py         # reads and validates configs/hyperparameters.json
│   ├── dataset.py               # downloads the raw dataset
│   ├── features.py              # cleaning, first-differencing, chronological 60 / 20 / 20 split
│   ├── plots.py                 # figure of the cleaned series
│   ├── diagnostics.py           # ADF / KPSS, ACF / PACF, SARIMA information criteria
│   ├── residual_checks.py       # Ljung-Box and Breusch-Godfrey on the selected models' residuals
│   ├── volatility_checks.py     # ARCH-LM, with and without an estimated variance break
│   ├── volatility_models.py     # GARCH(1,1) (normal and Student-t) vs constant variance, by BIC; residual checks by parametric bootstrap
│   ├── asymmetric_garch.py      # EGARCH and GJR-GARCH (Student-t), plain or break-adjusted, by BIC
│   ├── forecast_evaluation.py   # rolling-origin forecasts of every model vs baselines, validation and test
│   ├── var_diagnostics.py       # VAR lag order (AIC / BIC / HQIC) and Johansen cointegration
│   ├── var_models.py            # VAR / VECM candidates: estimation, stability, residual checks
│   ├── var_analysis.py          # Granger causality, impulse responses, variance decomposition
│   └── modeling/
│       └── train.py             # (template stub) model training
├── tests/                       # pytest: the split, each diagnostic step, GARCH, forecast metrics and baselines
└── tools/                       # make help
```

Every setting (split, test lags, model grids, bootstrap size, forecast horizons and so on) lives in `configs/hyperparameters.json`, each with a `justification`; `config_loader.py` rejects a missing, unknown, mistyped or out-of-range setting, and any setting without a justification, before an analysis runs.

The empty folders hold a `.gitkeep` file so they exist for future use, as in the template. `configs/`, `configurables.py`, `config_loader.py`, `diagnostics.py`, `residual_checks.py`, `volatility_checks.py`, `volatility_models.py`, `asymmetric_garch.py`, `forecast_evaluation.py`, `var_diagnostics.py`, `var_models.py`, `var_analysis.py` and `tools/` are this project's additions to the template.

## Datasets & attribution

[Fed Balance Sheet & M2 Money Supply 1950 - Present](https://www.kaggle.com/datasets/belbino/fed-balance-sheet-and-m2-money-supply-1950-present) by Belbin Beno R M on Kaggle, aggregated from the Federal Reserve Bank of St. Louis (FRED). Released under CC0: Public Domain.

## License

MIT, see [`LICENSE`](LICENSE).
