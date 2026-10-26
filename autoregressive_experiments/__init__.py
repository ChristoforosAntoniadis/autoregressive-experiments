"""
Autoregressive experiments on the Federal Reserve balance sheet.

The project's source package, laid out as in Cookiecutter Data Science:

- ``config``: project paths
- ``dataset``: downloads the Kaggle dataset into ``data/raw/``
- ``features``: cleaning into ``data/interim/`` and first-differencing into ``data/processed/``;
  also owns the chronological 60 / 20 / 20 train / validation / test split
- ``plots``: the figure of the cleaned series
- ``diagnostics``: stationarity, ACF / PACF and information-criteria reports (steps U1-U3)
- ``residual_checks``: Ljung-Box and Breusch-Godfrey on the SARIMA residuals (U4)
- ``volatility_checks``: ARCH-LM and the variance break (U5)
- ``volatility_models``: GARCH(1,1) on the SARIMA residuals (U5b)
- ``asymmetric_garch``: EGARCH and GJR-GARCH against GARCH (U5c)
- ``forecast_evaluation``: the shared rolling-origin evaluation harness (U6 and the systems)
- ``var_diagnostics``: VAR lag order and the Johansen test (V2-V3)
- ``var_models``: VAR / VECM candidates, stability and Portmanteau checks (V4-V6)
- ``var_analysis``: Granger causality, impulse responses and FEVD (V7-V8)
- ``modeling``: model training (still the template stub, not used by the pipeline)

Each analysis module is run by its own make target and writes its report to ``reports/``.
"""
