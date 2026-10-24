#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = autoregressive-experiments
PACKAGE = autoregressive_experiments

# Use the project's virtual environment when it exists, otherwise the Python on PATH.
ifeq ($(OS),Windows_NT)
VENV_PYTHON = .venv/Scripts/python.exe
else
VENV_PYTHON = .venv/bin/python
endif
PYTHON_INTERPRETER = $(if $(wildcard $(VENV_PYTHON)),$(VENV_PYTHON),python)

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Create the virtual environment in .venv
.PHONY: create_environment
create_environment:
	python -m venv .venv
	@echo Created .venv. Run make requirements next; the Makefile uses .venv automatically.

## Install the pinned runtime dependencies, the package itself, and the dev tools
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	$(PYTHON_INTERPRETER) -m pip install -e ".[dev]"

## Download the Kaggle dataset into data/raw/ (replaces what is there)
.PHONY: data
data:
	$(PYTHON_INTERPRETER) -m $(PACKAGE).dataset

## Clean the raw data into data/interim/ and first-difference it into data/processed/
.PHONY: features
features:
	$(PYTHON_INTERPRETER) -m $(PACKAGE).features

## Plot the cleaned series (no plot windows)
.PHONY: plots
plots: export MPLBACKEND = Agg
plots:
	$(PYTHON_INTERPRETER) -m $(PACKAGE).plots

## Write the stationarity, ACF / PACF and information-criteria reports (no plot windows)
.PHONY: diagnostics
diagnostics: export MPLBACKEND = Agg
diagnostics:
	$(PYTHON_INTERPRETER) -m $(PACKAGE).diagnostics

## Check the residuals of the recommended SARIMA models (needs make diagnostics; no plot windows)
.PHONY: residuals
residuals: export MPLBACKEND = Agg
residuals:
	$(PYTHON_INTERPRETER) -m $(PACKAGE).residual_checks

## Test the passing models' residuals for changing volatility (needs make diagnostics; no plot windows)
.PHONY: volatility
volatility: export MPLBACKEND = Agg
volatility:
	$(PYTHON_INTERPRETER) -m $(PACKAGE).volatility_checks

## Fit GARCH(1,1) on the passing models' residuals and check it (needs make diagnostics; no plot windows)
.PHONY: garch
garch: export MPLBACKEND = Agg
garch:
	$(PYTHON_INTERPRETER) -m $(PACKAGE).volatility_models

## Run the whole pipeline: features, plots, diagnostics, residual, volatility and GARCH checks
.PHONY: all
all: features plots diagnostics residuals volatility garch

## Run the test suite
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest

## Check code style and lint with ruff (reports only, changes nothing)
.PHONY: lint
lint:
	$(PYTHON_INTERPRETER) -m ruff format --check
	$(PYTHON_INTERPRETER) -m ruff check

## Fix lint issues and format the code with ruff (rewrites files)
.PHONY: format
format:
	$(PYTHON_INTERPRETER) -m ruff check --fix
	$(PYTHON_INTERPRETER) -m ruff format

## Delete Python caches and compiled files
.PHONY: clean
clean:
	$(PYTHON_INTERPRETER) -c "import pathlib, shutil; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__') if '.venv' not in p.parts]; [shutil.rmtree(p, ignore_errors=True) for p in ('.pytest_cache', '.ruff_cache')]"

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

## Show this list of commands
.PHONY: help
help:
	@$(PYTHON_INTERPRETER) tools/make_help.py Makefile
