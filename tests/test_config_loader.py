"""
Tests for the hyperparameter config: autoregressive_experiments.config_loader reading
configs/hyperparameters.json against the definitions in autoregressive_experiments.configurables.
"""

from dataclasses import fields
import json
from pathlib import Path
import re
from typing import Any

import pytest

from autoregressive_experiments.configs.config_loader import ConfigLoader, convert
from autoregressive_experiments.configs.configurables import HYPERPARAMETERS_PATH, Hyperparameters


def write_config(tmp_path: Path, change: Any) -> Path:
    """
    Copy the project's config, apply one change, and write it to a temporary file.

    Parameters
    ----------
    tmp_path : Path
        pytest's temporary folder.
    change : callable
        Takes the parsed config (a dict) and edits it in place.

    Returns
    -------
    Path
        The changed config file.
    """
    config = json.loads(HYPERPARAMETERS_PATH.read_text(encoding="utf-8"))
    change(config)
    path = tmp_path / "hyperparameters.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_the_project_config_loads_with_reasoning_for_every_setting() -> None:
    """The committed file passes every check, and each setting has its reasoning: one per
    field of every section."""
    loader = ConfigLoader()
    hyperparameters = loader.load()

    settings = sum(len(fields(getattr(hyperparameters, f.name))) for f in fields(Hyperparameters))
    assert len(loader.reasoning) == settings
    # Spot checks: JSON lists arrive as tuples, a JSON 2.0 as the float 2.0.
    assert hyperparameters.tests.significance_level == 0.05
    assert hyperparameters.forecast_evaluation.horizons == (1, 3, 6, 12)
    assert hyperparameters.residual_checks.aic_tie_margin == 2.0
    # Search spaces stay as JSON gave them: lists, with null as None.
    assert hyperparameters.ml.random_forest["max_depth"][0] is None


@pytest.mark.parametrize(
    ("description", "change", "message"),
    [
        (
            "missing reasoning",
            lambda c: c["split"]["train_percent"].pop("reasoning"),
            "split.train_percent: needs exactly the keys",
        ),
        (
            "empty reasoning",
            lambda c: c["split"]["train_percent"].update(reasoning="  "),
            "split.train_percent: the reasoning is missing or empty",
        ),
        (
            "an unknown setting",
            lambda c: c["split"].update(test_percent={"value": 20, "reasoning": "x"}),
            "split: unknown setting 'test_percent'",
        ),
        ("a missing section", lambda c: c.pop("var_models"), "missing section 'var_models'"),
        (
            "a string where an int belongs",
            lambda c: c["diagnostics"]["max_iterations"].update(value="200"),
            "diagnostics.max_iterations: '200' is not int",
        ),
        (
            "true where an int belongs",
            lambda c: c["diagnostics"]["seasonal_period"].update(value=True),
            "diagnostics.seasonal_period: True is not int",
        ),
        (
            "a value breaking its rule",
            lambda c: c["volatility_checks"]["break_trim"].update(value=0.7),
            "volatility_checks.break_trim: 0.7 breaks the rule: between 0 and 0.5",
        ),
        (
            "lags out of order",
            lambda c: c["var_models"]["var_lags"].update(value=[3, 1, 5]),
            "var_models.var_lags: (3, 1, 5) breaks the rule",
        ),
        (
            "a search space missing a setting",
            lambda c: c["ml"]["random_forest"]["value"].pop("max_depth"),
            "ml.random_forest: ",
        ),
        (
            "a search space with no values to try",
            lambda c: c["tuning"]["sma"]["value"].update(window=[]),
            "tuning.sma: ",
        ),
        (
            "a search range whose low end isn't below its high end",
            lambda c: c["nn"]["pytorch_mlp"]["value"]["dropout"].update(low=0.5),
            "nn.pytorch_mlp: ",
        ),
        (
            "no months left for test",
            lambda c: c["split"]["train_percent"].update(value=80),
            "train_percent + validation_percent must leave months for test",
        ),
        (
            "a band that doesn't match the significance level",
            lambda c: c["var_analysis"]["band_z"].update(value=2.58),
            "band_z must be within 0.01 of 1.9600",
        ),
    ],
)
def test_the_loader_rejects(tmp_path: Path, description: str, change: Any, message: str) -> None:
    """Each broken file is rejected with a message naming the setting and the problem."""
    loader = ConfigLoader(write_config(tmp_path, change))

    # The messages contain regex characters such as ( ) and +, so they are escaped.
    with pytest.raises(ValueError, match=re.escape(message)):
        loader.load()


def test_the_loader_reports_every_problem_at_once(tmp_path: Path) -> None:
    """Two broken settings give one error listing both, not just the first."""

    def two_mistakes(config: dict[str, Any]) -> None:
        """Break one value's type and remove another's reasoning."""
        config["diagnostics"]["max_iterations"]["value"] = "many"
        config["tests"]["significance_level"].pop("reasoning")

    with pytest.raises(ValueError) as error:
        ConfigLoader(write_config(tmp_path, two_mistakes)).load()

    assert "diagnostics.max_iterations" in str(error.value)
    assert "tests.significance_level" in str(error.value)


def test_a_top_level_that_is_not_an_object_is_rejected(tmp_path: Path) -> None:
    """A config must be an object of sections; a list is a type error."""
    path = tmp_path / "hyperparameters.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(TypeError, match="top level must be an object"):
        ConfigLoader(path).load()


def test_convert_by_hand() -> None:
    """Lists become tuples of the declared type; a whole number is accepted as a float; a
    wrong item is named in the message."""
    assert convert([1, 3, 5], tuple[int, ...]) == (1, 3, 5)
    assert convert(2, float) == 2.0
    assert isinstance(convert(2, float), float)
    with pytest.raises(TypeError, match="'3' is not int"):
        convert([1, "3"], tuple[int, ...])
