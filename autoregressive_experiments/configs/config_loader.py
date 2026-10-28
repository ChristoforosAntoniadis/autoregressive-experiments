"""
Loading and validation of the hyperparameter config.

Defines `ConfigLoader`, which reads ``hyperparameters.json`` (beside this file), checks every
entry against the definitions in ``configurables`` and returns them as one typed, frozen
`Hyperparameters` object. Every module takes its settings from `HYPERPARAMETERS`, loaded once
on import.

Each entry in the file is an object with exactly two keys: "value" and "reasoning". A
setting without reasoning is rejected: every choice must say why it was made (a
reference, a calculation or an earlier step's result).

Input: ``autoregressive_experiments/configs/hyperparameters.json`` (or another path given to
`ConfigLoader`).

Output: the validated `Hyperparameters` object and each setting's reasoning. Nothing is
written to disk.
"""

from collections.abc import Mapping
from dataclasses import Field, fields
import json
from pathlib import Path
import types
from typing import Any, get_args, get_origin, get_type_hints

from scipy import stats

from autoregressive_experiments.configs.configurables import (
    HYPERPARAMETERS_PATH,
    Hyperparameters,
)

# The two keys every entry must have, and no others.
ENTRY_KEYS = frozenset({"value", "reasoning"})
# How far the impulse-response band's z may sit from the exact normal quantile of the
# significance level: 1.96 vs 1.95996 for 5% (the conventional rounding passes).
BAND_Z_TOLERANCE = 0.01


class ConfigLoader:
    """
    Reads the hyperparameter file and validates it before any analysis runs.

    This class is responsible for:
    - reading the JSON file
    - checking that every section and setting exists, and that no unknown one does
    - checking that every entry has a value and non-empty reasoning
    - converting each value to its declared type (JSON lists become tuples)
    - checking each value against its rule, and the rules that link settings

    Attributes
    ----------
    path : Path
        The config file to read.
    reasoning : dict of tuple of (str, str) to str
        (section, setting) -> its reasoning, filled by load().

    Notes
    -----
    All problems are collected and raised together, so one run lists every mistake in the
    file instead of stopping at the first.
    """

    def __init__(self, path: Path = HYPERPARAMETERS_PATH) -> None:
        """
        Remember which file to load.

        Parameters
        ----------
        path : Path, default=HYPERPARAMETERS_PATH
            The JSON config file.
        """
        self.path = path
        self.reasoning: dict[tuple[str, str], str] = {}

    def load(self) -> Hyperparameters:
        """
        Read, check and convert the whole config.

        Returns
        -------
        Hyperparameters
            Every section's settings, typed and frozen.

        Raises
        ------
        FileNotFoundError
            If the file doesn't exist.
        TypeError
            If the file's top level isn't an object.
        ValueError
            If anything in the file is missing, unknown, of the wrong type, breaks its rule or
            lacks reasoning; the message lists every problem found.

        Notes
        -----
        Processing:
        1. Parse the JSON; the top level must hold exactly the sections of Hyperparameters.
        2. Build each section with read_section().
        3. Check the rules that link settings across sections (cross_checks()).
        4. Raise once with every problem, or return the result.
        """
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        errors: list[str] = []
        section_types = get_type_hints(Hyperparameters)
        if not isinstance(raw, dict):
            raise TypeError(f"{self.path.name}: the top level must be an object of sections")
        errors.extend(f"unknown section '{name}'" for name in raw if name not in section_types)
        sections: dict[str, Any] = {}
        for name, section_type in section_types.items():
            if name not in raw:
                errors.append(f"missing section '{name}'")
            elif not isinstance(raw[name], dict):
                errors.append(f"section '{name}' must be an object of settings")
            else:
                sections[name] = self.read_section(name, section_type, raw[name], errors)
        if errors:
            raise ValueError(f"{self.path.name}:\n  " + "\n  ".join(errors))
        hyperparameters = Hyperparameters(**sections)
        errors.extend(cross_checks(hyperparameters))
        if errors:
            raise ValueError(f"{self.path.name}:\n  " + "\n  ".join(errors))
        return hyperparameters

    def read_section(
        self, section: str, section_type: type, entries: dict[str, Any], errors: list[str]
    ) -> Any:
        """
        Check and convert one section's entries into its dataclass.

        Parameters
        ----------
        section : str
            The section's name, for messages.
        section_type : type
            Its dataclass in configurables.
        entries : dict of str to Any
            The section's raw entries from the JSON.
        errors : list of str
            Collects every problem found (mutated).

        Returns
        -------
        Any
            An instance of `section_type`, or None if any of its settings had a problem (the
            caller raises before using it).
        """
        hints = get_type_hints(section_type)
        definitions: dict[str, Field[Any]] = {f.name: f for f in fields(section_type)}
        errors.extend(
            f"{section}: unknown setting '{key}'" for key in entries if key not in definitions
        )
        values: dict[str, Any] = {}
        failed = False
        for key, definition in definitions.items():
            where = f"{section}.{key}"
            entry = entries.get(key)
            if not isinstance(entry, dict) or set(entry) != ENTRY_KEYS:
                errors.append(f"{where}: needs exactly the keys 'value' and 'reasoning'")
                failed = True
                continue
            reasoning = entry["reasoning"]
            if not isinstance(reasoning, str) or not reasoning.strip():
                errors.append(f"{where}: the reasoning is missing or empty")
                failed = True
            else:
                self.reasoning[(section, key)] = reasoning
            try:
                value = convert(entry["value"], hints[key])
            except TypeError as error:
                errors.append(f"{where}: {error}")
                failed = True
                continue
            if not definition.metadata["check"](value):
                errors.append(f"{where}: {value!r} breaks the rule: {definition.metadata['rule']}")
                failed = True
            values[key] = value
        return None if failed else section_type(**values)


def convert(value: Any, hint: Any) -> Any:
    """
    Convert a JSON value to a declared type, or explain why it can't be.

    Parameters
    ----------
    value : Any
        The value as parsed from JSON (int, float, str, bool, list or dict).
    hint : Any
        The declared type: int, float, str, bool, tuple[X, ...] or a Mapping / dict of str.

    Returns
    -------
    Any
        The value as that type; lists become tuples, and a whole-number int is accepted where a
        float is declared.

    Raises
    ------
    TypeError
        If the value doesn't have the declared type.

    Notes
    -----
    bool is rejected where int or float is declared, although Python treats it as an int:
    true in a lag setting is a mistake, not the number 1.
    """
    origin = get_origin(hint)
    if hint is bool:
        if isinstance(value, bool):
            return value
    elif hint is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    elif hint is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    elif hint is str:
        if isinstance(value, str):
            return value
    elif origin is tuple:
        if isinstance(value, list):
            item_type = get_args(hint)[0]
            return tuple(convert(item, item_type) for item in value)
    elif origin in (dict, Mapping):
        if isinstance(value, dict) and all(isinstance(key, str) for key in value):
            return value
    elif hint is Any:
        return value
    raise TypeError(f"{value!r} is not {type_name(hint)}")


def type_name(hint: Any) -> str:
    """
    A readable name of a declared type, for error messages.

    Parameters
    ----------
    hint : Any
        A type hint, e.g. int or tuple[int, ...].

    Returns
    -------
    str
        E.g. "int" or "a list of int".
    """
    if get_origin(hint) is tuple:
        return f"a list of {type_name(get_args(hint)[0])}"
    if get_origin(hint) in (dict, Mapping):
        return "an object"
    return hint.__name__ if isinstance(hint, (type, types.GenericAlias)) else str(hint)


def cross_checks(hyperparameters: Hyperparameters) -> list[str]:
    """
    The rules that link settings to each other.

    Parameters
    ----------
    hyperparameters : Hyperparameters
        The fully converted config.

    Returns
    -------
    list of str
        One message per broken rule; empty when all hold.

    Notes
    -----
    - Train and validation must leave months for test.
    - The primary VAR criterion must be one of the reported criteria.
    - The carried-forward VAR order must be one of the candidates checked in V4-V6.
    - The impulse-response band's z must match the significance level (within
      BAND_Z_TOLERANCE), so the bands are the level every test uses.
    """
    errors = []
    split = hyperparameters.split
    if split.train_percent + split.validation_percent >= 100:
        errors.append("split: train_percent + validation_percent must leave months for test")
    var = hyperparameters.var_diagnostics
    if var.primary_criterion not in var.criteria:
        errors.append("var_diagnostics: primary_criterion must be one of criteria")
    if hyperparameters.var_analysis.carried_lags not in hyperparameters.var_models.var_lags:
        errors.append("var_analysis: carried_lags must be one of var_models.var_lags")
    exact_z = stats.norm.ppf(1 - hyperparameters.tests.significance_level / 2)
    if abs(hyperparameters.var_analysis.band_z - exact_z) > BAND_Z_TOLERANCE:
        errors.append(
            f"var_analysis: band_z must be within {BAND_Z_TOLERANCE} of {exact_z:.4f}, the "
            "normal quantile of the significance level"
        )
    return errors


# Loaded once when first imported; every module reads its settings from here.
HYPERPARAMETERS = ConfigLoader().load()
