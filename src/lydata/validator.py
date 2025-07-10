"""Module to transform to and validate the CSV schema of the lydata datasets.

Here we define the function :py:func:`construct_schema` to dynamically create a
:py:class:`pandera.DataFrameSchema` that we can use to validate that a given
:py:class:`~pandas.DataFrame` conforms to the minimum requirements of the lyDATA
datasets.

Currently, we only publish the :py:func:`validate_datasets` function that validates all
datasets that are found by the function :py:func:`~lydata.loader.available_datasets`.
In the future, we may want to make this more flexible.

In this module, we also provide the :py:func:`transform_to_lyprox` function that can be
used to transform any raw data into the format that can be uploaded to the `LyProX`_
platform database.

.. _LyProX: https://lyprox.org
"""

import sys
from typing import Any

import pandas as pd
from loguru import logger
from pydantic import Field, create_model

from lydata import loader
from lydata.schema import BaseRecord, ModalityRecord
from lydata.utils import get_default_modalities


def flatten(
    nested: dict,
    prev_key: tuple = (),
    max_depth: int | None = None,
) -> dict:
    """Flatten ``nested`` dict by creating key tuples for each value at ``max_depth``.

    >>> nested = {"tumor": {"1": {"t_stage": 1, "size": 12.3}}}
    >>> flatten(nested)
    {('tumor', '1', 't_stage'): 1, ('tumor', '1', 'size'): 12.3}
    >>> mapping = {"patient": {"#": {"age": {"func": int, "columns": ["age"]}}}}
    >>> flatten(mapping, max_depth=3)
    {('patient', '#', 'age'): {'func': <class 'int'>, 'columns': ['age']}}

    Note that flattening an already flat dictionary will yield some weird results.
    """
    result = {}

    for key, value in nested.items():
        is_dict = isinstance(value, dict)
        has_reached_max_depth = max_depth is not None and len(prev_key) >= max_depth - 1

        if is_dict and not has_reached_max_depth:
            result.update(flatten(value, (*prev_key, key), max_depth))
        else:
            result[(*prev_key, key)] = value

    return result


def unflatten(flat: dict) -> dict:
    """Take a flat dictionary with tuples of keys and create nested dict from it.

    >>> flat = {('tumor', '1', 't_stage'): 1, ('tumor', '1', 'size'): 12.3}
    >>> unflatten(flat)
    {'tumor': {'1': {'t_stage': 1, 'size': 12.3}}}
    >>> mapping = {('patient', '#', 'age'): {'func': int, 'columns': ['age']}}
    >>> unflatten(mapping)
    {'patient': {'#': {'age': {'func': <class 'int'>, 'columns': ['age']}}}}
    """
    result = {}

    for keys, value in flat.items():
        current = result
        for key in keys[:-1]:
            current = current.setdefault(key, {})

        current[keys[-1]] = value

    return result


def move_value(mapping: dict[str, Any], from_key: str, to_key: str) -> None:
    """Move a key in a dictionary to another key."""
    if from_key not in mapping:
        raise KeyError(f"Key '{from_key}' not found in mapping.")

    mapping[to_key] = mapping.pop(from_key)


def create_modality_field(modality: str) -> tuple[type, Field]:
    """Create a field for a specific modality."""
    return (
        ModalityRecord,
        Field(
            default_factory=ModalityRecord,
            description=f"Involvement data for modality {modality}",
        ),
    )


def validate(dataset: pd.DataFrame) -> bool:
    """Validate the given dataset against the lyDATA schema."""
    top_lvl_cols = set(dataset.columns.get_level_values(0).unique())
    top_lvl_cols -= {"patient", "tumor"}

    modality_cols = set()
    for col in top_lvl_cols:
        if col in get_default_modalities():
            modality_cols.add(col)

    FullRecord = create_model(  # noqa: N806
        "FullRecord",
        __base__=BaseRecord,
        **{col: create_modality_field(col) for col in modality_cols},
    )

    for i, row in dataset.iterrows():
        record = unflatten(row.to_dict())

        move_value(record["patient"], from_key="#", to_key="_")
        move_value(record["tumor"], from_key="1", to_key="_")

        with logger.catch(message=f"Validation error for patient {i}"):
            validated_record = FullRecord(**record)
            logger.debug(f"Patient {i} is valid: {validated_record}")

    return True


if __name__ == "__main__":
    logger.enable("lydata")
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    dataset = next(loader.load_datasets())
    validate(dataset)
