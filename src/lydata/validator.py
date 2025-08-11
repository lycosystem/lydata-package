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
from collections.abc import Mapping
from typing import Any

from loguru import logger
from pandas import PeriodDtype
from pydantic import BaseModel, Field, PastDate, create_model

from lydata import loader
from lydata.accessor import LyDataAccessor, LyDataFrame  # noqa: F401
from lydata.schema import BaseRecord, ModalityRecord


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


def create_full_record_model(modalities: list[str]) -> type:
    """Create a Pydantic model for a full record with all modalities."""
    return create_model(
        "FullRecord",
        __base__=BaseRecord,
        **{mod: create_modality_field(mod) for mod in modalities},
    )


def validate(dataset: LyDataFrame) -> bool:
    """Validate the given dataset against the lyDATA schema."""
    modalities = dataset.ly.get_modalities()
    FullRecord = create_full_record_model(modalities)  # noqa: N806

    for i, row in dataset.iterrows():
        record = unflatten(row.to_dict())
        patient_id = record["patient"]["core"]["id"]

        # move_value(record["patient"], from_key="#", to_key="core")
        # move_value(record["tumor"], from_key="1", to_key="core")

        with logger.catch(message=f"Validation error for {patient_id=}"):
            validated_record = FullRecord(**record)
            logger.debug(f"Patient {i} is valid: {validated_record}")

    return True


def get_field_annotations(
    model: type[BaseModel],
) -> dict[str, Any]:
    """Get the field annotations of a three-level nested Pydantic model.

    >>> class Foo(BaseModel):
    ...     bar: int = 3
    >>> class Baz(BaseModel):
    ...     foo: Foo = Field(default_factory=Foo)
    >>> get_field_annotations(Baz)
    {'foo': {'bar': <class 'int'>}}
    """
    annotations = {}
    for field_name, field_info in model.model_fields.items():
        if issubclass(field_info.annotation, BaseModel):
            annotations[field_name] = get_field_annotations(field_info.annotation)
        else:
            annotations[field_name] = field_info.annotation

    return annotations


def cast_types(
    dataset: LyDataFrame,
    casters: Mapping[type, str] | None = None,
) -> LyDataFrame:
    """Cast the types of the ``dataset`` to the expected types.

    This function uses the annotations of the Pydantic schema to cast the individual
    columns of the ``dataset`` to the expected types. It uses the ``casters`` mapping
    to determine the type to cast to. By default, it uses the following mapping:

    .. code-block:: python

        {
            int: int,
            int | None: "Int64",
            float: float,
            float | None: "Float64",
            str: "string",
            str | None: "string",
            bool: bool,
            bool | None: "boolean",
            PastDate: PeriodDtype("D"),
        }

    That way, pandas uses e.g. the nullable integer type ``Int64`` if we specify in
    pydantic that a field can be an integer or None. If you want to use a different
    mapping, you can pass it as the ``casters`` argument.
    """
    if casters is None:
        casters = {
            int: int,
            int | None: "Int64",
            float: float,
            float | None: "Float64",
            str: "string",
            str | None: "string",
            bool: bool,
            bool | None: "boolean",
            PastDate: PeriodDtype("D"),
            PastDate | None: PeriodDtype("D"),
        }

    modalities = dataset.ly.get_modalities()
    FullRecord = create_full_record_model(modalities)  # noqa: N806
    annotations = get_field_annotations(FullRecord)
    annotations = flatten(annotations, max_depth=3)
    dtypes = {}

    for col, annotation in annotations.items():
        if col not in dataset.columns:
            continue

        dtypes[col] = casters.get(annotation, object)

    result = dataset.astype(dtypes)
    logger.debug(f"Cast types: {dtypes}")
    return result


if __name__ == "__main__":
    logger.enable("lydata")
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    dataset = next(
        loader.load_datasets(
            repo_name="lycosystem/lydata.private",
            ref="6c56a630f307ffea12a2f071f18316f605beaa08",
        )
    )
    dataset = cast_types(dataset)
    # validate(dataset)
