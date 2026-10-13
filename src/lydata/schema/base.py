"""Site-independent building blocks for the patient record schemas.

The functions in this module assemble the full record model from site-specific parts
(the list of lymph node levels and the patient and tumor records), which are defined in
the respective site modules, e.g. :py:mod:`lydata.schema.head_neck`.
"""

from __future__ import annotations

from typing import Annotated, Any

import pandas as pd
from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    PastDate,
    create_model,
)


def convert_nat(value: Any) -> Any:
    """Convert pandas NaT to None.

    pydantic throws an unspecific ``TypeError`` when ``pd.NaT`` is passed to a field.
    See [this issue on Github](https://github.com/pydantic/pydantic/issues/8039).
    """
    return None if pd.isna(value) else value


def create_lnl_field(lnl: str) -> tuple[type, Field]:
    """Create a field for a specific lymph node level."""
    return (
        Annotated[bool | None, BeforeValidator(lambda v: None if pd.isna(v) else v)],
        Field(default=None, description=f"LN {lnl} involvement"),
    )


class ModalityCore(BaseModel):
    """Basic info about a diagnostic/pathological modality."""

    date: Annotated[PastDate | None, BeforeValidator(convert_nat)] = Field(
        description="Date of the diagnostic or pathological modality.",
        default=None,
    )


def create_involvement_model(
    lnls: list[str],
    model_name: str = "UnilateralInvolvementInfo",
) -> type[BaseModel]:
    """Create a model holding the involvement of one side for all ``lnls``."""
    return create_model(
        model_name,
        **{lnl: create_lnl_field(lnl) for lnl in lnls},
    )


class BaseModalityRecord(BaseModel):
    """Basic record of a diagnostic or pathological modality.

    Site-specific subclasses add the ipsi- and contralateral involvement patterns
    under the keys ``ipsi`` and ``contra``.
    """

    core: ModalityCore = Field(
        title="Core",
        default_factory=ModalityCore,
    )


def create_modality_field(
    modality: str,
    modality_record: type[BaseModel],
) -> tuple[type, Field]:
    """Create a field for a specific modality."""
    return (
        modality_record,
        Field(
            title=modality,
            description=f"Involvement patterns as observed using {modality}.",
            default_factory=modality_record,
        ),
    )


def create_full_record_model(
    modalities: list[str],
    base_record: type[BaseModel],
    modality_record: type[BaseModel],
    model_name: str = "FullRecord",
    **kwargs: dict[str, Any],
) -> type:
    """Create a Pydantic model for a full record with all ``modalities``.

    The ``base_record`` holds the patient and tumor information of a specific site,
    while the ``modality_record`` defines the involvement information per modality.
    """
    return create_model(
        model_name,
        __base__=base_record,
        **{mod: create_modality_field(mod, modality_record) for mod in modalities},
        **kwargs,
    )
