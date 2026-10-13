"""Pydantic schema to define a single patient record.

This schema is useful for casting dtypes, as done in :py:func:`validator.cast_dtypes`,
validation via :py:func:`~validator.is_valid`, and for exporting a JSON schema that
may be used for all kinds of purposes, e.g. to automatically generate HTML forms using
a `JSON-Editor`_.

The site-independent building blocks are defined in :py:mod:`lydata.schema.base`,
while the site-specific parts (e.g. the lymph node levels and the patient and tumor
information) live in their own modules, like :py:mod:`lydata.schema.head_neck`. For
backwards compatibility, the head and neck schema is re-exported here and used by
default.

.. _JSON-Editor: https://json-editor.github.io/json-editor/
"""

from __future__ import annotations

from lydata.schema.base import (
    ModalityCore,
    convert_nat,
    create_lnl_field,
)
from lydata.schema.head_neck import (
    _LNLS,
    BaseRecord,
    ModalityRecord,
    PatientCore,
    PatientRecord,
    TumorCore,
    TumorRecord,
    UnilateralInvolvementInfo,
    create_full_record_model,
    create_modality_field,
)
from lydata.utils import get_default_modalities

__all__ = [
    "_LNLS",
    "BaseRecord",
    "ModalityCore",
    "ModalityRecord",
    "PatientCore",
    "PatientRecord",
    "TumorCore",
    "TumorRecord",
    "UnilateralInvolvementInfo",
    "convert_nat",
    "create_full_record_model",
    "create_lnl_field",
    "create_modality_field",
    "get_default_modalities",
]
