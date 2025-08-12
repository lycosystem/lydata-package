"""Pydantic schema to define a single patient record.

Based on such a schema, pandera can create a DataFrameSchema to validate lyDATA sets.
Also, it may be used to create a HTML form to enter patient data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any, Literal

import pandas as pd
from loguru import logger
from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    PastDate,
    RootModel,
    create_model,
    field_validator,
    model_validator,
)

from lydata.utils import get_default_modalities

_LNLS = [
    "I",
    "Ia",
    "Ib",
    "II",
    "IIa",
    "IIb",
    "III",
    "IV",
    "V",
    "Va",
    "Vb",
    "VI",
    "VII",
    "VIII",
    "IX",
    "X",
]


class PatientCore(BaseModel):
    """Basic required patient information.

    This includes demographic information, such as age and sex, as well as some risk
    factors for head and neck cancer, including HPV status, alcohol and nicotine abuse,
    etc.
    """

    id: str = Field(description="Unique but anonymized identifier for a patient.")
    institution: str = Field(description="Hospital where the patient was treated.")
    sex: Literal["male", "female"] = Field(description="Biological sex of the patient.")
    age: int = Field(
        ge=0,
        le=120,
        description="Age of the patient at the time of diagnosis in years.",
    )
    diagnose_date: PastDate = Field(description="Date of diagnosis of the patient.")
    alcohol_abuse: bool = Field(description="Whether the patient abused alcohol.")
    nicotine_abuse: bool = Field(description="Whether the patient abused nicotine.")
    pack_years: float | None = Field(
        default=None,
        ge=0,
        description="Number of pack years of nicotine abuse.",
    )
    hpv_status: bool | None = Field(
        default=None,
        description="Whether the patient was infected with HPV.",
    )
    neck_dissection: bool = Field(description="Did the patient have a neck dissection?")
    tnm_edition: int = Field(
        ge=6,
        le=8,
        default=8,
        description="Edition of the TNM classification used for staging.",
    )
    n_stage: int | None = Field(
        ge=0,
        le=3,
        description="N stage of the patient according to the TNM classification.",
    )
    m_stage: int | None = Field(
        ge=0,
        le=1,
        description="M stage of the patient according to the TNM classification.",
    )

    @field_validator("pack_years", "hpv_status", "m_stage", mode="before")
    @classmethod
    def nan_to_none(cls, value: Any) -> Any:
        """Convert NaN values to None."""
        return None if pd.isna(value) else value

    @model_validator(mode="after")
    def check_nicotine_and_pack_years(self) -> PatientCore:
        """Ensure that if nicotine abuse is False, pack_years is not > 0."""
        if not self.nicotine_abuse and (
            self.pack_years is not None and self.pack_years > 0
        ):
            raise ValueError("If nicotine abuse is False, pack_years cannot be > 0.")

        return self


class PatientRecord(BaseModel):
    """A patient's record.

    Because the final dataset has a three-level header, this record holds only the
    key ``core`` under which we store the actual patient information defined in the
    :py:class:`PatientCore` model.
    """

    core: PatientCore = Field(
        title="Core",
        description="Core information about the patient.",
        default_factory=PatientCore,
    )


class TumorCore(BaseModel):
    """Information about the tumor of a patient.

    This information characterizes the primary tumor via its location, ICD-O-3 subsite,
    T-category and so on.
    """

    location: str = Field(description="Primary tumor location.")
    subsite: str = Field(
        description="ICD-O-3 subsite of the primary tumor.",
        pattern=r"C[0-9]{2}(\.[0-9X])?",
    )
    central: bool | None = Field(
        description="Whether the tumor is located on the mid-sagittal line.",
        default=False,
    )
    extension: bool | None = Field(
        description="Whether the tumor extends over the mid-sagittal line.",
        default=False,
    )
    volume: float | None = Field(
        default=None,
        ge=0,
        description="Estimated volume of the tumor in cm³.",
    )
    stage_prefix: Literal["c", "p"] = Field(
        default="c",
        description="Prefix for the tumor stage, 'c' = clinical, 'p' = pathological.",
    )
    t_stage: int | None = Field(
        ge=0,
        le=4,
        description="T stage of the tumor according to the TNM classification.",
    )

    @field_validator("central", "volume", mode="before")
    @classmethod
    def nan_to_none(cls, value: Any) -> Any:
        """Convert NaN values to None."""
        return None if pd.isna(value) else value


class TumorRecord(BaseModel):
    """A tumor record of a patient.

    As with the patient record, this holds only the key ``core`` under which we
    store the actual tumor information defined in the :py:class:`TumorCore` model.
    """

    core: TumorCore = Field(
        title="Core",
        description="Core information about the tumor.",
        default_factory=TumorCore,
    )


def create_lnl_field(lnl: str) -> tuple[type, Field]:
    """Create a field for a specific lymph node level."""
    return (
        Annotated[bool | None, BeforeValidator(lambda v: None if pd.isna(v) else v)],
        Field(default=None, description=f"LN {lnl} involvement"),
    )


class ModalityCore(BaseModel):
    """Basic info about a diagnostic/pathological modality."""

    date: PastDate | None = Field(
        description="Date of the diagnostic or pathological modality.",
        default=None,
    )


UnilateralInvolvementInfo = create_model(
    "UnilateralInvolvementInfo",
    **{lnl: create_lnl_field(lnl) for lnl in _LNLS},
)


class ModalityRecord(BaseModel):
    """Involvement patterns of a diagnostic or pathological modality.

    This holds some basic information about the modality, which is currently limited to
    the date its information was collected (e.g. the date of the PET/CT scan).

    Most importantly, this holds the ipsi- and contralateral lymph node level
    involvement patterns under the respective keys ``ipsi`` and ``contra``.
    """

    core: ModalityCore = Field(
        title="Core",
        default_factory=ModalityCore,
    )
    ipsi: UnilateralInvolvementInfo = Field(
        title="Ipsilateral Involvement",
        description="Involvement patterns of the ipsilateral side.",
        default_factory=UnilateralInvolvementInfo,
    )
    contra: UnilateralInvolvementInfo = Field(
        title="Contralateral Involvement",
        description="Involvement patterns of the contralateral side.",
        default_factory=UnilateralInvolvementInfo,
    )


def create_modality_field(modality: str) -> tuple[type, Field]:
    """Create a field for a specific modality."""
    return (
        ModalityRecord,
        Field(
            title=modality,
            description=f"Involvement patterns as observed using {modality}.",
            default_factory=ModalityRecord,
        ),
    )


class BaseRecord(BaseModel):
    """A basic record of a patient.

    Contains at least the patient and tumor information in the same nested form
    as the data represents it.
    """

    patient: PatientRecord = Field(
        title="Patient",
        description=(
            "Characterizes the patient via demographic information and risk factors "
            "associated with head and neck cancer. In order to achieve the three-level "
            "header structure in the final table, there is a subkey `core` under which "
            "the actual patient information is stored."
        ),
        default_factory=PatientRecord,
    )
    tumor: TumorRecord = Field(
        title="Tumor",
        description=(
            "Characterizes the primary tumor via its location, ICD-O-3 subsite, "
            "T-category and so on. As with the patient record, this has a subkey "
            "`core` under which the actual tumor information is stored."
        ),
        default_factory=TumorRecord,
    )


def create_full_record_model(
    modalities: list[str],
    title: str = "FullRecord",
    **kwargs: dict[str, Any],
) -> type:
    """Create a Pydantic model for a full record with all modalities."""
    return create_model(
        title,
        __base__=BaseRecord,
        **{mod: create_modality_field(mod) for mod in modalities},
        **kwargs,
    )


def write_schema_to_file(
    schema: type[BaseModel] | None = None,
    file_path: Path = Path("schema.json"),
) -> None:
    """Write the Pydantic schema to a file."""
    if schema is None:
        modalities = get_default_modalities()
        schema = create_full_record_model(modalities, title="Record")

    root_schema = RootModel[list[schema]]

    with open(file_path, "w") as f:
        json_schema = root_schema.model_json_schema()
        f.write(json.dumps(json_schema, indent=2))

    logger.success(f"Schema written to {file_path}")


if __name__ == "__main__":
    logger.enable("lydata")
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    write_schema_to_file()
