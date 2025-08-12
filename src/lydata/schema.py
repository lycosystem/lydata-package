"""Pydantic schema to define a single patient record.

Based on such a schema, pandera can create a DataFrameSchema to validate lyDATA sets.
Also, it may be used to create a HTML form to enter patient data.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

import pandas as pd
from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    PastDate,
    create_model,
    field_validator,
    model_validator,
)

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


class PatientInfo(BaseModel):
    """Basic required patient information."""

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
    def check_nicotine_and_pack_years(self) -> PatientInfo:
        """Ensure that if nicotine abuse is False, pack_years is not > 0."""
        if not self.nicotine_abuse and (
            self.pack_years is not None and self.pack_years > 0
        ):
            raise ValueError("If nicotine abuse is False, pack_years cannot be > 0.")

        return self


class PatientRecord(BaseModel):
    """A patient's record.

    As of now, this only contains the patient information.
    """

    core: PatientInfo = Field(default_factory=PatientInfo)


class TumorInfo(BaseModel):
    """Information about the tumor of a patient."""

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

    As of now, this only contains the tumor information.
    """

    core: TumorInfo = Field(default_factory=TumorInfo)


def create_lnl_field(lnl: str) -> tuple[type, Field]:
    """Create a field for a specific lymph node level."""
    return (
        Annotated[bool | None, BeforeValidator(lambda v: None if pd.isna(v) else v)],
        Field(default=None, description=f"LN {lnl} involvement"),
    )


class ModalityInfo(BaseModel):
    """Basic info about a diagnostic/pathological modality.

    Contains only the date of the modality as of now.
    """

    date: PastDate | None = Field(
        description="Date of the diagnostic or pathological modality.",
        default=None,
    )


UnilateralInvolvementInfo = create_model(
    "UnilateralInvolvementInfo",
    **{lnl: create_lnl_field(lnl) for lnl in _LNLS},
)


class ModalityRecord(BaseModel):
    """A record of the involvement patterns of a diagnostic or pathological modality."""

    core: ModalityInfo = Field(default_factory=ModalityInfo)
    ipsi: UnilateralInvolvementInfo = Field(
        description="Unilateral involvement of the ipsilateral side.",
        default_factory=UnilateralInvolvementInfo,
    )
    contra: UnilateralInvolvementInfo = Field(
        description="Unilateral involvement of the contralateral side.",
        default_factory=UnilateralInvolvementInfo,
    )


def create_modality_field(modality: str) -> tuple[type, Field]:
    """Create a field for a specific modality."""
    return (
        ModalityRecord,
        Field(
            default_factory=ModalityRecord,
            description=f"Involvement data for modality {modality}",
        ),
    )


class BaseRecord(BaseModel):
    """A basic record of a patient.

    Contains at least the patient and tumor information in the same nested form
    as the data represents it.
    """

    patient: PatientRecord = Field(default_factory=PatientRecord)
    tumor: TumorRecord = Field(default_factory=TumorRecord)


def create_full_record_model(modalities: list[str]) -> type:
    """Create a Pydantic model for a full record with all modalities."""
    return create_model(
        "FullRecord",
        __base__=BaseRecord,
        **{mod: create_modality_field(mod) for mod in modalities},
    )
