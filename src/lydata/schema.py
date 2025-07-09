"""Pydantic schema to define a single patient record.

Based on such a schema, pandera can create a DataFrameSchema to validate lyDATA sets.
Also, it may be used to create a HTML form to enter patient data.
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class PatientInfo(BaseModel):
    """Basic required patient information."""

    id: str | int = Field(description="Unique but anonymized identifier for a patient.")
    institution: str = Field(description="Hospital where the patient was treated.")
    sex: Literal["male", "female"] = Field(description="Biological sex of the patient.")
    age: int = Field(
        ge=0,
        le=120,
        description="Age of the patient at the time of diagnosis in years.",
    )
    diagnose_date: date = Field(description="Date of diagnosis of the patient.")
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
    n_stage: int = Field(
        ge=0,
        le=4,
        description="N stage of the patient according to the TNM classification.",
    )
    m_stage: int | None = Field(
        ge=0,
        le=1,
        description="M stage of the patient according to the TNM classification.",
    )


class PatientRecord(BaseModel):
    """A patient's record.

    As of now, this only contains the patient information.
    """

    info: PatientInfo = Field(default_factory=PatientInfo, alias="_")


class TumorInfo(BaseModel):
    """Information about the tumor of a patient."""

    location: str = Field(description="Primary tumor location.")
    subsite: str = Field(
        description="ICD-O-3 subsite of the primary tumor.",
        pattern=r"C[0-9]{2}(\.[0-9X])?",
    )
    central: bool = Field(
        description="Whether the tumor is located on the mid-sagittal line.",
        default=False,
    )
    extension: bool = Field(
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
    t_stage: int = Field(
        ge=0,
        le=4,
        description="T stage of the tumor according to the TNM classification.",
    )


class TumorRecord(BaseModel):
    """A tumor record of a patient.

    As of now, this only contains the tumor information.
    """

    info: TumorInfo = Field(default_factory=TumorInfo, alias="_")


class CompleteRecord(BaseModel):
    """A complete patient record.

    This combines the patient and tumor records.
    """

    patient: PatientRecord = Field(default_factory=PatientRecord)
    tumor: TumorRecord = Field(default_factory=TumorRecord)
