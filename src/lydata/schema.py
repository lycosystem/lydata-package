"""Pydantic schema to define a single patient record.

This schema is useful for casting dtypes, as done in :py:func:`validator.cast_dtypes`,
validation via :py:func:`~validator.is_valid`, and for exporting a JSON schema that
may be used for all kinds of purposes, e.g. to automatically generate HTML forms using
a `JSON-Editor`_.

.. _JSON-Editor: https://json-editor.github.io/json-editor/
"""

from __future__ import annotations

import json
import math
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
    "IVa",
    "IVb",
    "V",
    "Va",
    "Vb",
    "Vc",
    "VI",
    "VIa",
    "VIb",
    "VII",
    "VIIa",
    "VIIb",
    "VIII",
    "IX",
    "X",
    "Xa",
    "Xb",
]

_MIDLINE_FIELDS = (
    "side",
    "volume_ipsi",
    "volume_contra",
    "extension",
    "extension_dist",
    "max_extent_ipsi",
    "max_extent_contra",
)


def convert_nat(value: Any) -> Any:
    """Convert pandas NaT to None.

    pydantic throws an unspecific ``TypeError`` when ``pd.NaT`` is passed to a field.
    See [this issue on Github](https://github.com/pydantic/pydantic/issues/8039).
    """
    return None if pd.isna(value) else value


class PatientCore(BaseModel):
    """Basic required patient information.

    This includes demographic information, such as age and sex, as well as some risk
    factors for head and neck cancer, including HPV status, alcohol and nicotine abuse,
    etc.
    """

    id: str = Field(
        description=(
            "Unique but anonymized identifier for a patient. We commonly use the "
            "format `YYYY-<abr.>-<num>`, where `<abr.>` is an abbreviation of the "
            "institution (hospital) where the patient was treated."
        )
    )
    institution: str = Field(
        description="Name of the institution/hospital where the patient was treated."
    )
    sex: Literal["male", "female"] = Field(description="Biological sex of the patient.")
    age: int = Field(
        ge=0,
        le=120,
        description="Age of the patient at the time of diagnosis in years.",
    )
    diagnose_date: Annotated[PastDate, BeforeValidator(convert_nat)] = Field(
        description="Date of diagnosis of the patient (format YYYY-MM-DD)."
    )
    alcohol_abuse: bool | None = Field(
        description="Whether the patient currently abuses alcohol."
    )
    nicotine_abuse: bool | None = Field(
        description="Whether the patient currently abuses nicotine."
    )
    pack_years: float | None = Field(
        default=None,
        ge=0,
        description="Number of pack years of nicotine abuse.",
    )
    hpv_status: bool | None = Field(
        default=None,
        description="Whether the patient was infected with HPV.",
    )
    neck_dissection: bool | None = Field(
        description=(
            "Whether the patient underwent neck dissection as part of their treatment."
        ),
    )
    tnm_edition: int = Field(
        ge=6,
        le=9,
        default=8,
        description="Edition of the TNM classification used for staging.",
    )
    n_stage_prefix: Literal["c", "p"] | None = Field(
        default=None,
        description=(
            "Prefix for the N stage, 'c' = clinical, 'p' = pathological. "
            "This is used to distinguish between clinical and pathological staging."
        ),
    )
    n_stage: int = Field(
        ge=-1,
        le=3,
        description=(
            "N stage of the patient according to the TNM classification. The value -1 "
            "is reserved for the NX stage, which means that the lymph nodes could not "
            "be assessed for involvement."
        ),
    )
    n_stage_suffix: Literal["a", "b", "c"] | None = Field(
        default=None,
        description=(
            "Suffix for the N-stage according to the TNM classification. "
            "Can be 'a', 'b', or 'c'."
        ),
    )
    m_stage: int | None = Field(
        default=None,
        ge=-1,
        le=1,
        description=(
            "M stage of the patient according to the TNM classification. The value -1 "
            "is reserved for the MX stage, which technically doesn't exist, but it is "
            "commonly used."
        ),
    )
    weight: float | None = Field(
        default=None,
        ge=0,
        description="Weight of the patient in kg at the time of diagnosis.",
    )

    @field_validator(
        "alcohol_abuse",
        "nicotine_abuse",
        "pack_years",
        "hpv_status",
        "n_stage_prefix",
        "n_stage_suffix",
        "neck_dissection",
        "m_stage",
        "weight",
        mode="before",
    )
    @classmethod
    def nan_to_none(cls, value: Any) -> Any:
        """Convert NaN values to None to avoid pydantic errors."""
        return None if pd.isna(value) else value

    @field_validator(
        "sex",
        "n_stage_prefix",
        "n_stage_suffix",
        mode="before",
    )
    @classmethod
    def to_lower(cls, value: Any) -> Any:
        """Convert some string fields to lower case before validation."""
        if isinstance(value, str):
            return value.lower()

        return value


class PatientRecord(BaseModel):
    """A patient's record.

    Because the final dataset has a three-level header, this record holds only the
    key ``core`` under which we store the actual patient information defined in the
    :py:class:`PatientCore` model.

    Alongside ``core``, this may at some point hold additional or optional information
    about the patient.
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
        description="Whether the tumor is located on the mid-sagittal plane.",
        default=False,
    )
    extension: bool | None = Field(
        description="Whether the tumor extends over the mid-sagittal plane.",
        default=False,
    )
    dist_to_midline: float | None = Field(
        default=None,
        description="Distance of the tumor to the mid-sagittal plane in mm.",
    )
    volume: float | None = Field(
        default=None,
        ge=0,
        description="Estimated volume of the tumor in cm³.",
    )
    t_stage_prefix: Literal["c", "p"] = Field(
        default="c",
        description="Prefix for the tumor stage, 'c' = clinical, 'p' = pathological.",
    )
    t_stage: int = Field(
        ge=-1,
        le=4,
        description=(
            "T stage of the tumor according to the TNM classification. -1 is reserved "
            "for the TX stage, meaning the presence of tumor could not be assessed."
        ),
    )
    t_stage_suffix: Literal["is", "a", "b"] | None = Field(
        default=None,
        description=(
            "Suffix for the T-stage according to the TNM classification. "
            "Can be 'a' or 'b'. The value 'is' is reserved for the Tis stage, in which "
            "case the `t_stage` should be 0."
        ),
    )
    side: Literal["left", "right", "central"] | None = Field(
        default=None,
        description="Side of the neck where the main tumor mass is located.",
    )

    @field_validator(
        "central",
        "extension",
        "dist_to_midline",
        "volume",
        "t_stage_suffix",
        "side",
        mode="before",
    )
    @classmethod
    def nan_to_none(cls, value: Any) -> Any:
        """Convert NaN values to None."""
        return None if pd.isna(value) else value

    @field_validator(
        "location",
        "t_stage_prefix",
        "t_stage_suffix",
        "side",
        mode="before",
    )
    @classmethod
    def to_lower(cls, value: Any) -> Any:
        """Convert string values to lower case."""
        if isinstance(value, str):
            return value.lower()

        return value

    @model_validator(mode="after")
    def check_tumor_side(self) -> TumorCore:
        """Ensure tumor side information is consistent with ``central``."""
        if self.side == "central" and not self.central:
            raise ValueError(f"{self.central=}, but {self.side=}.")

        return self

    @model_validator(mode="after")
    def check_t_stage(self) -> TumorCore:
        """Ensure T-category is valid."""
        if self.t_stage == -1 and self.t_stage_suffix is not None:
            raise ValueError(
                f"{self.t_stage_suffix=}, but should be `None`, since "
                f"{self.t_stage=}, indicating TX stage.",
            )

        if self.t_stage_suffix == "is" and self.t_stage != 0:
            raise ValueError(
                f"T-stage 'Tis' is indicated by t_stage=0 and t_stage_suffix='is'. "
                f"But got {self.t_stage=} and {self.t_stage_suffix=}.",
            )

        if self.t_stage_suffix in ["a", "b"] and self.t_stage not in [1, 2, 3, 4]:
            raise ValueError(
                f"T-stage suffix {self.t_stage_suffix=} is only valid for T-stages "
                f"1, 2, 3, or 4, but got {self.t_stage=}.",
            )

        return self


class TumorAlgo(BaseModel):
    """Automatically extracted features of the tumor of a patient.

    This holds features that describe the primary tumor's position relative to
    the patient's midline. In contrast to the clinically assessed fields in
    :py:class:`TumorCore`, these are computed from the GTVp contour and an automatically
    detected midline. The features include the GTVp volume on each side of the midline
    and the distance by which the tumor crosses the midline, or how far it lies
    from it. Ipsi- and contralateral are defined by the `side` in the
    :py:class:`TumorCore`. Hence, for a few patients the ipsilateral part of the GTVp
    is smaller than the contralateral one.
    """

    volume: float | None = Field(
        default=None,
        ge=0,
        description="Volume of the GTVp contour in cm³.",
    )
    volume_ipsi: float | None = Field(
        default=None,
        ge=0,
        description=(
            "Volume of the GTVp contour on the ipsilateral side of the midline in cm³."
        ),
    )
    volume_contra: float | None = Field(
        default=None,
        ge=0,
        description=(
            "Volume of the GTVp contour on the contralateral side of the midline, "
            "in cm³."
        ),
    )
    extension: bool | None = Field(
        default=None,
        description=("Whether the GTVp contour extends across the detected midline."),
    )
    extension_dist: float | None = Field(
        default=None,
        description=(
            "Signed distance in mm between the midline and the GTVp's most "
            "contralateral edge. Positive values mean the tumor crosses the "
            "midline by that distance, negative values mean it stays on the "
            "ipsilateral side and lies that far from the midline."
        ),
    )
    width: float | None = Field(
        default=None,
        ge=0,
        description=("Left-right extent of the GTVp contour in mm."),
    )
    ipsi_dist: float | None = Field(
        default=None,
        ge=0,
        description=(
            "Maximum distance from the midline that the tumor reaches on the "
            "ipsilateral side, in mm."
        ),
    )
    max_diameter: float | None = Field(
        default=None,
        ge=0,
        description=(
            "Largest distance between any two points on the GTVp contour in mm."
        ),
    )

    @model_validator(mode="after")
    def check_midline_fields_complete(self) -> TumorAlgo:
        """Ensure midline-dependent features are either all present or all missing."""
        missing = [f for f in _MIDLINE_FIELDS if getattr(self, f) is None]
        if 0 < len(missing) < len(_MIDLINE_FIELDS):
            raise ValueError(
                "Midline-dependent features are only partially present. "
                f"Missing: {missing}"
            )
        return self

    @field_validator(
        "volume",
        "volume_ipsi",
        "volume_contra",
        "extension",
        "extension_dist",
        "width",
        "ipsi_dist",
        "max_diameter",
        mode="before",
    )
    @classmethod
    def nan_to_none(cls, value: Any) -> Any:
        """Convert NaN values to None."""
        return None if pd.isna(value) else value

    def _has_midline_features(self) -> bool:
        """Whether all midline-dependent features are present."""
        return all(getattr(self, f) is not None for f in _MIDLINE_FIELDS)

    @model_validator(mode="after")
    def check_extension(self) -> TumorAlgo:
        """Ensure extension information is consistent with contra metrics."""
        if not self._has_midline_features():
            return self
        checks = {
            "volume_contra > 0": self.volume_contra > 0,
            "extension_dist > 0": self.extension_dist > 0,
            "max_extent_contra > 0": self.max_extent_contra > 0,
        }
        wrong = [name for name, holds in checks.items() if holds != self.extension]
        if wrong:
            raise ValueError(f"{self.extension=}, but these disagree: {wrong}.")
        return self

    @model_validator(mode="after")
    def check_volume_sum(self) -> TumorAlgo:
        """Ensure ipsi- and contralateral volumes add up to total volume."""
        if self.volume is None or not self._has_midline_features():
            return self
        total = self.volume_ipsi + self.volume_contra
        if not math.isclose(total, self.volume, rel_tol=1e-3):
            raise ValueError(
                f"{self.volume_ipsi=} + {self.volume_contra=} not equal "
                f"to {self.volume=}."
            )
        return self


class TumorRecord(BaseModel):
    """A tumor record of a patient.

    As with the :py:class:`PatientRecord`, this holds only the key ``core`` under which
    we store the actual tumor information defined in the :py:class:`TumorCore` model.
    The key ``algo`` holds automatically extracted information related to the midline
    defined in the :py:class:`TumorAlgo` model.
    """

    core: TumorCore = Field(
        title="Core",
        description="Core information about the tumor.",
        default_factory=TumorCore,
    )
    algo: TumorAlgo = Field(
        title="Algo",
        description="Automatically extracted, midline-related features of"
        " the primary tumor.",
        default_factory=TumorAlgo,
    )


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
    model_name: str = "FullRecord",
    **kwargs: dict[str, Any],
) -> type:
    """Create a Pydantic model for a full record with all ``modalities``."""
    return create_model(
        model_name,
        __base__=BaseRecord,
        **{mod: create_modality_field(mod) for mod in modalities},
        **kwargs,
    )


def _write_schema_to_file(
    schema: type[BaseModel] | None = None,
    file_path: Path = Path("schema.json"),
) -> None:
    """Write the Pydantic schema to a file."""
    if schema is None:
        modalities = get_default_modalities()
        schema = create_full_record_model(modalities, model_name="Record")

    root_schema = RootModel[list[schema]]

    with open(file_path, "w") as f:
        json_schema = root_schema.model_json_schema()
        f.write(json.dumps(json_schema, indent=2))

    logger.success(f"Schema written to {file_path}")


if __name__ == "__main__":
    logger.enable("lydata")
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    _write_schema_to_file()
