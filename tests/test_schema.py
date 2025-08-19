"""Check the pydantic schema for the lyDATA format works."""

import datetime
from typing import Any

import pytest

from lydata.schema import (
    BaseRecord,
    PatientCore,
    PatientRecord,
    TumorCore,
    TumorRecord,
)


@pytest.fixture
def patient_core_dict() -> dict[str, Any]:
    """Fixture for a sample patient info."""
    return {
        "id": "12345",
        "institution": "Test Hospital",
        "sex": "female",
        "age": 42,
        "diagnose_date": "2023-01-01",
        "alcohol_abuse": False,
        "nicotine_abuse": True,
        "pack_years": 10.0,
        "hpv_status": True,
        "neck_dissection": True,
        "tnm_edition": 8,
        "n_stage": 1,
        "m_stage": 0,
    }


@pytest.fixture
def tumor_core_dict() -> dict[str, Any]:
    """Fixture for a sample tumor info."""
    return {
        "location": "gums",
        "subsite": "C03.9",
        "central": False,
        "extension": True,
        "stage_prefix": "c",
        "t_stage": 2,
    }


def test_patient_core(patient_core_dict: dict[str, Any]) -> None:
    """Test the PatientInfo schema."""
    patient_info = PatientCore(**patient_core_dict)

    for key, dict_value in patient_core_dict.items():
        model_value = getattr(patient_info, key)
        if isinstance(model_value, datetime.date):
            model_value = model_value.isoformat()
        assert model_value == dict_value, f"Mismatch for {key}"


def test_tumor_core(tumor_core_dict: dict[str, Any]) -> None:
    """Test the TumorInfo schema."""
    tumor_core = TumorCore(**tumor_core_dict)

    for key, value in tumor_core_dict.items():
        assert getattr(tumor_core, key) == value, f"Mismatch for {key}"


@pytest.fixture
def patient_core(patient_core_dict: dict[str, Any]) -> PatientCore:
    """Fixture for a sample PatientInfo instance."""
    return PatientCore(**patient_core_dict)


@pytest.fixture
def tumor_core(tumor_core_dict: dict[str, Any]) -> TumorCore:
    """Fixture for a sample TumorInfo instance."""
    return TumorCore(**tumor_core_dict)


def test_patient_record(patient_core: PatientCore) -> None:
    """Test the PatientRecord schema."""
    record = PatientRecord(core=patient_core)

    assert record.core == patient_core, "PatientRecord info does not match PatientInfo"


def test_tumor_record(tumor_core: TumorCore) -> None:
    """Test the TumorRecord schema."""
    record = TumorRecord(core=tumor_core)

    assert record.core == tumor_core, "TumorRecord info does not match TumorInfo"


@pytest.fixture
def complete_record(patient_core: PatientCore, tumor_core: TumorCore) -> BaseRecord:
    """Fixture for a sample CompleteRecord instance."""
    return BaseRecord(
        patient=PatientRecord(core=patient_core),
        tumor=TumorRecord(core=tumor_core),
    )


def test_complete_record(complete_record: BaseRecord) -> None:
    """Test the CompleteRecord schema."""
    assert complete_record.patient.core.id == "12345", "Patient ID does not match"
