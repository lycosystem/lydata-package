"""Check the pydantic schema for the lyDATA format works."""

import datetime
from typing import Any

import pytest

from lydata.schema import (
    BaseRecord,
    PatientInfo,
    PatientRecord,
    TumorInfo,
    TumorRecord,
)


@pytest.fixture
def patient_info_dict() -> dict[str, Any]:
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
def tumor_info_dict() -> dict[str, Any]:
    """Fixture for a sample tumor info."""
    return {
        "location": "gums",
        "subsite": "C03.9",
        "central": False,
        "extension": True,
        "stage_prefix": "c",
        "t_stage": 2,
    }


def test_patient_info(patient_info_dict: dict[str, Any]) -> None:
    """Test the PatientInfo schema."""
    patient_info = PatientInfo(**patient_info_dict)

    for key, dict_value in patient_info_dict.items():
        model_value = getattr(patient_info, key)
        if isinstance(model_value, datetime.date):
            model_value = model_value.isoformat()
        assert model_value == dict_value, f"Mismatch for {key}"


def test_tumor_info(tumor_info_dict: dict[str, Any]) -> None:
    """Test the TumorInfo schema."""
    tumor_info = TumorInfo(**tumor_info_dict)

    for key, value in tumor_info_dict.items():
        assert getattr(tumor_info, key) == value, f"Mismatch for {key}"


@pytest.fixture
def patient_info(patient_info_dict: dict[str, Any]) -> PatientInfo:
    """Fixture for a sample PatientInfo instance."""
    return PatientInfo(**patient_info_dict)


@pytest.fixture
def tumor_info(tumor_info_dict: dict[str, Any]) -> TumorInfo:
    """Fixture for a sample TumorInfo instance."""
    return TumorInfo(**tumor_info_dict)


def test_patient_record(patient_info: PatientInfo) -> None:
    """Test the PatientRecord schema."""
    record = PatientRecord(_=patient_info)

    assert record.info == patient_info, "PatientRecord info does not match PatientInfo"


def test_tumor_record(tumor_info: TumorInfo) -> None:
    """Test the TumorRecord schema."""
    record = TumorRecord(_=tumor_info)

    assert record.info == tumor_info, "TumorRecord info does not match TumorInfo"


@pytest.fixture
def complete_record(patient_info: PatientInfo, tumor_info: TumorInfo) -> BaseRecord:
    """Fixture for a sample CompleteRecord instance."""
    return BaseRecord(
        patient=PatientRecord(_=patient_info),
        tumor=TumorRecord(_=tumor_info),
    )


def test_complete_record(complete_record: BaseRecord) -> None:
    """Test the CompleteRecord schema."""
    assert complete_record.patient.info.id == "12345", "Patient ID does not match"
