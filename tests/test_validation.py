"""Test the casting and validation of lydata datasets."""

import pandas as pd

from lydata.validator import cast_dtypes


def test_casting(clb_raw: pd.DataFrame) -> None:
    """Test the casting of a dataset."""
    clb_casted = cast_dtypes(clb_raw)

    assert clb_casted.patient.core.id.dtype == "string"
    assert clb_casted.patient.core.age.dtype == "Int64"
    assert clb_casted.patient.core.diagnose_date.dtype == "datetime64[ns]"
    assert clb_casted.tumor.core.t_stage.dtype == "Int64"
