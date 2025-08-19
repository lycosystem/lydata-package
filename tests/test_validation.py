"""Test the casting and validation of lydata datasets."""

from lydata import load_datasets
from lydata.validator import cast_dtypes


def test_casting() -> None:
    """Test the casting of a dataset."""
    clb_raw = next(
        load_datasets(
            year=2021,
            institution="clb",
            subsite="oropharynx",
            use_github=True,
            repo_name="lycosystem/lydata.private",
            ref="6c56a630f307ffea12a2f071f18316f605beaa08",
        )
    )
    clb_casted = cast_dtypes(clb_raw)

    assert clb_casted.patient.core.id.dtype == "string"
    assert clb_casted.patient.core.age.dtype == "int64"
    assert clb_casted.patient.core.diagnose_date.dtype == "period[D]"
    assert clb_casted.tumor.core.t_stage.dtype == "Int64"
