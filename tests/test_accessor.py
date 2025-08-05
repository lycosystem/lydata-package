"""Test the ``.ly`` accessor for lyDATA DataFrames."""

import pytest

import lydata  # noqa: F401
from lydata import loader


@pytest.fixture
def usz_df() -> lydata.LyDataFrame:
    """Fixture to load a sample DataFrame from the USZ dataset."""
    return next(
        loader.load_datasets(
            year=2021,
            institution="usz",
            repo_name="lycosystem/lydata.private",
            ref="fb55afa26ff78afa78274a86b131fb3014d0ceea",
        )
    )


def test_enhance(usz_df: lydata.LyDataFrame) -> None:
    """Test the enhance method of the ly accessor."""
    enhanced = usz_df.ly.enhance()
    assert enhanced.shape == (287, 244)
    assert "max_llh" in enhanced.columns
    assert "Ia" in enhanced.max_llh.ipsi
    assert "Ib" in enhanced.max_llh.ipsi
    assert "I" in enhanced.max_llh.ipsi
