"""Test the ``.ly`` accessor for lyDATA DataFrames."""

import lydata  # noqa: F401


def test_enhance(usz_2021_df: lydata.LyDataFrame) -> None:
    """Test the enhance method of the ly accessor."""
    enhanced = usz_2021_df.ly.enhance()
    assert enhanced.shape == (287, 250)
    assert "max_llh" in enhanced.columns
    assert "Ia" in enhanced.max_llh.ipsi
    assert "Ib" in enhanced.max_llh.ipsi
    assert "I" in enhanced.max_llh.ipsi
