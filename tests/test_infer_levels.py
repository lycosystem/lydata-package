"""Check that inferring sub- and super-levels works correctly."""

import pandas as pd
import pytest

import lydata  # noqa: F401


@pytest.fixture
def mock_data() -> pd.DataFrame:
    """Create a mock dataset for testing."""
    return pd.DataFrame({
        ("MRI", "ipsi",   "Ia" ): [True , False, False, None, None ],
        ("MRI", "ipsi",   "Ib" ): [False, True , False, None, False],
        ("MRI", "contra", "IIa"): [False, False, None , None, None ],
        ("MRI", "contra", "IIb"): [False, True , True , None, False],
        ("CT",  "ipsi",   "I"  ): [True , False, False, None, None ],
    })


def test_infer_superlevels(mock_data: pd.DataFrame) -> None:
    """Check that superlevels are inferred correctly."""
    inferred = mock_data.ly.infer_superlevels(modalities=["MRI"])

    expected_ipsi_I = [True, True, False, None, None]
    expected_contra_II = [False, True, True, None, None]

    for example in range(len(mock_data)):
        assert (
            inferred.iloc[example].MRI.ipsi.I
            == expected_ipsi_I[example]
        ), f"{example = } mismatch for ipsi I"
        assert (
            inferred.iloc[example].MRI.contra.II
            == expected_contra_II[example]
        ), f"{example = } mismatch for contra II"
