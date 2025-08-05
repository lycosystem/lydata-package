"""Tet some of the utility functions in `lydata.utils`."""

import pandas as pd

from lydata.utils import update_and_expand


def test_update_and_expand_using_p035(clb_raw: pd.DataFrame) -> None:
    """Check the `update_and_expand` function with a specific patient."""
    idx = clb_raw.ly.id == "P035"
    patient = clb_raw.loc[idx]
    combined = patient.ly.combine()
    combined = pd.concat({"test": combined}, axis="columns")
    augmented = combined.ly.augment(modality="test")
    augmented = pd.concat({"test": augmented}, axis="columns")
    result = update_and_expand(combined, augmented)
    assert len(result) == 1
    assert not pd.isna(result.iloc[0].test.ipsi.I)
