"""Fixtures for testing lydata functionality."""

import pandas as pd
import pytest

import lydata


@pytest.fixture(scope="session")
def clb_raw() -> pd.DataFrame:
    """Load the CLB dataset."""
    return next(
        lydata.load_datasets(
            year=2021,
            institution="clb",
            subsite="oropharynx",
            use_github=True,
            ref="4668ff6006764169411d6d198c126b020a7892b2",
        ),
    )
