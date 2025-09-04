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
            repo_name="lycosystem/lydata.private",
            ref="e68141fd5440d4cfa6491df14ca2203ddb7946b0",
            cast=True,
        ),
    )


@pytest.fixture(scope="session")
def usz_2021_df() -> pd.DataFrame:
    """Load the CLB dataset."""
    return next(
        lydata.load_datasets(
            year=2021,
            institution="usz",
            repo_name="lycosystem/lydata.private",
            ref="fb55afa26ff78afa78274a86b131fb3014d0ceea",
            cast=True,
        )
    )


@pytest.fixture(scope="session")
def usz_2025_df() -> lydata.LyDataFrame:
    """Fixture to load a sample DataFrame from the USZ 2025 dataset."""
    return next(
        lydata.load_datasets(
            year=2025,
            institution="usz",
            repo_name="lycosystem/lydata.private",
            ref="c11011aa928fe43f18e73e42577a0fcee5652d99",
            cast=True,
        )
    )
