"""Check that inferring sub- and super-levels works correctly."""

import pandas as pd
import pytest

import lydata  # noqa: F401
from lydata.augmentor import combine_and_augment_levels
from lydata.utils import get_default_modalities


@pytest.fixture(scope="module")
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


def test_clb_patient_17(clb_raw: pd.DataFrame) -> None:
    """Check the advanced combination and augmentation of diagnoses and levels."""
    modalities = get_default_modalities()
    modalities = {
        name: mod
        for name, mod in modalities.items()
        if name in clb_raw.columns.get_level_values(0)
    }
    clb_aug = combine_and_augment_levels(
        diagnoses=[clb_raw[mod] for mod in modalities.keys()],
        specificities=[mod.spec for mod in modalities.values()],
        sensitivities=[mod.sens for mod in modalities.values()],
    )
    assert len(clb_aug) == len(clb_raw), "Augmented data length mismatch"
    assert clb_aug.iloc[16].ipsi.I == False
    assert clb_aug.iloc[16].ipsi.Ia == False
    assert clb_aug.iloc[16].ipsi.Ib == False


def test_clb_patient_p011(clb_raw: pd.DataFrame) -> None:
    """Check that this patient's `NaN` values are handled correctly.

    In this patient, the sublvls are missing, therefore the superlvls should not be
    overridden by the augmentor.
    """
    idx = clb_raw.ly.id == "P011"
    patient = clb_raw.loc[idx]
    enhanced = patient.ly.enhance()
    assert enhanced.iloc[0].pathology.ipsi.II == patient.iloc[0].pathology.ipsi.II


def test_clb_patient_p035(clb_raw: pd.DataFrame) -> None:
    """Check that this patient's `NaN` values are handled correctly.

    In this patient, pathology reports ipsi.Ib as healthy, while diagnostic consensus
    reports ipsi.Ib as involved. This should correctly be combined to ipsi.Ib = False
    and the superlvl should also be set to False.
    """
    idx = clb_raw.ly.id == "P035"
    patient = clb_raw.loc[idx]
    enhanced = patient.ly.enhance()
    assert enhanced.iloc[0].max_llh.ipsi.I == False
    assert enhanced.iloc[0].max_llh.ipsi.Ib == False


def test_usz_patient_9() -> None:
    """Check the advanced combination and augmentation of diagnoses and levels."""
    usz_raw = next(
        lydata.load_datasets(year=2021, institution="usz", subsite="oropharynx")
    )
    modalities = get_default_modalities()
    modalities = {
        name: mod
        for name, mod in modalities.items()
        if name in usz_raw.columns.get_level_values(0)
    }
    usz_aug = combine_and_augment_levels(
        diagnoses=[usz_raw[mod] for mod in modalities.keys()],
        specificities=[mod.spec for mod in modalities.values()],
        sensitivities=[mod.sens for mod in modalities.values()],
    )
    assert len(usz_aug) == len(usz_raw), "Augmented data length mismatch"
    assert usz_aug.iloc[8].ipsi.III == False
