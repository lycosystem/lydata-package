"""Check that inferring sub- and super-levels works correctly."""

import pandas as pd

import lydata  # noqa: F401
from lydata.augmentor import combine_and_augment_levels
from lydata.utils import ModalityConfig, get_default_modalities


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


def test_2021_clb_001(clb_raw: pd.DataFrame) -> None:
    """Check that this patient's `NaN` values are handled correctly.

    In this patient, the sublvls are missing, therefore the superlvls should not be
    overridden by the augmentor.
    """
    idx = clb_raw.ly.id == "2021-CLB-001"
    patient = clb_raw.loc[idx]
    enhanced = patient.ly.enhance()
    assert enhanced.iloc[0].pathology.ipsi.II == patient.iloc[0].pathology.ipsi.II


def test_2021_clb_017(clb_raw: pd.DataFrame) -> None:
    """Check that this patient's `NaN` values are handled correctly.

    In this patient, pathology reports ipsi.Ib as healthy, while diagnostic consensus
    reports ipsi.Ib as involved. This should correctly be combined to ipsi.Ib = False
    and the superlvl should also be set to False.
    """
    idx = clb_raw.ly.id == "2021-CLB-017"
    patient = clb_raw.loc[idx]
    enhanced = patient.ly.enhance()
    assert len(patient) == len(enhanced) == 1, "Patient data length mismatch"
    assert enhanced.iloc[0].max_llh.ipsi.I == False
    assert enhanced.iloc[0].max_llh.ipsi.Ib == False


def test_2021_usz_009(usz_2021_df: pd.DataFrame) -> None:
    """Check the advanced combination and augmentation of diagnoses and levels."""
    modalities = get_default_modalities()
    modalities = {
        name: mod
        for name, mod in modalities.items()
        if name in usz_2021_df.columns.get_level_values(0)
    }
    usz_aug = combine_and_augment_levels(
        diagnoses=[usz_2021_df[mod] for mod in modalities.keys()],
        specificities=[mod.spec for mod in modalities.values()],
        sensitivities=[mod.sens for mod in modalities.values()],
    )
    assert len(usz_aug) == len(usz_2021_df), "Augmented data length mismatch"
    assert usz_aug.iloc[8].ipsi.III == False


def test_2025_usz_080(usz_2025_df: lydata.LyDataFrame) -> None:
    """Check that this patient..."""
    idx = usz_2025_df.ly.id == "2025-USZ-080"
    patient = usz_2025_df.loc[idx]
    enhanced = patient.ly.enhance()
    assert enhanced.iloc[0].max_llh.ipsi.II == True
    assert pd.isna(enhanced.iloc[0].max_llh.ipsi.IIa)
    assert pd.isna(enhanced.iloc[0].max_llh.ipsi.IIb)


def test_2025_usz_312(usz_2025_df: lydata.LyDataFrame) -> None:
    """Check that this patient..."""
    idx = usz_2025_df.ly.id == "2025-USZ-312"
    patient = usz_2025_df.loc[idx]
    assert len(patient) == 1
    assert patient.ly.date.iloc[0].strftime("%Y-%m-%d") == "2013-06-03"

    enhanced = patient.ly.enhance()
    assert len(enhanced) == 1
    assert enhanced.iloc[0].max_llh.ipsi.II == False


def test_2025_usz_075(usz_2025_df: lydata.LyDataFrame) -> None:
    """Ensure patient 2025-USZ-075 is correctly enhanced.

    This patient has a pathologically (FNA) confirmed contra II involvement, but PET
    and planning CT (pCT) are negative. Depending on the sensitivity and specificity
    values, this leads to a max_llh of True or False for the contra II level.
    """
    idx = usz_2025_df.ly.id == "2025-USZ-075"
    patient = usz_2025_df.loc[idx]
    assert len(patient) == 1
    assert patient.ly.date.iloc[0].strftime("%Y-%m-%d") == "2015-11-23"
    assert patient.FNA.contra.II.iloc[0] == True

    enhanced = patient.ly.enhance(
        modalities={
            "PET": ModalityConfig(spec=0.86, sens=0.79),
            "FNA": ModalityConfig(spec=0.98, sens=0.80, kind="pathological"),
            # "pCT": ModalityConfig(spec=0.86, sens=0.81),
        }
    )
    assert len(enhanced) == 1
    assert enhanced.iloc[0].max_llh.contra.II == True
