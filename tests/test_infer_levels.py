"""Check that inferring sub- and super-levels works correctly."""

import lydata  # noqa: F401
from lydata.augmentor import combine_and_augment_levels
from lydata.utils import get_default_modalities


def test_augment_levels() -> None:
    """Check the advanced combination and augmentation of diagnoses and levels."""
    clb_raw = next(lydata.load_datasets(institution="clb", subsite="oropharynx"))
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
