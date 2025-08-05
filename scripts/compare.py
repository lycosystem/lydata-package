# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "lydata @ git+https://github.com/lycosystem/lydata-package@01a62ec12d504365bfd51302526cccc371bfd0ce",
#     "typer",
# ]
# ///

"""Compare new and old datasets for differences."""

from pathlib import Path

import pandas as pd
from loguru import logger

from lydata.utils import _sort_all


def main(old_file: Path, new_file: Path) -> None:
    """Compare old and new datasets."""
    old = pd.read_csv(old_file, header=[0, 1, 2])
    logger.info(f"Loading old dataset from {old_file}")
    new = pd.read_csv(new_file, header=[0, 1, 2])
    logger.info(f"Loading new dataset from {new_file}")

    old, new = old.align(new, axis="columns", join="outer")
    is_equal = (old == new) | (old.isna() & new.isna())
    logger.info(f"Num of different cells (total): {(~is_equal).sum().sum()}")
    is_equal = (old.max_llh == new.max_llh) | (old.max_llh.isna() & new.max_llh.isna())
    logger.info(f"Num of different cells (max_llh): {(~is_equal).sum().sum()}")

    old = _sort_all(old)
    new = _sort_all(new)

    old_diff, new_diff = [], []

    for (_, old_row), (_, new_row) in zip(old.iterrows(), new.iterrows(), strict=True):
        is_equal = (old_row == new_row) | (old_row.isna() & new_row.isna())
        if not is_equal.all():
            old_diff.append(old_row)
            new_diff.append(new_row)

    old_diff = pd.DataFrame(old_diff, columns=old.columns)
    new_diff = pd.DataFrame(new_diff, columns=new.columns)

    old_file = old_file.with_suffix(".diff.csv")
    new_file = new_file.with_suffix(".diff.csv")

    old_diff.to_csv(old_file, index=False)
    logger.success(f"Saving {old_diff.shape=} diff to {old_file}")
    new_diff.to_csv(new_file, index=False)
    logger.success(f"Saving {new_diff.shape=} diff to {new_file}")


if __name__ == "__main__":
    import typer

    typer.run(main)
