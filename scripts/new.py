# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "lydata @ git+https://github.com/lycosystem/lydata-package@01a62ec12d504365bfd51302526cccc371bfd0ce",
# ]
# ///

"""Get all datasets and enhance them using the new lydata package."""

import pandas as pd
from loguru import logger

import lydata
from lydata.accessor import LyDataFrame

logger.enable("lydata")


def main() -> None:
    """Run the main function to load and enhance datasets."""
    full_dataset: LyDataFrame = pd.DataFrame()
    for dataset in lydata.load_datasets(
        repo_name="lycosystem/lydata.private",
        ref="ab04379a36b6946306041d1d38ad7e97df8ee7ba",
    ):
        full_dataset = pd.concat([full_dataset, dataset], ignore_index=True)
        logger.info(f"Added {len(dataset)=} rows to the full dataset.")

    enhanced_dataset = full_dataset.ly.enhance()
    added_cols = enhanced_dataset.shape[1] - full_dataset.shape[1]
    logger.info(f"Enhanced dataset has {added_cols} new columns.")

    enhanced_dataset.to_csv("new.csv", index=False)
    logger.success("Enhanced dataset saved to 'new.csv'.")


if __name__ == "__main__":
    main()
