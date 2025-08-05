# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "lydata==0.3.3",
# ]
# ///

"""Get all datasets and enhance them using the old lydata package."""

import pandas as pd
from loguru import logger

import lydata

logger.enable("lydata")


def main() -> None:
    """Run the main function to load and enhance datasets."""
    full_dataset = pd.DataFrame()
    for dataset in lydata.load_datasets(
        repo_name="lycosystem/lydata.private",
        ref="ab04379a36b6946306041d1d38ad7e97df8ee7ba",
    ):
        full_dataset = pd.concat([full_dataset, dataset], ignore_index=True)
        logger.info(f"Added {len(dataset)=} rows to the full dataset.")

    enhanced_dataset = lydata.infer_and_combine_levels(full_dataset)
    added_cols = enhanced_dataset.shape[1] - full_dataset.shape[1]
    logger.info(f"Enhanced dataset has {added_cols} new columns.")

    enhanced_dataset.to_csv("old.csv", index=False)
    logger.success("Enhanced dataset saved to 'old.csv'.")


if __name__ == "__main__":
    main()
