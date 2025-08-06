# Upgrade from lydata 0.3.3 to 0.4.0

Since we noticed [an issue with how the data is combined], a new version is in the works. To double check that now - finally - everything is working as expected, we need to compare the old and new data.

## Reproduce Comparison Files

This `scripts` directory contains the `new.py` and `old.py` scripts, which fetch the exact same data from the lyDATA repo, but once use the old version and once the new dev version.

> [!IMPORTANT]
> To run these scripts, you need to have [`uv`] installed, because - as far as I know - only this tool allows automatically installing the required dependencies from the [inline script metadata] and puts them in a temporary virtual environment.

Simply run the following command in the `scripts` directory:

```bash
make all
```

## Looking at the Differences

Now, you can compare the `old.pretty.csv` and `new.pretty.csv` files. This is best done e.g. inside VS Code:

1. Open `old.pretty.csv`
2. Press `Ctrl + Shift + P` to open the command palette
3. Type `Compare Active File With...` and select it
4. Select `new.pretty.csv` from the list

Now you have the old file on the left and the new file on the right. Any changes are highlighted and allow for a relatively easy comparison.

[an issue with how the data is combined]: https://github.com/lycosystem/lydata-package/issues/7
[`uv`]: https://docs.astral.sh/uv/
[inline script metadata]: https://packaging.python.org/en/latest/specifications/inline-script-metadata/#inline-script-metadata
