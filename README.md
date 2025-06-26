# Python Library for Loading and Manipulating lyDATA Tables

[![Build](https://github.com/lycosystem/lydata-package/actions/workflows/release.yml/badge.svg)](https://github.com/lycosystem/lydata-package/actions/workflows/release.yml)
[![Tests](https://github.com/lycosystem/lydata-package/actions/workflows/tests.yml/badge.svg)](https://github.com/lycosystem/lydata-package/actions/workflows/tests.yml)
[![Documentation Status](https://readthedocs.org/projects/lydata/badge/?version=stable)](https://lydata.readthedocs.io/stable/?badge=stable)
[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/lycosystem/lydata-package/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/lycosystem/lydata-package/blob/python-coverage-comment-action-data/htmlcov/index.html)

This repository provides a Python library for loading, manipulating, and validating the datasets available on [lyDATA](https://github.com/lycosystem/lydata).

> [!WARNING]
> This Python library is still highly experimental!
>
> Also, it has recently been spun off from the repository of datasets, [lyDATA](https://github.com/lycosystem/lydata), and some things might still not work as expected.

## Installation

### 1. Install from PyPI

You can install the library from PyPI using pip:

```bash
pip install lydata
```

### 2. Install from Source

If you want to install the library from source, you can clone the repository and install it using pip:

```bash
git clone https://github.com/lycosystem/lydata-package
cd lydata-package
pip install -e .
```

## Usage

The first and most common use case would probably listing and loading the published datasets:

```python
>>> import lydata
>>> for dataset_spec in lydata.available_datasets(
...     year=2023,              # show all datasets added in 2023
...     ref="61a17e",           # may be some specific hash/tag/branch
... ):
...     print(dataset_spec.name)
2023-clb-multisite
2023-isb-multisite

# return generator of datasets that include oropharyngeal tumor patients
>>> first_dataset = next(lydata.load_datasets(subsite="oropharynx"))
>>> print(first_dataset.head())
... # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
  patient                              ... positive_dissected
        #                              ...             contra
       id         institution     sex  ...                III   IV    V
0    P011  Centre Léon Bérard    male  ...                0.0  0.0  0.0
1    P012  Centre Léon Bérard  female  ...                0.0  0.0  0.0
2    P014  Centre Léon Bérard    male  ...                0.0  0.0  NaN
3    P015  Centre Léon Bérard    male  ...                0.0  0.0  NaN
4    P018  Centre Léon Bérard    male  ...                NaN  NaN  NaN
[5 rows x 82 columns]

```

And since the three-level header of the tables is a little unwieldy at times, we also provide some shortcodes via a custom pandas accessor. As soon as `lydata` is imported it can be used like this:

```python
>>> print(first_dataset.ly.age)
... # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
0      67
1      62
      ...
261    60
262    60
Name: (patient, #, age), Length: 263, dtype: int64

```

And we have implemented `Q` and `C` objects inspired by Django that allow easier querying of the tables:

```python
>>> from lydata import C

# select patients younger than 50 that are not HPV positive (includes NaNs)
>>> query_result = first_dataset.ly.query((C("age") < 50) & ~(C("hpv") == True))
>>> (query_result.ly.age < 50).all()
np.True_
>>> (query_result.ly.hpv == False).all()
np.True_

```

For more details and further examples or use-cases, have a look at the [official documentation](https://lydata.readthedocs.org/)
