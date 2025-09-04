# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/lycosystem/lydata-package/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                       |    Stmts |     Miss |   Cover |   Missing |
|--------------------------- | -------: | -------: | ------: | --------: |
| src/lydata/\_\_init\_\_.py |       13 |        0 |    100% |           |
| src/lydata/\_version.py    |       13 |        0 |    100% |           |
| src/lydata/accessor.py     |      151 |       11 |     93% |69, 71, 200, 270, 275, 375, 493, 563-565, 577-579 |
| src/lydata/augmentor.py    |       91 |        3 |     97% |80, 130, 219 |
| src/lydata/loader.py       |      116 |       33 |     72% |58-61, 76-79, 134-142, 192-194, 234, 248-260, 290, 355-360 |
| src/lydata/querier.py      |       78 |        2 |     97% |  175, 179 |
| src/lydata/schema.py       |      108 |       18 |     83% |183, 286, 292, 300, 306, 312, 442-452, 456-459 |
| src/lydata/types.py        |        5 |        0 |    100% |           |
| src/lydata/utils.py        |      138 |       18 |     87% |31-36, 115, 198-199, 263, 293, 296, 302, 311, 319, 322, 352, 361, 364 |
| src/lydata/validator.py    |       84 |       27 |     68% |81-98, 186-193, 199-211 |
|                  **TOTAL** |  **797** |  **112** | **86%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/lycosystem/lydata-package/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/lycosystem/lydata-package/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/lycosystem/lydata-package/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/lycosystem/lydata-package/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Flycosystem%2Flydata-package%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/lycosystem/lydata-package/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.