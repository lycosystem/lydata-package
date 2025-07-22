# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/lycosystem/lydata-package/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                       |    Stmts |     Miss |   Cover |   Missing |
|--------------------------- | -------: | -------: | ------: | --------: |
| src/lydata/\_\_init\_\_.py |       12 |        0 |    100% |           |
| src/lydata/\_version.py    |       13 |        3 |     77% |      8-11 |
| src/lydata/accessor.py     |      253 |       14 |     94% |59, 188, 298, 302, 424, 426, 475-477, 516, 638-640, 762 |
| src/lydata/loader.py       |       82 |       16 |     80% |83-84, 136-138, 174, 188-195, 281-286 |
| src/lydata/utils.py        |       80 |       18 |     78% |28-33, 102, 184-185, 259-268, 301-312 |
| src/lydata/validator.py    |       92 |       42 |     54% |93-99, 107-112, 133-152, 261, 316-352, 356 |
|                  **TOTAL** |  **532** |   **93** | **83%** |           |


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