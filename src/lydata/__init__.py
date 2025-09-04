"""Library for handling lymphatic involvement data."""

from loguru import logger

import lydata._version as _version
from lydata.accessor import LyDataFrame
from lydata.loader import (
    available_datasets,
    load_datasets,
)
from lydata.querier import C, Q
from lydata.validator import is_valid

__author__ = "Roman Ludwig"
__email__ = "roman.ludwig@usz.ch"
__uri__ = "https://github.com/lycosystem/lydata"
__version__ = _version.__version__

__all__ = [
    "LyDataFrame",
    "accessor",
    "Q",
    "C",
    "available_datasets",
    "load_datasets",
    "is_valid",
]

logger.disable("lydata")
logger.remove()
