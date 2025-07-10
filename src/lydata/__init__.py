"""Library for handling lymphatic involvement data."""

from loguru import logger

import lydata._version as _version
from lydata.accessor import C, Q
from lydata.loader import (
    available_datasets,
    load_datasets,
)
from lydata.utils import infer_and_combine_levels

__author__ = "Roman Ludwig"
__email__ = "roman.ludwig@usz.ch"
__uri__ = "https://github.com/lycosystem/lydata"
__version__ = _version.__version__

__all__ = [
    "accessor",
    "Q",
    "C",
    "available_datasets",
    "load_datasets",
    "infer_and_combine_levels",
]

logger.disable("lydata")
