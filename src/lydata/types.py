"""Protocol and type definitions for lydata package."""

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class CanExecute(Protocol):
    """Protocol for objects that can :py:func:`execute` on a DataFrame."""

    def execute(self, df: pd.DataFrame) -> pd.Series:
        """Provide a binary mask for the ``df`` DataFrame."""
        ...
