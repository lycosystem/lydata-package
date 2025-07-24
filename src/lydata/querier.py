"""Querier module for lydata package.

This module provides the :py:class:`Q` and :py:class:`C` classes for creating and
combining reusable queries to filter :py:class:`pandas.DataFrame` objects. These
classes are inspired by Django's ``Q`` objects and allow for a more readable and modular
way to filter and query data.

For example, we may want to keep only patient with tumors of T-category 3 or higher.
Then, we can write

.. code-block:: python

    from lydata import C
    has_t_stage = C("t_stage") >= 3

Now, through the equality comparison of an instance of :py:class:`C`, the
``has_t_stage`` is an instance of :py:class:`Q` that can be combined with other queries
and applied via our custom :py:class:`~lydata.accessor.LyDataAccessor` to a table:

.. code-block:: python

    is_old = C("age") >= 65
    data.ly.query(has_t_stage & is_old)

Internally, this works by calling the :py:meth:`Q.execute` method, which returns a
boolean mask to filter the DataFrame. So, the above example is equivalent to

.. code-block:: python

    (has_t_stage & is_old).execute(data)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import pandas as pd

from lydata import accessor  # noqa: F401
from lydata.types import CanExecute
from lydata.utils import _get_all_true


class CombineQMixin:
    """Mixin class for combining queries.

    Four operators are defined for combining queries:

    1. ``&`` for logical AND operations.
        The returned object is an :py:class:`AndQ` instance and - when executed -
        returns a boolean mask where both queries are satisfied. When the right-hand
        side is ``None``, the left-hand side query object is returned unchanged.
    2. ``|`` for logical OR operations.
        The returned object is an :py:class:`OrQ` instance and - when executed -
        returns a boolean mask where either query is satisfied. When the right-hand
        side is ``None``, the left-hand side query object is returned unchanged.
    3. ``~`` for inverting a query.
        The returned object is a :py:class:`NotQ` instance and - when executed -
        returns a boolean mask where the query is not satisfied.
    4. ``==`` for checking if two queries are equal.
        Two queries are equal if their column names, operators, and values are equal.
        Note that this does not check if the queries are semantically equal, i.e., if
        they would return the same result when executed.
    """

    def __and__(self, other: CanExecute | None) -> AndQ:
        """Combine two queries with a logical AND."""
        return self if other is None else AndQ(self, other)

    def __or__(self, other: CanExecute | None) -> OrQ:
        """Combine two queries with a logical OR."""
        return self if other is None else OrQ(self, other)

    def __invert__(self) -> NotQ:
        """Negate the query."""
        return NotQ(self)

    def __eq__(self, value):
        """Check if two queries are equal."""
        return (
            isinstance(value, self.__class__)
            and self.colname == value.colname
            and self.operator == value.operator
            and self.value == value.value
        )


class Q(CombineQMixin):
    """Combinable query object for filtering a DataFrame.

    The syntax for this object is similar to Django's ``Q`` object. It can be used to
    define queries in a more readable and modular way.

    .. caution::

        The column names are not checked upon instantiation. This is only done when the
        query is executed. In fact, the :py:class:`Q` object does not even know about
        the :py:class:`~pandas.DataFrame` it will be applied to in the beginning. On the
        flip side, this means a query may be reused for different DataFrames.

    The ``operator`` argument may be one of the following:

    - ``'=='``: Checks if ``column`` values are equal to the ``value``.
    - ``'<'``: Checks if ``column`` values are less than the ``value``.
    - ``'<='``: Checks if ``column`` values are less than or equal to ``value``.
    - ``'>'``: Checks if ``column`` values are greater than the ``value``.
    - ``'>='``: Checks if ``column`` values are greater than or equal to ``value``.
    - ``'!='``: Checks if ``column`` values are not equal to the ``value``. This is
      equivalent to ``~Q(column, '==', value)``.
    - ``'in'``: Checks if ``column`` values are in the list of ``value``. For this,
      pandas' :py:meth:`~pandas.Series.isin` method is used.
    - ``'contains'``: Checks if ``column`` values contain the string ``value``.
      Here, pandas' :py:meth:`~pandas.Series.str.contains` method is used.
    - ``'pass_to'``: Passes the column values to the callable ``value``. This is useful
        for custom filtering functions that may not be covered by the other operators.
    """

    _OPERATOR_MAP: dict[str, Callable[[pd.Series, Any], pd.Series]] = {
        "==": lambda series, value: series == value,
        "<": lambda series, value: series < value,
        "<=": lambda series, value: series <= value,
        ">": lambda series, value: series > value,
        ">=": lambda series, value: series >= value,
        "!=": lambda series, value: series != value,  # same as ~Q("col", "==", value)
        "in": lambda series, value: series.isin(value),  # value is a list
        "contains": lambda series, value: series.str.contains(value),  # value is a str
        "pass_to": lambda series, value: value(series),  # value is a callable
    }

    def __init__(
        self,
        column: str,
        operator: Literal["==", "<", "<=", ">", ">=", "!=", "in", "contains"],
        value: Any,
    ) -> None:
        """Create query object that can compare a ``column`` with a ``value``."""
        self.colname = column
        self.operator = operator
        self.value = value

    def __repr__(self) -> str:
        """Return a string representation of the query."""
        return f"Q({self.colname!r}, {self.operator!r}, {self.value!r})"

    def execute(self, df: pd.DataFrame) -> pd.Series:
        """Return a boolean mask where the query is satisfied for ``df``.

        >>> df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['foo', 'bar', 'baz']})
        >>> Q('col1', '<=', 2).execute(df)
        0     True
        1     True
        2    False
        Name: col1, dtype: bool
        >>> Q('col2', 'contains', 'ba').execute(df)
        0    False
        1     True
        2     True
        Name: col2, dtype: bool
        >>> Q('col1', 'pass_to', lambda x: x % 2 == 0).execute(df)
        0    False
        1     True
        2    False
        Name: col1, dtype: bool
        """
        column = df.ly[self.colname]
        return self._OPERATOR_MAP[self.operator](column, self.value)


class NoneQ(CombineQMixin):
    """Query object that always returns the entire DataFrame. Useful as default."""

    def __repr__(self) -> str:
        """Return a string representation of the query."""
        return "NoneQ()"

    def execute(self, df: pd.DataFrame) -> pd.Series:
        """Return a boolean mask with all entries set to ``True``."""
        return _get_all_true(df)


class AndQ(CombineQMixin):
    """Query object for combining two queries with a logical AND.

    >>> df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['foo', 'bar', 'baz']})
    >>> q1 = Q('col1', '!=', 3)
    >>> q2 = Q('col2', 'contains', 'ba')
    >>> and_q = q1 & q2
    >>> print(and_q)
    (Q('col1', '!=', 3) & Q('col2', 'contains', 'ba'))
    >>> isinstance(and_q, AndQ)
    True
    >>> and_q.execute(df)
    0    False
    1     True
    2    False
    dtype: bool
    >>> all((q1 & None).execute(df) == q1.execute(df))
    True
    """

    def __init__(self, q1: CanExecute, q2: CanExecute) -> None:
        """Combine two queries with a logical AND."""
        self.q1 = q1
        self.q2 = q2

    def __repr__(self) -> str:
        """Return a string representation of the query."""
        return f"({self.q1!r} & {self.q2!r})"

    def execute(self, df: pd.DataFrame) -> pd.Series:
        """Return a boolean mask where both queries are satisfied."""
        return self.q1.execute(df) & self.q2.execute(df)


class OrQ(CombineQMixin):
    """Query object for combining two queries with a logical OR.

    >>> df = pd.DataFrame({'col1': [1, 2, 3]})
    >>> q1 = Q('col1', '==', 1)
    >>> q2 = Q('col1', '==', 3)
    >>> or_q = q1 | q2
    >>> print(or_q)
    (Q('col1', '==', 1) | Q('col1', '==', 3))
    >>> isinstance(or_q, OrQ)
    True
    >>> or_q.execute(df)
    0     True
    1    False
    2     True
    Name: col1, dtype: bool
    >>> all((q1 | None).execute(df) == q1.execute(df))
    True
    """

    def __init__(self, q1: CanExecute, q2: CanExecute) -> None:
        """Combine two queries with a logical OR."""
        self.q1 = q1
        self.q2 = q2

    def __repr__(self) -> str:
        """Return a string representation of the query."""
        return f"({self.q1!r} | {self.q2!r})"

    def execute(self, df: pd.DataFrame) -> pd.Series:
        """Return a boolean mask where either query is satisfied."""
        return self.q1.execute(df) | self.q2.execute(df)


class NotQ(CombineQMixin):
    """Query object for negating a query.

    >>> df = pd.DataFrame({'col1': [1, 2, 3]})
    >>> q = Q('col1', '==', 2)
    >>> not_q = ~q
    >>> print(not_q)
    ~Q('col1', '==', 2)
    >>> isinstance(not_q, NotQ)
    True
    >>> not_q.execute(df)
    0     True
    1    False
    2     True
    Name: col1, dtype: bool
    >>> print(~(Q('col1', '==', 2) & Q('col1', '!=', 3)))
    ~(Q('col1', '==', 2) & Q('col1', '!=', 3))
    """

    def __init__(self, q: CanExecute) -> None:
        """Negate the given query ``q``."""
        self.q = q

    def __repr__(self) -> str:
        """Return a string representation of the query."""
        return f"~{self.q!r}"

    def execute(self, df: pd.DataFrame) -> pd.Series:
        """Return a boolean mask where the query is not satisfied."""
        return ~self.q.execute(df)


class C:
    """Wraps a column name and produces a :py:class:`Q` object upon comparison.

    This is basically a shorthand for creating a :py:class:`Q` object that avoids
    writing the operator and value in quotes. Thus, it may be more readable and allows
    IDEs to provide better autocompletion.

    .. caution::

        Just like for the :py:class:`Q` object, it is not checked upon instantiation
        whether the column name is valid. This is only done when the query is executed.
    """

    def __init__(self, *column: str) -> None:
        """Create a column object for comparison.

        For querying multi-level columns, both the syntax ``C('col1', 'col2')`` and
        ``C(('col1', 'col2'))`` is valid.

        >>> (C('col1', 'col2') == 1) == (C(('col1', 'col2')) == 1)
        True
        """
        self.column = column[0] if len(column) == 1 else column

    def __repr__(self) -> str:
        """Return a string representation of the column object.

        >>> repr(C('foo'))
        "C('foo')"
        >>> repr(C('foo', 'bar'))
        "C(('foo', 'bar'))"
        """
        return f"C({self.column!r})"

    def __eq__(self, value: Any) -> Q:
        """Create a query object for comparing equality.

        >>> C('foo') == 'bar'
        Q('foo', '==', 'bar')
        """
        return Q(self.column, "==", value)

    def __lt__(self, value: Any) -> Q:
        """Create a query object for comparing less than.

        >>> C('foo') < 42
        Q('foo', '<', 42)
        """
        return Q(self.column, "<", value)

    def __le__(self, value: Any) -> Q:
        """Create a query object for comparing less than or equal.

        >>> C('foo') <= 42
        Q('foo', '<=', 42)
        """
        return Q(self.column, "<=", value)

    def __gt__(self, value: Any) -> Q:
        """Create a query object for comparing greater than.

        >>> C('foo') > 42
        Q('foo', '>', 42)
        """
        return Q(self.column, ">", value)

    def __ge__(self, value: Any) -> Q:
        """Create a query object for comparing greater than or equal.

        >>> C('foo') >= 42
        Q('foo', '>=', 42)
        """
        return Q(self.column, ">=", value)

    def __ne__(self, value: Any) -> Q:
        """Create a query object for comparing inequality.

        >>> C('foo') != 'bar'
        Q('foo', '!=', 'bar')
        """
        return Q(self.column, "!=", value)

    def isin(self, value: list[Any]) -> Q:
        """Create a query object for checking if the column values are in a list.

        >>> C('foo').isin([1, 2, 3])
        Q('foo', 'in', [1, 2, 3])
        """
        return Q(self.column, "in", value)

    def contains(self, value: str) -> Q:
        """Create a query object for checking if the column values contain a string.

        >>> C('foo').contains('bar')
        Q('foo', 'contains', 'bar')
        """
        return Q(self.column, "contains", value)

    def pass_to(self, value: Callable[[pd.Series], pd.Series]) -> Q:
        """Create a query object that passes the column values to a callable.

        This is useful for custom filtering functions that may not be covered by the
        other operators.

        >>> C('foo').pass_to(lambda x: x > 42)   # doctest: +ELLIPSIS
        Q('foo', 'pass_to', <function <lambda> at ...>)
        """
        return Q(self.column, "pass_to", value)
