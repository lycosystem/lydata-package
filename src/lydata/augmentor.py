"""Provides functions for augmenting and enhancing the lyDATA tables.

This module does the heavy lifting of inferring the most likely true involvment based
on several - possibly conflicting - diagnoses and their sensitivities and
specificities. It also resolves the sub- and super-level involvement information,
e.g. if a sublevel is involved, the superlevel is also involved, and vice-versa.

All this is achieved in the :py:func:`combine_and_augment_levels` function, which is
also used by the :py:meth:`~lydata.accessor.LyDataAccessor.combine`,
:py:meth:`~lydata.accessor.LyDataAccessor.augment`, and
:py:meth:`~lydata.accessor.LyDataAccessor.enhance` methods of the
:py:class:`~lydata.accessor.LyDataAccessor` class.
"""

from collections.abc import Mapping, Sequence
from itertools import product
from typing import Literal

import numpy as np
import pandas as pd

from lydata.utils import _sort_by


def _keep_only_involvement(table: pd.DataFrame) -> pd.DataFrame:
    """Keep only the involvement information under ``"ipsi"`` and ``"contra"``.

    >>> table = pd.DataFrame({
    ...     ("ipsi", "I"): [True, False, None],
    ...     ("contra", "II"): [False, True, None],
    ...     ("foo", "bar"): [1, 2, 3],
    ... })
    >>> _keep_only_involvement(table)
        ipsi contra
           I     II
    0   True  False
    1  False   True
    2   None   None
    """
    return table.filter(regex=r"(ipsi|contra)", axis="columns")


def _align_tables(tables: Sequence[pd.DataFrame]) -> list[pd.DataFrame]:
    """Align all columns in the sequence of ``tables``.

    >>> one = pd.DataFrame({
    ...     ("x", "a"): [1, 2],
    ...     ("x", "b"): [3, 4],
    ...     ("y", "c"): [5, 6],
    ...     ("y", "b"): [19, 120],
    ... })
    >>> two = pd.DataFrame({
    ...     ("y", "c"): [91, 10],
    ...     ("y", "b"): [9, 10],
    ...     ("x", "a"): [7, 8],
    ... })
    >>> three = pd.DataFrame({
    ...     ("x", "c"): [71, 81],
    ...     ("y", "b"): [5, 6],
    ...     ("x", "a"): [5, 61],
    ... })
    >>> aligned = _align_tables([one, two, three])
    >>> aligned[0]  # doctest: +NORMALIZE_WHITESPACE
       x           y
       a  b   c    b  c
    0  1  3 NaN   19  5
    1  2  4 NaN  120  6
    >>> aligned[1]  # doctest: +NORMALIZE_WHITESPACE
       x           y
       a   b   c   b   c
    0  7 NaN NaN   9  91
    1  8 NaN NaN  10  10
    >>> aligned[2]  # doctest: +NORMALIZE_WHITESPACE
        x          y
        a   b   c  b   c
    0   5 NaN  71  5 NaN
    1  61 NaN  81  6 NaN
    """
    if len(tables) == 0:
        return []

    all_columns = tables[0].columns
    for table in tables[1:]:
        all_columns = all_columns.union(table.columns)

    return [table.reindex(columns=all_columns) for table in tables]


def _convert_to_float_matrix(diagnoses: Sequence[pd.DataFrame]) -> np.ndarray:
    """Convert a sequence of ``diagnoses`` to a 3D float matrix.

    >>> one = pd.DataFrame({"a": [1, None], "b": [3, 4]})
    >>> two = pd.DataFrame({"a": [5, 6], "b": [7, None]})
    >>> _convert_to_float_matrix([one, two])  # doctest: +NORMALIZE_WHITESPACE
    array([[[ 1.,  3.],
            [nan,  4.]],
           [[ 5.,  7.],
            [ 6., nan]]])
    """
    matrix = np.array(diagnoses)
    matrix[pd.isna(matrix)] = np.nan
    return np.astype(matrix, float)


def _compute_likelihoods(
    diagnosis_matrix: np.ndarray,
    sensitivities: np.ndarray,
    specificities: np.ndarray,
    method: Literal["max_llh", "rank"],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the likelihoods of true/false diagnoses using the given ``method``.

    The ``diagnosis_matrix`` is a 3D array of shape ``(n_modalities, n_patients,
    n_levels)``. It should contain ``1.0`` where the diagnosis was positive and ``0.0``
    where it was negative. It may also contain ``np.nan``.

    The ``sensitivities`` and ``specificities`` are 1D arrays of shape
    ``(n_modalities,)``. When choosing the ``method="max_llh"``, the likelihood of each
    diagnosis is combined into one likelihood for each patient and level. With
    ``method="rank"``, the likelihoods are computed for the most trustworthy diagnosis.

    Returns the likelihoods of true and false diagnoses as two separate arrays.
    """
    true_pos = sensitivities[:, None, None] * diagnosis_matrix
    false_neg = (1 - sensitivities[:, None, None]) * (1 - diagnosis_matrix)
    true_neg = specificities[:, None, None] * (1 - diagnosis_matrix)
    false_pos = (1 - specificities[:, None, None]) * diagnosis_matrix

    if method not in {"max_llh", "rank"}:
        raise ValueError(f"Unknown method {method}")

    agg_func = np.nanprod if method == "max_llh" else np.nanmax
    true_llh = agg_func(true_pos + false_neg, axis=0)
    false_llh = agg_func(true_neg + false_pos, axis=0)
    return true_llh, false_llh


def _compute_involved_probs(
    diagnosis_matrix: np.ndarray,
    sensitivities: np.ndarray,
    specificities: np.ndarray,
    method: Literal["max_llh", "rank"],
) -> np.ndarray:
    """Compute the probabilities of involvement for each diagnosis."""
    true_llhs, false_llhs = _compute_likelihoods(
        diagnosis_matrix=diagnosis_matrix,
        sensitivities=sensitivities,
        specificities=specificities,
        method=method,
    )
    return true_llhs / (true_llhs + false_llhs)


def combine_and_augment_levels(
    diagnoses: Sequence[pd.DataFrame],
    specificities: Sequence[float],
    sensitivities: Sequence[float],
    method: Literal["max_llh", "rank"] = "max_llh",
    sides: Sequence[Literal["ipsi", "contra"]] | None = None,
    subdivisions: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Combine ``diagnoses`` and add sub-/superlevel involvement info.

    Different diagnostic modalities may conflict with each other, e.g. on MRI an
    LNL may look metastatic, while FNA finds no malignancy. This function combines
    available diagnoses based on their ``sensitivities`` and ``specificities``
    into a sort of consensus. When choosing the ``method="max_llh"``, the most likely/
    probable diagnosis is chosen. If ``method="rank"``, the single most trustworthy
    diagnosis is kept.

    Additionally, the function may add and resolve sub- and superlevel involvement
    information. For example, some datasets report the overall involvement in LNL II,
    while others differentiate between sublevels IIa and IIb. Now, if IIa harbors
    disease, that means that the overall involvement in II is also true. By specifying
    ``subdivisions``, the function consistently updates these super- and sublevel
    involvement patterns.

    The returned :py:class:`~pandas.DataFrame` has a two-level multi-index: One level
    for each of the ``sides`` and the second level for the involvement levels. This
    means it i in the same format as the stack of input ``diagnoses``.

    See the accessor methods ``:py:meth:`~lydata.accessor.LyDataAccessor.augment`` and
    ``:py:meth:`~lydata.accessor.LyDataAccessor.combine`` for some examples.
    """
    diagnoses = [_keep_only_involvement(table) for table in diagnoses]
    diagnoses = _align_tables(diagnoses)
    matrix = _convert_to_float_matrix(diagnoses)
    all_nan_mask = np.all(np.isnan(matrix), axis=0)

    involved_probs = _compute_involved_probs(
        diagnosis_matrix=matrix,
        sensitivities=np.array(sensitivities),
        specificities=np.array(specificities),
        method=method,
    )

    combined = np.astype(involved_probs >= 0.5, object)
    combined[all_nan_mask] = None
    combined = pd.DataFrame(combined, columns=diagnoses[0].columns)

    healthy_probs = 1.0 - involved_probs
    involved_probs[all_nan_mask] = np.nan
    involved_probs = pd.DataFrame(involved_probs, columns=diagnoses[0].columns)
    healthy_probs[all_nan_mask] = np.nan
    healthy_probs = pd.DataFrame(healthy_probs, columns=diagnoses[0].columns)

    if sides is None:
        sides = ["ipsi", "contra"]

    if subdivisions is None:
        subdivisions = {
            "I": ["a", "b"],
            "II": ["a", "b"],
            "V": ["a", "b"],
        }

    for side, (superlvl, subids) in product(sides, subdivisions.items()):
        if side not in combined.columns:
            continue

        superlvl_col = (side, superlvl)
        sublvls = [superlvl + subid for subid in subids]
        sublvl_cols = [(side, sublvl) for sublvl in sublvls]

        if set([superlvl] + sublvls).isdisjoint(set(combined[side].columns)):
            continue

        for lvl in [superlvl] + sublvls:
            combined[(side, lvl)] = combined.get((side, lvl), [None] * len(combined))
            nans = [np.nan] * len(combined)
            involved_probs[(side, lvl)] = involved_probs.get((side, lvl), nans)
            healthy_probs[(side, lvl)] = healthy_probs.get((side, lvl), nans)

        is_super_unknown = combined[superlvl_col].isna()
        is_super_healthy = combined[superlvl_col] == False
        is_super_involved = combined[superlvl_col] == True

        is_any_sub_involved = combined[sublvl_cols].any(axis=1)
        is_one_sub_unknown = combined[sublvl_cols].isna().sum(axis=1) == 1
        are_all_subs_healthy = (combined[sublvl_cols] == False).all(axis=1)
        are_all_subs_unknown = combined[sublvl_cols].isna().all(axis=1)

        # Superlvl unknown => no conflict, use sublvl info
        combined.loc[is_super_unknown & is_any_sub_involved, superlvl_col] = True
        combined.loc[is_super_unknown & are_all_subs_healthy, superlvl_col] = False

        # No sublvl involved => no conflict, use superlvl info
        combined.loc[~is_any_sub_involved & is_super_healthy, sublvl_cols] = False

        # Conflicts
        # 1) Subs override superlvl
        super_healthy_prob_from_subs = np.nanprod(healthy_probs[sublvl_cols], axis=1)
        super_involved_prob_from_subs = 1.0 - super_healthy_prob_from_subs

        do_subs_determine_super_healthy = (
            is_super_involved
            & ~are_all_subs_unknown
            & (super_healthy_prob_from_subs > involved_probs[superlvl_col])
        )
        combined.loc[do_subs_determine_super_healthy, superlvl_col] = False

        do_subs_determine_super_involved = (
            is_super_healthy
            & ~are_all_subs_unknown
            & (super_involved_prob_from_subs > healthy_probs[superlvl_col])
        )
        combined.loc[do_subs_determine_super_involved, superlvl_col] = True

        # 2) Superlvl overrides subs
        does_super_determine_all_subs_healthy = (
            is_any_sub_involved
            & is_super_healthy
            & (healthy_probs[superlvl_col] > super_involved_prob_from_subs)
        )
        combined.loc[does_super_determine_all_subs_healthy, sublvl_cols] = False

        does_super_determine_subs_unknown = (
            are_all_subs_healthy
            & is_super_involved
            & (involved_probs[superlvl_col] > super_healthy_prob_from_subs)
        )
        combined.loc[does_super_determine_subs_unknown, sublvl_cols] = None

        for sublvl in sublvls:
            sublvl_col = (side, sublvl)
            is_sub_unknown = combined[sublvl_col].isna()
            does_super_determine_unknown_sub_involved = (
                is_super_involved
                & is_sub_unknown
                & is_one_sub_unknown
                & ~is_any_sub_involved
                & (involved_probs[superlvl_col] > super_healthy_prob_from_subs)
            )
            # The above combination of conditions means that the current `sublvl` is
            # unknown, while all others are healthy, while the superlvl is involved.
            # Then below, we change the sublvl to involved.
            combined.loc[does_super_determine_unknown_sub_involved, sublvl_col] = True

    combined = _sort_by(combined, which="lnl", level=1)
    return _sort_by(combined, which="mid", level=0)
