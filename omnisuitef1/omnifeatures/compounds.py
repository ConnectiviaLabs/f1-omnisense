"""Compound constants, normalization, and encoding."""

from __future__ import annotations

import pandas as pd

MODERN_COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD']
WET_COMPOUNDS = ['INTERMEDIATE', 'WET']
LEGACY_MAP = {'ULTRASOFT': 'SOFT', 'SUPERSOFT': 'MEDIUM', 'HYPERSOFT': 'SOFT'}
COMPOUND_MAP = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2}


def normalize_compounds(
    df: pd.DataFrame,
    compound_col: str = 'Compound',
) -> pd.DataFrame:
    """Filter wet/unknown compounds and normalize legacy names.

    Adds ``CompoundNorm`` and ``CompoundCode`` columns.
    Returns a filtered copy — does not mutate the input.
    """
    out = df[df[compound_col].notna()].copy()
    out = out[~out[compound_col].isin(WET_COMPOUNDS)].copy()
    out['CompoundNorm'] = out[compound_col].replace(LEGACY_MAP)
    out = out[out['CompoundNorm'].isin(MODERN_COMPOUNDS)].copy()
    out['CompoundCode'] = out['CompoundNorm'].map(COMPOUND_MAP)
    return out
