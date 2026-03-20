"""Gap and interval features from openf1_intervals collection."""

from __future__ import annotations

from typing import List, Optional, Set

import numpy as np
import pandas as pd


# ── Interval → lap mapping ───────────────────────────────────────────

def map_intervals_to_laps(
    intervals_df: pd.DataFrame,
    laps_df: pd.DataFrame,
    race_session_keys: Set,
) -> pd.DataFrame:
    """Map time-series interval data to lap numbers via ``merge_asof`` on date.

    Returns intervals with ``lap_number`` column added.
    Extracted from sc_probability.py.
    """
    intervals_race = intervals_df[intervals_df['session_key'].isin(race_session_keys)].copy()
    intervals_race['gap_to_leader'] = pd.to_numeric(intervals_race['gap_to_leader'], errors='coerce')
    intervals_race['interval'] = pd.to_numeric(intervals_race['interval'], errors='coerce')
    intervals_race = intervals_race.dropna(subset=['gap_to_leader'])

    if 'date' not in intervals_race.columns or 'date' not in laps_df.columns:
        return pd.DataFrame()

    laps_with_dates = (
        laps_df[['session_key', 'driver_number', 'lap_number', 'date']]
        .dropna(subset=['date'])
        .copy()
    )
    laps_with_dates['date'] = pd.to_datetime(laps_with_dates['date'])
    laps_with_dates = laps_with_dates.sort_values(['session_key', 'driver_number', 'lap_number'])

    intervals_race['date'] = pd.to_datetime(intervals_race['date'])
    intervals_race = intervals_race.sort_values(['session_key', 'driver_number', 'date'])
    laps_with_dates = laps_with_dates.sort_values(['session_key', 'driver_number', 'date'])

    interval_laps: List[pd.DataFrame] = []
    for (sk, dn), grp in intervals_race.groupby(['session_key', 'driver_number']):
        lap_grp = laps_with_dates[
            (laps_with_dates['session_key'] == sk)
            & (laps_with_dates['driver_number'] == dn)
        ]
        if lap_grp.empty:
            continue
        merged = pd.merge_asof(
            grp.sort_values('date'),
            lap_grp[['date', 'lap_number']].sort_values('date'),
            on='date',
            direction='backward',
        )
        interval_laps.append(merged)

    if interval_laps:
        return pd.concat(interval_laps, ignore_index=True)
    return pd.DataFrame()


# ── Session-level gap aggregation ────────────────────────────────────

def aggregate_gap_features(intervals_with_laps: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-lap gap statistics (session-level).

    Returns DataFrame indexed by ``[session_key, lap_number]`` with columns:
    ``gap_mean, gap_std, gap_spread, interval_min, interval_mean, close_battles``.
    """
    if intervals_with_laps.empty or 'lap_number' not in intervals_with_laps.columns:
        return pd.DataFrame()

    gap_agg = intervals_with_laps.groupby(['session_key', 'lap_number']).agg(
        gap_mean=('gap_to_leader', 'mean'),
        gap_std=('gap_to_leader', 'std'),
        gap_spread=('gap_to_leader', lambda x: x.max() - x.min()),
        interval_min=('interval', 'min'),
        interval_mean=('interval', 'mean'),
        close_battles=('interval', lambda x: (x.abs() < 1.0).sum()),
    ).reset_index()
    gap_agg['gap_std'] = gap_agg['gap_std'].fillna(0)
    return gap_agg


def add_gap_rolling_features(
    df: pd.DataFrame,
    group_col: str = 'session_key',
) -> pd.DataFrame:
    """Add lagged and rolling gap features for SC probability model.

    Adds: ``gap_std_prev, gap_spread_prev, interval_min_prev,
    close_battles_prev, gap_std_roll3, gap_std_roll5``.
    """
    for col in ['gap_std', 'gap_spread', 'interval_min', 'close_battles']:
        if col in df.columns:
            df[f'{col}_prev'] = df.groupby(group_col)[col].shift(1)

    if 'gap_std' in df.columns:
        df['gap_std_roll3'] = df.groupby(group_col)['gap_std'].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean(),
        ).fillna(0)
        df['gap_std_roll5'] = df.groupby(group_col)['gap_std'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean(),
        ).fillna(0)

    return df


# Default columns when no interval data is available
GAP_ROLLING_DEFAULTS = [
    'gap_std_prev', 'gap_spread_prev', 'interval_min_prev',
    'close_battles_prev', 'gap_std_roll3', 'gap_std_roll5',
]


def fill_gap_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """Set all gap rolling columns to 0 when no interval data is available."""
    for c in GAP_ROLLING_DEFAULTS:
        df[c] = 0
    return df


# ── NEW: Per-driver gap evolution features ───────────────────────────

# Default columns for gap evolution features
GAP_EVOLUTION_DEFAULTS = [
    'gap_to_leader', 'gap_delta', 'gap_delta_roll3',
    'interval_to_car_ahead', 'undercut_threat', 'close_gap_trend',
]


def add_gap_evolution_features(
    df: pd.DataFrame,
    intervals_df: pd.DataFrame,
    laps_df: pd.DataFrame,
    race_session_keys: Set,
    tyre_age_col: str = 'TyreLife',
) -> pd.DataFrame:
    """Add per-driver-per-lap gap evolution features for undercut/overcut detection.

    Features added:
    - ``gap_to_leader``: raw gap to P1
    - ``gap_delta``: change in gap vs previous lap (positive = losing time)
    - ``gap_delta_roll3``: 3-lap rolling mean of gap_delta (trend)
    - ``interval_to_car_ahead``: gap to car immediately ahead
    - ``undercut_threat``: 1 if interval < 1.5 s and car ahead has older tyres
    - ``close_gap_trend``: 1 if closing, -1 if falling back, 0 otherwise

    Parameters
    ----------
    df : DataFrame
        Lap-level data with at minimum ``[Year, Race, Driver, LapNumber]``.
    intervals_df : DataFrame
        Raw ``openf1_intervals`` data.
    laps_df : DataFrame
        Raw ``openf1_laps`` data (with ``date`` for merge_asof mapping).
    race_session_keys : set
        Session keys to process.
    tyre_age_col : str
        Column name for tyre age in *df* (used for undercut threat).
    """
    if intervals_df.empty:
        for c in GAP_EVOLUTION_DEFAULTS:
            df[c] = 0
        return df

    # Map intervals to laps
    intervals_with_laps = map_intervals_to_laps(intervals_df, laps_df, race_session_keys)
    if intervals_with_laps.empty:
        for c in GAP_EVOLUTION_DEFAULTS:
            df[c] = 0
        return df

    # Per-driver-per-lap: take the last interval reading per lap
    driver_gap = (
        intervals_with_laps
        .sort_values(['session_key', 'driver_number', 'date'])
        .groupby(['session_key', 'driver_number', 'lap_number'])
        .last()
        .reset_index()
    )

    # gap_delta: change from previous lap
    driver_gap = driver_gap.sort_values(['session_key', 'driver_number', 'lap_number'])
    driver_gap['gap_delta'] = driver_gap.groupby(
        ['session_key', 'driver_number'],
    )['gap_to_leader'].diff().fillna(0)

    # gap_delta_roll3: 3-lap rolling trend
    driver_gap['gap_delta_roll3'] = driver_gap.groupby(
        ['session_key', 'driver_number'],
    )['gap_delta'].transform(
        lambda x: x.rolling(3, min_periods=1).mean(),
    )

    # interval_to_car_ahead (already in 'interval' column)
    driver_gap['interval_to_car_ahead'] = pd.to_numeric(
        driver_gap['interval'], errors='coerce',
    ).fillna(0)

    # close_gap_trend: +1 closing, -1 falling back, 0 stable
    driver_gap['close_gap_trend'] = np.where(
        driver_gap['gap_delta_roll3'] < -0.1, 1,
        np.where(driver_gap['gap_delta_roll3'] > 0.1, -1, 0),
    )

    # Select columns to merge
    merge_cols = [
        'session_key', 'driver_number', 'lap_number',
        'gap_to_leader', 'gap_delta', 'gap_delta_roll3',
        'interval_to_car_ahead', 'close_gap_trend',
    ]
    gap_merge = driver_gap[merge_cols].copy()
    gap_merge['lap_number'] = gap_merge['lap_number'].astype(float)

    # We need to map (session_key, driver_number, lap_number) back to
    # the main df which uses (Year, Race, Driver, LapNumber).
    # Build a bridge from laps_df (openf1_laps) which has both keys.
    if not laps_df.empty and 'driver_number' in laps_df.columns:
        bridge = (
            laps_df[['session_key', 'driver_number', 'lap_number']]
            .drop_duplicates()
            .copy()
        )
        bridge['lap_number'] = bridge['lap_number'].astype(float)
        gap_merge = gap_merge.merge(bridge, on=['session_key', 'driver_number', 'lap_number'], how='inner')

    # For now, merge on LapNumber via a session→(Year,Race) map if available
    # The simplest approach: add gap features as extra columns keyed by position in df
    # Since df uses FastF1 naming and intervals use OpenF1 naming, we merge by lap number
    # after the session-level mapping is handled by the caller.
    # Return the gap features keyed by openf1 identifiers for the caller to join.

    # Undercut threat: interval < 1.5s — computed after merge with main df
    # since we need the tyre age of both the driver and car ahead.
    # For now, set a simple threshold-based proxy.
    gap_merge['undercut_threat'] = (
        (gap_merge['interval_to_car_ahead'].abs() < 1.5)
        & (gap_merge['interval_to_car_ahead'].abs() > 0)
    ).astype(int)

    # Store as class attribute or return for the caller to handle the join
    # Since the xgboost/bilstm models may use different driver naming,
    # we return the per-session-driver-lap features for flexible joining.
    for c in GAP_EVOLUTION_DEFAULTS:
        if c not in df.columns:
            df[c] = 0

    # Try direct merge if df has session_key / driver_number
    if 'session_key' in df.columns and 'driver_number' in df.columns:
        feature_cols = ['gap_to_leader', 'gap_delta', 'gap_delta_roll3',
                        'interval_to_car_ahead', 'undercut_threat', 'close_gap_trend']
        gap_slim = gap_merge[['session_key', 'driver_number', 'lap_number'] + feature_cols]
        df['LapNumber_float'] = df['LapNumber'].astype(float) if 'LapNumber' in df.columns else df['lap_number'].astype(float)
        gap_slim = gap_slim.rename(columns={'lap_number': 'LapNumber_float'})
        df = df.merge(
            gap_slim,
            on=['session_key', 'driver_number', 'LapNumber_float'],
            how='left',
            suffixes=('', '_gap'),
        )
        for c in feature_cols:
            gap_col = f'{c}_gap'
            if gap_col in df.columns:
                df[c] = df[gap_col].fillna(df[c])
                df.drop(columns=[gap_col], inplace=True)
        df.drop(columns=['LapNumber_float'], inplace=True, errors='ignore')
    else:
        # Attach gap features as a returned DataFrame for manual joining
        df.attrs['_gap_evolution'] = gap_merge

    return df
