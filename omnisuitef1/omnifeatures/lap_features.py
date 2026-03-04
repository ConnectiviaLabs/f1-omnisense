"""Lap-level feature engineering: lags, rolling, race progress, fuel, stints, degradation, encodings."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# ── Circuit name mapping (OpenF1 short name → FastF1 race name) ──────

CIRCUIT_TO_RACE = {
    'Sakhir': 'Bahrain Grand Prix',
    'Jeddah': 'Saudi Arabian Grand Prix',
    'Melbourne': 'Australian Grand Prix',
    'Baku': 'Azerbaijan Grand Prix',
    'Miami': 'Miami Grand Prix',
    'Imola': 'Emilia Romagna Grand Prix',
    'Monte Carlo': 'Monaco Grand Prix',
    'Catalunya': 'Spanish Grand Prix',
    'Madring': 'Spanish Grand Prix',
    'Montreal': 'Canadian Grand Prix',
    'Spielberg': 'Austrian Grand Prix',
    'Silverstone': 'British Grand Prix',
    'Hungaroring': 'Hungarian Grand Prix',
    'Spa-Francorchamps': 'Belgian Grand Prix',
    'Zandvoort': 'Dutch Grand Prix',
    'Monza': 'Italian Grand Prix',
    'Singapore': 'Singapore Grand Prix',
    'Suzuka': 'Japanese Grand Prix',
    'Lusail': 'Qatar Grand Prix',
    'Austin': 'United States Grand Prix',
    'Mexico City': 'Mexico City Grand Prix',
    'Interlagos': 'São Paulo Grand Prix',
    'Las Vegas': 'Las Vegas Grand Prix',
    'Yas Marina Circuit': 'Abu Dhabi Grand Prix',
    'Shanghai': 'Chinese Grand Prix',
}

# Race name aliases for historical data
_RACE_ALIASES = {
    'São Paulo Grand Prix': 'Brazilian Grand Prix',
    'Mexico City Grand Prix': 'Mexican Grand Prix',
}


# ── Outlier removal ──────────────────────────────────────────────────

def remove_lap_outliers(
    df: pd.DataFrame,
    group_cols: Optional[List[str]] = None,
    time_col: str = 'LapTime',
    lo_q: float = 0.05,
    hi_q: float = 0.95,
) -> pd.DataFrame:
    """Remove laps outside [lo_q, hi_q] quantiles per group. Returns filtered copy."""
    if group_cols is None:
        group_cols = ['Year', 'Race']
    q_lo = df.groupby(group_cols)[time_col].transform(lambda x: x.quantile(lo_q))
    q_hi = df.groupby(group_cols)[time_col].transform(lambda x: x.quantile(hi_q))
    return df[(df[time_col] >= q_lo) & (df[time_col] <= q_hi)].copy()


# ── Lag / rolling features ───────────────────────────────────────────

def add_lag_features(
    df: pd.DataFrame,
    group_cols: Optional[List[str]] = None,
    time_col: str = 'LapTime',
    lags: Optional[List[int]] = None,
    rolling_window: int = 3,
    sector_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Add lag features, rolling mean, delta from previous lap, and sector lags.

    Expects *df* sorted by group_cols + LapNumber.
    """
    if group_cols is None:
        group_cols = ['Year', 'Race', 'Driver']
    if lags is None:
        lags = [1, 2, 3]
    if sector_cols is None:
        sector_cols = ['Sector1Time', 'Sector2Time', 'Sector3Time']

    df = df.sort_values(group_cols + ['LapNumber']).copy()

    for lag in lags:
        df[f'{time_col}_lag{lag}'] = df.groupby(group_cols)[time_col].shift(lag)

    df[f'{time_col}_roll{rolling_window}'] = df.groupby(group_cols)[time_col].transform(
        lambda x: x.shift(1).rolling(rolling_window, min_periods=1).mean()
    )
    df[f'{time_col}Delta_prev'] = df[time_col] - df[f'{time_col}_lag1']

    for sector in sector_cols:
        if sector in df.columns:
            df[f'{sector}_lag1'] = df.groupby(group_cols)[sector].shift(1)

    return df


# ── Race progress & fuel ─────────────────────────────────────────────

def add_race_progress(df: pd.DataFrame) -> pd.DataFrame:
    """Add TotalLaps, RaceProgress, StintLap, FreshTyre columns."""
    race_total = df.groupby(['Year', 'Race'])['LapNumber'].max().reset_index()
    race_total.columns = ['Year', 'Race', 'TotalLaps']
    df = df.merge(race_total, on=['Year', 'Race'], how='left')
    df['RaceProgress'] = df['LapNumber'] / df['TotalLaps']
    df['StintLap'] = df['TyreLife']
    df['FreshTyre'] = df['FreshTyre'].fillna(True).astype(int)
    return df


def add_fuel_load(
    df: pd.DataFrame,
    start_kg: float = 110.0,
    burn_rate: float = 1.8,
) -> pd.DataFrame:
    """Add FuelLoad column: clip(start_kg - burn_rate * LapNumber, 0, start_kg)."""
    df['FuelLoad'] = np.clip(start_kg - burn_rate * df['LapNumber'], 0, start_kg)
    return df


def fuel_effect(lap_number: int, total_laps: int) -> float:
    """Fuel load effect on lap time. ~110 kg start, ~1.8 kg/lap burn, ~0.035 s/kg."""
    kg_burned = 1.8 * lap_number
    return kg_burned * 0.035


# ── Degradation curve lookup ─────────────────────────────────────────

def build_degradation_lookup(deg_curves: List[Dict[str, Any]]) -> Dict[Tuple, Dict]:
    """Build (circuit, compound, temp_band) → curve dict lookup."""
    lookup: Dict[Tuple, Dict] = {}
    for c in deg_curves:
        key = (c['circuit'], c['compound'], c['temp_band'])
        lookup[key] = c
    return lookup


def get_expected_degradation(
    row: pd.Series,
    deg_lookup: Dict[Tuple, Dict],
    circuit_col: str = 'Race',
    compound_col: str = 'CompoundNorm',
    tyre_life_col: str = 'TyreLife',
) -> float:
    """Look up expected degradation delta for a single lap.

    Falls back from circuit-specific to ``_global`` curve.
    """
    circuit = row[circuit_col]
    compound = row[compound_col]
    tyre_life = row[tyre_life_col]
    for key in [(circuit, compound, 'all'), ('_global', compound, 'all')]:
        if key in deg_lookup:
            c = deg_lookup[key]
            poly = list(reversed(c['coefficients'])) + [c['intercept']]
            return float(np.polyval(poly, tyre_life))
    return np.nan


def get_degradation(
    circuit: str,
    compound: str,
    tyre_life: int,
    deg_lookup: Dict[Tuple, Dict],
    temp_band: str = 'all',
) -> float:
    """Get expected lap time delta from tyre degradation (strategy simulator version).

    Falls back to linear rates if no curve found.
    """
    for key in [(circuit, compound, temp_band), (circuit, compound, 'all')]:
        if key in deg_lookup:
            c = deg_lookup[key]
            poly = list(reversed(c['coefficients'])) + [c.get('intercept', 0)]
            return max(0, float(np.polyval(poly, tyre_life)))
    rates = {'SOFT': 0.08, 'MEDIUM': 0.05, 'HARD': 0.03}
    return tyre_life * rates.get(compound, 0.05)


def add_expected_degradation(
    df: pd.DataFrame,
    deg_lookup: Dict[Tuple, Dict],
) -> pd.DataFrame:
    """Apply ``get_expected_degradation`` to all rows, adding ExpectedDegDelta column."""
    df['ExpectedDegDelta'] = df.apply(
        lambda row: get_expected_degradation(row, deg_lookup), axis=1,
    )
    return df


# ── OpenF1 stint expansion ───────────────────────────────────────────

def expand_openf1_stints(
    stints_raw: pd.DataFrame,
    sessions_df: pd.DataFrame,
    drivers_raw: pd.DataFrame,
    valid_races: Set[Tuple],
) -> pd.DataFrame:
    """Expand OpenF1 stint records to per-lap rows.

    Returns DataFrame with columns:
    ``[Year, Race, Driver, LapNumber, TyreAgeAtStart, StintNumber_of1]``.
    """
    if stints_raw.empty or sessions_df.empty or drivers_raw.empty:
        return pd.DataFrame(columns=['Year', 'Race', 'Driver', 'LapNumber',
                                     'TyreAgeAtStart', 'StintNumber_of1'])

    session_map = (
        sessions_df.drop_duplicates(subset='session_key')
        .set_index('session_key')[['year', 'circuit_short_name']]
        .to_dict('index')
    )
    driver_map = (
        drivers_raw.drop_duplicates(subset=['session_key', 'driver_number'])
        .set_index(['session_key', 'driver_number'])['name_acronym']
        .to_dict()
    )

    stint_rows: List[Dict] = []
    skipped_circuits: Set[str] = set()

    for _, s in stints_raw.iterrows():
        sk = s['session_key']
        if sk not in session_map:
            continue
        sess = session_map[sk]
        circuit = sess['circuit_short_name']
        race_name = CIRCUIT_TO_RACE.get(circuit)
        if race_name is None:
            skipped_circuits.add(circuit)
            continue

        year = sess['year']
        # Handle historical aliases
        if race_name in _RACE_ALIASES and (year, race_name) not in valid_races:
            race_name = _RACE_ALIASES[race_name]
        if race_name == 'Emilia Romagna Grand Prix' and (year, race_name) not in valid_races:
            continue
        if (year, race_name) not in valid_races:
            continue

        drv = driver_map.get((sk, s['driver_number']))
        if drv is None:
            continue

        lap_start_raw = s.get('lap_start', 0)
        lap_end_raw = s.get('lap_end', 0)
        if pd.isna(lap_start_raw) or pd.isna(lap_end_raw):
            continue
        lap_start = int(lap_start_raw or 0)
        lap_end = int(lap_end_raw or 0)
        if lap_end <= 0 or lap_start <= 0:
            continue

        for lap in range(lap_start, lap_end + 1):
            stint_rows.append({
                'Year': year,
                'Race': race_name,
                'Driver': drv,
                'LapNumber': float(lap),
                'TyreAgeAtStart': int(s.get('tyre_age_at_start', 0) or 0),
                'StintNumber_of1': int(s.get('stint_number', 1)),
            })

    if skipped_circuits:
        print(f'Unmapped circuits (skipped): {skipped_circuits}')

    return pd.DataFrame(stint_rows)


def join_stint_features(
    df: pd.DataFrame,
    stint_expanded: pd.DataFrame,
) -> pd.DataFrame:
    """Merge expanded stint data onto laps.

    Adds ``TyreAgeAtStart``, ``StintNumber_of1``, ``IsUsedTyre``.
    Falls back to defaults if stint data is empty.
    """
    if stint_expanded.empty:
        df['TyreAgeAtStart'] = 0
        df['StintNumber_of1'] = df.get('Stint', 1)
        df['IsUsedTyre'] = 0
        return df

    df = df.merge(
        stint_expanded[['Year', 'Race', 'Driver', 'LapNumber',
                        'TyreAgeAtStart', 'StintNumber_of1']],
        on=['Year', 'Race', 'Driver', 'LapNumber'],
        how='left',
    )
    df['TyreAgeAtStart'] = df['TyreAgeAtStart'].fillna(0).astype(int)
    if 'Stint' in df.columns:
        df['StintNumber_of1'] = df['StintNumber_of1'].fillna(df['Stint'])
    df['IsUsedTyre'] = (df['TyreAgeAtStart'] > 0).astype(int)
    return df


# ── Rank-based encodings ─────────────────────────────────────────────

def add_rank_encodings(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """Add CircuitCode, DriverCode, TeamCode (rank-based ordinal encoding).

    Returns ``(df_with_codes, encodings_dict)`` where *encodings_dict*
    contains ``circuit_rank``, ``driver_rank``, ``team_rank`` mappings.
    """
    # Circuit: rank by median lap time (fast→slow)
    circuit_speed = df.groupby('Race')['LapTime'].median().sort_values()
    circuit_rank = {race: i for i, race in enumerate(circuit_speed.index)}
    df['CircuitCode'] = df['Race'].map(circuit_rank)

    # Driver: rank by median delta from field median
    field_median = df.groupby(['Year', 'Race'])['LapTime'].transform('median')
    df['DriverDelta'] = df['LapTime'] - field_median
    driver_skill = df.groupby('Driver')['DriverDelta'].median().sort_values()
    driver_rank = {drv: i for i, drv in enumerate(driver_skill.index)}
    df['DriverCode'] = df['Driver'].map(driver_rank)

    # Team: rank by median delta from field median
    team_speed = df.groupby('Team')['DriverDelta'].median().sort_values()
    team_rank = {team: i for i, team in enumerate(team_speed.index)}
    df['TeamCode'] = df['Team'].map(team_rank)

    encodings = {
        'circuit_rank': circuit_rank,
        'driver_rank': driver_rank,
        'team_rank': team_rank,
    }
    return df, encodings
