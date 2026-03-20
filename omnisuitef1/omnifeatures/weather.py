"""Weather feature engineering: aggregation, joins, condition deltas."""

from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_race_weather(weather_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-reading weather to per-race medians.

    Returns DataFrame with columns:
    ``[Year, Race, TrackTemp, AirTemp, Humidity, Rainfall]``.
    """
    race_weather = weather_df.groupby(['Year', 'Race']).agg(
        TrackTemp=('TrackTemp', 'median'),
        AirTemp=('AirTemp', 'median'),
        Humidity=('Humidity', 'median'),
        Rainfall=('Rainfall', lambda x: x.any()),
    ).reset_index()
    race_weather['Rainfall'] = race_weather['Rainfall'].astype(int)
    return race_weather


def join_air_density(
    df: pd.DataFrame,
    density_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge race_air_density onto lap data, adding ``AirDensity`` column."""
    if not density_df.empty and 'race' in density_df.columns:
        density_merge = density_df[['race', 'year', 'air_density_kg_m3']].rename(
            columns={'race': 'Race', 'year': 'Year', 'air_density_kg_m3': 'AirDensity'},
        )
        df = df.merge(density_merge, on=['Year', 'Race'], how='left')
    else:
        df['AirDensity'] = np.nan
    return df


def join_telemetry_speed(
    df: pd.DataFrame,
    telem_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge telemetry_lap_summary onto lap data, adding ``avg_speed`` and ``top_speed``."""
    if not telem_df.empty:
        telem_merge = telem_df[['Race', 'Year', 'Driver', 'LapNumber', 'avg_speed', 'top_speed']].copy()
        telem_merge['LapNumber'] = telem_merge['LapNumber'].astype(float)
        df = df.merge(telem_merge, on=['Race', 'Year', 'Driver', 'LapNumber'], how='left')
    else:
        df['avg_speed'] = np.nan
        df['top_speed'] = np.nan
    return df


# ── Condition-based deltas ────────────────────────────────────────────

def heat_lap_delta(race_laps: pd.DataFrame, weather_df: pd.DataFrame) -> float:
    """Median lap time difference on hot tracks (TrackTemp > 45) vs overall."""
    if weather_df is None or weather_df.empty:
        return np.nan
    hot = weather_df[weather_df['TrackTemp'] > 45]
    if hot.empty:
        return np.nan
    hot_races = hot['Race'].unique()
    hot_laps = race_laps[race_laps['Race'].isin(hot_races)]['LapTime']
    all_laps = race_laps['LapTime']
    if hot_laps.dropna().empty or all_laps.dropna().empty:
        return np.nan
    return round(float(hot_laps.median() - all_laps.median()), 4)


def humidity_lap_delta(race_laps: pd.DataFrame, weather_df: pd.DataFrame) -> float:
    """Median lap time difference on humid races (Humidity > 70) vs overall."""
    if weather_df is None or weather_df.empty:
        return np.nan
    humid = weather_df[weather_df['Humidity'] > 70]
    if humid.empty:
        return np.nan
    humid_races = humid['Race'].unique()
    humid_laps = race_laps[race_laps['Race'].isin(humid_races)]['LapTime']
    all_laps = race_laps['LapTime']
    if humid_laps.dropna().empty or all_laps.dropna().empty:
        return np.nan
    return round(float(humid_laps.median() - all_laps.median()), 4)
