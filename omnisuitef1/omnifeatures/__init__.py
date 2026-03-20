"""OmniFeatures — F1-specific feature engineering for lap prediction models.

Quick start::

    from omnifeatures import normalize_compounds, add_lag_features, aggregate_race_weather

Modules:
    compounds     — compound normalization, encoding, constants
    lap_features  — lags, rolling, race progress, fuel, stints, degradation, encodings
    weather       — weather aggregation, air density, telemetry speed joins
    intervals     — gap features from openf1_intervals (session-level + per-driver evolution)
"""

from omnifeatures.compounds import (
    COMPOUND_MAP,
    LEGACY_MAP,
    MODERN_COMPOUNDS,
    WET_COMPOUNDS,
    normalize_compounds,
)
from omnifeatures.lap_features import (
    CIRCUIT_TO_RACE,
    add_expected_degradation,
    add_fuel_load,
    add_lag_features,
    add_race_progress,
    add_rank_encodings,
    build_degradation_lookup,
    expand_openf1_stints,
    fuel_effect,
    get_degradation,
    get_expected_degradation,
    join_stint_features,
    remove_lap_outliers,
)
from omnifeatures.weather import (
    aggregate_race_weather,
    heat_lap_delta,
    humidity_lap_delta,
    join_air_density,
    join_telemetry_speed,
)
from omnifeatures.intervals import (
    GAP_EVOLUTION_DEFAULTS,
    GAP_ROLLING_DEFAULTS,
    add_gap_evolution_features,
    add_gap_rolling_features,
    aggregate_gap_features,
    fill_gap_defaults,
    map_intervals_to_laps,
)

__all__ = [
    # compounds
    'MODERN_COMPOUNDS', 'WET_COMPOUNDS', 'LEGACY_MAP', 'COMPOUND_MAP',
    'normalize_compounds',
    # lap_features
    'CIRCUIT_TO_RACE',
    'remove_lap_outliers', 'add_lag_features', 'add_race_progress',
    'add_fuel_load', 'fuel_effect',
    'build_degradation_lookup', 'get_expected_degradation', 'get_degradation',
    'add_expected_degradation',
    'expand_openf1_stints', 'join_stint_features',
    'add_rank_encodings',
    # weather
    'aggregate_race_weather', 'join_air_density', 'join_telemetry_speed',
    'heat_lap_delta', 'humidity_lap_delta',
    # intervals
    'map_intervals_to_laps', 'aggregate_gap_features',
    'add_gap_rolling_features', 'fill_gap_defaults',
    'GAP_ROLLING_DEFAULTS', 'GAP_EVOLUTION_DEFAULTS',
    'add_gap_evolution_features',
]
