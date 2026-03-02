"""Shared fixtures for omnihealth tests — real data, real execution, no mocks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DATA_DIR = Path(__file__).resolve().parents[2] / "Data"
F1_CSV = DATA_DIR / "ALL_COMBINED.csv"


# ── Synthetic fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def normal_df():
    """50 rows of normal sensor data."""
    np.random.seed(42)
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=50, freq="h"),
        "vibration": np.random.normal(5.0, 0.3, 50),
        "temperature": np.random.normal(72.0, 2.0, 50),
        "pressure": np.random.normal(101.0, 3.0, 50),
        "rpm": np.random.normal(3000.0, 100.0, 50),
    })


@pytest.fixture
def anomalous_df():
    """50 rows with injected anomalies in vibration + rpm."""
    np.random.seed(42)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=50, freq="h"),
        "vibration": np.random.normal(5.0, 0.3, 50),
        "temperature": np.random.normal(72.0, 2.0, 50),
        "pressure": np.random.normal(101.0, 3.0, 50),
        "rpm": np.random.normal(3000.0, 100.0, 50),
    })
    df.loc[45:49, "vibration"] = [8, 9, 10, 11, 12]
    df.loc[47:49, "rpm"] = [1000, 800, 500]
    return df


@pytest.fixture
def trending_up_series():
    np.random.seed(42)
    return pd.Series(np.arange(100) * 0.5 + np.random.normal(0, 0.3, 100))


@pytest.fixture
def trending_down_series():
    np.random.seed(42)
    return pd.Series(-np.arange(100) * 0.5 + np.random.normal(0, 0.3, 100))


@pytest.fixture
def stable_series():
    np.random.seed(42)
    return pd.Series(np.random.normal(50, 1.0, 100))


@pytest.fixture
def seasonal_series():
    np.random.seed(42)
    t = np.arange(200)
    return pd.Series(50 + 5 * np.sin(2 * np.pi * t / 7) + np.random.normal(0, 0.5, 200))


@pytest.fixture
def degrading_series():
    np.random.seed(42)
    t = np.arange(50)
    return pd.Series(10 + t * 0.5 + t**1.5 * 0.01 + np.random.normal(0, 0.1, 50))


@pytest.fixture
def daily_df():
    np.random.seed(42)
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=100, freq="D"),
        "metric_a": np.random.normal(50.0, 5.0, 100),
        "metric_b": np.random.normal(200.0, 20.0, 100),
    })


@pytest.fixture
def minute_df():
    np.random.seed(42)
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=200, freq="min"),
        "sensor": np.random.normal(10.0, 1.0, 200),
    })


@pytest.fixture
def component_map():
    return {
        "Motor": ["vibration", "rpm"],
        "Thermal": ["temperature"],
        "Hydraulic": ["pressure"],
    }


# ── Real F1 data ────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def f1_raw():
    """Load a sample of the real F1 telemetry CSV (first 500 rows of one driver)."""
    if not F1_CSV.exists():
        pytest.skip("F1 CSV not found")
    df = pd.read_csv(F1_CSV, nrows=5000)
    # Filter to one driver for consistency
    if "Driver" in df.columns:
        drivers = df["Driver"].unique()
        df = df[df["Driver"] == drivers[0]].head(500).copy()
    # Parse date
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


@pytest.fixture(scope="session")
def f1_component_map():
    """F1 component mapping — real columns from the dataset."""
    return {
        "Power Unit": ["RPM", "Speed", "Throttle"],
        "Gearbox": ["nGear", "RPM"],
        "Brakes": ["Brake", "Speed"],
    }
