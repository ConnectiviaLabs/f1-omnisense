"""Time series forecasting: ETS + ARIMA + linear + LightGBM."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from omnidata._types import TabularDataset
from omnianalytics import feature_store
from omnianalytics._types import ForecastResult

logger = logging.getLogger(__name__)

HISTORY_PCT = 0.15  # show 15% of the driver's race history for context, focus on forecast


# ── Heuristics ────────────────────────────────────────────────────────────

def _enrich_heuristics(result: ForecastResult, raw_series: np.ndarray) -> ForecastResult:
    """Compute trend_direction, trend_pct, volatility, risk_flag and attach history."""
    vals = result.values
    if len(vals) >= 2:
        first, last = vals[0], vals[-1]
        pct = ((last - first) / abs(first) * 100) if first != 0 else 0.0
        result.trend_pct = round(pct, 2)
        if pct > 2:
            result.trend_direction = "rising"
        elif pct < -2:
            result.trend_direction = "falling"
        else:
            result.trend_direction = "stable"
    else:
        result.trend_direction = "stable"
        result.trend_pct = 0.0

    # Volatility: normalised average confidence band width
    if result.lower_bound and result.upper_bound and len(vals) > 0:
        widths = [u - l for u, l in zip(result.upper_bound, result.lower_bound)]
        mean_val = np.mean(np.abs(vals)) or 1.0
        result.volatility = round(np.mean(widths) / mean_val, 4)
    else:
        result.volatility = 0.0

    # Risk flag: forecast declining AND lower bound diverging
    if result.trend_direction == "falling" and result.volatility and result.volatility > 0.15:
        result.risk_flag = True

    # Historical tail — show 15% of the driver's races
    n = len(raw_series)
    tail_len = max(5, int(n * HISTORY_PCT))  # at least 5 points
    tail = raw_series[-tail_len:]
    result.history = [round(float(v), 4) for v in tail]
    result.history_timestamps = [str(n - tail_len + i) for i in range(tail_len)]

    return result


# ── Forecasting methods ──────────────────────────────────────────────────

def forecast_linear(
    series: pd.Series,
    *,
    horizon: int = 10,
    timestamps: Optional[List[str]] = None,
) -> ForecastResult:
    """Simple linear regression forecast (always available)."""
    values = series.dropna().values.astype(float)
    n = len(values)
    if n < 2:
        r = ForecastResult(
            column=series.name or "", method="linear", horizon=horizon,
            timestamps=timestamps or [str(i) for i in range(n, n + horizon)],
            values=[float(values[-1]) if n > 0 else 0.0] * horizon,
        )
        return _enrich_heuristics(r, values)

    x = np.arange(n)
    coeffs = np.polyfit(x, values, 1)
    slope, intercept = coeffs

    future_x = np.arange(n, n + horizon)
    predictions = slope * future_x + intercept

    # Confidence from residuals
    fitted = slope * x + intercept
    residual_std = np.std(values - fitted)
    steps = np.arange(1, horizon + 1)
    margin = 1.96 * residual_std * np.sqrt(1 + steps / n)

    # In-sample MAE/RMSE
    mae = float(np.mean(np.abs(values - fitted)))
    rmse = float(np.sqrt(np.mean((values - fitted) ** 2)))

    ts = timestamps or [str(i) for i in range(n, n + horizon)]

    r = ForecastResult(
        column=series.name or "",
        method="linear",
        horizon=horizon,
        timestamps=ts,
        values=[round(float(v), 4) for v in predictions],
        lower_bound=[round(float(v - m), 4) for v, m in zip(predictions, margin)],
        upper_bound=[round(float(v + m), 4) for v, m in zip(predictions, margin)],
        mae=round(mae, 4),
        rmse=round(rmse, 4),
    )
    return _enrich_heuristics(r, values)


def forecast_ets(
    series: pd.Series,
    *,
    horizon: int = 10,
    timestamps: Optional[List[str]] = None,
) -> ForecastResult:
    """Holt-Winters Exponential Smoothing — fast (~50ms) with trend + seasonality."""
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except ImportError:
        logger.info("statsmodels not installed, falling back to linear")
        return forecast_linear(series, horizon=horizon, timestamps=timestamps)

    values = series.dropna().values.astype(float)
    if len(values) < 5:
        return forecast_linear(series, horizon=horizon, timestamps=timestamps)

    try:
        # Try additive trend; skip seasonal if too few points
        model = ExponentialSmoothing(
            values,
            trend="add",
            seasonal=None,  # no seasonal — race-by-race data isn't periodic
            initialization_method="estimated",
        )
        fitted = model.fit(optimized=True)

        predictions = fitted.forecast(horizon)

        # Confidence bands from in-sample residuals
        in_sample = fitted.fittedvalues
        residuals = values[:len(in_sample)] - in_sample
        residual_std = np.std(residuals)
        steps = np.arange(1, horizon + 1)
        margin = 1.96 * residual_std * np.sqrt(steps)

        mae = float(np.mean(np.abs(residuals)))
        rmse = float(np.sqrt(np.mean(residuals ** 2)))

        ts = timestamps or [str(i) for i in range(len(values), len(values) + horizon)]

        r = ForecastResult(
            column=series.name or "",
            method="ets",
            horizon=horizon,
            timestamps=ts,
            values=[round(float(v), 4) for v in predictions],
            lower_bound=[round(float(v - m), 4) for v, m in zip(predictions, margin)],
            upper_bound=[round(float(v + m), 4) for v, m in zip(predictions, margin)],
            mae=round(mae, 4),
            rmse=round(rmse, 4),
        )
        return _enrich_heuristics(r, values)
    except Exception as e:
        logger.warning("ETS failed (%s), falling back to linear", e)
        return forecast_linear(series, horizon=horizon, timestamps=timestamps)


def forecast_arima(
    series: pd.Series,
    *,
    horizon: int = 10,
    order: tuple = (1, 1, 1),
    seasonal_order: Optional[tuple] = None,
    timestamps: Optional[List[str]] = None,
) -> ForecastResult:
    """ARIMA/SARIMA forecast. Falls back to ETS if slow, linear if unavailable."""
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        logger.info("statsmodels not installed, falling back to linear")
        return forecast_linear(series, horizon=horizon, timestamps=timestamps)

    values = series.dropna().values.astype(float)
    if len(values) < 10:
        return forecast_linear(series, horizon=horizon, timestamps=timestamps)

    try:
        kwargs = {"order": order, "enforce_stationarity": False, "enforce_invertibility": False}
        if seasonal_order:
            kwargs["seasonal_order"] = seasonal_order

        model = SARIMAX(values, **kwargs)
        fitted = model.fit(disp=False, maxiter=100)

        fc = fitted.get_forecast(steps=horizon)
        predictions = fc.predicted_mean
        ci = fc.conf_int(alpha=0.05)

        # In-sample metrics
        in_sample = fitted.fittedvalues
        residuals = values[len(values) - len(in_sample):] - in_sample
        mae = float(np.mean(np.abs(residuals)))
        rmse = float(np.sqrt(np.mean(residuals ** 2)))

        ts = timestamps or [str(i) for i in range(len(values), len(values) + horizon)]

        r = ForecastResult(
            column=series.name or "",
            method="arima",
            horizon=horizon,
            timestamps=ts,
            values=[round(float(v), 4) for v in predictions],
            lower_bound=[round(float(v), 4) for v in ci[:, 0]],
            upper_bound=[round(float(v), 4) for v in ci[:, 1]],
            mae=round(mae, 4),
            rmse=round(rmse, 4),
        )
        return _enrich_heuristics(r, values)
    except Exception as e:
        logger.warning("ARIMA failed (%s), falling back to linear", e)
        return forecast_linear(series, horizon=horizon, timestamps=timestamps)


def forecast_lightgbm(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    *,
    horizon: int = 10,
    timestamps: Optional[List[str]] = None,
) -> ForecastResult:
    """LightGBM regression forecast using engineered features."""
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        logger.info("lightgbm not installed, falling back to linear")
        return forecast_linear(df[target_col], horizon=horizon, timestamps=timestamps)

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).values

    if len(y) < 20:
        return forecast_linear(df[target_col], horizon=horizon, timestamps=timestamps)

    model = LGBMRegressor(
        n_estimators=100, num_leaves=15, max_depth=5,
        learning_rate=0.1, reg_alpha=0.5, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    model.fit(X, y)

    # Predict future by extending feature patterns
    last_row = X.iloc[-1:].values
    predictions = []
    for _ in range(horizon):
        pred = model.predict(last_row)[0]
        predictions.append(pred)
        # Shift features forward (simple carry-forward)
        last_row = last_row.copy()

    # Confidence from training residuals
    train_preds = model.predict(X)
    residual_std = np.std(y - train_preds)
    steps = np.arange(1, horizon + 1)
    margin = 1.96 * residual_std * np.sqrt(steps / len(y))

    mae = float(np.mean(np.abs(y - train_preds)))
    rmse = float(np.sqrt(np.mean((y - train_preds) ** 2)))

    ts = timestamps or [str(i) for i in range(len(y), len(y) + horizon)]

    r = ForecastResult(
        column=target_col,
        method="lightgbm",
        horizon=horizon,
        timestamps=ts,
        values=[round(float(v), 4) for v in predictions],
        lower_bound=[round(float(v - m), 4) for v, m in zip(predictions, margin)],
        upper_bound=[round(float(v + m), 4) for v, m in zip(predictions, margin)],
        mae=round(mae, 4),
        rmse=round(rmse, 4),
    )
    return _enrich_heuristics(r, y)


def forecast_sf(
    series: pd.Series,
    *,
    horizon: int = 10,
    timestamps: Optional[List[str]] = None,
) -> ForecastResult:
    """StatsForecast AutoETS + AutoARIMA + AutoTheta ensemble — fast and accurate."""
    try:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoETS, AutoARIMA, AutoTheta
    except ImportError:
        logger.info("statsforecast not installed, falling back to ets")
        return forecast_ets(series, horizon=horizon, timestamps=timestamps)

    values = series.dropna().values.astype(float)
    if len(values) < 5:
        return forecast_linear(series, horizon=horizon, timestamps=timestamps)

    try:
        # StatsForecast expects a DataFrame with unique_id, ds, y
        sf_df = pd.DataFrame({
            "unique_id": "driver",
            "ds": pd.RangeIndex(len(values)),
            "y": values,
        })

        models = [AutoETS(season_length=1), AutoARIMA(season_length=1)]
        if len(values) >= 10:
            models.append(AutoTheta(season_length=1))

        sf = StatsForecast(models=models, freq=1, n_jobs=1)
        sf.fit(sf_df)
        forecast_df = sf.predict(h=horizon, level=[95])

        # Average across models for ensemble prediction
        model_cols = [c for c in forecast_df.columns if not c.startswith("ds") and "-lo-" not in c and "-hi-" not in c and c != "unique_id"]
        lo_cols = [c for c in forecast_df.columns if "-lo-95" in c]
        hi_cols = [c for c in forecast_df.columns if "-hi-95" in c]

        predictions = forecast_df[model_cols].mean(axis=1).values
        lower = forecast_df[lo_cols].mean(axis=1).values if lo_cols else predictions
        upper = forecast_df[hi_cols].mean(axis=1).values if hi_cols else predictions

        # In-sample fitted values for MAE/RMSE
        fitted_df = sf.fitted_[0] if hasattr(sf, 'fitted_') else None
        if fitted_df is not None and len(fitted_df) > 0:
            fitted_col = model_cols[0] if model_cols else None
            if fitted_col and fitted_col in fitted_df.columns:
                in_sample = fitted_df[fitted_col].values
                residuals = values[:len(in_sample)] - in_sample
                mae = float(np.mean(np.abs(residuals)))
                rmse = float(np.sqrt(np.mean(residuals ** 2)))
            else:
                mae = rmse = None
        else:
            # Fallback: compute from cross-val-like residual estimate
            mae = rmse = None

        ts = timestamps or [str(i) for i in range(len(values), len(values) + horizon)]

        r = ForecastResult(
            column=series.name or "",
            method="statsforecast",
            horizon=horizon,
            timestamps=ts,
            values=[round(float(v), 4) for v in predictions],
            lower_bound=[round(float(v), 4) for v in lower],
            upper_bound=[round(float(v), 4) for v in upper],
            mae=round(mae, 4) if mae is not None else None,
            rmse=round(rmse, 4) if rmse is not None else None,
        )
        return _enrich_heuristics(r, values)
    except Exception as e:
        logger.warning("StatsForecast failed (%s), falling back to ETS", e)
        return forecast_ets(series, horizon=horizon, timestamps=timestamps)


# ── Main dispatch ─────────────────────────────────────────────────────────

def forecast(
    dataset: TabularDataset,
    column: str,
    *,
    horizon: int = 10,
    method: str = "auto",
    session_key: Optional[int] = None,
    driver_number: Optional[int] = None,
    db=None,
) -> ForecastResult:
    """Forecast a single metric column.

    method="auto": tries ETS (fast), falls back to linear.
    method="arima": slower but more accurate for long series.
    method="ets": Holt-Winters exponential smoothing (~50ms).
    """
    # ── Feature store cache check ──
    cache_computation = f"forecast:{column}:{horizon}:{method}"
    if session_key is not None and driver_number is not None and db is not None:
        cached = feature_store.get(db, session_key, driver_number, cache_computation)
        if cached is not None:
            logger.info("feature_store HIT: %s session=%s driver=%s", cache_computation, session_key, driver_number)
            return ForecastResult.from_dict(cached)

    if column not in dataset.df.columns:
        raise ValueError(f"Column '{column}' not in dataset")

    series = pd.to_numeric(dataset.df[column], errors="coerce").dropna()
    series.name = column

    # Build timestamps from dataset's timestamp column
    timestamps = None
    ts_col = dataset.profile.timestamp_col
    if ts_col and ts_col in dataset.df.columns:
        ts_series = pd.to_datetime(dataset.df[ts_col], errors="coerce").dropna()
        if len(ts_series) > 1:
            freq = ts_series.diff().median()
            last_ts = ts_series.iloc[-1]
            future_ts = [last_ts + freq * (i + 1) for i in range(horizon)]
            timestamps = [str(t) for t in future_ts]

    if method == "arima":
        result = forecast_arima(series, horizon=horizon, timestamps=timestamps)
    elif method == "linear":
        result = forecast_linear(series, horizon=horizon, timestamps=timestamps)
    elif method == "ets":
        result = forecast_ets(series, horizon=horizon, timestamps=timestamps)
    elif method == "statsforecast" or method == "sf":
        result = forecast_sf(series, horizon=horizon, timestamps=timestamps)
    elif method == "lightgbm":
        feature_cols = [c for c in dataset.profile.metric_cols if c != column]
        if feature_cols:
            result = forecast_lightgbm(
                dataset.df, column, feature_cols, horizon=horizon, timestamps=timestamps,
            )
        else:
            result = forecast_linear(series, horizon=horizon, timestamps=timestamps)
    else:
        # auto: StatsForecast ensemble (AutoETS + AutoARIMA + AutoTheta), falls back to ETS/linear
        result = forecast_sf(series, horizon=horizon, timestamps=timestamps)

    # ── Cache result in feature store ──
    if session_key is not None and driver_number is not None and db is not None:
        try:
            feature_store.put(db, session_key, driver_number, cache_computation, result.to_dict())
        except Exception:
            logger.debug("Failed to cache %s for session=%s driver=%s", cache_computation, session_key, driver_number)

    return result


def forecast_anomaly_features(
    dataset: TabularDataset,
    anomaly_result: "AnomalyResult",
    *,
    horizon: int = 10,
    method: str = "auto",
) -> Dict[str, List[ForecastResult]]:
    """Forecast the top SHAP features from HIGH/CRITICAL anomalies.

    For each HIGH/CRITICAL point that has shap_top_features,
    forecasts each identified feature. Returns grouped by severity.
    """
    from omnianalytics._types import SeverityLevel

    results: Dict[str, List[ForecastResult]] = {"high": [], "critical": []}
    seen_features = set()

    for score in anomaly_result.scores:
        if score.severity not in (SeverityLevel.HIGH, SeverityLevel.CRITICAL):
            continue

        top_features = score.model_scores.get("shap_top_features", [])
        if not top_features:
            continue

        severity_key = score.severity.value
        for feat_info in top_features:
            feat_name = feat_info["feature"]
            if feat_name in seen_features:
                continue
            seen_features.add(feat_name)

            if feat_name not in dataset.df.columns:
                continue

            try:
                fc = forecast(dataset, feat_name, horizon=horizon, method=method)
                results[severity_key].append(fc)
            except Exception as e:
                logger.warning("Failed to forecast %s: %s", feat_name, e)

    return results
