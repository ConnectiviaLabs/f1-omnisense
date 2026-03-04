"""Time series forecasting: ARIMA + linear + LightGBM."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from omnidata._types import TabularDataset
from omnianalytics._types import ForecastResult

logger = logging.getLogger(__name__)


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
        return ForecastResult(
            column=series.name or "", method="linear", horizon=horizon,
            timestamps=timestamps or [str(i) for i in range(n, n + horizon)],
            values=[float(values[-1]) if n > 0 else 0.0] * horizon,
        )

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

    return ForecastResult(
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


def forecast_arima(
    series: pd.Series,
    *,
    horizon: int = 10,
    order: tuple = (1, 1, 1),
    seasonal_order: Optional[tuple] = None,
    timestamps: Optional[List[str]] = None,
) -> ForecastResult:
    """ARIMA/SARIMA forecast. Falls back to linear if statsmodels unavailable."""
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

        return ForecastResult(
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

    return ForecastResult(
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


def forecast(
    dataset: TabularDataset,
    column: str,
    *,
    horizon: int = 10,
    method: str = "auto",
) -> ForecastResult:
    """Forecast a single metric column.

    method="auto": tries ARIMA, falls back to linear.
    """
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
        return forecast_arima(series, horizon=horizon, timestamps=timestamps)
    elif method == "linear":
        return forecast_linear(series, horizon=horizon, timestamps=timestamps)
    elif method == "lightgbm":
        feature_cols = [c for c in dataset.profile.metric_cols if c != column]
        if feature_cols:
            return forecast_lightgbm(
                dataset.df, column, feature_cols, horizon=horizon, timestamps=timestamps,
            )
        return forecast_linear(series, horizon=horizon, timestamps=timestamps)
    else:
        # auto: ARIMA first, falls back internally
        return forecast_arima(series, horizon=horizon, timestamps=timestamps)


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
