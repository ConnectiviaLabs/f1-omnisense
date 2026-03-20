"""Data profiling, sampling, fingerprinting, and grounding verification.

Ported from DataSense:
  - wise_kex_framework._profile_data, _compute_trend_label, build_outlier_summary
  - kex_optimizer.intelligent_sampling, anomaly_aware_sampling
  - data_fingerprinting.compute_data_fingerprint, verify_insight_grounding
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from omnikex._types import DataFingerprint, DataProfile, GroundingResult


# ── Data profiling ───────────────────────────────────────────────────────────

def profile_data(data: pd.DataFrame) -> DataProfile:
    """Create a comprehensive profile of a DataFrame."""
    numeric_cols = data.select_dtypes(include="number").columns.tolist()
    datetime_cols = data.select_dtypes(include="datetime").columns.tolist()
    categorical_cols = data.select_dtypes(include=["object", "category"]).columns.tolist()
    entity_cols = find_entity_columns(data)

    total_cells = len(data) * len(data.columns)
    missing = int(data.isnull().sum().sum())
    completeness = round((1 - missing / total_cells) * 100, 2) if total_cells > 0 else 100.0

    # Per-column stats for numeric columns
    column_stats: Dict[str, Dict[str, Any]] = {}
    for col in numeric_cols[:15]:
        s = data[col].dropna()
        if len(s) == 0:
            continue
        column_stats[col] = {
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std()), 4) if len(s) > 1 else 0.0,
            "min": round(float(s.min()), 4),
            "max": round(float(s.max()), 4),
            "trend": compute_trend_label(s),
            "nulls_pct": round(float(data[col].isna().mean() * 100), 1),
        }

    # Detect patterns
    patterns: List[str] = []
    if datetime_cols:
        patterns.append("temporal_analysis_available")
    if len(numeric_cols) >= 2:
        patterns.append("correlation_analysis_available")
    if categorical_cols:
        patterns.append("segmentation_analysis_available")

    return DataProfile(
        row_count=len(data),
        column_count=len(data.columns),
        memory_mb=round(data.memory_usage(deep=True).sum() / 1024**2, 2),
        completeness_pct=completeness,
        numeric_cols=numeric_cols,
        datetime_cols=datetime_cols,
        categorical_cols=categorical_cols,
        entity_cols=entity_cols,
        column_stats=column_stats,
        patterns_detected=patterns,
    )


# ── Trend computation ────────────────────────────────────────────────────────

def compute_trend_label(series: pd.Series, threshold_pct: float = 10.0) -> str:
    """Compute trend direction by comparing first vs last quarter of a series."""
    n = len(series)
    if n < 8:
        return "INSUFFICIENT_DATA"
    quarter = max(n // 4, 1)
    first_q = float(series.iloc[:quarter].mean())
    last_q = float(series.iloc[-quarter:].mean())
    if abs(first_q) < 1e-6:
        return "STABLE" if abs(last_q) < 1e-6 else "RISING"
    pct_change = ((last_q - first_q) / abs(first_q)) * 100
    if pct_change > threshold_pct:
        return f"RISING (+{pct_change:.1f}%)"
    elif pct_change < -threshold_pct:
        return f"FALLING ({pct_change:.1f}%)"
    return "STABLE"


# ── Entity detection ─────────────────────────────────────────────────────────

def find_entity_columns(data: pd.DataFrame) -> List[str]:
    """Detect low-cardinality integer columns that are entity IDs, not measurements."""
    entity_cols: List[str] = []
    id_hints = {"id", "code", "key", "machine", "sensor", "device", "unit",
                "station", "node", "zone", "line", "plant", "site", "batch"}

    for col in data.select_dtypes(include=["int8", "int16", "int32", "int64"]).columns:
        n_unique = data[col].nunique()
        if n_unique < 2 or n_unique > 100:
            continue
        col_lower = col.lower().replace("_", " ").replace("-", " ")
        name_match = any(hint in col_lower for hint in id_hints)
        ratio = n_unique / max(len(data), 1)
        if name_match or ratio < 0.01:
            entity_cols.append(col)
    return entity_cols


# ── Outlier summary ──────────────────────────────────────────────────────────

def build_outlier_summary(
    data: pd.DataFrame,
    anomaly_df: Optional[pd.DataFrame] = None,
    top_n: int = 5,
) -> str:
    """Build a top/bottom-N outlier summary grouped by entity columns.

    Designed for the anomaly pillar — surfaces which entities have the most
    extreme metric values so the LLM can make specific, grounded claims.
    """
    entity_cols = find_entity_columns(data)
    if not entity_cols:
        return ""

    entity_col = entity_cols[0]
    numeric_cols = [c for c in data.select_dtypes(include="number").columns
                    if c not in entity_cols]

    if not numeric_cols or data[entity_col].nunique() < 3:
        return ""

    lines: List[str] = [f"[OUTLIER SUMMARY BY {entity_col.upper()}]"]
    grouped = data.groupby(entity_col)

    for col in numeric_cols[:8]:
        means = grouped[col].mean().dropna()
        if len(means) < 3:
            continue

        top = means.nlargest(top_n)
        bottom = means.nsmallest(top_n)
        overall_mean = data[col].mean()
        overall_std = data[col].std()

        lines.append(f"\n**{col}** (overall mean={overall_mean:.4f}, std={overall_std:.4f})")
        if overall_std > 1e-9:
            top_items = ", ".join(
                f"{entity_col}={idx}: {val:.4f} (+{((val - overall_mean) / overall_std):.1f}σ)"
                for idx, val in top.items()
            )
            bottom_items = ", ".join(
                f"{entity_col}={idx}: {val:.4f} ({((val - overall_mean) / overall_std):.1f}σ)"
                for idx, val in bottom.items()
            )
        else:
            top_items = ", ".join(f"{entity_col}={idx}: {val:.4f}" for idx, val in top.items())
            bottom_items = ", ".join(f"{entity_col}={idx}: {val:.4f}" for idx, val in bottom.items())
        lines.append(f"  Highest {top_n}: {top_items}")
        lines.append(f"  Lowest {top_n}: {bottom_items}")

    # Anomaly concentration if anomaly_df provided
    if anomaly_df is not None and entity_col in anomaly_df.columns:
        anomaly_level_col = None
        for candidate in ["Anomaly_Level", "Enhanced_Anomaly_Score", "IsolationForest_Anomaly"]:
            if candidate in anomaly_df.columns:
                anomaly_level_col = candidate
                break

        if anomaly_level_col:
            lines.append(f"\n**Anomaly concentration** (by {anomaly_level_col})")
            if anomaly_df[anomaly_level_col].dtype in ["object", "category"]:
                anomaly_counts = (
                    anomaly_df[anomaly_df[anomaly_level_col] != "Normal"]
                    .groupby(entity_col).size().nlargest(top_n)
                )
            else:
                anomaly_counts = (
                    anomaly_df.groupby(entity_col)[anomaly_level_col]
                    .mean().nlargest(top_n)
                )
            if not anomaly_counts.empty:
                items = ", ".join(
                    f"{entity_col}={idx}: {val:.2f}" for idx, val in anomaly_counts.items()
                )
                lines.append(f"  Most anomalous: {items}")

    return "\n".join(lines)


# ── Sampling ─────────────────────────────────────────────────────────────────

def intelligent_sampling(df: pd.DataFrame, max_rows: int = 8000) -> pd.DataFrame:
    """Simple random sampling if DataFrame exceeds max_rows."""
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=42)


def anomaly_aware_sampling(df: pd.DataFrame, max_rows: int = 8000) -> pd.DataFrame:
    """Sample with bias towards high-severity anomalies."""
    if len(df) <= max_rows:
        return df

    anomaly_level_col = None
    for col in ["Anomaly_Level", "anomaly_level", "severity", "Severity"]:
        if col in df.columns:
            anomaly_level_col = col
            break

    if anomaly_level_col and not df[anomaly_level_col].isna().all():
        try:
            grouped = df.groupby(anomaly_level_col, group_keys=False)
            total_size = len(df)
            samples = []
            for level, group in grouped:
                proportion = len(group) / total_size
                weight = 1.0
                if pd.api.types.is_numeric_dtype(df[anomaly_level_col]):
                    max_level = df[anomaly_level_col].max()
                    min_level = df[anomaly_level_col].min()
                    if max_level > min_level:
                        normalized = (level - min_level) / (max_level - min_level)
                        weight = 1.0 + normalized
                group_samples = max(1, min(int(proportion * weight * max_rows), len(group)))
                samples.append(group.sample(n=group_samples, random_state=42))

            result = pd.concat(samples)
            if len(result) > max_rows:
                result = result.sample(n=max_rows, random_state=42)
            elif len(result) < max_rows:
                remaining = min(max_rows - len(result), len(df) - len(result))
                if remaining > 0:
                    additional = df.drop(result.index).sample(n=remaining, random_state=42)
                    result = pd.concat([result, additional])
            return result
        except Exception:
            return df.sample(n=max_rows, random_state=42)

    return intelligent_sampling(df, max_rows)


# ── Data context for LLM ────────────────────────────────────────────────────

def build_data_context(data: pd.DataFrame, max_sample_rows: int = 30) -> str:
    """Build supplementary data context: per-group stats + sample CSV."""
    if len(data) == 0:
        return "\n(empty dataset — no data available)\n"

    parts: List[str] = []
    numeric_cols = data.select_dtypes(include="number").columns.tolist()
    cat_cols = data.select_dtypes(include=["object", "category"]).columns.tolist()

    # Per-group breakdown
    group_col = _find_group_column(data, cat_cols)
    if group_col and numeric_cols:
        block = _build_per_group_stats(data, group_col, numeric_cols[:6])
        if block:
            parts.append(block)

    # Categorical distributions
    if cat_cols:
        cat_lines: List[str] = []
        for col in cat_cols[:5]:
            vc = data[col].value_counts().head(5)
            dist = ", ".join(f"{k}={v}" for k, v in vc.items())
            cat_lines.append(f"- **{col}**: {dist}")
        if cat_lines:
            parts.append("**Categories**\n" + "\n".join(cat_lines))

    # Time range
    dt_cols = data.select_dtypes(include="datetime").columns.tolist()
    for col in dt_cols:
        s = data[col].dropna()
        if len(s) > 0:
            parts.append(f"**Time range** ({col}): {s.min()} → {s.max()}")

    # Representative sample
    sample = _smart_sample(data, max_sample_rows)
    csv_str = sample.to_csv(index=False, float_format="%.3f")
    parts.append(
        f"**Data Sample** ({len(sample)} of {len(data):,} rows)\n"
        f"```csv\n{csv_str}```"
    )

    return "\n\n".join(parts)


def build_stats_table(data: pd.DataFrame, max_cols: int = 12) -> str:
    """Build a markdown stats table: Column | Mean | Std | Min | Max | Trend."""
    entity_cols = find_entity_columns(data)
    num_cols = [c for c in data.select_dtypes(include="number").columns.tolist()
                if c not in entity_cols][:max_cols]
    if not num_cols:
        return ""

    lines = [
        f"**Column Statistics** (computed from all {len(data):,} rows)",
        "| Column | Mean | Std | Min | Max | Trend |",
        "|--------|------|-----|-----|-----|-------|",
    ]
    for col in num_cols:
        s = data[col].dropna()
        if len(s) == 0:
            continue
        trend = compute_trend_label(s)
        lines.append(
            f"| {col} | {s.mean():.4f} | {s.std():.4f} | "
            f"{s.min():.4f} | {s.max():.4f} | {trend} |"
        )
    return "\n".join(lines)


def _find_group_column(data: pd.DataFrame, cat_cols: List[str]) -> Optional[str]:
    """Find the best grouping column for per-group breakdowns."""
    for col in cat_cols:
        try:
            n_unique = data[col].nunique()
        except TypeError:
            continue
        if 2 <= n_unique <= 20:
            return col
    entity_cols = find_entity_columns(data)
    for col in entity_cols:
        try:
            n_unique = data[col].nunique()
        except TypeError:
            continue
        if 2 <= n_unique <= 50:
            return col
    return None


def _build_per_group_stats(
    data: pd.DataFrame,
    group_col: str,
    numeric_cols: List[str],
) -> str:
    """Build a per-group stats table."""
    groups = data.groupby(group_col)
    if len(groups) < 2:
        return ""

    variability = {}
    for col in numeric_cols:
        s = data[col].dropna()
        if len(s) > 0 and s.std() > 0:
            variability[col] = s.std() / (abs(s.mean()) + 1e-9)
    top_cols = sorted(variability, key=variability.get, reverse=True)[:4]
    if not top_cols:
        return ""

    header = f"| {group_col} | N |"
    sep = "|---|---|"
    for col in top_cols:
        header += f" {col} (mean±std) |"
        sep += "---|"

    rows: List[str] = []
    for gval, gdata in groups:
        if len(gdata) < 2:
            continue
        row = f"| {gval} | {len(gdata)} |"
        for col in top_cols:
            s = gdata[col].dropna()
            if len(s) > 0:
                row += f" {s.mean():.2f}±{s.std():.2f} |"
            else:
                row += " — |"
        rows.append(row)

    if not rows:
        return ""

    return (
        f"**Per-{group_col} Breakdown** (from all {len(data):,} rows)\n"
        + header + "\n" + sep + "\n" + "\n".join(rows)
    )


def _smart_sample(data: pd.DataFrame, max_rows: int = 30) -> pd.DataFrame:
    """Representative sample: head + middle + tail."""
    if len(data) <= max_rows:
        return data
    n_head = max_rows // 3
    n_tail = max_rows // 3
    n_mid = max_rows - n_head - n_tail

    head = data.head(n_head)
    tail = data.tail(n_tail)
    mid_slice = data.iloc[n_head:len(data) - n_tail] if n_tail > 0 else data.iloc[n_head:]
    n_mid = max(0, min(n_mid, len(mid_slice)))
    mid = mid_slice.sample(n_mid, random_state=42) if n_mid > 0 and len(mid_slice) > 0 else pd.DataFrame()
    return pd.concat([head, mid, tail])


# ── Data fingerprinting ──────────────────────────────────────────────────────

def compute_fingerprint(
    data: pd.DataFrame,
    mode: str = "realtime",
    extra: Optional[Dict[str, Any]] = None,
) -> DataFingerprint:
    """Compute a verifiable data fingerprint for grounding verification."""
    if data is None or not hasattr(data, "shape"):
        return DataFingerprint(mode=mode, row_count=0, column_count=0, columns=[])

    numeric_stats: Dict[str, Dict[str, Any]] = {}
    for col in data.select_dtypes(include="number").columns[:15]:
        col_data = data[col].dropna()
        if len(col_data) > 0:
            trend_raw = compute_trend_label(col_data)
            # Strip percentage for fingerprint
            trend_clean = trend_raw.split(" ")[0] if " " in trend_raw else trend_raw
            numeric_stats[col] = {
                "mean": round(float(col_data.mean()), 4),
                "min": round(float(col_data.min()), 4),
                "max": round(float(col_data.max()), 4),
                "std": round(float(col_data.std()), 4) if len(col_data) > 1 else 0.0,
                "nulls_pct": round(float(data[col].isna().mean() * 100), 1),
                "trend": trend_clean,
            }

    # Group-level stats
    group_stats: Dict[str, Dict[str, Any]] = {}
    cat_cols = data.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) > 0 and numeric_stats:
        group_col = None
        for col in cat_cols:
            try:
                if 2 <= data[col].nunique() <= 20:
                    group_col = col
                    break
            except TypeError:
                continue
        if group_col is not None:
            for gval, gdata in data.groupby(group_col):
                if len(gdata) < 4:
                    continue
                gstats: Dict[str, Dict[str, Any]] = {}
                for ncol in list(numeric_stats.keys())[:10]:
                    gcol_data = gdata[ncol].dropna()
                    if len(gcol_data) > 0:
                        gstats[ncol] = {
                            "mean": round(float(gcol_data.mean()), 4),
                            "min": round(float(gcol_data.min()), 4),
                            "max": round(float(gcol_data.max()), 4),
                        }
                group_stats[str(gval)] = gstats

    extra_data = dict(extra) if extra else {}
    if group_stats:
        extra_data["group_stats"] = group_stats

    # Time range
    dt_cols = data.select_dtypes(include="datetime").columns
    if len(dt_cols) > 0:
        col = dt_cols[0]
        col_data = data[col].dropna()
        if len(col_data) > 0:
            extra_data["time_range"] = {"start": str(col_data.min()), "end": str(col_data.max())}

    return DataFingerprint(
        mode=mode,
        row_count=int(data.shape[0]),
        column_count=int(data.shape[1]),
        columns=list(data.columns[:50]),
        numeric_stats=numeric_stats,
        extra=extra_data,
    )


def compute_data_hash(df: pd.DataFrame) -> str:
    """Generate an 8-char MD5 hex hash for caching purposes."""
    components = [
        str(df.shape),
        str(sorted(df.columns.tolist())),
        str(df.dtypes.value_counts().to_dict()),
        str(df.iloc[:min(10, len(df))].values.tolist()) if len(df) > 0 else "",
    ]
    return hashlib.md5("".join(components).encode()).hexdigest()[:8]


# ── Grounding verification ───────────────────────────────────────────────────

def verify_grounding(insight_text: str, fingerprint: DataFingerprint) -> GroundingResult:
    """Verify LLM-generated claims against source data fingerprint.

    Three layers:
      1. Numerical fidelity — are cited numbers real?
      2. Trend fidelity — do claimed trends match computed trends?
      3. Group attribution — values attributed to groups match group stats?
    """
    fp_dict = fingerprint.to_dict()
    # Merge extra into top level for compatibility
    fp_dict.update(fp_dict.pop("extra", {}))

    result = _verify_insight_grounding(insight_text, fp_dict)
    return GroundingResult(
        grounding_score=result["grounding_score"],
        total_claims=result["total_claims"],
        verified_claims=result["verified_claims"],
        unverified_claims=result["unverified_claims"],
        details=result["details"],
    )


def _verify_insight_grounding(insight_text: str, fingerprint: Dict) -> Dict[str, Any]:
    """Core grounding verification ported from DataSense data_fingerprinting.py."""
    _EMPTY: Dict[str, Any] = {
        "grounding_score": 1.0, "total_claims": 0, "verified_claims": 0,
        "unverified_claims": 0, "details": [],
    }
    if not insight_text or not fingerprint:
        return _EMPTY

    # Build lookup of verifiable values
    known_values: Dict[str, float] = {}

    for key in ("row_count", "column_count", "total_anomalies",
                "anomaly_percentage", "total_records",
                "consensus_mean", "agreement_score", "model_count"):
        if key in fingerprint:
            known_values[key] = float(fingerprint[key])

    for level, count in fingerprint.get("severity_distribution", {}).items():
        known_values[f"severity_{level.lower()}"] = float(count)

    for m_name, m_stats in fingerprint.get("model_predictions", {}).items():
        for stat_key, stat_val in m_stats.items():
            known_values[f"model_{m_name}_{stat_key}"] = float(stat_val)

    for col, stats in fingerprint.get("numeric_stats", {}).items():
        for stat_key, stat_val in stats.items():
            if stat_key == "trend":
                continue
            known_values[f"col_{col}_{stat_key}"] = float(stat_val)

    for group_name, group_cols in fingerprint.get("group_stats", {}).items():
        for col, col_stats in group_cols.items():
            for stat_key, stat_val in col_stats.items():
                known_values[f"group_{group_name}_{col}_{stat_key}"] = float(stat_val)

    if not known_values:
        return _EMPTY

    claims = _extract_claims(insight_text, fingerprint)
    details: List[Dict[str, Any]] = []
    numeric_score = 0.0

    # Layer 1: Numerical fidelity
    for claim in claims:
        match = _find_best_match(claim["value"], claim.get("context", ""), known_values)
        if match:
            numeric_score += 1.0
            details.append({
                "claim": claim["text"], "value": claim["value"],
                "status": "verified", "matched_field": match["field"],
                "expected": match["expected"], "layer": "numerical",
            })
        else:
            details.append({
                "claim": claim["text"], "value": claim["value"],
                "status": "unverified", "matched_field": None,
                "expected": None, "layer": "numerical",
            })

    # Layer 2: Trend fidelity
    trend_claims = _extract_trend_claims(insight_text, fingerprint)
    trend_score = sum(1.0 for tc in trend_claims if tc["status"] == "verified")
    details.extend(trend_claims)

    # Layer 3: Group attribution
    group_claims = _extract_group_claims(insight_text, fingerprint)
    group_score = sum(1.0 for gc in group_claims if gc["status"] == "verified")
    details.extend(group_claims)

    total_all = len(claims) + len(trend_claims) + len(group_claims)
    total_score = numeric_score + trend_score + group_score

    return {
        "grounding_score": round(total_score / total_all if total_all > 0 else 1.0, 3),
        "total_claims": total_all,
        "verified_claims": int(round(total_score)),
        "unverified_claims": total_all - int(round(total_score)),
        "details": details,
    }


def _extract_claims(text: str, fingerprint: Dict) -> List[Dict[str, Any]]:
    """Extract verifiable numerical claims from insight text."""
    claims: List[Dict[str, Any]] = []
    seen: set = set()

    # Count phrases
    count_patterns = [
        (r'(\d[\d,]*)\s+anomal(?:y|ies)', 'anomaly_count'),
        (r'(\d[\d,]*)\s+rows?', 'row_count'),
        (r'(\d[\d,]*)\s+columns?', 'column_count'),
        (r'(\d[\d,]*)\s+records?', 'record_count'),
        (r'(\d[\d,]*)\s+models?', 'model_count'),
    ]
    for pattern, ctx in count_patterns:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            val = float(m.group(1).replace(',', ''))
            key = f"{ctx}_{val}"
            if key not in seen:
                seen.add(key)
                claims.append({"text": m.group(0).strip(), "value": val, "context": ctx})

    # Percentages
    for m in re.finditer(r'(\d+\.?\d*)\s*(?:%|percent)', text, re.IGNORECASE):
        val = float(m.group(1))
        key = f"pct_{val}"
        if key not in seen:
            seen.add(key)
            claims.append({"text": m.group(0).strip(), "value": val, "context": "percentage"})

    # Labeled values
    stat_patterns = [
        (r'(?:mean|average)s?\s*(?:of|:|\s+is|around|approximately|about|~)?\s*(\-?\d+\.?\d*)', 'mean'),
        (r'(?:minimum|min)\s*(?:of|:|\s+is|around|approximately|about|~)?\s*(\-?\d+\.?\d*)', 'min'),
        (r'(?:maximum|max)\s*(?:of|:|\s+is|around|approximately|about|~)?\s*(\-?\d+\.?\d*)', 'max'),
        (r'(?:std|standard deviation)\s*(?:of|:|\s+is|around|approximately|about|~)?\s*(\-?\d+\.?\d*)', 'std'),
    ]
    for pattern, ctx in stat_patterns:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            val = float(m.group(1))
            key = f"{ctx}_{val}"
            if key not in seen:
                seen.add(key)
                claims.append({"text": m.group(0).strip(), "value": val, "context": ctx})

    # Column-specific values
    for col_name in fingerprint.get("numeric_stats", {}):
        escaped = re.escape(col_name)
        for m in re.finditer(rf'{escaped}\b[^.]*?(\-?\d+\.?\d*)', text, re.IGNORECASE):
            val = float(m.group(1))
            key = f"col_{col_name}_{val}"
            if key not in seen:
                seen.add(key)
                claims.append({"text": m.group(0).strip()[:80], "value": val, "context": f"column_{col_name}"})

    # Table cell values
    for m in re.finditer(r'\|\s*(\-?\d+\.?\d+)\s*\|', text):
        val = float(m.group(1))
        key = f"table_{val}"
        if key not in seen:
            seen.add(key)
            claims.append({"text": m.group(0).strip(), "value": val, "context": "table_value"})

    return claims


def _find_best_match(
    claim_value: float,
    context: str,
    known_values: Dict[str, float],
    rel_tol: float = 0.10,
    abs_tol: float = 0.5,
) -> Optional[Dict[str, Any]]:
    """Try to match a claimed numeric value against known fingerprint values."""
    # Narrow candidates by context
    ctx = context.lower()
    candidates = known_values

    if "column_" in ctx:
        col_name = ctx.replace("column_", "")
        candidates = {k: v for k, v in known_values.items() if col_name.lower() in k.lower()}
    elif "row" in ctx or "record" in ctx:
        candidates = {k: v for k, v in known_values.items() if "row" in k or "record" in k}
    elif "percentage" in ctx:
        candidates = {k: v for k, v in known_values.items() if "pct" in k or "percentage" in k}

    if not candidates:
        candidates = known_values

    for field, expected in candidates.items():
        if _values_match(claim_value, expected, rel_tol, abs_tol):
            return {"field": field, "expected": expected}

    return None


def _values_match(claimed: float, expected: float, rel_tol: float = 0.10, abs_tol: float = 0.5) -> bool:
    """Check if a claimed value matches an expected value within tolerance."""
    if claimed == expected:
        return True
    # Rounding-aware
    if claimed == int(claimed) and round(expected) == int(claimed):
        return True
    if claimed == round(claimed, 1) and round(expected, 1) == round(claimed, 1):
        return True
    if expected == 0:
        return abs(claimed) <= abs_tol
    return abs(claimed - expected) / abs(expected) <= rel_tol or abs(claimed - expected) <= abs_tol


# Trend synonyms for grounding
_TREND_SYNONYMS = {
    "rising": "RISING", "increasing": "RISING", "upward": "RISING",
    "falling": "FALLING", "decreasing": "FALLING", "downward": "FALLING",
    "stable": "STABLE", "flat": "STABLE",
}

_TREND_INLINE = re.compile(
    r'(\w[\w_]*)\s+(?:is|shows?|exhibits?|has)\s+(?:a\s+)?'
    r'(rising|falling|stable|increasing|decreasing|upward|downward)\s+'
    r'(?:trend|pattern|trajectory|direction)',
    re.IGNORECASE,
)


def _extract_trend_claims(text: str, fingerprint: Dict) -> List[Dict[str, Any]]:
    """Extract trend direction claims and verify against fingerprint trends."""
    col_trends: Dict[str, str] = {}
    for col, stats in fingerprint.get("numeric_stats", {}).items():
        trend = stats.get("trend")
        if trend and trend != "INSUFFICIENT_DATA":
            col_trends[col.lower()] = trend

    if not col_trends:
        return []

    claims: List[Dict[str, Any]] = []
    seen: set = set()

    for m in _TREND_INLINE.finditer(text):
        col_mention = m.group(1).lower().replace(" ", "_")
        claimed_trend = _TREND_SYNONYMS.get(m.group(2).lower(), m.group(2).upper())
        key = f"trend_{col_mention}_{claimed_trend}"
        if key in seen:
            continue
        seen.add(key)

        actual = col_trends.get(col_mention)
        if actual is None:
            for cn, t in col_trends.items():
                if col_mention in cn or cn in col_mention:
                    actual = t
                    break

        if actual is not None:
            status = "verified" if actual == claimed_trend else "unverified"
            claims.append({
                "claim": m.group(0).strip()[:80], "value": claimed_trend,
                "status": status, "expected": actual, "layer": "trend",
                "matched_field": f"trend_{col_mention}",
            })

    return claims


def _extract_group_claims(text: str, fingerprint: Dict) -> List[Dict[str, Any]]:
    """Extract claims that attribute a value to a specific group and verify."""
    group_stats = fingerprint.get("group_stats", {})
    if not group_stats:
        return []

    claims: List[Dict[str, Any]] = []
    seen: set = set()

    for group_name, group_cols in group_stats.items():
        escaped = re.escape(group_name)
        for m in re.finditer(rf'{escaped}\b[^.{{0,60}}?](\-?\d+\.?\d*)', text, re.IGNORECASE):
            val = float(m.group(1))
            key = f"group_{group_name}_{val}"
            if key in seen:
                continue
            seen.add(key)

            matched = False
            for col, col_stats in group_cols.items():
                for stat_key, stat_val in col_stats.items():
                    if _values_match(val, stat_val):
                        claims.append({
                            "claim": m.group(0).strip()[:80], "value": val,
                            "status": "verified", "layer": "group",
                            "matched_field": f"group_{group_name}_{col}_{stat_key}",
                            "expected": stat_val,
                        })
                        matched = True
                        break
                if matched:
                    break

            if not matched:
                claims.append({
                    "claim": m.group(0).strip()[:80], "value": val,
                    "status": "unverified", "layer": "group",
                    "matched_field": None, "expected": None,
                })

    return claims
