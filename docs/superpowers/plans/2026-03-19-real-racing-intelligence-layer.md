# Real Racing Intelligence Layer Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four intelligence features to Real Racing that go beyond AiM RaceStudio 3 — track anomalies, predictive degradation, lap deltas, and system risk heatmap — using statistical thresholds on raw telemetry (11,560 samples) instead of the ML ensemble (which fails with only 7 laps).

**Architecture:** New backend function `run_track_anomalies()` in `pipeline/aim_anomaly.py` runs statistical threshold rules on raw `aim_telemetry` data, computes per-channel degradation via linear regression on lap averages, and generates lap-over-lap delta analysis. New endpoint `GET /api/aim/track-anomalies/{session_id}` serves the results. Frontend adds an "Anomalies" tab to `TrackMapOverlay` and intelligence cards below the map.

**Tech Stack:** Python (numpy, scipy.stats for linregress), FastAPI, React, SVG, TypeScript

**Spec:** `docs/superpowers/specs/2026-03-19-real-racing-intelligence-layer-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `pipeline/aim_anomaly.py` | Modify | Add `run_track_anomalies()` — threshold detection, degradation, lap deltas |
| `pipeline/aim_router.py` | Modify | Add `GET /api/aim/track-anomalies/{session_id}` endpoint |
| `frontend/src/app/api/local.ts` | Modify | Add `aim.trackAnomalies(id)` API call |
| `frontend/src/app/components/RealRacing.tsx` | Modify | Add Anomalies tab to TrackMapOverlay, add intelligence cards |

---

## Task 1: Backend — Statistical anomaly detection on raw telemetry

**Files:**
- Modify: `pipeline/aim_anomaly.py`

- [ ] **Step 1: Add threshold detection function**

Add to `pipeline/aim_anomaly.py` after the existing `_action_from_severity` function:

```python
# ── Track-level anomaly detection (statistical thresholds) ────────────


def _detect_threshold_anomalies(session_id: str) -> tuple[list[dict], list[dict]]:
    """Run statistical threshold rules on raw telemetry.

    Returns (anomaly_zones, point_events).
    Zones are contiguous segments; points are single-sample events.
    """
    db = _get_db()

    docs = list(db["aim_telemetry"].find(
        {"session_id": session_id},
        {"_id": 0, "time_s": 1, "lap": 1,
         "GPS_Lat": 1, "GPS_Lon": 1, "GPS_Speed_kmh": 1,
         "FrontBrake_bar": 1, "RearBrake_bar": 1,
         "OilTemp_C": 1, "WaterTemp_C": 1, "OilPressure_bar": 1,
         "FuelPressure_bar": 1, "Throttle_pct": 1,
         "LateralAcc_g": 1, "ExternalVoltage_V": 1},
    ).sort("time_s", 1))

    if not docs:
        return [], []

    df = pd.DataFrame(docs)

    # Session-wide stats
    stats = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        series = df[col].dropna()
        if len(series) < 10:
            continue
        stats[col] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "p5": float(series.quantile(0.05)),
            "p95": float(series.quantile(0.95)),
            "p99": float(series.quantile(0.99)),
        }

    zones: list[dict] = []
    points: list[dict] = []

    # Identify outlap/inlap by lap time — exclude from analysis
    lap_times = df.groupby("lap")["time_s"].agg(["min", "max"])
    lap_times["duration"] = lap_times["max"] - lap_times["min"]
    median_duration = lap_times["duration"].median()
    hot_laps = set(
        lap_times[lap_times["duration"] < median_duration * 1.5].index
    )

    for lap_num in hot_laps:
        lap_df = df[df["lap"] == lap_num].copy()
        if lap_df.empty:
            continue

        # ── Rule 1: Brake pressure drop during braking zone ──
        if "FrontBrake_bar" in stats:
            braking = lap_df[lap_df["FrontBrake_bar"] > 2.0]  # in braking zone
            if len(braking) > 5:
                brake_mean = float(braking["FrontBrake_bar"].mean())
                brake_std = float(braking["FrontBrake_bar"].std()) or 1.0
                drops = braking[braking["FrontBrake_bar"] < brake_mean - 2 * brake_std]
                if len(drops) >= 3:
                    # Group consecutive drops into zones
                    start = drops.iloc[0]
                    end = drops.iloc[-1]
                    zones.append({
                        "type": "segment",
                        "system": "Brakes",
                        "severity": "HIGH" if len(drops) > 5 else "MEDIUM",
                        "channel": "FrontBrake_bar",
                        "reason": f"Brake pressure dropped {((brake_mean - float(drops['FrontBrake_bar'].mean())) / brake_mean * 100):.0f}% below braking zone mean",
                        "lap": int(lap_num),
                        "start_time_s": float(start["time_s"]),
                        "end_time_s": float(end["time_s"]),
                        "start_gps": {"lat": float(start.get("GPS_Lat", 0)), "lon": float(start.get("GPS_Lon", 0))},
                        "end_gps": {"lat": float(end.get("GPS_Lat", 0)), "lon": float(end.get("GPS_Lon", 0))},
                    })

        # ── Rule 2: Oil temp spike (>95th percentile) ──
        if "OilTemp_C" in stats:
            threshold = stats["OilTemp_C"]["p95"]
            spikes = lap_df[lap_df["OilTemp_C"] > threshold]
            if len(spikes) > 0:
                worst = spikes.loc[spikes["OilTemp_C"].idxmax()]
                points.append({
                    "type": "point",
                    "system": "Thermal",
                    "severity": "HIGH" if float(worst["OilTemp_C"]) > stats["OilTemp_C"]["p99"] else "MEDIUM",
                    "channel": "OilTemp_C",
                    "reason": f"Oil temp {float(worst['OilTemp_C']):.1f}°C (95th pct = {threshold:.1f}°C)",
                    "lap": int(lap_num),
                    "time_s": float(worst["time_s"]),
                    "gps": {"lat": float(worst.get("GPS_Lat", 0)), "lon": float(worst.get("GPS_Lon", 0))},
                    "value": round(float(worst["OilTemp_C"]), 1),
                })

        # ── Rule 3: Water temp spike (>95th percentile) ──
        if "WaterTemp_C" in stats:
            threshold = stats["WaterTemp_C"]["p95"]
            spikes = lap_df[lap_df["WaterTemp_C"] > threshold]
            if len(spikes) > 0:
                worst = spikes.loc[spikes["WaterTemp_C"].idxmax()]
                points.append({
                    "type": "point",
                    "system": "Thermal",
                    "severity": "HIGH" if float(worst["WaterTemp_C"]) > stats["WaterTemp_C"]["p99"] else "MEDIUM",
                    "channel": "WaterTemp_C",
                    "reason": f"Water temp {float(worst['WaterTemp_C']):.1f}°C (95th pct = {threshold:.1f}°C)",
                    "lap": int(lap_num),
                    "time_s": float(worst["time_s"]),
                    "gps": {"lat": float(worst.get("GPS_Lat", 0)), "lon": float(worst.get("GPS_Lon", 0))},
                    "value": round(float(worst["WaterTemp_C"]), 1),
                })

        # ── Rule 4: Oil pressure drop (<5th percentile) ──
        if "OilPressure_bar" in stats:
            threshold = stats["OilPressure_bar"]["p5"]
            drops = lap_df[lap_df["OilPressure_bar"] < threshold]
            if len(drops) > 0:
                worst = drops.loc[drops["OilPressure_bar"].idxmin()]
                points.append({
                    "type": "point",
                    "system": "Thermal",
                    "severity": "HIGH",
                    "channel": "OilPressure_bar",
                    "reason": f"Oil pressure {float(worst['OilPressure_bar']):.2f} bar (5th pct = {threshold:.2f} bar)",
                    "lap": int(lap_num),
                    "time_s": float(worst["time_s"]),
                    "gps": {"lat": float(worst.get("GPS_Lat", 0)), "lon": float(worst.get("GPS_Lon", 0))},
                    "value": round(float(worst["OilPressure_bar"]), 2),
                })

        # ── Rule 5: Lateral G spike (>99th percentile — possible slide) ──
        if "LateralAcc_g" in stats:
            threshold = stats["LateralAcc_g"]["p99"]
            spikes = lap_df[lap_df["LateralAcc_g"].abs() > abs(threshold)]
            if len(spikes) > 0:
                worst = spikes.loc[spikes["LateralAcc_g"].abs().idxmax()]
                points.append({
                    "type": "point",
                    "system": "Suspension",
                    "severity": "MEDIUM",
                    "channel": "LateralAcc_g",
                    "reason": f"Lateral G spike: {float(worst['LateralAcc_g']):.2f}g (99th pct = {abs(threshold):.2f}g)",
                    "lap": int(lap_num),
                    "time_s": float(worst["time_s"]),
                    "gps": {"lat": float(worst.get("GPS_Lat", 0)), "lon": float(worst.get("GPS_Lon", 0))},
                    "value": round(float(worst["LateralAcc_g"]), 2),
                })

        # ── Rule 6: Voltage drop (<11.5V) ──
        if "ExternalVoltage_V" in lap_df.columns:
            drops = lap_df[lap_df["ExternalVoltage_V"] < 11.5]
            if len(drops) > 0:
                worst = drops.loc[drops["ExternalVoltage_V"].idxmin()]
                points.append({
                    "type": "point",
                    "system": "Electronics",
                    "severity": "HIGH" if float(worst["ExternalVoltage_V"]) < 10.5 else "MEDIUM",
                    "channel": "ExternalVoltage_V",
                    "reason": f"Battery voltage drop: {float(worst['ExternalVoltage_V']):.1f}V",
                    "lap": int(lap_num),
                    "time_s": float(worst["time_s"]),
                    "gps": {"lat": float(worst.get("GPS_Lat", 0)), "lon": float(worst.get("GPS_Lon", 0))},
                    "value": round(float(worst["ExternalVoltage_V"]), 1),
                })

    return zones, points
```

- [ ] **Step 2: Add degradation detection function**

Add after `_detect_threshold_anomalies`:

```python
def _detect_degradation(session_id: str) -> list[dict]:
    """Detect per-channel degradation trends across laps using linear regression."""
    from scipy.stats import linregress

    db = _get_db()
    lap_docs = list(db["aim_lap_summary"].find(
        {"session_id": session_id},
        {"_id": 0},
    ).sort("lap_number", 1))

    if len(lap_docs) < 3:
        return []

    df = pd.DataFrame(lap_docs)

    # Exclude outlaps (first and last if duration > 1.5× median)
    if "lap_time_s" in df.columns and len(df) > 2:
        median_time = df["lap_time_s"].median()
        df = df[df["lap_time_s"] < median_time * 1.5].reset_index(drop=True)

    if len(df) < 3:
        return []

    channels = [
        ("avg_OilPressure_bar", "Oil Pressure", "Thermal", "bar"),
        ("avg_OilTemp_C", "Oil Temperature", "Thermal", "°C"),
        ("avg_WaterTemp_C", "Water Temperature", "Thermal", "°C"),
        ("avg_FuelPressure_bar", "Fuel Pressure", "Electronics", "bar"),
        ("avg_ExternalVoltage_V", "Battery Voltage", "Electronics", "V"),
        ("avg_FrontBrake_bar", "Front Brake Pressure", "Brakes", "bar"),
    ]

    degradations = []
    x = np.arange(len(df))

    for col, label, system, unit in channels:
        if col not in df.columns:
            continue
        values = df[col].dropna()
        if len(values) < 3:
            continue

        slope, intercept, r_value, p_value, std_err = linregress(x[:len(values)], values)

        mean_val = float(values.mean())
        if mean_val == 0:
            continue

        rate_pct = (slope / mean_val) * 100

        # Flag if trend is significant and rate > 1% per lap
        if p_value < 0.1 and abs(rate_pct) > 1.0:
            direction = "increasing" if slope > 0 else "decreasing"

            # Determine severity
            if abs(rate_pct) > 5:
                severity = "HIGH"
            elif abs(rate_pct) > 2:
                severity = "MEDIUM"
            else:
                severity = "LOW"

            # Is this direction bad?
            bad_increasing = col in ("avg_OilTemp_C", "avg_WaterTemp_C")
            bad_decreasing = col in ("avg_OilPressure_bar", "avg_FuelPressure_bar", "avg_ExternalVoltage_V", "avg_FrontBrake_bar")
            is_concerning = (direction == "increasing" and bad_increasing) or (direction == "decreasing" and bad_decreasing)

            if not is_concerning:
                continue

            message = f"{label} {direction} {abs(rate_pct):.1f}% per lap"
            if severity in ("HIGH", "MEDIUM"):
                message += " — investigate before next session"

            degradations.append({
                "system": system,
                "channel": col.replace("avg_", ""),
                "label": label,
                "unit": unit,
                "trend": direction,
                "rate_per_lap": round(float(slope), 4),
                "rate_pct_per_lap": round(rate_pct, 1),
                "r_squared": round(float(r_value ** 2), 3),
                "message": message,
                "severity": severity,
                "lap_values": [round(float(v), 2) for v in values.tolist()],
            })

    return degradations
```

- [ ] **Step 3: Add lap delta analysis function**

Add after `_detect_degradation`:

```python
def _compute_lap_deltas(session_id: str) -> list[dict]:
    """Compare consecutive hot laps to find where time was gained/lost."""
    db = _get_db()
    session = db["aim_sessions"].find_one({"session_id": session_id}, {"_id": 0, "laps": 1})
    if not session or not session.get("laps"):
        return []

    laps = session["laps"]
    if len(laps) < 2:
        return []

    # Exclude outlaps
    median_time = np.median([l["lap_time_s"] for l in laps])
    hot_laps = [l for l in laps if l["lap_time_s"] < median_time * 1.5]
    if len(hot_laps) < 2:
        return []

    # Get per-lap summary stats for comparison
    summaries = {
        doc["lap_number"]: doc
        for doc in db["aim_lap_summary"].find(
            {"session_id": session_id},
            {"_id": 0},
        )
    }

    deltas = []
    for i in range(1, len(hot_laps)):
        prev = hot_laps[i - 1]
        curr = hot_laps[i]
        delta_s = curr["lap_time_s"] - prev["lap_time_s"]
        faster = delta_s < 0

        reasons = []
        prev_sum = summaries.get(prev["lap_number"], {})
        curr_sum = summaries.get(curr["lap_number"], {})

        # Compare key channels
        comparisons = [
            ("avg_FrontBrake_bar", "Brake pressure", "bar"),
            ("max_FrontBrake_bar", "Peak brake force", "bar"),
            ("avg_GPS_Speed_kmh", "Avg speed", "km/h"),
            ("max_GPS_Speed_kmh", "Top speed", "km/h"),
            ("avg_Throttle_pct", "Throttle application", "%"),
            ("avg_LateralAcc_g", "Cornering G", "g"),
            ("brake_application_pct", "Brake usage", "%"),
        ]

        for col, label, unit in comparisons:
            prev_val = prev_sum.get(col)
            curr_val = curr_sum.get(col)
            if prev_val is None or curr_val is None:
                continue
            diff = curr_val - prev_val
            pct_diff = (diff / prev_val * 100) if prev_val != 0 else 0

            if abs(pct_diff) > 2:  # Only flag >2% changes
                direction = "more" if diff > 0 else "less"
                reasons.append({
                    "channel": col,
                    "detail": f"{abs(pct_diff):.1f}% {direction} {label.lower()} ({curr_val:.1f} vs {prev_val:.1f} {unit})",
                })

        deltas.append({
            "lap": curr["lap_number"],
            "delta_s": round(delta_s, 3),
            "vs_lap": prev["lap_number"],
            "faster": faster,
            "reasons": reasons[:4],  # Top 4 reasons
        })

    return deltas
```

- [ ] **Step 4: Add the unified `run_track_anomalies` function**

Add after `_compute_lap_deltas`:

```python
def run_track_anomalies(session_id: str) -> dict:
    """Run the full track-level intelligence pipeline.

    Returns anomaly zones, point events, degradation trends, and lap deltas.
    """
    meta = load_aim_session_meta(session_id)

    zones, point_events = _detect_threshold_anomalies(session_id)
    degradation = _detect_degradation(session_id)
    lap_deltas = _compute_lap_deltas(session_id)

    return {
        "session_id": session_id,
        "driver": meta.get("driver", ""),
        "track": meta.get("track", ""),
        "anomaly_zones": zones,
        "point_events": point_events,
        "degradation": degradation,
        "lap_deltas": lap_deltas,
        "summary": {
            "total_anomalies": len(zones) + len(point_events),
            "degradation_count": len(degradation),
            "worst_severity": max(
                [z.get("severity", "NORMAL") for z in zones] +
                [p.get("severity", "NORMAL") for p in point_events] +
                ["NORMAL"],
                key=lambda s: ["NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"].index(s)
                if s in ["NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"] else 0,
            ),
        },
    }
```

- [ ] **Step 5: Test the backend locally**

```bash
PYTHONPATH="omnisuitef1:pipeline:." python3 -c "
from pipeline.aim_anomaly import run_track_anomalies
import json
result = run_track_anomalies('aim_20251021_130000_ignacio_ruiz')
print(json.dumps(result, indent=2, default=str))
"
```

Expected: JSON with anomaly_zones, point_events (oil temp spikes on later laps), degradation (oil temp trending up ~3.8°C/lap), and lap_deltas.

- [ ] **Step 6: Commit**

```bash
git add pipeline/aim_anomaly.py
git commit -m "feat(aim): add track-level anomaly detection, degradation, and lap deltas"
```

---

## Task 2: Backend — API endpoint

**Files:**
- Modify: `pipeline/aim_router.py`

- [ ] **Step 1: Add the endpoint**

Add after the existing `get_anomaly` endpoint in `pipeline/aim_router.py`:

```python
@router.get("/track-anomalies/{session_id}")
async def get_track_anomalies(session_id: str):
    """Track-level intelligence: anomaly zones, point events, degradation, lap deltas."""
    from pipeline.aim_anomaly import run_track_anomalies
    try:
        return run_track_anomalies(session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Track anomaly analysis failed: %s", e)
        raise HTTPException(status_code=500, detail="Analysis failed")
```

- [ ] **Step 2: Commit**

```bash
git add pipeline/aim_router.py
git commit -m "feat(aim): add GET /api/aim/track-anomalies endpoint"
```

---

## Task 3: Frontend — API client + data fetching

**Files:**
- Modify: `frontend/src/app/api/local.ts`
- Modify: `frontend/src/app/components/RealRacing.tsx` (state + fetch)

- [ ] **Step 1: Add API call to local.ts**

In `frontend/src/app/api/local.ts`, add to the `aim` object after the `compare` method:

```typescript
  trackAnomalies: (id: string) => fetchLocal<any>(`aim/track-anomalies/${id}`),
```

- [ ] **Step 2: Add state and fetch to RealRacing.tsx**

In the `RealRacing` component, add state after the existing `anomaly` state:

```typescript
const [trackAnomalies, setTrackAnomalies] = useState<any>(null);
```

In the `useEffect` that loads session data, add to the `Promise.all`:

```typescript
aim.trackAnomalies(selectedId).catch(() => null),
```

And destructure in the `.then`:

```typescript
.then(([sess, tel, trk, hlth, anom, trkAnom]) => {
    // ...existing...
    setTrackAnomalies(trkAnom);
})
```

- [ ] **Step 3: Build and verify**

```bash
cd frontend && npx vite build
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/app/api/local.ts frontend/src/app/components/RealRacing.tsx
git commit -m "feat(real-racing): fetch track anomalies data"
```

---

## Task 4: Frontend — Anomalies tab + intelligence cards

**Files:**
- Modify: `frontend/src/app/components/RealRacing.tsx`

- [ ] **Step 1: Add Anomalies tab to TrackMapOverlay**

In the `TrackMapOverlay` component, add `'anomalies'` to the mode type and add a third tab button:

```typescript
const [mode, setMode] = useState<'compare' | 'metric' | 'anomalies'>('compare');
```

Add the third tab button after the "Metric" button:

```typescript
<button
  onClick={() => setMode('anomalies')}
  className={`text-[10px] px-2.5 py-1 rounded-full border transition-colors ${
    mode === 'anomalies'
      ? 'bg-primary text-background border-primary font-medium'
      : 'border-border text-muted-foreground hover:text-foreground'
  }`}
>
  Anomalies
</button>
```

- [ ] **Step 2: Add anomalies prop and SVG markers**

Add `trackAnomalies` to the component props:

```typescript
function TrackMapOverlay({ trackData, laps, activeLap, trackAnomalies }: {
  trackData: any[];
  laps: { lap_number: number; lap_time_s: number }[];
  activeLap: number | null;
  trackAnomalies: any;
})
```

Inside the SVG, after the existing render groups, add anomaly markers when in anomaly mode:

```typescript
{/* Anomaly mode: dim base track + overlay markers */}
{mode === 'anomalies' && trackAnomalies && (
  <>
    {/* Anomaly zones (highlighted segments) */}
    {(trackAnomalies.anomaly_zones || []).map((zone: any, i: number) => {
      if (!zone.start_gps?.lat || !zone.end_gps?.lat) return null;
      const p1 = toSvg({ GPS_Lat: zone.start_gps.lat, GPS_Lon: zone.start_gps.lon });
      const p2 = toSvg({ GPS_Lat: zone.end_gps.lat, GPS_Lon: zone.end_gps.lon });
      const color = zone.severity === 'CRITICAL' ? RED : zone.severity === 'HIGH' ? PAPAYA : AMBER;
      return (
        <g key={`zone-${i}`}>
          <line x1={p1.x} y1={p1.y} x2={p2.x} y2={p2.y}
            stroke={color} strokeWidth={8} strokeLinecap="round" opacity={0.4} />
          <line x1={p1.x} y1={p1.y} x2={p2.x} y2={p2.y}
            stroke={color} strokeWidth={3} strokeLinecap="round" opacity={0.9} />
        </g>
      );
    })}
    {/* Point events (dots) */}
    {(trackAnomalies.point_events || []).map((pt: any, i: number) => {
      if (!pt.gps?.lat) return null;
      const p = toSvg({ GPS_Lat: pt.gps.lat, GPS_Lon: pt.gps.lon });
      const SYSTEM_COLORS: Record<string, string> = {
        Brakes: RED, Thermal: PAPAYA, Suspension: PURPLE,
        Electronics: CYAN, 'Power Unit': LIME, Drivetrain: AMBER, 'Tyre Management': PINK,
      };
      const color = SYSTEM_COLORS[pt.system] || AMBER;
      return (
        <g key={`pt-${i}`}>
          <circle cx={p.x} cy={p.y} r={7} fill={color} opacity={0.25} />
          <circle cx={p.x} cy={p.y} r={4} fill={color} stroke="white" strokeWidth={1} />
          <title>{`${pt.system}: ${pt.reason} (Lap ${pt.lap})`}</title>
        </g>
      );
    })}
  </>
)}
```

For anomaly mode, render the base track as dim gray instead of the normal colored lines. Update the `renderGroups` logic:

```typescript
const renderGroups = useMemo(() => {
  if (mode === 'compare') {
    return [...selectedLaps].map(lap => ({ lap, points: lapGroups.get(lap) || [] }));
  }
  if (mode === 'anomalies') {
    // Render all data as dim base track
    return [{ lap: 0, points: trackData }];
  }
  const pts = activeLap != null ? (lapGroups.get(activeLap) || []) : trackData;
  return [{ lap: activeLap ?? 0, points: pts }];
}, [mode, selectedLaps, activeLap, lapGroups, trackData]);
```

Update the SVG color logic for anomaly mode:

```typescript
const segColor = mode === 'compare'
  ? color!
  : mode === 'anomalies'
  ? 'rgba(255,255,255,0.15)'
  : metricColor(metric, pt[metric] ?? 0, metricBounds.min, metricBounds.max);
```

- [ ] **Step 3: Add anomaly legend below map in anomalies mode**

After the existing metric controls, add:

```typescript
{mode === 'anomalies' && trackAnomalies && (
  <div className="mt-3 space-y-1.5">
    <div className="flex flex-wrap gap-2 text-[10px]">
      {[
        { label: 'Brakes', color: RED },
        { label: 'Thermal', color: PAPAYA },
        { label: 'Suspension', color: PURPLE },
        { label: 'Electronics', color: CYAN },
      ].map(s => (
        <div key={s.label} className="flex items-center gap-1 text-muted-foreground">
          <span className="w-2.5 h-2.5 rounded-full" style={{ background: s.color }} />
          {s.label}
        </div>
      ))}
    </div>
    <div className="text-[10px] text-muted-foreground">
      {(trackAnomalies.summary?.total_anomalies || 0)} anomalies detected
    </div>
  </div>
)}
```

- [ ] **Step 4: Add intelligence cards below the track map**

After the existing `{/* ── Section 5: Health Dashboard ── */}` section, add:

```typescript
{/* ── Section 6: Intelligence Cards ── */}
{!telLoading && trackAnomalies && (
  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
    {/* Anomaly Alerts */}
    {(trackAnomalies.anomaly_zones?.length > 0 || trackAnomalies.point_events?.length > 0) && (
      <div className="bg-card border border-border rounded-xl p-4">
        <div className="flex items-center gap-2 mb-3">
          <AlertTriangle className="w-4 h-4 text-red-400" />
          <span className="text-sm font-medium text-foreground">Anomaly Alerts</span>
          <span className="ml-auto text-[10px] text-muted-foreground">
            {(trackAnomalies.anomaly_zones?.length || 0) + (trackAnomalies.point_events?.length || 0)} issues
          </span>
        </div>
        <div className="space-y-2 max-h-48 overflow-y-auto">
          {[...(trackAnomalies.anomaly_zones || []), ...(trackAnomalies.point_events || [])]
            .sort((a: any, b: any) => {
              const order = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'NORMAL'];
              return order.indexOf(a.severity) - order.indexOf(b.severity);
            })
            .map((a: any, i: number) => (
              <div key={i} className="flex items-start gap-2 text-[11px]">
                <span className="w-2 h-2 rounded-full mt-1 shrink-0"
                  style={{ background: a.severity === 'HIGH' ? RED : a.severity === 'CRITICAL' ? RED : AMBER }} />
                <div>
                  <span className="text-foreground font-medium">{a.system}</span>
                  <span className="text-muted-foreground"> · Lap {a.lap} · {a.severity}</span>
                  <div className="text-muted-foreground">{a.reason}</div>
                </div>
              </div>
            ))}
        </div>
      </div>
    )}

    {/* Degradation Watch */}
    {trackAnomalies.degradation?.length > 0 && (
      <div className="bg-card border border-border rounded-xl p-4">
        <div className="flex items-center gap-2 mb-3">
          <BarChart3 className="w-4 h-4 text-amber-400" />
          <span className="text-sm font-medium text-foreground">Degradation Watch</span>
          <span className="ml-auto text-[10px] text-muted-foreground">
            {trackAnomalies.degradation.length} trends
          </span>
        </div>
        <div className="space-y-3">
          {trackAnomalies.degradation.map((d: any, i: number) => (
            <div key={i}>
              <div className="flex items-center justify-between text-[11px]">
                <span className="text-foreground font-medium">{d.label}</span>
                <span className={`px-1.5 py-0.5 rounded text-[9px] font-medium ${
                  d.severity === 'HIGH' ? 'bg-red-500/10 text-red-400' : 'bg-amber-500/10 text-amber-400'
                }`}>
                  {d.rate_pct_per_lap > 0 ? '+' : ''}{d.rate_pct_per_lap}%/lap
                </span>
              </div>
              <div className="text-[10px] text-muted-foreground mt-0.5">{d.message}</div>
              {/* Mini sparkline */}
              <div className="flex items-end gap-px mt-1.5 h-6">
                {d.lap_values.map((v: number, j: number) => {
                  const min = Math.min(...d.lap_values);
                  const max = Math.max(...d.lap_values);
                  const pct = max === min ? 50 : ((v - min) / (max - min)) * 100;
                  return (
                    <div key={j} className="flex-1 rounded-sm"
                      style={{
                        height: `${Math.max(10, pct)}%`,
                        background: d.severity === 'HIGH' ? RED : AMBER,
                        opacity: 0.3 + (j / d.lap_values.length) * 0.7,
                      }}
                      title={`Lap ${j + 1}: ${v} ${d.unit}`}
                    />
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      </div>
    )}

    {/* Lap Deltas — full width */}
    {trackAnomalies.lap_deltas?.length > 0 && (
      <div className="lg:col-span-2 bg-card border border-border rounded-xl p-4">
        <div className="flex items-center gap-2 mb-3">
          <Timer className="w-4 h-4 text-primary" />
          <span className="text-sm font-medium text-foreground">Lap-over-Lap Analysis</span>
        </div>
        <div className="space-y-2">
          {trackAnomalies.lap_deltas.map((d: any, i: number) => (
            <div key={i} className="flex items-start gap-3 text-[11px]">
              <div className="shrink-0 w-20 text-right">
                <span className="text-muted-foreground">L{d.vs_lap}→L{d.lap}</span>
              </div>
              <div className={`shrink-0 w-16 font-mono font-bold ${d.faster ? 'text-green-400' : 'text-red-400'}`}>
                {d.delta_s > 0 ? '+' : ''}{d.delta_s.toFixed(3)}s
              </div>
              <div className="text-muted-foreground flex-1">
                {d.reasons.length > 0
                  ? d.reasons.map((r: any) => r.detail).join(' · ')
                  : 'No significant channel changes detected'}
              </div>
            </div>
          ))}
        </div>
      </div>
    )}

    {/* All nominal state */}
    {!trackAnomalies.anomaly_zones?.length && !trackAnomalies.point_events?.length && !trackAnomalies.degradation?.length && (
      <div className="lg:col-span-2 bg-card border border-border rounded-xl p-4 text-center">
        <div className="text-green-400 text-sm font-medium">All Systems Nominal</div>
        <div className="text-[11px] text-muted-foreground mt-1">No anomalies or degradation trends detected</div>
      </div>
    )}
  </div>
)}
```

- [ ] **Step 5: Update TrackMapOverlay usage to pass trackAnomalies**

Where `TrackMapOverlay` is used in the render:

```typescript
<TrackMapOverlay trackData={trackData} laps={session?.laps || []} activeLap={activeLap} trackAnomalies={trackAnomalies} />
```

- [ ] **Step 6: Build and verify**

```bash
cd frontend && npx vite build
```

- [ ] **Step 7: Commit**

```bash
git add frontend/src/app/components/RealRacing.tsx
git commit -m "feat(real-racing): add anomalies tab, degradation watch, and lap delta cards"
```

---

## Task 5: Deploy and verify

- [ ] **Step 1: Push to both remotes**

```bash
git push origin main && git push marip main
```

- [ ] **Step 2: Deploy frontend**

```bash
cd frontend && vercel deploy --prod
```

- [ ] **Step 3: Verify on live site**

Check Real Racing section:
- Anomalies tab shows dim gray track with colored markers for detected issues
- Anomaly Alerts card lists flagged issues sorted by severity
- Degradation Watch card shows oil temp trend with sparkline
- Lap-over-Lap Analysis shows time deltas with channel-level reasons
- If no anomalies: "All Systems Nominal" green state
