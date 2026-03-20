# Real Racing Intelligence Layer — Design Spec

## Summary

Add four intelligence features to the Real Racing section that go beyond what AiM RaceStudio 3 offers, showcasing OmniSense's AI/ML strengths:

1. **Auto-detected anomalies on the track map** — flag braking zones, overheating corners, pressure drops at exact GPS locations
2. **Predictive degradation** — "Oil pressure trending down 3% per lap — investigate before next session"
3. **Lap-over-lap trend alerts** — auto-highlight where the driver gained/lost time with reasons
4. **System risk heatmap on track** — overlay which track sections stress which systems most

## Architecture

### Hybrid anomaly detection (ML ensemble + statistical thresholds)

- **Lap-level ML** (existing): `run_aim_anomaly()` scores each lap per system via the AnomalyEnsemble — provides high-level "this system is degrading" intelligence
- **Sample-level thresholds** (new): run statistical rules on raw telemetry to pinpoint exact GPS locations where values exceed bounds (>2σ from lap mean, percentile violations, rate-of-change spikes)

### New backend endpoint

`GET /api/aim/track-anomalies/{session_id}` — returns:
```json
{
  "session_id": "...",
  "anomaly_zones": [
    {
      "type": "segment",
      "system": "Brakes",
      "severity": "HIGH",
      "channel": "FrontBrake_bar",
      "reason": "Brake pressure dropped 45% below lap mean",
      "lap": 3,
      "start_time_s": 142.3,
      "end_time_s": 143.8,
      "start_gps": { "lat": 40.123, "lon": -3.456 },
      "end_gps": { "lat": 40.124, "lon": -3.457 }
    }
  ],
  "point_events": [
    {
      "type": "point",
      "system": "Thermal",
      "severity": "MEDIUM",
      "channel": "OilTemp_C",
      "reason": "Oil temp spike: 128°C (95th pct = 115°C)",
      "lap": 5,
      "time_s": 287.4,
      "gps": { "lat": 40.125, "lon": -3.458 },
      "value": 128.3
    }
  ],
  "degradation": [
    {
      "system": "Thermal",
      "channel": "OilPressure_bar",
      "trend": "decreasing",
      "rate_per_lap": -0.12,
      "rate_pct_per_lap": -3.1,
      "message": "Oil pressure trending down 3.1% per lap — investigate before next session",
      "severity": "MEDIUM",
      "lap_values": [3.89, 3.85, 3.78, 3.72, 3.65, 3.58, 3.51]
    }
  ],
  "lap_deltas": [
    {
      "lap": 4,
      "delta_s": -0.82,
      "vs_lap": 3,
      "faster": true,
      "reasons": [
        { "zone": "Turn 3-4", "channel": "FrontBrake_bar", "detail": "Later braking by 0.15s" },
        { "zone": "Straight", "channel": "Throttle_pct", "detail": "2.3% more throttle application" }
      ]
    }
  ]
}
```

### Statistical threshold rules

Applied to raw `aim_telemetry` data per lap:

| Rule | Channel | Condition | Anomaly type |
|------|---------|-----------|-------------|
| Brake pressure drop | FrontBrake_bar | >2σ below lap mean during braking zone | segment |
| Oil temp spike | OilTemp_C | Above 95th percentile of session | point |
| Water temp spike | WaterTemp_C | Above 95th percentile of session | point |
| Oil pressure drop | OilPressure_bar | Below 5th percentile of session | point |
| Throttle inconsistency | Throttle_pct | >2σ deviation in same track section across laps | segment |
| Lateral G spike | LateralAcc_g | Above 99th percentile (possible slide/snap) | point |
| Voltage drop | ExternalVoltage_V | Below 11.5V | point |

### Degradation detection

For each monitored channel, run linear regression on per-lap averages:
- If slope is significant (p < 0.1) and rate > 1% per lap, flag as degradation
- Channels: OilPressure_bar, OilTemp_C, WaterTemp_C, FuelPressure_bar, ExternalVoltage_V, FrontBrake_bar

### Lap delta analysis

Compare consecutive laps sector-by-sector:
- Split each lap into N segments by distance (or time)
- Compare segment times between laps
- For segments with >0.1s difference, identify which telemetry channels changed most (brake point, throttle %, speed delta)

## File changes

| File | Action | Responsibility |
|------|--------|---------------|
| `pipeline/aim_anomaly.py` | Modify | Add `run_track_anomalies()`, threshold rules, degradation detection, lap delta analysis |
| `pipeline/aim_router.py` | Modify | Add `GET /api/aim/track-anomalies/{session_id}` |
| `frontend/src/app/api/local.ts` | Modify | Add `aim.trackAnomalies(id)` |
| `frontend/src/app/components/RealRacing.tsx` | Modify | Add anomaly markers to TrackMapOverlay, add intelligence cards below map |
| `frontend/vercel.json` | No change needed | `/api/aim/*` rewrite already covers this |

## Frontend: Track map markers

### Anomaly segments (highlighted track sections)

- Render as thicker SVG lines (5px) with glow effect over the racing line
- Color by severity: MEDIUM=amber, HIGH=orange, CRITICAL=red
- Semi-transparent so the racing line shows through
- On hover: tooltip with system, channel, reason, value

### Point events (pulsing dots)

- SVG circles at GPS location, 6px radius
- Color by system: Brakes=red, Thermal=orange, Suspension=purple, Electronics=cyan, Power Unit=green, Drivetrain=amber, Tyre=pink
- CSS pulse animation for HIGH/CRITICAL severity
- On click: tooltip with details

### New "Anomaly" toggle on TrackMapOverlay

Add a third mode tab alongside "Lap Compare" and "Metric":
```
[Lap Compare] [Metric] [Anomalies]
```

In Anomaly mode:
- Base track rendered as dim gray line
- Anomaly segments and point events overlaid
- Legend showing system colors
- Filter checkboxes for severity levels

## Frontend: Intelligence cards below map

### Layout
```
┌─────────────────────────────────────────────┐
│          TrackMapOverlay (existing)          │
└─────────────────────────────────────────────┘

┌──────────────────┐ ┌──────────────────────┐
│ 🔴 Anomaly Alerts │ │ 📉 Degradation Watch │
│ 3 issues found   │ │ 2 trends flagged     │
│                  │ │                      │
│ • Brake drop T3  │ │ • Oil press -3.1%/lap│
│   Lap 3, HIGH    │ │   ⚠ Investigate      │
│ • Oil spike T7   │ │ • Water temp +1.2/lap│
│   Lap 5, MEDIUM  │ │   📊 Monitor         │
└──────────────────┘ └──────────────────────┘

┌─────────────────────────────────────────────┐
│ ⏱ Lap-over-Lap Deltas                       │
│                                             │
│ L3→L4: -0.82s ✅  Later braking T3, more   │
│                    throttle on straight      │
│ L4→L5: +0.34s 🔻  Early lift T5, brake      │
│                    pressure drop T7          │
└─────────────────────────────────────────────┘
```

## Edge cases

- **No anomalies detected**: Show "All systems nominal" green state
- **Missing GPS data**: Skip point/segment markers, still show cards
- **Single lap session**: Skip lap deltas, still run thresholds and degradation
- **Threshold tuning**: Use session percentiles, not absolute values — adapts to different cars/tracks

## Out of scope (for now)

- Real-time anomaly detection during live sessions
- Custom threshold configuration UI
- Anomaly history across multiple sessions (future: session comparison feature)
