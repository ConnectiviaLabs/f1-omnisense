# Track Map Per-Lap Overlay — Design Spec

## Summary

Replace the existing `TrackMap` component in `RealRacing.tsx` with `TrackMapOverlay` — a two-mode track visualization that supports lap-to-lap racing line comparison and per-metric color-coded overlays.

## Motivation

The current track map shows a single speed-colored path. Drivers and engineers need to compare racing lines across laps and visualize where braking, throttle, and cornering differ — this is the core workflow in AiM RaceStudio 3 that we're replicating.

## Architecture

**Frontend-only change.** No backend modifications needed — the existing `/api/aim/track/{session_id}` endpoint already returns GPS points with a `lap` field per point.

### File changes

| File | Change |
|------|--------|
| `frontend/src/app/components/RealRacing.tsx` | Replace `TrackMap` with `TrackMapOverlay` |

### Data flow

1. `RealRacing` fetches `aim.track(sessionId)` with **no lap filter** (gets all laps' GPS data)
2. `useMemo` groups points by `lap` number into `Map<number, Point[]>`
3. Lap Compare tab renders one SVG polyline per active lap
4. Metric Overlay tab colors line segments by selected channel value

## Component: `TrackMapOverlay`

### Props

```typescript
interface TrackMapOverlayProps {
  trackData: any[];           // All GPS points with lap field
  laps: { lap_number: number; lap_time_s: number }[];  // From session
  activeLap: number | null;   // Currently selected lap in parent
}
```

### Internal state

```typescript
const [mode, setMode] = useState<'compare' | 'metric'>('compare');
const [activeLaps, setActiveLaps] = useState<Set<number>>(new Set());  // max 3
const [metric, setMetric] = useState<string>('GPS_Speed_kmh');
```

### Two modes

#### Lap Compare mode

- Each active lap rendered as a distinct-colored SVG polyline
- Lap colors: orange (#FF8000), cyan (#47C7FC), lime (#22C55E), purple (#A855F7), pink (#EC4899), teal (#14B8A6), amber (#F59E0B)
- Pill buttons below the map to toggle laps on/off
- Max 3 laps active at once — toggling a 4th deactivates the oldest
- Active lap polyline is 3px stroke; others are 2px
- On mount, auto-activate lap 1

#### Metric Overlay mode

- Uses `activeLap` from the parent (or all laps if null)
- Dropdown to select metric: Speed (GPS_Speed_kmh), Braking (FrontBrake_bar), Throttle (Throttle_pct), Lateral G (LateralAcc_g), Gear (Gear)
- Each line segment colored by the metric value using a min→max gradient
- Color scales:
  - Speed: green → yellow → orange → red (slow → fast)
  - Braking: transparent → red (no brake → full brake)
  - Throttle: transparent → green (off → full)
  - Lateral G: blue → white → red (left → neutral → right)
  - Gear: discrete colors per gear number (1=red, 2=orange, 3=yellow, 4=green, 5=cyan, 6=blue)
- Gradient legend bar below the map

### SVG rendering

- Same viewBox approach as current `TrackMap`: compute bounds from GPS_Lon/GPS_Lat (or X_m/Y_m fallback), pad, normalize to SVG coords
- Start/finish marker: green circle at first point
- Background: `var(--background)` with 8px border-radius

### Layout

```
┌─────────────────────────────────────────────┐
│ 🧭 Track Map    [Lap Compare] [Metric ▾]   │
├─────────────────────────────────────────────┤
│                                             │
│              SVG Track Map                  │
│                                             │
├─────────────────────────────────────────────┤
│ Lap Compare: [L1] [L2] [L3] [L4] ...       │
│   — or —                                    │
│ Metric: [Speed ▾]                           │
│ [Slow ████████████████████████ Fast]        │
└─────────────────────────────────────────────┘
```

## Edge cases

- **No GPS data**: Return null (same as current)
- **Single lap session**: Lap Compare shows one lap, no toggle needed
- **Missing metric values**: Skip segments where the channel is null/undefined
- **Large datasets**: Track endpoint already downsamples to ~2K points

## Out of scope

- Delta time visualization (future enhancement)
- 3D track rendering
- Animation/playback along the track
