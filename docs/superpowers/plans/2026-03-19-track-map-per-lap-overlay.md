# Track Map Per-Lap Overlay Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the basic `TrackMap` component in RealRacing.tsx with a two-mode overlay that supports lap-to-lap racing line comparison and per-metric color-coded visualization.

**Architecture:** Frontend-only change. The existing `/api/aim/track/{session_id}` endpoint already returns GPS points with a `lap` field. We group points client-side by lap, render SVG polylines per lap in Compare mode, and color-code segments by metric value in Metric Overlay mode.

**Tech Stack:** React, SVG, TypeScript, Recharts (existing), Lucide icons (existing)

**Spec:** `docs/superpowers/specs/2026-03-19-track-map-per-lap-overlay-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `frontend/src/app/components/RealRacing.tsx` | Modify | Replace `TrackMap` with `TrackMapOverlay`, update data fetching to remove lap filter on track endpoint |

All changes are in one file since `TrackMap` is already defined inline in `RealRacing.tsx` and the new component follows the same pattern.

---

## Chunk 1: Implementation

### Task 1: Update track data fetching to load all laps

**Files:**
- Modify: `frontend/src/app/components/RealRacing.tsx:422-431`

- [ ] **Step 1: Change track fetch to omit lap filter**

In the `useEffect` that loads session data (line ~422), change the `aim.track()` call to always fetch all laps (no lap filter), so the overlay component has all GPS data available:

```typescript
// Before:
aim.track(selectedId, activeLap ?? undefined),

// After:
aim.track(selectedId),  // always fetch all laps for overlay
```

The telemetry fetch keeps using `activeLap` — only the track map needs all laps.

- [ ] **Step 2: Verify the app still loads without errors**

Run: `npx vite build` from `frontend/`
Expected: Build passes

- [ ] **Step 3: Commit**

```bash
git add frontend/src/app/components/RealRacing.tsx
git commit -m "feat(real-racing): fetch all laps for track overlay"
```

---

### Task 2: Build TrackMapOverlay component — Lap Compare mode

**Files:**
- Modify: `frontend/src/app/components/RealRacing.tsx` — replace `TrackMap` function (lines 223-286)

- [ ] **Step 1: Define constants and types**

Add above the component, after the existing `CHART_GROUPS`:

```typescript
// ── Lap overlay colors ──────────────────────────────────────────────
const LAP_COLORS = [PAPAYA, CYAN, LIME, PURPLE, PINK, TEAL, AMBER];

const METRIC_OPTIONS: { key: string; label: string; unit: string }[] = [
  { key: 'GPS_Speed_kmh', label: 'Speed', unit: 'km/h' },
  { key: 'FrontBrake_bar', label: 'Braking', unit: 'bar' },
  { key: 'Throttle_pct', label: 'Throttle', unit: '%' },
  { key: 'LateralAcc_g', label: 'Lateral G', unit: 'g' },
  { key: 'Gear', label: 'Gear', unit: '' },
];

// Color scales per metric
function metricColor(key: string, value: number, min: number, max: number): string {
  const range = max - min || 1;
  const ratio = Math.max(0, Math.min(1, (value - min) / range));

  if (key === 'Gear') {
    const gearColors = ['#EF4444', '#F59E0B', '#EAB308', '#22C55E', '#47C7FC', '#3B82F6', '#A855F7'];
    return gearColors[Math.round(value)] || '#666';
  }
  if (key === 'FrontBrake_bar') {
    return `rgba(239, 68, 68, ${0.1 + ratio * 0.9})`;
  }
  if (key === 'Throttle_pct') {
    return `rgba(34, 197, 94, ${0.1 + ratio * 0.9})`;
  }
  if (key === 'LateralAcc_g') {
    // blue (left) → white (neutral) → red (right)
    if (ratio < 0.5) {
      const t = ratio * 2;
      return `rgb(${Math.round(100 + 155 * t)}, ${Math.round(100 + 155 * t)}, 255)`;
    } else {
      const t = (ratio - 0.5) * 2;
      return `rgb(255, ${Math.round(255 - 155 * t)}, ${Math.round(255 - 155 * t)})`;
    }
  }
  // Speed: green → yellow → orange → red
  const r = Math.round(255 * ratio);
  const g = Math.round(255 * (1 - ratio * 0.5));
  const b = Math.round(80 * (1 - ratio));
  return `rgb(${r},${g},${b})`;
}
```

- [ ] **Step 2: Build the TrackMapOverlay component**

Replace the entire `TrackMap` function (lines 223-286) with:

```typescript
function TrackMapOverlay({ trackData, laps, activeLap }: {
  trackData: any[];
  laps: { lap_number: number; lap_time_s: number }[];
  activeLap: number | null;
}) {
  const [mode, setMode] = useState<'compare' | 'metric'>('compare');
  const [selectedLaps, setSelectedLaps] = useState<Set<number>>(new Set([1]));
  const [metric, setMetric] = useState('GPS_Speed_kmh');

  // Group points by lap
  const lapGroups = useMemo(() => {
    const groups = new Map<number, any[]>();
    for (const pt of trackData) {
      const lap = pt.lap ?? 1;
      if (!groups.has(lap)) groups.set(lap, []);
      groups.get(lap)!.push(pt);
    }
    return groups;
  }, [trackData]);

  // Available lap numbers
  const lapNumbers = useMemo(() => [...lapGroups.keys()].sort((a, b) => a - b), [lapGroups]);

  // Toggle lap selection (max 3)
  const toggleLap = (lap: number) => {
    setSelectedLaps(prev => {
      const next = new Set(prev);
      if (next.has(lap)) {
        next.delete(lap);
      } else {
        if (next.size >= 3) {
          // Remove oldest (first in set)
          const oldest = next.values().next().value;
          next.delete(oldest);
        }
        next.add(lap);
      }
      return next;
    });
  };

  // Points for current mode
  const renderPoints = useMemo(() => {
    if (mode === 'compare') {
      return [...selectedLaps].map(lap => ({
        lap,
        points: lapGroups.get(lap) || [],
      }));
    }
    // Metric mode: use activeLap or all data
    const pts = activeLap != null ? (lapGroups.get(activeLap) || []) : trackData;
    return [{ lap: activeLap ?? 0, points: pts }];
  }, [mode, selectedLaps, activeLap, lapGroups, trackData]);

  if (!trackData.length) return null;

  // SVG bounds from all data
  const xs = trackData.map(d => d.X_m ?? d.GPS_Lon).filter(Boolean);
  const ys = trackData.map(d => d.Y_m ?? d.GPS_Lat).filter(Boolean);
  if (!xs.length || !ys.length) return null;

  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const rangeX = maxX - minX || 1, rangeY = maxY - minY || 1;
  const pad = 20, w = 500, h = 400;

  const toSvg = (d: any) => ({
    x: pad + ((d.X_m ?? d.GPS_Lon) - minX) / rangeX * (w - 2 * pad),
    y: pad + (1 - ((d.Y_m ?? d.GPS_Lat) - minY) / rangeY) * (h - 2 * pad),
  });

  // Metric min/max for color scale
  const metricBounds = useMemo(() => {
    if (mode !== 'metric') return { min: 0, max: 1 };
    const vals = trackData.map(d => d[metric]).filter((v: any) => v != null);
    return { min: Math.min(...vals), max: Math.max(...vals) };
  }, [mode, metric, trackData]);

  return (
    <div className="bg-card border border-border rounded-xl p-4">
      {/* Header */}
      <div className="flex items-center gap-2 mb-3">
        <Navigation className="w-4 h-4 text-primary" />
        <span className="text-sm font-medium text-foreground">Track Map</span>
        <div className="ml-auto flex gap-1">
          <button
            onClick={() => setMode('compare')}
            className={`text-[10px] px-2.5 py-1 rounded-full border transition-colors ${
              mode === 'compare'
                ? 'bg-primary text-background border-primary font-medium'
                : 'border-border text-muted-foreground hover:text-foreground'
            }`}
          >
            Lap Compare
          </button>
          <button
            onClick={() => setMode('metric')}
            className={`text-[10px] px-2.5 py-1 rounded-full border transition-colors ${
              mode === 'metric'
                ? 'bg-primary text-background border-primary font-medium'
                : 'border-border text-muted-foreground hover:text-foreground'
            }`}
          >
            Metric
          </button>
        </div>
      </div>

      {/* SVG Map */}
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full" style={{ background: 'var(--background)', borderRadius: 8 }}>
        {renderPoints.map(({ lap, points }, groupIdx) => {
          const color = mode === 'compare'
            ? LAP_COLORS[lapNumbers.indexOf(lap) % LAP_COLORS.length]
            : undefined;
          const strokeW = mode === 'compare'
            ? ([...selectedLaps].pop() === lap ? 3 : 2)
            : 2.5;

          return points.map((pt: any, i: number) => {
            if (i === 0) return null;
            const prev = points[i - 1];
            const p1 = toSvg(prev);
            const p2 = toSvg(pt);
            const segColor = mode === 'compare'
              ? color!
              : metricColor(metric, pt[metric] ?? 0, metricBounds.min, metricBounds.max);

            return (
              <line
                key={`${groupIdx}-${i}`}
                x1={p1.x} y1={p1.y}
                x2={p2.x} y2={p2.y}
                stroke={segColor}
                strokeWidth={strokeW}
                strokeLinecap="round"
              />
            );
          });
        })}
        {/* Start marker */}
        {trackData.length > 0 && (() => {
          const start = toSvg(trackData[0]);
          return <circle cx={start.x} cy={start.y} r={5} fill={LIME} stroke="white" strokeWidth={1.5} />;
        })()}
      </svg>

      {/* Controls */}
      {mode === 'compare' ? (
        <div className="flex gap-1.5 mt-3 flex-wrap">
          {lapNumbers.map((lap, i) => (
            <button
              key={lap}
              onClick={() => toggleLap(lap)}
              className={`text-[10px] px-2.5 py-1 rounded-full border transition-colors ${
                selectedLaps.has(lap)
                  ? 'font-medium border-transparent text-background'
                  : 'border-border text-muted-foreground hover:text-foreground'
              }`}
              style={selectedLaps.has(lap) ? { background: LAP_COLORS[i % LAP_COLORS.length] } : undefined}
            >
              L{lap}
            </button>
          ))}
          <span className="text-[9px] text-muted-foreground self-center ml-1">max 3</span>
        </div>
      ) : (
        <div className="mt-3 space-y-2">
          <select
            value={metric}
            onChange={e => setMetric(e.target.value)}
            className="bg-secondary text-foreground text-[11px] rounded-lg px-3 py-1.5 border border-border focus:outline-none focus:ring-1 focus:ring-primary"
          >
            {METRIC_OPTIONS.map(m => (
              <option key={m.key} value={m.key}>{m.label}{m.unit ? ` (${m.unit})` : ''}</option>
            ))}
          </select>
          <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
            <span>{metric === 'Gear' ? '1' : 'Low'}</span>
            <div className="flex-1 h-2 rounded-full" style={{
              background: metric === 'FrontBrake_bar'
                ? 'linear-gradient(to right, rgba(239,68,68,0.1), rgb(239,68,68))'
                : metric === 'Throttle_pct'
                ? 'linear-gradient(to right, rgba(34,197,94,0.1), rgb(34,197,94))'
                : metric === 'LateralAcc_g'
                ? 'linear-gradient(to right, rgb(100,100,255), white, rgb(255,100,100))'
                : metric === 'Gear'
                ? 'linear-gradient(to right, #EF4444, #F59E0B, #EAB308, #22C55E, #47C7FC, #3B82F6, #A855F7)'
                : 'linear-gradient(to right, rgb(0,255,80), rgb(255,200,0), rgb(255,128,0), rgb(255,0,0))'
            }} />
            <span>{metric === 'Gear' ? '7' : 'High'}</span>
          </div>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 3: Update TrackMap usage in the render section**

In the main `RealRacing` component's return JSX (line ~572), replace:

```typescript
// Before:
<TrackMap trackData={trackData} />

// After:
<TrackMapOverlay trackData={trackData} laps={session?.laps || []} activeLap={activeLap} />
```

- [ ] **Step 4: Build and verify**

Run: `npx vite build` from `frontend/`
Expected: Build passes with no errors

- [ ] **Step 5: Commit**

```bash
git add frontend/src/app/components/RealRacing.tsx
git commit -m "feat(real-racing): add TrackMapOverlay with lap compare and metric modes"
```

---

### Task 3: Deploy and verify

**Files:** None (deployment only)

- [ ] **Step 1: Push to both remotes**

```bash
git push origin main && git push marip main
```

- [ ] **Step 2: Deploy frontend to Vercel**

```bash
cd frontend && vercel deploy --prod
```

- [ ] **Step 3: Verify on live site**

Open the Real Racing section, select a session, and confirm:
- Lap Compare mode shows colored polylines per lap with toggle pills
- Metric Overlay mode colors the track by selected channel
- Switching between modes works
- Lap toggle max 3 enforcement works
