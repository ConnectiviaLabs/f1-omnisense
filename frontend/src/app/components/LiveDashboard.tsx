import { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import {
  AreaChart, Area, LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip, ReferenceLine,
} from 'recharts';
import {
  Gauge, Zap, Timer, Loader2, AlertCircle,
  Flag, MapPin, Cloud, Wind, Play, Pause, RotateCcw, Radio, Bug,
  ChevronUp, ChevronDown, Trophy,
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { usePolling } from '../hooks/usePolling';
import * as openf1 from '../api/openf1';
import type { OpenF1Session, OpenF1Lap, OpenF1Driver, OpenF1Stint } from '../types';
import { ScrollArea } from './ui/scroll-area';
import { TrackMapGL } from './TrackMapGL';
import { LiveAnalyticsSidebar } from './LiveAnalyticsSidebar';
import { getCircuitByShortName } from '../data/circuits';
import { HealthGauge } from './HealthGauge';
import { StatusBadge } from './StatusBadge';
import {
  type VehicleData,
  levelColor, levelBg, MAINTENANCE_LABELS, SEVERITY_COLORS,
  parseAnomalyDrivers,
} from './anomalyHelpers';

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload?.length) {
    return (
      <div className="bg-[#0D1117] border border-[rgba(255,128,0,0.2)] rounded-lg p-2 text-[12px]">
        <div className="text-muted-foreground mb-1">{label}</div>
        {payload.map((entry: any, index: number) => (
          <div key={index} className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full" style={{ backgroundColor: entry.color }} />
            <span className="text-muted-foreground">{entry.name}:</span>
            <span className="text-foreground font-mono">{typeof entry.value === 'number' ? entry.value.toFixed(3) : entry.value}</span>
          </div>
        ))}
      </div>
    );
  }
  return null;
};

const SPEED_OPTIONS = [1, 2, 5, 10, 30];

const COMPOUND_COLORS: Record<string, string> = {
  SOFT: '#ef4444', MEDIUM: '#f59e0b', HARD: '#e8e8f0', INTERMEDIATE: '#22c55e', WET: '#3b82f6',
};

function formatElapsed(ms: number): string {
  const totalSec = Math.floor(ms / 1000);
  const h = Math.floor(totalSec / 3600);
  const m = Math.floor((totalSec % 3600) / 60);
  const s = totalSec % 60;
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  return `${m}:${String(s).padStart(2, '0')}`;
}

function flagColor(flag: string | null, category: string): string {
  if (flag === 'GREEN') return 'text-green-400';
  if (flag === 'YELLOW' || flag === 'DOUBLE YELLOW') return 'text-amber-400';
  if (flag === 'RED') return 'text-red-400';
  if (category === 'SafetyCar') return 'text-amber-400';
  if (category === 'DRS') return 'text-cyan-400';
  return 'text-muted-foreground';
}

function flagDotColor(flag: string | null): string {
  if (flag === 'RED') return '#ef4444';
  if (flag === 'YELLOW' || flag === 'DOUBLE YELLOW') return '#f59e0b';
  return '';
}

function sectorColor(value: number | null, overallBest: number, personalBest: number): string {
  if (!value) return 'text-muted-foreground';
  if (value <= overallBest + 0.001) return 'text-purple-400';
  if (value <= personalBest + 0.001) return 'text-green-400';
  return 'text-foreground';
}

export function LiveDashboard() {
  const [selectedSession, setSelectedSession] = useState<OpenF1Session | null>(null);
  const [replayActive, setReplayActive] = useState(false);
  const [replayTime, setReplayTime] = useState(0);
  const [replaySpeed, setReplaySpeed] = useState(5);
  const [playing, setPlaying] = useState(false);
  const [showDebug, setShowDebug] = useState(false);
  const [selectedDriverNum, setSelectedDriverNum] = useState<number | null>(null);
  const [anomalyVehicles, setAnomalyVehicles] = useState<VehicleData[]>([]);
  const lastFrameRef = useRef(0);
  const scrubRef = useRef<HTMLDivElement>(null);
  const isDraggingRef = useRef(false);
  const prevPositionsRef = useRef(new Map<number, number>());

  // Fetch all sessions
  const { data: sessions, loading: sessionsLoading } = usePolling({
    fetcher: () => openf1.getAllSessions(),
    interval: 60000,
  });

  // Auto-select latest past Race session
  useEffect(() => {
    if (sessions && sessions.length > 0 && !selectedSession) {
      const now = Date.now();
      const pastSessions = sessions.filter(s => new Date(s.date_start).getTime() < now);
      const pastRaces = pastSessions.filter(s => s.session_type === 'Race');
      const pick = pastRaces.length > 0 ? pastRaces[pastRaces.length - 1]
        : pastSessions.length > 0 ? pastSessions[pastSessions.length - 1]
        : sessions[sessions.length - 1];
      setSelectedSession(pick);
    }
  }, [sessions, selectedSession]);

  const sessionKey = selectedSession?.session_key;

  const sessionStartMs = useMemo(() => {
    if (!selectedSession?.date_start) return 0;
    return new Date(selectedSession.date_start).getTime();
  }, [selectedSession]);

  const sessionEndMs = useMemo(() => {
    if (!selectedSession?.date_end) return 0;
    return new Date(selectedSession.date_end).getTime();
  }, [selectedSession]);

  const sessionDuration = sessionEndMs - sessionStartMs;

  useEffect(() => {
    setReplayTime(0);
    setPlaying(false);
  }, [sessionKey]);

  // --- Anomaly data (one-time fetch) ---
  useEffect(() => {
    fetch('/api/pipeline/anomaly')
      .then(r => r.json())
      .then(data => setAnomalyVehicles(parseAnomalyDrivers(data)))
      .catch(() => {});
  }, []);

  // --- Data Fetching ---
  const { data: drivers } = usePolling({
    fetcher: () => sessionKey ? openf1.getDrivers(sessionKey) : Promise.resolve([]),
    interval: 30000, enabled: !!sessionKey,
  });
  const { data: rawPositions } = usePolling({
    fetcher: () => sessionKey ? openf1.getPositions(sessionKey) : Promise.resolve([]),
    interval: 10000, enabled: !!sessionKey,
  });
  const { data: weather } = usePolling({
    fetcher: () => sessionKey ? openf1.getWeather(sessionKey) : Promise.resolve([]),
    interval: 30000, enabled: !!sessionKey,
  });
  const { data: laps } = usePolling({
    fetcher: () => sessionKey ? openf1.getLaps(sessionKey) : Promise.resolve([]),
    interval: 15000, enabled: !!sessionKey,
  });
  const { data: stints } = usePolling({
    fetcher: () => sessionKey ? openf1.getStints(sessionKey) : Promise.resolve([]),
    interval: 30000, enabled: !!sessionKey,
  });
  const { data: pitStops } = usePolling({
    fetcher: () => sessionKey ? openf1.getPitStops(sessionKey) : Promise.resolve([]),
    interval: 30000, enabled: !!sessionKey,
  });
  const { data: raceControl } = usePolling({
    fetcher: () => sessionKey ? openf1.getRaceControl(sessionKey) : Promise.resolve([]),
    interval: 30000, enabled: !!sessionKey,
  });
  const { data: rawIntervals } = usePolling({
    fetcher: () => sessionKey ? openf1.getIntervals(sessionKey) : Promise.resolve([]),
    interval: 60000, enabled: !!sessionKey,
  });

  // Auto-select first McLaren driver when drivers load
  useEffect(() => {
    if (drivers && drivers.length > 0 && selectedDriverNum === null) {
      const mclaren = drivers.find(d => d.team_name?.toLowerCase().includes('mclaren'));
      setSelectedDriverNum(mclaren?.driver_number ?? drivers[0].driver_number);
    }
  }, [drivers, selectedDriverNum]);

  // --- Replay Animation ---
  useEffect(() => {
    if (!playing || !replayActive) return;
    lastFrameRef.current = performance.now();
    let rafId: number;
    const tick = (now: number) => {
      const delta = now - lastFrameRef.current;
      lastFrameRef.current = now;
      setReplayTime(prev => {
        const next = prev + delta * replaySpeed;
        if (next >= sessionDuration) { setPlaying(false); return sessionDuration; }
        return next;
      });
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, [playing, replayActive, replaySpeed, sessionDuration]);

  // --- Replay Filtering ---
  const cutoffMs = sessionStartMs + replayTime;

  const filteredPositions = useMemo(() => {
    if (!rawPositions) return null;
    const cutoff = replayActive ? cutoffMs : Infinity;
    const latest = new Map<number, typeof rawPositions[0]>();
    for (const pos of rawPositions) {
      const t = new Date(pos.date).getTime();
      if (t > cutoff) continue;
      const existing = latest.get(pos.driver_number);
      if (!existing || t > new Date(existing.date).getTime()) {
        latest.set(pos.driver_number, pos);
      }
    }
    return Array.from(latest.values()).sort((a, b) => a.position - b.position);
  }, [rawPositions, replayActive, cutoffMs]);

  // Compute currentMaxLap directly from cutoffMs (light loop, no array creation)
  // This value changes only at lap boundaries, preventing downstream heavy memos from recomputing every frame
  const currentMaxLap = useMemo(() => {
    if (!laps || laps.length === 0) return 0;
    if (!replayActive) return Math.max(...laps.map(l => l.lap_number));
    let maxLap = 0;
    for (const l of laps) {
      if (new Date(l.date_start).getTime() <= cutoffMs && l.lap_number > maxLap) {
        maxLap = l.lap_number;
      }
    }
    return maxLap;
  }, [laps, replayActive, cutoffMs]);

  // Filter laps by lap boundary (not timestamp) — only recomputes when currentMaxLap changes
  const filteredLaps = useMemo(() => {
    if (!replayActive || !laps) return laps;
    return laps.filter(l => l.lap_number <= currentMaxLap);
  }, [laps, replayActive, currentMaxLap]);

  const filteredWeather = useMemo(() => {
    if (!replayActive || !weather) return weather;
    return weather.filter(w => new Date(w.date).getTime() <= cutoffMs);
  }, [weather, replayActive, cutoffMs]);

  const filteredPitStops = useMemo(() => {
    if (!replayActive || !pitStops) return pitStops;
    return pitStops.filter(p => p.lap_number <= currentMaxLap);
  }, [pitStops, replayActive, currentMaxLap]);

  const filteredRaceControl = useMemo(() => {
    if (!raceControl) return [];
    if (!replayActive) return raceControl;
    return raceControl.filter(rc => new Date(rc.date).getTime() <= cutoffMs);
  }, [raceControl, replayActive, cutoffMs]);

  const totalLaps = useMemo(() => {
    if (!laps || laps.length === 0) return 0;
    return Math.max(...laps.map(l => l.lap_number));
  }, [laps]);

  const filteredStints = useMemo(() => {
    if (!replayActive || !stints) return stints;
    return stints.filter(s => s.lap_start <= currentMaxLap);
  }, [stints, replayActive, currentMaxLap]);

  // --- Derived Data ---
  const activePositions = filteredPositions;
  const activeLaps = filteredLaps;
  const activeWeather = filteredWeather;
  const activePitStops = filteredPitStops;
  const activeStints = filteredStints;
  const latestWeather = activeWeather && activeWeather.length > 0 ? activeWeather[activeWeather.length - 1] : null;

  const driverMap = useMemo(() => {
    const map = new Map<number, OpenF1Driver>();
    if (drivers) drivers.forEach(d => map.set(d.driver_number, d));
    return map;
  }, [drivers]);

  // Car track positions — estimate each driver's fractional lap progress for the track map
  // Uses lap date_start + sector durations to interpolate position at cutoffMs
  const carTrackPositions = useMemo(() => {
    if (!replayActive || !laps || laps.length === 0 || !activePositions || activePositions.length === 0) return [];

    const cars: import('./TrackMap').CarPosition[] = [];

    for (const pos of activePositions) {
      const driver = driverMap.get(pos.driver_number);
      if (!driver) continue;

      // Find the latest lap for this driver that started before cutoff
      let currentLap: typeof laps[0] | null = null;
      for (const l of laps) {
        if (l.driver_number !== pos.driver_number) continue;
        const lapStart = new Date(l.date_start).getTime();
        if (lapStart <= cutoffMs) {
          if (!currentLap || l.lap_number > currentLap.lap_number) {
            currentLap = l;
          }
        }
      }

      if (!currentLap) continue;

      const lapStartMs = new Date(currentLap.date_start).getTime();
      const elapsed = (cutoffMs - lapStartMs) / 1000; // seconds into this lap
      const lapDuration = currentLap.lap_duration || 90; // fallback ~90s

      // Fractional progress through the lap (0 → 1)
      const fraction = Math.min(Math.max(elapsed / lapDuration, 0), 1);

      cars.push({
        driverNumber: pos.driver_number,
        code: driver.name_acronym,
        x: fraction,
        y: 0,
        color: driver.team_colour ? `#${driver.team_colour}` : '#ffffff',
      });
    }

    return cars;
  }, [replayActive, laps, activePositions, driverMap, cutoffMs]);

  // Position deltas — computed from ref, ref updated in useEffect (not during render)
  const positionDeltas = useMemo(() => {
    const deltas = new Map<number, number>();
    if (activePositions) {
      for (const pos of activePositions) {
        const prev = prevPositionsRef.current.get(pos.driver_number);
        if (prev !== undefined) {
          deltas.set(pos.driver_number, prev - pos.position); // positive = improved
        }
      }
    }
    return deltas;
  }, [activePositions]);

  // Update previous positions ref AFTER render (useEffect, not useMemo)
  useEffect(() => {
    if (activePositions) {
      const newMap = new Map<number, number>();
      for (const pos of activePositions) newMap.set(pos.driver_number, pos.position);
      prevPositionsRef.current = newMap;
    }
  }, [activePositions]);

  // Fastest lap
  const fastestLapInfo = useMemo(() => {
    if (!activeLaps) return null;
    let best: { driver_number: number; lap_number: number; lap_duration: number } | null = null;
    for (const lap of activeLaps) {
      if (lap.lap_duration && !lap.is_pit_out_lap) {
        if (!best || lap.lap_duration < best.lap_duration) {
          best = { driver_number: lap.driver_number, lap_number: lap.lap_number, lap_duration: lap.lap_duration };
        }
      }
    }
    return best;
  }, [activeLaps]);

  // Sector bests
  const sectorBests = useMemo(() => {
    let overallS1 = Infinity, overallS2 = Infinity, overallS3 = Infinity;
    const personal = new Map<number, { s1: number; s2: number; s3: number }>();
    if (activeLaps) {
      for (const lap of activeLaps) {
        const p = personal.get(lap.driver_number) ?? { s1: Infinity, s2: Infinity, s3: Infinity };
        if (lap.duration_sector_1) {
          if (lap.duration_sector_1 < overallS1) overallS1 = lap.duration_sector_1;
          if (lap.duration_sector_1 < p.s1) p.s1 = lap.duration_sector_1;
        }
        if (lap.duration_sector_2) {
          if (lap.duration_sector_2 < overallS2) overallS2 = lap.duration_sector_2;
          if (lap.duration_sector_2 < p.s2) p.s2 = lap.duration_sector_2;
        }
        if (lap.duration_sector_3) {
          if (lap.duration_sector_3 < overallS3) overallS3 = lap.duration_sector_3;
          if (lap.duration_sector_3 < p.s3) p.s3 = lap.duration_sector_3;
        }
        personal.set(lap.driver_number, p);
      }
    }
    return { overallS1, overallS2, overallS3, personal };
  }, [activeLaps]);

  // Index laps by driver+lap for O(1) lookup (avoids O(n) find per driver)
  const lapIndex = useMemo(() => {
    const idx = new Map<string, OpenF1Lap>();
    if (activeLaps) {
      for (const l of activeLaps) {
        idx.set(`${l.driver_number}-${l.lap_number}`, l);
      }
    }
    return idx;
  }, [activeLaps]);

  // Sector times for current lap
  const sectorTimesData = useMemo(() => {
    if (!activeLaps || !activePositions) return [];
    const targetLap = currentMaxLap > 0 ? currentMaxLap : 0;
    return activePositions.map(pos => {
      const lap = lapIndex.get(`${pos.driver_number}-${targetLap}`);
      const driver = driverMap.get(pos.driver_number);
      return {
        driver_number: pos.driver_number,
        acronym: driver?.name_acronym ?? `D${pos.driver_number}`,
        team_colour: driver?.team_colour ?? '555555',
        s1: lap?.duration_sector_1 ?? null,
        s2: lap?.duration_sector_2 ?? null,
        s3: lap?.duration_sector_3 ?? null,
        total: lap?.lap_duration ?? null,
      };
    });
  }, [activeLaps, activePositions, currentMaxLap, driverMap, lapIndex]);

  // Gap chart data — downsample intervals to one per lap per driver
  // Keyed on currentMaxLap (not per-frame cutoffMs) to avoid filtering 168K records at 60fps
  const gapChartData = useMemo(() => {
    if (!rawIntervals || !activeLaps || !activePositions) return [];
    const topDrivers = activePositions.slice(0, 5).map(p => p.driver_number);
    const topDriverSet = new Set(topDrivers);

    // Build lap timestamps (only for laps up to currentMaxLap)
    const lapTimes = new Map<number, number>();
    for (const lap of activeLaps) {
      if (!lapTimes.has(lap.lap_number)) {
        lapTimes.set(lap.lap_number, new Date(lap.date_start).getTime());
      }
    }

    // Find cutoff timestamp for filtering intervals
    let maxLapTs = 0;
    for (const [, ts] of lapTimes) {
      if (ts > maxLapTs) maxLapTs = ts;
    }

    // Group relevant intervals by driver (single pass, filtered by time + driver)
    const byDriver = new Map<number, typeof rawIntervals>();
    for (const iv of rawIntervals) {
      if (!topDriverSet.has(iv.driver_number)) continue;
      if (replayActive && new Date(iv.date).getTime() > maxLapTs + 120000) continue; // generous buffer
      if (!byDriver.has(iv.driver_number)) byDriver.set(iv.driver_number, []);
      byDriver.get(iv.driver_number)!.push(iv);
    }

    const sortedLaps = Array.from(lapTimes.entries()).sort((a, b) => a[0] - b[0]);
    const data: any[] = [];

    for (const [lapNum, lapTs] of sortedLaps) {
      const point: any = { lap: lapNum };
      for (const dNum of topDrivers) {
        const driver = driverMap.get(dNum);
        const key = driver?.name_acronym ?? `D${dNum}`;
        const driverIvs = byDriver.get(dNum);
        if (!driverIvs || driverIvs.length === 0) { point[key] = null; continue; }
        // Find closest interval to lap timestamp
        let closest = driverIvs[0];
        let minDist = Math.abs(new Date(closest.date).getTime() - lapTs);
        for (let i = 1; i < driverIvs.length; i++) {
          const dist = Math.abs(new Date(driverIvs[i].date).getTime() - lapTs);
          if (dist < minDist) { minDist = dist; closest = driverIvs[i]; }
        }
        point[key] = closest.gap_to_leader ?? null;
      }
      data.push(point);
    }
    return data;
  }, [rawIntervals, activeLaps, activePositions, driverMap, replayActive, currentMaxLap]);

  // Lap chart data
  const lapChartData = useMemo(() => {
    if (!activeLaps || !activePositions) return [];
    const topDrivers = activePositions.slice(0, 5).map(p => p.driver_number);
    const driverLaps = new Map<number, OpenF1Lap[]>();
    activeLaps.forEach(l => {
      if (topDrivers.includes(l.driver_number) && l.lap_duration) {
        if (!driverLaps.has(l.driver_number)) driverLaps.set(l.driver_number, []);
        driverLaps.get(l.driver_number)!.push(l);
      }
    });
    const maxLaps = Math.max(...Array.from(driverLaps.values()).map(ls => ls.length), 0);
    const data: any[] = [];
    for (let i = 0; i < maxLaps; i++) {
      const point: any = { lap: i + 1 };
      topDrivers.forEach(dNum => {
        const dl = driverLaps.get(dNum);
        const driver = driverMap.get(dNum);
        const key = driver?.name_acronym || `D${dNum}`;
        point[key] = dl?.[i]?.lap_duration ?? null;
      });
      data.push(point);
    }
    return data;
  }, [activeLaps, activePositions, driverMap]);

  // Stint summary — all stints per driver for timeline
  const stintsByDriver = useMemo(() => {
    const map = new Map<number, OpenF1Stint[]>();
    activeStints?.forEach(s => {
      if (!map.has(s.driver_number)) map.set(s.driver_number, []);
      map.get(s.driver_number)!.push(s);
    });
    return map;
  }, [activeStints]);

  // Latest race control events (last 3 for ticker)
  const latestEvents = useMemo(() => filteredRaceControl.slice(-3).reverse(), [filteredRaceControl]);

  // Head-to-head comparison (top 2 drivers)
  const headToHead = useMemo(() => {
    if (!activePositions || activePositions.length < 2 || !activeLaps) return null;
    const d1 = activePositions[0];
    const d2 = activePositions[1];
    const drv1 = driverMap.get(d1.driver_number);
    const drv2 = driverMap.get(d2.driver_number);
    if (!drv1 || !drv2) return null;

    let bestLap1 = Infinity, bestLap2 = Infinity;
    let bestS1_1 = Infinity, bestS2_1 = Infinity, bestS3_1 = Infinity;
    let bestS1_2 = Infinity, bestS2_2 = Infinity, bestS3_2 = Infinity;
    let topSpeed1 = 0, topSpeed2 = 0;

    for (const l of activeLaps) {
      if (l.driver_number === d1.driver_number) {
        if (l.lap_duration && l.lap_duration < bestLap1) bestLap1 = l.lap_duration;
        if (l.duration_sector_1 && l.duration_sector_1 < bestS1_1) bestS1_1 = l.duration_sector_1;
        if (l.duration_sector_2 && l.duration_sector_2 < bestS2_1) bestS2_1 = l.duration_sector_2;
        if (l.duration_sector_3 && l.duration_sector_3 < bestS3_1) bestS3_1 = l.duration_sector_3;
        if (l.st_speed && l.st_speed > topSpeed1) topSpeed1 = l.st_speed;
      } else if (l.driver_number === d2.driver_number) {
        if (l.lap_duration && l.lap_duration < bestLap2) bestLap2 = l.lap_duration;
        if (l.duration_sector_1 && l.duration_sector_1 < bestS1_2) bestS1_2 = l.duration_sector_1;
        if (l.duration_sector_2 && l.duration_sector_2 < bestS2_2) bestS2_2 = l.duration_sector_2;
        if (l.duration_sector_3 && l.duration_sector_3 < bestS3_2) bestS3_2 = l.duration_sector_3;
        if (l.st_speed && l.st_speed > topSpeed2) topSpeed2 = l.st_speed;
      }
    }

    const pits1 = (activePitStops ?? []).filter(p => p.driver_number === d1.driver_number).length;
    const pits2 = (activePitStops ?? []).filter(p => p.driver_number === d2.driver_number).length;

    return {
      d1: { code: drv1.name_acronym, color: `#${drv1.team_colour}`, pos: d1.position,
             bestLap: bestLap1 === Infinity ? null : bestLap1,
             bestS1: bestS1_1 === Infinity ? null : bestS1_1,
             bestS2: bestS2_1 === Infinity ? null : bestS2_1,
             bestS3: bestS3_1 === Infinity ? null : bestS3_1,
             topSpeed: topSpeed1 || null, pits: pits1 },
      d2: { code: drv2.name_acronym, color: `#${drv2.team_colour}`, pos: d2.position,
             bestLap: bestLap2 === Infinity ? null : bestLap2,
             bestS1: bestS1_2 === Infinity ? null : bestS1_2,
             bestS2: bestS2_2 === Infinity ? null : bestS2_2,
             bestS3: bestS3_2 === Infinity ? null : bestS3_2,
             topSpeed: topSpeed2 || null, pits: pits2 },
    };
  }, [activePositions, activeLaps, driverMap, activePitStops]);

  // Scrub bar markers
  const flagMarkers = useMemo(() => {
    if (!raceControl || sessionDuration <= 0) return [];
    return raceControl
      .filter(rc => rc.flag === 'RED' || rc.flag === 'YELLOW' || rc.flag === 'DOUBLE YELLOW')
      .map(rc => ({
        pct: (new Date(rc.date).getTime() - sessionStartMs) / sessionDuration,
        color: flagDotColor(rc.flag),
      }))
      .filter(m => m.pct >= 0 && m.pct <= 1);
  }, [raceControl, sessionStartMs, sessionDuration]);

  const pitMarkers = useMemo(() => {
    if (!pitStops || sessionDuration <= 0) return [];
    return pitStops
      .map(p => ({
        pct: (new Date(p.date).getTime() - sessionStartMs) / sessionDuration,
      }))
      .filter(m => m.pct >= 0 && m.pct <= 1);
  }, [pitStops, sessionStartMs, sessionDuration]);

  // --- Scrub Handlers ---
  const handleScrub = useCallback((e: React.PointerEvent | PointerEvent) => {
    if (!scrubRef.current || sessionDuration <= 0) return;
    const rect = scrubRef.current.getBoundingClientRect();
    const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    setReplayTime(pct * sessionDuration);
  }, [sessionDuration]);

  const scrubCleanupRef = useRef<(() => void) | null>(null);

  const onScrubDown = useCallback((e: React.PointerEvent) => {
    isDraggingRef.current = true;
    handleScrub(e);
    const onMove = (ev: PointerEvent) => handleScrub(ev);
    const onUp = () => {
      isDraggingRef.current = false;
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      scrubCleanupRef.current = null;
    };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    scrubCleanupRef.current = () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
    };
  }, [handleScrub]);

  // Cleanup scrub listeners on unmount
  useEffect(() => {
    return () => { scrubCleanupRef.current?.(); };
  }, []);

  // Toggle replay
  const toggleReplay = useCallback(() => {
    if (replayActive) {
      setReplayActive(false);
      setPlaying(false);
      setReplayTime(0);
    } else {
      setReplayActive(true);
      setReplayTime(0);
      setPlaying(false);
    }
  }, [replayActive]);

  // --- Selected driver anomaly data ---
  const selectedDriver = useMemo(() => drivers?.find(d => d.driver_number === selectedDriverNum) ?? null, [drivers, selectedDriverNum]);
  const selectedVehicle = useMemo(() => {
    if (!selectedDriver || anomalyVehicles.length === 0) return null;
    return anomalyVehicles.find(v => v.code === selectedDriver.name_acronym) ?? null;
  }, [selectedDriver, anomalyVehicles]);

  // Race-by-race trend for the selected driver's systems
  const healthTrendData = useMemo(() => {
    if (!selectedVehicle) return [];
    return selectedVehicle.races.map(r => {
      const row: Record<string, any> = { race: r.race };
      for (const [sysName, sys] of Object.entries(r.systems)) {
        row[sysName] = sys.health;
      }
      return row;
    });
  }, [selectedVehicle]);

  if (sessionsLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-6 h-6 text-[#FF8000] animate-spin" />
        <span className="ml-3 text-muted-foreground text-sm">Loading sessions from OpenF1...</span>
      </div>
    );
  }

  if (!selectedSession) {
    return (
      <div className="flex items-center justify-center h-64">
        <AlertCircle className="w-6 h-6 text-amber-400" />
        <span className="ml-3 text-muted-foreground text-sm">No sessions available</span>
      </div>
    );
  }

  const topDriverColors = ['#FF8000', '#00d4ff', '#22c55e', '#f59e0b', '#ef4444'];
  const scrubPct = sessionDuration > 0 ? (replayTime / sessionDuration) * 100 : 0;

  return (
    <div className="space-y-4">
      {/* Session Info Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-6">
          <div>
            <div className="text-[12px] text-muted-foreground tracking-widest">{selectedSession.session_type}</div>
            <div className="text-[#FF8000] text-sm">{selectedSession.circuit_short_name} — {selectedSession.session_name}</div>
          </div>
          <div className="flex items-center gap-4 text-sm">
            <div className="bg-[#222838] rounded-lg px-3 py-1.5 font-mono flex items-center gap-1.5">
              <MapPin className="w-3 h-3 text-[#FF8000]" />
              <span className="text-foreground">{selectedSession.circuit_short_name}</span>
            </div>
            <div className="bg-[#222838] rounded-lg px-3 py-1.5 font-mono flex items-center gap-1.5">
              <Flag className="w-3 h-3 text-muted-foreground" />
              <span className="text-foreground">{selectedSession.country_name}</span>
            </div>
            <select
              value={selectedSession.session_key}
              onChange={(e) => {
                const s = sessions?.find(s => s.session_key === Number(e.target.value));
                if (s) setSelectedSession(s);
              }}
              aria-label="Select session"
              className="appearance-none bg-[#222838] border border-[rgba(255,128,0,0.12)] rounded-lg px-3 py-1.5 text-sm text-foreground focus:outline-none cursor-pointer"
            >
              {sessions
                ?.filter(s => new Date(s.date_start).getTime() < Date.now())
                .slice(-30)
                .reverse()
                .filter((s, i, arr) => arr.findIndex(x => x.session_key === s.session_key) === i)
                .map(s => (
                <option key={s.session_key} value={s.session_key}>
                  {s.circuit_short_name} — {s.session_name} ({s.date_start?.slice(0, 10)})
                </option>
              ))}
            </select>
            {drivers && drivers.length > 0 && (
              <select
                value={selectedDriverNum ?? ''}
                onChange={(e) => setSelectedDriverNum(Number(e.target.value))}
                aria-label="Select driver"
                className="appearance-none bg-[#222838] border border-[rgba(255,128,0,0.12)] rounded-lg px-3 py-1.5 text-sm text-foreground focus:outline-none cursor-pointer"
              >
                {drivers.map(d => (
                  <option key={d.driver_number} value={d.driver_number}>
                    {d.name_acronym} — {d.full_name}
                  </option>
                ))}
              </select>
            )}
          </div>
        </div>
        {latestWeather && (
          <div className="flex items-center gap-3 text-[12px] text-muted-foreground">
            <div className="flex items-center gap-1">
              <Cloud className="w-3 h-3" />
              Track {latestWeather.track_temperature}°C
            </div>
            <div>Air {latestWeather.air_temperature}°C</div>
            <div className="flex items-center gap-1">
              <Wind className="w-3 h-3" />
              {latestWeather.wind_speed} km/h
            </div>
            <div>{latestWeather.rainfall ? 'Rain' : 'Dry'}</div>
          </div>
        )}
      </div>

      {/* Replay Control Bar */}
      <div className="bg-[#222838] border border-[rgba(255,128,0,0.20)] border-t-[#FF8000]/40 rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-3">
        <div className="flex items-center gap-4">
          <button
            onClick={toggleReplay}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-mono transition-colors ${
              replayActive
                ? 'bg-[#FF8000]/15 text-[#FF8000] border border-[#FF8000]/30'
                : 'bg-green-500/10 text-green-400 border border-green-500/20 hover:bg-green-500/20'
            }`}
          >
            {replayActive ? <RotateCcw className="w-3 h-3" /> : <Radio className="w-3 h-3" />}
            {replayActive ? 'REPLAY' : 'LIVE'}
          </button>

          {replayActive && (
            <>
              <button
                onClick={() => setPlaying(!playing)}
                className="w-8 h-8 rounded-lg bg-[#FF8000]/10 flex items-center justify-center hover:bg-[#FF8000]/20 transition-colors"
              >
                {playing ? <Pause className="w-4 h-4 text-[#FF8000]" /> : <Play className="w-4 h-4 text-[#FF8000]" />}
              </button>

              <div className="flex items-center gap-1">
                {SPEED_OPTIONS.map(sp => (
                  <button
                    key={sp}
                    onClick={() => setReplaySpeed(sp)}
                    className={`px-2 py-1 rounded text-[12px] font-mono transition-colors ${
                      replaySpeed === sp
                        ? 'bg-[#FF8000] text-[#0D1117]'
                        : 'text-muted-foreground hover:text-foreground hover:bg-[#222838]'
                    }`}
                  >
                    {sp}x
                  </button>
                ))}
              </div>

              {/* Animated Scrub Bar */}
              <div className="flex-1 flex items-center gap-3">
                <div
                  ref={scrubRef}
                  className="flex-1 h-3 bg-[#222838] rounded-full relative cursor-pointer select-none"
                  onPointerDown={onScrubDown}
                >
                  {/* Filled track */}
                  <motion.div
                    className="absolute top-0 left-0 h-full rounded-full bg-gradient-to-r from-[#FF8000] to-[#FF8000]/70"
                    style={{ width: `${scrubPct}%` }}
                  />
                  {/* Flag markers */}
                  {flagMarkers.map((m, i) => (
                    <div
                      key={`flag-${i}`}
                      className="absolute top-0 w-1 h-full rounded-full"
                      style={{ left: `${m.pct * 100}%`, backgroundColor: m.color, opacity: 0.7 }}
                    />
                  ))}
                  {/* Pit markers */}
                  {pitMarkers.map((m, i) => (
                    <div
                      key={`pit-${i}`}
                      className="absolute top-0 w-px h-full"
                      style={{ left: `${m.pct * 100}%`, backgroundColor: '#fff', opacity: 0.3 }}
                    />
                  ))}
                  {/* Thumb */}
                  <motion.div
                    className="absolute top-1/2 -translate-y-1/2 w-4 h-4 rounded-full bg-[#FF8000] shadow-[0_0_8px_rgba(255,128,0,0.5)] border-2 border-[#0D1117]"
                    style={{ left: `calc(${scrubPct}% - 8px)` }}
                    transition={{ type: 'spring', stiffness: 400, damping: 30 }}
                  />
                </div>
              </div>

              <div className="text-right shrink-0">
                <div className="text-sm font-mono text-foreground">
                  {formatElapsed(replayTime)} / {formatElapsed(sessionDuration)}
                </div>
                <div className="text-[12px] text-muted-foreground">
                  L{currentMaxLap}{totalLaps > 0 ? ` / L${totalLaps}` : ''}
                </div>
              </div>
            </>
          )}

          {!replayActive && (
            <div className="flex-1 text-[12px] text-muted-foreground">
              Switch to <span className="text-[#FF8000]">REPLAY</span> to stream historical race data
            </div>
          )}

          <button
            onClick={() => setShowDebug(d => !d)}
            className={`p-1.5 rounded-lg transition-colors ${showDebug ? 'bg-amber-500/20 text-amber-400' : 'text-muted-foreground hover:text-foreground hover:bg-[#222838]'}`}
            title="Toggle debug info"
          >
            <Bug className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Race control ticker */}
        {replayActive && latestEvents.length > 0 && (
          <div className="flex items-center gap-4 mt-2 pt-2 border-t border-[rgba(255,128,0,0.12)]">
            {latestEvents.map((ev, i) => (
              <div key={`${ev.date}-${i}`} className={`flex items-center gap-1.5 text-[12px] ${flagColor(ev.flag, ev.category)}`}>
                <span className="w-1.5 h-1.5 rounded-full bg-current shrink-0" />
                <span className="truncate max-w-[300px]">{ev.message}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Debug Panel */}
      {showDebug && (
        <div className="bg-[#0D1117] border border-amber-500/30 rounded-xl p-3 text-[12px] font-mono space-y-1">
          <div className="text-amber-400 font-semibold mb-1">DEBUG — Data Pipeline</div>
          <div className="text-muted-foreground">
            Session: <span className="text-foreground">{sessionKey}</span> |
            Start: <span className="text-foreground">{sessionStartMs ? new Date(sessionStartMs).toISOString() : 'N/A'}</span> |
            End: <span className="text-foreground">{sessionEndMs ? new Date(sessionEndMs).toISOString() : 'N/A'}</span> |
            Duration: <span className="text-foreground">{formatElapsed(sessionDuration)}</span>
          </div>
          {replayActive && (
            <div className="text-muted-foreground">
              Cutoff: <span className="text-cyan-400">{new Date(cutoffMs).toISOString()}</span> |
              ReplayTime: <span className="text-cyan-400">{formatElapsed(replayTime)}</span>
            </div>
          )}
          <div className="text-amber-400">
            RAW: positions={rawPositions?.length ?? 'null'} | laps={laps?.length ?? 'null'} | drivers={drivers?.length ?? 'null'} | weather={weather?.length ?? 'null'} | intervals={rawIntervals?.length ?? 'null'} | pits={pitStops?.length ?? 'null'} | raceCtrl={raceControl?.length ?? 'null'}
          </div>
          <div className="text-green-400">
            FILTERED: positions={filteredPositions?.length ?? 'null'} | laps={filteredLaps?.length ?? 'null'} | weather={filteredWeather?.length ?? 'null'} | pits={filteredPitStops?.length ?? 'null'} | maxLap={currentMaxLap}
          </div>
        </div>
      )}

      {/* Circuit Track Map + Live Analytics Sidebar */}
      {(() => {
        const circuit = selectedSession ? getCircuitByShortName(selectedSession.circuit_short_name) : undefined;
        if (!circuit) return (
          <div className="bg-[#1A1F2E] rounded-xl border border-[rgba(255,128,0,0.12)] p-6 text-center">
            <div className="text-amber-400 text-sm">No track map for circuit: "{selectedSession?.circuit_short_name}"</div>
          </div>
        );
        return (
          <div className="grid grid-cols-12 gap-4">
            {/* Left: MapLibre Track Map */}
            <div className="col-span-7">
              <TrackMapGL
                geojsonPath={circuit.geojsonPath}
                circuitName={circuit.circuitName}
                locality={circuit.locality}
                country={circuit.country}
                lengthKm={circuit.lengthKm}
                lat={circuit.lat}
                lng={circuit.lng}
                height={550}
                turns={circuit.turns}
                drsZones={circuit.drsZones}
                cars={replayActive && carTrackPositions.length > 0 ? carTrackPositions : undefined}
              />
            </div>
            {/* Right: Analytics Sidebar + Head-to-Head */}
            <div className="col-span-5 flex flex-col gap-4">
              <LiveAnalyticsSidebar
                positions={activePositions}
                sectorTimesData={sectorTimesData}
                sectorBests={sectorBests}
                stints={activeStints}
                raceControlEvents={filteredRaceControl ?? []}
                gapChartData={gapChartData}
                fastestLap={fastestLapInfo}
                driverMap={driverMap}
                positionDeltas={positionDeltas}
                currentMaxLap={currentMaxLap}
                replayActive={replayActive}
                replayTime={replayTime}
                height={headToHead ? 360 : 550}
              />
              {headToHead && (
                <div className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.20)] border-t-2 border-t-[#FF8000] rounded-xl shadow-[0_4px_12px_rgba(0,0,0,0.4)] p-3">
                  <h3 className="text-[11px] text-[#FF8000] tracking-widest mb-2 font-semibold">HEAD TO HEAD</h3>
                  <div className="grid grid-cols-[1fr_auto_1fr] gap-3 items-center">
                    <div className="text-right">
                      <div className="text-base font-bold font-mono" style={{ color: headToHead.d1.color }}>{headToHead.d1.code}</div>
                      <div className="text-[11px] text-muted-foreground">P{headToHead.d1.pos}</div>
                    </div>
                    <div className="text-[11px] text-muted-foreground text-center">VS</div>
                    <div className="text-left">
                      <div className="text-base font-bold font-mono" style={{ color: headToHead.d2.color }}>{headToHead.d2.code}</div>
                      <div className="text-[11px] text-muted-foreground">P{headToHead.d2.pos}</div>
                    </div>
                  </div>
                  <div className="mt-2 space-y-0.5">
                    {[
                      { label: 'BEST LAP', v1: headToHead.d1.bestLap, v2: headToHead.d2.bestLap, fmt: (v: number | null) => v ? `${v.toFixed(3)}s` : '—', lower: true },
                      { label: 'S1', v1: headToHead.d1.bestS1, v2: headToHead.d2.bestS1, fmt: (v: number | null) => v ? `${v.toFixed(3)}` : '—', lower: true },
                      { label: 'S2', v1: headToHead.d1.bestS2, v2: headToHead.d2.bestS2, fmt: (v: number | null) => v ? `${v.toFixed(3)}` : '—', lower: true },
                      { label: 'S3', v1: headToHead.d1.bestS3, v2: headToHead.d2.bestS3, fmt: (v: number | null) => v ? `${v.toFixed(3)}` : '—', lower: true },
                      { label: 'TOP SPD', v1: headToHead.d1.topSpeed, v2: headToHead.d2.topSpeed, fmt: (v: number | null) => v ? `${v}` : '—', lower: false },
                      { label: 'PITS', v1: headToHead.d1.pits, v2: headToHead.d2.pits, fmt: (v: number | null) => v !== null ? `${v}` : '—', lower: null },
                    ].map(({ label, v1, v2, fmt, lower }) => {
                      const win1 = v1 !== null && v2 !== null && lower !== null ? (lower ? v1 < v2 : v1 > v2) : false;
                      const win2 = v1 !== null && v2 !== null && lower !== null ? (lower ? v2 < v1 : v2 > v1) : false;
                      return (
                        <div key={label} className="grid grid-cols-[1fr_auto_1fr] gap-3 items-center">
                          <div className={`text-right text-[11px] font-mono ${win1 ? 'text-green-400' : 'text-foreground'}`}>{fmt(v1)}</div>
                          <div className="text-[10px] text-muted-foreground text-center w-14">{label}</div>
                          <div className={`text-left text-[11px] font-mono ${win2 ? 'text-green-400' : 'text-foreground'}`}>{fmt(v2)}</div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          </div>
        );
      })()}

      {/* ── RACE OVERVIEW ── */}
      <div className="flex items-center gap-2 mt-2">
        <div className="h-px flex-1 bg-[rgba(255,128,0,0.10)]" />
        <span className="text-[10px] tracking-[0.25em] text-[#FF8000]/60 font-semibold">RACE OVERVIEW</span>
        <div className="h-px flex-1 bg-[rgba(255,128,0,0.10)]" />
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-4 gap-3">
        <AnimatedKPI
          icon={<Gauge className="w-4 h-4 text-[#FF8000]" />}
          label="Drivers"
          value={`${activePositions?.length ?? '—'}`}
          sub="In session"
          color="text-[#FF8000]"
        />
        <AnimatedKPI
          icon={<Timer className="w-4 h-4 text-cyan-400" />}
          label="Total Laps"
          value={`${currentMaxLap || '—'}`}
          sub={replayActive ? 'Laps so far' : 'Laps completed'}
          color="text-cyan-400"
        />
        <AnimatedKPI
          icon={<Zap className="w-4 h-4 text-green-400" />}
          label="Pit Stops"
          value={`${activePitStops?.length ?? '—'}`}
          sub="Total this session"
          color="text-green-400"
        />
        <AnimatedKPI
          icon={<Trophy className="w-4 h-4 text-purple-400" />}
          label="Fastest Lap"
          value={fastestLapInfo ? `${fastestLapInfo.lap_duration.toFixed(3)}s` : '---'}
          sub={fastestLapInfo ? `${driverMap.get(fastestLapInfo.driver_number)?.name_acronym ?? '?'} — L${fastestLapInfo.lap_number}` : 'No data yet'}
          color="text-purple-400"
        />
      </div>

      {/* ── DRIVER HEALTH ── */}
      {selectedVehicle && (
        <>
          <div className="flex items-center gap-2 mt-2">
            <div className="h-px flex-1 bg-[rgba(255,128,0,0.10)]" />
            <span className="text-[10px] tracking-[0.25em] text-[#FF8000]/60 font-semibold">DRIVER HEALTH — {selectedDriver?.name_acronym}</span>
            <div className="h-px flex-1 bg-[rgba(255,128,0,0.10)]" />
          </div>

          <div className="grid grid-cols-12 gap-4">
            {/* Overall Health + KPIs */}
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              className="col-span-3 bg-[#1A1F2E] border border-[rgba(255,128,0,0.20)] rounded-xl shadow-[0_4px_12px_rgba(0,0,0,0.4)] p-4 flex flex-col items-center justify-center gap-3"
            >
              <HealthGauge value={selectedVehicle.overallHealth} size={100} label="Overall" />
              <StatusBadge status={selectedVehicle.level} />
              <div className="w-full space-y-2 mt-2">
                <div className="flex items-center justify-between text-[11px]">
                  <span className="text-muted-foreground">DNF Risk</span>
                  <span className="font-mono" style={{ color: selectedVehicle.overallHealth >= 80 ? '#22c55e' : selectedVehicle.overallHealth >= 60 ? '#FF8000' : '#ef4444' }}>
                    {Math.max(0, 100 - selectedVehicle.overallHealth).toFixed(0)}%
                  </span>
                </div>
                <div className="flex items-center justify-between text-[11px]">
                  <span className="text-muted-foreground">Systems</span>
                  <span className="font-mono text-foreground">{selectedVehicle.systems.length}</span>
                </div>
                <div className="flex items-center justify-between text-[11px]">
                  <span className="text-muted-foreground">Last Race</span>
                  <span className="font-mono text-foreground text-[10px]">{selectedVehicle.lastRace}</span>
                </div>
              </div>
            </motion.div>

            {/* System Health Cards */}
            <div className="col-span-9 grid grid-cols-3 gap-3">
              {selectedVehicle.systems.map((sys) => {
                const Icon = sys.icon;
                const maint = sys.maintenanceAction ? MAINTENANCE_LABELS[sys.maintenanceAction] ?? MAINTENANCE_LABELS.none : null;
                return (
                  <motion.div
                    key={sys.name}
                    initial={{ opacity: 0, y: 12 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.20)] rounded-xl shadow-[0_4px_12px_rgba(0,0,0,0.4)] p-4"
                  >
                    <div className="flex items-center gap-2 mb-3">
                      <div className="w-7 h-7 rounded-lg flex items-center justify-center" style={{ backgroundColor: levelBg(sys.level) }}>
                        <Icon className="w-4 h-4" style={{ color: levelColor(sys.level) }} />
                      </div>
                      <div className="flex-1">
                        <div className="text-[11px] text-muted-foreground">{sys.name}</div>
                        <div className="text-sm font-mono" style={{ color: levelColor(sys.level) }}>{sys.health.toFixed(0)}%</div>
                      </div>
                      <StatusBadge status={sys.level} size="sm" />
                    </div>

                    {/* Health bar */}
                    <div className="w-full h-1.5 bg-[#222838] rounded-full overflow-hidden mb-3">
                      <div
                        className="h-full rounded-full transition-all duration-500"
                        style={{ width: `${sys.health}%`, backgroundColor: levelColor(sys.level) }}
                      />
                    </div>

                    {/* Severity probability bars */}
                    {sys.severityProbabilities && (
                      <div className="space-y-1 mb-3">
                        <div className="text-[9px] text-muted-foreground tracking-wider">SEVERITY DISTRIBUTION</div>
                        <div className="flex h-2 rounded-full overflow-hidden">
                          {Object.entries(sys.severityProbabilities)
                            .sort(([,a], [,b]) => b - a)
                            .map(([level, prob]) => (
                              <div
                                key={level}
                                style={{ width: `${prob * 100}%`, backgroundColor: SEVERITY_COLORS[level] ?? '#6b7280' }}
                                title={`${level}: ${(prob * 100).toFixed(0)}%`}
                              />
                            ))}
                        </div>
                      </div>
                    )}

                    {/* Model consensus */}
                    <div className="flex items-center justify-between text-[10px] text-muted-foreground mb-2">
                      <span>Models: {sys.voteCount}/{sys.totalModels}</span>
                    </div>

                    {/* Maintenance badge */}
                    {maint && (
                      <div className="flex items-center gap-1.5 text-[10px] px-2 py-1 rounded-md" style={{ backgroundColor: maint.color + '15', color: maint.color }}>
                        <maint.icon className="w-3 h-3" />
                        <span>{maint.label}</span>
                      </div>
                    )}

                    {/* Top SHAP features */}
                    {sys.metrics.length > 0 && (
                      <div className="mt-2 space-y-0.5">
                        <div className="text-[9px] text-muted-foreground tracking-wider">TOP FEATURES</div>
                        {sys.metrics.slice(0, 3).map(m => (
                          <div key={m.label} className="flex items-center justify-between text-[10px]">
                            <span className="text-muted-foreground truncate mr-2">{m.label}</span>
                            <span className="font-mono text-foreground">{m.value}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </motion.div>
                );
              })}
            </div>
          </div>

          {/* Race-by-Race Health Trend */}
          {healthTrendData.length > 1 && (
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.20)] rounded-xl shadow-[0_4px_12px_rgba(0,0,0,0.4)] p-4"
            >
              <h3 className="text-sm text-foreground tracking-widest mb-3 font-medium">HEALTH TREND</h3>
              <div className="h-[160px]">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={healthTrendData}>
                    <XAxis dataKey="race" tick={{ fontSize: 9, fill: '#666' }} interval="preserveStartEnd" />
                    <YAxis domain={[0, 100]} tick={{ fontSize: 9, fill: '#666' }} width={30} />
                    <Tooltip content={<CustomTooltip />} />
                    <Area type="monotone" dataKey="Speed" stroke="#FF8000" fill="#FF8000" fillOpacity={0.08} strokeWidth={1.5} dot={false} />
                    <Area type="monotone" dataKey="Lap Pace" stroke="#00d4ff" fill="#00d4ff" fillOpacity={0.08} strokeWidth={1.5} dot={false} />
                    <Area type="monotone" dataKey="Tyre Management" stroke="#22c55e" fill="#22c55e" fillOpacity={0.08} strokeWidth={1.5} dot={false} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              <div className="flex items-center gap-4 mt-2 justify-center">
                <div className="flex items-center gap-1.5 text-[10px]"><span className="w-2 h-2 rounded-full bg-[#FF8000]" /> Speed</div>
                <div className="flex items-center gap-1.5 text-[10px]"><span className="w-2 h-2 rounded-full bg-[#00d4ff]" /> Lap Pace</div>
                <div className="flex items-center gap-1.5 text-[10px]"><span className="w-2 h-2 rounded-full bg-[#22c55e]" /> Tyre Management</div>
              </div>
            </motion.div>
          )}
        </>
      )}

      {/* Row 1: Positions + Lap Times */}
      <div className="grid grid-cols-12 gap-4">
        {/* Position Board */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          className="col-span-5 bg-[#1A1F2E] border border-[rgba(255,128,0,0.20)] rounded-xl shadow-[0_4px_12px_rgba(0,0,0,0.4)] p-4"
        >
          <h3 className="text-sm text-foreground tracking-widest mb-3 font-medium">
            {replayActive ? 'MCLAREN POSITIONS' : 'LIVE POSITIONS'}
          </h3>
          <div className="space-y-1">
            <AnimatePresence mode="popLayout">
              {activePositions?.slice(0, 20).map((pos) => {
                const driver = driverMap.get(pos.driver_number);
                const delta = positionDeltas.get(pos.driver_number) ?? 0;
                const hasFastestLap = fastestLapInfo?.driver_number === pos.driver_number;
                return (
                  <motion.div
                    key={pos.driver_number}
                    layout
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ layout: { duration: 0.4, ease: 'easeInOut' } }}
                    className="flex items-center gap-3 px-2 py-1.5 rounded-lg hover:bg-[#222838] transition-colors"
                  >
                    <span className="text-foreground font-mono text-sm w-6 text-right">P{pos.position}</span>
                    <div className="w-1 h-5 rounded-full" style={{ backgroundColor: driver ? `#${driver.team_colour}` : '#555' }} />
                    <span className="text-[#FF8000] font-mono text-sm w-6">{pos.driver_number}</span>
                    <span className="text-foreground text-sm flex-1 truncate">
                      {driver?.broadcast_name ?? `Driver ${pos.driver_number}`}
                    </span>
                    {/* Position delta */}
                    <span className="w-4 flex items-center justify-center">
                      {delta > 0 && <ChevronUp className="w-3 h-3 text-green-400" />}
                      {delta < 0 && <ChevronDown className="w-3 h-3 text-red-400" />}
                    </span>
                    {/* Fastest lap badge */}
                    {hasFastestLap && (
                      <span className="text-[11px] font-mono text-purple-400 bg-purple-400/10 px-1.5 rounded">FL</span>
                    )}
                    <span className="text-[12px] text-muted-foreground truncate max-w-[80px]">
                      {driver?.team_name ?? ''}
                    </span>
                  </motion.div>
                );
              })}
            </AnimatePresence>
            {(!activePositions || activePositions.length === 0) && (
              <div className="text-center text-muted-foreground text-sm py-4">
                {replayActive && replayTime === 0 ? 'Press play to start replay' : 'No position data — try selecting a Race session'}
              </div>
            )}
          </div>
        </motion.div>

        {/* Lap Times Chart */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.05 }}
          className="col-span-7 bg-[#1A1F2E] border border-[rgba(255,128,0,0.20)] rounded-xl shadow-[0_4px_12px_rgba(0,0,0,0.4)] p-4"
        >
          <div className="flex items-center justify-between mb-3">
            <div>
              <h3 className="text-sm text-foreground">Lap Times</h3>
              <p className="text-[12px] text-muted-foreground">
                {replayActive ? `Replay — ${replaySpeed}x speed` : 'Real-time lap duration'}
              </p>
            </div>
            <div className="flex items-center gap-3 text-[12px]">
              {activePositions?.slice(0, 5).map((pos, i) => {
                const driver = driverMap.get(pos.driver_number);
                return (
                  <span key={pos.driver_number} className="flex items-center gap-1">
                    <span className="w-3 h-0.5 rounded" style={{ backgroundColor: topDriverColors[i] }} />
                    {driver?.name_acronym || `D${pos.driver_number}`}
                  </span>
                );
              })}
            </div>
          </div>
          <div className="h-[220px]">
            {lapChartData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={lapChartData}>
                  <XAxis dataKey="lap" stroke="#8888a0" fontSize={10} tickFormatter={(v) => `L${v}`} />
                  <YAxis stroke="#8888a0" fontSize={10} domain={['dataMin - 2', 'dataMax + 2']} tickFormatter={(v) => `${v.toFixed(1)}s`} />
                  <Tooltip content={<CustomTooltip />} />
                  {fastestLapInfo && (
                    <ReferenceLine
                      y={fastestLapInfo.lap_duration}
                      stroke="#a855f7"
                      strokeDasharray="4 2"
                      strokeOpacity={0.5}
                    />
                  )}
                  {activePositions?.slice(0, 5).map((pos, i) => {
                    const driver = driverMap.get(pos.driver_number);
                    const key = driver?.name_acronym || `D${pos.driver_number}`;
                    return (
                      <Line key={pos.driver_number} type="monotone" dataKey={key} stroke={topDriverColors[i]} strokeWidth={1.5} dot={false} name={key} connectNulls />
                    );
                  })}
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-full text-muted-foreground text-sm">
                {replayActive && replayTime === 0 ? 'Press play to start replay' : activeLaps ? 'Processing lap data...' : 'Loading lap data...'}
              </div>
            )}
          </div>
        </motion.div>
      </div>

      {/* ── ANALYTICS ── */}
      <div className="flex items-center gap-2 mt-2">
        <div className="h-px flex-1 bg-[rgba(255,128,0,0.10)]" />
        <span className="text-[10px] tracking-[0.25em] text-[#FF8000]/60 font-semibold">ANALYTICS</span>
        <div className="h-px flex-1 bg-[rgba(255,128,0,0.10)]" />
      </div>

      {/* Row 2: Sector Times + Gap to Leader */}
      <div className="grid grid-cols-12 gap-4">
        {/* Sector Times */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="col-span-5 bg-[#1A1F2E] border border-[rgba(255,128,0,0.20)] rounded-xl shadow-[0_4px_12px_rgba(0,0,0,0.4)] p-4"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm text-foreground tracking-widest font-medium">SECTOR TIMES</h3>
            <span className="text-[12px] text-muted-foreground font-mono">Lap {currentMaxLap}</span>
          </div>
          {sectorTimesData.length > 0 ? (
            <div className="space-y-0.5">
              {/* Header */}
              <div className="grid grid-cols-[60px_1fr_1fr_1fr_1fr] gap-1 text-[11px] text-muted-foreground pb-1 border-b border-[rgba(255,128,0,0.12)]">
                <span>Driver</span>
                <span className="text-right">S1</span>
                <span className="text-right">S2</span>
                <span className="text-right">S3</span>
                <span className="text-right">Total</span>
              </div>
              {sectorTimesData.map(row => {
                const pb = sectorBests.personal.get(row.driver_number) ?? { s1: Infinity, s2: Infinity, s3: Infinity };
                return (
                  <div key={row.driver_number} className="grid grid-cols-[60px_1fr_1fr_1fr_1fr] gap-1 text-[12px] font-mono py-1 hover:bg-[#222838] rounded transition-colors">
                    <span className="flex items-center gap-1.5">
                      <span className="w-1 h-3 rounded-full" style={{ backgroundColor: `#${row.team_colour}` }} />
                      <span className="text-foreground">{row.acronym}</span>
                    </span>
                    <span className={`text-right ${sectorColor(row.s1, sectorBests.overallS1, pb.s1)}`}>
                      {row.s1?.toFixed(3) ?? '—'}
                    </span>
                    <span className={`text-right ${sectorColor(row.s2, sectorBests.overallS2, pb.s2)}`}>
                      {row.s2?.toFixed(3) ?? '—'}
                    </span>
                    <span className={`text-right ${sectorColor(row.s3, sectorBests.overallS3, pb.s3)}`}>
                      {row.s3?.toFixed(3) ?? '—'}
                    </span>
                    <span className="text-right text-foreground">
                      {row.total?.toFixed(3) ?? '—'}
                    </span>
                  </div>
                );
              })}
              {/* Legend */}
              <div className="flex items-center gap-4 pt-2 text-[11px]">
                <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-purple-400" /> Overall best</span>
                <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-green-400" /> Personal best</span>
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-center h-[120px] text-muted-foreground text-sm">
              {replayActive && replayTime === 0 ? 'Press play to start replay' : 'No sector data'}
            </div>
          )}
        </motion.div>

        {/* Gap to Leader Chart */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15 }}
          className="col-span-7 bg-[#1A1F2E] border border-[rgba(255,128,0,0.20)] rounded-xl shadow-[0_4px_12px_rgba(0,0,0,0.4)] p-4"
        >
          <div className="flex items-center justify-between mb-3">
            <div>
              <h3 className="text-sm text-foreground font-medium">Gap to Leader</h3>
              <p className="text-[12px] text-muted-foreground">Interval in seconds per lap</p>
            </div>
          </div>
          <div className="h-[220px]">
            {gapChartData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={gapChartData}>
                  <XAxis dataKey="lap" stroke="#8888a0" fontSize={10} tickFormatter={(v) => `L${v}`} />
                  <YAxis stroke="#8888a0" fontSize={10} tickFormatter={(v) => `${v.toFixed(1)}s`} />
                  <Tooltip content={<CustomTooltip />} />
                  {activePositions?.slice(0, 5).map((pos, i) => {
                    const driver = driverMap.get(pos.driver_number);
                    const key = driver?.name_acronym || `D${pos.driver_number}`;
                    return (
                      <Line key={pos.driver_number} type="monotone" dataKey={key} stroke={topDriverColors[i]} strokeWidth={1.5} dot={false} name={key} connectNulls />
                    );
                  })}
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-full text-muted-foreground text-sm">
                {replayActive && replayTime === 0 ? 'Press play to start replay' : 'No interval data'}
              </div>
            )}
          </div>
        </motion.div>
      </div>

      {/* Row 3: Stint Timeline (full width) */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.20)] rounded-xl shadow-[0_4px_12px_rgba(0,0,0,0.4)] p-4"
      >
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm text-foreground tracking-widest font-medium">TIRE STRATEGY</h3>
          <div className="flex items-center gap-3 text-[11px]">
            {Object.entries(COMPOUND_COLORS).map(([name, color]) => (
              <span key={name} className="flex items-center gap-1">
                <span className="w-3 h-2 rounded-sm" style={{ backgroundColor: color, opacity: 0.8 }} />
                <span className="text-muted-foreground">{name}</span>
              </span>
            ))}
          </div>
        </div>
        {activePositions && activePositions.length > 0 ? (
          <StintTimeline
            positions={activePositions}
            stintsByDriver={stintsByDriver}
            pitStops={activePitStops ?? []}
            driverMap={driverMap}
            currentMaxLap={currentMaxLap}
            totalLaps={totalLaps || currentMaxLap}
          />
        ) : (
          <div className="flex items-center justify-center h-[80px] text-muted-foreground text-sm">
            {replayActive && replayTime === 0 ? 'Press play to start replay' : 'No stint data'}
          </div>
        )}
      </motion.div>

      {/* ── REFERENCE ── */}
      <div className="flex items-center gap-2 mt-2">
        <div className="h-px flex-1 bg-[rgba(255,128,0,0.10)]" />
        <span className="text-[10px] tracking-[0.25em] text-[#FF8000]/60 font-semibold">REFERENCE</span>
        <div className="h-px flex-1 bg-[rgba(255,128,0,0.10)]" />
      </div>

      {/* Row 4: Race Control Log + Weather */}
      <div className="grid grid-cols-12 gap-4">
        {/* Race Control Event Log */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.25 }}
          className="col-span-8 bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm text-muted-foreground tracking-widest">RACE CONTROL</h3>
            <span className="text-[12px] text-muted-foreground">{filteredRaceControl.length} events</span>
          </div>
          <ScrollArea className="h-[180px]">
            <div className="space-y-0.5 pr-3">
              {filteredRaceControl.length > 0 ? (
                filteredRaceControl.slice().reverse().map((ev, i) => (
                  <motion.div
                    key={`${ev.date}-${ev.message}-${i}`}
                    initial={{ opacity: 0, x: -8 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.2 }}
                    className={`flex items-center gap-2 text-[12px] py-1 rounded px-1 hover:bg-[#222838] ${flagColor(ev.flag, ev.category)}`}
                  >
                    <span className="w-1.5 h-1.5 rounded-full bg-current shrink-0" />
                    {ev.lap_number != null && (
                      <span className="text-muted-foreground font-mono w-6 shrink-0">L{ev.lap_number}</span>
                    )}
                    <span className="truncate flex-1">{ev.message}</span>
                  </motion.div>
                ))
              ) : (
                <div className="text-center text-muted-foreground text-sm py-4">No events</div>
              )}
            </div>
          </ScrollArea>
        </motion.div>

        {/* Weather Trend */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="col-span-4 bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4"
        >
          <h3 className="text-sm text-muted-foreground tracking-widest mb-3">WEATHER TREND</h3>
          {activeWeather && activeWeather.length > 0 ? (
            <div className="h-[130px]">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={activeWeather.slice(-30)}>
                  <defs>
                    <linearGradient id="tempGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#FF8000" stopOpacity={0.3} />
                      <stop offset="100%" stopColor="#FF8000" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <Area type="monotone" dataKey="track_temperature" stroke="#FF8000" fill="url(#tempGrad)" strokeWidth={1.5} dot={false} name="Track Temp" />
                  <Area type="monotone" dataKey="air_temperature" stroke="#00d4ff" fill="transparent" strokeWidth={1} dot={false} name="Air Temp" />
                  <YAxis hide domain={['dataMin - 2', 'dataMax + 2']} />
                  <Tooltip content={<CustomTooltip />} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="flex items-center justify-center h-[130px] text-muted-foreground text-sm">
              No weather data
            </div>
          )}
          {latestWeather && (
            <div className="grid grid-cols-2 gap-2 mt-2 text-[12px]">
              <div>
                <span className="text-muted-foreground">Humidity: </span>
                <span className="font-mono text-foreground">{latestWeather.humidity}%</span>
              </div>
              <div>
                <span className="text-muted-foreground">Pressure: </span>
                <span className="font-mono text-foreground">{latestWeather.pressure} hPa</span>
              </div>
            </div>
          )}
        </motion.div>
      </div>
    </div>
  );
}

// --- Sub-Components ---

function AnimatedKPI({ icon, label, value, sub, color }: {
  icon: React.ReactNode; label: string; value: string; sub: string; color: string;
}) {
  return (
    <div className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] border-t-2 border-t-[rgba(255,128,0,0.25)] rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
      <div className="flex items-center gap-2 mb-2">
        {icon}
        <span className="text-[12px] text-muted-foreground tracking-widest uppercase">{label}</span>
      </div>
      <AnimatePresence mode="wait">
        <motion.div
          key={value}
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -6 }}
          transition={{ duration: 0.15 }}
          className={`text-2xl font-mono font-semibold ${color}`}
        >
          {value}
        </motion.div>
      </AnimatePresence>
      <div className="text-[12px] text-muted-foreground mt-1">{sub}</div>
    </div>
  );
}

function StintTimeline({ positions, stintsByDriver, pitStops, driverMap, currentMaxLap, totalLaps }: {
  positions: { driver_number: number; position: number }[];
  stintsByDriver: Map<number, OpenF1Stint[]>;
  pitStops: { driver_number: number; lap_number: number; pit_duration: number | null; date: string }[];
  driverMap: Map<number, OpenF1Driver>;
  currentMaxLap: number;
  totalLaps: number;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(600);

  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver(entries => {
      for (const entry of entries) setWidth(entry.contentRect.width);
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  const driverRows = positions.map(p => p.driver_number);
  const rowHeight = 32;
  const barHeight = 20;
  const labelWidth = 56;
  const maxLap = Math.max(totalLaps, currentMaxLap, 1);
  const chartWidth = width - labelWidth;

  return (
    <div ref={containerRef} className="w-full overflow-hidden">
      <svg width={width} height={driverRows.length * rowHeight + 24} className="select-none">
        {driverRows.map((dNum, i) => {
          const driver = driverMap.get(dNum);
          const driverStints = stintsByDriver.get(dNum) ?? [];
          const y = i * rowHeight;

          return (
            <g key={dNum}>
              {/* Driver label */}
              <text
                x={0}
                y={y + rowHeight / 2 + 4}
                fill="#8888a0"
                fontSize={10}
                fontFamily="ui-monospace, monospace"
              >
                {driver?.name_acronym ?? `D${dNum}`}
              </text>

              {/* Stint bars */}
              {driverStints.map((stint, si) => {
                const x1 = labelWidth + (stint.lap_start / maxLap) * chartWidth;
                const endLap = stint.lap_end > 0 ? stint.lap_end : currentMaxLap;
                const x2 = labelWidth + (endLap / maxLap) * chartWidth;
                const barColor = COMPOUND_COLORS[stint.compound] ?? '#8888a0';
                return (
                  <motion.rect
                    key={si}
                    x={x1}
                    y={y + (rowHeight - barHeight) / 2}
                    width={Math.max(x2 - x1, 2)}
                    height={barHeight}
                    fill={barColor}
                    opacity={0.75}
                    rx={4}
                    initial={{ scaleX: 0 }}
                    animate={{ scaleX: 1 }}
                    transition={{ duration: 0.4, ease: 'easeOut' }}
                    style={{ transformOrigin: `${x1}px ${y + rowHeight / 2}px` }}
                  />
                );
              })}

              {/* Pit stop markers */}
              {pitStops
                .filter(p => p.driver_number === dNum)
                .map((pit, pi) => {
                  const px = labelWidth + (pit.lap_number / maxLap) * chartWidth;
                  return (
                    <line
                      key={`pit-${pi}`}
                      x1={px}
                      y1={y + 2}
                      x2={px}
                      y2={y + rowHeight - 2}
                      stroke="#ffffff"
                      strokeWidth={1.5}
                      strokeDasharray="2 2"
                      opacity={0.5}
                    />
                  );
                })}
            </g>
          );
        })}

        {/* Lap axis */}
        {Array.from({ length: Math.ceil(maxLap / 10) + 1 }, (_, i) => i * 10).filter(l => l <= maxLap).map(lap => {
          const x = labelWidth + (lap / maxLap) * chartWidth;
          return (
            <text
              key={`axis-${lap}`}
              x={x}
              y={driverRows.length * rowHeight + 16}
              fill="#8888a0"
              fontSize={9}
              textAnchor="middle"
              fontFamily="ui-monospace, monospace"
            >
              L{lap}
            </text>
          );
        })}
      </svg>
    </div>
  );
}
