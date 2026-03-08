import { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import {
  AreaChart, Area, LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip, ReferenceLine,
} from 'recharts';
import {
  Gauge, Zap, Timer, Loader2, AlertCircle,
  Flag, MapPin, Cloud, Wind, Play, Pause, Bug,
  Trophy,
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { usePolling } from '../hooks/usePolling';
import * as openf1 from '../api/openf1';
import type { OpenF1Session, OpenF1Lap, OpenF1Driver, OpenF1Stint } from '../types';
import { TrackMapGL } from './TrackMapGL';
import { LiveAnalyticsSidebar } from './LiveAnalyticsSidebar';
import { getCircuitByShortName } from '../data/circuits';

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload?.length) {
    return (
      <div className="bg-background border border-[rgba(255,128,0,0.2)] rounded-lg p-2 text-[12px]">
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

export function LiveDashboard() {
  const [selectedSession, setSelectedSession] = useState<OpenF1Session | null>(null);
  const [replayActive, setReplayActive] = useState(false);
  const [replayTime, setReplayTime] = useState(0);
  const [replaySpeed, setReplaySpeed] = useState(5);
  const [playing, setPlaying] = useState(false);
  const [showDebug, setShowDebug] = useState(false);
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

  // Detect if selected session is currently live (between start and end, or end not yet set)
  const isSessionLive = useMemo(() => {
    if (!selectedSession?.date_start) return false;
    const now = Date.now();
    const started = sessionStartMs > 0 && now >= sessionStartMs;
    const notEnded = !sessionEndMs || now <= sessionEndMs + 600_000; // 10min grace after official end
    return started && notEnded;
  }, [selectedSession, sessionStartMs, sessionEndMs]);

  useEffect(() => {
    setReplayTime(0);
    setPlaying(false);
    // Auto-enable replay for historical sessions
    setReplayActive(!isSessionLive && sessionDuration > 0);
  }, [sessionKey, isSessionLive, sessionDuration]);

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
  const carTrackPositions = useMemo(() => {
    if (!replayActive || !laps || laps.length === 0 || !activePositions || activePositions.length === 0) return [];

    const cars: import('./TrackMap').CarPosition[] = [];

    for (const pos of activePositions) {
      const driver = driverMap.get(pos.driver_number);
      if (!driver) continue;

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
      const elapsed = (cutoffMs - lapStartMs) / 1000;
      const lapDuration = currentLap.lap_duration || 90;
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

  // Position deltas
  const positionDeltas = useMemo(() => {
    const deltas = new Map<number, number>();
    if (activePositions) {
      for (const pos of activePositions) {
        const prev = prevPositionsRef.current.get(pos.driver_number);
        if (prev !== undefined) {
          deltas.set(pos.driver_number, prev - pos.position);
        }
      }
    }
    return deltas;
  }, [activePositions]);

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

  // Index laps by driver+lap for O(1) lookup
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

  // Gap chart data
  const gapChartData = useMemo(() => {
    if (!rawIntervals || !activeLaps || !activePositions) return [];
    const topDrivers = activePositions.slice(0, 5).map(p => p.driver_number);
    const topDriverSet = new Set(topDrivers);

    const lapTimes = new Map<number, number>();
    for (const lap of activeLaps) {
      if (!lapTimes.has(lap.lap_number)) {
        lapTimes.set(lap.lap_number, new Date(lap.date_start).getTime());
      }
    }

    let maxLapTs = 0;
    for (const [, ts] of lapTimes) {
      if (ts > maxLapTs) maxLapTs = ts;
    }

    const byDriver = new Map<number, typeof rawIntervals>();
    for (const iv of rawIntervals) {
      if (!topDriverSet.has(iv.driver_number)) continue;
      if (replayActive && new Date(iv.date).getTime() > maxLapTs + 120000) continue;
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

  // Stint summary
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

  useEffect(() => {
    return () => { scrubCleanupRef.current?.(); };
  }, []);

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
            <div className="bg-secondary rounded-lg px-3 py-1.5 font-mono flex items-center gap-1.5">
              <MapPin className="w-3 h-3 text-[#FF8000]" />
              <span className="text-foreground">{selectedSession.circuit_short_name}</span>
            </div>
            <div className="bg-secondary rounded-lg px-3 py-1.5 font-mono flex items-center gap-1.5">
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
              className="appearance-none bg-secondary border border-border rounded-lg px-3 py-1.5 text-sm text-foreground focus:outline-none cursor-pointer"
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

      {/* Session Control Bar */}
      <div className={`bg-secondary border rounded-lg shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-3 ${
        isSessionLive ? 'border-green-500/30 border-t-green-500/60' : 'border-[rgba(255,128,0,0.20)] border-t-[#FF8000]/40'
      }`}>
        <div className="flex items-center gap-4">
          {isSessionLive ? (
            /* Live session indicator — not a toggle, just status */
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-mono bg-green-500/10 text-green-400 border border-green-500/20">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-green-400" />
              </span>
              LIVE
            </div>
          ) : (
            /* Historical session — replay toggle */
            <button
              onClick={toggleReplay}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-mono transition-colors ${
                replayActive
                  ? 'bg-[#FF8000]/15 text-[#FF8000] border border-[#FF8000]/30'
                  : 'bg-secondary text-muted-foreground border border-border hover:text-foreground'
              }`}
            >
              {replayActive ? <Pause className="w-3 h-3" /> : <Play className="w-3 h-3" />}
              RACE REVIEW
            </button>
          )}

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
                        : 'text-muted-foreground hover:text-foreground hover:bg-secondary'
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
                  className="flex-1 h-3 bg-secondary rounded-full relative cursor-pointer select-none"
                  onPointerDown={onScrubDown}
                >
                  <motion.div
                    className="absolute top-0 left-0 h-full rounded-full bg-gradient-to-r from-[#FF8000] to-[#FF8000]/70"
                    style={{ width: `${scrubPct}%` }}
                  />
                  {flagMarkers.map((m, i) => (
                    <div
                      key={`flag-${i}`}
                      className="absolute top-0 w-1 h-full rounded-full"
                      style={{ left: `${m.pct * 100}%`, backgroundColor: m.color, opacity: 0.7 }}
                    />
                  ))}
                  {pitMarkers.map((m, i) => (
                    <div
                      key={`pit-${i}`}
                      className="absolute top-0 w-px h-full"
                      style={{ left: `${m.pct * 100}%`, backgroundColor: '#fff', opacity: 0.3 }}
                    />
                  ))}
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

          {!replayActive && !isSessionLive && (
            <div className="flex-1 text-[12px] text-muted-foreground">
              Click <span className="text-[#FF8000]">RACE REVIEW</span> to replay this session
            </div>
          )}
          {isSessionLive && (
            <div className="flex-1 text-[12px] text-green-400/80">
              Streaming live session data — {selectedSession.session_name}
            </div>
          )}

          <button
            onClick={() => setShowDebug(d => !d)}
            className={`p-1.5 rounded-lg transition-colors ${showDebug ? 'bg-amber-500/20 text-amber-400' : 'text-muted-foreground hover:text-foreground hover:bg-secondary'}`}
            title="Toggle debug info"
          >
            <Bug className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Race control ticker */}
        {replayActive && latestEvents.length > 0 && (
          <div className="flex items-center gap-4 mt-2 pt-2 border-t border-border">
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
        <div className="bg-background border border-amber-500/30 rounded-lg p-3 text-[12px] font-mono space-y-1">
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

      {/* Circuit Track Map + Live Analytics Sidebar */}
      {(() => {
        const circuit = selectedSession ? getCircuitByShortName(selectedSession.circuit_short_name) : undefined;
        if (!circuit) return (
          <div className="bg-card rounded-lg border border-border p-6 text-center">
            <div className="text-amber-400 text-sm">No track map for circuit: "{selectedSession?.circuit_short_name}"</div>
          </div>
        );
        return (
          <div className="grid grid-cols-12 gap-4">
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
            <div className="col-span-5">
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
                height={550}
              />
            </div>
          </div>
        );
      })()}

      {/* Lap Times Chart (full width) */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.05 }}
        className="bg-card border border-[rgba(255,128,0,0.20)] rounded-lg shadow-[0_4px_12px_rgba(0,0,0,0.4)] p-4"
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

      {/* ── ANALYTICS ── */}
      <div className="flex items-center gap-2 mt-2">
        <div className="h-px flex-1 bg-[rgba(255,128,0,0.10)]" />
        <span className="text-[10px] tracking-[0.25em] text-[#FF8000]/60 font-semibold">ANALYTICS</span>
        <div className="h-px flex-1 bg-[rgba(255,128,0,0.10)]" />
      </div>

      {/* Gap to Leader Chart (full width) */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="bg-card border border-[rgba(255,128,0,0.20)] rounded-lg shadow-[0_4px_12px_rgba(0,0,0,0.4)] p-4"
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

      {/* Stint Timeline (full width) */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="bg-card border border-[rgba(255,128,0,0.20)] rounded-lg shadow-[0_4px_12px_rgba(0,0,0,0.4)] p-4"
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

      {/* Weather Trend (full width) */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.25 }}
        className="bg-card border border-border rounded-lg shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4"
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
                <Area type="monotone" dataKey="track_temperature" stroke="#FF8000" fill="url(#tempGrad)" strokeWidth={1.5} dot={false} name="Track °C" />
                <Area type="monotone" dataKey="air_temperature" stroke="#00d4ff" fill="transparent" strokeWidth={1} dot={false} name="Air °C" />
                <Area type="monotone" dataKey="wind_speed" stroke="#f59e0b" fill="transparent" strokeWidth={1} strokeDasharray="4 2" dot={false} name="Wind km/h" />
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
        {activeWeather && activeWeather.length > 0 && (
          <div className="flex items-center gap-3 mt-1.5 justify-center text-[10px]">
            <span className="flex items-center gap-1"><span className="w-3 h-0.5 rounded bg-[#FF8000]" /> Track</span>
            <span className="flex items-center gap-1"><span className="w-3 h-0.5 rounded bg-[#00d4ff]" /> Air</span>
            <span className="flex items-center gap-1"><span className="w-3 h-0.5 rounded bg-[#f59e0b] opacity-70" style={{ borderTop: '1px dashed #f59e0b' }} /> Wind</span>
          </div>
        )}
        {latestWeather && (
          <div className="grid grid-cols-2 gap-x-3 gap-y-1 mt-2 text-[12px]">
            <div>
              <span className="text-muted-foreground">Humidity: </span>
              <span className="font-mono text-foreground">{latestWeather.humidity}%</span>
            </div>
            <div>
              <span className="text-muted-foreground">Pressure: </span>
              <span className="font-mono text-foreground">{latestWeather.pressure} hPa</span>
            </div>
            <div className="flex items-center gap-1">
              <Wind className="w-3 h-3 text-amber-400" />
              <span className="text-muted-foreground">Wind: </span>
              <span className="font-mono text-foreground">{latestWeather.wind_speed} km/h</span>
            </div>
            <div>
              <span className={`font-mono ${latestWeather.rainfall ? 'text-blue-400' : 'text-green-400'}`}>
                {latestWeather.rainfall ? '🌧 Rain' : '☀ Dry'}
              </span>
            </div>
          </div>
        )}
      </motion.div>
    </div>
  );
}

// --- Sub-Components ---

function AnimatedKPI({ icon, label, value, sub, color }: {
  icon: React.ReactNode; label: string; value: string; sub: string; color: string;
}) {
  return (
    <div className="bg-card border border-border border-t-2 border-t-[rgba(255,128,0,0.25)] rounded-lg shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
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
              <text
                x={0}
                y={y + rowHeight / 2 + 4}
                fill="#8888a0"
                fontSize={10}
                fontFamily="ui-monospace, monospace"
              >
                {driver?.name_acronym ?? `D${dNum}`}
              </text>

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
