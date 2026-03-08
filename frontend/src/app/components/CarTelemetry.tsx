import React, { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, AreaChart, Area, BarChart, Bar, Cell,
  ReferenceLine,
} from 'recharts';
import {
  Gauge, Zap, Timer, Loader2, AlertCircle, Wind, GitCompareArrows, Disc,
  ArrowUp, ArrowDown, Activity, TrendingUp,
} from 'lucide-react';
import KexBriefingCard from './KexBriefingCard';
import { HealthGauge } from './HealthGauge';
import { StatusBadge } from './StatusBadge';
import { parseAnomalyDrivers, type VehicleData } from './anomalyHelpers';
import { getCarTelemetryKex, type CarTelemetryKex } from '../api/driverIntel';
// All data fetched from MongoDB via JSON API endpoints

const MCLAREN_LOGO = 'https://media.formula1.com/image/upload/c_lfill,w_96/q_auto/v1740000000/common/f1/2026/mclaren/2026mclarenlogowhite.webp';

const DRIVER_INFO: Record<string, { name: string; nationality: string; flag: string; number: number }> = {
  NOR: { name: 'Lando Norris', nationality: 'British', flag: '\u{1F1EC}\u{1F1E7}', number: 4 },
  PIA: { name: 'Oscar Piastri', nationality: 'Australian', flag: '\u{1F1E6}\u{1F1FA}', number: 81 },
};

const compoundColors: Record<string, string> = {
  SOFT: '#ef4444', MEDIUM: '#f59e0b', HARD: '#e8e8f0', INTERMEDIATE: '#22c55e', WET: '#3b82f6',
};

interface TelemetryMeta {
  years: number[];
  drivers: string[];
  races_by_year: Record<string, string[]>;
  drivers_by_year?: Record<string, string[]>;
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload?.length) {
    return (
      <div className="bg-background border border-[rgba(255,128,0,0.2)] rounded-lg p-2 text-[12px]">
        <div className="text-muted-foreground mb-1">{label}</div>
        {payload.map((entry: any, index: number) => (
          <div key={index} className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full" style={{ backgroundColor: entry.color }} />
            <span className="text-muted-foreground">{entry.name}:</span>
            <span className="text-foreground font-mono">{typeof entry.value === 'number' ? entry.value.toFixed(1) : entry.value}</span>
          </div>
        ))}
      </div>
    );
  }
  return null;
};

function KPI({ icon, label, value, detail, color = 'text-foreground' }: { icon: React.ReactNode; label: string; value: string; detail: string; color?: string }) {
  return (
    <div className="bg-card border border-border rounded-lg shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-3">
      <div className="flex items-center gap-2 mb-1.5">
        {icon}
        <span className="text-[12px] text-muted-foreground tracking-wider">{label}</span>
      </div>
      <div className={`text-lg font-mono ${color}`}>{value}</div>
      <div className="text-[12px] text-muted-foreground mt-0.5">{detail}</div>
    </div>
  );
}

function DeltaKPI({ label, val1, val2, label1, label2, unit, icon, color1 = '#FF8000', color2 = '#22d3ee' }: {
  label: string; val1: string; val2: string; label1: string; label2: string;
  unit: string; icon: React.ReactNode; color1?: string; color2?: string;
}) {
  const n1 = parseFloat(val1);
  const n2 = parseFloat(val2);
  const diff = n1 - n2;
  const absDiff = Math.abs(diff).toFixed(1);
  const firstWins = diff > 0;
  return (
    <div className="bg-card border border-border rounded-lg shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-3">
      <div className="flex items-center gap-2 mb-2">
        {icon}
        <span className="text-[12px] text-muted-foreground tracking-wider">{label}</span>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div>
          <div className="text-[11px] text-muted-foreground mb-0.5">{label1}</div>
          <div className="text-base font-mono" style={{ color: color1 }}>{val1}{unit}</div>
        </div>
        <div>
          <div className="text-[11px] text-muted-foreground mb-0.5">{label2}</div>
          <div className="text-base font-mono" style={{ color: color2 }}>{val2}{unit}</div>
        </div>
      </div>
      {diff !== 0 && !isNaN(diff) && (
        <div className="mt-2 flex items-center gap-1 text-[12px]">
          {firstWins
            ? <ArrowUp className="w-3 h-3" style={{ color: color1 }} />
            : <ArrowDown className="w-3 h-3" style={{ color: color2 }} />
          }
          <span className="text-muted-foreground">
            {firstWins ? label1 : label2} +{absDiff}{unit}
          </span>
        </div>
      )}
    </div>
  );
}

interface CarSummary {
  race: string;
  avgSpeed: number;
  topSpeed: number;
  avgRPM: number;
  maxRPM: number;
  avgThrottle: number;
  brakePct: number;
  drsPct: number;
  compounds: string[];
  samples: number;
}

type Tab = 'detail' | 'season' | 'h2h' | 'racecompare';

/* ── helpers ── */
function computeCarKpis(data: Record<string, any>[]) {
  if (!data.length) return null;
  const speeds = data.map(r => Number(r.Speed) || 0);
  const rpms = data.map(r => Number(r.RPM) || 0);
  const drsCount = data.filter(r => Number(r.DRS) >= 10).length;
  const compounds = [...new Set(data.map(r => r.Compound).filter(Boolean))] as string[];
  return {
    topSpeed: Math.max(...speeds).toFixed(0),
    avgSpeed: (speeds.reduce((a, b) => a + b, 0) / speeds.length).toFixed(1),
    avgRPM: (rpms.reduce((a, b) => a + b, 0) / rpms.length).toFixed(0),
    drsActivations: drsCount,
    drsPct: ((drsCount / data.length) * 100).toFixed(1),
    avgThrottle: (data.map(r => Number(r.Throttle) || 0).reduce((a, b) => a + b, 0) / data.length).toFixed(1),
    brakePct: ((data.filter(r => r.Brake === true || r.Brake === 'True' || r.Brake === '1' || r.Brake === 1).length / data.length) * 100).toFixed(1),
    compounds,
  };
}

function buildLapTimes(data: Record<string, any>[]) {
  const lapMap = new Map<number, { times: number[]; compound: string }>();
  for (const r of data) {
    const lap = Number(r.LapNumber);
    const lt = r.LapTime;
    if (!lap || !lt) continue;
    let seconds = 0;
    const match = lt.match(/(\d+):(\d+):(\d+\.?\d*)/);
    if (match) seconds = Number(match[1]) * 3600 + Number(match[2]) * 60 + Number(match[3]);
    if (seconds > 0 && seconds < 300) {
      if (!lapMap.has(lap)) lapMap.set(lap, { times: [], compound: r.Compound || '' });
      lapMap.get(lap)!.times.push(seconds);
    }
  }
  return Array.from(lapMap.entries())
    .map(([lap, { times, compound }]) => ({ lap, time: times[0], compound }))
    .sort((a, b) => a.lap - b.lap);
}

function downsample(data: Record<string, any>[], maxPoints = 500) {
  const step = Math.max(1, Math.floor(data.length / maxPoints));
  return data.filter((_, i) => i % step === 0);
}

export function CarTelemetry() {
  const [meta, setMeta] = useState<TelemetryMeta | null>(null);
  const [tab, setTab] = useState<Tab>('detail');
  const [year, setYear] = useState<number>(0);
  const [race, setRace] = useState('');
  const [driver, setDriver] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [rawData, setRawData] = useState<Record<string, any>[]>([]);
  const [stintsData, setStintsData] = useState<Record<string, any>[]>([]);
  const [seasonSummary, setSeasonSummary] = useState<CarSummary[]>([]);
  const [summaryLoading, setSummaryLoading] = useState(false);

  // KeX briefing state
  const [kex, setKex] = useState<CarTelemetryKex | null>(null);
  const [kexLoading, setKexLoading] = useState(false);

  // Anomaly health state
  const [anomalyVehicles, setAnomalyVehicles] = useState<VehicleData[]>([]);

  // H2H state
  const [h2hMode, setH2hMode] = useState<'race' | 'season'>('race');
  const [h2hDriver2, setH2hDriver2] = useState('');
  const [h2hLoading, setH2hLoading] = useState(false);
  const [h2hRaceData, setH2hRaceData] = useState<Record<string, any>[]>([]);
  const [h2hSeason1, setH2hSeason1] = useState<CarSummary[]>([]);
  const [h2hSeason2, setH2hSeason2] = useState<CarSummary[]>([]);

  // Compare state
  const [compareMode, setCompareMode] = useState<'race' | 'year'>('race');
  const [race2, setRace2] = useState('');
  const [year2, setYear2] = useState<number>(0);
  const [compareLoading, setCompareLoading] = useState(false);
  const [compareData1, setCompareData1] = useState<Record<string, any>[]>([]);
  const [compareData2, setCompareData2] = useState<Record<string, any>[]>([]);
  const [compareSeason1, setCompareSeason1] = useState<CarSummary[]>([]);
  const [compareSeason2, setCompareSeason2] = useState<CarSummary[]>([]);

  // Load metadata on mount
  useEffect(() => {
    fetch('/api/local/mccar-summary/meta')
      .then(r => r.json())
      .then((m: TelemetryMeta) => {
        setMeta(m);
        if (m.years.length) {
          const latestYear = m.years[m.years.length - 1];
          setYear(latestYear);
          const prevYear = m.years.length > 1 ? m.years[m.years.length - 2] : latestYear;
          setYear2(prevYear);
          const yearRaces = m.races_by_year[String(latestYear)] || [];
          if (yearRaces.length) setRace(yearRaces[0]);
          if (yearRaces.length > 1) setRace2(yearRaces[1]);
        }
        if (m.drivers.length) {
          setDriver(m.drivers[0]);
          setH2hDriver2(m.drivers.length > 1 ? m.drivers[1] : m.drivers[0]);
        }
      })
      .catch(() => {});
  }, []);

  // Fetch anomaly health data once
  useEffect(() => {
    fetch('/api/pipeline/anomaly')
      .then(r => r.json())
      .then(data => setAnomalyVehicles(parseAnomalyDrivers(data)))
      .catch(() => {});
  }, []);

  // Fetch KeX briefing when driver/year/race/tab changes
  useEffect(() => {
    setKex(null);
    if (!driver || !year) return;
    if (tab !== 'detail' && tab !== 'season') return;
    setKexLoading(true);
    const raceParam = tab === 'detail' && race ? race : undefined;
    getCarTelemetryKex(driver, year, raceParam)
      .then(setKex)
      .catch(() => setKex(null))
      .finally(() => setKexLoading(false));
  }, [driver, year, race, tab]);

  const selectedVehicle = useMemo(() => {
    if (!driver || anomalyVehicles.length === 0) return null;
    return anomalyVehicles.find(v => v.code === driver) ?? null;
  }, [driver, anomalyVehicles]);

  const years = meta?.years || [];
  const drivers = meta?.drivers_by_year?.[String(year)] || meta?.drivers || [];
  const races = meta?.races_by_year[String(year)] || [];

  // Reset driver selection when year changes (selected driver may not exist in new year)
  useEffect(() => {
    if (drivers.length && !drivers.includes(driver)) {
      setDriver(drivers[0]);
    }
  }, [year, drivers]); // eslint-disable-line react-hooks/exhaustive-deps

  // Load per-race telemetry from MongoDB (detail tab)
  useEffect(() => {
    if (tab !== 'detail' || !year || !race) return;
    setLoading(true);
    setError(null);
    Promise.allSettled([
      fetch(`/api/local/mccar-race-telemetry/${year}/${race}`).then(r => r.json()).then(d => setRawData(Array.isArray(d) ? d : [])),
      fetch(`/api/local/mccar-race-stints/${year}/${race}`).then(r => r.json()).then(d => setStintsData(d)),
    ]).then(results => {
      if (results[0].status === 'rejected') { setError('Race data not available'); setRawData([]); }
    }).finally(() => setLoading(false));
  }, [year, race, tab]);

  // Load season summary (season tab)
  useEffect(() => {
    if (tab !== 'season' || !year || !driver) return;
    setSummaryLoading(true);
    fetch(`/api/local/mccar-summary/${year}/${driver}`)
      .then(res => res.json())
      .then(data => setSeasonSummary(data))
      .catch(() => setSeasonSummary([]))
      .finally(() => setSummaryLoading(false));
  }, [year, driver, tab]);

  // H2H race — load from MongoDB (contains both drivers)
  useEffect(() => {
    if (tab !== 'h2h' || h2hMode !== 'race' || !year || !race) return;
    setH2hLoading(true);
    fetch(`/api/local/mccar-race-telemetry/${year}/${race}`)
      .then(r => r.json())
      .then(d => setH2hRaceData(d))
      .catch(() => setH2hRaceData([]))
      .finally(() => setH2hLoading(false));
  }, [year, race, tab, h2hMode]);

  // H2H season — both drivers
  useEffect(() => {
    if (tab !== 'h2h' || h2hMode !== 'season' || !driver || !h2hDriver2 || !year) return;
    setH2hLoading(true);
    Promise.all([
      fetch(`/api/local/mccar-summary/${year}/${driver}`).then(r => r.json()).catch(() => []),
      fetch(`/api/local/mccar-summary/${year}/${h2hDriver2}`).then(r => r.json()).catch(() => []),
    ]).then(([d1, d2]) => { setH2hSeason1(d1); setH2hSeason2(d2); })
      .finally(() => setH2hLoading(false));
  }, [year, driver, h2hDriver2, tab, h2hMode]);

  // Compare: race vs race (from MongoDB)
  useEffect(() => {
    if (tab !== 'racecompare' || compareMode !== 'race' || !year || !race || !race2) return;
    setCompareLoading(true);
    Promise.all([
      fetch(`/api/local/mccar-race-telemetry/${year}/${race}`).then(r => r.json()).catch(() => []),
      fetch(`/api/local/mccar-race-telemetry/${year}/${race2}`).then(r => r.json()).catch(() => []),
    ]).then(([d1, d2]) => { setCompareData1(d1); setCompareData2(d2); })
      .finally(() => setCompareLoading(false));
  }, [year, race, race2, driver, tab, compareMode]);

  // Compare: year vs year
  useEffect(() => {
    if (tab !== 'racecompare' || compareMode !== 'year' || !driver || !year || !year2) return;
    setCompareLoading(true);
    Promise.all([
      fetch(`/api/local/mccar-summary/${year}/${driver}`).then(r => r.json()).catch(() => []),
      fetch(`/api/local/mccar-summary/${year2}/${driver}`).then(r => r.json()).catch(() => []),
    ]).then(([s1, s2]) => { setCompareSeason1(s1); setCompareSeason2(s2); })
      .finally(() => setCompareLoading(false));
  }, [driver, year, year2, tab, compareMode]);

  const driverData = useMemo(() => rawData.filter(r => r.Driver === driver), [rawData, driver]);

  const kpis = useMemo(() => {
    if (!driverData.length) return null;
    const speeds = driverData.map(r => Number(r.Speed) || 0);
    const rpms = driverData.map(r => Number(r.RPM) || 0);
    const drsCount = driverData.filter(r => Number(r.DRS) >= 10).length;
    const compounds = [...new Set(driverData.map(r => r.Compound).filter(Boolean))];
    return {
      topSpeed: Math.max(...speeds).toFixed(0),
      avgRPM: (rpms.reduce((a, b) => a + b, 0) / rpms.length).toFixed(0),
      drsActivations: drsCount,
      compounds,
    };
  }, [driverData]);

  const speedTrace = useMemo(() => {
    if (!driverData.length) return [];
    return downsample(driverData).map(r => ({
      dist: (Number(r.Distance) / 1000).toFixed(2),
      speed: Number(r.Speed) || 0,
      rpm: Number(r.RPM) || 0,
      gear: Number(r.nGear) || 0,
      throttle: Number(r.Throttle) || 0,
      brake: r.Brake === true || r.Brake === 'True' || r.Brake === '1' || r.Brake === 1 ? 100 : 0,
    }));
  }, [driverData]);

  const lapTimes = useMemo(() => buildLapTimes(driverData), [driverData]);

  const drsPerLap = useMemo(() => {
    if (!driverData.length) return [];
    const lapDrs = new Map<number, number>();
    for (const r of driverData) {
      const lap = Number(r.LapNumber);
      if (!lap) continue;
      if (Number(r.DRS) >= 10) lapDrs.set(lap, (lapDrs.get(lap) || 0) + 1);
    }
    return Array.from(lapDrs.entries()).map(([lap, count]) => ({ lap, drs: count })).sort((a, b) => a.lap - b.lap);
  }, [driverData]);

  const raceStints = useMemo(() => {
    if (!stintsData.length) return [];
    return stintsData
      .filter(r => {
        const matchDriver = r.driver_acronym === driver;
        const matchRace = r.meeting_name?.includes(race);
        const matchYear = String(r.year) === String(year);
        const isRace = r.session_name === 'Race' || r.session_type === 'Race';
        return matchDriver && matchRace && matchYear && isRace;
      })
      .map(r => ({
        stint: Number(r.stint_number),
        compound: r.compound || '',
        lapStart: Number(r.lap_start),
        lapEnd: Number(r.lap_end),
        stintLaps: Number(r.stint_laps) || (Number(r.lap_end) - Number(r.lap_start)),
        tyreAge: Number(r.tyre_age_at_start) || 0,
      }))
      .sort((a, b) => a.stint - b.stint);
  }, [stintsData, race, year, driver]);

  const topSpeedPerLap = useMemo(() => {
    if (!driverData.length) return [];
    const lapMax = new Map<number, number>();
    for (const r of driverData) {
      const lap = Number(r.LapNumber);
      const spd = Number(r.Speed);
      if (!lap || !spd) continue;
      const prev = lapMax.get(lap) ?? 0;
      if (spd > prev) lapMax.set(lap, spd);
    }
    return Array.from(lapMax.entries()).map(([lap, speed]) => ({ lap, speed })).sort((a, b) => a.lap - b.lap);
  }, [driverData]);

  const highlightRace = race;

  const driverInfo = DRIVER_INFO[driver];

  return (
    <div className="space-y-4">
      {/* ── Hero Header ── */}
      <div className="relative bg-card border border-[rgba(255,128,0,0.15)] rounded-lg overflow-hidden">
        <div className="absolute inset-0 opacity-[0.07]" style={{ background: 'linear-gradient(135deg, #FF8000 0%, transparent 60%)' }} />
        <div className="absolute top-0 left-0 right-0 h-[2px]" style={{ background: 'linear-gradient(90deg, #FF8000, rgba(255,128,0,0.4), transparent)' }} />
        <div className="relative flex items-center gap-4 p-4">
          <img src={MCLAREN_LOGO} alt="McLaren" className="h-8 w-8 object-contain opacity-80" />
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-lg font-bold text-foreground">{driverInfo?.name || driver}</span>
              {driverInfo && <span className="text-base">{driverInfo.flag}</span>}
              <span className="text-[11px] text-primary/60 font-mono">#{driverInfo?.number || ''}</span>
            </div>
            <div className="flex items-center gap-2 text-[12px] text-muted-foreground">
              <span className="text-primary font-semibold">McLaren</span>
              <span className="opacity-40">|</span>
              <span>{tab === 'detail' && race ? `${race} GP ${year}` : `${year} Season`}</span>
              {tab === 'detail' && driverData.length > 0 && (
                <><span className="opacity-40">|</span><span>{driverData.length.toLocaleString()} pts</span></>
              )}
              {tab === 'season' && seasonSummary.length > 0 && (
                <><span className="opacity-40">|</span><span>{seasonSummary.length} races</span></>
              )}
            </div>
          </div>
          {/* Anomaly health gauge */}
          {selectedVehicle && (
            <div className="flex items-center gap-3">
              <HealthGauge value={selectedVehicle.overallHealth} size={44} showLabel={false} />
              <div className="flex flex-col items-end gap-0.5">
                <StatusBadge status={selectedVehicle.level} size="sm" />
                <span className="text-[10px] text-muted-foreground">{selectedVehicle.overallHealth.toFixed(0)}% health</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── Tab Switcher + Controls ── */}
      <div className="flex items-center gap-3 flex-wrap">
        <div className="flex items-center gap-1 bg-card rounded-lg p-0.5 border border-border">
          {([
            { id: 'detail' as Tab, label: 'Race Detail', icon: Activity },
            { id: 'season' as Tab, label: 'Season', icon: TrendingUp },
            { id: 'h2h' as Tab, label: 'Head to Head', icon: GitCompareArrows },
            { id: 'racecompare' as Tab, label: 'Compare', icon: Disc },
          ]).map(t => (
            <button type="button" key={t.id} onClick={() => setTab(t.id)}
              className={`text-sm px-4 py-1.5 rounded-md transition-all flex items-center gap-1.5 ${
                tab === t.id ? 'bg-primary/10 text-primary' : 'text-muted-foreground hover:text-foreground'
              }`}
            ><t.icon className="w-3.5 h-3.5" />{t.label}</button>
          ))}
        </div>
        {/* Year */}
        {tab !== 'racecompare' || compareMode !== 'year' ? (
          <select value={year} onChange={e => { const y = Number(e.target.value); setYear(y); const yr = meta?.races_by_year[String(y)] || []; if (yr.length) { setRace(yr[0]); if (yr.length > 1) setRace2(yr[1]); } }} aria-label="Select year"
            className="bg-card border border-border rounded-lg text-sm text-foreground px-3 py-1.5 outline-none"
          >
            {years.map(y => <option key={y} value={y}>{y}</option>)}
          </select>
        ) : null}
        {/* Race A */}
        {(tab === 'detail' || (tab === 'h2h' && h2hMode === 'race') || (tab === 'racecompare' && compareMode === 'race')) && (
          <select value={race} onChange={e => setRace(e.target.value)} aria-label="Select race"
            className="bg-card border border-border rounded-lg text-sm text-foreground px-3 py-1.5 outline-none"
          >
            {races.map(r => <option key={r} value={r}>{r} GP</option>)}
          </select>
        )}
        {/* Race B */}
        {tab === 'racecompare' && compareMode === 'race' && (
          <>
            <span className="text-[12px] text-muted-foreground">vs</span>
            <select value={race2} onChange={e => setRace2(e.target.value)} aria-label="Select second race"
              className="bg-card border border-border rounded-lg text-sm text-foreground px-3 py-1.5 outline-none"
            >
              {races.map(r => <option key={r} value={r}>{r} GP</option>)}
            </select>
          </>
        )}
        {/* Driver pills */}
        {tab !== 'h2h' && drivers.length > 1 && (
          <div className="flex items-center gap-1.5">
            {drivers.map(d => (
              <button type="button" key={d} onClick={() => setDriver(d)}
                className={`px-3 py-1 rounded-full text-sm font-semibold transition-all border ${
                  driver === d
                    ? 'bg-primary text-white border-primary shadow-[0_0_12px_rgba(255,128,0,0.3)]'
                    : 'bg-transparent text-primary border-primary/30 hover:border-primary/60'
                }`}
              >{d}</button>
            ))}
          </div>
        )}
        {/* H2H driver pickers + sub-mode */}
        {tab === 'h2h' && (
          <>
            <div className="flex items-center gap-1.5">
              {drivers.map(d => (
                <button type="button" key={`d1-${d}`} onClick={() => setDriver(d)}
                  className={`px-3 py-1 rounded-full text-sm font-semibold transition-all border ${
                    driver === d
                      ? 'bg-primary text-white border-primary shadow-[0_0_12px_rgba(255,128,0,0.3)]'
                      : 'bg-transparent text-primary border-primary/30 hover:border-primary/60'
                  }`}
                >{d}</button>
              ))}
            </div>
            <span className="text-[12px] text-muted-foreground font-semibold">vs</span>
            <div className="flex items-center gap-1.5">
              {drivers.map(d => (
                <button type="button" key={`d2-${d}`} onClick={() => setH2hDriver2(d)}
                  className={`px-3 py-1 rounded-full text-sm font-semibold transition-all border ${
                    h2hDriver2 === d
                      ? 'bg-[#22d3ee] text-white border-[#22d3ee] shadow-[0_0_12px_rgba(34,211,238,0.3)]'
                      : 'bg-transparent text-[#22d3ee] border-[#22d3ee]/30 hover:border-[#22d3ee]/60'
                  }`}
                >{d}</button>
              ))}
            </div>
            <div className="flex items-center gap-1 bg-card rounded-lg p-0.5 border border-border">
              {([{ id: 'race' as const, label: 'Single Race' }, { id: 'season' as const, label: 'Full Season' }]).map(m => (
                <button type="button" key={m.id} onClick={() => setH2hMode(m.id)}
                  className={`text-sm px-3 py-1.5 rounded-md transition-all ${h2hMode === m.id ? 'bg-primary/10 text-primary' : 'text-muted-foreground hover:text-foreground'}`}
                >{m.label}</button>
              ))}
            </div>
          </>
        )}
        {/* Compare sub-mode */}
        {tab === 'racecompare' && (
          <div className="flex items-center gap-1 bg-card rounded-lg p-0.5 border border-border">
            {([{ id: 'race' as const, label: 'Race vs Race' }, { id: 'year' as const, label: 'Year vs Year' }]).map(m => (
              <button type="button" key={m.id} onClick={() => setCompareMode(m.id)}
                className={`text-sm px-3 py-1.5 rounded-md transition-all ${compareMode === m.id ? 'bg-primary/10 text-primary' : 'text-muted-foreground hover:text-foreground'}`}
              >{m.label}</button>
            ))}
          </div>
        )}
      </div>

      {/* ── System Health Compact Bar ── */}
      {selectedVehicle && selectedVehicle.systems.length > 0 && (tab === 'detail' || tab === 'season') && (
        <div className="flex items-center gap-4 bg-card border border-border rounded-lg px-4 py-2">
          <span className="text-[10px] tracking-[0.2em] text-primary/50 font-semibold">SYSTEMS</span>
          {selectedVehicle.systems.slice(0, 4).map(sys => (
            <div key={sys.name} className="flex items-center gap-2">
              <span className="text-[11px] text-muted-foreground">{sys.name}</span>
              <div className="w-16 h-1.5 bg-secondary rounded-full overflow-hidden">
                <div className="h-full rounded-full transition-all" style={{
                  width: `${sys.health}%`,
                  backgroundColor: sys.level === 'nominal' ? '#22c55e' : sys.level === 'warning' ? '#FF8000' : '#ef4444',
                }} />
              </div>
              <span className="text-[10px] font-mono text-muted-foreground">{sys.health}%</span>
            </div>
          ))}
        </div>
      )}

      {error && tab === 'detail' && (
        <div className="flex items-center gap-2 text-amber-400 text-sm bg-amber-500/10 rounded-lg p-3">
          <AlertCircle className="w-4 h-4" />{error}
        </div>
      )}

      {/* ── Tab Content with Transitions ── */}
      <AnimatePresence mode="wait">
        <motion.div key={tab} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.2 }}>
          {tab === 'detail' && (
            <>
              <RaceDetailView kpis={kpis} speedTrace={speedTrace} lapTimes={lapTimes}
                drsPerLap={drsPerLap} raceStints={raceStints} topSpeedPerLap={topSpeedPerLap} race={race} year={year} loading={loading} />
              {!loading && driverData.length > 0 && (
                <div className="mt-4">
                  <KexBriefingCard title="WISE Race Analysis" icon="brain" kex={kex} loading={kexLoading} loadingText="Analyzing race telemetry..." />
                </div>
              )}
            </>
          )}
          {tab === 'season' && (
            <>
              <SeasonCompareView seasonSummary={seasonSummary} loading={summaryLoading}
                year={year} driver={driver} highlightRace={highlightRace} />
              {!summaryLoading && seasonSummary.length > 0 && (
                <div className="mt-4">
                  <KexBriefingCard title="WISE Season Analysis" icon="sparkles" kex={kex} loading={kexLoading} loadingText="Analyzing season telemetry..." />
                </div>
              )}
            </>
          )}
          {tab === 'h2h' && (
            <H2HView mode={h2hMode} loading={h2hLoading} year={year} race={race}
              driver1={driver} driver2={h2hDriver2}
              raceData={h2hRaceData} season1={h2hSeason1} season2={h2hSeason2} />
          )}
          {tab === 'racecompare' && (
            <CompareView mode={compareMode} loading={compareLoading} year={year} year2={year2} driver={driver}
              race1={race} race2={race2} data1={compareData1} data2={compareData2}
              season1={compareSeason1} season2={compareSeason2} years={years} onYear2Change={setYear2} />
          )}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}

/* ─── Race Detail View ─── */

function RaceDetailView({ kpis, speedTrace, lapTimes, drsPerLap, raceStints, topSpeedPerLap, race, year, loading }: {
  kpis: { topSpeed: string; avgRPM: string; drsActivations: number; compounds: string[] } | null;
  speedTrace: { dist: string; speed: number; rpm: number; gear: number; throttle: number; brake: number }[];
  lapTimes: { lap: number; time: number; compound: string }[];
  drsPerLap: { lap: number; drs: number }[];
  raceStints: { stint: number; compound: string; lapStart: number; lapEnd: number; stintLaps: number; tyreAge: number }[];
  topSpeedPerLap: { lap: number; speed: number }[];
  race: string;
  year: number;
  loading: boolean;
}) {
  if (loading) {
    return (
      <div className="flex items-center justify-center h-[400px]">
        <Loader2 className="w-6 h-6 text-primary animate-spin" />
        <span className="ml-2 text-sm text-muted-foreground">Loading telemetry...</span>
      </div>
    );
  }

  return (
    <>
      {kpis && (
        <div className="grid grid-cols-4 gap-3">
          <KPI icon={<Gauge className="w-4 h-4 text-primary" />} label="TOP SPEED" value={`${kpis.topSpeed} km/h`} detail={`${race} GP ${year}`} color="text-primary" />
          <KPI icon={<Zap className="w-4 h-4 text-cyan-400" />} label="AVG RPM" value={kpis.avgRPM} detail="Engine average" color="text-cyan-400" />
          <KPI icon={<Wind className="w-4 h-4 text-green-400" />} label="DRS ACTIVATIONS" value={String(kpis.drsActivations)} detail="Total samples with DRS open" color="text-green-400" />
          <KPI icon={<Timer className="w-4 h-4 text-amber-400" />} label="TIRE COMPOUNDS" value={kpis.compounds.join(' / ')} detail={`${raceStints.length} stints`} color="text-amber-400" />
        </div>
      )}

      {speedTrace.length > 0 && (
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-card border border-border rounded-lg shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
            <h3 className="text-sm text-foreground mb-1">Speed Trace</h3>
            <p className="text-[12px] text-muted-foreground mb-3">Speed (km/h) over distance</p>
            <div className="h-[220px]">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={speedTrace}>
                  <defs>
                    <linearGradient id="speedGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#FF8000" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#FF8000" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="dist" tick={{ fill: '#8888a0', fontSize: 9 }} tickCount={8} />
                  <YAxis tick={{ fill: '#8888a0', fontSize: 9 }} domain={[0, 'auto']} />
                  <Tooltip content={<CustomTooltip />} />
                  <Area type="monotone" dataKey="speed" stroke="#FF8000" fill="url(#speedGrad)" strokeWidth={1.5} dot={false} name="Speed (km/h)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div className="bg-card border border-border rounded-lg shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
            <h3 className="text-sm text-foreground mb-1">RPM & Gear</h3>
            <p className="text-[12px] text-muted-foreground mb-3">Engine RPM and gear selection</p>
            <div className="h-[220px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={speedTrace}>
                  <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="dist" tick={{ fill: '#8888a0', fontSize: 9 }} tickCount={8} />
                  <YAxis yAxisId="rpm" tick={{ fill: '#8888a0', fontSize: 9 }} />
                  <YAxis yAxisId="gear" orientation="right" domain={[0, 8]} tick={{ fill: '#8888a0', fontSize: 9 }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Line yAxisId="rpm" type="monotone" dataKey="rpm" stroke="#22d3ee" strokeWidth={1} dot={false} name="RPM" />
                  <Line yAxisId="gear" type="stepAfter" dataKey="gear" stroke="#a78bfa" strokeWidth={1.5} dot={false} name="Gear" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {speedTrace.length > 0 && (
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-card border border-border rounded-lg shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
            <h3 className="text-sm text-foreground mb-1">Throttle & Brake</h3>
            <p className="text-[12px] text-muted-foreground mb-3">Throttle application (%) and brake zones</p>
            <div className="h-[220px]">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={speedTrace}>
                  <defs>
                    <linearGradient id="throttleGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#22c55e" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#22c55e" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="dist" tick={{ fill: '#8888a0', fontSize: 9 }} tickCount={8} />
                  <YAxis tick={{ fill: '#8888a0', fontSize: 9 }} domain={[0, 100]} />
                  <Tooltip content={<CustomTooltip />} />
                  <Area type="monotone" dataKey="throttle" stroke="#22c55e" fill="url(#throttleGrad)" strokeWidth={1} dot={false} name="Throttle %" />
                  <Area type="monotone" dataKey="brake" stroke="#ef4444" fill="#ef444420" strokeWidth={1} dot={false} name="Brake" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
          {lapTimes.length > 0 && (
            <div className="bg-card border border-border rounded-lg shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
              <h3 className="text-sm text-foreground mb-1">Lap Time Progression</h3>
              <p className="text-[12px] text-muted-foreground mb-3">Lap time by compound</p>
              <div className="h-[220px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={lapTimes}>
                    <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="lap" tick={{ fill: '#8888a0', fontSize: 9 }} />
                    <YAxis tick={{ fill: '#8888a0', fontSize: 9 }} domain={['auto', 'auto']} />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey="time" name="Lap Time (s)" radius={[2, 2, 0, 0]}>
                      {lapTimes.map((entry, i) => (
                        <Cell key={i} fill={compoundColors[entry.compound] || '#8888a0'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </div>
      )}

      <div className="grid grid-cols-2 gap-3">
        {drsPerLap.length > 0 && (
          <div className="bg-card border border-border rounded-lg shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
            <h3 className="text-sm text-foreground mb-1">DRS Usage by Lap</h3>
            <p className="text-[12px] text-muted-foreground mb-3">DRS activation samples per lap</p>
            <div className="h-[200px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={drsPerLap}>
                  <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="lap" tick={{ fill: '#8888a0', fontSize: 9 }} />
                  <YAxis tick={{ fill: '#8888a0', fontSize: 9 }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="drs" fill="#22c55e" name="DRS Activations" radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
        {raceStints.length > 0 && (
          <div className="bg-card border border-border rounded-lg shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
            <h3 className="text-sm text-foreground mb-1">Tire Strategy</h3>
            <p className="text-[12px] text-muted-foreground mb-3">Stint breakdown — compound, laps & age</p>
            <div className="space-y-2">
              {raceStints.map((s, i) => (
                <div key={i} className="flex items-center gap-3">
                  <span className="text-[12px] text-muted-foreground w-12">Stint {s.stint}</span>
                  <div className="flex-1 h-6 bg-secondary rounded-md overflow-hidden relative">
                    <div className="h-full rounded-md flex items-center px-2"
                      style={{ backgroundColor: `${compoundColors[s.compound] || '#8888a0'}30`, borderLeft: `3px solid ${compoundColors[s.compound] || '#8888a0'}`, width: `${Math.min(100, (s.stintLaps / 30) * 100)}%` }}
                    >
                      <span className="text-[12px] font-mono text-foreground">{s.compound}</span>
                    </div>
                  </div>
                  <span className="text-[12px] text-muted-foreground w-20 text-right">L{s.lapStart}–{s.lapEnd}</span>
                  <span className="text-[12px] font-mono text-foreground w-16 text-right">{s.stintLaps} laps</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Speed Trap — top speed per lap */}
      {topSpeedPerLap.length > 0 && (
        <div className="bg-card border border-border rounded-lg shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
          <h3 className="text-sm text-foreground mb-1">Speed Trap</h3>
          <p className="text-[12px] text-muted-foreground mb-3">Max speed recorded per lap (km/h)</p>
          <div className="h-[200px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={topSpeedPerLap}>
                <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="lap" tick={{ fill: '#8888a0', fontSize: 9 }} tickFormatter={(v) => `L${v}`} />
                <YAxis tick={{ fill: '#8888a0', fontSize: 9 }} domain={['dataMin - 10', 'dataMax + 5']} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="speed" fill="#FF8000" name="Top Speed" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </>
  );
}

/* ─── Season Compare View ─── */

function SeasonCompareView({ seasonSummary, loading, year, driver, highlightRace }: {
  seasonSummary: CarSummary[]; loading: boolean; year: number; driver: string; highlightRace: string;
}) {
  if (loading) {
    return <div className="flex items-center justify-center h-[400px]"><Loader2 className="w-6 h-6 text-primary animate-spin" /><span className="ml-2 text-sm text-muted-foreground">Computing season telemetry summary...</span></div>;
  }
  if (!seasonSummary.length) {
    return <div className="flex items-center justify-center h-[400px] text-muted-foreground text-sm">No telemetry data available for {year}</div>;
  }

  const seasonAvgSpeed = seasonSummary.reduce((s, r) => s + r.avgSpeed, 0) / seasonSummary.length;
  const seasonTopSpeed = Math.max(...seasonSummary.map(r => r.topSpeed));
  const seasonAvgRPM = seasonSummary.reduce((s, r) => s + r.avgRPM, 0) / seasonSummary.length;
  const seasonAvgThrottle = seasonSummary.reduce((s, r) => s + r.avgThrottle, 0) / seasonSummary.length;
  const fastestRace = seasonSummary.reduce((best, r) => r.topSpeed > best.topSpeed ? r : best, seasonSummary[0]);
  const heaviestBraking = seasonSummary.reduce((best, r) => r.brakePct > best.brakePct ? r : best, seasonSummary[0]);

  return (
    <>
      <div className="grid grid-cols-6 gap-3">
        <KPI icon={<Gauge className="w-4 h-4 text-primary" />} label="SEASON TOP SPEED" value={`${seasonTopSpeed.toFixed(0)} km/h`} detail={`${fastestRace.race} GP`} color="text-primary" />
        <KPI icon={<Gauge className="w-4 h-4 text-cyan-400" />} label="SEASON AVG SPEED" value={`${seasonAvgSpeed.toFixed(1)} km/h`} detail={`Across ${seasonSummary.length} races`} color="text-cyan-400" />
        <KPI icon={<Zap className="w-4 h-4 text-purple-400" />} label="SEASON AVG RPM" value={seasonAvgRPM.toFixed(0)} detail="Engine average" color="text-purple-400" />
        <KPI icon={<Wind className="w-4 h-4 text-green-400" />} label="AVG THROTTLE" value={`${seasonAvgThrottle.toFixed(1)}%`} detail="Season average" color="text-green-400" />
        <KPI icon={<Disc className="w-4 h-4 text-red-400" />} label="HEAVIEST BRAKING" value={`${heaviestBraking.brakePct.toFixed(1)}%`} detail={`${heaviestBraking.race} GP`} color="text-red-400" />
        <KPI icon={<GitCompareArrows className="w-4 h-4 text-amber-400" />} label="FASTEST CIRCUIT" value={fastestRace.race} detail={`${fastestRace.topSpeed.toFixed(0)} km/h top`} color="text-amber-400" />
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="bg-card border border-border rounded-lg shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
          <h3 className="text-sm text-foreground mb-1">Top Speed by Race</h3>
          <p className="text-[12px] text-muted-foreground mb-3">{driver} — {year} season</p>
          <div className="h-[260px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={seasonSummary}>
                <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="race" tick={{ fill: '#8888a0', fontSize: 8 }} angle={-45} textAnchor="end" height={60} interval={0} />
                <YAxis tick={{ fill: '#8888a0', fontSize: 9 }} domain={['auto', 'auto']} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="topSpeed" name="Top Speed (km/h)" radius={[2, 2, 0, 0]}>
                  {seasonSummary.map((entry, i) => <Cell key={i} fill={entry.race === highlightRace ? '#FF8000' : '#FF800080'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div className="bg-card border border-border rounded-lg shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
          <h3 className="text-sm text-foreground mb-1">Average Speed by Race</h3>
          <p className="text-[12px] text-muted-foreground mb-3">Mean speed across all telemetry samples</p>
          <div className="h-[260px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={seasonSummary}>
                <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="race" tick={{ fill: '#8888a0', fontSize: 8 }} angle={-45} textAnchor="end" height={60} interval={0} />
                <YAxis tick={{ fill: '#8888a0', fontSize: 9 }} domain={['auto', 'auto']} />
                <Tooltip content={<CustomTooltip />} />
                <ReferenceLine y={seasonAvgSpeed} stroke="#22d3ee" strokeDasharray="4 2" strokeOpacity={0.6} />
                <Bar dataKey="avgSpeed" name="Avg Speed (km/h)" radius={[2, 2, 0, 0]}>
                  {seasonSummary.map((entry, i) => <Cell key={i} fill={entry.race === highlightRace ? '#22d3ee' : '#22d3ee80'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="bg-card border border-border rounded-lg shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
          <h3 className="text-sm text-foreground mb-1">Throttle vs Brake by Race</h3>
          <p className="text-[12px] text-muted-foreground mb-3">Average throttle % and braking %</p>
          <div className="h-[260px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={seasonSummary}>
                <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="race" tick={{ fill: '#8888a0', fontSize: 8 }} angle={-45} textAnchor="end" height={60} interval={0} />
                <YAxis tick={{ fill: '#8888a0', fontSize: 9 }} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="avgThrottle" name="Throttle %" fill="#22c55e" radius={[2, 2, 0, 0]} />
                <Bar dataKey="brakePct" name="Brake %" fill="#ef4444" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div className="bg-card border border-border rounded-lg shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
          <h3 className="text-sm text-foreground mb-1">DRS Usage by Race</h3>
          <p className="text-[12px] text-muted-foreground mb-3">% of telemetry samples with DRS open</p>
          <div className="h-[260px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={seasonSummary}>
                <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="race" tick={{ fill: '#8888a0', fontSize: 8 }} angle={-45} textAnchor="end" height={60} interval={0} />
                <YAxis tick={{ fill: '#8888a0', fontSize: 9 }} domain={[0, 'auto']} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="drsPct" name="DRS %" radius={[2, 2, 0, 0]}>
                  {seasonSummary.map((entry, i) => <Cell key={i} fill={entry.race === highlightRace ? '#22c55e' : '#22c55e80'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Season Summary Table */}
      <div className="bg-card border border-border rounded-lg shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
        <h3 className="text-sm text-foreground mb-3 flex items-center gap-2">
          <GitCompareArrows className="w-3 h-3 text-primary" />{driver} — {year} Season Telemetry Summary
        </h3>
        <div className="overflow-x-auto">
          <div className="grid grid-cols-[160px_80px_80px_80px_80px_80px_80px_80px_120px] gap-px bg-[rgba(255,128,0,0.12)] rounded-lg overflow-hidden min-w-[840px]">
            {['Race', 'Avg Spd', 'Top Spd', 'Avg RPM', 'Max RPM', 'Throttle', 'Brake %', 'DRS %', 'Compounds'].map(h => (
              <div key={h} className="bg-secondary px-3 py-2 text-[11px] text-muted-foreground tracking-wider">{h}</div>
            ))}
            {seasonSummary.map((r, i) => (
              <React.Fragment key={i}>
                <div className={`px-3 py-1.5 text-sm ${r.race === highlightRace ? 'bg-primary/10 text-primary' : 'bg-card text-foreground'}`}>{r.race} GP</div>
                <div className={`px-3 py-1.5 text-sm font-mono ${r.race === highlightRace ? 'bg-primary/10' : 'bg-card'} text-foreground`}>{r.avgSpeed.toFixed(1)}</div>
                <div className={`px-3 py-1.5 text-sm font-mono ${r.race === highlightRace ? 'bg-primary/10' : 'bg-card'} ${r.topSpeed >= seasonTopSpeed - 5 ? 'text-primary' : 'text-foreground'}`}>{r.topSpeed.toFixed(0)}</div>
                <div className={`px-3 py-1.5 text-sm font-mono ${r.race === highlightRace ? 'bg-primary/10' : 'bg-card'} text-foreground`}>{r.avgRPM}</div>
                <div className={`px-3 py-1.5 text-sm font-mono ${r.race === highlightRace ? 'bg-primary/10' : 'bg-card'} text-foreground`}>{r.maxRPM}</div>
                <div className={`px-3 py-1.5 text-sm font-mono ${r.race === highlightRace ? 'bg-primary/10' : 'bg-card'} text-green-400`}>{r.avgThrottle.toFixed(1)}%</div>
                <div className={`px-3 py-1.5 text-sm font-mono ${r.race === highlightRace ? 'bg-primary/10' : 'bg-card'} ${r.brakePct > 15 ? 'text-red-400' : 'text-foreground'}`}>{r.brakePct.toFixed(1)}</div>
                <div className={`px-3 py-1.5 text-sm font-mono ${r.race === highlightRace ? 'bg-primary/10' : 'bg-card'} text-foreground`}>{r.drsPct.toFixed(1)}</div>
                <div className={`px-3 py-1.5 text-[12px] ${r.race === highlightRace ? 'bg-primary/10' : 'bg-card'}`}>
                  {r.compounds.map((c, ci) => <span key={ci} className="font-mono mr-1" style={{ color: compoundColors[c] || '#8888a0' }}>{c}</span>)}
                </div>
              </React.Fragment>
            ))}
          </div>
        </div>
      </div>
    </>
  );
}

/* ─── Head-to-Head View ─── */

function H2HView({ mode, loading, year, race, driver1, driver2, raceData, season1, season2 }: {
  mode: 'race' | 'season'; loading: boolean; year: number; race: string;
  driver1: string; driver2: string;
  raceData: Record<string, any>[]; season1: CarSummary[]; season2: CarSummary[];
}) {
  if (loading) {
    return <div className="flex items-center justify-center h-[400px]"><Loader2 className="w-6 h-6 text-primary animate-spin" /><span className="ml-2 text-sm text-muted-foreground">Loading head-to-head data...</span></div>;
  }
  if (mode === 'race') return <H2HRaceView raceData={raceData} race={race} year={year} driver1={driver1} driver2={driver2} />;
  return <H2HSeasonView season1={season1} season2={season2} year={year} driver1={driver1} driver2={driver2} />;
}

function H2HRaceView({ raceData, race, year, driver1, driver2 }: { raceData: Record<string, any>[]; race: string; year: number; driver1: string; driver2: string }) {
  const D1_COLOR = '#FF8000';
  const D2_COLOR = '#22d3ee';
  const d1Data = useMemo(() => raceData.filter(r => r.Driver === driver1), [raceData, driver1]);
  const d2Data = useMemo(() => raceData.filter(r => r.Driver === driver2), [raceData, driver2]);
  const d1Kpis = useMemo(() => computeCarKpis(d1Data), [d1Data]);
  const d2Kpis = useMemo(() => computeCarKpis(d2Data), [d2Data]);

  const mergedSpeed = useMemo(() => {
    const d1Sampled = downsample(d1Data);
    const d2Sampled = downsample(d2Data);
    const map = new Map<string, { dist: string; d1Speed?: number; d2Speed?: number }>();
    for (const r of d1Sampled) {
      const dist = (Number(r.Distance) / 1000).toFixed(1);
      map.set(dist, { ...map.get(dist), dist, d1Speed: Number(r.Speed) || 0 });
    }
    for (const r of d2Sampled) {
      const dist = (Number(r.Distance) / 1000).toFixed(1);
      map.set(dist, { ...map.get(dist), dist, d2Speed: Number(r.Speed) || 0 });
    }
    return Array.from(map.values()).sort((a, b) => parseFloat(a.dist) - parseFloat(b.dist));
  }, [d1Data, d2Data]);

  const d1Laps = useMemo(() => buildLapTimes(d1Data), [d1Data]);
  const d2Laps = useMemo(() => buildLapTimes(d2Data), [d2Data]);
  const mergedLapTimes = useMemo(() => {
    const map = new Map<number, { lap: number; d1Time?: number; d2Time?: number }>();
    for (const d of d1Laps) map.set(d.lap, { ...map.get(d.lap), lap: d.lap, d1Time: d.time });
    for (const d of d2Laps) map.set(d.lap, { ...map.get(d.lap), lap: d.lap, d2Time: d.time });
    return Array.from(map.values()).sort((a, b) => a.lap - b.lap);
  }, [d1Laps, d2Laps]);

  if (!d1Data.length && !d2Data.length) {
    return <div className="flex items-center justify-center h-[300px] text-muted-foreground text-sm">No data available for {race} GP {year}</div>;
  }

  return (
    <>
      <div className="grid grid-cols-4 gap-3">
        <DeltaKPI label="TOP SPEED" val1={d1Kpis?.topSpeed || '—'} val2={d2Kpis?.topSpeed || '—'} label1={driver1} label2={driver2} unit=" km/h"
          icon={<Gauge className="w-4 h-4 text-primary" />} color1={D1_COLOR} color2={D2_COLOR} />
        <DeltaKPI label="AVG RPM" val1={d1Kpis?.avgRPM || '—'} val2={d2Kpis?.avgRPM || '—'} label1={driver1} label2={driver2} unit=""
          icon={<Zap className="w-4 h-4 text-cyan-400" />} color1={D1_COLOR} color2={D2_COLOR} />
        <DeltaKPI label="AVG THROTTLE" val1={d1Kpis?.avgThrottle || '—'} val2={d2Kpis?.avgThrottle || '—'} label1={driver1} label2={driver2} unit="%"
          icon={<Wind className="w-4 h-4 text-green-400" />} color1={D1_COLOR} color2={D2_COLOR} />
        <DeltaKPI label="DRS USAGE" val1={d1Kpis?.drsPct || '—'} val2={d2Kpis?.drsPct || '—'} label1={driver1} label2={driver2} unit="%"
          icon={<Wind className="w-4 h-4 text-green-400" />} color1={D1_COLOR} color2={D2_COLOR} />
      </div>

      {mergedSpeed.length > 0 && (
        <div className="bg-card border border-border rounded-lg shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
          <h3 className="text-sm text-foreground mb-1">Speed Trace Overlay — {race} GP {year}</h3>
          <p className="text-[12px] text-muted-foreground mb-3">
            <span style={{ color: D1_COLOR }}>■</span> {driver1} vs <span style={{ color: D2_COLOR }}>■</span> {driver2} — km/h over distance
          </p>
          <div className="h-[280px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={mergedSpeed}>
                <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="dist" tick={{ fill: '#8888a0', fontSize: 9 }} tickCount={10} />
                <YAxis tick={{ fill: '#8888a0', fontSize: 9 }} domain={[0, 'auto']} />
                <Tooltip content={<CustomTooltip />} />
                <Line type="monotone" dataKey="d1Speed" stroke={D1_COLOR} strokeWidth={1.5} dot={false} name={`${driver1} Speed`} />
                <Line type="monotone" dataKey="d2Speed" stroke={D2_COLOR} strokeWidth={1.5} dot={false} name={`${driver2} Speed`} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {mergedLapTimes.length > 0 && (
        <div className="bg-card border border-border rounded-lg shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
          <h3 className="text-sm text-foreground mb-1">Lap Times — {driver1} vs {driver2}</h3>
          <p className="text-[12px] text-muted-foreground mb-3">Lap time (seconds) per lap</p>
          <div className="h-[260px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={mergedLapTimes}>
                <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="lap" tick={{ fill: '#8888a0', fontSize: 9 }} />
                <YAxis tick={{ fill: '#8888a0', fontSize: 9 }} domain={['auto', 'auto']} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="d1Time" name={`${driver1} (s)`} fill={D1_COLOR} radius={[2, 2, 0, 0]} />
                <Bar dataKey="d2Time" name={`${driver2} (s)`} fill={D2_COLOR} radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </>
  );
}

function H2HSeasonView({ season1, season2, year, driver1, driver2 }: { season1: CarSummary[]; season2: CarSummary[]; year: number; driver1: string; driver2: string }) {
  const D1_COLOR = '#FF8000';
  const D2_COLOR = '#22d3ee';
  const merged = useMemo(() => {
    const allRaces = [...new Set([...season1.map(r => r.race), ...season2.map(r => r.race)])];
    return allRaces.map(race => {
      const s1 = season1.find(r => r.race === race);
      const s2 = season2.find(r => r.race === race);
      return {
        race,
        d1Top: s1?.topSpeed ?? 0, d2Top: s2?.topSpeed ?? 0,
        d1Avg: s1?.avgSpeed ?? 0, d2Avg: s2?.avgSpeed ?? 0,
        d1Throttle: s1?.avgThrottle ?? 0, d2Throttle: s2?.avgThrottle ?? 0,
        d1Brake: s1?.brakePct ?? 0, d2Brake: s2?.brakePct ?? 0,
      };
    });
  }, [season1, season2]);

  const avg = (arr: CarSummary[], key: keyof CarSummary) => arr.length ? (arr.reduce((s, r) => s + (r[key] as number), 0) / arr.length).toFixed(1) : '—';
  const topOf = (arr: CarSummary[]) => arr.length ? Math.max(...arr.map(r => r.topSpeed)).toFixed(0) : '—';

  if (!merged.length) return <div className="flex items-center justify-center h-[300px] text-muted-foreground text-sm">No season data for {year}</div>;

  return (
    <>
      <div className="grid grid-cols-4 gap-3">
        <DeltaKPI label="TOP SPEED" val1={topOf(season1)} val2={topOf(season2)} label1={driver1} label2={driver2} unit=" km/h" icon={<Gauge className="w-4 h-4 text-primary" />} color1={D1_COLOR} color2={D2_COLOR} />
        <DeltaKPI label="AVG SPEED" val1={avg(season1, 'avgSpeed')} val2={avg(season2, 'avgSpeed')} label1={driver1} label2={driver2} unit=" km/h" icon={<Gauge className="w-4 h-4 text-cyan-400" />} color1={D1_COLOR} color2={D2_COLOR} />
        <DeltaKPI label="AVG THROTTLE" val1={avg(season1, 'avgThrottle')} val2={avg(season2, 'avgThrottle')} label1={driver1} label2={driver2} unit="%" icon={<Wind className="w-4 h-4 text-green-400" />} color1={D1_COLOR} color2={D2_COLOR} />
        <DeltaKPI label="AVG BRAKE" val1={avg(season1, 'brakePct')} val2={avg(season2, 'brakePct')} label1={driver1} label2={driver2} unit="%" icon={<Disc className="w-4 h-4 text-red-400" />} color1={D1_COLOR} color2={D2_COLOR} />
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="bg-card border border-border rounded-lg shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
          <h3 className="text-sm text-foreground mb-1">Top Speed by Race — {driver1} vs {driver2}</h3>
          <p className="text-[12px] text-muted-foreground mb-3">{year} season</p>
          <div className="h-[280px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={merged}>
                <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="race" tick={{ fill: '#8888a0', fontSize: 8 }} angle={-45} textAnchor="end" height={60} interval={0} />
                <YAxis tick={{ fill: '#8888a0', fontSize: 9 }} domain={['auto', 'auto']} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="d1Top" name={driver1} fill={D1_COLOR} radius={[2, 2, 0, 0]} />
                <Bar dataKey="d2Top" name={driver2} fill={D2_COLOR} radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div className="bg-card border border-border rounded-lg shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
          <h3 className="text-sm text-foreground mb-1">Avg Speed by Race — {driver1} vs {driver2}</h3>
          <p className="text-[12px] text-muted-foreground mb-3">{year} season</p>
          <div className="h-[280px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={merged}>
                <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="race" tick={{ fill: '#8888a0', fontSize: 8 }} angle={-45} textAnchor="end" height={60} interval={0} />
                <YAxis tick={{ fill: '#8888a0', fontSize: 9 }} domain={['auto', 'auto']} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="d1Avg" name={driver1} fill={D1_COLOR} radius={[2, 2, 0, 0]} />
                <Bar dataKey="d2Avg" name={driver2} fill={D2_COLOR} radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="bg-card border border-border rounded-lg shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
          <h3 className="text-sm text-foreground mb-1">Throttle % — {driver1} vs {driver2}</h3>
          <p className="text-[12px] text-muted-foreground mb-3">{year} season</p>
          <div className="h-[280px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={merged}>
                <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="race" tick={{ fill: '#8888a0', fontSize: 8 }} angle={-45} textAnchor="end" height={60} interval={0} />
                <YAxis tick={{ fill: '#8888a0', fontSize: 9 }} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="d1Throttle" name={driver1} fill={D1_COLOR} radius={[2, 2, 0, 0]} />
                <Bar dataKey="d2Throttle" name={driver2} fill={D2_COLOR} radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div className="bg-card border border-border rounded-lg shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
          <h3 className="text-sm text-foreground mb-1">Braking % — {driver1} vs {driver2}</h3>
          <p className="text-[12px] text-muted-foreground mb-3">{year} season</p>
          <div className="h-[280px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={merged}>
                <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="race" tick={{ fill: '#8888a0', fontSize: 8 }} angle={-45} textAnchor="end" height={60} interval={0} />
                <YAxis tick={{ fill: '#8888a0', fontSize: 9 }} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="d1Brake" name={driver1} fill={D1_COLOR} radius={[2, 2, 0, 0]} />
                <Bar dataKey="d2Brake" name={driver2} fill={D2_COLOR} radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </>
  );
}

/* ─── Compare View ─── */

function CompareView({ mode, loading, year, year2, driver, race1, race2, data1, data2, season1, season2, years, onYear2Change }: {
  mode: 'race' | 'year'; loading: boolean; year: number; year2: number; driver: string;
  race1: string; race2: string; data1: Record<string, any>[]; data2: Record<string, any>[];
  season1: CarSummary[]; season2: CarSummary[]; years: number[]; onYear2Change: (y: number) => void;
}) {
  if (loading) {
    return <div className="flex items-center justify-center h-[400px]"><Loader2 className="w-6 h-6 text-primary animate-spin" /><span className="ml-2 text-sm text-muted-foreground">Loading comparison...</span></div>;
  }
  if (mode === 'race') return <RaceVsRaceView driver={driver} year={year} race1={race1} race2={race2} data1={data1} data2={data2} />;
  return <YearVsYearView driver={driver} year1={year} year2={year2} season1={season1} season2={season2} years={years} onYear2Change={onYear2Change} />;
}

function RaceVsRaceView({ driver, year, race1, race2, data1, data2 }: {
  driver: string; year: number; race1: string; race2: string;
  data1: Record<string, any>[]; data2: Record<string, any>[];
}) {
  const d1 = useMemo(() => data1.filter(r => r.Driver === driver), [data1, driver]);
  const d2 = useMemo(() => data2.filter(r => r.Driver === driver), [data2, driver]);
  const kpis1 = useMemo(() => computeCarKpis(d1), [d1]);
  const kpis2 = useMemo(() => computeCarKpis(d2), [d2]);

  const laps1 = useMemo(() => buildLapTimes(d1), [d1]);
  const laps2 = useMemo(() => buildLapTimes(d2), [d2]);
  const mergedLaps = useMemo(() => {
    const map = new Map<number, any>();
    for (const d of laps1) map.set(d.lap, { lap: d.lap, race1Time: d.time });
    for (const d of laps2) map.set(d.lap, { ...map.get(d.lap), lap: d.lap, race2Time: d.time });
    return Array.from(map.values()).sort((a: any, b: any) => a.lap - b.lap);
  }, [laps1, laps2]);

  const R1_COLOR = '#FF8000';
  const R2_COLOR = '#a78bfa';

  if (!d1.length && !d2.length) return <div className="flex items-center justify-center h-[300px] text-muted-foreground text-sm">No data available</div>;

  return (
    <>
      <div className="grid grid-cols-4 gap-3">
        <DeltaKPI label="TOP SPEED" val1={kpis1?.topSpeed || '—'} val2={kpis2?.topSpeed || '—'} label1={race1} label2={race2} unit=" km/h" icon={<Gauge className="w-4 h-4 text-primary" />} color1={R1_COLOR} color2={R2_COLOR} />
        <DeltaKPI label="AVG SPEED" val1={kpis1?.avgSpeed || '—'} val2={kpis2?.avgSpeed || '—'} label1={race1} label2={race2} unit=" km/h" icon={<Gauge className="w-4 h-4 text-cyan-400" />} color1={R1_COLOR} color2={R2_COLOR} />
        <DeltaKPI label="AVG THROTTLE" val1={kpis1?.avgThrottle || '—'} val2={kpis2?.avgThrottle || '—'} label1={race1} label2={race2} unit="%" icon={<Wind className="w-4 h-4 text-green-400" />} color1={R1_COLOR} color2={R2_COLOR} />
        <DeltaKPI label="BRAKE %" val1={kpis1?.brakePct || '—'} val2={kpis2?.brakePct || '—'} label1={race1} label2={race2} unit="%" icon={<Disc className="w-4 h-4 text-red-400" />} color1={R1_COLOR} color2={R2_COLOR} />
      </div>

      <div className="flex items-center gap-4 text-[12px] text-muted-foreground px-1">
        <span><span style={{ color: R1_COLOR }}>■</span> {race1} GP</span>
        <span><span style={{ color: R2_COLOR }}>■</span> {race2} GP</span>
      </div>

      {mergedLaps.length > 0 && (
        <div className="bg-card border border-border rounded-lg shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
          <h3 className="text-sm text-foreground mb-1">Lap Times — {race1} vs {race2}</h3>
          <p className="text-[12px] text-muted-foreground mb-3">{driver} — {year}</p>
          <div className="h-[280px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={mergedLaps}>
                <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="lap" tick={{ fill: '#8888a0', fontSize: 9 }} />
                <YAxis tick={{ fill: '#8888a0', fontSize: 9 }} domain={['auto', 'auto']} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="race1Time" name={`${race1} (s)`} fill={R1_COLOR} radius={[2, 2, 0, 0]} />
                <Bar dataKey="race2Time" name={`${race2} (s)`} fill={R2_COLOR} radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </>
  );
}

function YearVsYearView({ driver, year1, year2, season1, season2, years, onYear2Change }: {
  driver: string; year1: number; year2: number; season1: CarSummary[]; season2: CarSummary[];
  years: number[]; onYear2Change: (y: number) => void;
}) {
  const avg = (arr: CarSummary[], key: keyof CarSummary) => arr.length ? (arr.reduce((s, r) => s + (r[key] as number), 0) / arr.length).toFixed(1) : '—';
  const topOf = (arr: CarSummary[]) => arr.length ? Math.max(...arr.map(r => r.topSpeed)).toFixed(0) : '—';

  const commonRaces = useMemo(() => {
    const r2 = new Set(season2.map(r => r.race));
    return season1.filter(r => r2.has(r.race)).map(r => r.race);
  }, [season1, season2]);

  const merged = useMemo(() => commonRaces.map(race => {
    const s1 = season1.find(r => r.race === race)!;
    const s2 = season2.find(r => r.race === race)!;
    return { race, topY1: s1.topSpeed, topY2: s2.topSpeed, avgY1: s1.avgSpeed, avgY2: s2.avgSpeed, throttleY1: s1.avgThrottle, throttleY2: s2.avgThrottle, brakeY1: s1.brakePct, brakeY2: s2.brakePct };
  }), [commonRaces, season1, season2]);

  const Y1_COLOR = '#a78bfa';
  const Y2_COLOR = '#FF8000';

  if (!season1.length && !season2.length) return <div className="flex items-center justify-center h-[300px] text-muted-foreground text-sm">No season data for {driver}</div>;

  return (
    <>
      {/* Year selectors */}
      <div className="flex items-center gap-2">
        <select value={year1} disabled aria-label="Year 1" className="bg-card border border-border rounded-lg text-sm text-foreground px-3 py-1.5 outline-none opacity-60">
          <option value={year1}>{year1}</option>
        </select>
        <span className="text-[12px] text-muted-foreground">vs</span>
        <select value={year2} onChange={e => onYear2Change(Number(e.target.value))} aria-label="Year 2"
          className="bg-card border border-border rounded-lg text-sm text-foreground px-3 py-1.5 outline-none"
        >
          {years.filter(y => y !== year1).map(y => <option key={y} value={y}>{y}</option>)}
        </select>
      </div>

      <div className="grid grid-cols-4 gap-3">
        <DeltaKPI label="TOP SPEED" val1={topOf(season1)} val2={topOf(season2)} label1={String(year1)} label2={String(year2)} unit=" km/h" icon={<Gauge className="w-4 h-4 text-primary" />} color1={Y1_COLOR} color2={Y2_COLOR} />
        <DeltaKPI label="AVG SPEED" val1={avg(season1, 'avgSpeed')} val2={avg(season2, 'avgSpeed')} label1={String(year1)} label2={String(year2)} unit=" km/h" icon={<Gauge className="w-4 h-4 text-cyan-400" />} color1={Y1_COLOR} color2={Y2_COLOR} />
        <DeltaKPI label="AVG THROTTLE" val1={avg(season1, 'avgThrottle')} val2={avg(season2, 'avgThrottle')} label1={String(year1)} label2={String(year2)} unit="%" icon={<Wind className="w-4 h-4 text-green-400" />} color1={Y1_COLOR} color2={Y2_COLOR} />
        <DeltaKPI label="AVG BRAKE" val1={avg(season1, 'brakePct')} val2={avg(season2, 'brakePct')} label1={String(year1)} label2={String(year2)} unit="%" icon={<Disc className="w-4 h-4 text-red-400" />} color1={Y1_COLOR} color2={Y2_COLOR} />
      </div>

      <div className="flex items-center gap-4 text-[12px] text-muted-foreground px-1">
        <span><span style={{ color: Y1_COLOR }}>■</span> {year1}</span>
        <span><span style={{ color: Y2_COLOR }}>■</span> {year2}</span>
        <span className="ml-auto">{commonRaces.length} common races</span>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="bg-card border border-border rounded-lg shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
          <h3 className="text-sm text-foreground mb-1">Top Speed — {year1} vs {year2}</h3>
          <p className="text-[12px] text-muted-foreground mb-3">{driver} — common circuits</p>
          <div className="h-[280px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={merged}>
                <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="race" tick={{ fill: '#8888a0', fontSize: 8 }} angle={-45} textAnchor="end" height={60} interval={0} />
                <YAxis tick={{ fill: '#8888a0', fontSize: 9 }} domain={['auto', 'auto']} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="topY1" name={String(year1)} fill={Y1_COLOR} radius={[2, 2, 0, 0]} />
                <Bar dataKey="topY2" name={String(year2)} fill={Y2_COLOR} radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div className="bg-card border border-border rounded-lg shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
          <h3 className="text-sm text-foreground mb-1">Avg Speed — {year1} vs {year2}</h3>
          <p className="text-[12px] text-muted-foreground mb-3">{driver} — common circuits</p>
          <div className="h-[280px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={merged}>
                <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="race" tick={{ fill: '#8888a0', fontSize: 8 }} angle={-45} textAnchor="end" height={60} interval={0} />
                <YAxis tick={{ fill: '#8888a0', fontSize: 9 }} domain={['auto', 'auto']} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="avgY1" name={String(year1)} fill={Y1_COLOR} radius={[2, 2, 0, 0]} />
                <Bar dataKey="avgY2" name={String(year2)} fill={Y2_COLOR} radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="bg-card border border-border rounded-lg shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
          <h3 className="text-sm text-foreground mb-1">Throttle % — {year1} vs {year2}</h3>
          <p className="text-[12px] text-muted-foreground mb-3">{driver} — common circuits</p>
          <div className="h-[280px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={merged}>
                <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="race" tick={{ fill: '#8888a0', fontSize: 8 }} angle={-45} textAnchor="end" height={60} interval={0} />
                <YAxis tick={{ fill: '#8888a0', fontSize: 9 }} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="throttleY1" name={String(year1)} fill={Y1_COLOR} radius={[2, 2, 0, 0]} />
                <Bar dataKey="throttleY2" name={String(year2)} fill={Y2_COLOR} radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div className="bg-card border border-border rounded-lg shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
          <h3 className="text-sm text-foreground mb-1">Braking % — {year1} vs {year2}</h3>
          <p className="text-[12px] text-muted-foreground mb-3">{driver} — common circuits</p>
          <div className="h-[280px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={merged}>
                <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="race" tick={{ fill: '#8888a0', fontSize: 8 }} angle={-45} textAnchor="end" height={60} interval={0} />
                <YAxis tick={{ fill: '#8888a0', fontSize: 9 }} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="brakeY1" name={String(year1)} fill={Y1_COLOR} radius={[2, 2, 0, 0]} />
                <Bar dataKey="brakeY2" name={String(year2)} fill={Y2_COLOR} radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </>
  );
}
