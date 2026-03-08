import React, { useState, useEffect, useMemo } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, AreaChart, Area, BarChart, Bar,
  ReferenceLine, Cell,
} from 'recharts';
import {
  Heart, Thermometer, Loader2, AlertCircle, Swords, User, Trophy, GitCompareArrows,
  ArrowUp, ArrowDown, ShieldAlert, Eye, FileText, Bell, CheckCircle2, Activity,
} from 'lucide-react';
import { fetchCSV, parseCSV } from '../api/local';
import { HealthGauge } from './HealthGauge';
import { StatusBadge } from './StatusBadge';
import {
  parseAnomalyDrivers, levelColor, levelBg,
  MAINTENANCE_LABELS, SEVERITY_COLORS,
  type VehicleData,
} from './anomalyHelpers';

const RACES_2023 = [
  'Bahrain', 'Saudi Arabian', 'Australian', 'Miami', 'Monaco', 'Spanish',
  'Canadian', 'British', 'Hungarian', 'Dutch', 'Italian', 'Singapore',
  'Japanese', 'Mexico City', 'Las Vegas', 'Abu Dhabi',
];

const RACES_2024 = [
  'Bahrain', 'Saudi Arabian', 'Australian', 'Japanese', 'Monaco', 'Canadian',
  'Spanish', 'British', 'Hungarian', 'Belgian', 'Dutch', 'Italian',
  'Azerbaijan', 'Singapore', 'Mexico City', 'Las Vegas', 'Abu Dhabi',
  'Emilia Romagna',
];

const NOR_COLOR = '#FF8000';
const PIA_COLOR = '#22d3ee';

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
    <div className="bg-card border border-border rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-3">
      <div className="flex items-center gap-2 mb-1.5">
        {icon}
        <span className="text-[12px] text-muted-foreground tracking-wider">{label}</span>
      </div>
      <div className={`text-lg font-mono ${color}`}>{value}</div>
      <div className="text-[12px] text-muted-foreground mt-0.5">{detail}</div>
    </div>
  );
}

function DeltaKPI({ label, norVal, piaVal, unit, icon, higher = 'neutral' }: {
  label: string; norVal: string; piaVal: string; unit: string; icon: React.ReactNode;
  higher?: 'better' | 'worse' | 'neutral'; // reserved for future color logic
}) {
  const norNum = parseFloat(norVal);
  const piaNum = parseFloat(piaVal);
  const diff = norNum - piaNum;
  const absDiff = Math.abs(diff).toFixed(1);
  const norWins = diff > 0;
  return (
    <div className="bg-card border border-border rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-3">
      <div className="flex items-center gap-2 mb-2">
        {icon}
        <span className="text-[12px] text-muted-foreground tracking-wider">{label}</span>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div>
          <div className="text-[11px] text-muted-foreground mb-0.5">NOR</div>
          <div className="text-base font-mono" style={{ color: NOR_COLOR }}>{norVal}{unit}</div>
        </div>
        <div>
          <div className="text-[11px] text-muted-foreground mb-0.5">PIA</div>
          <div className="text-base font-mono" style={{ color: PIA_COLOR }}>{piaVal}{unit}</div>
        </div>
      </div>
      {diff !== 0 && !isNaN(diff) && (
        <div className="mt-2 flex items-center gap-1 text-[12px]">
          {norWins
            ? <ArrowUp className="w-3 h-3 text-[#FF8000]" />
            : <ArrowDown className="w-3 h-3 text-cyan-400" />
          }
          <span className="text-muted-foreground">
            {norWins ? 'NOR' : 'PIA'} +{absDiff}{unit}
          </span>
        </div>
      )}
    </div>
  );
}

interface CareerData {
  driver_name: string;
  driver_code: string;
  nationality: string;
  date_of_birth: string;
  num_seasons: string;
  seasons: string;
  races: string;
  wins: string;
  podiums: string;
  poles: string;
  dnfs: string;
  total_points: string;
  points_per_race: string;
  win_rate_pct: string;
  podium_rate_pct: string;
  best_finish: string;
}

interface RaceSummary {
  race: string;
  avgHR: number;
  peakHR: number;
  avgTemp: number;
  battleIntensity: number;
  airTemp: number;
  trackTemp: number;
  samples: number;
}

type Tab = 'detail' | 'season' | 'h2h' | 'racecompare';

/* ── helpers ── */
function computeBioKpis(data: Record<string, string>[]) {
  if (!data.length) return null;
  const hrs = data.map(r => Number(r.HeartRate_bpm) || 0).filter(v => v > 0);
  const temps = data.map(r => Number(r.CockpitTemp_C) || 0).filter(v => v > 0);
  const battles = data.map(r => Number(r.BattleIntensity) || 0);
  return {
    avgHR: hrs.length ? (hrs.reduce((a, b) => a + b, 0) / hrs.length).toFixed(1) : '—',
    peakHR: hrs.length ? Math.max(...hrs).toFixed(1) : '—',
    avgTemp: temps.length ? (temps.reduce((a, b) => a + b, 0) / temps.length).toFixed(1) : '—',
    battleIntensity: battles.length ? (battles.reduce((a, b) => a + b, 0) / battles.length * 100).toFixed(0) : '—',
  };
}

function buildHrPerLap(data: Record<string, string>[]) {
  const lapMap = new Map<number, number[]>();
  for (const r of data) {
    const lap = Number(r.LapNumber);
    const hr = Number(r.HeartRate_bpm);
    if (!lap || !hr) continue;
    if (!lapMap.has(lap)) lapMap.set(lap, []);
    lapMap.get(lap)!.push(hr);
  }
  return Array.from(lapMap.entries())
    .map(([lap, hrs]) => ({ lap, avgHR: hrs.reduce((a, b) => a + b, 0) / hrs.length }))
    .sort((a, b) => a.lap - b.lap);
}

function buildBattlePerLap(data: Record<string, string>[]) {
  const lapMap = new Map<number, number[]>();
  for (const r of data) {
    const lap = Number(r.LapNumber);
    if (!lap) continue;
    const bi = Number(r.BattleIntensity) || 0;
    if (!lapMap.has(lap)) lapMap.set(lap, []);
    lapMap.get(lap)!.push(bi);
  }
  return Array.from(lapMap.entries())
    .map(([lap, vals]) => ({ lap, intensity: (vals.reduce((a, b) => a + b, 0) / vals.length) * 100 }))
    .sort((a, b) => a.lap - b.lap);
}

function downsampleTrace(data: Record<string, string>[], maxPoints = 500) {
  const step = Math.max(1, Math.floor(data.length / maxPoints));
  return data.filter((_, i) => i % step === 0);
}

// ─── Biometric Intelligence helpers ──────────────────────────────────
const BIO_MAINTENANCE: Record<string, { label: string; icon: React.ElementType; color: string }> = {
  alert_and_remediate: { label: 'Immediate Action', icon: ShieldAlert, color: '#ef4444' },
  alert:              { label: 'Schedule Review',   icon: Bell,        color: '#FF8000' },
  log_and_monitor:    { label: 'Monitor',           icon: Eye,         color: '#eab308' },
  log:                { label: 'Logged',            icon: FileText,    color: '#6b7280' },
  none:               { label: 'No Action',         icon: CheckCircle2, color: '#22c55e' },
};

interface ThermalRace {
  race: string;
  health: number;
  action: string;
  probs: Record<string, number>;
  features: Record<string, number>;
}

function BiometricIntelligence({ driverCode }: { driverCode: string }) {
  const [races, setRaces] = useState<ThermalRace[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/pipeline/anomaly')
      .then(r => {
        if (!r.ok) throw new Error(`API ${r.status}`);
        return r.json();
      })
      .then(data => {
        // Handle both direct snapshot and nested structure
        const drivers = data.drivers ?? [];
        const drv = drivers.find((d: any) => d.code === driverCode);
        if (!drv?.races?.length) return;
        const mapped = drv.races.map((r: any) => {
          const t = r.systems?.['Tyre Management'] ?? r.systems?.['Lap Pace'] ?? {};
          return {
            race: r.race,
            health: t.health ?? 100,
            action: t.maintenance_action ?? 'none',
            probs: t.severity_probabilities ?? {},
            features: t.features ?? {},
          };
        });
        setRaces(mapped);
      })
      .catch(err => console.warn('[BiometricIntelligence]', err))
      .finally(() => setLoading(false));
  }, [driverCode]);

  if (loading) {
    return (
      <div className="flex items-center gap-2 p-3">
        <Activity className="w-4 h-4 text-[#FF8000] animate-spin" />
        <span className="text-[12px] text-muted-foreground">Loading biometric intelligence...</span>
      </div>
    );
  }

  if (!races.length) return null;

  const actionRaces = races.filter(r => r.action !== 'none');
  const avgHealth = Math.round(races.reduce((a, r) => a + r.health, 0) / races.length);
  const worstRace = races.reduce((a, b) => a.health < b.health ? a : b);

  return (
    <div className="bg-card border border-border rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-foreground flex items-center gap-2">
          <Activity className="w-3.5 h-3.5 text-[#FF8000]" />
          Biometric Risk Intelligence
        </h3>
        <div className="flex items-center gap-3 text-[11px]">
          <span className="text-muted-foreground">Season Avg</span>
          <span className="font-mono font-semibold" style={{ color: avgHealth >= 75 ? '#22c55e' : avgHealth >= 50 ? '#FF8000' : '#ef4444' }}>
            {avgHealth}%
          </span>
          {actionRaces.length > 0 && (
            <span className="text-[#FF8000]">{actionRaces.length} race{actionRaces.length > 1 ? 's' : ''} flagged</span>
          )}
        </div>
      </div>

      {/* Risk timeline — one bar per race */}
      <div className="flex gap-0.5 mb-3 h-8">
        {races.map((r, i) => {
          const info = BIO_MAINTENANCE[r.action] ?? BIO_MAINTENANCE.none;
          const healthColor = r.health >= 75 ? '#22c55e' : r.health >= 50 ? '#FF8000' : '#ef4444';
          return (
            <div key={i} className="flex-1 relative group cursor-default" title={`${r.race}: ${r.health}% — ${info.label}`}>
              <div className="h-full rounded-sm" style={{ background: `${healthColor}30`, borderBottom: `2px solid ${healthColor}` }}>
                <div className="h-full rounded-sm" style={{ background: healthColor, opacity: 0.3, height: `${r.health}%` }} />
              </div>
              {r.action !== 'none' && (
                <div className="absolute -top-0.5 left-1/2 -translate-x-1/2 w-1.5 h-1.5 rounded-full" style={{ background: info.color }} />
              )}
              {/* Tooltip on hover */}
              <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 hidden group-hover:block z-10 pointer-events-none">
                <div className="bg-background border border-[rgba(255,128,0,0.2)] rounded px-2 py-1 text-[10px] whitespace-nowrap">
                  <div className="font-medium text-foreground">{r.race}</div>
                  <div className="text-muted-foreground">Health: <span className="font-mono" style={{ color: healthColor }}>{r.health}%</span></div>
                  {r.features.HeartRate_bpm && <div className="text-muted-foreground">HR: {r.features.HeartRate_bpm} bpm</div>}
                  {r.features.CockpitTemp_C && <div className="text-muted-foreground">Cockpit: {r.features.CockpitTemp_C}°C</div>}
                  {r.action !== 'none' && <div style={{ color: info.color }}>{info.label}</div>}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Flagged races */}
      {actionRaces.length > 0 && (
        <div className="space-y-1.5">
          {actionRaces.slice(0, 3).map((r, i) => {
            const info = BIO_MAINTENANCE[r.action] ?? BIO_MAINTENANCE.none;
            const Icon = info.icon;
            const order = ['normal', 'low', 'medium', 'high', 'critical'];
            const colors: Record<string, string> = { normal: '#22c55e', low: '#6b7280', medium: '#eab308', high: '#FF8000', critical: '#ef4444' };
            const total = Object.values(r.probs).reduce((a, b) => a + b, 0);
            return (
              <div key={i} className="flex items-center gap-2 rounded-lg p-2 bg-background border border-border">
                <Icon className="w-3 h-3 flex-shrink-0" style={{ color: info.color }} />
                <span className="text-[11px] text-foreground w-24 truncate">{r.race}</span>
                <span className="text-[11px] font-mono" style={{ color: r.health >= 75 ? '#22c55e' : r.health >= 50 ? '#FF8000' : '#ef4444' }}>{r.health}%</span>
                {total > 0.01 && (
                  <div className="flex h-1.5 flex-1 rounded-full overflow-hidden bg-secondary">
                    {order.map(sev => {
                      const pct = (r.probs[sev] ?? 0) * 100;
                      if (pct < 1) return null;
                      return <div key={sev} style={{ width: `${pct}%`, background: colors[sev] }} />;
                    })}
                  </div>
                )}
                <div className="flex items-center gap-1 px-1.5 py-0.5 rounded text-[9px] font-medium flex-shrink-0"
                  style={{ background: `${info.color}15`, color: info.color, border: `1px solid ${info.color}25` }}>
                  {info.label}
                </div>
              </div>
            );
          })}
          {actionRaces.length > 3 && (
            <div className="text-[11px] text-muted-foreground pl-5">+{actionRaces.length - 3} more flagged</div>
          )}
        </div>
      )}

      {/* Worst race callout */}
      {worstRace.health < 70 && (
        <div className="mt-2 flex items-center gap-2 text-[11px] px-2 py-1.5 rounded bg-[rgba(239,68,68,0.06)] border border-[rgba(239,68,68,0.15)]">
          <AlertCircle className="w-3 h-3 text-red-400 flex-shrink-0" />
          <span className="text-muted-foreground">Lowest biometric health:</span>
          <span className="text-foreground font-medium">{worstRace.race}</span>
          <span className="font-mono text-red-400">{worstRace.health}%</span>
          {worstRace.features.HeartRate_bpm && (
            <span className="text-muted-foreground">· HR {worstRace.features.HeartRate_bpm} bpm</span>
          )}
          {worstRace.features.CockpitTemp_C && (
            <span className="text-muted-foreground">· {worstRace.features.CockpitTemp_C}°C cockpit</span>
          )}
        </div>
      )}
    </div>
  );
}

export function DriverBiometrics() {
  const [tab, setTab] = useState<Tab>('detail');
  const [year, setYear] = useState(2024);
  const [race, setRace] = useState('Bahrain');
  const [driver, setDriver] = useState('NOR');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [rawData, setRawData] = useState<Record<string, string>[]>([]);
  const [careerData, setCareerData] = useState<Record<string, string>[]>([]);
  const [seasonSummary, setSeasonSummary] = useState<RaceSummary[]>([]);
  const [summaryLoading, setSummaryLoading] = useState(false);

  // Anomaly / vehicle health state
  const [anomalyVehicles, setAnomalyVehicles] = useState<VehicleData[]>([]);

  useEffect(() => {
    fetch('/api/pipeline/anomaly')
      .then(r => r.json())
      .then(data => setAnomalyVehicles(parseAnomalyDrivers(data)))
      .catch(() => {});
  }, []);

  const selectedVehicle = useMemo(
    () => anomalyVehicles.find(v => v.code === driver) ?? null,
    [anomalyVehicles, driver],
  );

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

  // H2H state
  const [h2hMode, setH2hMode] = useState<'race' | 'season'>('race');
  const [h2hLoading, setH2hLoading] = useState(false);
  const [h2hRaceData, setH2hRaceData] = useState<Record<string, string>[]>([]);
  const [h2hSeasonNor, setH2hSeasonNor] = useState<RaceSummary[]>([]);
  const [h2hSeasonPia, setH2hSeasonPia] = useState<RaceSummary[]>([]);

  // Compare state
  const [compareMode, setCompareMode] = useState<'race' | 'year'>('race');
  const [race2, setRace2] = useState('Saudi Arabian');
  const [compareLoading, setCompareLoading] = useState(false);
  const [compareData1, setCompareData1] = useState<Record<string, string>[]>([]);
  const [compareData2, setCompareData2] = useState<Record<string, string>[]>([]);
  const [compareSeason23, setCompareSeason23] = useState<RaceSummary[]>([]);
  const [compareSeason24, setCompareSeason24] = useState<RaceSummary[]>([]);

  const races = year === 2024 ? RACES_2024 : RACES_2023;

  // Load career data once
  useEffect(() => {
    fetchCSV('mccsv/driver_career').then(csv => setCareerData(parseCSV(csv))).catch(() => {});
  }, []);

  // Load per-race biometrics (detail tab)
  useEffect(() => {
    if (tab !== 'detail') return;
    setLoading(true);
    setError(null);
    const raceName = `${year}_${race.replace(/ /g, '_')}_Grand_Prix_Race_biometrics`;
    fetchCSV(`mcdriver/${year}/${raceName}.csv`)
      .then(csv => { setRawData(parseCSV(csv)); setError(null); })
      .catch(() => { setError('Biometric data not available for this race'); setRawData([]); })
      .finally(() => setLoading(false));
  }, [year, race, tab]);

  // Load season summary (season tab)
  useEffect(() => {
    if (tab !== 'season') return;
    setSummaryLoading(true);
    fetch(`/api/mcdriver-summary/${year}/${driver}`)
      .then(res => res.json())
      .then(data => setSeasonSummary(data))
      .catch(() => setSeasonSummary([]))
      .finally(() => setSummaryLoading(false));
  }, [year, driver, tab]);

  // H2H race data — load CSV for selected race (contains both drivers)
  useEffect(() => {
    if (tab !== 'h2h' || h2hMode !== 'race') return;
    setH2hLoading(true);
    const raceName = `${year}_${race.replace(/ /g, '_')}_Grand_Prix_Race_biometrics`;
    fetchCSV(`mcdriver/${year}/${raceName}.csv`)
      .then(csv => setH2hRaceData(parseCSV(csv)))
      .catch(() => setH2hRaceData([]))
      .finally(() => setH2hLoading(false));
  }, [year, race, tab, h2hMode]);

  // H2H season data — load summaries for both drivers
  useEffect(() => {
    if (tab !== 'h2h' || h2hMode !== 'season') return;
    setH2hLoading(true);
    Promise.all([
      fetch(`/api/mcdriver-summary/${year}/NOR`).then(r => r.json()).catch(() => []),
      fetch(`/api/mcdriver-summary/${year}/PIA`).then(r => r.json()).catch(() => []),
    ]).then(([nor, pia]) => {
      setH2hSeasonNor(nor);
      setH2hSeasonPia(pia);
    }).finally(() => setH2hLoading(false));
  }, [year, tab, h2hMode]);

  // Compare: race vs race
  useEffect(() => {
    if (tab !== 'racecompare' || compareMode !== 'race') return;
    setCompareLoading(true);
    const name1 = `${year}_${race.replace(/ /g, '_')}_Grand_Prix_Race_biometrics`;
    const name2 = `${year}_${race2.replace(/ /g, '_')}_Grand_Prix_Race_biometrics`;
    Promise.all([
      fetchCSV(`mcdriver/${year}/${name1}.csv`).then(csv => parseCSV(csv)).catch(() => []),
      fetchCSV(`mcdriver/${year}/${name2}.csv`).then(csv => parseCSV(csv)).catch(() => []),
    ]).then(([d1, d2]) => {
      setCompareData1(d1);
      setCompareData2(d2);
    }).finally(() => setCompareLoading(false));
  }, [year, race, race2, driver, tab, compareMode]);

  // Compare: year vs year
  useEffect(() => {
    if (tab !== 'racecompare' || compareMode !== 'year') return;
    setCompareLoading(true);
    Promise.all([
      fetch(`/api/mcdriver-summary/2023/${driver}`).then(r => r.json()).catch(() => []),
      fetch(`/api/mcdriver-summary/2024/${driver}`).then(r => r.json()).catch(() => []),
    ]).then(([s23, s24]) => {
      setCompareSeason23(s23);
      setCompareSeason24(s24);
    }).finally(() => setCompareLoading(false));
  }, [driver, tab, compareMode]);

  const driverData = useMemo(() => rawData.filter(r => r.Driver === driver), [rawData, driver]);

  const career = useMemo((): CareerData | null => {
    const code = driver === 'NOR' ? 'norris' : 'piastri';
    const row = careerData.find(r => r.driver_id === code);
    return (row as unknown) as CareerData | null;
  }, [careerData, driver]);

  const kpis = useMemo(() => computeBioKpis(driverData), [driverData]);

  const hrTrace = useMemo(() => {
    if (!driverData.length) return [];
    return downsampleTrace(driverData).map(r => ({
      lap: Number(r.LapNumber) || 0,
      hr: Number(r.HeartRate_bpm) || 0,
      speed: Number(r.Speed) || 0,
    }));
  }, [driverData]);

  const tempTrace = useMemo(() => {
    if (!driverData.length) return [];
    return downsampleTrace(driverData).map(r => ({
      lap: Number(r.LapNumber) || 0,
      cockpit: Number(r.CockpitTemp_C) || 0,
      air: Number(r.AirTemp_C) || 0,
      track: Number(r.TrackTemp_C) || 0,
    }));
  }, [driverData]);

  const battlePerLap = useMemo(() => buildBattlePerLap(driverData), [driverData]);
  const hrPerLap = useMemo(() => buildHrPerLap(driverData), [driverData]);

  const highlightRace = race;

  return (
    <div className="space-y-4">
      {/* Tab Switcher + Controls */}
      <div className="flex items-center gap-3 flex-wrap">
        {/* Tabs */}
        <div className="flex items-center gap-1 bg-card rounded-lg p-0.5 border border-border">
          {([
            { id: 'detail' as Tab, label: 'Race Detail' },
            { id: 'season' as Tab, label: 'Season' },
            { id: 'h2h' as Tab, label: 'Head to Head' },
            { id: 'racecompare' as Tab, label: 'Compare' },
          ]).map(t => (
            <button type="button" key={t.id} onClick={() => setTab(t.id)}
              className={`text-sm px-4 py-1.5 rounded-md transition-all ${
                tab === t.id ? 'bg-[#FF8000]/10 text-[#FF8000]' : 'text-muted-foreground hover:text-foreground'
              }`}
            >{t.label}</button>
          ))}
        </div>
        {/* Year */}
        {tab !== 'racecompare' || compareMode !== 'year' ? (
          <div className="flex items-center gap-1 bg-card rounded-lg p-0.5 border border-border">
            {[2023, 2024].map(y => (
              <button type="button" key={y} onClick={() => { setYear(y); setRace(y === 2024 ? RACES_2024[0] : RACES_2023[0]); setRace2(y === 2024 ? RACES_2024[1] : RACES_2023[1]); }}
                className={`text-sm px-3 py-1.5 rounded-md transition-all ${year === y ? 'bg-[#FF8000]/10 text-[#FF8000]' : 'text-muted-foreground hover:text-foreground'}`}
              >{y}</button>
            ))}
          </div>
        ) : null}
        {/* Race A (detail, h2h race mode, racecompare race mode) */}
        {(tab === 'detail' || (tab === 'h2h' && h2hMode === 'race') || (tab === 'racecompare' && compareMode === 'race')) && (
          <select value={race} onChange={e => setRace(e.target.value)} aria-label="Select race"
            className="bg-card border border-border rounded-lg text-sm text-foreground px-3 py-1.5 outline-none"
          >
            {races.map(r => <option key={r} value={r}>{r} GP</option>)}
          </select>
        )}
        {/* Race B (racecompare race mode) */}
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
        {/* Driver (not in h2h — h2h always shows both) */}
        {tab !== 'h2h' && (
          <div className="flex items-center gap-1 bg-card rounded-lg p-0.5 border border-border">
            {['NOR', 'PIA'].map(d => (
              <button type="button" key={d} onClick={() => setDriver(d)}
                className={`text-sm px-3 py-1.5 rounded-md transition-all ${driver === d ? 'bg-[#FF8000]/10 text-[#FF8000]' : 'text-muted-foreground hover:text-foreground'}`}
              >{d}</button>
            ))}
          </div>
        )}
        {/* H2H sub-mode toggle */}
        {tab === 'h2h' && (
          <div className="flex items-center gap-1 bg-card rounded-lg p-0.5 border border-border">
            {([{ id: 'race' as const, label: 'Single Race' }, { id: 'season' as const, label: 'Full Season' }]).map(m => (
              <button type="button" key={m.id} onClick={() => setH2hMode(m.id)}
                className={`text-sm px-3 py-1.5 rounded-md transition-all ${h2hMode === m.id ? 'bg-[#FF8000]/10 text-[#FF8000]' : 'text-muted-foreground hover:text-foreground'}`}
              >{m.label}</button>
            ))}
          </div>
        )}
        {/* Compare sub-mode toggle */}
        {tab === 'racecompare' && (
          <div className="flex items-center gap-1 bg-card rounded-lg p-0.5 border border-border">
            {([{ id: 'race' as const, label: 'Race vs Race' }, { id: 'year' as const, label: '2023 vs 2024' }]).map(m => (
              <button type="button" key={m.id} onClick={() => setCompareMode(m.id)}
                className={`text-sm px-3 py-1.5 rounded-md transition-all ${compareMode === m.id ? 'bg-[#FF8000]/10 text-[#FF8000]' : 'text-muted-foreground hover:text-foreground'}`}
              >{m.label}</button>
            ))}
          </div>
        )}
        {/* Sample/race count */}
        {tab === 'detail' && (
          <span className="text-[12px] text-muted-foreground">{driverData.length.toLocaleString()} samples</span>
        )}
        {tab === 'season' && seasonSummary.length > 0 && (
          <span className="text-[12px] text-muted-foreground">{seasonSummary.length} races</span>
        )}
      </div>

      {error && tab === 'detail' && (
        <div className="flex items-center gap-2 text-amber-400 text-sm bg-amber-500/10 rounded-lg p-3">
          <AlertCircle className="w-4 h-4" />
          {error}
        </div>
      )}

      {tab === 'detail' && (
        <RaceDetailView career={career} kpis={kpis} hrTrace={hrTrace} tempTrace={tempTrace}
          battlePerLap={battlePerLap} hrPerLap={hrPerLap} race={race} year={year} loading={loading} driver={driver} />
      )}

      {/* Vehicle Health — from anomaly pipeline (shown on all tabs) */}
      {selectedVehicle && (
        <>
          <div className="flex items-center gap-2 mt-2">
            <div className="h-px flex-1 bg-[rgba(255,128,0,0.10)]" />
            <span className="text-[10px] tracking-[0.25em] text-[#FF8000]/60 font-semibold">VEHICLE HEALTH — {driver}</span>
            <div className="h-px flex-1 bg-[rgba(255,128,0,0.10)]" />
          </div>

          <div className="grid grid-cols-12 gap-4">
            <div className="col-span-3 bg-card border border-[rgba(255,128,0,0.20)] rounded-xl shadow-[0_4px_12px_rgba(0,0,0,0.4)] p-4 flex flex-col items-center justify-center gap-3">
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
            </div>

            <div className="col-span-9 grid grid-cols-3 gap-3">
              {selectedVehicle.systems.map((sys) => {
                const Icon = sys.icon;
                const maint = sys.maintenanceAction ? MAINTENANCE_LABELS[sys.maintenanceAction] ?? MAINTENANCE_LABELS.none : null;
                return (
                  <div key={sys.name} className="bg-card border border-[rgba(255,128,0,0.20)] rounded-xl shadow-[0_4px_12px_rgba(0,0,0,0.4)] p-4">
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
                    <div className="w-full h-1.5 bg-secondary rounded-full overflow-hidden mb-3">
                      <div className="h-full rounded-full transition-all duration-500" style={{ width: `${sys.health}%`, backgroundColor: levelColor(sys.level) }} />
                    </div>
                    {sys.severityProbabilities && (
                      <div className="space-y-1 mb-3">
                        <div className="text-[9px] text-muted-foreground tracking-wider">SEVERITY DISTRIBUTION</div>
                        <div className="flex h-2 rounded-full overflow-hidden">
                          {Object.entries(sys.severityProbabilities)
                            .sort(([,a], [,b]) => (b as number) - (a as number))
                            .map(([level, prob]) => (
                              <div key={level} style={{ width: `${(prob as number) * 100}%`, backgroundColor: SEVERITY_COLORS[level] ?? '#6b7280' }} title={`${level}: ${((prob as number) * 100).toFixed(0)}%`} />
                            ))}
                        </div>
                      </div>
                    )}
                    <div className="flex items-center justify-between text-[10px] text-muted-foreground mb-2">
                      <span>Models: {sys.voteCount}/{sys.totalModels}</span>
                    </div>
                    {maint && (
                      <div className="flex items-center gap-1.5 text-[10px] px-2 py-1 rounded-md" style={{ backgroundColor: maint.color + '15', color: maint.color }}>
                        <maint.icon className="w-3 h-3" />
                        <span>{maint.label}</span>
                      </div>
                    )}
                    {sys.metrics.length > 0 && (
                      <div className="mt-2 space-y-0.5">
                        <div className="text-[9px] text-muted-foreground tracking-wider">TOP FEATURES</div>
                        {sys.metrics.slice(0, 3).map((m: { label: string; value: string }) => (
                          <div key={m.label} className="flex items-center justify-between text-[10px]">
                            <span className="text-muted-foreground truncate mr-2">{m.label}</span>
                            <span className="font-mono text-foreground">{m.value}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {healthTrendData.length > 1 && (
            <div className="bg-card border border-[rgba(255,128,0,0.20)] rounded-xl shadow-[0_4px_12px_rgba(0,0,0,0.4)] p-4">
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
            </div>
          )}
        </>
      )}

      {tab === 'season' && (
        <SeasonCompareView seasonSummary={seasonSummary} loading={summaryLoading}
          year={year} driver={driver} highlightRace={highlightRace} />
      )}
      {tab === 'h2h' && (
        <HeadToHeadView
          mode={h2hMode} loading={h2hLoading} year={year} race={race}
          raceData={h2hRaceData} seasonNor={h2hSeasonNor} seasonPia={h2hSeasonPia}
        />
      )}
      {tab === 'racecompare' && (
        <RaceCompareView
          mode={compareMode} loading={compareLoading} year={year} driver={driver}
          race1={race} race2={race2}
          data1={compareData1} data2={compareData2}
          season23={compareSeason23} season24={compareSeason24}
        />
      )}
    </div>
  );
}

/* ─── Race Detail View ─── */

function RaceDetailView({ career, kpis, hrTrace, tempTrace, battlePerLap, hrPerLap, race, year, loading, driver }: {
  career: CareerData | null;
  kpis: { avgHR: string; peakHR: string; avgTemp: string; battleIntensity: string } | null;
  hrTrace: { lap: number; hr: number; speed: number }[];
  tempTrace: { lap: number; cockpit: number; air: number; track: number }[];
  battlePerLap: { lap: number; intensity: number }[];
  hrPerLap: { lap: number; avgHR: number }[];
  race: string;
  year: number;
  loading: boolean;
  driver: string;
}) {
  if (loading) {
    return (
      <div className="flex items-center justify-center h-[400px]">
        <Loader2 className="w-6 h-6 text-[#FF8000] animate-spin" />
        <span className="ml-2 text-sm text-muted-foreground">Loading biometrics...</span>
      </div>
    );
  }

  return (
    <>
      {/* Bio Card + KPIs */}
      <div className="grid grid-cols-12 gap-3">
        {career && (
          <div className="col-span-4 bg-card border border-border rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 rounded-full bg-[#FF8000]/10 flex items-center justify-center">
                <User className="w-5 h-5 text-[#FF8000]" />
              </div>
              <div>
                <h3 className="text-sm text-foreground font-medium">{career.driver_name}</h3>
                <div className="text-[12px] text-muted-foreground">{career.nationality} · Born {career.date_of_birth}</div>
              </div>
            </div>
            <div className="grid grid-cols-3 gap-2">
              <div className="text-center">
                <div className="text-sm font-mono text-[#FF8000]">{career.races}</div>
                <div className="text-[11px] text-muted-foreground">RACES</div>
              </div>
              <div className="text-center">
                <div className="text-sm font-mono text-green-400">{career.wins}</div>
                <div className="text-[11px] text-muted-foreground">WINS</div>
              </div>
              <div className="text-center">
                <div className="text-sm font-mono text-amber-400">{career.podiums}</div>
                <div className="text-[11px] text-muted-foreground">PODIUMS</div>
              </div>
              <div className="text-center">
                <div className="text-sm font-mono text-cyan-400">{career.poles}</div>
                <div className="text-[11px] text-muted-foreground">POLES</div>
              </div>
              <div className="text-center">
                <div className="text-sm font-mono text-foreground">{career.total_points}</div>
                <div className="text-[11px] text-muted-foreground">POINTS</div>
              </div>
              <div className="text-center">
                <div className="text-sm font-mono text-foreground">{career.num_seasons}</div>
                <div className="text-[11px] text-muted-foreground">SEASONS</div>
              </div>
            </div>
            <div className="mt-3 pt-3 border-t border-border space-y-1">
              <div className="flex justify-between text-[12px]">
                <span className="text-muted-foreground">Win Rate</span>
                <span className="text-foreground font-mono">{career.win_rate_pct}%</span>
              </div>
              <div className="flex justify-between text-[12px]">
                <span className="text-muted-foreground">Podium Rate</span>
                <span className="text-foreground font-mono">{career.podium_rate_pct}%</span>
              </div>
              <div className="flex justify-between text-[12px]">
                <span className="text-muted-foreground">Points/Race</span>
                <span className="text-foreground font-mono">{career.points_per_race}</span>
              </div>
              <div className="flex justify-between text-[12px]">
                <span className="text-muted-foreground">Seasons at McLaren</span>
                <span className="text-foreground font-mono">{career.seasons}</span>
              </div>
            </div>
          </div>
        )}

        <div className={`${career ? 'col-span-8' : 'col-span-12'} grid grid-cols-4 gap-3`}>
          {kpis && (
            <>
              <KPI icon={<Heart className="w-4 h-4 text-red-400" />} label="AVG HEART RATE" value={`${kpis.avgHR} bpm`} detail={`${race} GP ${year}`} color="text-red-400" />
              <KPI icon={<Heart className="w-4 h-4 text-rose-500" />} label="PEAK HEART RATE" value={`${kpis.peakHR} bpm`} detail="Maximum recorded" color="text-rose-500" />
              <KPI icon={<Thermometer className="w-4 h-4 text-amber-400" />} label="AVG COCKPIT TEMP" value={`${kpis.avgTemp}°C`} detail="In-cockpit estimate" color="text-amber-400" />
              <KPI icon={<Swords className="w-4 h-4 text-purple-400" />} label="BATTLE INTENSITY" value={`${kpis.battleIntensity}%`} detail="Close racing metric" color="text-purple-400" />
            </>
          )}
        </div>
      </div>

      {/* Biometric Risk Intelligence — anomaly detection on driver bio data */}
      <BiometricIntelligence driverCode={driver} />

      {/* Heart Rate Trace + Temperature */}
      {hrTrace.length > 0 && (
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-card border border-border rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
            <h3 className="text-sm text-foreground mb-1">Heart Rate Trace</h3>
            <p className="text-[12px] text-muted-foreground mb-3">BPM over race — zones: green &lt;150, amber 150-170, red &gt;170</p>
            <div className="h-[220px]">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={hrTrace}>
                  <defs>
                    <linearGradient id="hrGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="lap" tick={{ fill: '#8888a0', fontSize: 9 }} />
                  <YAxis tick={{ fill: '#8888a0', fontSize: 9 }} domain={[120, 200]} />
                  <Tooltip content={<CustomTooltip />} />
                  <ReferenceLine y={150} stroke="#22c55e" strokeDasharray="3 3" strokeOpacity={0.5} />
                  <ReferenceLine y={170} stroke="#f59e0b" strokeDasharray="3 3" strokeOpacity={0.5} />
                  <Area type="monotone" dataKey="hr" stroke="#ef4444" fill="url(#hrGrad)" strokeWidth={1.5} dot={false} name="Heart Rate (bpm)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="bg-card border border-border rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
            <h3 className="text-sm text-foreground mb-1">Temperature Profile</h3>
            <p className="text-[12px] text-muted-foreground mb-3">Cockpit vs air vs track temperature</p>
            <div className="h-[220px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={tempTrace}>
                  <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="lap" tick={{ fill: '#8888a0', fontSize: 9 }} />
                  <YAxis tick={{ fill: '#8888a0', fontSize: 9 }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Line type="monotone" dataKey="cockpit" stroke="#f59e0b" strokeWidth={1.5} dot={false} name="Cockpit °C" />
                  <Line type="monotone" dataKey="air" stroke="#22d3ee" strokeWidth={1} dot={false} name="Air °C" strokeDasharray="4 2" />
                  <Line type="monotone" dataKey="track" stroke="#a78bfa" strokeWidth={1} dot={false} name="Track °C" strokeDasharray="4 2" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* HR vs Speed + Battle Intensity */}
      {hrTrace.length > 0 && (
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-card border border-border rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
            <h3 className="text-sm text-foreground mb-1">Heart Rate vs Speed</h3>
            <p className="text-[12px] text-muted-foreground mb-3">HR spikes correlated with speed changes</p>
            <div className="h-[220px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={hrTrace}>
                  <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="lap" tick={{ fill: '#8888a0', fontSize: 9 }} />
                  <YAxis yAxisId="hr" tick={{ fill: '#8888a0', fontSize: 9 }} domain={[120, 200]} />
                  <YAxis yAxisId="speed" orientation="right" tick={{ fill: '#8888a0', fontSize: 9 }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Line yAxisId="hr" type="monotone" dataKey="hr" stroke="#ef4444" strokeWidth={1.5} dot={false} name="HR (bpm)" />
                  <Line yAxisId="speed" type="monotone" dataKey="speed" stroke="#FF8000" strokeWidth={1} dot={false} name="Speed (km/h)" opacity={0.6} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {battlePerLap.length > 0 && (
            <div className="bg-card border border-border rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
              <h3 className="text-sm text-foreground mb-1">Battle Intensity by Lap</h3>
              <p className="text-[12px] text-muted-foreground mb-3">Close racing metric — % of intervals under 1 second</p>
              <div className="h-[220px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={battlePerLap}>
                    <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="lap" tick={{ fill: '#8888a0', fontSize: 9 }} />
                    <YAxis tick={{ fill: '#8888a0', fontSize: 9 }} domain={[0, 100]} />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey="intensity" fill="#a78bfa" name="Battle Intensity %" radius={[2, 2, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Avg HR per Lap */}
      {hrPerLap.length > 0 && (
        <div className="bg-card border border-border rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
          <h3 className="text-sm text-foreground mb-1">Average Heart Rate by Lap</h3>
          <p className="text-[12px] text-muted-foreground mb-3">Lap-by-lap average BPM showing fatigue and adrenaline patterns</p>
          <div className="h-[200px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={hrPerLap}>
                <defs>
                  <linearGradient id="hrLapGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#FF8000" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#FF8000" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="lap" tick={{ fill: '#8888a0', fontSize: 9 }} />
                <YAxis tick={{ fill: '#8888a0', fontSize: 9 }} domain={['auto', 'auto']} />
                <Tooltip content={<CustomTooltip />} />
                <ReferenceLine y={150} stroke="#22c55e" strokeDasharray="3 3" strokeOpacity={0.4} />
                <ReferenceLine y={170} stroke="#f59e0b" strokeDasharray="3 3" strokeOpacity={0.4} />
                <Area type="monotone" dataKey="avgHR" stroke="#FF8000" fill="url(#hrLapGrad)" strokeWidth={1.5} dot={false} name="Avg HR (bpm)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Career Stats Table */}
      {career && (
        <div className="bg-card border border-border rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
          <h3 className="text-sm text-foreground mb-3 flex items-center gap-2">
            <Trophy className="w-3 h-3 text-[#FF8000]" />
            McLaren Career Summary — {career.driver_name}
          </h3>
          <div className="grid grid-cols-[1fr_1fr_1fr_1fr_1fr_1fr_1fr_1fr] gap-px bg-[rgba(255,128,0,0.12)] rounded-lg overflow-hidden">
            {['Seasons', 'Races', 'Wins', 'Podiums', 'Poles', 'DNFs', 'Points', 'Pts/Race'].map(h => (
              <div key={h} className="bg-secondary px-3 py-2 text-[11px] text-muted-foreground tracking-wider text-center">{h}</div>
            ))}
            {[career.num_seasons, career.races, career.wins, career.podiums, career.poles, career.dnfs, career.total_points, career.points_per_race].map((v, i) => (
              <div key={i} className="bg-card px-3 py-2 text-sm font-mono text-foreground text-center">{v}</div>
            ))}
          </div>
        </div>
      )}
    </>
  );
}

/* ─── Season Compare View ─── */

function SeasonCompareView({ seasonSummary, loading, year, driver, highlightRace }: {
  seasonSummary: RaceSummary[];
  loading: boolean;
  year: number;
  driver: string;
  highlightRace: string;
}) {
  if (loading) {
    return (
      <div className="flex items-center justify-center h-[400px]">
        <Loader2 className="w-6 h-6 text-[#FF8000] animate-spin" />
        <span className="ml-2 text-sm text-muted-foreground">Computing season summary...</span>
      </div>
    );
  }

  if (!seasonSummary.length) {
    return (
      <div className="flex items-center justify-center h-[400px] text-muted-foreground text-sm">
        No biometric data available for {year}
      </div>
    );
  }

  const seasonAvgHR = seasonSummary.reduce((s, r) => s + r.avgHR, 0) / seasonSummary.length;
  const seasonPeakHR = Math.max(...seasonSummary.map(r => r.peakHR));
  const seasonAvgTemp = seasonSummary.reduce((s, r) => s + r.avgTemp, 0) / seasonSummary.length;
  const seasonAvgBattle = seasonSummary.reduce((s, r) => s + r.battleIntensity, 0) / seasonSummary.length;
  const hardestRace = seasonSummary.reduce((best, r) => r.avgHR > best.avgHR ? r : best, seasonSummary[0]);
  const hottestRace = seasonSummary.reduce((best, r) => r.avgTemp > best.avgTemp ? r : best, seasonSummary[0]);

  return (
    <>
      <div className="grid grid-cols-6 gap-3">
        <KPI icon={<Heart className="w-4 h-4 text-red-400" />} label="SEASON AVG HR" value={`${seasonAvgHR.toFixed(1)} bpm`} detail={`${year} average across ${seasonSummary.length} races`} color="text-red-400" />
        <KPI icon={<Heart className="w-4 h-4 text-rose-500" />} label="SEASON PEAK HR" value={`${seasonPeakHR.toFixed(1)} bpm`} detail={`Highest at ${hardestRace.race} GP`} color="text-rose-500" />
        <KPI icon={<Thermometer className="w-4 h-4 text-amber-400" />} label="SEASON AVG TEMP" value={`${seasonAvgTemp.toFixed(1)}°C`} detail={`Hottest: ${hottestRace.race} GP`} color="text-amber-400" />
        <KPI icon={<Swords className="w-4 h-4 text-purple-400" />} label="AVG BATTLE" value={`${seasonAvgBattle.toFixed(1)}%`} detail="Season average intensity" color="text-purple-400" />
        <KPI icon={<GitCompareArrows className="w-4 h-4 text-cyan-400" />} label="HARDEST RACE" value={hardestRace.race} detail={`${hardestRace.avgHR.toFixed(1)} bpm avg`} color="text-cyan-400" />
        <KPI icon={<Thermometer className="w-4 h-4 text-orange-400" />} label="HOTTEST COCKPIT" value={`${hottestRace.avgTemp.toFixed(1)}°C`} detail={`${hottestRace.race} GP`} color="text-orange-400" />
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="bg-card border border-border rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
          <h3 className="text-sm text-foreground mb-1">Average Heart Rate by Race</h3>
          <p className="text-[12px] text-muted-foreground mb-3">{driver} — {year} season, BPM per race</p>
          <div className="h-[260px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={seasonSummary}>
                <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="race" tick={{ fill: '#8888a0', fontSize: 8 }} angle={-45} textAnchor="end" height={60} interval={0} />
                <YAxis tick={{ fill: '#8888a0', fontSize: 9 }} domain={['auto', 'auto']} />
                <Tooltip content={<CustomTooltip />} />
                <ReferenceLine y={seasonAvgHR} stroke="#FF8000" strokeDasharray="4 2" strokeOpacity={0.6} />
                <Bar dataKey="avgHR" name="Avg HR (bpm)" radius={[2, 2, 0, 0]}>
                  {seasonSummary.map((entry, i) => (
                    <Cell key={i} fill={entry.race === highlightRace ? '#FF8000' : entry.avgHR > seasonAvgHR + 2 ? '#ef4444' : '#ef444480'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-card border border-border rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
          <h3 className="text-sm text-foreground mb-1">Peak Heart Rate by Race</h3>
          <p className="text-[12px] text-muted-foreground mb-3">Maximum BPM recorded per race</p>
          <div className="h-[260px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={seasonSummary}>
                <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="race" tick={{ fill: '#8888a0', fontSize: 8 }} angle={-45} textAnchor="end" height={60} interval={0} />
                <YAxis tick={{ fill: '#8888a0', fontSize: 9 }} domain={['auto', 'auto']} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="peakHR" name="Peak HR (bpm)" radius={[2, 2, 0, 0]}>
                  {seasonSummary.map((entry, i) => (
                    <Cell key={i} fill={entry.race === highlightRace ? '#FF8000' : entry.peakHR > 185 ? '#f43f5e' : '#f43f5e80'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="bg-card border border-border rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
          <h3 className="text-sm text-foreground mb-1">Cockpit Temperature by Race</h3>
          <p className="text-[12px] text-muted-foreground mb-3">Average cockpit vs air temperature per race</p>
          <div className="h-[260px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={seasonSummary}>
                <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="race" tick={{ fill: '#8888a0', fontSize: 8 }} angle={-45} textAnchor="end" height={60} interval={0} />
                <YAxis tick={{ fill: '#8888a0', fontSize: 9 }} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="avgTemp" name="Cockpit °C" fill="#f59e0b" radius={[2, 2, 0, 0]} />
                <Bar dataKey="airTemp" name="Air °C" fill="#22d3ee60" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-card border border-border rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
          <h3 className="text-sm text-foreground mb-1">Battle Intensity by Race</h3>
          <p className="text-[12px] text-muted-foreground mb-3">Close racing metric — higher = more wheel-to-wheel</p>
          <div className="h-[260px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={seasonSummary}>
                <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="race" tick={{ fill: '#8888a0', fontSize: 8 }} angle={-45} textAnchor="end" height={60} interval={0} />
                <YAxis tick={{ fill: '#8888a0', fontSize: 9 }} domain={[0, 'auto']} />
                <Tooltip content={<CustomTooltip />} />
                <ReferenceLine y={seasonAvgBattle} stroke="#a78bfa" strokeDasharray="4 2" strokeOpacity={0.6} />
                <Bar dataKey="battleIntensity" name="Battle Intensity %" radius={[2, 2, 0, 0]}>
                  {seasonSummary.map((entry, i) => (
                    <Cell key={i} fill={entry.race === highlightRace ? '#FF8000' : '#a78bfa'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Season Summary Table */}
      <div className="bg-card border border-border rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
        <h3 className="text-sm text-foreground mb-3 flex items-center gap-2">
          <GitCompareArrows className="w-3 h-3 text-[#FF8000]" />
          {driver} — {year} Season Biometric Summary
        </h3>
        <div className="overflow-x-auto">
          <div className="grid grid-cols-[180px_80px_80px_80px_80px_80px_80px_80px] gap-px bg-[rgba(255,128,0,0.12)] rounded-lg overflow-hidden min-w-[740px]">
            {['Race', 'Avg HR', 'Peak HR', 'Cockpit °C', 'Air °C', 'Track °C', 'Battle %', 'Samples'].map(h => (
              <div key={h} className="bg-secondary px-3 py-2 text-[11px] text-muted-foreground tracking-wider">{h}</div>
            ))}
            {seasonSummary.map((r, i) => (
              <React.Fragment key={i}>
                <div className={`px-3 py-1.5 text-sm ${r.race === highlightRace ? 'bg-[#FF8000]/10 text-[#FF8000]' : 'bg-card text-foreground'}`}>
                  {r.race} GP
                </div>
                <div className={`px-3 py-1.5 text-sm font-mono ${r.race === highlightRace ? 'bg-[#FF8000]/10' : 'bg-card'} ${r.avgHR > seasonAvgHR + 2 ? 'text-red-400' : 'text-foreground'}`}>
                  {r.avgHR.toFixed(1)}
                </div>
                <div className={`px-3 py-1.5 text-sm font-mono ${r.race === highlightRace ? 'bg-[#FF8000]/10' : 'bg-card'} ${r.peakHR > 185 ? 'text-rose-500' : 'text-foreground'}`}>
                  {r.peakHR.toFixed(1)}
                </div>
                <div className={`px-3 py-1.5 text-sm font-mono ${r.race === highlightRace ? 'bg-[#FF8000]/10' : 'bg-card'} ${r.avgTemp > 50 ? 'text-amber-400' : 'text-foreground'}`}>
                  {r.avgTemp.toFixed(1)}
                </div>
                <div className={`px-3 py-1.5 text-sm font-mono ${r.race === highlightRace ? 'bg-[#FF8000]/10' : 'bg-card'} text-foreground`}>
                  {r.airTemp.toFixed(1)}
                </div>
                <div className={`px-3 py-1.5 text-sm font-mono ${r.race === highlightRace ? 'bg-[#FF8000]/10' : 'bg-card'} text-foreground`}>
                  {r.trackTemp.toFixed(1)}
                </div>
                <div className={`px-3 py-1.5 text-sm font-mono ${r.race === highlightRace ? 'bg-[#FF8000]/10' : 'bg-card'} ${r.battleIntensity > 40 ? 'text-purple-400' : 'text-foreground'}`}>
                  {r.battleIntensity.toFixed(1)}
                </div>
                <div className={`px-3 py-1.5 text-[12px] font-mono ${r.race === highlightRace ? 'bg-[#FF8000]/10' : 'bg-card'} text-muted-foreground`}>
                  {r.samples.toLocaleString()}
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

function HeadToHeadView({ mode, loading, year, race, raceData, seasonNor, seasonPia }: {
  mode: 'race' | 'season';
  loading: boolean;
  year: number;
  race: string;
  raceData: Record<string, string>[];
  seasonNor: RaceSummary[];
  seasonPia: RaceSummary[];
}) {
  if (loading) {
    return (
      <div className="flex items-center justify-center h-[400px]">
        <Loader2 className="w-6 h-6 text-[#FF8000] animate-spin" />
        <span className="ml-2 text-sm text-muted-foreground">Loading head-to-head data...</span>
      </div>
    );
  }

  if (mode === 'race') return <H2HRaceView raceData={raceData} race={race} year={year} />;
  return <H2HSeasonView seasonNor={seasonNor} seasonPia={seasonPia} year={year} />;
}

function H2HRaceView({ raceData, race, year }: { raceData: Record<string, string>[]; race: string; year: number }) {
  const norData = useMemo(() => raceData.filter(r => r.Driver === 'NOR'), [raceData]);
  const piaData = useMemo(() => raceData.filter(r => r.Driver === 'PIA'), [raceData]);
  const norKpis = useMemo(() => computeBioKpis(norData), [norData]);
  const piaKpis = useMemo(() => computeBioKpis(piaData), [piaData]);

  const norHrPerLap = useMemo(() => buildHrPerLap(norData), [norData]);
  const piaHrPerLap = useMemo(() => buildHrPerLap(piaData), [piaData]);
  const norBattle = useMemo(() => buildBattlePerLap(norData), [norData]);
  const piaBattle = useMemo(() => buildBattlePerLap(piaData), [piaData]);

  // Merge HR per lap for overlay chart
  const mergedHrPerLap = useMemo(() => {
    const map = new Map<number, { lap: number; norHR?: number; piaHR?: number }>();
    for (const d of norHrPerLap) { map.set(d.lap, { ...map.get(d.lap), lap: d.lap, norHR: d.avgHR }); }
    for (const d of piaHrPerLap) { map.set(d.lap, { ...map.get(d.lap), lap: d.lap, piaHR: d.avgHR }); }
    return Array.from(map.values()).sort((a, b) => a.lap - b.lap);
  }, [norHrPerLap, piaHrPerLap]);

  // Merge battle per lap
  const mergedBattle = useMemo(() => {
    const map = new Map<number, { lap: number; norBattle?: number; piaBattle?: number }>();
    for (const d of norBattle) { map.set(d.lap, { ...map.get(d.lap), lap: d.lap, norBattle: d.intensity }); }
    for (const d of piaBattle) { map.set(d.lap, { ...map.get(d.lap), lap: d.lap, piaBattle: d.intensity }); }
    return Array.from(map.values()).sort((a, b) => a.lap - b.lap);
  }, [norBattle, piaBattle]);

  // HR trace overlay
  const mergedHrTrace = useMemo(() => {
    const norSampled = downsampleTrace(norData);
    const piaSampled = downsampleTrace(piaData);
    const map = new Map<number, { lap: number; norHR?: number; piaHR?: number }>();
    for (const r of norSampled) {
      const lap = Number(r.LapNumber) || 0;
      const hr = Number(r.HeartRate_bpm) || 0;
      if (lap && hr) map.set(lap, { ...map.get(lap), lap, norHR: hr });
    }
    for (const r of piaSampled) {
      const lap = Number(r.LapNumber) || 0;
      const hr = Number(r.HeartRate_bpm) || 0;
      if (lap && hr) map.set(lap, { ...map.get(lap), lap, piaHR: hr });
    }
    return Array.from(map.values()).sort((a, b) => a.lap - b.lap);
  }, [norData, piaData]);

  if (!norData.length && !piaData.length) {
    return <div className="flex items-center justify-center h-[300px] text-muted-foreground text-sm">No data available for {race} GP {year}</div>;
  }

  return (
    <>
      {/* Delta KPI Cards */}
      <div className="grid grid-cols-4 gap-3">
        <DeltaKPI label="AVG HEART RATE" norVal={norKpis?.avgHR || '—'} piaVal={piaKpis?.avgHR || '—'} unit=" bpm"
          icon={<Heart className="w-4 h-4 text-red-400" />} />
        <DeltaKPI label="PEAK HEART RATE" norVal={norKpis?.peakHR || '—'} piaVal={piaKpis?.peakHR || '—'} unit=" bpm"
          icon={<Heart className="w-4 h-4 text-rose-500" />} />
        <DeltaKPI label="AVG COCKPIT TEMP" norVal={norKpis?.avgTemp || '—'} piaVal={piaKpis?.avgTemp || '—'} unit="°C"
          icon={<Thermometer className="w-4 h-4 text-amber-400" />} />
        <DeltaKPI label="BATTLE INTENSITY" norVal={norKpis?.battleIntensity || '—'} piaVal={piaKpis?.battleIntensity || '—'} unit="%"
          icon={<Swords className="w-4 h-4 text-purple-400" />} />
      </div>

      {/* HR Trace Overlay */}
      {mergedHrTrace.length > 0 && (
        <div className="bg-card border border-border rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
          <h3 className="text-sm text-foreground mb-1">Heart Rate Overlay — {race} GP {year}</h3>
          <p className="text-[12px] text-muted-foreground mb-3">
            <span style={{ color: NOR_COLOR }}>■</span> NOR vs <span style={{ color: PIA_COLOR }}>■</span> PIA
          </p>
          <div className="h-[260px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={mergedHrTrace}>
                <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="lap" tick={{ fill: '#8888a0', fontSize: 9 }} />
                <YAxis tick={{ fill: '#8888a0', fontSize: 9 }} domain={[120, 200]} />
                <Tooltip content={<CustomTooltip />} />
                <ReferenceLine y={150} stroke="#22c55e" strokeDasharray="3 3" strokeOpacity={0.3} />
                <ReferenceLine y={170} stroke="#f59e0b" strokeDasharray="3 3" strokeOpacity={0.3} />
                <Line type="monotone" dataKey="norHR" stroke={NOR_COLOR} strokeWidth={1.5} dot={false} name="NOR HR (bpm)" />
                <Line type="monotone" dataKey="piaHR" stroke={PIA_COLOR} strokeWidth={1.5} dot={false} name="PIA HR (bpm)" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Avg HR per Lap + Battle Intensity */}
      <div className="grid grid-cols-2 gap-3">
        {mergedHrPerLap.length > 0 && (
          <div className="bg-card border border-border rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
            <h3 className="text-sm text-foreground mb-1">Average HR per Lap</h3>
            <p className="text-[12px] text-muted-foreground mb-3">Lap-by-lap comparison</p>
            <div className="h-[240px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={mergedHrPerLap}>
                  <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="lap" tick={{ fill: '#8888a0', fontSize: 9 }} />
                  <YAxis tick={{ fill: '#8888a0', fontSize: 9 }} domain={['auto', 'auto']} />
                  <Tooltip content={<CustomTooltip />} />
                  <Line type="monotone" dataKey="norHR" stroke={NOR_COLOR} strokeWidth={1.5} dot={false} name="NOR Avg HR" />
                  <Line type="monotone" dataKey="piaHR" stroke={PIA_COLOR} strokeWidth={1.5} dot={false} name="PIA Avg HR" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
        {mergedBattle.length > 0 && (
          <div className="bg-card border border-border rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
            <h3 className="text-sm text-foreground mb-1">Battle Intensity per Lap</h3>
            <p className="text-[12px] text-muted-foreground mb-3">Close racing comparison</p>
            <div className="h-[240px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={mergedBattle}>
                  <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="lap" tick={{ fill: '#8888a0', fontSize: 9 }} />
                  <YAxis tick={{ fill: '#8888a0', fontSize: 9 }} domain={[0, 100]} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="norBattle" fill={NOR_COLOR} name="NOR Battle %" radius={[2, 2, 0, 0]} />
                  <Bar dataKey="piaBattle" fill={PIA_COLOR} name="PIA Battle %" radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>
    </>
  );
}

function H2HSeasonView({ seasonNor, seasonPia, year }: { seasonNor: RaceSummary[]; seasonPia: RaceSummary[]; year: number }) {
  // Merge season data by race
  const mergedSeason = useMemo(() => {
    const allRaces = [...new Set([...seasonNor.map(r => r.race), ...seasonPia.map(r => r.race)])];
    return allRaces.map(race => {
      const nor = seasonNor.find(r => r.race === race);
      const pia = seasonPia.find(r => r.race === race);
      return {
        race,
        norAvgHR: nor?.avgHR ?? 0,
        piaAvgHR: pia?.avgHR ?? 0,
        norPeakHR: nor?.peakHR ?? 0,
        piaPeakHR: pia?.peakHR ?? 0,
        norBattle: nor?.battleIntensity ?? 0,
        piaBattle: pia?.battleIntensity ?? 0,
        norTemp: nor?.avgTemp ?? 0,
        piaTemp: pia?.avgTemp ?? 0,
      };
    });
  }, [seasonNor, seasonPia]);

  // Season aggregate KPIs
  const norAvg = seasonNor.length ? (seasonNor.reduce((s, r) => s + r.avgHR, 0) / seasonNor.length).toFixed(1) : '—';
  const piaAvg = seasonPia.length ? (seasonPia.reduce((s, r) => s + r.avgHR, 0) / seasonPia.length).toFixed(1) : '—';
  const norPeak = seasonNor.length ? Math.max(...seasonNor.map(r => r.peakHR)).toFixed(1) : '—';
  const piaPeak = seasonPia.length ? Math.max(...seasonPia.map(r => r.peakHR)).toFixed(1) : '—';
  const norTemp = seasonNor.length ? (seasonNor.reduce((s, r) => s + r.avgTemp, 0) / seasonNor.length).toFixed(1) : '—';
  const piaTemp = seasonPia.length ? (seasonPia.reduce((s, r) => s + r.avgTemp, 0) / seasonPia.length).toFixed(1) : '—';
  const norBattle = seasonNor.length ? (seasonNor.reduce((s, r) => s + r.battleIntensity, 0) / seasonNor.length).toFixed(1) : '—';
  const piaBattle = seasonPia.length ? (seasonPia.reduce((s, r) => s + r.battleIntensity, 0) / seasonPia.length).toFixed(1) : '—';

  if (!mergedSeason.length) {
    return <div className="flex items-center justify-center h-[300px] text-muted-foreground text-sm">No season data available for {year}</div>;
  }

  return (
    <>
      <div className="grid grid-cols-4 gap-3">
        <DeltaKPI label="SEASON AVG HR" norVal={norAvg} piaVal={piaAvg} unit=" bpm" icon={<Heart className="w-4 h-4 text-red-400" />} />
        <DeltaKPI label="SEASON PEAK HR" norVal={norPeak} piaVal={piaPeak} unit=" bpm" icon={<Heart className="w-4 h-4 text-rose-500" />} />
        <DeltaKPI label="AVG COCKPIT TEMP" norVal={norTemp} piaVal={piaTemp} unit="°C" icon={<Thermometer className="w-4 h-4 text-amber-400" />} />
        <DeltaKPI label="AVG BATTLE" norVal={norBattle} piaVal={piaBattle} unit="%" icon={<Swords className="w-4 h-4 text-purple-400" />} />
      </div>

      {/* Avg HR by Race — dual bars */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-card border border-border rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
          <h3 className="text-sm text-foreground mb-1">Average HR by Race — NOR vs PIA</h3>
          <p className="text-[12px] text-muted-foreground mb-3">{year} season comparison</p>
          <div className="h-[280px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={mergedSeason}>
                <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="race" tick={{ fill: '#8888a0', fontSize: 8 }} angle={-45} textAnchor="end" height={60} interval={0} />
                <YAxis tick={{ fill: '#8888a0', fontSize: 9 }} domain={['auto', 'auto']} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="norAvgHR" name="NOR" fill={NOR_COLOR} radius={[2, 2, 0, 0]} />
                <Bar dataKey="piaAvgHR" name="PIA" fill={PIA_COLOR} radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-card border border-border rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
          <h3 className="text-sm text-foreground mb-1">Peak HR by Race — NOR vs PIA</h3>
          <p className="text-[12px] text-muted-foreground mb-3">Max BPM per race</p>
          <div className="h-[280px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={mergedSeason}>
                <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="race" tick={{ fill: '#8888a0', fontSize: 8 }} angle={-45} textAnchor="end" height={60} interval={0} />
                <YAxis tick={{ fill: '#8888a0', fontSize: 9 }} domain={['auto', 'auto']} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="norPeakHR" name="NOR" fill={NOR_COLOR} radius={[2, 2, 0, 0]} />
                <Bar dataKey="piaPeakHR" name="PIA" fill={PIA_COLOR} radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Battle + Temp comparison */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-card border border-border rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
          <h3 className="text-sm text-foreground mb-1">Battle Intensity — NOR vs PIA</h3>
          <p className="text-[12px] text-muted-foreground mb-3">Who races closer wheel-to-wheel?</p>
          <div className="h-[280px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={mergedSeason}>
                <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="race" tick={{ fill: '#8888a0', fontSize: 8 }} angle={-45} textAnchor="end" height={60} interval={0} />
                <YAxis tick={{ fill: '#8888a0', fontSize: 9 }} domain={[0, 'auto']} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="norBattle" name="NOR" fill={NOR_COLOR} radius={[2, 2, 0, 0]} />
                <Bar dataKey="piaBattle" name="PIA" fill={PIA_COLOR} radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-card border border-border rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
          <h3 className="text-sm text-foreground mb-1">Cockpit Temperature — NOR vs PIA</h3>
          <p className="text-[12px] text-muted-foreground mb-3">Avg cockpit temp per race</p>
          <div className="h-[280px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={mergedSeason}>
                <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="race" tick={{ fill: '#8888a0', fontSize: 8 }} angle={-45} textAnchor="end" height={60} interval={0} />
                <YAxis tick={{ fill: '#8888a0', fontSize: 9 }} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="norTemp" name="NOR °C" fill={NOR_COLOR} radius={[2, 2, 0, 0]} />
                <Bar dataKey="piaTemp" name="PIA °C" fill={PIA_COLOR} radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </>
  );
}

/* ─── Race Compare View ─── */

function RaceCompareView({ mode, loading, year, driver, race1, race2, data1, data2, season23, season24 }: {
  mode: 'race' | 'year';
  loading: boolean;
  year: number;
  driver: string;
  race1: string;
  race2: string;
  data1: Record<string, string>[];
  data2: Record<string, string>[];
  season23: RaceSummary[];
  season24: RaceSummary[];
}) {
  if (loading) {
    return (
      <div className="flex items-center justify-center h-[400px]">
        <Loader2 className="w-6 h-6 text-[#FF8000] animate-spin" />
        <span className="ml-2 text-sm text-muted-foreground">Loading comparison data...</span>
      </div>
    );
  }

  if (mode === 'race') return <RaceVsRaceView driver={driver} year={year} race1={race1} race2={race2} data1={data1} data2={data2} />;
  return <YearVsYearView driver={driver} season23={season23} season24={season24} />;
}

function RaceVsRaceView({ driver, year, race1, race2, data1, data2 }: {
  driver: string; year: number; race1: string; race2: string;
  data1: Record<string, string>[]; data2: Record<string, string>[];
}) {
  const d1 = useMemo(() => data1.filter(r => r.Driver === driver), [data1, driver]);
  const d2 = useMemo(() => data2.filter(r => r.Driver === driver), [data2, driver]);
  const kpis1 = useMemo(() => computeBioKpis(d1), [d1]);
  const kpis2 = useMemo(() => computeBioKpis(d2), [d2]);

  const hr1 = useMemo(() => buildHrPerLap(d1), [d1]);
  const hr2 = useMemo(() => buildHrPerLap(d2), [d2]);
  const battle1 = useMemo(() => buildBattlePerLap(d1), [d1]);
  const battle2 = useMemo(() => buildBattlePerLap(d2), [d2]);

  const mergedHr = useMemo(() => {
    const map = new Map<number, any>();
    for (const d of hr1) map.set(d.lap, { lap: d.lap, race1HR: d.avgHR });
    for (const d of hr2) map.set(d.lap, { ...map.get(d.lap), lap: d.lap, race2HR: d.avgHR });
    return Array.from(map.values()).sort((a: any, b: any) => a.lap - b.lap);
  }, [hr1, hr2]);

  const mergedBattle = useMemo(() => {
    const map = new Map<number, any>();
    for (const d of battle1) map.set(d.lap, { lap: d.lap, race1Battle: d.intensity });
    for (const d of battle2) map.set(d.lap, { ...map.get(d.lap), lap: d.lap, race2Battle: d.intensity });
    return Array.from(map.values()).sort((a: any, b: any) => a.lap - b.lap);
  }, [battle1, battle2]);

  const RACE1_COLOR = '#FF8000';
  const RACE2_COLOR = '#a78bfa';

  if (!d1.length && !d2.length) {
    return <div className="flex items-center justify-center h-[300px] text-muted-foreground text-sm">No data available for comparison</div>;
  }

  return (
    <>
      {/* Side-by-side KPIs */}
      <div className="grid grid-cols-4 gap-3">
        <DeltaKPI label="AVG HR" norVal={kpis1?.avgHR || '—'} piaVal={kpis2?.avgHR || '—'} unit=" bpm"
          icon={<Heart className="w-4 h-4 text-red-400" />} />
        <DeltaKPI label="PEAK HR" norVal={kpis1?.peakHR || '—'} piaVal={kpis2?.peakHR || '—'} unit=" bpm"
          icon={<Heart className="w-4 h-4 text-rose-500" />} />
        <DeltaKPI label="COCKPIT TEMP" norVal={kpis1?.avgTemp || '—'} piaVal={kpis2?.avgTemp || '—'} unit="°C"
          icon={<Thermometer className="w-4 h-4 text-amber-400" />} />
        <DeltaKPI label="BATTLE %" norVal={kpis1?.battleIntensity || '—'} piaVal={kpis2?.battleIntensity || '—'} unit="%"
          icon={<Swords className="w-4 h-4 text-purple-400" />} />
      </div>

      {/* Legend */}
      <div className="flex items-center gap-4 text-[12px] text-muted-foreground px-1">
        <span><span style={{ color: RACE1_COLOR }}>■</span> {race1} GP</span>
        <span><span style={{ color: RACE2_COLOR }}>■</span> {race2} GP</span>
      </div>

      {/* HR and Battle overlay */}
      <div className="grid grid-cols-2 gap-3">
        {mergedHr.length > 0 && (
          <div className="bg-card border border-border rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
            <h3 className="text-sm text-foreground mb-1">Avg HR per Lap — {race1} vs {race2}</h3>
            <p className="text-[12px] text-muted-foreground mb-3">{driver} — {year}</p>
            <div className="h-[260px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={mergedHr}>
                  <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="lap" tick={{ fill: '#8888a0', fontSize: 9 }} />
                  <YAxis tick={{ fill: '#8888a0', fontSize: 9 }} domain={['auto', 'auto']} />
                  <Tooltip content={<CustomTooltip />} />
                  <Line type="monotone" dataKey="race1HR" stroke={RACE1_COLOR} strokeWidth={1.5} dot={false} name={`${race1} HR`} />
                  <Line type="monotone" dataKey="race2HR" stroke={RACE2_COLOR} strokeWidth={1.5} dot={false} name={`${race2} HR`} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
        {mergedBattle.length > 0 && (
          <div className="bg-card border border-border rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
            <h3 className="text-sm text-foreground mb-1">Battle Intensity — {race1} vs {race2}</h3>
            <p className="text-[12px] text-muted-foreground mb-3">{driver} — {year}</p>
            <div className="h-[260px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={mergedBattle}>
                  <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="lap" tick={{ fill: '#8888a0', fontSize: 9 }} />
                  <YAxis tick={{ fill: '#8888a0', fontSize: 9 }} domain={[0, 100]} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="race1Battle" name={`${race1} %`} fill={RACE1_COLOR} radius={[2, 2, 0, 0]} />
                  <Bar dataKey="race2Battle" name={`${race2} %`} fill={RACE2_COLOR} radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>
    </>
  );
}

function YearVsYearView({ driver, season23, season24 }: { driver: string; season23: RaceSummary[]; season24: RaceSummary[] }) {
  // Season averages
  const avg = (arr: RaceSummary[], key: keyof RaceSummary) =>
    arr.length ? (arr.reduce((s, r) => s + (r[key] as number), 0) / arr.length).toFixed(1) : '—';
  const peak = (arr: RaceSummary[]) => arr.length ? Math.max(...arr.map(r => r.peakHR)).toFixed(1) : '—';

  // Common races for direct comparison
  const commonRaces = useMemo(() => {
    const races24 = new Set(season24.map(r => r.race));
    return season23.filter(r => races24.has(r.race)).map(r => r.race);
  }, [season23, season24]);

  const mergedCommon = useMemo(() => commonRaces.map(race => {
    const r23 = season23.find(r => r.race === race)!;
    const r24 = season24.find(r => r.race === race)!;
    return {
      race,
      avgHR23: r23.avgHR, avgHR24: r24.avgHR,
      peakHR23: r23.peakHR, peakHR24: r24.peakHR,
      battle23: r23.battleIntensity, battle24: r24.battleIntensity,
      temp23: r23.avgTemp, temp24: r24.avgTemp,
    };
  }), [commonRaces, season23, season24]);

  const Y23_COLOR = '#a78bfa';
  const Y24_COLOR = '#FF8000';

  if (!season23.length && !season24.length) {
    return <div className="flex items-center justify-center h-[300px] text-muted-foreground text-sm">No season data available for {driver}</div>;
  }

  return (
    <>
      <div className="grid grid-cols-4 gap-3">
        <DeltaKPI label="AVG HR" norVal={avg(season23, 'avgHR')} piaVal={avg(season24, 'avgHR')} unit=" bpm"
          icon={<Heart className="w-4 h-4 text-red-400" />} />
        <DeltaKPI label="PEAK HR" norVal={peak(season23)} piaVal={peak(season24)} unit=" bpm"
          icon={<Heart className="w-4 h-4 text-rose-500" />} />
        <DeltaKPI label="AVG TEMP" norVal={avg(season23, 'avgTemp')} piaVal={avg(season24, 'avgTemp')} unit="°C"
          icon={<Thermometer className="w-4 h-4 text-amber-400" />} />
        <DeltaKPI label="AVG BATTLE" norVal={avg(season23, 'battleIntensity')} piaVal={avg(season24, 'battleIntensity')} unit="%"
          icon={<Swords className="w-4 h-4 text-purple-400" />} />
      </div>

      {/* Legend */}
      <div className="flex items-center gap-4 text-[12px] text-muted-foreground px-1">
        <span><span style={{ color: Y23_COLOR }}>■</span> 2023 Season</span>
        <span><span style={{ color: Y24_COLOR }}>■</span> 2024 Season</span>
        <span className="ml-auto">{commonRaces.length} common races compared</span>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="bg-card border border-border rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
          <h3 className="text-sm text-foreground mb-1">Avg HR by Race — 2023 vs 2024</h3>
          <p className="text-[12px] text-muted-foreground mb-3">{driver} — common circuits</p>
          <div className="h-[280px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={mergedCommon}>
                <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="race" tick={{ fill: '#8888a0', fontSize: 8 }} angle={-45} textAnchor="end" height={60} interval={0} />
                <YAxis tick={{ fill: '#8888a0', fontSize: 9 }} domain={['auto', 'auto']} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="avgHR23" name="2023" fill={Y23_COLOR} radius={[2, 2, 0, 0]} />
                <Bar dataKey="avgHR24" name="2024" fill={Y24_COLOR} radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-card border border-border rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
          <h3 className="text-sm text-foreground mb-1">Peak HR by Race — 2023 vs 2024</h3>
          <p className="text-[12px] text-muted-foreground mb-3">{driver} — common circuits</p>
          <div className="h-[280px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={mergedCommon}>
                <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="race" tick={{ fill: '#8888a0', fontSize: 8 }} angle={-45} textAnchor="end" height={60} interval={0} />
                <YAxis tick={{ fill: '#8888a0', fontSize: 9 }} domain={['auto', 'auto']} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="peakHR23" name="2023" fill={Y23_COLOR} radius={[2, 2, 0, 0]} />
                <Bar dataKey="peakHR24" name="2024" fill={Y24_COLOR} radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="bg-card border border-border rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
          <h3 className="text-sm text-foreground mb-1">Battle Intensity — 2023 vs 2024</h3>
          <p className="text-[12px] text-muted-foreground mb-3">{driver} — common circuits</p>
          <div className="h-[280px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={mergedCommon}>
                <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="race" tick={{ fill: '#8888a0', fontSize: 8 }} angle={-45} textAnchor="end" height={60} interval={0} />
                <YAxis tick={{ fill: '#8888a0', fontSize: 9 }} domain={[0, 'auto']} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="battle23" name="2023" fill={Y23_COLOR} radius={[2, 2, 0, 0]} />
                <Bar dataKey="battle24" name="2024" fill={Y24_COLOR} radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-card border border-border rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
          <h3 className="text-sm text-foreground mb-1">Cockpit Temp — 2023 vs 2024</h3>
          <p className="text-[12px] text-muted-foreground mb-3">{driver} — common circuits</p>
          <div className="h-[280px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={mergedCommon}>
                <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="race" tick={{ fill: '#8888a0', fontSize: 8 }} angle={-45} textAnchor="end" height={60} interval={0} />
                <YAxis tick={{ fill: '#8888a0', fontSize: 9 }} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="temp23" name="2023 °C" fill={Y23_COLOR} radius={[2, 2, 0, 0]} />
                <Bar dataKey="temp24" name="2024 °C" fill={Y24_COLOR} radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </>
  );
}
