import { useState, useEffect, useMemo, useCallback } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  BarChart, Bar, AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
} from 'recharts';
import { Users, Search, Loader2, Target, Zap, ChevronRight, X, GitCompare, Disc, ArrowLeft, TrendingUp, TrendingDown, Minus, Gauge, Activity } from 'lucide-react';
import type { DriverPerformanceMarker, DriverOvertakeProfile, DriverTelemetryProfile } from '../types';
import * as api from '../api/driverIntel';
import type { DriverKex, SimilarDriver, ComparisonKex } from '../api/driverIntel';
import { HealthGauge } from './HealthGauge';
import KexBriefingCard from './KexBriefingCard';
import { StatusBadge } from './StatusBadge';
import {
  type VehicleData,
  parseAnomalyDrivers, levelColor, levelBg,
  MAINTENANCE_LABELS, SEVERITY_COLORS,
} from './anomalyHelpers';

/* ─── Constants ─── */

const teamColors: Record<string, string> = {
  'Red Bull': '#3671C6', 'McLaren': '#FF8000', 'Ferrari': '#E8002D',
  'Mercedes': '#27F4D2', 'Aston Martin': '#229971', 'Alpine': '#FF87BC',
  'Williams': '#64C4FF', 'RB': '#6692FF', 'Kick Sauber': '#52E252',
  'Haas F1 Team': '#B6BABD', 'AlphaTauri': '#6692FF', 'Alfa Romeo': '#C92D4B',
  'Racing Point': '#F596C8', 'Renault': '#FFF500', 'Toro Rosso': '#469BFF',
  'Force India': '#F596C8',
};

const NATIONALITY_FLAGS: Record<string, string> = {
  'British': '\u{1F1EC}\u{1F1E7}', 'Dutch': '\u{1F1F3}\u{1F1F1}', 'Spanish': '\u{1F1EA}\u{1F1F8}',
  'Monegasque': '\u{1F1F2}\u{1F1E8}', 'Mexican': '\u{1F1F2}\u{1F1FD}', 'Australian': '\u{1F1E6}\u{1F1FA}',
  'Canadian': '\u{1F1E8}\u{1F1E6}', 'German': '\u{1F1E9}\u{1F1EA}', 'French': '\u{1F1EB}\u{1F1F7}',
  'Finnish': '\u{1F1EB}\u{1F1EE}', 'Thai': '\u{1F1F9}\u{1F1ED}', 'Japanese': '\u{1F1EF}\u{1F1F5}',
  'Chinese': '\u{1F1E8}\u{1F1F3}', 'Danish': '\u{1F1E9}\u{1F1F0}', 'American': '\u{1F1FA}\u{1F1F8}',
  'Italian': '\u{1F1EE}\u{1F1F9}', 'Brazilian': '\u{1F1E7}\u{1F1F7}', 'New Zealander': '\u{1F1F3}\u{1F1FF}',
  'Argentine': '\u{1F1E6}\u{1F1F7}', 'Polish': '\u{1F1F5}\u{1F1F1}', 'Indonesian': '\u{1F1EE}\u{1F1E9}',
  'Russian': '\u{1F1F7}\u{1F1FA}', 'Indian': '\u{1F1EE}\u{1F1F3}', 'Belgian': '\u{1F1E7}\u{1F1EA}',
  'Colombian': '\u{1F1E8}\u{1F1F4}', 'Venezuelan': '\u{1F1FB}\u{1F1EA}', 'Swedish': '\u{1F1F8}\u{1F1EA}',
  'Malaysian': '\u{1F1F2}\u{1F1FE}', 'Austrian': '\u{1F1E6}\u{1F1F9}', 'Swiss': '\u{1F1E8}\u{1F1ED}',
  'South African': '\u{1F1FF}\u{1F1E6}', 'Irish': '\u{1F1EE}\u{1F1EA}', 'Portuguese': '\u{1F1F5}\u{1F1F9}',
  'Hungarian': '\u{1F1ED}\u{1F1FA}', 'Czech': '\u{1F1E8}\u{1F1FF}', 'Korean': '\u{1F1F0}\u{1F1F7}',
  'Chilean': '\u{1F1E8}\u{1F1F1}', 'Uruguayan': '\u{1F1FA}\u{1F1FE}', 'Peruvian': '\u{1F1F5}\u{1F1EA}',
  'Romanian': '\u{1F1F7}\u{1F1F4}', 'Norwegian': '\u{1F1F3}\u{1F1F4}', 'Emirati': '\u{1F1E6}\u{1F1EA}',
  'Saudi': '\u{1F1F8}\u{1F1E6}', 'Qatari': '\u{1F1F6}\u{1F1E6}', 'Bahraini': '\u{1F1E7}\u{1F1ED}',
};

const F1_CDN = 'https://media.formula1.com/image/upload/c_lfill,w_96/q_auto/v1740000000/common/f1/2026';
const TEAM_LOGOS: Record<string, string> = {
  red_bull: `${F1_CDN}/redbullracing/2026redbullracinglogowhite.webp`,
  mclaren: `${F1_CDN}/mclaren/2026mclarenlogowhite.webp`,
  ferrari: `${F1_CDN}/ferrari/2026ferrarilogowhite.webp`,
  mercedes: `${F1_CDN}/mercedes/2026mercedeslogowhite.webp`,
  aston_martin: `${F1_CDN}/astonmartin/2026astonmartinlogowhite.webp`,
  alpine: `${F1_CDN}/alpine/2026alpinelogowhite.webp`,
  williams: `${F1_CDN}/williams/2026williamslogowhite.webp`,
  rb: `${F1_CDN}/racingbulls/2026racingbullslogowhite.webp`,
  sauber: `${F1_CDN}/audi/2026audilogowhite.webp`,
  haas: `${F1_CDN}/haasf1team/2026haasf1teamlogowhite.webp`,
};
const TEAM_NAME_TO_LOGO: Record<string, string> = {
  'Red Bull': 'red_bull', 'McLaren': 'mclaren', 'Ferrari': 'ferrari',
  'Mercedes': 'mercedes', 'Aston Martin': 'aston_martin', 'Alpine': 'alpine',
  'Williams': 'williams', 'RB': 'rb', 'Kick Sauber': 'sauber', 'Haas F1 Team': 'haas',
  'Haas': 'haas', 'Sauber': 'sauber', 'Racing Bulls': 'rb',
};

const COMPOUND_COLORS: Record<string, string> = {
  SOFT: '#FF3333', MEDIUM: '#FFC300', HARD: '#EEEEEE',
  INTERMEDIATE: '#39B54A', WET: '#0072C6',
};

const COMPARE_COLORS = ['#FF8000', '#3671C6', '#E8002D', '#27F4D2'];

type Tab = 'overview' | 'performance' | 'compare';
type SortKey = 'name' | 'team' | 'country' | 'races' | 'health' | 'standing';

/** Compute health trend from last 3 races: 'up' | 'down' | 'stable' */
function getHealthTrend(vehicle: VehicleData | undefined): 'up' | 'down' | 'stable' {
  if (!vehicle || vehicle.races.length < 2) return 'stable';
  const recent = vehicle.races.slice(-3);
  const avgHealth = (race: typeof recent[0]) => {
    const vals = Object.values(race.systems).map(s => s.health);
    return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
  };
  const first = avgHealth(recent[0]);
  const last = avgHealth(recent[recent.length - 1]);
  const diff = last - first;
  if (diff > 3) return 'up';
  if (diff < -3) return 'down';
  return 'stable';
}

const TrendIcon = ({ trend }: { trend: 'up' | 'down' | 'stable' }) => {
  if (trend === 'up') return <TrendingUp className="w-3 h-3 text-green-400" />;
  if (trend === 'down') return <TrendingDown className="w-3 h-3 text-red-400" />;
  return <Minus className="w-3 h-3 text-muted-foreground/50" />;
};

/* ─── Utilities ─── */

function normalize(val: number | null, min: number, max: number, invert = false): number {
  if (val == null) return 0;
  const clamped = Math.max(min, Math.min(max, val));
  const pct = ((clamped - min) / (max - min)) * 100;
  return invert ? 100 - pct : pct;
}

function formatDriverName(id: string) {
  return id.replace(/_/g, ' ').replace(/\b\w/g, (c: string) => c.toUpperCase());
}

/* ─── Shared Components ─── */

function Divider({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-3 py-1">
      <div className="h-px flex-1 bg-[rgba(255,128,0,0.10)]" />
      <span className="text-[10px] tracking-[0.25em] text-[#FF8000]/60 font-semibold select-none">{label}</span>
      <div className="h-px flex-1 bg-[rgba(255,128,0,0.10)]" />
    </div>
  );
}

function StatCard({ label, value, unit, precision = 2, accent }: {
  label: string; value: number | null | undefined; unit: string; precision?: number; accent?: boolean;
}) {
  const display = value != null ? value.toFixed(precision) : '\u2014';
  return (
    <div className={`rounded-lg p-3 transition-colors ${accent ? 'bg-[#FF8000]/5 border border-[#FF8000]/20' : 'bg-background border border-border'}`}>
      <div className="text-[10px] text-muted-foreground mb-1.5 tracking-wide uppercase">{label}</div>
      <div className="flex items-baseline gap-1.5">
        <span className={`font-mono text-lg leading-none ${accent ? 'text-[#FF8000]' : 'text-foreground'}`}>{display}</span>
        <span className="text-[10px] text-muted-foreground">{unit}</span>
      </div>
    </div>
  );
}

function CustomTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-background/95 backdrop-blur-sm border border-[rgba(255,128,0,0.15)] rounded-lg px-3 py-2 shadow-lg">
      {payload.map((p: any) => (
        <div key={p.name} className="flex items-center gap-2 text-[11px]">
          <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: p.color || p.stroke }} />
          <span className="text-muted-foreground">{p.name}</span>
          <span className="font-mono text-foreground ml-auto">{typeof p.value === 'number' ? p.value.toFixed(2) : p.value}</span>
        </div>
      ))}
    </div>
  );
}

function SectionHeader({ icon: Icon, label, sub }: { icon: React.ElementType; label: string; sub?: string }) {
  return (
    <div className="flex items-center gap-2.5 mb-3">
      <div className="w-7 h-7 rounded-lg bg-[#FF8000]/10 flex items-center justify-center">
        <Icon className="w-3.5 h-3.5 text-[#FF8000]" />
      </div>
      <div>
        <h3 className="text-sm font-semibold text-foreground leading-none">{label}</h3>
        {sub && <p className="text-[10px] text-muted-foreground mt-0.5">{sub}</p>}
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════
   MAIN EXPORT
   ═══════════════════════════════════════════════ */

export function DriverIntel({ showTabBar = true, prefetchedVehicles }: { showTabBar?: boolean; prefetchedVehicles?: VehicleData[] }) {
  const [tab, setTab] = useState<Tab>('overview');
  const [selectedDriver, setSelectedDriver] = useState<string | null>(null);
  const [anomalyVehicles, setAnomalyVehicles] = useState<VehicleData[]>(prefetchedVehicles ?? []);

  useEffect(() => {
    if (prefetchedVehicles && prefetchedVehicles.length > 0) {
      setAnomalyVehicles(prefetchedVehicles);
      return;
    }
    fetch('/api/pipeline/anomaly')
      .then(r => r.json())
      .then(data => setAnomalyVehicles(parseAnomalyDrivers(data)))
      .catch(() => {});
  }, [prefetchedVehicles]);

  const tabs: { id: Tab; label: string; icon: React.ElementType }[] = [
    { id: 'overview', label: 'Driver Grid', icon: Users },
    { id: 'performance', label: 'Profile', icon: Target },
    { id: 'compare', label: 'Compare', icon: GitCompare },
  ];

  return (
    <div className="space-y-5">
      {showTabBar && (
        <div className="flex items-center gap-1 bg-background/60 p-1 rounded-lg w-fit border border-border">
          {tabs.map(t => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`relative text-[13px] px-4 py-2 rounded-lg transition-all flex items-center gap-2 font-medium ${
                tab === t.id
                  ? 'bg-[#FF8000]/10 text-[#FF8000]'
                  : 'text-muted-foreground hover:text-foreground hover:bg-card/60'
              }`}
            >
              <t.icon className="w-3.5 h-3.5" />
              {t.label}
            </button>
          ))}
        </div>
      )}

      <AnimatePresence mode="wait">
        <motion.div
          key={tab}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          transition={{ duration: 0.2 }}
        >
          {tab === 'overview' && <DriverGrid onSelect={(d) => { setSelectedDriver(d); setTab('performance'); }} anomalyVehicles={anomalyVehicles} />}
          {tab === 'performance' && <PerformanceProfile driverCode={selectedDriver} onSelect={setSelectedDriver} anomalyVehicles={anomalyVehicles} onCompare={() => setTab('compare')} onBack={() => setTab('overview')} />}
          {tab === 'compare' && <CompareDrivers />}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}

/* ═══════════════════════════════════════════════
   DRIVER GRID
   ═══════════════════════════════════════════════ */

function DriverGrid({ onSelect, anomalyVehicles }: { onSelect: (code: string) => void; anomalyVehicles: VehicleData[] }) {
  const [drivers, setDrivers] = useState<any[]>([]);
  const [standings, setStandings] = useState<Record<string, { position: number; points: number }>>({});
  const [perfMarkers, setPerfMarkers] = useState<Record<string, any>>({});
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [sortBy, setSortBy] = useState<SortKey>('name');
  const [sortAsc, setSortAsc] = useState(true);

  useEffect(() => {
    api.getOpponentDrivers().then(data => {
      setDrivers(data.drivers || []);
      setLoading(false);
    }).catch(() => setLoading(false));

    // Fetch driver performance markers for card stats
    fetch('/api/driver_intel/performance_markers')
      .then(r => r.json())
      .then((data: any[]) => {
        if (!Array.isArray(data)) return;
        const map: Record<string, any> = {};
        data.forEach(m => { if (m.Driver) map[m.Driver] = m; });
        setPerfMarkers(map);
      })
      .catch(() => {});

    fetch('/api/jolpica/driver_standings')
      .then(r => r.json())
      .then((data: any[]) => {
        const map: Record<string, { position: number; points: number }> = {};
        if (!Array.isArray(data)) return;
        // Get latest season standings per driver
        const latestSeason = Math.max(...data.map(s => s.season ?? 0), 0);
        data
          .filter(s => s.season === latestSeason)
          .forEach(s => {
            if (s.driver_code) map[s.driver_code] = { position: Number(s.position ?? 0), points: Number(s.points ?? 0) };
          });
        setStandings(map);
      })
      .catch(() => {});
  }, []);

  const filtered = useMemo(() => {
    if (!search) return drivers;
    const q = search.toLowerCase();
    return drivers.filter((d: any) =>
      (d.driver_id || '').toLowerCase().includes(q) ||
      (d.team || '').toLowerCase().includes(q) ||
      (d.nationality || '').toLowerCase().includes(q)
    );
  }, [drivers, search]);

  const sorted = useMemo(() => {
    const arr = [...filtered];
    const dir = sortAsc ? 1 : -1;
    arr.sort((a: any, b: any) => {
      switch (sortBy) {
        case 'name': return dir * (a.driver_id || '').localeCompare(b.driver_id || '');
        case 'team': return dir * (a.team || '').localeCompare(b.team || '');
        case 'country': return dir * (a.nationality || '').localeCompare(b.nationality || '');
        case 'races': return dir * ((a.total_races ?? 0) - (b.total_races ?? 0));
        case 'health': {
          const codeA = a.driver_code || a.code || a.driver_id?.slice(0, 3).toUpperCase();
          const codeB = b.driver_code || b.code || b.driver_id?.slice(0, 3).toUpperCase();
          const hA = anomalyVehicles.find(v => v.code === codeA)?.overallHealth ?? 0;
          const hB = anomalyVehicles.find(v => v.code === codeB)?.overallHealth ?? 0;
          return dir * (hA - hB);
        }
        case 'standing': {
          const codeA = a.driver_code || a.code || a.driver_id?.slice(0, 3).toUpperCase();
          const codeB = b.driver_code || b.code || b.driver_id?.slice(0, 3).toUpperCase();
          const pA = standings[codeA]?.position ?? 99;
          const pB = standings[codeB]?.position ?? 99;
          return dir * (pA - pB);
        }
        default: return 0;
      }
    });
    return arr;
  }, [filtered, sortBy, sortAsc, anomalyVehicles, standings]);

  const toggleSort = (key: SortKey) => {
    if (sortBy === key) setSortAsc(!sortAsc);
    else { setSortBy(key); setSortAsc(key === 'name' || key === 'team' || key === 'country'); }
  };

  if (loading) {
    return <div className="flex items-center justify-center py-20"><Loader2 className="w-6 h-6 text-[#FF8000] animate-spin" /></div>;
  }

  const sortBtn = (key: SortKey, label: string) => (
    <button
      type="button"
      onClick={() => toggleSort(key)}
      className={`px-2.5 py-1 rounded-md text-[11px] font-medium transition-colors ${
        sortBy === key
          ? 'bg-[#FF8000]/15 text-[#FF8000] border border-[#FF8000]/25'
          : 'text-muted-foreground hover:text-foreground border border-transparent hover:border-border'
      }`}
    >
      {label} {sortBy === key ? (sortAsc ? '\u2191' : '\u2193') : ''}
    </button>
  );

  const criticalCount = anomalyVehicles.filter(v => v.level === 'critical').length;
  const warningCount = anomalyVehicles.filter(v => v.level === 'warning').length;
  const nominalCount = anomalyVehicles.filter(v => v.level === 'nominal').length;
  const avgHealth = anomalyVehicles.length
    ? anomalyVehicles.reduce((sum, v) => sum + v.overallHealth, 0) / anomalyVehicles.length
    : 0;

  return (
    <div className="space-y-4">
      {/* Fleet Summary Bar */}
      {anomalyVehicles.length > 0 && (
        <div className="flex items-center gap-4 px-4 py-2.5 bg-background rounded-lg border border-border text-[12px]">
          <span className="text-muted-foreground font-medium">{anomalyVehicles.length} drivers monitored</span>
          <div className="w-px h-4 bg-[rgba(255,128,0,0.12)]" />
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-red-400" />
            <span className="text-red-400 font-mono">{criticalCount}</span>
            <span className="text-muted-foreground/60">critical</span>
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-[#FF8000]" />
            <span className="text-[#FF8000] font-mono">{warningCount}</span>
            <span className="text-muted-foreground/60">warning</span>
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-green-400" />
            <span className="text-green-400 font-mono">{nominalCount}</span>
            <span className="text-muted-foreground/60">nominal</span>
          </span>
          <div className="w-px h-4 bg-[rgba(255,128,0,0.12)]" />
          <span className="text-muted-foreground">Avg Health</span>
          <span className="font-mono font-medium" style={{ color: avgHealth >= 80 ? '#4ade80' : avgHealth >= 60 ? '#FF8000' : '#f87171' }}>
            {avgHealth.toFixed(1)}%
          </span>
        </div>
      )}

      {/* Toolbar */}
      <div className="flex items-center gap-3 flex-wrap">
        <div className="relative w-72">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground pointer-events-none" />
          <input
            type="text"
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search drivers, teams\u2026"
            className="w-full pl-10 pr-4 py-2 text-sm bg-background border border-[rgba(255,128,0,0.10)] rounded-lg text-foreground placeholder:text-muted-foreground/60 focus:outline-none focus:border-[#FF8000]/30 transition-colors"
          />
        </div>
        <div className="flex items-center gap-1">
          <span className="text-[10px] text-muted-foreground mr-1 uppercase tracking-wider">Sort</span>
          {sortBtn('name', 'Name')}
          {sortBtn('team', 'Team')}
          {sortBtn('country', 'Country')}
          {sortBtn('races', 'Races')}
          {sortBtn('health', 'Health')}
          {sortBtn('standing', 'Standing')}
        </div>
        <span className="text-[11px] text-muted-foreground ml-auto font-mono">{sorted.length} drivers</span>
      </div>

      {/* ── McLaren Drivers — prominent cards ── */}
      {(() => {
        const mclarenDrivers = sorted.filter((d: any) => (d.team || '') === 'McLaren');
        const otherDrivers = sorted.filter((d: any) => (d.team || '') !== 'McLaren');

        const renderDriverCard = (d: any, i: number, isMcLaren: boolean) => {
          const dCode = d.driver_code || d.code || d.driver_id?.slice(0, 3).toUpperCase();
          const vehicle = anomalyVehicles.find(v => v.code === dCode);
          const team = d.team || vehicle?.team || '';
          const color = teamColors[team] || '#666';
          const standing = standings[dCode];
          const trend = getHealthTrend(vehicle);
          const perf = perfMarkers[dCode];
          return (
            <motion.button
              key={d.driver_id}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.25, delay: Math.min(i * 0.02, 0.4) }}
              onClick={() => onSelect(d.driver_id)}
              className={`text-left rounded-lg transition-all group relative overflow-hidden ${
                isMcLaren
                  ? 'bg-card border border-[#FF8000]/25 hover:border-[#FF8000]/50 shadow-[0_0_20px_rgba(255,128,0,0.06)]'
                  : 'bg-card border border-border hover:border-[rgba(255,128,0,0.25)]'
              }`}
            >
              {/* Team color left accent stripe */}
              <div className={`absolute top-0 left-0 bottom-0 rounded-l-xl ${isMcLaren ? 'w-[4px]' : 'w-[3px]'}`} style={{ background: color }} />

              <div className={isMcLaren ? 'p-5 pl-6' : 'p-4 pl-5'}>
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-1.5 min-w-0">
                    {TEAM_NAME_TO_LOGO[team] && TEAM_LOGOS[TEAM_NAME_TO_LOGO[team]] && (
                      <img src={TEAM_LOGOS[TEAM_NAME_TO_LOGO[team]]} alt={team} className={`object-contain shrink-0 ${isMcLaren ? 'h-5 w-5' : 'h-4 w-4'}`} />
                    )}
                    <span className={`font-medium tracking-wide truncate ${isMcLaren ? 'text-[12px]' : 'text-[11px]'}`} style={{ color }}>{team}</span>
                  </div>
                  <div className="flex items-center gap-1.5 shrink-0">
                    {vehicle && <TrendIcon trend={trend} />}
                    {vehicle && <HealthGauge value={vehicle.overallHealth} size={isMcLaren ? 38 : 32} showLabel={false} strokeWidth={3} />}
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  {vehicle && vehicle.number > 0 && (
                    <span className={`font-black font-mono leading-none opacity-25 ${isMcLaren ? 'text-[22px]' : 'text-[18px]'}`} style={{ color }}>
                      {vehicle.number}
                    </span>
                  )}
                  <div className={`text-foreground font-bold leading-tight group-hover:text-[#FF8000] transition-colors ${isMcLaren ? 'text-[17px]' : 'text-[15px]'}`}>
                    {formatDriverName(d.driver_id)}
                  </div>
                </div>

                {standing && (
                  <div className="flex items-center gap-2 mt-1.5 text-[11px]">
                    <span className="font-mono font-bold text-[#FF8000]">P{standing.position}</span>
                    <span className="text-muted-foreground font-mono">{standing.points} pts</span>
                  </div>
                )}

                <div className="flex items-center gap-2.5 mt-2 text-[11px] text-muted-foreground">
                  {d.nationality && NATIONALITY_FLAGS[d.nationality] && (
                    <span className="text-sm leading-none">{NATIONALITY_FLAGS[d.nationality]}</span>
                  )}
                  {d.total_races != null && <span className="font-mono">{d.total_races} <span className="text-muted-foreground/60">races</span></span>}
                  {d.wins != null && d.wins > 0 && <span className="font-mono text-[#FF8000]">{d.wins}W</span>}
                </div>

                {/* Driver performance stats */}
                {perf && (
                  <div className={`mt-3 flex flex-wrap gap-x-3 gap-y-1 ${isMcLaren ? 'text-[11px]' : 'text-[10px]'}`}>
                    {perf.avg_top_speed_kmh != null && (
                      <div><span className="text-muted-foreground/60">Top Spd </span><span className="font-mono font-semibold text-foreground">{Math.round(perf.avg_top_speed_kmh)}</span></div>
                    )}
                    {perf.throttle_smoothness != null && (
                      <div><span className="text-muted-foreground/60">Throttle </span><span className="font-mono font-semibold text-foreground">{perf.throttle_smoothness.toFixed(2)}</span></div>
                    )}
                    {perf.lap_time_consistency_std != null && (
                      <div><span className="text-muted-foreground/60">Consistency </span><span className="font-mono font-semibold text-foreground">{perf.lap_time_consistency_std.toFixed(2)}s</span></div>
                    )}
                    {perf.late_race_delta_s != null && (
                      <div>
                        <span className="text-muted-foreground/60">Late Race </span>
                        <span className={`font-mono font-semibold ${perf.late_race_delta_s < 0 ? 'text-green-400' : 'text-red-400'}`}>
                          {perf.late_race_delta_s > 0 ? '+' : ''}{perf.late_race_delta_s.toFixed(2)}s
                        </span>
                      </div>
                    )}
                    {perf.total_race_laps != null && (
                      <div><span className="text-muted-foreground/60">Laps </span><span className="font-mono font-semibold text-foreground">{perf.total_race_laps}</span></div>
                    )}
                  </div>
                )}

                <ChevronRight className="w-3.5 h-3.5 text-muted-foreground/30 absolute bottom-3 right-3 group-hover:text-[#FF8000]/60 transition-colors" />
              </div>
            </motion.button>
          );
        };

        return (
          <>
            {mclarenDrivers.length > 0 && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {mclarenDrivers.map((d: any, i: number) => renderDriverCard(d, i, true))}
              </div>
            )}

            {otherDrivers.length > 0 && mclarenDrivers.length > 0 && (
              <div className="flex items-center gap-3 mt-2">
                <div className="h-px flex-1 bg-[rgba(255,128,0,0.08)]" />
                <span className="text-[10px] text-muted-foreground/50 uppercase tracking-widest">All Drivers</span>
                <div className="h-px flex-1 bg-[rgba(255,128,0,0.08)]" />
              </div>
            )}

            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
              {otherDrivers.map((d: any, i: number) => renderDriverCard(d, i, false))}
            </div>
          </>
        );
      })()}

      {filtered.length === 0 && (
        <div className="text-center py-12 text-muted-foreground text-sm">No drivers match your search.</div>
      )}
    </div>
  );
}

/* ═══════════════════════════════════════════════
   PERFORMANCE PROFILE
   ═══════════════════════════════════════════════ */

function PerformanceProfile({ driverCode, onSelect, anomalyVehicles, onCompare, onBack }: {
  driverCode: string | null;
  onSelect: (code: string) => void;
  anomalyVehicles: VehicleData[];
  onCompare?: () => void;
  onBack?: () => void;
}) {
  const [markers, setMarkers] = useState<DriverPerformanceMarker[]>([]);
  const [overtakes, setOvertakes] = useState<DriverOvertakeProfile[]>([]);
  const [telemetry, setTelemetry] = useState<DriverTelemetryProfile[]>([]);
  const [allMarkers, setAllMarkers] = useState<DriverPerformanceMarker[]>([]);
  const [allOvertakes, setAllOvertakes] = useState<DriverOvertakeProfile[]>([]);
  const [drivers, setDrivers] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [compounds, setCompounds] = useState<any[]>([]);
  const [kex, setKex] = useState<DriverKex | null>(null);
  const [kexLoading, setKexLoading] = useState(false);

  useEffect(() => {
    setKex(null);
    if (!driverCode) return;
    const driver = drivers.find((d: any) => d.driver_id === driverCode);
    const code = driver?.driver_code || driver?.code || driverCode.slice(0, 3).toUpperCase();
    setKexLoading(true);
    api.getDriverKex(code)
      .then(setKex)
      .catch(() => setKex(null))
      .finally(() => setKexLoading(false));
  }, [driverCode, drivers]);

  useEffect(() => {
    Promise.all([
      api.getPerformanceMarkers(),
      api.getOvertakeProfiles(),
      api.getTelemetryProfiles(),
      api.getOpponentDrivers(),
    ]).then(([m, o, t, d]) => {
      setAllMarkers(m);
      setAllOvertakes(o);
      setTelemetry(t);
      setDrivers(d.drivers || []);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!driverCode) return;
    const driver = drivers.find((d: any) => d.driver_id === driverCode);
    const code = driver?.driver_code || driver?.code || driverCode.slice(0, 3).toUpperCase();
    setMarkers(allMarkers.filter(m => m.Driver === code));
    setOvertakes(allOvertakes.filter(o => o.driver_code === code));
    api.getDriverCompounds(driverCode).then(setCompounds).catch(() => setCompounds([]));
  }, [driverCode, drivers, allMarkers, allOvertakes]);

  const driverTelemetry = useMemo(() => {
    if (!driverCode || !drivers.length) return null;
    const driver = drivers.find((d: any) => d.driver_id === driverCode);
    const code = driver?.driver_code || driver?.code || driverCode.slice(0, 3).toUpperCase();
    return telemetry.find(t => t.driver_code === code) || null;
  }, [driverCode, drivers, telemetry]);

  const radarData = useMemo(() => {
    const m = markers[0];
    const o = overtakes[0];
    const t = driverTelemetry;
    if (!m && !o && !t) return [];
    return [
      { metric: 'Consistency', value: normalize(m?.lap_time_consistency_std ?? null, 0, 30, true), fullMark: 100 },
      { metric: 'Tyre Mgmt', value: normalize(m?.degradation_slope_s_per_lap ?? null, -0.3, 0.1, true), fullMark: 100 },
      { metric: 'Overtaking', value: normalize(o?.overtake_ratio ?? null, 0.5, 1.5, false), fullMark: 100 },
      { metric: 'Late Race', value: normalize(m?.late_race_delta_s ?? null, -30, 5, true), fullMark: 100 },
      { metric: 'Top Speed', value: normalize(t?.avg_race_speed_kmh ?? null, 170, 220, false), fullMark: 100 },
      { metric: 'Braking', value: normalize(t?.avg_braking_g ?? null, 2, 5, false), fullMark: 100 },
    ];
  }, [markers, overtakes, driverTelemetry]);

  const overtakeBarData = useMemo(() => {
    const o = overtakes[0];
    if (!o) return [];
    const avgMade = allOvertakes.reduce((s, x) => s + x.overtakes_per_race, 0) / (allOvertakes.length || 1);
    const avgLost = allOvertakes.reduce((s, x) => s + x.times_overtaken_per_race, 0) / (allOvertakes.length || 1);
    return [
      { metric: 'OT Made/Race', driver: o.overtakes_per_race, avg: +avgMade.toFixed(1) },
      { metric: 'OT Lost/Race', driver: o.times_overtaken_per_race, avg: +avgLost.toFixed(1) },
      { metric: 'OT Ratio', driver: o.overtake_ratio, avg: +(allOvertakes.reduce((s, x) => s + x.overtake_ratio, 0) / (allOvertakes.length || 1)).toFixed(2) },
    ];
  }, [overtakes, allOvertakes]);

  const selectedVehicle = useMemo(() => {
    if (!driverCode || anomalyVehicles.length === 0) return null;
    const driver = drivers.find((d: any) => d.driver_id === driverCode);
    const code = driver?.driver_code || driver?.code || driverCode.slice(0, 3).toUpperCase();
    return anomalyVehicles.find(v => v.code === code) ?? null;
  }, [driverCode, drivers, anomalyVehicles]);

  const healthTrendData = useMemo(() => {
    if (!selectedVehicle) return [];
    return selectedVehicle.races.slice(-10).map(r => {
      const row: Record<string, any> = { race: r.race };
      for (const [sysName, sys] of Object.entries(r.systems)) {
        row[sysName] = sys.health;
      }
      return row;
    });
  }, [selectedVehicle]);

  if (loading) {
    return <div className="flex items-center justify-center py-20"><Loader2 className="w-6 h-6 text-[#FF8000] animate-spin" /></div>;
  }

  if (!driverCode) {
    return (
      <div className="text-center py-20 text-muted-foreground">
        <Target className="w-8 h-8 mx-auto mb-3 opacity-50" />
        <p className="text-sm">Select a driver from the Driver Grid tab to view their profile.</p>
      </div>
    );
  }

  const driverName = formatDriverName(driverCode);
  const m = markers[0];
  const t = driverTelemetry;
  const driverObj = drivers.find((d: any) => d.driver_id === driverCode);
  const teamName = driverObj?.team || '';
  const teamColor = teamColors[teamName] || '#666';

  return (
    <div className="space-y-5">
      {/* ── Hero Header ── */}
      <div className="relative bg-card border border-[rgba(255,128,0,0.10)] rounded-lg overflow-hidden">
        {/* Team color gradient accent */}
        <div className="absolute inset-0 opacity-[0.04]" style={{ background: `linear-gradient(135deg, ${teamColor} 0%, transparent 60%)` }} />
        <div className="absolute top-0 left-0 right-0 h-[2px]" style={{ background: `linear-gradient(90deg, ${teamColor}, ${teamColor}40, transparent)` }} />

        <div className="relative px-5 py-4 flex items-center gap-4">
          {onBack && (
            <button type="button" aria-label="Back to driver grid" onClick={onBack} className="w-8 h-8 rounded-lg bg-background/60 flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-background transition-colors">
              <ArrowLeft className="w-4 h-4" />
            </button>
          )}

          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-3">
              <h2 className="text-xl font-bold text-foreground truncate">{driverName}</h2>
              {teamName && (
                <span className="flex items-center gap-1.5 text-[11px] font-semibold tracking-wide px-2.5 py-0.5 rounded-md" style={{ color: teamColor, background: `${teamColor}15`, border: `1px solid ${teamColor}25` }}>
                  {TEAM_NAME_TO_LOGO[teamName] && TEAM_LOGOS[TEAM_NAME_TO_LOGO[teamName]] && (
                    <img src={TEAM_LOGOS[TEAM_NAME_TO_LOGO[teamName]]} alt={teamName} className="h-4 w-4 object-contain" />
                  )}
                  {teamName}
                </span>
              )}
            </div>
            {driverObj?.nationality && (
              <div className="flex items-center gap-1.5 mt-1 text-[12px] text-muted-foreground">
                {NATIONALITY_FLAGS[driverObj.nationality] && <span className="text-sm">{NATIONALITY_FLAGS[driverObj.nationality]}</span>}
                {driverObj.nationality}
                {driverObj.total_races != null && <span className="ml-3 font-mono">{driverObj.total_races} races</span>}
                {driverObj.wins != null && driverObj.wins > 0 && <span className="ml-2 font-mono text-[#FF8000]">{driverObj.wins} wins</span>}
              </div>
            )}
          </div>

          {/* Driver selector */}
          <select
            value={driverCode}
            onChange={e => onSelect(e.target.value)}
            aria-label="Select driver"
            className="bg-background border border-[rgba(255,128,0,0.10)] rounded-lg px-3 py-2 text-[12px] text-foreground focus:outline-none focus:border-[#FF8000]/30 transition-colors cursor-pointer"
          >
            {drivers.map((d: any) => (
              <option key={d.driver_id} value={d.driver_id}>
                {formatDriverName(d.driver_id)} — {d.team || ''}
              </option>
            ))}
          </select>

          {selectedVehicle && <HealthGauge value={selectedVehicle.overallHealth} size={52} showLabel={false} strokeWidth={5} />}
        </div>
      </div>

      {/* ── Main Content Grid ── */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">

        {/* LEFT: Radar + Compound Breakdown */}
        <div className="lg:col-span-5 space-y-4">

          {/* Radar Chart */}
          <div className="bg-card border border-border rounded-lg p-4">
            <SectionHeader icon={Target} label="Performance Radar" sub="Normalized 0–100 across 6 dimensions" />
            {radarData.length > 0 ? (
              <>
                <ResponsiveContainer width="100%" height={260}>
                  <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="72%">
                    <PolarGrid stroke="rgba(255,255,255,0.05)" gridType="polygon" />
                    <PolarAngleAxis dataKey="metric" tick={{ fill: '#9ca3af', fontSize: 10, fontWeight: 500 }} />
                    <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                    <Radar name={driverName} dataKey="value" stroke="#FF8000" strokeWidth={2} fill="#FF8000" fillOpacity={0.15} dot={{ r: 3, fill: '#FF8000', strokeWidth: 0 }} />
                  </RadarChart>
                </ResponsiveContainer>
                <div className="grid grid-cols-6 gap-1 mt-1">
                  {radarData.map(d => (
                    <div key={d.metric} className="text-center">
                      <div className="text-[9px] text-muted-foreground/70 truncate">{d.metric}</div>
                      <div className="text-[12px] font-mono font-semibold text-foreground">{d.value.toFixed(0)}</div>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <div className="h-[260px] flex items-center justify-center text-muted-foreground text-sm">No data available</div>
            )}
          </div>

          {/* Compound Breakdown */}
          {compounds.length > 0 && (
            <div className="bg-card border border-border rounded-lg p-4">
              <SectionHeader icon={Disc} label="Compound Profiles" sub="Per-tyre performance breakdown" />
              <div className="space-y-2">
                {['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']
                  .map(name => compounds.find((c: any) => c.compound === name))
                  .filter(Boolean)
                  .map((c: any) => {
                    const color = COMPOUND_COLORS[c.compound] ?? '#888';
                    const maxLife = Math.max(...compounds.map((x: any) => x.avg_tyre_life ?? 0), 1);
                    const lifePct = ((c.avg_tyre_life ?? 0) / maxLife) * 100;
                    return (
                      <div key={c.compound} className="bg-background rounded-lg p-3 border border-[rgba(255,128,0,0.04)]">
                        <div className="flex items-center gap-2.5 mb-2">
                          <span className="w-3 h-3 rounded-full shrink-0" style={{ background: color }} />
                          <span className="text-[12px] font-semibold text-foreground flex-1">{c.compound}</span>
                          <span className="text-[10px] text-muted-foreground font-mono">{c.total_laps} laps sampled</span>
                        </div>
                        <div className="grid grid-cols-2 gap-3">
                          <div>
                            <div className="text-[9px] text-muted-foreground/70 uppercase tracking-wider mb-1">Degradation</div>
                            <span className="text-sm font-mono text-foreground">
                              {c.degradation_slope != null ? `${c.degradation_slope.toFixed(4)}` : '\u2014'}
                            </span>
                            <span className="text-[9px] text-muted-foreground ml-1">s/lap</span>
                          </div>
                          <div>
                            <div className="text-[9px] text-muted-foreground/70 uppercase tracking-wider mb-1">Avg Life</div>
                            <div className="flex items-center gap-2">
                              <span className="text-sm font-mono text-foreground">
                                {c.avg_tyre_life != null ? c.avg_tyre_life.toFixed(1) : '\u2014'}
                              </span>
                              <span className="text-[9px] text-muted-foreground">laps</span>
                            </div>
                            <div className="h-1 bg-secondary rounded-full mt-1.5 overflow-hidden">
                              <div className="h-full rounded-full transition-all duration-500" style={{ width: `${lifePct}%`, backgroundColor: color }} />
                            </div>
                          </div>
                        </div>
                      </div>
                    );
                  })}
              </div>
            </div>
          )}

          {/* Overtake Chart */}
          {overtakeBarData.length > 0 && (
            <div className="bg-card border border-border rounded-lg p-4">
              <SectionHeader icon={TrendingUp} label="Overtaking" sub="vs grid average" />
              <ResponsiveContainer width="100%" height={160}>
                <BarChart data={overtakeBarData} layout="vertical">
                  <XAxis type="number" tick={{ fill: '#666', fontSize: 10 }} />
                  <YAxis type="category" dataKey="metric" tick={{ fill: '#888', fontSize: 10 }} width={90} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="driver" name={driverName} fill="#FF8000" radius={[0, 4, 4, 0]} />
                  <Bar dataKey="avg" name="Grid Avg" fill="#333" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        {/* RIGHT: Metrics + Telemetry */}
        <div className="lg:col-span-7 space-y-4">

          {/* Key Metrics */}
          <div className="bg-card border border-border rounded-lg p-4">
            <SectionHeader icon={Zap} label="Key Metrics" />
            <div className="grid grid-cols-2 xl:grid-cols-4 gap-2.5">
              <StatCard label="Tyre Degradation" value={m?.degradation_slope_s_per_lap} unit="s/lap" precision={3} accent />
              <StatCard label="Late Race Delta" value={m?.late_race_delta_s} unit="s" precision={2} />
              <StatCard label="Consistency" value={m?.lap_time_consistency_std} unit="std" precision={2} />
              <StatCard label="Avg Stint" value={m?.avg_stint_length} unit="laps" precision={1} />
              <StatCard label="Sector 1 CV" value={m?.sector1_cv} unit="%" precision={2} />
              <StatCard label="Sector 2 CV" value={m?.sector2_cv} unit="%" precision={2} />
              <StatCard label="Heat Sensitivity" value={m?.heat_lap_delta_s} unit="s" precision={2} />
              <StatCard label="Humidity Effect" value={m?.humidity_lap_delta_s} unit="s" precision={2} />
            </div>
          </div>

          {/* Telemetry Style */}
          {t && (
            <div className="bg-card border border-border rounded-lg p-4">
              <SectionHeader icon={Activity} label="Telemetry Style" />
              <div className="grid grid-cols-2 xl:grid-cols-3 gap-2.5">
                <StatCard label="Avg Speed" value={t.avg_race_speed_kmh} unit="km/h" precision={1} />
                <StatCard label="Avg Braking G" value={t.avg_braking_g} unit="G" precision={2} />
                <StatCard label="Full Throttle" value={t.full_throttle_ratio * 100} unit="%" precision={1} />
                <StatCard label="DRS Gain" value={t.drs_speed_gain_kmh} unit="km/h" precision={1} />
                <StatCard label="Brake\u2192Throttle" value={t.brake_to_throttle_avg_s * 1000} unit="ms" precision={0} />
                <StatCard label="Braking Consistency" value={t.braking_consistency} unit="std" precision={2} />
              </div>
            </div>
          )}

          {/* Compare CTA */}
          {onCompare && (
            <button
              onClick={onCompare}
              className="w-full bg-card border border-dashed border-[rgba(255,128,0,0.20)] rounded-lg p-3.5 flex items-center justify-center gap-3 hover:border-[#FF8000]/40 hover:bg-[#FF8000]/[0.03] transition-all group"
            >
              <GitCompare className="w-4 h-4 text-[#FF8000]/50 group-hover:text-[#FF8000]" />
              <span className="text-[13px] text-muted-foreground group-hover:text-foreground transition-colors">
                Compare <span className="text-[#FF8000] font-medium">{driverName}</span> against other drivers
              </span>
              <ChevronRight className="w-4 h-4 text-muted-foreground/40 group-hover:text-[#FF8000]/60" />
            </button>
          )}

          {/* System Health */}
          {selectedVehicle && (
            <>
              <Divider label="SYSTEM HEALTH" />

              <div className="grid grid-cols-12 gap-3">
                {/* Overall gauge */}
                <div className="col-span-3 bg-card border border-border rounded-lg p-4 flex flex-col items-center justify-center gap-2">
                  <HealthGauge value={selectedVehicle.overallHealth} size={90} label="Overall" />
                  <StatusBadge status={selectedVehicle.level} />
                  <div className="w-full space-y-1.5 mt-1.5">
                    <div className="flex items-center justify-between text-[10px]">
                      <span className="text-muted-foreground">DNF Risk</span>
                      <span className="font-mono" style={{ color: selectedVehicle.overallHealth >= 80 ? '#22c55e' : selectedVehicle.overallHealth >= 60 ? '#FF8000' : '#ef4444' }}>
                        {Math.max(0, 100 - selectedVehicle.overallHealth).toFixed(0)}%
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-[10px]">
                      <span className="text-muted-foreground">Last Race</span>
                      <span className="font-mono text-foreground text-[9px]">{selectedVehicle.lastRace}</span>
                    </div>
                  </div>
                </div>

                {/* System cards */}
                <div className="col-span-9 grid grid-cols-3 gap-2.5">
                  {selectedVehicle.systems.map((sys) => {
                    const Icon = sys.icon;
                    const maint = sys.maintenanceAction ? MAINTENANCE_LABELS[sys.maintenanceAction] ?? MAINTENANCE_LABELS.none : null;
                    return (
                      <div key={sys.name} className="bg-card border border-border rounded-lg p-3">
                        <div className="flex items-center gap-2 mb-2.5">
                          <div className="w-6 h-6 rounded-md flex items-center justify-center" style={{ backgroundColor: levelBg(sys.level) }}>
                            <Icon className="w-3.5 h-3.5" style={{ color: levelColor(sys.level) }} />
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="text-[10px] text-muted-foreground truncate">{sys.name}</div>
                            <div className="text-[13px] font-mono font-semibold" style={{ color: levelColor(sys.level) }}>{sys.health.toFixed(0)}%</div>
                          </div>
                          <StatusBadge status={sys.level} size="sm" />
                        </div>

                        <div className="w-full h-1 bg-secondary rounded-full overflow-hidden mb-2.5">
                          <div className="h-full rounded-full transition-all duration-500" style={{ width: `${sys.health}%`, backgroundColor: levelColor(sys.level) }} />
                        </div>

                        {sys.severityProbabilities && (
                          <div className="space-y-0.5 mb-2.5">
                            <div className="text-[8px] text-muted-foreground/70 tracking-wider uppercase">Severity</div>
                            <div className="flex h-1.5 rounded-full overflow-hidden">
                              {Object.entries(sys.severityProbabilities)
                                .sort(([,a], [,b]) => b - a)
                                .map(([sev, prob]) => (
                                  <div key={sev} style={{ width: `${prob * 100}%`, backgroundColor: SEVERITY_COLORS[sev] ?? '#6b7280' }} title={`${sev}: ${(prob * 100).toFixed(0)}%`} />
                                ))}
                            </div>
                          </div>
                        )}

                        <div className="flex items-center justify-between text-[9px] text-muted-foreground/70 mb-1.5">
                          <span>Models: {sys.voteCount}/{sys.totalModels}</span>
                        </div>

                        {maint && (
                          <div className="flex items-center gap-1 text-[9px] px-2 py-0.5 rounded-md" style={{ backgroundColor: maint.color + '12', color: maint.color }}>
                            <maint.icon className="w-2.5 h-2.5" />
                            <span>{maint.label}</span>
                          </div>
                        )}

                        {sys.metrics.length > 0 && (
                          <div className="mt-1.5 space-y-0.5">
                            <div className="text-[8px] text-muted-foreground/70 tracking-wider uppercase">Top Features</div>
                            {sys.metrics.slice(0, 3).map(met => (
                              <div key={met.label} className="flex items-center justify-between text-[9px]">
                                <span className="text-muted-foreground truncate mr-1">{met.label}</span>
                                <span className="font-mono text-foreground">{met.value}</span>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Health Trend */}
              {healthTrendData.length > 1 && (
                <div className="bg-card border border-border rounded-lg p-4">
                  <SectionHeader icon={TrendingUp} label="Health Trend" sub="Last 10 races" />
                  <div className="h-[140px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={healthTrendData}>
                        <XAxis dataKey="race" tick={{ fontSize: 9, fill: '#555' }} interval="preserveStartEnd" />
                        <YAxis domain={[0, 100]} tick={{ fontSize: 9, fill: '#555' }} width={28} />
                        <Tooltip content={<CustomTooltip />} />
                        {(() => {
                          const systemNames = healthTrendData.length > 0
                            ? Object.keys(healthTrendData[0]).filter(k => k !== 'race')
                            : [];
                          const colors = ['#FF8000', '#00d4ff', '#22c55e', '#a78bfa', '#f472b6'];
                          return systemNames.map((name, i) => (
                            <Area key={name} type="monotone" dataKey={name} stroke={colors[i % colors.length]} fill={colors[i % colors.length]} fillOpacity={0.06} strokeWidth={1.5} dot={false} />
                          ));
                        })()}
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="flex items-center gap-4 mt-2 justify-center">
                    {healthTrendData.length > 0 && Object.keys(healthTrendData[0]).filter(k => k !== 'race').map((name, i) => {
                      const colors = ['#FF8000', '#00d4ff', '#22c55e', '#a78bfa', '#f472b6'];
                      return (
                        <div key={name} className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
                          <span className="w-2 h-0.5 rounded" style={{ backgroundColor: colors[i % colors.length] }} /> {name}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </>
          )}

          {/* KeX Briefing */}
          {driverCode && (
            <KexBriefingCard
              title="WISE Driver Briefing"
              kex={kex}
              loading={kexLoading}
              loadingText="Extracting driver intelligence\u2026"
            />
          )}
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════
   COMPARE DRIVERS
   ═══════════════════════════════════════════════ */

function CompareDrivers() {
  const [allMarkers, setAllMarkers] = useState<DriverPerformanceMarker[]>([]);
  const [allOvertakes, setAllOvertakes] = useState<DriverOvertakeProfile[]>([]);
  const [allTelemetry, setAllTelemetry] = useState<DriverTelemetryProfile[]>([]);
  const [drivers, setDrivers] = useState<any[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(true);
  const [similarDrivers, setSimilarDrivers] = useState<SimilarDriver[]>([]);
  const [comparisonKex, setComparisonKex] = useState<ComparisonKex | null>(null);
  const [comparisonLoading, setComparisonLoading] = useState(false);
  const [opponentProfiles, setOpponentProfiles] = useState<Record<string, any>>({});

  useEffect(() => {
    Promise.all([
      api.getPerformanceMarkers(),
      api.getOvertakeProfiles(),
      api.getTelemetryProfiles(),
      api.getOpponentDrivers(),
    ]).then(([m, o, t, d]) => {
      setAllMarkers(m);
      setAllOvertakes(o);
      setAllTelemetry(t);
      setDrivers(d.drivers || []);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  const toggleDriver = useCallback((id: string) => {
    setSelected(prev => {
      if (prev.includes(id)) return prev.filter(x => x !== id);
      if (prev.length >= 4) return prev;
      return [...prev, id];
    });
  }, []);

  const getDriverCode = useCallback((driverId: string) => {
    const driver = drivers.find((d: any) => d.driver_id === driverId);
    return driver?.driver_code || driver?.code || driverId.slice(0, 3).toUpperCase();
  }, [drivers]);

  useEffect(() => {
    if (selected.length !== 1) { setSimilarDrivers([]); return; }
    const code = getDriverCode(selected[0]);
    api.getSimilarDrivers(code, 5).then(setSimilarDrivers).catch(() => setSimilarDrivers([]));
  }, [selected, getDriverCode]);

  // Fetch full opponent profiles for selected drivers
  useEffect(() => {
    if (selected.length < 2) return;
    selected.forEach(driverId => {
      if (opponentProfiles[driverId]) return;
      api.getOpponentDriver(driverId).then(profile => {
        setOpponentProfiles(prev => ({ ...prev, [driverId]: profile }));
      }).catch(() => {});
    });
  }, [selected, opponentProfiles]);

  // Trigger KeX comparison when 2+ drivers selected
  useEffect(() => {
    if (selected.length < 2) { setComparisonKex(null); return; }
    const codes = selected.map(getDriverCode);
    setComparisonLoading(true);
    api.getComparisonKex(codes)
      .then(setComparisonKex)
      .catch(() => setComparisonKex(null))
      .finally(() => setComparisonLoading(false));
  }, [selected, getDriverCode]);

  const getDriverData = useCallback((code: string) => {
    const m = allMarkers.find(x => x.Driver === code);
    const o = allOvertakes.find(x => x.driver_code === code);
    const t = allTelemetry.find(x => x.driver_code === code);
    return { m, o, t };
  }, [allMarkers, allOvertakes, allTelemetry]);

  const compareRadarData = useMemo(() => {
    if (selected.length < 2) return [];
    const hasProfiles = selected.every(id => opponentProfiles[id]);
    const metrics: { label: string; key: string; min: number; max: number; invert: boolean }[] = [
      { label: 'Qualifying', key: 'q3_appearance_rate', min: 0, max: 1, invert: false },
      { label: 'Race Pace', key: 'avg_finish_position', min: 1, max: 20, invert: true },
      { label: 'Overtaking', key: 'avg_positions_gained', min: -3, max: 5, invert: false },
      { label: 'Tyre Life', key: 'avg_tyre_life', min: 10, max: 35, invert: false },
      { label: 'Strategy', key: 'undercut_aggression_score', min: 0, max: 1, invert: false },
      { label: 'Consistency', key: 'g_consistency', min: 0, max: 1, invert: false },
      { label: 'Braking', key: 'avg_braking_g', min: 2, max: 5, invert: false },
      { label: 'Late Race', key: 'late_race_position_loss', min: 0, max: 5, invert: true },
      { label: 'Top Speed', key: 'avg_top_speed', min: 280, max: 340, invert: false },
      { label: 'Starts', key: 'avg_positions_gained_lap1_to_5', min: -2, max: 4, invert: false },
    ];
    return metrics.map(({ label, key, min, max, invert }) => {
      const row: any = { metric: label };
      selected.forEach(driverId => {
        const code = getDriverCode(driverId);
        if (hasProfiles) {
          const p = opponentProfiles[driverId];
          row[code] = normalize(p?.[key] ?? null, min, max, invert);
        } else {
          const { m, o, t } = getDriverData(code);
          let val = 0;
          switch (label) {
            case 'Qualifying': val = normalize(m?.lap_time_consistency_std ?? null, 0, 30, true); break;
            case 'Overtaking': val = normalize(o?.overtake_ratio ?? null, 0.5, 1.5, false); break;
            case 'Top Speed': val = normalize(t?.avg_race_speed_kmh ?? null, 170, 220, false); break;
            case 'Braking': val = normalize(t?.avg_braking_g ?? null, 2, 5, false); break;
            default: val = 0;
          }
          row[code] = val;
        }
      });
      return row;
    });
  }, [selected, getDriverCode, getDriverData, opponentProfiles]);

  const comparisonStats = useMemo(() => {
    if (selected.length < 2) return [];
    const codes = selected.map(getDriverCode);
    const hasProfiles = selected.every(id => opponentProfiles[id]);
    const fmt = (v: number | null | undefined, p = 2) => v != null ? v.toFixed(p) : '\u2014';
    const pct = (v: number | null | undefined) => v != null ? `${(v * 100).toFixed(1)}%` : '\u2014';
    const op = (code: string) => {
      const id = selected.find(id => getDriverCode(id) === code);
      return id ? opponentProfiles[id] : null;
    };
    type Row = { label: string; unit: string; values: Record<string, string>; bestDir?: 'low' | 'high' };
    const rows: Row[] = [];

    if (hasProfiles) {
      // Rich metrics from opponent_profiles
      rows.push(
        { label: 'Avg Finish', unit: 'pos', bestDir: 'low', values: Object.fromEntries(codes.map(c => [c, fmt(op(c)?.avg_finish_position, 1)])) },
        { label: 'Avg Grid', unit: 'pos', bestDir: 'low', values: Object.fromEntries(codes.map(c => [c, fmt(op(c)?.avg_grid_position, 1)])) },
        { label: 'Q3 Rate', unit: '', bestDir: 'high', values: Object.fromEntries(codes.map(c => [c, pct(op(c)?.q3_appearance_rate)])) },
        { label: 'Positions Gained', unit: '/race', bestDir: 'high', values: Object.fromEntries(codes.map(c => [c, fmt(op(c)?.avg_positions_gained, 2)])) },
        { label: 'Top Speed', unit: 'km/h', bestDir: 'high', values: Object.fromEntries(codes.map(c => [c, fmt(op(c)?.avg_top_speed, 1)])) },
        { label: 'Braking G', unit: 'G', bestDir: 'high', values: Object.fromEntries(codes.map(c => [c, fmt(op(c)?.avg_braking_g, 2)])) },
        { label: 'Tyre Life', unit: 'laps', bestDir: 'high', values: Object.fromEntries(codes.map(c => [c, fmt(op(c)?.avg_tyre_life, 1)])) },
        { label: 'Long Stint', unit: 'laps', bestDir: 'high', values: Object.fromEntries(codes.map(c => [c, fmt(op(c)?.long_stint_capability, 1)])) },
        { label: 'Pit Duration', unit: 's', bestDir: 'low', values: Object.fromEntries(codes.map(c => [c, fmt(op(c)?.avg_pit_duration_s, 1)])) },
        { label: 'Undercut Aggression', unit: '', bestDir: 'high', values: Object.fromEntries(codes.map(c => [c, fmt(op(c)?.undercut_aggression_score, 2)])) },
        { label: 'Tyre Extension', unit: '', bestDir: 'high', values: Object.fromEntries(codes.map(c => [c, fmt(op(c)?.tyre_extension_bias, 2)])) },
        { label: 'Late Race Loss', unit: 'pos', bestDir: 'low', values: Object.fromEntries(codes.map(c => [c, fmt(op(c)?.late_race_position_loss, 2)])) },
        { label: 'Position Volatility', unit: '', bestDir: 'low', values: Object.fromEntries(codes.map(c => [c, fmt(op(c)?.position_volatility, 2)])) },
        { label: 'Throttle Smoothness', unit: '', bestDir: 'high', values: Object.fromEntries(codes.map(c => [c, fmt(op(c)?.throttle_smoothness, 3)])) },
        { label: 'G Consistency', unit: '', bestDir: 'high', values: Object.fromEntries(codes.map(c => [c, fmt(op(c)?.g_consistency, 3)])) },
        { label: 'DNF Rate', unit: '', bestDir: 'low', values: Object.fromEntries(codes.map(c => [c, pct(op(c)?.dnf_rate)])) },
        { label: 'Points/Race', unit: '', bestDir: 'high', values: Object.fromEntries(codes.map(c => [c, fmt(op(c)?.avg_points_per_race, 1)])) },
      );
    } else {
      // Fallback to aggregate collections
      const dataMap = Object.fromEntries(codes.map(c => [c, getDriverData(c)]));
      rows.push(
        { label: 'Avg Speed', unit: 'km/h', bestDir: 'high', values: Object.fromEntries(codes.map(c => [c, fmt(dataMap[c].t?.avg_race_speed_kmh, 1)])) },
        { label: 'Braking G', unit: 'G', bestDir: 'high', values: Object.fromEntries(codes.map(c => [c, fmt(dataMap[c].t?.avg_braking_g)])) },
        { label: 'Full Throttle', unit: '%', bestDir: 'high', values: Object.fromEntries(codes.map(c => [c, fmt(dataMap[c].t ? dataMap[c].t!.full_throttle_ratio * 100 : null, 1)])) },
        { label: 'DRS Gain', unit: 'km/h', bestDir: 'high', values: Object.fromEntries(codes.map(c => [c, fmt(dataMap[c].t?.drs_speed_gain_kmh, 1)])) },
        { label: 'Tyre Degradation', unit: 's/lap', bestDir: 'low', values: Object.fromEntries(codes.map(c => [c, fmt(dataMap[c].m?.degradation_slope_s_per_lap, 3)])) },
        { label: 'Lap Consistency', unit: 'std', bestDir: 'low', values: Object.fromEntries(codes.map(c => [c, fmt(dataMap[c].m?.lap_time_consistency_std)])) },
        { label: 'Late Race Delta', unit: 's', bestDir: 'low', values: Object.fromEntries(codes.map(c => [c, fmt(dataMap[c].m?.late_race_delta_s)])) },
        { label: 'OT Made/Race', unit: '', bestDir: 'high', values: Object.fromEntries(codes.map(c => [c, fmt(dataMap[c].o?.overtakes_per_race, 1)])) },
        { label: 'OT Ratio', unit: '', bestDir: 'high', values: Object.fromEntries(codes.map(c => [c, fmt(dataMap[c].o?.overtake_ratio)])) },
      );
    }
    return rows;
  }, [selected, getDriverCode, getDriverData, opponentProfiles]);

  const filteredDrivers = drivers.filter((d: any) => {
    if (!search) return true;
    const s = search.toLowerCase();
    const code = (d.driver_code || d.code || '').toLowerCase();
    const name = (d.driver_id || '').toLowerCase();
    const team = (d.team || d.constructor || '').toLowerCase();
    return code.includes(s) || name.includes(s) || team.includes(s);
  });

  if (loading) {
    return <div className="flex items-center justify-center py-20"><Loader2 className="w-6 h-6 text-[#FF8000] animate-spin" /></div>;
  }

  const selectedCodes = selected.map(getDriverCode);

  return (
    <div className="space-y-4">
      {/* Selected drivers bar */}
      <div className="flex items-center gap-2 flex-wrap min-h-[40px] bg-background/40 rounded-lg px-4 py-2 border border-border">
        {selected.length === 0 ? (
          <span className="text-[12px] text-muted-foreground">Pick 2–4 drivers to compare</span>
        ) : (
          selected.map((driverId, i) => {
            const code = getDriverCode(driverId);
            return (
              <button
                key={driverId}
                onClick={() => toggleDriver(driverId)}
                className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-[12px] border transition-all hover:opacity-80"
                style={{ borderColor: COMPARE_COLORS[i] + '40', color: COMPARE_COLORS[i], background: COMPARE_COLORS[i] + '08' }}
              >
                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: COMPARE_COLORS[i] }} />
                <span className="font-bold">{code}</span>
                <span className="text-muted-foreground/70 text-[10px] hidden sm:inline">{formatDriverName(driverId)}</span>
                <X className="w-3 h-3 opacity-40 hover:opacity-100" />
              </button>
            );
          })
        )}
        <span className="text-[10px] text-muted-foreground/60 ml-auto font-mono">{selected.length}/4</span>
      </div>

      <div className="flex gap-4">
        {/* Driver picker */}
        <div className="w-[190px] shrink-0 space-y-2">
          <div className="relative">
            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground pointer-events-none" />
            <input
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder="Search\u2026"
              className="w-full bg-background border border-border rounded-lg pl-8 pr-3 py-1.5 text-[12px] text-foreground placeholder:text-muted-foreground/50 focus:outline-none focus:border-[#FF8000]/25 transition-colors"
            />
          </div>
          <div className="space-y-0.5 max-h-[60vh] overflow-y-auto pr-1">
            {filteredDrivers.map((d: any) => {
              const isSelected = selected.includes(d.driver_id);
              const idx = selected.indexOf(d.driver_id);
              const code = d.driver_code || d.code || d.driver_id?.slice(0, 3).toUpperCase();
              const team = d.team || '';
              const tColor = teamColors[team] || '#555';
              return (
                <button
                  key={d.driver_id}
                  onClick={() => toggleDriver(d.driver_id)}
                  disabled={!isSelected && selected.length >= 4}
                  className={`w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-left text-[11px] transition-all ${
                    isSelected
                      ? 'bg-[rgba(255,128,0,0.06)] border border-border'
                      : 'hover:bg-card disabled:opacity-25 border border-transparent'
                  }`}
                >
                  {isSelected ? (
                    <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: COMPARE_COLORS[idx] }} />
                  ) : (
                    <span className="w-1.5 h-4 rounded-sm shrink-0" style={{ backgroundColor: tColor + '60' }} />
                  )}
                  <span className={`font-semibold ${isSelected ? 'text-foreground' : 'text-muted-foreground'}`}>{code}</span>
                  <span className="text-[9px] text-muted-foreground/60 truncate">{team}</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Main comparison area */}
        <div className="flex-1 min-w-0 space-y-4">
          {selected.length < 2 ? (
            <div className="flex flex-col items-center justify-center py-16 text-muted-foreground">
              {selected.length === 1 && similarDrivers.length > 0 ? (
                <>
                  <GitCompare className="w-7 h-7 mb-3 opacity-30" />
                  <p className="text-[13px] mb-1">Similar to <span className="text-[#FF8000] font-semibold">{getDriverCode(selected[0])}</span></p>
                  <p className="text-[10px] opacity-50 mb-4">Vector profile similarity — click to add</p>
                  <div className="space-y-1.5 w-full max-w-xs">
                    {similarDrivers.map((s, i) => {
                      const pct = Math.round(s.score * 100);
                      const color = teamColors[s.team] || '#6b7280';
                      const driverEntry = drivers.find((d: any) => (d.driver_code || d.code) === s.driver_code);
                      return (
                        <button
                          type="button"
                          key={s.driver_code}
                          onClick={() => { if (driverEntry) toggleDriver(driverEntry.driver_id); }}
                          className="w-full flex items-center gap-3 px-3 py-2 rounded-lg bg-background hover:bg-[#FF8000]/[0.03] border border-transparent hover:border-[#FF8000]/15 transition-all group"
                        >
                          <span className="text-[10px] font-mono text-muted-foreground/50 w-3">{i + 1}</span>
                          {TEAM_NAME_TO_LOGO[s.team] && TEAM_LOGOS[TEAM_NAME_TO_LOGO[s.team]] ? (
                            <img src={TEAM_LOGOS[TEAM_NAME_TO_LOGO[s.team]]} alt={s.team} className="w-4 h-4 object-contain shrink-0" />
                          ) : (
                            <span className="w-1.5 h-4 rounded-sm shrink-0" style={{ backgroundColor: color }} />
                          )}
                          <div className="flex-1 text-left">
                            <span className="text-[11px] font-semibold text-foreground group-hover:text-[#FF8000] transition-colors">{s.driver_code}</span>
                            <span className="text-[9px] text-muted-foreground/60 ml-1.5">{driverEntry ? formatDriverName(driverEntry.driver_id) : ''}</span>
                            <span className="text-[8px] ml-1" style={{ color }}>{s.team}</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <div className="w-14 h-1 bg-secondary rounded-full overflow-hidden">
                              <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: color }} />
                            </div>
                            <span className="text-[10px] font-mono w-8 text-right" style={{ color }}>{pct}%</span>
                          </div>
                        </button>
                      );
                    })}
                  </div>
                </>
              ) : (
                <>
                  <Users className="w-7 h-7 mb-3 opacity-30" />
                  <p className="text-[13px]">Select at least 2 drivers</p>
                  <p className="text-[10px] mt-1 opacity-50">Performance radar + head-to-head stats</p>
                </>
              )}
            </div>
          ) : (
            <>
              {/* Radar */}
              <div className="bg-card border border-border rounded-lg p-4">
                <SectionHeader icon={Target} label="Driver Profile Overlay" sub="Individual strengths side-by-side — each shape is one driver" />
                {/* Legend */}
                <div className="flex items-center justify-center gap-4 mb-2">
                  {selected.map((driverId, i) => {
                    const code = getDriverCode(driverId);
                    const driver = drivers.find((d: any) => d.driver_id === driverId);
                    const teamKey = TEAM_NAME_TO_LOGO[driver?.team];
                    return (
                      <span key={driverId} className="flex items-center gap-1.5 text-[11px] font-semibold" style={{ color: COMPARE_COLORS[i] }}>
                        <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: COMPARE_COLORS[i] }} />
                        {teamKey && TEAM_LOGOS[teamKey] && (
                          <img src={TEAM_LOGOS[teamKey]} alt="" className="h-3.5 w-3.5 object-contain" />
                        )}
                        {code}
                      </span>
                    );
                  })}
                </div>
                <ResponsiveContainer width="100%" height={380}>
                  <RadarChart data={compareRadarData} cx="50%" cy="50%" outerRadius="68%">
                    <PolarGrid stroke="rgba(255,255,255,0.05)" gridType="polygon" />
                    <PolarAngleAxis dataKey="metric" tick={{ fill: '#9ca3af', fontSize: 9, fontWeight: 500 }} />
                    <PolarRadiusAxis angle={90} domain={[0, 100]} tick={false} axisLine={false} />
                    <Tooltip
                      contentStyle={{
                        background: '#1A1F2E',
                        border: '1px solid rgba(255,255,255,0.1)',
                        fontSize: 11,
                        borderRadius: 8,
                      }}
                      formatter={(value: number, name: string) => [`${Math.round(value)}`, name]}
                    />
                    {selected.map((driverId, i) => {
                      const code = getDriverCode(driverId);
                      return (
                        <Radar
                          key={driverId}
                          name={code}
                          dataKey={code}
                          stroke={COMPARE_COLORS[i]}
                          strokeWidth={2}
                          fill={COMPARE_COLORS[i]}
                          fillOpacity={0.08}
                          dot={{ r: 3, fill: COMPARE_COLORS[i], strokeWidth: 0 }}
                        />
                      );
                    })}
                  </RadarChart>
                </ResponsiveContainer>
              </div>

              {/* KeX Comparison Briefing (Gen UI) */}
              <KexBriefingCard
                title={`WISE Matchup Analysis — ${selectedCodes.join(' vs ')}`}
                icon="sparkles"
                kex={comparisonKex}
                loading={comparisonLoading}
                loadingText="WISE analyzing what differentiates these drivers…"
              />

              {/* Stats Table */}
              <div className="bg-card border border-border rounded-lg p-4">
                <SectionHeader icon={Gauge} label="Head-to-Head" />
                <div className="overflow-x-auto">
                  <table className="w-full text-[11px]">
                    <thead>
                      <tr className="border-b border-border">
                        <th className="text-left py-2.5 pr-4 text-muted-foreground/70 font-medium text-[10px] uppercase tracking-wider">Metric</th>
                        {selectedCodes.map((code, i) => (
                          <th key={code} className="text-center px-3 py-2.5 font-bold text-[12px]" style={{ color: COMPARE_COLORS[i] }}>{code}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {comparisonStats.map(row => {
                        const numVals = selectedCodes.map(c => parseFloat(row.values[c])).filter(v => !isNaN(v));
                        const best = row.bestDir === 'high' ? Math.max(...numVals) : Math.min(...numVals);
                        return (
                          <tr key={row.label} className="border-b border-[rgba(255,128,0,0.03)] hover:bg-[rgba(255,128,0,0.02)] transition-colors">
                            <td className="py-2 pr-4 text-muted-foreground">
                              {row.label}
                              {row.unit && <span className="text-[9px] ml-1 opacity-40">{row.unit}</span>}
                            </td>
                            {selectedCodes.map((code) => {
                              const val = row.values[code];
                              const num = parseFloat(val);
                              const isBest = !isNaN(num) && num === best && numVals.length > 1;
                              return (
                                <td key={code} className="text-center px-3 py-2 font-mono">
                                  <span className={isBest ? 'text-[#FF8000] font-semibold' : 'text-foreground'}>{val}</span>
                                </td>
                              );
                            })}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>

            </>
          )}
        </div>
      </div>
    </div>
  );
}
