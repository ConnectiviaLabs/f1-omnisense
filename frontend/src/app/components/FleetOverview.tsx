import { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import {
  ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip,
} from 'recharts';
import {
  AlertTriangle, CheckCircle2, XCircle, Activity,
  CircleDot, Shield, TrendingUp, TrendingDown, Minus, Fuel,
  X, Loader2, Sparkles, Gauge, ChevronRight,
} from 'lucide-react';
import KexBriefingCard from './KexBriefingCard';
import { ForecastChart } from './ForecastChart';
import { HealthGauge } from './HealthGauge';
import { AgentActivity } from './AgentActivity';
import { getCarTelemetryKex, getAnomalyKex, type CarTelemetryKex } from '../api/driverIntel';
import {
  type HealthLevel, type VehicleData,
  mapLevel, levelColor, levelBg,
  MAINTENANCE_LABELS, SEVERITY_COLORS,
} from './anomalyHelpers';

/* ── Team colors & logos (match DriverIntel) ── */
const teamColors: Record<string, string> = {
  'Red Bull': '#3671C6', 'Red Bull Racing': '#3671C6', 'McLaren': '#FF8000', 'Ferrari': '#E8002D',
  'Mercedes': '#27F4D2', 'Aston Martin': '#229971', 'Alpine': '#FF87BC', 'Alpine F1 Team': '#FF87BC',
  'Williams': '#64C4FF', 'RB': '#6692FF', 'RB F1 Team': '#6692FF', 'Racing Bulls': '#6692FF',
  'Kick Sauber': '#52E252', 'Sauber': '#52E252', 'Alfa Romeo': '#C92D4B',
  'Haas F1 Team': '#B6BABD', 'Haas': '#B6BABD',
  'AlphaTauri': '#6692FF', 'Toro Rosso': '#469BFF',
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
  'Red Bull': 'red_bull', 'Red Bull Racing': 'red_bull', 'McLaren': 'mclaren', 'Ferrari': 'ferrari',
  'Mercedes': 'mercedes', 'Aston Martin': 'aston_martin', 'Alpine': 'alpine', 'Alpine F1 Team': 'alpine',
  'Williams': 'williams', 'RB': 'rb', 'RB F1 Team': 'rb', 'Racing Bulls': 'rb',
  'Kick Sauber': 'sauber', 'Sauber': 'sauber', 'Alfa Romeo': 'sauber',
  'Haas F1 Team': 'haas', 'Haas': 'haas', 'AlphaTauri': 'rb', 'Toro Rosso': 'rb',
};

function getHealthTrend(vehicle: VehicleData): 'up' | 'down' | 'stable' {
  if (!vehicle.races || vehicle.races.length < 2) return 'stable';
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

/* ── Tyre compound colors ── */
const COMPOUND_COLORS: Record<string, string> = {
  SOFT: '#FF3333', MEDIUM: '#FFD700', HARD: '#CCCCCC',
  INTERMEDIATE: '#00CC44', WET: '#0088FF',
};

/* ── Race telemetry summary type ── */
interface RaceTelemetry {
  race: string;
  avgSpeed: number;
  topSpeed: number;
  avgRPM: number;
  maxRPM: number;
  avgThrottle: number;
  brakePct: number;
  drsPct: number;
  compounds: string[];
}

/* ── OmniHealth component type ── */
interface OmniComponent {
  component: string;
  health_pct: number;
  severity: string;
  action: string;
  confidence?: number;
}

/* ── Stint type ── */
interface StintData {
  driver_acronym: string;
  meeting_name: string;
  compound: string;
  stint_number: number;
  lap_start: number;
  lap_end: number;
  stint_laps: number;
  tyre_age_at_start: number;
}

function MaintenanceBadge({ action }: { action?: string }) {
  const info = MAINTENANCE_LABELS[action ?? 'none'] ?? MAINTENANCE_LABELS.none;
  const Icon = info.icon;
  return (
    <div className="flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium"
      style={{ background: `${info.color}15`, color: info.color, border: `1px solid ${info.color}25` }}>
      <Icon className="w-2.5 h-2.5" />
      {info.label}
    </div>
  );
}

function SeverityBar({ probabilities }: { probabilities?: Record<string, number> }) {
  if (!probabilities) return null;
  const order = ['normal', 'low', 'medium', 'high', 'critical'] as const;
  const total = Object.values(probabilities).reduce((a, b) => a + b, 0);
  if (total < 0.01) return null;
  return (
    <div className="flex h-1.5 rounded-full overflow-hidden bg-background" title="Risk distribution">
      {order.map(sev => {
        const pct = (probabilities[sev] ?? 0) * 100;
        if (pct < 1) return null;
        return <div key={sev} style={{ width: `${pct}%`, background: SEVERITY_COLORS[sev] }} />;
      })}
    </div>
  );
}


// 3D car models — team name → GLB served from public/models/
const TEAM_CAR_MODEL: Record<string, { label: string; url: string }> = {
  'McLaren':           { label: 'MCL39',    url: '/models/mcl39.glb' },
  'Red Bull Racing':   { label: 'RB21',     url: '/models/red_bull.glb' },
  'Red Bull':          { label: 'RB21',     url: '/models/red_bull.glb' },
  'Ferrari':           { label: 'SF-25',    url: '/models/ferrari.glb' },
  'Mercedes':          { label: 'W16',      url: '/models/mercedes.glb' },
  'Aston Martin':      { label: 'AMR25',    url: '/models/aston_martin.glb' },
  'Alpine':            { label: 'A525',     url: '/models/alpine.glb' },
  'Alpine F1 Team':    { label: 'A525',     url: '/models/alpine.glb' },
  'Williams':          { label: 'FW47',     url: '/models/williams.glb' },
  'Racing Bulls':      { label: 'VCARB 02', url: '/models/rb.glb' },
  'RB':                { label: 'VCARB 02', url: '/models/rb.glb' },
  'RB F1 Team':        { label: 'VCARB 02', url: '/models/rb.glb' },
  'AlphaTauri':        { label: 'VCARB 02', url: '/models/rb.glb' },
  'Toro Rosso':        { label: 'VCARB 02', url: '/models/rb.glb' },
  'Kick Sauber':       { label: 'C45',      url: '/models/sauber.glb' },
  'Sauber':            { label: 'C45',      url: '/models/sauber.glb' },
  'Alfa Romeo':        { label: 'C45',      url: '/models/sauber.glb' },
  'Haas F1 Team':      { label: 'VF-25',    url: '/models/haas.glb' },
  'Haas':              { label: 'VF-25',    url: '/models/haas.glb' },
};
const DEFAULT_CAR_MODEL = { label: 'MCL39', url: '/models/mcl39.glb' };

// ─── Component ──────────────────────────────────────────────────────
interface FleetOverviewProps {
  prefetchedVehicles: VehicleData[];
  prefetchLoading: boolean;
  /** Which pillar tab is active (driven by sidebar). */
  defaultSection?: 'telemetry' | 'anomaly' | 'forecast';
}

type FleetTab = 'telemetry' | 'anomaly' | 'forecast';

export function FleetOverview({ prefetchedVehicles, prefetchLoading, defaultSection }: FleetOverviewProps) {
  const [vehicles, setVehicles] = useState<VehicleData[]>([]);
  const [selectedCar, setSelectedCar] = useState<VehicleData | null>(null);
  const [loading, setLoading] = useState(true);
  const [liveSource, setLiveSource] = useState<'cached' | 'live' | null>(null);

  // Active pillar — driven by sidebar
  const activeTab: FleetTab = (defaultSection as FleetTab) ?? 'telemetry';

  // Session picker for agent runs
  const [sessions, setSessions] = useState<{ session_key: number; meeting_name: string; session_type: string; year: number }[]>([]);
  const [selectedSessionKey, setSelectedSessionKey] = useState<number | undefined>();

  useEffect(() => {
    fetch('/api/local/openf1/sessions?session_type=Race')
      .then(r => r.json())
      .then((data: any[]) => {
        const mapped = data.slice(0, 30).map(s => ({
          session_key: s.session_key,
          meeting_name: s.meeting_name || s.circuit_short_name || `Session ${s.session_key}`,
          session_type: s.session_type || 'Race',
          year: s.year || new Date(s.date_start).getFullYear(),
        }));
        setSessions(mapped);
        if (mapped.length) setSelectedSessionKey(mapped[0].session_key);
      })
      .catch(() => {});
  }, []);

  // Per-tab KeX state (lazy-loaded)
  const [anomalyKex, setAnomalyKex] = useState<any>(null);
  const [anomalyKexLoading, setAnomalyKexLoading] = useState(false);
  const [telemetryKex, setTelemetryKex] = useState<CarTelemetryKex | null>(null);
  const [telemetryKexLoading, setTelemetryKexLoading] = useState(false);

  // Driver profile stats (fetched on driver select)
  interface DriverStats {
    position?: number; points?: number; wins?: number; nationality?: string;
    races?: number; podiums?: number; dnfs?: number; avgGrid?: number; avgFinish?: number;
    throttleSmoothness?: number; brakeOverlap?: number; lapConsistency?: number;
    avgTopSpeed?: number; degradationSlope?: number; lateRaceDelta?: number;
    overtakesMade?: number; timesOvertaken?: number; overtakeNet?: number;
  }
  const [driverStats, setDriverStats] = useState<DriverStats>({});
  const [statsLoading, setStatsLoading] = useState(false);

  // Fetch driver stats when a driver is selected
  useEffect(() => {
    if (!selectedCar) { setDriverStats({}); return; }
    const code = selectedCar.code;
    setStatsLoading(true);

    Promise.allSettled([
      fetch('/api/jolpica/driver_standings').then(r => r.json()),
      fetch('/api/jolpica/race_results').then(r => r.json()),
      fetch(`/api/driver_intel/performance_markers?driver=${code}`).then(r => r.json()),
      fetch(`/api/driver_intel/overtake_profiles?driver=${code}`).then(r => r.json()),
    ]).then(([standingsRes, resultsRes, markersRes, overtakeRes]) => {
      const stats: DriverStats = {};

      if (standingsRes.status === 'fulfilled') {
        const all = standingsRes.value as any[];
        const driverStandings = all
          .filter((s: any) => s.driver_code === code)
          .sort((a: any, b: any) => b.season - a.season);
        if (driverStandings.length) {
          const latest = driverStandings[0];
          stats.position = latest.position;
          stats.points = latest.points;
          stats.wins = latest.wins;
          stats.nationality = latest.nationality;
        }
      }

      if (resultsRes.status === 'fulfilled') {
        const all = resultsRes.value as any[];
        const driverResults = all.filter((r: any) => r.driver_code === code);
        const latestSeason = Math.max(...driverResults.map((r: any) => r.season), 0);
        const seasonResults = driverResults.filter((r: any) => r.season === latestSeason);
        stats.races = seasonResults.length;
        stats.podiums = seasonResults.filter((r: any) => {
          const pos = parseInt(r.position_text);
          return !isNaN(pos) && pos <= 3;
        }).length;
        stats.dnfs = seasonResults.filter((r: any) =>
          r.status && r.status !== 'Finished' && !r.status.startsWith('+')
        ).length;
        const grids = seasonResults.map((r: any) => r.grid).filter((g: any) => g > 0);
        const finishes = seasonResults.map((r: any) => parseInt(r.position_text)).filter((p: number) => !isNaN(p) && p > 0);
        if (grids.length) stats.avgGrid = +(grids.reduce((a: number, b: number) => a + b, 0) / grids.length).toFixed(1);
        if (finishes.length) stats.avgFinish = +(finishes.reduce((a: number, b: number) => a + b, 0) / finishes.length).toFixed(1);
      }

      if (markersRes.status === 'fulfilled') {
        const arr = markersRes.value as any[];
        if (arr.length) {
          const m = arr[0];
          stats.throttleSmoothness = m.throttle_smoothness;
          stats.brakeOverlap = m.brake_overlap_rate;
          stats.lapConsistency = m.lap_time_consistency_std;
          stats.avgTopSpeed = m.avg_top_speed_kmh;
          stats.degradationSlope = m.degradation_slope_s_per_lap;
          stats.lateRaceDelta = m.late_race_delta_s;
        }
      }

      if (overtakeRes.status === 'fulfilled') {
        const arr = overtakeRes.value as any[];
        if (arr.length) {
          const o = arr[0];
          stats.overtakesMade = o.total_overtakes_made;
          stats.timesOvertaken = o.total_times_overtaken;
          stats.overtakeNet = o.overtake_net;
        }
      }

      setDriverStats(stats);
    }).finally(() => setStatsLoading(false));
  }, [selectedCar]);

  // ── Race telemetry trends (mccar-summary) ──
  const [raceTelemetry, setRaceTelemetry] = useState<RaceTelemetry[]>([]);
  useEffect(() => {
    if (!selectedCar) { setRaceTelemetry([]); return; }
    fetch(`/api/mccar-summary/2024/${selectedCar.code}`) // TODO: parameterize season when 2025 data available
      .then(r => r.ok ? r.json() : [])
      .then(data => setRaceTelemetry(Array.isArray(data) ? data : []))
      .catch(() => setRaceTelemetry([]));
  }, [selectedCar?.code]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── OmniHealth detailed assessment ──
  const [omniComponents, setOmniComponents] = useState<OmniComponent[]>([]);
  const [omniLoading, setOmniLoading] = useState(false);
  useEffect(() => {
    if (!selectedCar || activeTab !== 'anomaly') { setOmniComponents([]); return; }
    setOmniLoading(true);
    fetch(`/api/omni/health/assess/${selectedCar.code}`)
      .then(r => r.ok ? r.json() : null)
      .then(data => setOmniComponents(data?.components ?? []))
      .catch(() => setOmniComponents([]))
      .finally(() => setOmniLoading(false));
  }, [selectedCar?.code, activeTab]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Tyre stints (last race from race telemetry) ──
  const [stints, setStints] = useState<StintData[]>([]);
  const lastRaceName = raceTelemetry.length > 0 ? raceTelemetry[raceTelemetry.length - 1].race : null;
  useEffect(() => {
    if (!selectedCar || !lastRaceName) { setStints([]); return; }
    const meetingName = lastRaceName + ' Grand Prix';
    fetch(`/api/mccar-race-stints/2024/${encodeURIComponent(meetingName)}`) // TODO: parameterize season
      .then(r => r.ok ? r.json() : [])
      .then(data => {
        const all = Array.isArray(data) ? data : [];
        setStints(all.filter((s: StintData) =>
          s.driver_acronym === selectedCar.code && s.meeting_name === meetingName
        ));
      })
      .catch(() => setStints([]));
  }, [selectedCar?.code, lastRaceName]); // eslint-disable-line react-hooks/exhaustive-deps

  // Use pre-fetched anomaly vehicles from App
  useEffect(() => {
    if (prefetchedVehicles.length > 0) {
      setVehicles(prefetchedVehicles);
      setLoading(prefetchLoading);
    } else if (!prefetchLoading) {
      setLoading(false);
    }
  }, [prefetchedVehicles, prefetchLoading]);

  // Try OmniHealth live data — non-blocking, falls back to cached anomaly snapshot
  useEffect(() => {
    if (vehicles.length === 0) return;
    const fetchLive = async () => {
      try {
        const res = await fetch('/api/omni/health/fleet');
        if (!res.ok) return;
        const data = await res.json();
        if (!data.drivers?.length) return;

        setVehicles(prev => prev.map(v => {
          const live = data.drivers.find((d: any) => d.code === v.code);
          if (!live || live.error || !live.components) return v;

          const liveSystemMap = new Map<string, any>();
          for (const comp of live.components) {
            liveSystemMap.set(comp.component, comp);
          }

          const systems = v.systems.map(sys => {
            const lc = liveSystemMap.get(sys.name);
            if (!lc) return sys;
            return {
              ...sys,
              health: Math.round(lc.health_pct),
              level: mapLevel(lc.severity),
              maintenanceAction: lc.action,
            };
          });

          return {
            ...v,
            overallHealth: Math.round(live.overall_health),
            level: mapLevel(live.overall_risk),
            systems,
          };
        }));
        setLiveSource('live');
      } catch {
        setLiveSource('cached');
      }
    };
    fetchLive();
  }, [vehicles.length > 0]); // eslint-disable-line react-hooks/exhaustive-deps

  // Lazy-load KeX per active tab + selected driver
  useEffect(() => {
    setAnomalyKex(null);
    if (!selectedCar || activeTab !== 'anomaly') return;
    setAnomalyKexLoading(true);
    getAnomalyKex(selectedCar.code)
      .then(setAnomalyKex)
      .catch(() => setAnomalyKex(null))
      .finally(() => setAnomalyKexLoading(false));
  }, [selectedCar?.code, activeTab]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    setTelemetryKex(null);
    if (!selectedCar || activeTab !== 'telemetry') return;
    setTelemetryKexLoading(true);
    getCarTelemetryKex(selectedCar.code, 2024)
      .then(setTelemetryKex)
      .catch(() => setTelemetryKex(null))
      .finally(() => setTelemetryKexLoading(false));
  }, [selectedCar?.code, activeTab]); // eslint-disable-line react-hooks/exhaustive-deps

  // Race-by-race trend from anomaly data
  const trendYears = useMemo(() => {
    if (!selectedCar) return [];
    const years = new Set<number>();
    for (const r of selectedCar.races) {
      const m = r.race.match(/^(\d{4})\s/);
      if (m) years.add(Number(m[1]));
    }
    return [...years].sort((a, b) => b - a);
  }, [selectedCar]);

  const [trendYear, setTrendYear] = useState<number | null>(null);

  useEffect(() => {
    setTrendYear(trendYears[0] ?? null);
  }, [trendYears]);

  const TREND_SYSTEMS = ['Power Unit', 'Brakes', 'Drivetrain', 'Suspension', 'Thermal', 'Electronics', 'Tyre Management'] as const;
  const TREND_ABBR: Record<string, string> = {
    'Power Unit': 'PU', 'Brakes': 'BRK', 'Drivetrain': 'DRV',
    'Suspension': 'SUS', 'Thermal': 'THR', 'Electronics': 'ELC', 'Tyre Management': 'TYR',
  };

  const trendData = useMemo(() => {
    if (!selectedCar) return [];
    return selectedCar.races
      .filter(r => !trendYear || r.race.startsWith(`${trendYear} `))
      .map(r => {
        const systems = r.systems;

        const actions = Object.values(systems)
          .map(s => s.maintenance_action)
          .filter((a): a is string => !!a && a !== 'none');
        const actionPriority = ['alert_and_remediate', 'alert', 'log_and_monitor', 'log'];
        const topAction = actionPriority.find(a => actions.includes(a)) ?? 'none';

        const sysHealth: Record<string, number> = {};
        for (const sys of TREND_SYSTEMS) {
          sysHealth[sys] = systems[sys]?.health ?? 0;
        }

        return {
          race: r.race.replace(/^\d{4}\s+/, ''),
          sysHealth,
          maintenanceAction: topAction,
        };
      });
  }, [selectedCar, trendYear]);

  // Split vehicles: McLaren first, then rest sorted by health
  const mclarenVehicles = useMemo(() => vehicles.filter(v => v.team === 'McLaren'), [vehicles]);
  const otherVehicles = useMemo(() =>
    vehicles.filter(v => v.team !== 'McLaren').sort((a, b) => a.overallHealth - b.overallHealth),
    [vehicles],
  );

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Activity className="w-5 h-5 text-primary animate-spin" />
        <span className="ml-2 text-sm text-muted-foreground">Loading anomaly detection data...</span>
      </div>
    );
  }

  if (!loading && vehicles.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-64 gap-3">
        <Shield className="w-8 h-8 text-muted-foreground/30" />
        <span className="text-sm text-muted-foreground">No anomaly data available</span>
        <span className="text-[11px] text-muted-foreground/60">Run the anomaly detection pipeline to populate this view</span>
      </div>
    );
  }

  const healthColor = (h: number) => h >= 75 ? '#22c55e' : h >= 50 ? '#FF8000' : '#ef4444';

  return (
    <div className="space-y-4">
      {/* Fleet status bar */}
      <div className="flex items-center gap-4 flex-wrap">
        <div className="flex items-center gap-2 bg-card rounded-lg px-3 py-2 border border-border">
          <CircleDot className="w-3 h-3 text-primary" />
          <span className="text-[12px] text-muted-foreground">{vehicles.length} drivers monitored</span>
          {liveSource && (
            <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${
              liveSource === 'live'
                ? 'bg-green-500/10 text-green-400 border border-green-500/20'
                : 'bg-gray-500/10 text-gray-400 border border-gray-500/20'
            }`}>
              {liveSource === 'live' ? 'LIVE' : 'CACHED'}
            </span>
          )}
        </div>
        {(['nominal', 'warning', 'critical'] as HealthLevel[]).map(level => {
          const count = vehicles.filter(v => v.level === level).length;
          if (count === 0) return null;
          return (
            <div key={level} className="flex items-center gap-1.5 text-[12px]">
              <div className="w-2 h-2 rounded-full" style={{ background: levelColor(level) }} />
              <span style={{ color: levelColor(level) }}>{count} {level}</span>
            </div>
          );
        })}
      </div>

      {/* McLaren Drivers — collapse when a car is selected */}
      {!selectedCar && mclarenVehicles.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {mclarenVehicles.map((v, i) => {
            const color = teamColors[v.team] || '#FF8000';
            const logoKey = TEAM_NAME_TO_LOGO[v.team];
            const logoUrl = logoKey ? TEAM_LOGOS[logoKey] : null;
            const carModel = TEAM_CAR_MODEL[v.team] ?? DEFAULT_CAR_MODEL;
            const trend = getHealthTrend(v);
            const critCount = v.systems.filter(s => s.level === 'critical').length;
            const warnCount = v.systems.filter(s => s.level === 'warning').length;
            const topAction = v.systems
              .map(s => s.maintenanceAction)
              .filter((a): a is string => !!a && a !== 'none')[0];
            return (
              <motion.button
                key={v.code || v.number}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.25, delay: i * 0.05 }}
                onClick={() => setSelectedCar(v)}
                className="text-left rounded-lg transition-all group relative overflow-hidden bg-card border border-primary/25 hover:border-primary/50 shadow-[0_0_20px_rgba(255,128,0,0.06)]"
              >
                <div className="absolute top-0 left-0 bottom-0 w-[4px] rounded-l-xl" style={{ background: color }} />
                <div className="p-5 pl-6">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-1.5 min-w-0">
                      {logoUrl && <img src={logoUrl} alt={v.team} className="h-5 w-5 object-contain shrink-0" />}
                      <span className="text-[12px] font-medium tracking-wide truncate" style={{ color }}>{v.team}</span>
                      <span className="text-[10px] text-muted-foreground/60 font-mono">{carModel.label}</span>
                    </div>
                    <div className="flex items-center gap-1.5 shrink-0">
                      <TrendIcon trend={trend} />
                      <HealthGauge value={v.overallHealth} size={38} showLabel={false} strokeWidth={3} />
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    {v.number > 0 && (
                      <span className="text-[22px] font-black font-mono leading-none opacity-25" style={{ color }}>{v.number}</span>
                    )}
                    <div className="text-[17px] text-foreground font-bold leading-tight group-hover:text-primary transition-colors">
                      {v.driver}
                    </div>
                  </div>

                  {/* Car health summary */}
                  <div className="flex items-center gap-2.5 mt-1.5 text-[11px]">
                    <span className="font-mono font-bold" style={{ color: levelColor(v.level) }}>{v.overallHealth}%</span>
                    <span className="uppercase text-[10px] tracking-wider font-medium" style={{ color: levelColor(v.level) }}>{v.level}</span>
                  </div>

                  {/* Car status indicators */}
                  <div className="flex items-center gap-2.5 mt-2 text-[11px] text-muted-foreground">
                    {critCount > 0 && <span className="font-mono text-red-400">{critCount} critical</span>}
                    {warnCount > 0 && <span className="font-mono text-primary">{warnCount} warning</span>}
                    {critCount === 0 && warnCount === 0 && <span className="font-mono text-green-400">All systems nominal</span>}
                    {topAction && <MaintenanceBadge action={topAction} />}
                  </div>

                  <div className="mt-3 space-y-1">
                    {v.systems.map(sys => (
                      <div key={sys.name} className="flex items-center gap-1.5">
                        <span className="text-[10px] w-12 text-muted-foreground/70 truncate">{sys.name}</span>
                        <div className="flex-1 h-1.5 bg-secondary rounded-full overflow-hidden">
                          <div className="h-full rounded-full transition-all duration-700" style={{ width: `${sys.health}%`, backgroundColor: levelColor(sys.level) }} />
                        </div>
                      </div>
                    ))}
                  </div>

                  <ChevronRight className="w-3.5 h-3.5 text-muted-foreground/30 absolute bottom-3 right-3 group-hover:text-primary/60 transition-colors" />
                </div>
              </motion.button>
            );
          })}
        </div>
      )}

      {/* All other drivers — hidden when a car is selected */}
      {otherVehicles.length > 0 && !selectedCar && (
        <>
          <div className="flex items-center justify-between px-1">
            <span className="text-[12px] font-medium text-muted-foreground uppercase tracking-wider">All Drivers</span>
            <span className="text-[10px] text-muted-foreground">{otherVehicles.length} drivers</span>
          </div>
          <div className="grid grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-2">
            {otherVehicles.map((v, i) => {
              const color = teamColors[v.team] || '#666';
              const logoKey = TEAM_NAME_TO_LOGO[v.team];
              const logoUrl = logoKey ? TEAM_LOGOS[logoKey] : null;
              const carModel = TEAM_CAR_MODEL[v.team] ?? DEFAULT_CAR_MODEL;
              const trend = getHealthTrend(v);
              const critCount = v.systems.filter(s => s.level === 'critical').length;
              const warnCount = v.systems.filter(s => s.level === 'warning').length;
              return (
                <motion.button
                  key={v.code || v.number}
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.25, delay: Math.min(i * 0.02, 0.4) }}
                  onClick={() => setSelectedCar(v)}
                  className="text-left rounded-lg transition-all group relative overflow-hidden bg-card border border-border hover:border-[rgba(255,128,0,0.25)]"
                >
                  <div className="absolute top-0 left-0 bottom-0 w-[2px] rounded-l-xl" style={{ background: color }} />
                  <div className="p-2.5 pl-3.5">
                    <div className="flex items-center justify-between mb-1">
                      <div className="flex items-center gap-1 min-w-0">
                        {logoUrl && <img src={logoUrl} alt={v.team} className="h-3.5 w-3.5 object-contain shrink-0" />}
                        <span className="text-[10px] font-medium tracking-wide truncate" style={{ color }}>{v.team}</span>
                      </div>
                      <div className="flex items-center gap-1 shrink-0">
                        <TrendIcon trend={trend} />
                        <HealthGauge value={v.overallHealth} size={26} showLabel={false} strokeWidth={2.5} />
                      </div>
                    </div>

                    <div className="flex items-center gap-1.5">
                      {v.number > 0 && (
                        <span className="text-[14px] font-black font-mono leading-none opacity-25" style={{ color }}>{v.number}</span>
                      )}
                      <div>
                        <div className="text-[12px] text-foreground font-bold leading-tight group-hover:text-primary transition-colors">
                          {v.driver}
                        </div>
                        <div className="text-[8px] text-muted-foreground/60 font-mono">{carModel.label}</div>
                      </div>
                    </div>

                    {/* Car health + status */}
                    <div className="flex items-center gap-1.5 mt-1 text-[10px]">
                      <span className="font-mono font-bold" style={{ color: levelColor(v.level) }}>{v.overallHealth}%</span>
                      <span className="uppercase text-[8px] tracking-wider" style={{ color: levelColor(v.level) }}>{v.level}</span>
                      {critCount > 0 && <span className="font-mono text-red-400">{critCount}C</span>}
                      {warnCount > 0 && <span className="font-mono text-primary">{warnCount}W</span>}
                    </div>

                    <div className="mt-2 space-y-0.5">
                      {v.systems.map(sys => (
                        <div key={sys.name} className="flex items-center gap-1">
                          <span className="text-[7px] w-8 text-muted-foreground/70 truncate">{sys.name}</span>
                          <div className="flex-1 h-[3px] bg-secondary rounded-full overflow-hidden">
                            <div className="h-full rounded-full transition-all duration-700" style={{ width: `${sys.health}%`, backgroundColor: levelColor(sys.level) }} />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </motion.button>
              );
            })}
          </div>
        </>
      )}

      {/* Driver Detail Panel — shown when a driver is selected */}
      {selectedCar && (() => {
        const carModel = TEAM_CAR_MODEL[selectedCar.team] ?? DEFAULT_CAR_MODEL;
        const s = driverStats;

        return (
        <div className="space-y-4">
          {/* Driver Profile Header */}
          <div className="bg-card rounded-lg border border-primary/20 overflow-hidden">
            <div className="h-1 bg-gradient-to-r from-primary via-primary/60 to-transparent" />
            <div className="px-5 py-4 flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className="w-14 h-14 rounded-lg bg-primary/10 border border-primary/30 flex items-center justify-center">
                  <span className="text-xl font-bold font-mono text-primary">#{selectedCar.number}</span>
                </div>
                <div>
                  <div className="text-lg font-semibold text-foreground">{selectedCar.driver}</div>
                  <div className="flex items-center gap-2 text-[13px] text-muted-foreground">
                    <span>{selectedCar.team}</span>
                    <span className="text-primary/40">|</span>
                    <span className="text-primary/80">{carModel.label}</span>
                    {s.nationality && <>
                      <span className="text-primary/40">|</span>
                      <span>{s.nationality}</span>
                    </>}
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="text-right">
                  <div className="text-2xl font-mono font-bold" style={{ color: levelColor(selectedCar.level) }}>
                    {selectedCar.overallHealth}%
                  </div>
                  <div className="text-[11px] uppercase tracking-wider font-medium" style={{ color: levelColor(selectedCar.level) }}>
                    {selectedCar.level}
                  </div>
                </div>
                <button type="button" title="Close" onClick={() => setSelectedCar(null)} className="text-muted-foreground hover:text-foreground ml-2">
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Quick stats row */}
            {!statsLoading && (s.position || s.races) && (
              <div className="px-5 pb-3 flex flex-wrap gap-x-6 gap-y-1 text-[12px]">
                {s.position != null && (
                  <div><span className="text-muted-foreground">P</span><span className="font-mono font-bold text-foreground ml-0.5">{s.position}</span></div>
                )}
                {s.points != null && (
                  <div><span className="text-muted-foreground">Points </span><span className="font-mono font-bold text-foreground">{s.points}</span></div>
                )}
                {s.wins != null && s.wins > 0 && (
                  <div><span className="text-muted-foreground">Wins </span><span className="font-mono font-bold text-primary">{s.wins}</span></div>
                )}
                {s.podiums != null && s.podiums > 0 && (
                  <div><span className="text-muted-foreground">Podiums </span><span className="font-mono font-bold text-foreground">{s.podiums}</span></div>
                )}
                {s.races != null && (
                  <div><span className="text-muted-foreground">Races </span><span className="font-mono font-bold text-foreground">{s.races}</span></div>
                )}
                {s.dnfs != null && s.dnfs > 0 && (
                  <div><span className="text-muted-foreground">DNFs </span><span className="font-mono font-bold text-red-400">{s.dnfs}</span></div>
                )}
                {s.avgGrid != null && (
                  <div><span className="text-muted-foreground">Avg Grid </span><span className="font-mono text-foreground">{s.avgGrid}</span></div>
                )}
                {s.avgFinish != null && (
                  <div><span className="text-muted-foreground">Avg Finish </span><span className="font-mono text-foreground">{s.avgFinish}</span></div>
                )}
              </div>
            )}
            {statsLoading && (
              <div className="px-5 pb-3 flex items-center gap-2 text-[11px] text-muted-foreground">
                <Loader2 className="w-3 h-3 animate-spin text-primary" /> Loading driver stats...
              </div>
            )}
          </div>

          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.2 }}
              className="space-y-4"
            >

          {/* TELEMETRY TAB */}
          {activeTab === 'telemetry' && (<>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {/* 3D Car Viewer — 2/3 */}
            <div className="lg:col-span-2 bg-card rounded-lg border border-border p-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-medium text-foreground flex items-center gap-2">
                  <Shield className="w-3.5 h-3.5 text-primary" />
                  {carModel.label} — Vehicle Stress
                </h3>
                <div className="flex gap-3">
                  {(['nominal', 'warning', 'critical'] as HealthLevel[]).map(l => (
                    <div key={l} className="flex items-center gap-1.5">
                      <div className="w-2 h-2 rounded-full" style={{ background: levelColor(l) }} />
                      <span className="text-[10px] text-muted-foreground capitalize">{l}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div className="rounded-lg overflow-hidden border border-border">
                <iframe
                  src={`/glb_viewer.html?url=${carModel.url}&title=${encodeURIComponent(carModel.label)}&stress=${encodeURIComponent(JSON.stringify(selectedCar.systems.map(sys => ({ name: sys.name, health: sys.health, level: sys.level, metrics: sys.metrics }))))}`}
                  className="w-full h-[480px] border-0 bg-background"
                  title={`3D — ${carModel.label}`}
                />
              </div>

              {/* Race-by-Race Telemetry Trends — full width stacked under 3D model */}
              {raceTelemetry.length > 1 && (
                <div className="mt-3 space-y-2">
                  {([
                    { key: 'avgSpeed', label: 'Avg Speed', unit: 'km/h', color: '#FF8000' },
                    { key: 'avgThrottle', label: 'Throttle', unit: '%', color: '#22c55e' },
                    { key: 'brakePct', label: 'Brake', unit: '%', color: '#ef4444' },
                  ] as const).map(({ key, label, unit, color }) => (
                    <div key={key} className="bg-background rounded-lg p-2">
                      <ResponsiveContainer width="100%" height={85}>
                        <AreaChart data={raceTelemetry} margin={{ top: 2, right: 2, bottom: 0, left: 2 }}>
                          <defs>
                            <linearGradient id={`grad-${key}`} x1="0" y1="0" x2="0" y2="1">
                              <stop offset="0%" stopColor={color} stopOpacity={0.3} />
                              <stop offset="100%" stopColor={color} stopOpacity={0} />
                            </linearGradient>
                          </defs>
                          <XAxis dataKey="race" hide />
                          <YAxis hide domain={['dataMin - 2', 'dataMax + 2']} />
                          <Tooltip
                            contentStyle={{ background: '#161B22', border: '1px solid #30363D', borderRadius: 8, fontSize: 10 }}
                            formatter={(v: number) => [`${v.toFixed(1)} ${unit}`, label]}
                            labelFormatter={(l: string) => l}
                          />
                          <Area type="monotone" dataKey={key} stroke={color} strokeWidth={1.5} fill={`url(#grad-${key})`} dot={false} />
                        </AreaChart>
                      </ResponsiveContainer>
                      <div className="flex items-center justify-between mt-1">
                        <span className="text-[10px] text-muted-foreground">{label}</span>
                        <span className="text-[10px] font-mono font-semibold" style={{ color }}>
                          {raceTelemetry[raceTelemetry.length - 1][key].toFixed(1)}{unit}
                        </span>
                      </div>
                    </div>
                  ))}
                  {/* Summary stats row */}
                  <div className="flex flex-wrap gap-3 mt-1 text-[10px]">
                    {(() => {
                      const latest = raceTelemetry[raceTelemetry.length - 1];
                      return <>
                        <div><span className="text-muted-foreground">Top Speed </span><span className="font-mono font-semibold text-foreground">{latest.topSpeed} km/h</span></div>
                        <div><span className="text-muted-foreground">RPM </span><span className="font-mono font-semibold text-foreground">{Math.round(latest.avgRPM)}/{Math.round(latest.maxRPM)}</span></div>
                        <div><span className="text-muted-foreground">DRS </span><span className="font-mono font-semibold text-foreground">{latest.drsPct.toFixed(1)}%</span></div>
                        <span className="text-muted-foreground/50">({raceTelemetry.length} races)</span>
                      </>;
                    })()}
                  </div>
                </div>
              )}
            </div>

            {/* Driver Performance Card — 1/3 */}
            <div className="lg:col-span-1 space-y-3">
              {/* Performance Metrics */}
              {(s.avgTopSpeed || s.throttleSmoothness || s.overtakesMade) && (
                <div className="bg-card rounded-lg border border-border p-3">
                  <h3 className="text-[12px] font-medium text-foreground mb-2.5 flex items-center gap-1.5">
                    <Activity className="w-3 h-3 text-primary" />
                    Performance Profile
                  </h3>
                  <div className="space-y-2">
                    {s.avgTopSpeed != null && (
                      <div className="flex items-center justify-between text-[11px]">
                        <span className="text-muted-foreground">Top Speed</span>
                        <span className="font-mono font-semibold text-foreground">{Math.round(s.avgTopSpeed)} km/h</span>
                      </div>
                    )}
                    {s.throttleSmoothness != null && (
                      <div className="flex items-center justify-between text-[11px]">
                        <span className="text-muted-foreground">Throttle Smoothness</span>
                        <span className="font-mono font-semibold text-foreground">{s.throttleSmoothness.toFixed(2)}</span>
                      </div>
                    )}
                    {s.brakeOverlap != null && (
                      <div className="flex items-center justify-between text-[11px]">
                        <span className="text-muted-foreground">Brake Overlap</span>
                        <span className="font-mono font-semibold text-foreground">{(s.brakeOverlap * 100).toFixed(2)}%</span>
                      </div>
                    )}
                    {s.lapConsistency != null && (
                      <div className="flex items-center justify-between text-[11px]">
                        <span className="text-muted-foreground">Lap Consistency (σ)</span>
                        <span className="font-mono font-semibold text-foreground">{s.lapConsistency.toFixed(2)}s</span>
                      </div>
                    )}
                    {s.degradationSlope != null && (
                      <div className="flex items-center justify-between text-[11px]">
                        <span className="text-muted-foreground">Tyre Degradation</span>
                        <span className="font-mono font-semibold" style={{ color: s.degradationSlope < -0.2 ? '#ef4444' : '#22c55e' }}>
                          {s.degradationSlope > 0 ? '+' : ''}{s.degradationSlope.toFixed(3)} s/lap
                        </span>
                      </div>
                    )}
                    {s.lateRaceDelta != null && (
                      <div className="flex items-center justify-between text-[11px]">
                        <span className="text-muted-foreground">Late Race Pace</span>
                        <span className="font-mono font-semibold" style={{ color: s.lateRaceDelta < 0 ? '#22c55e' : '#ef4444' }}>
                          {s.lateRaceDelta > 0 ? '+' : ''}{s.lateRaceDelta.toFixed(2)}s
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Overtake Stats */}
              {s.overtakesMade != null && (
                <div className="bg-card rounded-lg border border-border p-3">
                  <h3 className="text-[12px] font-medium text-foreground mb-2.5 flex items-center gap-1.5">
                    <Sparkles className="w-3 h-3 text-primary" />
                    Overtake Profile
                  </h3>
                  <div className="grid grid-cols-3 gap-2 text-center">
                    <div className="bg-background rounded-lg p-2">
                      <div className="text-lg font-mono font-bold text-[#22c55e]">{s.overtakesMade}</div>
                      <div className="text-[10px] text-muted-foreground">Made</div>
                    </div>
                    <div className="bg-background rounded-lg p-2">
                      <div className="text-lg font-mono font-bold text-red-400">{s.timesOvertaken}</div>
                      <div className="text-[10px] text-muted-foreground">Lost</div>
                    </div>
                    <div className="bg-background rounded-lg p-2">
                      <div className="text-lg font-mono font-bold" style={{ color: (s.overtakeNet ?? 0) >= 0 ? '#22c55e' : '#ef4444' }}>
                        {(s.overtakeNet ?? 0) >= 0 ? '+' : ''}{s.overtakeNet}
                      </div>
                      <div className="text-[10px] text-muted-foreground">Net</div>
                    </div>
                  </div>
                </div>
              )}

              {/* System Health — stacked in sidebar */}
              <div className="bg-card rounded-lg border border-border p-3">
                <h3 className="text-[12px] font-medium text-foreground mb-2 flex items-center gap-1.5">
                  <Gauge className="w-3 h-3 text-primary" />
                  System Health
                </h3>
                <div className="space-y-1.5">
                  {selectedCar.systems.map(sys => {
                    const Icon = sys.icon;
                    return (
                      <div key={sys.name} className="rounded-lg p-2 border" style={{ background: levelBg(sys.level), borderColor: `${levelColor(sys.level)}15` }}>
                        <div className="flex items-center justify-between mb-1">
                          <div className="flex items-center gap-1.5">
                            <Icon className="w-3 h-3" style={{ color: levelColor(sys.level) }} />
                            <span className="text-[12px] font-medium text-foreground">{sys.name}</span>
                          </div>
                          <div className="flex items-center gap-1.5">
                            <span className="text-[12px] font-mono font-semibold" style={{ color: levelColor(sys.level) }}>
                              {sys.health}%
                            </span>
                            {sys.level === 'nominal' && <CheckCircle2 className="w-2.5 h-2.5 text-green-400" />}
                            {sys.level === 'warning' && <AlertTriangle className="w-2.5 h-2.5 text-primary" />}
                            {sys.level === 'critical' && <XCircle className="w-2.5 h-2.5 text-red-400" />}
                          </div>
                        </div>
                        <div className="h-1 bg-background rounded-full overflow-hidden mb-1.5">
                          <div className="h-full rounded-full transition-all duration-700"
                            style={{ width: `${sys.health}%`, background: levelColor(sys.level) }} />
                        </div>
                        <SeverityBar probabilities={sys.severityProbabilities} />
                        {sys.maintenanceAction && sys.maintenanceAction !== 'none' && (
                          <div className="mt-1.5"><MaintenanceBadge action={sys.maintenanceAction} /></div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>


          {/* Tyre Strategy — last race */}
          {stints.length > 0 && lastRaceName && (() => {
            const sorted = [...stints].sort((a, b) => a.stint_number - b.stint_number);
            const totalLaps = sorted.reduce((s, t) => s + t.stint_laps, 0);
            return (
            <div className="bg-card rounded-lg border border-border p-4">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-[12px] font-medium text-foreground flex items-center gap-1.5">
                  <Fuel className="w-3.5 h-3.5 text-primary" />
                  Tyre Strategy — {lastRaceName}
                </h3>
                <span className="text-[10px] text-muted-foreground font-mono">{totalLaps} laps</span>
              </div>
              {/* Timeline bar */}
              <div className="flex items-center gap-1 h-14 rounded-lg overflow-hidden bg-background mb-4">
                {sorted.map(st => {
                  const pct = (st.stint_laps / totalLaps) * 100;
                  const color = COMPOUND_COLORS[st.compound] ?? '#6b7280';
                  return (
                    <div
                      key={st.stint_number}
                      className="h-full flex flex-col items-center justify-center gap-0.5 font-mono"
                      style={{ width: `${pct}%`, background: `${color}20`, borderLeft: st.stint_number > sorted[0].stint_number ? `2px solid ${color}60` : 'none' }}
                    >
                      <span className="text-[12px] font-bold" style={{ color }}>{st.compound}</span>
                      <span className="text-[10px] text-muted-foreground">{st.stint_laps} laps</span>
                    </div>
                  );
                })}
              </div>
              {/* Stint detail cards */}
              <div className={`grid gap-2 ${sorted.length <= 3 ? 'grid-cols-' + sorted.length : 'grid-cols-2 md:grid-cols-4'}`}>
                {sorted.map(st => {
                  const color = COMPOUND_COLORS[st.compound] ?? '#6b7280';
                  return (
                    <div key={st.stint_number} className="bg-background rounded-lg p-3 border" style={{ borderColor: `${color}30` }}>
                      <div className="flex items-center gap-2 mb-2">
                        <div className="w-3 h-3 rounded-full" style={{ background: color }} />
                        <span className="text-[11px] font-semibold text-foreground">Stint {st.stint_number}</span>
                      </div>
                      <div className="space-y-1.5">
                        <div className="flex items-center justify-between text-[10px]">
                          <span className="text-muted-foreground">Compound</span>
                          <span className="font-mono font-bold" style={{ color }}>{st.compound}</span>
                        </div>
                        <div className="flex items-center justify-between text-[10px]">
                          <span className="text-muted-foreground">Laps</span>
                          <span className="font-mono font-semibold text-foreground">{st.lap_start}–{st.lap_end} ({st.stint_laps})</span>
                        </div>
                        <div className="flex items-center justify-between text-[10px]">
                          <span className="text-muted-foreground">Tyre Age</span>
                          <span className="font-mono font-semibold text-foreground">{st.tyre_age_at_start > 0 ? `${st.tyre_age_at_start} laps used` : 'New'}</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
            );
          })()}

          <KexBriefingCard
            title="WISE Telemetry Briefing"
            icon="brain"
            kex={telemetryKex}
            loading={telemetryKexLoading}
            loadingText="Analyzing car telemetry data\u2026"
          />
          </>)}

          {/* ANOMALY TAB */}
          {activeTab === 'anomaly' && (<>
          {/* 3D Stress Model */}
          <div className="bg-card rounded-lg border border-border p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-medium text-foreground flex items-center gap-2">
                <Shield className="w-3.5 h-3.5 text-primary" />
                {carModel.label} — Anomaly Stress Map
              </h3>
              <div className="flex gap-3">
                {(['nominal', 'warning', 'critical'] as HealthLevel[]).map(l => (
                  <div key={l} className="flex items-center gap-1.5">
                    <div className="w-2 h-2 rounded-full" style={{ background: levelColor(l) }} />
                    <span className="text-[10px] text-muted-foreground capitalize">{l}</span>
                  </div>
                ))}
              </div>
            </div>
            <div className="rounded-lg overflow-hidden border border-border">
              <iframe
                src={`/glb_viewer.html?url=${carModel.url}&title=${encodeURIComponent(carModel.label)}&stress=${encodeURIComponent(JSON.stringify(selectedCar.systems.map(sys => ({ name: sys.name, health: sys.health, level: sys.level, metrics: sys.metrics }))))}`}
                className="w-full h-[400px] border-0 bg-background"
                title={`3D Anomaly — ${carModel.label}`}
              />
            </div>
          </div>
          {/* System Detail Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {selectedCar.systems.map(sys => {
              const MaintenanceInfo = MAINTENANCE_LABELS[sys.maintenanceAction ?? 'none'] ?? MAINTENANCE_LABELS.none;
              const MIcon = MaintenanceInfo.icon;
              return (
                <div key={sys.name} className="bg-card rounded-lg border border-border p-3">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-[12px] font-medium text-foreground">{sys.name}</span>
                    <span className="text-[11px] font-mono" style={{ color: levelColor(sys.level) }}>
                      {sys.health.toFixed(0)}%
                    </span>
                  </div>
                  {/* Health bar */}
                  <div className="w-full h-1.5 rounded-full bg-background mb-2">
                    <div className="h-full rounded-full transition-all" style={{ width: `${sys.health}%`, background: levelColor(sys.level) }} />
                  </div>
                  {/* Severity distribution */}
                  {sys.severityProbabilities && (
                    <div className="mb-2">
                      <SeverityBar probabilities={sys.severityProbabilities} />
                    </div>
                  )}
                  {/* Maintenance badge */}
                  <div className="flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium w-fit"
                    style={{ background: `${MaintenanceInfo.color}15`, color: MaintenanceInfo.color, border: `1px solid ${MaintenanceInfo.color}25` }}>
                    <MIcon className="w-2.5 h-2.5" />
                    {MaintenanceInfo.label}
                  </div>
                  {/* Top SHAP features */}
                  {sys.metrics.length > 0 && (
                    <div className="mt-2 space-y-0.5">
                      {sys.metrics.slice(0, 3).map((m, i) => (
                        <div key={i} className="flex items-center justify-between text-[10px]">
                          <span className="text-muted-foreground truncate mr-2">{m.label}</span>
                          <span className="font-mono text-foreground">{m.value}</span>
                        </div>
                      ))}
                    </div>
                  )}
                  {/* Model consensus */}
                  <div className="mt-2 text-[10px] text-muted-foreground">
                    {sys.voteCount}/{sys.totalModels} models agree
                  </div>
                </div>
              );
            })}
          </div>

          {/* OmniHealth Component Assessment */}
          {omniComponents.length > 0 && (
            <div className="bg-card rounded-lg border border-border p-4">
              <h3 className="text-[12px] font-medium text-foreground mb-3 flex items-center gap-1.5">
                <Shield className="w-3 h-3 text-primary" />
                OmniHealth Assessment
                <span className="text-[10px] text-muted-foreground font-normal ml-auto">Live diagnostic</span>
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
                {omniComponents.map(comp => {
                  const sevColor = comp.severity === 'critical' ? '#ef4444'
                    : comp.severity === 'high' ? '#f97316'
                    : comp.severity === 'medium' ? '#eab308'
                    : '#22c55e';
                  return (
                    <div key={comp.component} className="bg-background rounded-lg p-3 border border-border">
                      <div className="flex items-center justify-between mb-1.5">
                        <span className="text-[11px] font-medium text-foreground">{comp.component}</span>
                        <span className="text-[11px] font-mono font-bold" style={{ color: sevColor }}>
                          {comp.health_pct.toFixed(0)}%
                        </span>
                      </div>
                      <div className="w-full h-1.5 rounded-full bg-card mb-2">
                        <div className="h-full rounded-full transition-all" style={{ width: `${comp.health_pct}%`, background: sevColor }} />
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-[10px] px-1.5 py-0.5 rounded font-medium"
                          style={{ background: `${sevColor}15`, color: sevColor, border: `1px solid ${sevColor}25` }}>
                          {comp.severity}
                        </span>
                        <span className="text-[10px] text-muted-foreground">{comp.action.replace(/_/g, ' ')}</span>
                      </div>
                      {comp.confidence != null && (
                        <div className="mt-1.5 text-[10px] text-muted-foreground">
                          Confidence: <span className="font-mono text-foreground">{(comp.confidence * 100).toFixed(0)}%</span>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}
          {omniLoading && (
            <div className="flex items-center gap-2 text-[11px] text-muted-foreground py-2 justify-center">
              <Loader2 className="w-3 h-3 animate-spin text-primary" /> Loading OmniHealth assessment...
            </div>
          )}

          {/* Season Health Trend Table */}
          {trendData.length > 0 && (
            <div className="bg-card rounded-lg border border-border p-3">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-[12px] font-medium text-foreground flex items-center gap-1.5">
                  <TrendingUp className="w-3 h-3 text-primary" />
                  Season Health Trend
                </h3>
                {trendYears.length > 1 && (
                  <div className="flex items-center gap-1">
                    {trendYears.slice(0, 3).map(y => (
                      <button
                        key={y}
                        onClick={() => setTrendYear(y)}
                        className={`px-2 py-0.5 rounded text-[10px] font-mono transition-colors ${
                          trendYear === y
                            ? 'bg-primary text-black font-medium'
                            : 'bg-background text-muted-foreground hover:text-foreground border border-border'
                        }`}
                      >
                        {y}
                      </button>
                    ))}
                    {trendYears.length > 3 && (
                      <details className="relative">
                        <summary className="px-2 py-0.5 rounded text-[10px] font-mono bg-background text-muted-foreground hover:text-foreground border border-border cursor-pointer list-none">
                          +{trendYears.length - 3}
                        </summary>
                        <div className="absolute right-0 top-full mt-1 z-10 bg-card border border-[rgba(255,128,0,0.2)] rounded-lg p-1 flex flex-col gap-0.5">
                          {trendYears.slice(3).map(y => (
                            <button
                              key={y}
                              onClick={() => setTrendYear(y)}
                              className={`px-3 py-1 rounded text-[10px] font-mono transition-colors text-left ${
                                trendYear === y
                                  ? 'bg-primary text-black font-medium'
                                  : 'text-muted-foreground hover:text-foreground hover:bg-[rgba(255,128,0,0.1)]'
                              }`}
                            >
                              {y}
                            </button>
                          ))}
                        </div>
                      </details>
                    )}
                  </div>
                )}
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-[11px]">
                  <thead className="sticky top-0 bg-card">
                    <tr className="border-b border-border">
                      <th className="text-left py-1 text-muted-foreground font-normal">Race</th>
                      {TREND_SYSTEMS.map(sys => (
                        <th key={sys} className="text-right py-1 text-muted-foreground font-normal" title={sys}>{TREND_ABBR[sys]}</th>
                      ))}
                      <th className="text-right py-1 text-muted-foreground font-normal">Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {trendData.slice(-10).map((d, i) => (
                      <tr key={i} className="border-b border-[rgba(255,128,0,0.04)] hover:bg-[rgba(255,128,0,0.02)]">
                        <td className="py-0.5 text-foreground">{d.race}</td>
                        {TREND_SYSTEMS.map(sys => {
                          const h = d.sysHealth[sys] ?? 0;
                          return (
                            <td key={sys} className="py-0.5 text-right font-mono" style={{ color: healthColor(h) }}>{h}%</td>
                          );
                        })}
                        <td className="py-0.5 text-right">
                          {d.maintenanceAction !== 'none' && <MaintenanceBadge action={d.maintenanceAction} />}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          <KexBriefingCard
            title="WISE Anomaly Briefing"
            icon="sparkles"
            kex={anomalyKex}
            loading={anomalyKexLoading}
            loadingText="Extracting anomaly intelligence\u2026"
          />

          {/* Agent Analysis — session picker + live agent activity */}
          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <select
                title="Select race session for agent analysis"
                value={selectedSessionKey ?? ''}
                onChange={e => setSelectedSessionKey(Number(e.target.value))}
                className="bg-secondary border border-border rounded-md px-3 py-1.5 text-[11px] font-mono text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
              >
                {sessions.map(s => (
                  <option key={s.session_key} value={s.session_key}>
                    {s.year} {s.meeting_name} — {s.session_type}
                  </option>
                ))}
                {sessions.length === 0 && <option value="">Loading sessions...</option>}
              </select>
            </div>
            <AgentActivity
              sessionKey={selectedSessionKey}
              driverNumber={selectedCar.number}
            />
          </div>
          </>)}

          {/* FORECAST TAB */}
          {activeTab === 'forecast' && (
            <ForecastChart driverCode={selectedCar.code} />
          )}

            </motion.div>
          </AnimatePresence>

        </div>
        );
      })()}
    </div>
  );
}
