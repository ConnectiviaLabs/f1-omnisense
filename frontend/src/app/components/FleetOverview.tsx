import { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import {
  AlertTriangle, CheckCircle2, XCircle, Activity,
  CircleDot, Shield, TrendingUp, TrendingDown, Minus, Cpu,
  Plus, X, Car, Save, Upload, Box, Loader2, Sparkles, Gauge,
} from 'lucide-react';
import {
  ResponsiveContainer, ComposedChart, Area, Line, XAxis, YAxis, Tooltip, ReferenceLine,
} from 'recharts';
import KexBriefingCard from './KexBriefingCard';
import * as model3dApi from '../api/model3d';
import type { Job } from '../api/model3d';
import {
  type HealthLevel, type VehicleData, type FeatureForecast,
  mapLevel, levelColor, levelBg,
  MAINTENANCE_LABELS, SEVERITY_COLORS,
} from './anomalyHelpers';

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
    <div className="flex h-1.5 rounded-full overflow-hidden bg-[#0D1117]" title="Risk distribution">
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

// NOTE: Car3DViewer standalone section removed from fleet grid view.
// ModelGen3D (AI 3D generation) can be re-enabled in a settings/admin panel.

// ─── Registered vehicle type ────────────────────────────────────────
interface RegisteredVehicle {
  model: string;
  driverName: string;
  driverNumber: number;
  driverCode: string;
  teamName: string;
  chassisId: string;
  engineSpec: string;
  season: number;
  notes: string;
  createdAt: string;
}

const EMPTY_FORM: Omit<RegisteredVehicle, 'createdAt'> = {
  model: '', driverName: '', driverNumber: 0, driverCode: '',
  teamName: 'McLaren', chassisId: '', engineSpec: '', season: new Date().getFullYear(), notes: '',
};

// ─── Component ──────────────────────────────────────────────────────
interface FleetOverviewProps {
  prefetchedVehicles: VehicleData[];
  prefetchedForecasts: Record<string, FeatureForecast[]>;
  prefetchLoading: boolean;
  /** Which pillar tab is active (used by PrimeCar). Phase 2 will filter sections. */
  defaultSection?: 'telemetry' | 'anomaly' | 'forecast';
}

interface AnomalyKex {
  driver_code: string;
  text: string;
  model_used: string;
  provider_used: string;
  scores: Record<string, number>;
  summary: string;
  generated_at: number;
}

async function getAnomalyKex(driverCode: string): Promise<AnomalyKex> {
  const res = await fetch(`/api/anomaly/kex/${encodeURIComponent(driverCode)}`, { method: 'POST' });
  if (!res.ok) throw new Error(`Anomaly KeX failed: ${res.status}`);
  return res.json();
}

export function FleetOverview({ prefetchedVehicles, prefetchedForecasts, prefetchLoading, defaultSection: _defaultSection }: FleetOverviewProps) {
  const [vehicles, setVehicles] = useState<VehicleData[]>([]);
  const [selectedCar, setSelectedCar] = useState<VehicleData | null>(null);
  const [loading, setLoading] = useState(true);
  const [liveSource, setLiveSource] = useState<'cached' | 'live' | null>(null);

  // KeX state
  const [kex, setKex] = useState<AnomalyKex | null>(null);
  const [kexLoading, setKexLoading] = useState(false);

  // Registration state
  const [showRegister, setShowRegister] = useState(false);
  const [regForm, setRegForm] = useState(EMPTY_FORM);
  const [registeredVehicles, setRegisteredVehicles] = useState<RegisteredVehicle[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [regError, setRegError] = useState('');

  // Use pre-fetched forecasts keyed by driver code
  const forecasts = selectedCar ? (prefetchedForecasts[selectedCar.code] ?? []) : [];
  const forecastLoading = selectedCar ? !(selectedCar.code in prefetchedForecasts) : false;

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

      // Latest season standings
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

      // Race results — compute career stats for latest season
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

      // Performance markers
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

      // Overtake profile
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

  // 3D generation state (inside Register Car modal)
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [gen3dProvider, setGen3dProvider] = useState<'hunyuan' | 'meshy'>('hunyuan');
  const [isDragOver, setIsDragOver] = useState(false);
  const [activeJob, setActiveJob] = useState<Job | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, []);

  const handleFile = useCallback((file: File) => {
    setImageFile(file);
    setImagePreview(URL.createObjectURL(file));
  }, []);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) handleFile(file);
  }, [handleFile]);

  // Fetch registered vehicles
  useEffect(() => {
    fetch('/api/fleet-vehicles')
      .then(r => r.json())
      .then(data => { if (Array.isArray(data)) setRegisteredVehicles(data); })
      .catch(() => {});
  }, []);

  const startGen3dPolling = (jobId: string) => {
    pollRef.current = setInterval(async () => {
      try {
        const updated = await model3dApi.getJobStatus(jobId);
        setActiveJob(updated);
        if (updated.status === 'completed' || updated.status === 'failed') {
          if (pollRef.current) clearInterval(pollRef.current);
          pollRef.current = null;
        }
      } catch { /* ignore poll errors */ }
    }, 3000);
  };

  const handleRegister = async () => {
    if (!regForm.model || !regForm.driverName || !regForm.driverNumber || !regForm.driverCode) {
      setRegError('Model, driver name, number, and code are required.');
      return;
    }
    setSubmitting(true);
    setRegError('');
    try {
      const res = await fetch('/api/fleet-vehicles', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(regForm),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || 'Registration failed');
      }
      const created = await res.json();
      setRegisteredVehicles(prev => [created, ...prev]);

      // If reference image was uploaded, trigger 3D generation
      if (imageFile) {
        try {
          const genName = regForm.model.replace(/[^a-zA-Z0-9_-]/g, '_') || 'car_model';
          const result = await model3dApi.submitGeneration({
            image: imageFile,
            model_name: genName,
            provider: gen3dProvider,
            enable_pbr: true,
          });
          const job: Job = {
            job_id: result.job_id,
            model_name: result.model_name,
            provider: result.provider,
            status: 'queued',
            progress: 0,
            glb_url: null,
            error: null,
            created_at: new Date().toISOString(),
            completed_at: null,
          };
          setActiveJob(job);
          startGen3dPolling(result.job_id);
        } catch { /* 3D gen is optional, don't block registration */ }
      }

      setRegForm(EMPTY_FORM);
      setImageFile(null);
      setImagePreview(null);
      setShowRegister(false);
    } catch (e: any) {
      setRegError(e.message);
    } finally {
      setSubmitting(false);
    }
  };

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

          // Merge live OmniHealth components into existing SystemHealth shape
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

  // Auto-generate KeX when selected car changes (backend auto-regenerates if data changed)
  useEffect(() => {
    setKex(null);
    if (!selectedCar) return;
    setKexLoading(true);
    getAnomalyKex(selectedCar.code)
      .then(setKex)
      .catch(() => setKex(null))
      .finally(() => setKexLoading(false));
  }, [selectedCar?.code]); // eslint-disable-line react-hooks/exhaustive-deps

  // Race-by-race trend from anomaly data
  // Extract available years from the selected car's race history
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

  // Reset year filter when driver changes (default to latest year)
  useEffect(() => {
    setTrendYear(trendYears[0] ?? null);
  }, [trendYears]);

  const trendData = useMemo(() => {
    if (!selectedCar) return [];
    return selectedCar.races
      .filter(r => !trendYear || r.race.startsWith(`${trendYear} `))
      .map(r => {
        const systems = r.systems;
        const speedFeats = systems['Speed']?.features ?? {};
        const paceFeats = systems['Lap Pace']?.features ?? {};
        const tyreFeats = systems['Tyre Management']?.features ?? {};

        // Collect maintenance actions across all systems for this race
        const actions = Object.values(systems)
          .map(s => s.maintenance_action)
          .filter((a): a is string => !!a && a !== 'none');
        const actionPriority = ['alert_and_remediate', 'alert', 'log_and_monitor', 'log'];
        const topAction = actionPriority.find(a => actions.includes(a)) ?? 'none';

        return {
          race: r.race.replace(/^\d{4}\s+/, ''),
          speedHealth: systems['Speed']?.health ?? 0,
          paceHealth: systems['Lap Pace']?.health ?? 0,
          tyreHealth: systems['Tyre Management']?.health ?? 0,
          speedST: speedFeats['SpeedST'] ?? 0,
          lapTime: paceFeats['LapTime'] ?? 0,
          tyreLife: tyreFeats['TyreLife'] ?? 0,
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
        <Activity className="w-5 h-5 text-[#FF8000] animate-spin" />
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

  return (
    <div className="space-y-4">
      {/* Fleet status bar */}
      <div className="flex items-center gap-4 flex-wrap">
        <div className="flex items-center gap-2 bg-[#1A1F2E] rounded-lg px-3 py-2 border border-[rgba(255,128,0,0.12)]">
          <CircleDot className="w-3 h-3 text-[#FF8000]" />
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
        <button
          type="button"
          onClick={() => setShowRegister(true)}
          className="ml-auto flex items-center gap-1.5 bg-[#FF8000] hover:bg-[#FF8000]/80 text-black text-[12px] font-medium rounded-lg px-3 py-2 transition-colors"
        >
          <Plus className="w-3.5 h-3.5" />
          Register Car
        </button>
      </div>

      {/* ── McLaren Drivers — full cards ── */}
      {mclarenVehicles.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {mclarenVehicles.map(v => {
            const isSelected = selectedCar?.code === v.code;
            return (
              <button
                key={v.code || v.number}
                onClick={() => setSelectedCar(isSelected ? null : v)}
                className={`text-left bg-[#1A1F2E] rounded-xl border p-5 transition-all group ${
                  isSelected
                    ? 'border-[#FF8000] ring-1 ring-[#FF8000]/30 shadow-[0_0_20px_rgba(255,128,0,0.1)]'
                    : 'border-[rgba(255,128,0,0.12)] hover:border-[rgba(255,128,0,0.3)]'
                }`}
              >
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-lg flex items-center justify-center text-sm font-bold font-mono bg-[#FF8000]/15 text-[#FF8000] border border-[#FF8000]/40">
                      #{v.number}
                    </div>
                    <div>
                      <div className="text-sm font-medium text-[#FF8000]">{v.driver}</div>
                      <div className="text-[12px] text-muted-foreground">{v.team} &middot; {v.lastRace}</div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-lg font-mono font-bold" style={{ color: levelColor(v.level) }}>
                      {v.overallHealth}%
                    </div>
                    <div className="text-[11px] uppercase tracking-wider" style={{ color: levelColor(v.level) }}>
                      {v.level}
                    </div>
                  </div>
                </div>
                <div className="h-2 bg-[#0D1117] rounded-full overflow-hidden mb-4">
                  <div
                    className="h-full rounded-full transition-all duration-700"
                    style={{ width: `${v.overallHealth}%`, background: `linear-gradient(90deg, ${levelColor(v.level)}, ${levelColor(v.level)}88)` }}
                  />
                </div>
                <div className="grid grid-cols-3 gap-2">
                  {v.systems.map(sys => {
                    const Icon = sys.icon;
                    return (
                      <div key={sys.name} className="bg-[#0D1117] rounded-lg p-2">
                        <div className="flex items-center gap-1.5 mb-1">
                          <Icon className="w-2.5 h-2.5" style={{ color: levelColor(sys.level) }} />
                          <span className="text-[10px] text-muted-foreground truncate">{sys.name}</span>
                        </div>
                        <div className="h-1 bg-[#222838] rounded-full overflow-hidden">
                          <div className="h-full rounded-full" style={{ width: `${sys.health}%`, background: levelColor(sys.level) }} />
                        </div>
                        <div className="text-[11px] font-mono mt-0.5" style={{ color: levelColor(sys.level) }}>{sys.health}%</div>
                      </div>
                    );
                  })}
                </div>
              </button>
            );
          })}
        </div>
      )}

      {/* ── Grid: all other drivers as compact rows ── */}
      {otherVehicles.length > 0 && (
        <div className="bg-[#1A1F2E] rounded-xl border border-[rgba(255,128,0,0.12)] overflow-hidden">
          <div className="px-4 py-2.5 border-b border-[rgba(255,128,0,0.08)] flex items-center justify-between">
            <span className="text-[12px] font-medium text-muted-foreground">All Drivers</span>
            <span className="text-[10px] text-muted-foreground">{otherVehicles.length} drivers</span>
          </div>
          <div className="max-h-[400px] overflow-y-auto">
            {otherVehicles.map(v => {
              const isSelected = selectedCar?.code === v.code;
              return (
                <button
                  key={v.code || v.number}
                  onClick={() => setSelectedCar(isSelected ? null : v)}
                  className={`w-full text-left flex items-center gap-3 px-4 py-2.5 border-b border-[rgba(255,128,0,0.04)] transition-colors ${
                    isSelected
                      ? 'bg-[#FF8000]/5 border-l-2 border-l-[#FF8000]'
                      : 'hover:bg-[rgba(255,128,0,0.02)]'
                  }`}
                >
                  <div className="w-8 h-8 rounded-lg flex items-center justify-center text-[11px] font-bold font-mono shrink-0"
                    style={{ background: levelBg(v.level), color: levelColor(v.level), border: `1px solid ${levelColor(v.level)}22` }}>
                    #{v.number}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className={`text-[12px] font-medium ${isSelected ? 'text-[#FF8000]' : 'text-foreground'}`}>{v.driver}</span>
                      <span className="text-[10px] text-muted-foreground truncate">{v.team}</span>
                    </div>
                  </div>
                  {/* Mini system bars */}
                  <div className="flex items-center gap-2 shrink-0">
                    {v.systems.map(sys => (
                      <div key={sys.name} className="flex items-center gap-1" title={`${sys.name}: ${sys.health}%`}>
                        <div className="w-12 h-1.5 bg-[#0D1117] rounded-full overflow-hidden">
                          <div className="h-full rounded-full" style={{ width: `${sys.health}%`, background: levelColor(sys.level) }} />
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className="w-12 text-right shrink-0">
                    <span className="text-[12px] font-mono font-semibold" style={{ color: levelColor(v.level) }}>
                      {v.overallHealth}%
                    </span>
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* ─── Driver Detail Panel — shown when a driver is selected ─── */}
      {selectedCar && (() => {
        const carModel = TEAM_CAR_MODEL[selectedCar.team] ?? DEFAULT_CAR_MODEL;
        const s = driverStats;
        const healthColor = (h: number) => h >= 75 ? '#22c55e' : h >= 50 ? '#FF8000' : '#ef4444';

        return (
        <div className="space-y-4">
          {/* ── Driver Profile Header ── */}
          <div className="bg-[#1A1F2E] rounded-xl border border-[#FF8000]/20 overflow-hidden">
            <div className="h-1 bg-gradient-to-r from-[#FF8000] via-[#FF8000]/60 to-transparent" />
            <div className="px-5 py-4 flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className="w-14 h-14 rounded-xl bg-[#FF8000]/10 border border-[#FF8000]/30 flex items-center justify-center">
                  <span className="text-xl font-bold font-mono text-[#FF8000]">#{selectedCar.number}</span>
                </div>
                <div>
                  <div className="text-lg font-semibold text-foreground">{selectedCar.driver}</div>
                  <div className="flex items-center gap-2 text-[13px] text-muted-foreground">
                    <span>{selectedCar.team}</span>
                    <span className="text-[#FF8000]/40">|</span>
                    <span className="text-[#FF8000]/80">{carModel.label}</span>
                    {s.nationality && <>
                      <span className="text-[#FF8000]/40">|</span>
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
                  <div><span className="text-muted-foreground">Wins </span><span className="font-mono font-bold text-[#FF8000]">{s.wins}</span></div>
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
                <Loader2 className="w-3 h-3 animate-spin text-[#FF8000]" /> Loading driver stats...
              </div>
            )}
          </div>

          {/* ── 3D Car + Driver Stats ── */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {/* 3D Car Viewer — 2/3 */}
            <div className="lg:col-span-2 bg-[#1A1F2E] rounded-xl border border-[rgba(255,128,0,0.12)] p-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-medium text-foreground flex items-center gap-2">
                  <Shield className="w-3.5 h-3.5 text-[#FF8000]" />
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
              <div className="rounded-lg overflow-hidden border border-[rgba(255,128,0,0.12)]">
                <iframe
                  src={`/glb_viewer.html?url=${carModel.url}&title=${encodeURIComponent(carModel.label)}&stress=${encodeURIComponent(JSON.stringify(selectedCar.systems.map(sys => ({ name: sys.name, health: sys.health, level: sys.level, metrics: sys.metrics }))))}`}
                  className="w-full h-[480px] border-0 bg-[#0D1117]"
                  title={`3D — ${carModel.label}`}
                />
              </div>
            </div>

            {/* Driver Performance Card — 1/3 */}
            <div className="lg:col-span-1 space-y-3">
              {/* Performance Metrics */}
              {(s.avgTopSpeed || s.throttleSmoothness || s.overtakesMade) && (
                <div className="bg-[#1A1F2E] rounded-xl border border-[rgba(255,128,0,0.12)] p-3">
                  <h3 className="text-[12px] font-medium text-foreground mb-2.5 flex items-center gap-1.5">
                    <Activity className="w-3 h-3 text-[#FF8000]" />
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
                <div className="bg-[#1A1F2E] rounded-xl border border-[rgba(255,128,0,0.12)] p-3">
                  <h3 className="text-[12px] font-medium text-foreground mb-2.5 flex items-center gap-1.5">
                    <Sparkles className="w-3 h-3 text-[#FF8000]" />
                    Overtake Profile
                  </h3>
                  <div className="grid grid-cols-3 gap-2 text-center">
                    <div className="bg-[#0D1117] rounded-lg p-2">
                      <div className="text-lg font-mono font-bold text-[#22c55e]">{s.overtakesMade}</div>
                      <div className="text-[9px] text-muted-foreground">Made</div>
                    </div>
                    <div className="bg-[#0D1117] rounded-lg p-2">
                      <div className="text-lg font-mono font-bold text-red-400">{s.timesOvertaken}</div>
                      <div className="text-[9px] text-muted-foreground">Lost</div>
                    </div>
                    <div className="bg-[#0D1117] rounded-lg p-2">
                      <div className="text-lg font-mono font-bold" style={{ color: (s.overtakeNet ?? 0) >= 0 ? '#22c55e' : '#ef4444' }}>
                        {(s.overtakeNet ?? 0) >= 0 ? '+' : ''}{s.overtakeNet}
                      </div>
                      <div className="text-[9px] text-muted-foreground">Net</div>
                    </div>
                  </div>
                </div>
              )}

              {/* System Health — stacked in sidebar */}
              <div className="bg-[#1A1F2E] rounded-xl border border-[rgba(255,128,0,0.12)] p-3">
                <h3 className="text-[12px] font-medium text-foreground mb-2 flex items-center gap-1.5">
                  <Gauge className="w-3 h-3 text-[#FF8000]" />
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
                            {sys.level === 'warning' && <AlertTriangle className="w-2.5 h-2.5 text-[#FF8000]" />}
                            {sys.level === 'critical' && <XCircle className="w-2.5 h-2.5 text-red-400" />}
                          </div>
                        </div>
                        <div className="h-1 bg-[#0D1117] rounded-full overflow-hidden mb-1.5">
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

          {/* ── WISE Anomaly Briefing ── */}
          <KexBriefingCard
            title="WISE Anomaly Briefing"
            icon="sparkles"
            kex={kex}
            loading={kexLoading}
            loadingText="Extracting anomaly intelligence\u2026"
          />

          {/* ── Season Health Trend ── */}
          {trendData.length > 0 && (
            <div className="bg-[#1A1F2E] rounded-xl border border-[rgba(255,128,0,0.12)] p-3">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-[12px] font-medium text-foreground flex items-center gap-1.5">
                  <TrendingUp className="w-3 h-3 text-[#FF8000]" />
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
                            ? 'bg-[#FF8000] text-black font-medium'
                            : 'bg-[#0D1117] text-muted-foreground hover:text-foreground border border-[rgba(255,128,0,0.12)]'
                        }`}
                      >
                        {y}
                      </button>
                    ))}
                    {trendYears.length > 3 && (
                      <details className="relative">
                        <summary className="px-2 py-0.5 rounded text-[10px] font-mono bg-[#0D1117] text-muted-foreground hover:text-foreground border border-[rgba(255,128,0,0.12)] cursor-pointer list-none">
                          +{trendYears.length - 3}
                        </summary>
                        <div className="absolute right-0 top-full mt-1 z-10 bg-[#1A1F2E] border border-[rgba(255,128,0,0.2)] rounded-lg p-1 flex flex-col gap-0.5">
                          {trendYears.slice(3).map(y => (
                            <button
                              key={y}
                              onClick={() => setTrendYear(y)}
                              className={`px-3 py-1 rounded text-[10px] font-mono transition-colors text-left ${
                                trendYear === y
                                  ? 'bg-[#FF8000] text-black font-medium'
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
                  <thead className="sticky top-0 bg-[#1A1F2E]">
                    <tr className="border-b border-[rgba(255,128,0,0.12)]">
                      <th className="text-left py-1 text-muted-foreground font-normal">Race</th>
                      <th className="text-right py-1 text-muted-foreground font-normal">Speed</th>
                      <th className="text-right py-1 text-muted-foreground font-normal">Pace</th>
                      <th className="text-right py-1 text-muted-foreground font-normal">Tyres</th>
                      <th className="text-right py-1 text-muted-foreground font-normal">Trap km/h</th>
                      <th className="text-right py-1 text-muted-foreground font-normal">Lap (s)</th>
                      <th className="text-right py-1 text-muted-foreground font-normal">Tyre Life</th>
                      <th className="text-right py-1 text-muted-foreground font-normal">Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {trendData.slice(-10).map((d, i) => (
                      <tr key={i} className="border-b border-[rgba(255,128,0,0.04)] hover:bg-[rgba(255,128,0,0.02)]">
                        <td className="py-0.5 text-foreground">{d.race}</td>
                        <td className="py-0.5 text-right font-mono" style={{ color: healthColor(d.speedHealth) }}>{d.speedHealth}%</td>
                        <td className="py-0.5 text-right font-mono" style={{ color: healthColor(d.paceHealth) }}>{d.paceHealth}%</td>
                        <td className="py-0.5 text-right font-mono" style={{ color: healthColor(d.tyreHealth) }}>{d.tyreHealth}%</td>
                        <td className="py-0.5 text-right font-mono text-foreground">{d.speedST ? Math.round(d.speedST) : '—'}</td>
                        <td className="py-0.5 text-right font-mono text-foreground">{d.lapTime ? d.lapTime.toFixed(1) : '—'}</td>
                        <td className="py-0.5 text-right font-mono text-foreground">{d.tyreLife ? Math.round(d.tyreLife) : '—'}</td>
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

          {/* ── Feature Forecasts ── */}
          {forecastLoading && (
            <div className="flex items-center gap-2 text-[12px] text-muted-foreground py-2">
              <Loader2 className="w-3.5 h-3.5 animate-spin text-[#FF8000]" />
              Forecasting anomaly features...
            </div>
          )}
          {forecasts.length > 0 && (
            <div className="bg-[#1A1F2E] rounded-xl border border-[rgba(255,128,0,0.12)] p-3">
              <h3 className="text-[12px] font-medium text-foreground mb-3 flex items-center gap-1.5">
                <TrendingUp className="w-3 h-3 text-[#FF8000]" />
                Feature Forecasts
                <span className="text-[10px] text-muted-foreground font-normal ml-1">Critical/High anomaly drivers</span>
              </h3>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                {forecasts.map(fc => {
                  const label = fc.column
                    .replace(/_mean$/, '')
                    .replace(/([A-Z])/g, ' $1')
                    .replace(/^[\s]/, '')
                    .replace(/I(\d)/, '(I$1)')
                    .replace(/_/g, ' ')
                    .trim();

                  const TrendIcon = fc.trend_direction === 'rising' ? TrendingUp
                    : fc.trend_direction === 'falling' ? TrendingDown : Minus;
                  const trendColor = fc.trend_direction === 'rising' ? '#22c55e'
                    : fc.trend_direction === 'falling' ? '#ef4444' : '#6b7280';

                  const historyData = (fc.history ?? []).map((v, i) => ({
                    step: fc.history_timestamps?.[i] ?? `H${i}`,
                    actual: v, value: null as number | null,
                    lower: null as number | null, upper: null as number | null,
                  }));
                  const bridgeStep = fc.history?.length
                    ? { step: 'now', actual: fc.history[fc.history.length - 1], value: fc.data[0]?.value ?? null, lower: null as number | null, upper: null as number | null }
                    : null;
                  const forecastData = fc.data.map(d => ({
                    step: d.step, actual: null as number | null,
                    value: d.value, lower: d.lower, upper: d.upper,
                  }));
                  const chartData = [...historyData, ...(bridgeStep ? [bridgeStep] : []), ...forecastData];

                  return (
                    <div key={fc.column} className="bg-[#0D1117] rounded-lg p-3 border border-[rgba(255,128,0,0.08)]">
                      <div className="flex items-center justify-between mb-1.5">
                        <div className="flex items-center gap-1.5">
                          <span className="text-[11px] font-medium text-foreground">{label}</span>
                          <TrendIcon className="w-3 h-3" style={{ color: trendColor }} />
                          {fc.trend_pct != null && (
                            <span className="text-[9px] font-mono font-semibold px-1 py-0.5 rounded"
                              style={{ color: trendColor, background: `${trendColor}15` }}>
                              {fc.trend_pct > 0 ? '+' : ''}{fc.trend_pct.toFixed(1)}%
                            </span>
                          )}
                        </div>
                        <div className="flex items-center gap-1.5">
                          {fc.risk_flag && (
                            <span className="text-[8px] font-bold uppercase px-1.5 py-0.5 rounded bg-red-500/10 text-red-400 border border-red-500/20">Risk</span>
                          )}
                          <span className="text-[9px] text-muted-foreground font-mono uppercase">{fc.method}</span>
                        </div>
                      </div>
                      <ResponsiveContainer width="100%" height={180}>
                        <ComposedChart data={chartData} margin={{ top: 4, right: 4, bottom: 0, left: 4 }}>
                          <defs>
                            <linearGradient id={`fc-${fc.column}`} x1="0" y1="0" x2="0" y2="1">
                              <stop offset="0%" stopColor="#FF8000" stopOpacity={0.3} />
                              <stop offset="100%" stopColor="#FF8000" stopOpacity={0} />
                            </linearGradient>
                          </defs>
                          <XAxis dataKey="step" tick={{ fontSize: 8, fill: '#6b7280' }} axisLine={false} tickLine={false} />
                          <YAxis tick={{ fontSize: 8, fill: '#6b7280' }} axisLine={false} tickLine={false} width={40} />
                          <Tooltip
                            contentStyle={{ background: '#1A1F2E', border: '1px solid rgba(255,128,0,0.2)', borderRadius: 8, fontSize: 11 }}
                            labelStyle={{ color: '#9ca3af' }}
                            formatter={(v: any, name: string) => v != null ? [Number(v).toFixed(2), name] : ['-', name]}
                          />
                          <ReferenceLine x="now" stroke="rgba(255,128,0,0.4)" strokeDasharray="4 3" label={{ value: 'now', position: 'top', fontSize: 8, fill: '#FF8000' }} />
                          <Area type="monotone" dataKey="upper" stroke="none" fill="#FF8000" fillOpacity={0.06} name="Upper" connectNulls={false} />
                          <Area type="monotone" dataKey="lower" stroke="none" fill="#0D1117" fillOpacity={1} name="Lower" connectNulls={false} />
                          <Area type="monotone" dataKey="value" stroke="#FF8000" fill={`url(#fc-${fc.column})`} strokeWidth={1.5} dot={false} name="Forecast" connectNulls={false} />
                          <Line type="monotone" dataKey="actual" stroke="#6b7280" strokeWidth={1.5} strokeDasharray="4 2" dot={{ r: 2, fill: '#6b7280' }} name="Actual" connectNulls />
                        </ComposedChart>
                      </ResponsiveContainer>
                      <div className="flex items-center gap-3 mt-1.5 text-[9px] text-muted-foreground font-mono">
                        {fc.mae != null && <span>MAE {fc.mae.toFixed(2)}</span>}
                        {fc.rmse != null && <span>RMSE {fc.rmse.toFixed(2)}</span>}
                        {fc.volatility != null && (
                          <span style={{ color: fc.volatility > 0.2 ? '#FF8000' : undefined }}>Vol {(fc.volatility * 100).toFixed(0)}%</span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}


        </div>
        );
      })()}

      {/* ─── Registered Vehicles ─────────────────────────────────────── */}
      {registeredVehicles.length > 0 && (
        <div>
          <h3 className="text-sm font-medium text-foreground mb-3 flex items-center gap-2">
            <Car className="w-3.5 h-3.5 text-[#FF8000]" />
            Registered Vehicles
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {registeredVehicles.map((rv, i) => (
              <div key={i} className="bg-[#1A1F2E] rounded-xl border border-[rgba(255,128,0,0.12)] p-4">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <div className="w-8 h-8 rounded-lg bg-[#FF8000]/10 border border-[#FF8000]/20 flex items-center justify-center text-[12px] font-bold font-mono text-[#FF8000]">
                      #{rv.driverNumber}
                    </div>
                    <div>
                      <div className="text-sm font-medium text-foreground">{rv.driverName}</div>
                      <div className="text-[11px] text-muted-foreground">{rv.teamName} {rv.model}</div>
                    </div>
                  </div>
                  <span className="text-[10px] text-muted-foreground bg-[#0D1117] px-2 py-0.5 rounded font-mono">{rv.driverCode}</span>
                </div>
                <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-[11px] mt-2">
                  {rv.chassisId && (
                    <><span className="text-muted-foreground">Chassis</span><span className="text-foreground font-mono">{rv.chassisId}</span></>
                  )}
                  {rv.engineSpec && (
                    <><span className="text-muted-foreground">Engine</span><span className="text-foreground font-mono">{rv.engineSpec}</span></>
                  )}
                  <span className="text-muted-foreground">Season</span><span className="text-foreground font-mono">{rv.season}</span>
                </div>
                {rv.notes && (
                  <p className="text-[11px] text-muted-foreground mt-2 italic border-t border-[rgba(255,128,0,0.06)] pt-2">{rv.notes}</p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 3D Generation Progress (shown after registration with image) */}
      {activeJob && (
        <div className="bg-[#1A1F2E] rounded-xl border border-[rgba(255,128,0,0.12)] p-4">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-medium text-foreground flex items-center gap-2">
              {activeJob.status === 'completed' ? (
                <CheckCircle2 className="w-3.5 h-3.5 text-green-400" />
              ) : activeJob.status === 'failed' ? (
                <XCircle className="w-3.5 h-3.5 text-red-400" />
              ) : (
                <Loader2 className="w-3.5 h-3.5 text-[#FF8000] animate-spin" />
              )}
              3D Generation: {activeJob.model_name}
            </h3>
            <span className="text-[12px] px-2 py-0.5 rounded-full bg-[#FF8000]/10 text-[#FF8000]">
              {activeJob.provider}
            </span>
          </div>
          <div className="h-1.5 bg-[#0D1117] rounded-full overflow-hidden mb-2">
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{
                width: `${activeJob.progress}%`,
                background: activeJob.status === 'failed' ? '#ef4444' : '#FF8000',
              }}
            />
          </div>
          <div className="flex items-center justify-between text-[12px]">
            <span className="text-muted-foreground">
              {activeJob.status === 'queued' ? 'Queued...' : activeJob.status === 'generating' ? 'Generating 3D...' : activeJob.status === 'completed' ? 'Complete' : activeJob.status === 'failed' ? 'Failed' : activeJob.status}
            </span>
            <span className="text-muted-foreground">{activeJob.progress}%</span>
          </div>
          {activeJob.error && <p className="text-[12px] text-red-400 mt-2">{activeJob.error}</p>}
          {activeJob.status === 'completed' && activeJob.glb_url && (
            <a
              href={activeJob.glb_url}
              download
              className="inline-flex items-center gap-1.5 mt-2 px-3 py-1.5 rounded-lg text-[12px] bg-[#FF8000]/10 text-[#FF8000] border border-[#FF8000]/20 hover:bg-[#FF8000]/20 transition-all"
            >
              Download GLB
            </a>
          )}
        </div>
      )}

      {/* ─── Registration Modal ───────────────────────────────────────── */}
      {showRegister && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.2)] rounded-2xl w-full max-w-lg mx-4 shadow-2xl">
            {/* Header */}
            <div className="flex items-center justify-between px-5 py-4 border-b border-[rgba(255,128,0,0.12)]">
              <h2 className="text-sm font-semibold text-foreground flex items-center gap-2">
                <Car className="w-4 h-4 text-[#FF8000]" />
                Register New Car
              </h2>
              <button type="button" title="Close" onClick={() => { setShowRegister(false); setRegError(''); }} className="text-muted-foreground hover:text-foreground">
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Form */}
            <div className="px-5 py-4 space-y-3 max-h-[70vh] overflow-y-auto">
              {regError && (
                <div className="text-[12px] text-red-400 bg-red-400/10 border border-red-400/20 rounded-lg px-3 py-2">
                  {regError}
                </div>
              )}

              <div className="grid grid-cols-2 gap-3">
                <label className="space-y-1">
                  <span className="text-[11px] text-muted-foreground">Car Model *</span>
                  <input
                    type="text" placeholder="MCL38"
                    value={regForm.model} onChange={e => setRegForm(f => ({ ...f, model: e.target.value }))}
                    className="w-full bg-[#0D1117] border border-[rgba(255,128,0,0.15)] rounded-lg px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground/40 focus:outline-none focus:border-[#FF8000]/50"
                  />
                </label>
                <label className="space-y-1">
                  <span className="text-[11px] text-muted-foreground">Team Name</span>
                  <input
                    type="text" placeholder="McLaren"
                    value={regForm.teamName} onChange={e => setRegForm(f => ({ ...f, teamName: e.target.value }))}
                    className="w-full bg-[#0D1117] border border-[rgba(255,128,0,0.15)] rounded-lg px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground/40 focus:outline-none focus:border-[#FF8000]/50"
                  />
                </label>
              </div>

              <div className="grid grid-cols-3 gap-3">
                <label className="space-y-1">
                  <span className="text-[11px] text-muted-foreground">Driver Name *</span>
                  <input
                    type="text" placeholder="Lando Norris"
                    value={regForm.driverName} onChange={e => setRegForm(f => ({ ...f, driverName: e.target.value }))}
                    className="w-full bg-[#0D1117] border border-[rgba(255,128,0,0.15)] rounded-lg px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground/40 focus:outline-none focus:border-[#FF8000]/50"
                  />
                </label>
                <label className="space-y-1">
                  <span className="text-[11px] text-muted-foreground">Number *</span>
                  <input
                    type="number" placeholder="4"
                    value={regForm.driverNumber || ''} onChange={e => setRegForm(f => ({ ...f, driverNumber: Number(e.target.value) }))}
                    className="w-full bg-[#0D1117] border border-[rgba(255,128,0,0.15)] rounded-lg px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground/40 focus:outline-none focus:border-[#FF8000]/50"
                  />
                </label>
                <label className="space-y-1">
                  <span className="text-[11px] text-muted-foreground">Code *</span>
                  <input
                    type="text" placeholder="NOR" maxLength={3}
                    value={regForm.driverCode} onChange={e => setRegForm(f => ({ ...f, driverCode: e.target.value.toUpperCase() }))}
                    className="w-full bg-[#0D1117] border border-[rgba(255,128,0,0.15)] rounded-lg px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground/40 focus:outline-none focus:border-[#FF8000]/50 uppercase"
                  />
                </label>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <label className="space-y-1">
                  <span className="text-[11px] text-muted-foreground">Chassis ID</span>
                  <input
                    type="text" placeholder="MCL38-001"
                    value={regForm.chassisId} onChange={e => setRegForm(f => ({ ...f, chassisId: e.target.value }))}
                    className="w-full bg-[#0D1117] border border-[rgba(255,128,0,0.15)] rounded-lg px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground/40 focus:outline-none focus:border-[#FF8000]/50"
                  />
                </label>
                <label className="space-y-1">
                  <span className="text-[11px] text-muted-foreground">Season</span>
                  <input
                    type="number" placeholder="2024"
                    value={regForm.season} onChange={e => setRegForm(f => ({ ...f, season: Number(e.target.value) }))}
                    className="w-full bg-[#0D1117] border border-[rgba(255,128,0,0.15)] rounded-lg px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground/40 focus:outline-none focus:border-[#FF8000]/50"
                  />
                </label>
              </div>

              <label className="space-y-1 block">
                <span className="text-[11px] text-muted-foreground">Engine Spec</span>
                <input
                  type="text" placeholder="Mercedes-AMG F1 M14"
                  value={regForm.engineSpec} onChange={e => setRegForm(f => ({ ...f, engineSpec: e.target.value }))}
                  className="w-full bg-[#0D1117] border border-[rgba(255,128,0,0.15)] rounded-lg px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground/40 focus:outline-none focus:border-[#FF8000]/50"
                />
              </label>

              <label className="space-y-1 block">
                <span className="text-[11px] text-muted-foreground">Notes</span>
                <textarea
                  rows={2} placeholder="Additional notes..."
                  value={regForm.notes} onChange={e => setRegForm(f => ({ ...f, notes: e.target.value }))}
                  className="w-full bg-[#0D1117] border border-[rgba(255,128,0,0.15)] rounded-lg px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground/40 focus:outline-none focus:border-[#FF8000]/50 resize-none"
                />
              </label>

              {/* 3D Reference Image */}
              <div className="pt-3 border-t border-[rgba(255,128,0,0.08)]">
                <span className="text-[11px] text-muted-foreground flex items-center gap-1.5 mb-2">
                  <Upload className="w-3 h-3 text-[#FF8000]" />
                  Reference Image (optional — generates 3D model)
                </span>
                <div
                  onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
                  onDragLeave={() => setIsDragOver(false)}
                  onDrop={onDrop}
                  onClick={() => fileInputRef.current?.click()}
                  className={`relative border-2 border-dashed rounded-lg p-4 text-center cursor-pointer transition-all ${
                    isDragOver
                      ? 'border-[#FF8000] bg-[#FF8000]/5'
                      : imagePreview
                        ? 'border-[rgba(255,128,0,0.2)] bg-[#0D1117]'
                        : 'border-[rgba(255,128,0,0.12)] hover:border-[rgba(255,128,0,0.3)] bg-[#0D1117]'
                  }`}
                >
                  {imagePreview ? (
                    <img src={imagePreview} alt="Preview" className="max-h-32 mx-auto rounded-lg object-contain" />
                  ) : (
                    <div className="space-y-1">
                      <Box className="w-6 h-6 mx-auto text-muted-foreground" />
                      <p className="text-[12px] text-muted-foreground">Drop an F1 car image or click to browse</p>
                      <p className="text-[11px] text-muted-foreground/50">PNG, JPG up to 10MB</p>
                    </div>
                  )}
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    aria-label="Upload reference image"
                    className="hidden"
                    onChange={(e) => { if (e.target.files?.[0]) handleFile(e.target.files[0]); }}
                  />
                </div>

                {imagePreview && (
                  <div className="flex gap-2 mt-2">
                    <button
                      type="button"
                      onClick={() => setGen3dProvider('hunyuan')}
                      className={`flex-1 flex items-center justify-center gap-1.5 px-2 py-1.5 rounded-lg text-[11px] border transition-all ${
                        gen3dProvider === 'hunyuan'
                          ? 'bg-[#3b82f6]/10 border-[#3b82f6]/30 text-[#3b82f6]'
                          : 'bg-transparent border-[rgba(255,128,0,0.12)] text-muted-foreground hover:text-foreground'
                      }`}
                    >
                      <Cpu className="w-3 h-3" /> Hunyuan3D (Free)
                    </button>
                    <button
                      type="button"
                      onClick={() => setGen3dProvider('meshy')}
                      className={`flex-1 flex items-center justify-center gap-1.5 px-2 py-1.5 rounded-lg text-[11px] border transition-all ${
                        gen3dProvider === 'meshy'
                          ? 'bg-[#FF8000]/10 border-[#FF8000]/30 text-[#FF8000]'
                          : 'bg-transparent border-[rgba(255,128,0,0.12)] text-muted-foreground hover:text-foreground'
                      }`}
                    >
                      <Sparkles className="w-3 h-3" /> Meshy.ai (Pro)
                    </button>
                  </div>
                )}
              </div>
            </div>

            {/* Footer */}
            <div className="flex items-center justify-end gap-2 px-5 py-3 border-t border-[rgba(255,128,0,0.12)]">
              <button
                type="button"
                onClick={() => { setShowRegister(false); setRegError(''); setImageFile(null); setImagePreview(null); }}
                className="px-4 py-2 text-[12px] text-muted-foreground hover:text-foreground rounded-lg transition-colors"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleRegister}
                disabled={submitting}
                className="flex items-center gap-1.5 px-4 py-2 bg-[#FF8000] hover:bg-[#FF8000]/80 disabled:opacity-50 text-black text-[12px] font-medium rounded-lg transition-colors"
              >
                <Save className="w-3.5 h-3.5" />
                {submitting ? 'Registering...' : 'Register'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
