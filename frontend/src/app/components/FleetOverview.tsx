import { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import {
  AlertTriangle, CheckCircle2, XCircle, Activity,
  CircleDot, Shield, TrendingUp, Cpu,
  Plus, X, Car, Save, Upload, Box, Loader2, Sparkles, Gauge,
} from 'lucide-react';
import {
  ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip,
} from 'recharts';
import * as model3dApi from '../api/model3d';
import type { Job } from '../api/model3d';
import {
  type HealthLevel, type VehicleData,
  mapLevel, levelColor, levelBg,
  MAINTENANCE_LABELS, SEVERITY_COLORS, parseAnomalyDrivers,
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


// 3D car models — served from public/models/ via symlinks
const DRIVER_CAR_MODEL: Record<number, { label: string; url: string }> = {
  4:  { label: 'MCL38', url: '/models/mcl38.glb' },   // NOR — 2024 car
  81: { label: 'MCL39', url: '/models/mcl39.glb' },    // PIA — 2025 car
};

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
export function FleetOverview() {
  const [vehicles, setVehicles] = useState<VehicleData[]>([]);
  const [selectedCar, setSelectedCar] = useState<VehicleData | null>(null);
  const [loading, setLoading] = useState(true);
  const [liveSource, setLiveSource] = useState<'cached' | 'live' | null>(null);

  // Registration state
  const [showRegister, setShowRegister] = useState(false);
  const [regForm, setRegForm] = useState(EMPTY_FORM);
  const [registeredVehicles, setRegisteredVehicles] = useState<RegisteredVehicle[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [regError, setRegError] = useState('');

  // Forecast state
  interface ForecastPoint { step: string; value: number; lower: number; upper: number }
  interface FeatureForecast { column: string; method: string; data: ForecastPoint[]; mae?: number; rmse?: number }
  const [forecasts, setForecasts] = useState<FeatureForecast[]>([]);
  const [forecastLoading, setForecastLoading] = useState(false);

  // Fetch forecasts for Critical/High SHAP features when driver selected
  useEffect(() => {
    if (!selectedCar) { setForecasts([]); return; }
    const criticalSystems = selectedCar.systems.filter(s => s.level === 'critical' || s.level === 'warning');
    const features = criticalSystems.flatMap(s => s.metrics.slice(0, 2).map(m => m.label));
    const unique = [...new Set(features)];
    if (unique.length === 0) { setForecasts([]); return; }

    let cancelled = false;
    setForecastLoading(true);
    Promise.all(
      unique.map(col =>
        fetch(`/api/omni/analytics/forecast/${selectedCar.code}?column=${encodeURIComponent(col)}&horizon=5`, { method: 'POST' })
          .then(r => r.ok ? r.json() : null)
          .catch(() => null)
      )
    ).then(results => {
      if (cancelled) return;
      const fcs: FeatureForecast[] = [];
      for (const r of results) {
        if (!r?.values) continue;
        fcs.push({
          column: r.column,
          method: r.method,
          mae: r.mae,
          rmse: r.rmse,
          data: r.values.map((v: number, i: number) => ({
            step: r.timestamps?.[i] ?? `+${i + 1}`,
            value: v,
            lower: r.lower_bound?.[i] ?? v,
            upper: r.upper_bound?.[i] ?? v,
          })),
        });
      }
      setForecasts(fcs);
      setForecastLoading(false);
    });
    return () => { cancelled = true; };
  }, [selectedCar?.code, selectedCar?.level]); // eslint-disable-line react-hooks/exhaustive-deps

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

  useEffect(() => {
    const load = async () => {
      try {
        const res = await fetch('/api/pipeline/anomaly');
        const data = await res.json();
        setVehicles(parseAnomalyDrivers(data));
      } catch (err) {
        console.error('Fleet anomaly data load error:', err);
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

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

  // Race-by-race trend from anomaly data
  const trendData = useMemo(() => {
    if (!selectedCar) return [];
    return selectedCar.races.map(r => {
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
        race: r.race,
        speedHealth: systems['Speed']?.health ?? 0,
        paceHealth: systems['Lap Pace']?.health ?? 0,
        tyreHealth: systems['Tyre Management']?.health ?? 0,
        speedST: speedFeats['SpeedST'] ?? 0,
        lapTime: paceFeats['LapTime'] ?? 0,
        tyreLife: tyreFeats['TyreLife'] ?? 0,
        maintenanceAction: topAction,
      };
    });
  }, [selectedCar]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Activity className="w-5 h-5 text-[#FF8000] animate-spin" />
        <span className="ml-2 text-sm text-muted-foreground">Loading anomaly detection data...</span>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Fleet status bar */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2 bg-[#1A1F2E] rounded-lg px-3 py-2 border border-[rgba(255,128,0,0.12)]">
          <CircleDot className="w-3 h-3 text-[#FF8000]" />
          <span className="text-[12px] text-muted-foreground">{vehicles.length + registeredVehicles.length} vehicles</span>
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

      {/* Driver cards — always visible, clickable to select */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {vehicles.map(v => {
          const isSelected = selectedCar?.number === v.number;
          return (
            <button
              key={v.number}
              onClick={() => setSelectedCar(isSelected ? null : v)}
              className={`text-left bg-[#1A1F2E] rounded-xl border p-5 transition-all group ${
                isSelected
                  ? 'border-[#FF8000] ring-1 ring-[#FF8000]/30 shadow-[0_0_20px_rgba(255,128,0,0.1)]'
                  : 'border-[rgba(255,128,0,0.12)] hover:border-[rgba(255,128,0,0.2)]'
              }`}
            >
              {/* Header */}
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div className={`w-10 h-10 rounded-lg flex items-center justify-center text-sm font-bold font-mono ${
                    isSelected ? 'bg-[#FF8000]/15 text-[#FF8000] border border-[#FF8000]/40' : ''
                  }`}
                    style={!isSelected ? { background: levelBg(v.level), color: levelColor(v.level), border: `1px solid ${levelColor(v.level)}22` } : undefined}>
                    #{v.number}
                  </div>
                  <div>
                    <div className={`text-sm font-medium transition-colors ${
                      isSelected ? 'text-[#FF8000]' : 'text-foreground group-hover:text-[#FF8000]'
                    }`}>{v.driver}</div>
                    <div className="text-[12px] text-muted-foreground">Last: {v.lastRace}</div>
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

              {/* Overall health bar */}
              <div className="h-2 bg-[#0D1117] rounded-full overflow-hidden mb-4">
                <div
                  className="h-full rounded-full transition-all duration-700"
                  style={{ width: `${v.overallHealth}%`, background: `linear-gradient(90deg, ${levelColor(v.level)}, ${levelColor(v.level)}88)` }}
                />
              </div>

              {/* System mini-bars */}
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

              {/* Selection indicator */}
              <div className={`mt-3 flex items-center justify-end gap-1 text-[11px] transition-colors ${
                isSelected
                  ? 'text-[#FF8000]/70'
                  : 'text-muted-foreground/50 group-hover:text-[#FF8000]/60'
              }`}>
                {isSelected ? 'Selected — click to collapse' : 'Click for details'}
              </div>
            </button>
          );
        })}
      </div>

      {/* ─── Detail Section — inline below cards when a driver is selected ─── */}
      {selectedCar && (
        <div className="space-y-4">
          {/* Section divider with driver name */}
          <div className="flex items-center gap-3 pt-2">
            <div className="h-px flex-1 bg-gradient-to-r from-[#FF8000]/30 to-transparent" />
            <div className="flex items-center gap-2">
              <div className="w-2.5 h-2.5 rounded-full" style={{ background: levelColor(selectedCar.level) }} />
              <span className="text-sm font-medium text-[#FF8000]">#{selectedCar.number} {selectedCar.driver}</span>
              <span className="text-sm font-mono" style={{ color: levelColor(selectedCar.level) }}>
                {selectedCar.overallHealth}% Overall
              </span>
            </div>
            <div className="h-px flex-1 bg-gradient-to-l from-[#FF8000]/30 to-transparent" />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {/* Stress Heatmap + 3D Model — takes 2/3 */}
            <div className="lg:col-span-2 space-y-4">
              <div className="bg-[#1A1F2E] rounded-xl border border-[rgba(255,128,0,0.12)] p-4">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-sm font-medium text-foreground flex items-center gap-2">
                    <Shield className="w-3.5 h-3.5 text-[#FF8000]" />
                    Vehicle Stress Heatmap
                  </h3>
                </div>

                {/* 3D model with stress hotspots rendered in Three.js */}
                <div className="rounded-lg overflow-hidden border border-[rgba(255,128,0,0.12)]">
                  {(() => {
                    const carModel = DRIVER_CAR_MODEL[selectedCar.number] ?? DRIVER_CAR_MODEL[4];
                    return (
                      <iframe
                        src={`/glb_viewer.html?url=${carModel.url}&title=${encodeURIComponent(carModel.label)}&stress=${encodeURIComponent(JSON.stringify(selectedCar.systems.map(s => ({ name: s.name, health: s.health, level: s.level, metrics: s.metrics }))))}`}
                        className="w-full h-[500px] border-0 bg-[#0D1117]"
                        title={`3D — ${carModel.label}`}
                      />
                    );
                  })()}
                </div>

                <div className="flex justify-center gap-4 mt-3">
                  {(['nominal', 'warning', 'critical'] as HealthLevel[]).map(l => (
                    <div key={l} className="flex items-center gap-1.5">
                      <div className="w-2 h-2 rounded-full" style={{ background: levelColor(l) }} />
                      <span className="text-[11px] text-muted-foreground capitalize">{l}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* System Summary — compact 1/3 sidebar */}
            <div className="lg:col-span-1 bg-[#1A1F2E] rounded-xl border border-[rgba(255,128,0,0.12)] p-3">
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
                        <div
                          className="h-full rounded-full transition-all duration-700"
                          style={{ width: `${sys.health}%`, background: levelColor(sys.level) }}
                        />
                      </div>
                      <SeverityBar probabilities={sys.severityProbabilities} />
                      {sys.maintenanceAction && sys.maintenanceAction !== 'none' && (
                        <div className="mt-1.5">
                          <MaintenanceBadge action={sys.maintenanceAction} />
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Race-by-race anomaly trend — compact */}
          {trendData.length > 0 && (
            <div className="bg-[#1A1F2E] rounded-xl border border-[rgba(255,128,0,0.12)] p-3">
              <h3 className="text-[12px] font-medium text-foreground mb-2 flex items-center gap-1.5">
                <TrendingUp className="w-3 h-3 text-[#FF8000]" />
                Season Health Trend
              </h3>
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
                    {trendData.map((d, i) => {
                      const healthColor = (h: number) => h >= 75 ? '#22c55e' : h >= 50 ? '#FF8000' : '#ef4444';
                      return (
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
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Feature Forecasts — from Critical/High SHAP features */}
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
                {forecasts.map(fc => (
                  <div key={fc.column} className="bg-[#0D1117] rounded-lg p-3 border border-[rgba(255,128,0,0.08)]">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-[11px] font-medium text-foreground">{fc.column}</span>
                      <div className="flex items-center gap-2">
                        <span className="text-[9px] text-muted-foreground font-mono uppercase">{fc.method}</span>
                        {fc.rmse != null && (
                          <span className="text-[9px] text-muted-foreground font-mono">RMSE {fc.rmse.toFixed(2)}</span>
                        )}
                      </div>
                    </div>
                    <ResponsiveContainer width="100%" height={120}>
                      <AreaChart data={fc.data} margin={{ top: 4, right: 4, bottom: 0, left: 4 }}>
                        <defs>
                          <linearGradient id={`fc-${fc.column}`} x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#FF8000" stopOpacity={0.3} />
                            <stop offset="100%" stopColor="#FF8000" stopOpacity={0} />
                          </linearGradient>
                        </defs>
                        <XAxis dataKey="step" tick={{ fontSize: 9, fill: '#6b7280' }} axisLine={false} tickLine={false} />
                        <YAxis tick={{ fontSize: 9, fill: '#6b7280' }} axisLine={false} tickLine={false} width={40} />
                        <Tooltip
                          contentStyle={{ background: '#1A1F2E', border: '1px solid rgba(255,128,0,0.2)', borderRadius: 8, fontSize: 11 }}
                          labelStyle={{ color: '#9ca3af' }}
                        />
                        <Area type="monotone" dataKey="upper" stroke="none" fill="#FF8000" fillOpacity={0.08} name="Upper bound" />
                        <Area type="monotone" dataKey="lower" stroke="none" fill="#0D1117" fillOpacity={1} name="Lower bound" />
                        <Area type="monotone" dataKey="value" stroke="#FF8000" fill={`url(#fc-${fc.column})`} strokeWidth={1.5} dot={false} name="Forecast" />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                ))}
              </div>
            </div>
          )}

        </div>
      )}

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
