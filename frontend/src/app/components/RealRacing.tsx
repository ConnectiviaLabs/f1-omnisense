import { useState, useEffect, useMemo } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer,
} from 'recharts';
import {
  Activity, Timer, Gauge, Thermometer, Zap, AlertTriangle,
  ChevronDown, ChevronUp, Upload, Loader2, Car, Fuel,
  Navigation, BarChart3,
} from 'lucide-react';
import { aim } from '../api/local';
import { HealthGauge } from './HealthGauge';
import type {
  AiMSession, AiMSessionSummary, AiMHealthResult,
  AiMSystemAnomaly,
} from '../types';

// ── Theme colors ────────────────────────────────────────────────────────

const PAPAYA = '#FF8000';
const CYAN = '#47C7FC';
const LIME = '#22C55E';
const RED = '#EF4444';
const AMBER = '#F59E0B';
const PURPLE = '#A855F7';
const PINK = '#EC4899';
const TEAL = '#14B8A6';

const SEVERITY_COLOR: Record<string, string> = {
  NORMAL: LIME,
  LOW: CYAN,
  MEDIUM: AMBER,
  HIGH: PAPAYA,
  CRITICAL: RED,
};

// ── Chart channel groups ────────────────────────────────────────────────

interface ChartGroup {
  id: string;
  label: string;
  icon: React.ElementType;
  channels: { key: string; color: string; label: string }[];
  yDomain?: [number, number];
}

const CHART_GROUPS: ChartGroup[] = [
  {
    id: 'speed-throttle',
    label: 'Speed & Throttle',
    icon: Gauge,
    channels: [
      { key: 'GPS_Speed_kmh', color: PAPAYA, label: 'Speed (km/h)' },
      { key: 'Throttle_pct', color: LIME, label: 'Throttle (%)' },
    ],
  },
  {
    id: 'brakes',
    label: 'Brakes',
    icon: AlertTriangle,
    channels: [
      { key: 'FrontBrake_bar', color: RED, label: 'Front Brake (bar)' },
      { key: 'RearBrake_bar', color: AMBER, label: 'Rear Brake (bar)' },
    ],
  },
  {
    id: 'thermal',
    label: 'Thermal',
    icon: Thermometer,
    channels: [
      { key: 'OilTemp_C', color: PAPAYA, label: 'Oil Temp (°C)' },
      { key: 'WaterTemp_C', color: CYAN, label: 'Water Temp (°C)' },
      { key: 'OilPressure_bar', color: LIME, label: 'Oil Pressure (bar)' },
    ],
  },
  {
    id: 'imu',
    label: 'IMU / G-Forces',
    icon: Activity,
    channels: [
      { key: 'LateralAcc_g', color: CYAN, label: 'Lateral G' },
      { key: 'InlineAcc_g', color: PAPAYA, label: 'Inline G' },
      { key: 'VerticalAcc_g', color: LIME, label: 'Vertical G' },
    ],
  },
  {
    id: 'dynamics',
    label: 'Rotational Dynamics',
    icon: Navigation,
    channels: [
      { key: 'YawRate_dps', color: PURPLE, label: 'Yaw (°/s)' },
      { key: 'RollRate_dps', color: PINK, label: 'Roll (°/s)' },
      { key: 'PitchRate_dps', color: TEAL, label: 'Pitch (°/s)' },
    ],
  },
  {
    id: 'engine',
    label: 'Engine',
    icon: Fuel,
    channels: [
      { key: 'Lambda', color: LIME, label: 'Lambda' },
      { key: 'AFR', color: PAPAYA, label: 'AFR' },
      { key: 'TPS_raw', color: CYAN, label: 'TPS Raw' },
    ],
  },
  {
    id: 'timing',
    label: 'Timing Deltas',
    icon: Timer,
    channels: [
      { key: 'PredictiveTime', color: PAPAYA, label: 'Predictive' },
      { key: 'BestRunDiff', color: LIME, label: 'Best Run Δ' },
      { key: 'BestTodayDiff', color: CYAN, label: 'Best Today Δ' },
      { key: 'PrevLapDiff', color: AMBER, label: 'Prev Lap Δ' },
    ],
  },
  {
    id: 'track-3d',
    label: 'Track Position',
    icon: Car,
    channels: [
      { key: 'X_m', color: PAPAYA, label: 'X (m)' },
      { key: 'Y_m', color: CYAN, label: 'Y (m)' },
      { key: 'Z_m', color: LIME, label: 'Z (m)' },
      { key: 'Distance_m', color: PURPLE, label: 'Distance (m)' },
    ],
  },
];

// ── KPI Card ────────────────────────────────────────────────────────────

function KPI({ icon: Icon, label, value, detail, color }: {
  icon: React.ElementType; label: string; value: string; detail?: string; color?: string;
}) {
  return (
    <div className="bg-card border border-border rounded-xl px-4 py-3 flex items-center gap-3 min-w-[160px]">
      <div className="p-2 rounded-lg bg-secondary">
        <Icon className="w-4 h-4" style={{ color: color || PAPAYA }} />
      </div>
      <div className="min-w-0">
        <div className="text-[10px] text-muted-foreground tracking-wider uppercase">{label}</div>
        <div className="text-foreground text-lg font-semibold leading-tight">{value}</div>
        {detail && <div className="text-[11px] text-muted-foreground">{detail}</div>}
      </div>
    </div>
  );
}

// ── Collapsible Chart Card ──────────────────────────────────────────────

function ChartCard({ group, data }: { group: ChartGroup; data: any[] }) {
  const [open, setOpen] = useState(true);
  const Icon = group.icon;

  // Check which channels actually have data
  const activeChannels = useMemo(() =>
    group.channels.filter(ch => data.some(d => d[ch.key] != null)),
    [group.channels, data],
  );

  if (activeChannels.length === 0) return null;

  return (
    <div className="bg-card border border-border rounded-xl overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-2 px-4 py-2.5 hover:bg-secondary/50 transition-colors"
      >
        <Icon className="w-4 h-4 text-primary" />
        <span className="text-sm font-medium text-foreground flex-1 text-left">{group.label}</span>
        <div className="flex gap-1.5">
          {activeChannels.map(ch => (
            <span key={ch.key} className="w-2.5 h-2.5 rounded-full" style={{ background: ch.color }} />
          ))}
        </div>
        {open ? <ChevronUp className="w-3.5 h-3.5 text-muted-foreground" /> : <ChevronDown className="w-3.5 h-3.5 text-muted-foreground" />}
      </button>
      {open && (
        <div className="px-2 pb-3">
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={data} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis
                dataKey="time_s"
                tick={{ fontSize: 10, fill: 'var(--muted-foreground)' }}
                tickFormatter={(v: number) => `${v.toFixed(0)}s`}
              />
              <YAxis tick={{ fontSize: 10, fill: 'var(--muted-foreground)' }} width={50} domain={group.yDomain} />
              <Tooltip
                contentStyle={{ background: 'var(--card)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 11 }}
                labelFormatter={(v: number) => `${Number(v).toFixed(2)}s`}
              />
              {activeChannels.map(ch => (
                <Line
                  key={ch.key}
                  type="monotone"
                  dataKey={ch.key}
                  stroke={ch.color}
                  dot={false}
                  strokeWidth={1.5}
                  name={ch.label}
                  connectNulls
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
          <div className="flex gap-3 px-3 mt-1">
            {activeChannels.map(ch => (
              <div key={ch.key} className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
                <span className="w-2 h-2 rounded-full" style={{ background: ch.color }} />
                {ch.label}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Track Map (XY scatter) ──────────────────────────────────────────────

function TrackMap({ trackData }: { trackData: any[] }) {
  if (!trackData.length) return null;

  // Get XY bounds for SVG viewBox
  const xs = trackData.map(d => d.X_m ?? d.GPS_Lon).filter(Boolean);
  const ys = trackData.map(d => d.Y_m ?? d.GPS_Lat).filter(Boolean);
  if (!xs.length || !ys.length) return null;

  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const rangeX = maxX - minX || 1, rangeY = maxY - minY || 1;
  const pad = 20;
  const w = 500, h = 400;

  const points = trackData.map(d => {
    const x = pad + ((d.X_m ?? d.GPS_Lon) - minX) / rangeX * (w - 2 * pad);
    const y = pad + (1 - ((d.Y_m ?? d.GPS_Lat) - minY) / rangeY) * (h - 2 * pad);
    const speed = d.GPS_Speed_kmh ?? d.Track_Speed_kmh ?? 0;
    return { x, y, speed };
  });

  const maxSpeed = Math.max(...points.map(p => p.speed), 1);

  return (
    <div className="bg-card border border-border rounded-xl p-4">
      <div className="flex items-center gap-2 mb-3">
        <Navigation className="w-4 h-4 text-primary" />
        <span className="text-sm font-medium text-foreground">Track Map</span>
        <span className="text-[10px] text-muted-foreground ml-auto">Speed-colored</span>
      </div>
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full" style={{ background: 'var(--background)', borderRadius: 8 }}>
        {points.map((p, i) => {
          if (i === 0) return null;
          const prev = points[i - 1];
          const ratio = p.speed / maxSpeed;
          const r = Math.round(255 * ratio);
          const g = Math.round(255 * (1 - ratio * 0.5));
          const b = Math.round(80 * (1 - ratio));
          return (
            <line
              key={i}
              x1={prev.x} y1={prev.y}
              x2={p.x} y2={p.y}
              stroke={`rgb(${r},${g},${b})`}
              strokeWidth={2.5}
              strokeLinecap="round"
            />
          );
        })}
        {/* Start marker */}
        {points.length > 0 && (
          <circle cx={points[0].x} cy={points[0].y} r={5} fill={LIME} stroke="white" strokeWidth={1.5} />
        )}
      </svg>
      <div className="flex items-center gap-2 mt-2 text-[10px] text-muted-foreground">
        <span>Slow</span>
        <div className="flex-1 h-2 rounded-full" style={{
          background: 'linear-gradient(to right, rgb(0,255,80), rgb(255,200,0), rgb(255,128,0), rgb(255,0,0))'
        }} />
        <span>Fast</span>
      </div>
    </div>
  );
}

// ── Lap Time Bar Chart ──────────────────────────────────────────────────

function LapTimesChart({ laps }: { laps: { lap_number: number; lap_time_s: number }[] }) {
  if (!laps.length) return null;
  const bestLap = Math.min(...laps.map(l => l.lap_time_s));

  return (
    <div className="bg-card border border-border rounded-xl p-4">
      <div className="flex items-center gap-2 mb-3">
        <Timer className="w-4 h-4 text-primary" />
        <span className="text-sm font-medium text-foreground">Lap Times</span>
      </div>
      <div className="space-y-1.5">
        {laps.map(l => {
          const isBest = l.lap_time_s === bestLap;
          const pct = (bestLap / l.lap_time_s) * 100;
          return (
            <div key={l.lap_number} className="flex items-center gap-2">
              <span className={`text-[11px] w-10 text-right font-mono ${isBest ? 'text-primary font-bold' : 'text-muted-foreground'}`}>
                L{l.lap_number}
              </span>
              <div className="flex-1 h-5 bg-secondary rounded-md overflow-hidden relative">
                <div
                  className="h-full rounded-md transition-all"
                  style={{ width: `${pct}%`, background: isBest ? PAPAYA : 'var(--muted-foreground)', opacity: isBest ? 1 : 0.4 }}
                />
                <span className={`absolute right-2 top-0.5 text-[10px] font-mono ${isBest ? 'text-primary font-bold' : 'text-muted-foreground'}`}>
                  {l.lap_time_s.toFixed(3)}s
                </span>
              </div>
              {isBest && <span className="text-[9px] text-primary font-bold">BEST</span>}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Health Dashboard ────────────────────────────────────────────────────

function HealthDashboard({ health, anomaly }: { health: AiMHealthResult | null; anomaly: any }) {
  if (!health) return null;

  return (
    <div className="bg-card border border-border rounded-xl p-4">
      <div className="flex items-center gap-2 mb-4">
        <Activity className="w-4 h-4 text-primary" />
        <span className="text-sm font-medium text-foreground">7-System Health Assessment</span>
        <span className={`ml-auto text-xs font-medium px-2 py-0.5 rounded-full`}
          style={{ background: SEVERITY_COLOR[health.severity] + '20', color: SEVERITY_COLOR[health.severity] }}>
          {health.severity}
        </span>
      </div>

      {/* Overall gauge */}
      <div className="flex justify-center mb-4">
        <HealthGauge value={health.overall_health} size={100} label="Overall" />
      </div>

      {/* Per-system gauges */}
      <div className="grid grid-cols-4 lg:grid-cols-7 gap-3">
        {health.systems.map(s => (
          <div key={s.system} className="flex flex-col items-center gap-1">
            <HealthGauge value={s.health_pct} size={70} label={s.system} strokeWidth={6} />
            <span className="text-[9px] text-center text-muted-foreground leading-tight">{s.system}</span>
            <span className="text-[8px] font-mono px-1.5 rounded"
              style={{ color: SEVERITY_COLOR[s.severity] }}>
              {s.severity}
            </span>
          </div>
        ))}
      </div>

      {/* Anomaly timeline per system */}
      {anomaly?.systems && (
        <div className="mt-4 space-y-2">
          <div className="text-[10px] text-muted-foreground tracking-wider uppercase">Per-Lap Anomaly Timeline</div>
          {anomaly.systems
            .filter((s: AiMSystemAnomaly) => s.anomaly_scores?.length > 0)
            .map((s: AiMSystemAnomaly) => (
              <div key={s.system} className="flex items-center gap-2">
                <span className="text-[10px] text-muted-foreground w-24 truncate">{s.system}</span>
                <div className="flex-1 flex gap-0.5">
                  {s.anomaly_scores.map(sc => (
                    <div
                      key={sc.lap}
                      className="flex-1 h-4 rounded-sm"
                      style={{ background: SEVERITY_COLOR[sc.severity] || LIME, opacity: 0.3 + sc.score * 0.7 }}
                      title={`Lap ${sc.lap}: ${sc.severity} (${sc.score.toFixed(3)})`}
                    />
                  ))}
                </div>
                <span className="text-[10px] text-muted-foreground w-12 text-right">{s.health_pct.toFixed(0)}%</span>
              </div>
            ))}
        </div>
      )}
    </div>
  );
}

// ── Main Component ──────────────────────────────────────────────────────

export function RealRacing() {
  // ── State ──
  const [sessions, setSessions] = useState<AiMSessionSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [session, setSession] = useState<AiMSession | null>(null);
  const [telemetry, setTelemetry] = useState<any[]>([]);
  const [trackData, setTrackData] = useState<any[]>([]);
  const [health, setHealth] = useState<AiMHealthResult | null>(null);
  const [anomaly, setAnomaly] = useState<any>(null);
  const [activeLap, setActiveLap] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [telLoading, setTelLoading] = useState(false);
  const [uploading, setUploading] = useState(false);

  // ── Load sessions on mount ──
  useEffect(() => {
    aim.sessions()
      .then(data => {
        setSessions(data || []);
        if (data?.length > 0) setSelectedId(data[0].session_id);
      })
      .catch(err => console.error('Failed to load AiM sessions:', err))
      .finally(() => setLoading(false));
  }, []);

  // ── Load session data when selection changes ──
  useEffect(() => {
    if (!selectedId) return;
    setTelLoading(true);

    Promise.all([
      aim.session(selectedId),
      aim.telemetry(selectedId, activeLap ?? undefined),
      aim.track(selectedId),  // always fetch all laps for overlay
      aim.health(selectedId).catch(() => null),
      aim.anomaly(selectedId).catch(() => null),
    ]).then(([sess, tel, trk, hlth, anom]) => {
      setSession(sess);
      setTelemetry(tel?.data || []);
      setTrackData(trk?.track_xy || trk?.gps || []);
      setHealth(hlth);
      setAnomaly(anom);
    }).catch(err => {
      console.error('Failed to load session data:', err);
    }).finally(() => {
      setTelLoading(false);
    });
  }, [selectedId, activeLap]);

  // ── Upload handler ──
  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    try {
      const result = await aim.upload(file);
      // Refresh sessions
      const data = await aim.sessions();
      setSessions(data || []);
      setSelectedId(result.session_id);
    } catch (err) {
      console.error('Upload failed:', err);
    } finally {
      setUploading(false);
    }
  };

  // ── KPIs ──
  const kpis = useMemo(() => {
    if (!session?.summary) return [];
    const s = session.summary;
    return [
      { icon: Timer, label: 'Best Lap', value: s.best_lap_time_s ? `${s.best_lap_time_s.toFixed(3)}s` : '—', detail: s.best_lap_number ? `Lap ${s.best_lap_number}` : '', color: PAPAYA },
      { icon: Gauge, label: 'Top Speed', value: s.max_GPS_Speed_kmh ? `${s.max_GPS_Speed_kmh.toFixed(1)}` : '—', detail: 'km/h', color: CYAN },
      { icon: Activity, label: 'Max Lateral G', value: s.max_LateralAcc_g ? `${s.max_LateralAcc_g.toFixed(2)}g` : '—', color: PURPLE },
      { icon: AlertTriangle, label: 'Max Brake', value: s.max_FrontBrake_bar ? `${s.max_FrontBrake_bar.toFixed(1)}` : '—', detail: 'bar', color: RED },
      { icon: Thermometer, label: 'Avg Oil Temp', value: s.avg_OilTemp_C ? `${s.avg_OilTemp_C.toFixed(1)}°C` : '—', color: AMBER },
      { icon: Zap, label: 'Alerts', value: String(s.alert_count ?? 0), color: s.alert_count > 0 ? RED : LIME },
    ];
  }, [session]);

  // ── Loading state ──
  if (loading) {
    return (
      <div className="flex items-center justify-center h-96 gap-3 text-muted-foreground">
        <Loader2 className="w-5 h-5 animate-spin" />
        <span className="text-sm">Loading AiM sessions...</span>
      </div>
    );
  }

  return (
    <div className="space-y-4 pt-4">
      {/* ── Section 1: Session Bar ── */}
      <div className="flex items-center gap-3 flex-wrap">
        {/* Session selector */}
        <div className="flex items-center gap-2">
          <Car className="w-4 h-4 text-primary" />
          <select
            value={selectedId || ''}
            onChange={e => { setSelectedId(e.target.value); setActiveLap(null); }}
            className="bg-secondary text-foreground text-sm rounded-lg px-3 py-1.5 border border-border focus:outline-none focus:ring-1 focus:ring-primary"
          >
            {sessions.length === 0 && <option value="">No sessions</option>}
            {sessions.map(s => (
              <option key={s.session_id} value={s.session_id}>
                {s.driver} — {s.track} ({s.date})
              </option>
            ))}
          </select>
        </div>

        {/* Lap pills */}
        {session?.laps && (
          <div className="flex items-center gap-1.5">
            <button
              onClick={() => setActiveLap(null)}
              className={`text-[11px] px-2.5 py-1 rounded-full border transition-colors ${
                activeLap === null
                  ? 'bg-primary text-background border-primary font-medium'
                  : 'border-border text-muted-foreground hover:text-foreground hover:border-foreground/30'
              }`}
            >
              All
            </button>
            {session.laps.map(l => (
              <button
                key={l.lap_number}
                onClick={() => setActiveLap(l.lap_number)}
                className={`text-[11px] px-2.5 py-1 rounded-full border transition-colors ${
                  activeLap === l.lap_number
                    ? 'bg-primary text-background border-primary font-medium'
                    : 'border-border text-muted-foreground hover:text-foreground hover:border-foreground/30'
                }`}
              >
                L{l.lap_number}
              </button>
            ))}
          </div>
        )}

        {/* Upload */}
        <label className="ml-auto flex items-center gap-1.5 text-[11px] text-muted-foreground hover:text-foreground bg-secondary rounded-lg px-3 py-1.5 border border-border cursor-pointer transition-colors">
          {uploading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Upload className="w-3.5 h-3.5" />}
          <span>Upload .xrk</span>
          <input type="file" accept=".xrk" onChange={handleUpload} className="hidden" />
        </label>
      </div>

      {/* Session meta bar */}
      {session && (
        <div className="flex items-center gap-4 text-[11px] text-muted-foreground bg-card border border-border rounded-lg px-4 py-2">
          <span><span className="text-foreground font-medium">{session.driver}</span> — {session.vehicle}</span>
          <span>{session.track}</span>
          <span>{session.date} {session.time}</span>
          <span>ECU: {session.ecu}</span>
          <span>{session.lap_count} laps · {session.duration_s.toFixed(1)}s</span>
        </div>
      )}

      {/* ── Section 2: KPI Row ── */}
      {kpis.length > 0 && (
        <div className="flex gap-3 overflow-x-auto pb-1">
          {kpis.map(k => <KPI key={k.label} {...k} />)}
        </div>
      )}

      {/* Loading overlay for telemetry */}
      {telLoading && (
        <div className="flex items-center gap-2 text-muted-foreground text-sm py-2">
          <Loader2 className="w-4 h-4 animate-spin" />
          Loading telemetry...
        </div>
      )}

      {/* ── Section 3+4: Track Map + Telemetry Charts ── */}
      {!telLoading && telemetry.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Left: Track Map + Lap Times */}
          <div className="space-y-4">
            <TrackMap trackData={trackData} />
            {session?.laps && <LapTimesChart laps={session.laps} />}
          </div>

          {/* Right: Telemetry Charts */}
          <div className="space-y-3">
            {CHART_GROUPS.map(group => (
              <ChartCard key={group.id} group={group} data={telemetry} />
            ))}
          </div>
        </div>
      )}

      {/* ── Section 5: Health Dashboard ── */}
      {!telLoading && <HealthDashboard health={health} anomaly={anomaly} />}

      {/* Empty state */}
      {!loading && sessions.length === 0 && (
        <div className="flex flex-col items-center justify-center h-64 gap-3 text-muted-foreground">
          <BarChart3 className="w-10 h-10 text-primary/30" />
          <span className="text-sm">No AiM sessions found</span>
          <span className="text-[11px]">Upload an .xrk file or run the ingestion pipeline</span>
        </div>
      )}
    </div>
  );
}
