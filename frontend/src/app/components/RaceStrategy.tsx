import { useState, useEffect, useMemo } from 'react';
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, ReferenceLine,
} from 'recharts';
import {
  Loader2, Flag, Timer, Users, TrendingUp, AlertTriangle, Disc, Gauge, Zap,
  ChevronDown, ChevronUp, Shield, Activity, Brain, Cpu, Play, ClipboardList,
} from 'lucide-react';
import { strategy } from '../api/local';
import { getOpponentDrivers } from '../api/driverIntel';
import { COMPOUND_COLORS as compoundColors } from '../constants/teams';

// ─── Prep Mode Types ─────────────────────────────────────────────────

interface PrepSystemHealth { health: number; level: string; }
interface PrepAnomalyData { overall_health: number; overall_level: string; last_race: string; race_count: number; systems: Record<string, PrepSystemHealth>; }
interface PrepEltData { baseline_pace_s: number; driver_advantage_s: number; predicted_pace_s: number; }
interface PrepDegCurve { compound: string; temp_band: string; coefficients: number[]; deg_per_lap_s: number | null; r_squared: number | null; n_stints: number | null; }
interface PrepCircuitData { pit_loss_s?: number; air_density?: { density_kg_m3: number; temperature_c: number; humidity_pct: number; pressure_hpa: number }; intelligence?: Record<string, unknown>; }
interface PrepReport {
  report_type: string; race_name: string; season: number; team: string; drivers: string[];
  generated_at: string; generation_time_s: number; briefing: string; model_used: string;
  pillars: { anomaly: Record<string, PrepAnomalyData>; elt_pace: Record<string, PrepEltData>; degradation_curves: PrepDegCurve[]; strategy_simulation: Record<string, unknown> | null; circuit: PrepCircuitData };
}

const RACES_2024 = [
  'Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 'Australian Grand Prix',
  'Japanese Grand Prix', 'Chinese Grand Prix', 'Miami Grand Prix',
  'Emilia Romagna Grand Prix', 'Monaco Grand Prix', 'Canadian Grand Prix',
  'Spanish Grand Prix', 'Austrian Grand Prix', 'British Grand Prix',
  'Hungarian Grand Prix', 'Belgian Grand Prix', 'Dutch Grand Prix',
  'Italian Grand Prix', 'Azerbaijan Grand Prix', 'Singapore Grand Prix',
  'United States Grand Prix', 'Mexico City Grand Prix', 'São Paulo Grand Prix',
  'Las Vegas Grand Prix', 'Qatar Grand Prix', 'Abu Dhabi Grand Prix',
];

const healthColor = (h: number) => h >= 80 ? '#05DF72' : h >= 60 ? '#f59e0b' : '#ef4444';
const levelBg = (level: string) => {
  switch (level) {
    case 'normal': return 'bg-green-500/10 text-green-400 border-green-500/20';
    case 'medium': return 'bg-amber-500/10 text-amber-400 border-amber-500/20';
    case 'high': return 'bg-red-500/10 text-red-400 border-red-500/20';
    case 'critical': return 'bg-red-500/20 text-red-400 border-red-500/30';
    default: return 'bg-muted text-muted-foreground border-border';
  }
};

const compoundColor: Record<string, string> = {
  SOFT: '#ef4444', MEDIUM: '#f59e0b', HARD: '#888', INTERMEDIATE: '#22c55e', WET: '#3b82f6',
};

// ─── Constants ──────────────────────────────────────────────────────

const compoundTextColors: Record<string, string> = {
  SOFT: '#ef4444', MEDIUM: '#f59e0b', HARD: '#d4d4d8',
  INTERMEDIATE: '#22c55e', WET: '#3b82f6', UNKNOWN: '#6b7280',
};

// ─── Shared helpers ─────────────────────────────────────────────────
const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-background border border-[rgba(255,128,0,0.2)] rounded-lg p-2 text-[12px]">
      <div className="text-muted-foreground mb-1">{label}</div>
      {payload.map((e: any, i: number) => (
        <div key={i} className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full" style={{ backgroundColor: e.color }} />
          <span className="text-muted-foreground">{e.name}:</span>
          <span className="text-foreground font-mono">{typeof e.value === 'number' ? e.value.toFixed(2) : e.value}</span>
        </div>
      ))}
    </div>
  );
};

function Divider({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-3 pt-2">
      <div className="h-px flex-1 bg-[rgba(255,128,0,0.10)]" />
      <span className="text-[10px] tracking-[0.25em] text-primary/60 font-semibold">{label}</span>
      <div className="h-px flex-1 bg-[rgba(255,128,0,0.10)]" />
    </div>
  );
}

function KPI({ icon, label, value, detail }: { icon: React.ReactNode; label: string; value: string; detail: string }) {
  return (
    <div className="bg-card border border-border rounded-lg p-3">
      <div className="flex items-center gap-2 mb-1.5">
        {icon}
        <span className="text-[12px] text-muted-foreground tracking-wider">{label}</span>
      </div>
      <div className="text-lg font-mono text-foreground">{value}</div>
      <div className="text-[12px] text-muted-foreground mt-0.5">{detail}</div>
    </div>
  );
}

function isMcLaren(team: string) {
  return team?.toLowerCase().includes('mclaren');
}

// ─── Component ──────────────────────────────────────────────────────
export function RaceStrategy() {
  const [loading, setLoading] = useState(true);
  const [trackerData, setTrackerData] = useState<any>(null);
  const [simData, setSimData] = useState<any>(null);
  const [degData, setDegData] = useState<any>(null);
  const [eltData, setEltData] = useState<any>(null);
  const [scData, setScData] = useState<any>(null);
  const [battleData, setBattleData] = useState<any>(null);
  const [opponentData, setOpponentData] = useState<any>(null);
  const [modelHealth, setModelHealth] = useState<any>(null);
  const [xgbData, setXgbData] = useState<any>(null);
  const [bilstmData, setBilstmData] = useState<any>(null);
  const [expandedDriver, setExpandedDriver] = useState<string | null>(null);

  // Lap prediction form
  const [predCircuit, setPredCircuit] = useState('Bahrain Grand Prix');
  const [predDriver, setPredDriver] = useState('NOR');
  const [predCompound, setPredCompound] = useState('MEDIUM');
  const [predLapStart, setPredLapStart] = useState(1);
  const [predLapEnd, setPredLapEnd] = useState(25);
  const [predResult, setPredResult] = useState<any>(null);
  const [bilstmResult, setBilstmResult] = useState<any>(null);
  const [predModel, setPredModel] = useState<'xgboost' | 'bilstm' | 'both'>('both');
  const [predLoading, setPredLoading] = useState(false);

  // Prep Mode state
  const [prepExpanded, setPrepExpanded] = useState(false);
  const [prepRace, setPrepRace] = useState('Austrian Grand Prix');
  const [prepReport, setPrepReport] = useState<PrepReport | null>(null);
  const [prepLoading, setPrepLoading] = useState(false);
  const [prepError, setPrepError] = useState<string | null>(null);

  // Load latest prep report on mount
  useEffect(() => {
    fetch('/api/advantage/prep-mode/latest?season=2024&team=McLaren')
      .then(r => r.ok ? r.json() : null)
      .then(data => { if (data) { setPrepReport(data); setPrepRace(data.race_name); } })
      .catch(() => {});
  }, []);

  const generatePrep = async () => {
    setPrepLoading(true);
    setPrepError(null);
    try {
      const res = await fetch('/api/advantage/prep-mode/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ race_name: prepRace, season: 2024, team: 'McLaren', force: true }),
      });
      if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
      setPrepReport(await res.json());
    } catch (e: unknown) {
      setPrepError(e instanceof Error ? e.message : 'Generation failed');
    } finally {
      setPrepLoading(false);
    }
  };

  useEffect(() => {
    setLoading(true);
    Promise.all([
      strategy.tracker().catch(() => null),
      strategy.simulations().catch(() => null),
      strategy.degradation().catch(() => null),
      strategy.elt().catch(() => null),
      strategy.scProbability().catch(() => null),
      strategy.battleIntel().catch(() => null),
      getOpponentDrivers().catch(() => null),
      fetch('/api/omni/dapt/health').then(r => r.json()).catch(() => null),
      strategy.xgboost().catch(() => null),
      strategy.bilstm().catch(() => null),
    ]).then(([tracker, sims, deg, elt, sc, battle, opponents, health, xgb, bilstm]) => {
      setTrackerData(tracker);
      setSimData(sims);
      setDegData(deg);
      setEltData(elt);
      setScData(sc);
      setBattleData(battle);
      setOpponentData(opponents);
      setModelHealth(health);
      setXgbData(xgb);
      setBilstmData(bilstm);
      setLoading(false);
    });
  }, []);

  const session = trackerData?.session;
  const drivers = trackerData?.drivers || [];
  const cliffLaps = trackerData?.cliff_laps || {};
  const optimalStrategies = simData?.simulations?.[0]?.optimal_strategies || [];
  const simMeta = simData?.simulations?.[0];

  // Degradation curve data for chart
  const degCurveData = useMemo(() => {
    if (!degData?.curves?.length) return [];
    const compounds = ['SOFT', 'MEDIUM', 'HARD'];
    const maxLap = 40;
    const points: any[] = [];

    for (let lap = 1; lap <= maxLap; lap++) {
      const point: any = { lap };
      for (const compound of compounds) {
        const curve = degData.curves.find(
          (c: any) => c.compound === compound && (c.temp_band === 'all' || !c.temp_band)
        );
        if (curve?.coefficients) {
          const coeffs = [...curve.coefficients].reverse();
          const intercept = curve.intercept || 0;
          let val = intercept;
          for (let i = 0; i < coeffs.length; i++) {
            val += coeffs[i] * Math.pow(lap, i + 1);
          }
          point[compound] = Math.max(0, parseFloat(val.toFixed(3)));
        }
      }
      points.push(point);
    }
    return points;
  }, [degData]);

  // SC rates sorted for chart
  const scRateData = useMemo(() => {
    if (!scData?.circuit_sc_rates) return [];
    return Object.entries(scData.circuit_sc_rates)
      .map(([circuit, rate]) => ({ circuit, rate: rate as number }))
      .sort((a, b) => b.rate - a.rate)
      .slice(0, 15);
  }, [scData]);

  // Feature importances sorted
  const featureData = useMemo(() => {
    if (!scData?.feature_importances) return [];
    return Object.entries(scData.feature_importances)
      .map(([feature, importance]) => ({ feature, importance: importance as number }))
      .filter(f => f.importance > 0)
      .sort((a, b) => b.importance - a.importance)
      .slice(0, 10);
  }, [scData]);

  // XGBoost feature importances
  const xgbFeatureData = useMemo(() => {
    if (!xgbData?.feature_importances) return [];
    return Object.entries(xgbData.feature_importances)
      .map(([feature, importance]) => ({ feature, importance: importance as number }))
      .filter(f => f.importance > 0.001)
      .sort((a, b) => b.importance - a.importance)
      .slice(0, 12);
  }, [xgbData]);

  // XGBoost circuit accuracy sorted by MAE
  const xgbCircuitData = useMemo(() => {
    if (!xgbData?.circuit_accuracy) return [];
    return Object.entries(xgbData.circuit_accuracy)
      .map(([circuit, stats]: [string, any]) => ({
        circuit: circuit.replace(' Grand Prix', ''),
        mae: stats.mae,
        median_ae: stats.median_ae,
        n_laps: stats.n_laps,
      }))
      .sort((a, b) => a.mae - b.mae);
  }, [xgbData]);

  // Driver deltas for chart
  const driverDeltaData = useMemo(() => {
    if (!eltData?.driver_deltas) return [];
    return eltData.driver_deltas.slice(0, 15).map((d: any) => ({
      driver: d.driver,
      delta: d.avg_delta,
      races: d.races,
    }));
  }, [eltData]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="w-6 h-6 animate-spin text-primary" />
        <span className="ml-2 text-muted-foreground">Loading strategy data...</span>
      </div>
    );
  }

  return (
    <div className="space-y-4 pb-6">

      {/* ── Race Prep Mode ── */}
      <div className="bg-card border border-border rounded-lg">
        <button
          onClick={() => setPrepExpanded(!prepExpanded)}
          className="w-full flex items-center justify-between px-4 py-3 hover:bg-background/50 transition-colors rounded-lg"
        >
          <div className="flex items-center gap-2">
            <ClipboardList className="w-4 h-4 text-primary" />
            <span className="text-sm font-semibold text-foreground">Race Prep Mode</span>
            {prepReport && (
              <span className="text-[10px] text-muted-foreground font-mono">
                {prepReport.race_name} · {prepReport.drivers.join(', ')}
              </span>
            )}
          </div>
          {prepExpanded ? <ChevronUp className="w-4 h-4 text-muted-foreground" /> : <ChevronDown className="w-4 h-4 text-muted-foreground" />}
        </button>

        {prepExpanded && (
          <div className="px-4 pb-4 space-y-4 border-t border-border pt-3">
            {/* Controls */}
            <div className="flex items-center gap-3">
              <select
                title="Select race"
                value={prepRace}
                onChange={(e) => setPrepRace(e.target.value)}
                className="bg-background border border-border rounded-lg px-3 py-1.5 text-sm text-foreground focus:outline-none focus:border-primary/30"
              >
                {RACES_2024.map(r => (
                  <option key={r} value={r}>{r}</option>
                ))}
              </select>
              <button
                type="button"
                onClick={generatePrep}
                disabled={prepLoading}
                className="flex items-center gap-2 px-4 py-1.5 bg-primary text-primary-foreground rounded-lg text-[12px] font-medium hover:bg-primary/90 disabled:opacity-50 transition-colors"
              >
                {prepLoading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
                {prepLoading ? 'Generating...' : 'Generate'}
              </button>
              {prepReport && (
                <span className="text-[10px] text-muted-foreground font-mono">
                  {prepReport.generation_time_s}s | {prepReport.model_used} | {prepReport.team}
                </span>
              )}
            </div>

            {prepError && (
              <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-3 text-[12px] text-red-400 flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 shrink-0" />
                {prepError}
              </div>
            )}

            {prepLoading && !prepReport && (
              <div className="bg-background border border-border rounded-lg p-12 text-center">
                <Loader2 className="w-8 h-8 animate-spin text-primary mx-auto mb-3" />
                <div className="text-[12px] text-muted-foreground">Generating pre-race intelligence...</div>
                <div className="text-[10px] text-muted-foreground/60 mt-1">Anomaly + ELT + Degradation + Strategy context + LLM synthesis</div>
              </div>
            )}

            {prepReport && (() => {
              const p = prepReport.pillars;
              return (
                <>
                  {/* LLM Briefing */}
                  <Divider label="PRE-RACE BRIEFING" />
                  <div className="bg-background border border-border rounded-lg p-4">
                    <div className="text-[12px] text-foreground/90 whitespace-pre-line leading-relaxed">
                      {prepReport.briefing}
                    </div>
                  </div>

                  {/* System Health */}
                  {Object.keys(p.anomaly).length > 0 && (
                    <>
                      <Divider label="SYSTEM HEALTH" />
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        {Object.entries(p.anomaly).map(([driver, data]) => (
                          <div key={driver} className="bg-background border border-border rounded-lg p-3">
                            <div className="flex items-center justify-between mb-2">
                              <span className="text-[12px] font-medium">{driver}</span>
                              <div className="flex items-center gap-2">
                                <span className="text-lg font-mono font-bold" style={{ color: healthColor(data.overall_health) }}>
                                  {data.overall_health}%
                                </span>
                                <span className={`text-[9px] px-1.5 py-0.5 rounded border ${levelBg(data.overall_level)}`}>
                                  {data.overall_level}
                                </span>
                              </div>
                            </div>
                            <div className="grid grid-cols-2 gap-1.5">
                              {Object.entries(data.systems).map(([sys, sh]) => (
                                <div key={sys} className="flex items-center justify-between text-[10px] px-2 py-1 rounded bg-card border border-border/50">
                                  <span className="text-muted-foreground truncate mr-2">{sys}</span>
                                  <span className="font-mono" style={{ color: healthColor(sh.health) }}>{sh.health}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        ))}
                      </div>
                    </>
                  )}

                  {/* ELT Pace */}
                  {Object.keys(p.elt_pace).length > 0 && (
                    <>
                      <Divider label="PACE PREDICTION" />
                      <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                        {Object.entries(p.elt_pace).map(([driver, data]) => (
                          <div key={driver} className="bg-background border border-border rounded-lg p-3 text-center">
                            <div className="text-[10px] text-muted-foreground mb-1">{driver}</div>
                            <div className="text-xl font-mono font-bold text-primary">{data.predicted_pace_s.toFixed(3)}s</div>
                            <div className="text-[9px] text-muted-foreground mt-1">
                              Baseline {data.baseline_pace_s.toFixed(3)}s |
                              Advantage <span className={data.driver_advantage_s < 0 ? 'text-green-400' : 'text-amber-400'}>
                                {data.driver_advantage_s > 0 ? '+' : ''}{data.driver_advantage_s.toFixed(3)}s
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </>
                  )}

                  {/* Tyre Degradation */}
                  {p.degradation_curves.length > 0 && (
                    <>
                      <Divider label="TYRE DEGRADATION" />
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                        {p.degradation_curves
                          .filter(c => c.temp_band === 'all')
                          .map((c, i) => (
                            <div key={i} className="bg-background border border-border rounded-lg p-3 text-center">
                              <div className="text-[10px] font-mono font-bold mb-1" style={{ color: compoundColor[c.compound] || '#888' }}>
                                {c.compound}
                              </div>
                              <div className="text-sm font-mono text-foreground">
                                {c.deg_per_lap_s != null ? `${c.deg_per_lap_s.toFixed(4)} s/lap` : '—'}
                              </div>
                              {c.r_squared != null && <div className="text-[9px] text-muted-foreground">R²={c.r_squared.toFixed(3)}</div>}
                              {c.n_stints != null && <div className="text-[9px] text-muted-foreground">{c.n_stints} stints</div>}
                            </div>
                          ))}
                      </div>
                    </>
                  )}

                  {/* Circuit Conditions */}
                  {(p.circuit.pit_loss_s || p.circuit.air_density) && (
                    <>
                      <Divider label="CIRCUIT CONDITIONS" />
                      <div className="flex gap-3">
                        {p.circuit.pit_loss_s != null && (
                          <div className="bg-background border border-border rounded-lg px-4 py-2 text-center">
                            <div className="text-[9px] text-muted-foreground">Pit Loss</div>
                            <div className="text-sm font-mono font-bold text-primary">{p.circuit.pit_loss_s.toFixed(1)}s</div>
                          </div>
                        )}
                        {p.circuit.air_density && (
                          <>
                            <div className="bg-background border border-border rounded-lg px-4 py-2 text-center">
                              <div className="text-[9px] text-muted-foreground">Temperature</div>
                              <div className="text-sm font-mono font-bold">{p.circuit.air_density.temperature_c}°C</div>
                            </div>
                            <div className="bg-background border border-border rounded-lg px-4 py-2 text-center">
                              <div className="text-[9px] text-muted-foreground">Air Density</div>
                              <div className="text-sm font-mono font-bold">{p.circuit.air_density.density_kg_m3} kg/m³</div>
                            </div>
                            <div className="bg-background border border-border rounded-lg px-4 py-2 text-center">
                              <div className="text-[9px] text-muted-foreground">Humidity</div>
                              <div className="text-sm font-mono font-bold">{p.circuit.air_density.humidity_pct}%</div>
                            </div>
                          </>
                        )}
                      </div>
                    </>
                  )}
                </>
              );
            })()}
          </div>
        )}
      </div>

      {/* ── Section 1: Session KPIs ── */}
      {session && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-3">
            <KPI
              icon={<Flag className="w-4 h-4 text-primary" />}
              label="RACE"
              value={session.meeting}
              detail={`${session.circuit} — ${session.year}`}
            />
            <KPI
              icon={<Gauge className="w-4 h-4 text-primary" />}
              label="TOTAL LAPS"
              value={String(session.total_laps)}
              detail={`Session ${session.session_key}`}
            />
            <KPI
              icon={<Users className="w-4 h-4 text-primary" />}
              label="DRIVERS"
              value={String(drivers.length)}
              detail={`${simMeta?.n_strategies_evaluated || '—'} strategies evaluated`}
            />
            <KPI
              icon={<Timer className="w-4 h-4 text-primary" />}
              label="PIT LOSS"
              value={`${simMeta?.pit_loss?.toFixed(1) || '22.0'}s`}
              detail="Time lost per pit stop"
            />
          </div>
        </>
      )}

      {/* ── Section 2: Stint Timeline ── */}
      <Divider label="STINT TIMELINE" />
      <div className="bg-card border border-border rounded-lg p-4">
        <div className="flex items-center gap-4 mb-3">
          <h3 className="text-foreground font-semibold text-sm">Pit Strategy Tracker</h3>
          <div className="flex items-center gap-3 text-[11px]">
            {['SOFT', 'MEDIUM', 'HARD'].map(c => (
              <div key={c} className="flex items-center gap-1.5">
                <span className="w-3 h-2 rounded-sm" style={{ backgroundColor: compoundColors[c] }} />
                <span className="text-muted-foreground">{c}</span>
              </div>
            ))}
          </div>
          {Object.keys(cliffLaps).length > 0 && (
            <div className="ml-auto text-[11px] text-muted-foreground">
              Cliff: {Object.entries(cliffLaps).map(([c, lap]) => (
                <span key={c} className="ml-2">
                  <span style={{ color: compoundTextColors[c] }}>{c}</span> L{String(lap)}
                </span>
              ))}
            </div>
          )}
        </div>

        <div className="space-y-1 max-h-[500px] overflow-y-auto">
          {drivers.map((d: any) => {
            const isExpanded = expandedDriver === d.driver;
            const isMcL = isMcLaren(d.team);
            return (
              <div
                key={d.driver_number}
                className={`rounded-lg transition cursor-pointer ${
                  isMcL ? 'border-l-2 border-l-primary' : ''
                } ${isExpanded ? 'bg-secondary' : 'hover:bg-[#1e2433]'}`}
                onClick={() => setExpandedDriver(isExpanded ? null : d.driver)}
              >
                <div className="flex items-center gap-3 px-3 py-1.5">
                  {/* Driver name */}
                  <div className="w-12 shrink-0">
                    <span className={`text-sm font-mono ${isMcL ? 'text-primary font-semibold' : 'text-foreground'}`}>
                      {d.driver}
                    </span>
                  </div>

                  {/* Team */}
                  <div className="w-28 shrink-0">
                    <span className="text-[11px] text-muted-foreground truncate">{d.team}</span>
                  </div>

                  {/* Stint bars */}
                  <div className="flex-1 flex items-center h-6 gap-px rounded overflow-hidden bg-background">
                    {d.stints.map((s: any, i: number) => {
                      const pct = session ? (s.stint_laps / session.total_laps) * 100 : 50;
                      return (
                        <div
                          key={i}
                          className="h-full flex items-center justify-center text-[10px] font-mono"
                          style={{
                            width: `${pct}%`,
                            backgroundColor: compoundColors[s.compound] || '#6b7280',
                            color: s.compound === 'HARD' ? '#1A1F2E' : '#fff',
                            minWidth: '20px',
                          }}
                          title={`${s.compound} — ${s.stint_laps} laps (L${s.lap_start}-L${s.lap_end})`}
                        >
                          {s.stint_laps}
                        </div>
                      );
                    })}
                  </div>

                  {/* Stops count */}
                  <div className="w-16 shrink-0 text-right">
                    <span className="text-[11px] text-muted-foreground">{d.total_stops} stop{d.total_stops !== 1 ? 's' : ''}</span>
                  </div>

                  {/* Expand icon */}
                  {isExpanded
                    ? <ChevronUp className="w-3 h-3 text-muted-foreground shrink-0" />
                    : <ChevronDown className="w-3 h-3 text-muted-foreground shrink-0" />
                  }
                </div>

                {/* Expanded details */}
                {isExpanded && (
                  <div className="px-3 pb-2 pt-1 border-t border-border">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-[11px]">
                      {d.stints.map((s: any, i: number) => (
                        <div key={i} className="bg-background rounded-lg p-2">
                          <div className="flex items-center gap-1.5 mb-1">
                            <Disc className="w-3 h-3" style={{ color: compoundTextColors[s.compound] }} />
                            <span style={{ color: compoundTextColors[s.compound] }} className="font-semibold">{s.compound}</span>
                          </div>
                          <div className="text-muted-foreground">
                            Stint {s.stint_number}: L{s.lap_start}–L{s.lap_end} ({s.stint_laps} laps)
                          </div>
                          {s.tyre_age_at_start > 0 && (
                            <div className="text-muted-foreground">Used tyre: +{s.tyre_age_at_start} laps</div>
                          )}
                          {s.cliff_lap && (
                            <div className="text-yellow-500">Cliff: lap {s.cliff_lap}</div>
                          )}
                          {s.predicted_pit_window && (
                            <div className="text-primary">Pit window: L{s.predicted_pit_window}</div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* ── Section 2b: Battle Intel ── */}
      {battleData?.battles?.length > 0 && (
        <>
          <Divider label="BATTLE INTEL" />
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Undercut threats */}
            <div className="bg-card border border-border rounded-lg p-4">
              <div className="flex items-center gap-2 mb-3">
                <Zap className="w-4 h-4 text-red-400" />
                <h3 className="text-foreground font-semibold text-sm">Undercut Threats</h3>
                <span className="ml-auto text-[11px] text-muted-foreground">
                  {battleData.undercut_count} driver{battleData.undercut_count !== 1 ? 's' : ''} within 1.5s
                </span>
              </div>
              <div className="space-y-1.5">
                {battleData.battles
                  .filter((b: any) => b.undercut_threat)
                  .map((b: any) => (
                    <div
                      key={b.driver_number}
                      className={`flex items-center justify-between px-3 py-1.5 rounded-lg bg-background ${
                        isMcLaren(b.team) ? 'border-l-2 border-l-primary' : ''
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <span className={`text-sm font-mono ${isMcLaren(b.team) ? 'text-primary font-semibold' : 'text-foreground'}`}>
                          {b.driver}
                        </span>
                        <span className="text-[10px] text-muted-foreground">{b.team}</span>
                      </div>
                      <div className="flex items-center gap-3 text-[12px] font-mono">
                        <span className="text-red-400">
                          {b.interval != null ? `${b.interval > 0 ? '+' : ''}${b.interval.toFixed(2)}s` : '—'}
                        </span>
                        {b.trend === 1 && <span className="text-green-400 text-[10px]">CLOSING</span>}
                        {b.trend === -1 && <span className="text-red-400 text-[10px]">FALLING</span>}
                      </div>
                    </div>
                  ))}
                {battleData.undercut_count === 0 && (
                  <div className="text-[12px] text-muted-foreground text-center py-2">No undercut threats detected</div>
                )}
              </div>
            </div>

            {/* Gap trends */}
            <div className="bg-card border border-border rounded-lg p-4">
              <div className="flex items-center gap-2 mb-3">
                <TrendingUp className="w-4 h-4 text-green-400" />
                <h3 className="text-foreground font-semibold text-sm">Gap Trends</h3>
                <span className="ml-auto text-[11px] text-muted-foreground">
                  {battleData.closing_count} closing
                </span>
              </div>
              <div className="space-y-1 max-h-[300px] overflow-y-auto">
                {battleData.battles.slice(0, 20).map((b: any) => (
                  <div
                    key={b.driver_number}
                    className={`flex items-center justify-between px-3 py-1 rounded-lg hover:bg-secondary ${
                      isMcLaren(b.team) ? 'border-l-2 border-l-primary' : ''
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <span className={`text-[12px] font-mono w-10 ${isMcLaren(b.team) ? 'text-primary font-semibold' : 'text-foreground'}`}>
                        {b.driver}
                      </span>
                      <span className="text-[10px] text-muted-foreground w-20 truncate">{b.team}</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <span className="text-[11px] font-mono text-muted-foreground w-16 text-right">
                        {b.gap_to_leader != null
                          ? b.gap_to_leader === 0 ? 'LEADER' : `+${b.gap_to_leader.toFixed(1)}s`
                          : '—'}
                      </span>
                      <span className={`text-[11px] font-mono w-16 text-right ${
                        b.interval != null && Math.abs(b.interval) < 1.5 ? 'text-red-400' : 'text-muted-foreground'
                      }`}>
                        {b.interval != null ? `${b.interval.toFixed(2)}s` : '—'}
                      </span>
                      <span className="w-5 text-center">
                        {b.trend === 1 && <ChevronUp className="w-3.5 h-3.5 text-green-400 inline" />}
                        {b.trend === -1 && <ChevronDown className="w-3.5 h-3.5 text-red-400 inline" />}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </>
      )}

      {/* ── Section 2c: Opponent Strategy Patterns ── */}
      {opponentData?.drivers?.length > 0 && (
        <>
          <Divider label="OPPONENT STRATEGY PATTERNS" />
          <div className="bg-card border border-border rounded-lg p-4">
            <div className="flex items-center gap-2 mb-3">
              <Shield className="w-4 h-4 text-purple-400" />
              <h3 className="text-foreground font-semibold text-sm">Rival Strategy Profiles</h3>
              <span className="ml-auto text-[11px] text-muted-foreground">
                {opponentData.count} drivers profiled
              </span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-[12px]">
                <thead>
                  <tr className="text-muted-foreground border-b border-border">
                    <th className="text-left py-1.5 pr-2">Driver</th>
                    <th className="text-left py-1.5 pr-2">Name</th>
                    <th className="text-left py-1.5 pr-2">Nationality</th>
                    <th className="text-right py-1.5 pr-2">Races</th>
                    <th className="text-right py-1.5 pr-2">Wins</th>
                    <th className="text-right py-1.5 pr-2">Podiums</th>
                    <th className="text-right py-1.5">Seasons</th>
                  </tr>
                </thead>
                <tbody>
                  {opponentData.drivers.slice(0, 20).map((d: any) => (
                    <tr
                      key={d.driver_id}
                      className="border-b border-[rgba(255,128,0,0.04)]"
                    >
                      <td className="py-1.5 pr-2 font-mono text-foreground">
                        {d.driver_code || d.driver_id?.slice(0, 3).toUpperCase()}
                      </td>
                      <td className="py-1.5 pr-2 text-muted-foreground">{d.forename} {d.surname}</td>
                      <td className="py-1.5 pr-2 text-muted-foreground">{d.nationality ?? '—'}</td>
                      <td className="py-1.5 pr-2 text-right font-mono text-muted-foreground">{d.total_races ?? '—'}</td>
                      <td className="py-1.5 pr-2 text-right font-mono text-foreground">{d.total_wins ?? '—'}</td>
                      <td className="py-1.5 pr-2 text-right font-mono text-foreground">{d.total_podiums ?? '—'}</td>
                      <td className="py-1.5 text-right font-mono text-muted-foreground">
                        {d.seasons?.length ?? '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* ── Section 3: Optimal Strategies ── */}
      {optimalStrategies.length > 0 && (
        <>
          <Divider label="OPTIMAL STRATEGIES" />
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Table */}
            <div className="bg-card border border-border rounded-lg p-4">
              <h3 className="text-foreground font-semibold text-sm mb-3">Simulated Optimal Strategy</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-[12px]">
                  <thead>
                    <tr className="text-muted-foreground border-b border-border">
                      <th className="text-left py-1.5 pr-2">Pos</th>
                      <th className="text-left py-1.5 pr-2">Driver</th>
                      <th className="text-left py-1.5 pr-2">Team</th>
                      <th className="text-left py-1.5 pr-2">Strategy</th>
                      <th className="text-right py-1.5">Gap</th>
                    </tr>
                  </thead>
                  <tbody>
                    {optimalStrategies.slice(0, 20).map((s: any, i: number) => (
                      <tr
                        key={i}
                        className={`border-b border-[rgba(255,128,0,0.04)] ${
                          isMcLaren(s.team) ? 'bg-primary/5' : ''
                        }`}
                      >
                        <td className="py-1.5 pr-2 font-mono text-muted-foreground">{i + 1}</td>
                        <td className={`py-1.5 pr-2 font-mono ${isMcLaren(s.team) ? 'text-primary font-semibold' : 'text-foreground'}`}>
                          {s.driver}
                        </td>
                        <td className="py-1.5 pr-2 text-muted-foreground">{s.team}</td>
                        <td className="py-1.5 pr-2 text-foreground">{s.strategy}</td>
                        <td className="py-1.5 text-right font-mono text-muted-foreground">
                          {s.gap_to_leader === 0 ? 'LEADER' : `+${s.gap_to_leader.toFixed(1)}s`}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Gap chart */}
            <div className="bg-card border border-border rounded-lg p-4">
              <h3 className="text-foreground font-semibold text-sm mb-3">Gap to Leader</h3>
              <ResponsiveContainer width="100%" height={350}>
                <BarChart
                  data={optimalStrategies.slice(0, 15)}
                  layout="vertical"
                  margin={{ left: 5, right: 10 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
                  <XAxis type="number" tick={{ fill: '#8b949e', fontSize: 11 }} unit="s" />
                  <YAxis type="category" dataKey="driver" tick={{ fill: '#8b949e', fontSize: 11 }} width={40} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="gap_to_leader" name="Gap" radius={[0, 4, 4, 0]}>
                    {optimalStrategies.slice(0, 15).map((s: any, i: number) => (
                      <Cell key={i} fill={isMcLaren(s.team) ? '#FF8000' : '#3b82f6'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </>
      )}

      {/* ── Section 4: Tyre Degradation Curves ── */}
      {degCurveData.length > 0 && (
        <>
          <Divider label="TYRE DEGRADATION" />
          <div className="bg-card border border-border rounded-lg p-4">
            <div className="flex items-center gap-3 mb-3">
              <h3 className="text-foreground font-semibold text-sm">Degradation Curves — {degData?.circuit}</h3>
              <div className="flex items-center gap-3 text-[11px] ml-auto">
                {['SOFT', 'MEDIUM', 'HARD'].map(c => {
                  const curve = degData?.curves?.find(
                    (cv: any) => cv.compound === c && (cv.temp_band === 'all' || !cv.temp_band)
                  );
                  return (
                    <div key={c} className="flex items-center gap-1.5">
                      <span className="w-3 h-0.5 rounded" style={{ backgroundColor: compoundColors[c] }} />
                      <span className="text-muted-foreground">{c}</span>
                      {curve?.cliff_lap && (
                        <span className="text-yellow-500 text-[10px]">cliff L{curve.cliff_lap}</span>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={degCurveData} margin={{ left: 5, right: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis
                  dataKey="lap"
                  label={{ value: 'Tyre Life (laps)', position: 'bottom', fill: '#8b949e', fontSize: 11, offset: -5 }}
                  tick={{ fill: '#8b949e', fontSize: 11 }}
                />
                <YAxis
                  label={{ value: 'Degradation (s)', angle: -90, position: 'insideLeft', fill: '#8b949e', fontSize: 11 }}
                  tick={{ fill: '#8b949e', fontSize: 11 }}
                />
                <Tooltip content={<CustomTooltip />} />
                <Line type="monotone" dataKey="SOFT" stroke={compoundColors.SOFT} strokeWidth={2} dot={false} name="Soft" />
                <Line type="monotone" dataKey="MEDIUM" stroke={compoundColors.MEDIUM} strokeWidth={2} dot={false} name="Medium" />
                <Line type="monotone" dataKey="HARD" stroke={compoundColors.HARD} strokeWidth={2} dot={false} name="Hard" />
                {/* Cliff reference lines */}
                {degData?.curves?.filter((c: any) => c.cliff_lap && (c.temp_band === 'all' || !c.temp_band)).map((c: any) => (
                  <ReferenceLine
                    key={c.compound}
                    x={c.cliff_lap}
                    stroke={compoundColors[c.compound]}
                    strokeDasharray="5 5"
                    strokeOpacity={0.5}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </>
      )}

      {/* ── Section 5: SC Probability & ELT ── */}
      <Divider label="MODELS" />
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">

        {/* SC Probability */}
        {scData?.metadata && (
          <div className="bg-card border border-border rounded-lg p-4">
            <div className="flex items-center gap-2 mb-3">
              <AlertTriangle className="w-4 h-4 text-yellow-500" />
              <h3 className="text-foreground font-semibold text-sm">Safety Car Probability</h3>
              <span className="ml-auto text-[11px] text-muted-foreground">
                {scData.metadata.algorithm} v{scData.metadata.model_version}
              </span>
            </div>

            {/* Model metrics */}
            <div className="grid grid-cols-3 gap-2 mb-3">
              {[
                { label: 'ROC-AUC', value: scData.metadata.validation?.roc_auc?.toFixed(3) },
                { label: 'PR-AUC', value: scData.metadata.validation?.pr_auc?.toFixed(3) },
                { label: 'F1', value: scData.metadata.validation?.f1?.toFixed(3) },
              ].map(m => (
                <div key={m.label} className="bg-background rounded-lg p-2 text-center">
                  <div className="text-[10px] text-muted-foreground">{m.label}</div>
                  <div className="text-sm font-mono text-foreground">{m.value}</div>
                </div>
              ))}
            </div>

            {/* Feature importances with descriptions */}
            {featureData.length > 0 && (
              <>
                <div className="text-[11px] text-muted-foreground mb-2">Top Feature Importances</div>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={featureData} layout="vertical" margin={{ left: 10, right: 10 }}>
                    <XAxis type="number" tick={{ fill: '#8b949e', fontSize: 10 }} />
                    <YAxis type="category" dataKey="feature" tick={{ fill: '#8b949e', fontSize: 10 }} width={110} />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey="importance" name="Importance" fill="#eab308" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
                <div className="mt-2 space-y-0.5">
                  {featureData.slice(0, 5).map(f => {
                    const desc: Record<string, string> = {
                      lap_number: 'How far into the race — late-race incidents are more common',
                      position_spread: 'Spread of positions — closer packs mean more contact risk',
                      tyre_age_max: 'Oldest tyres on grid — degraded rubber increases lock-ups',
                      speed_std: 'Speed variance across field — mixed pace leads to incidents',
                      drs_enabled: 'DRS activation status — high-speed overtakes raise collision risk',
                      pit_stops_total: 'Cumulative pit stops — busy pit phases correlate with SC',
                      track_temp: 'Track temperature affecting grip and tyre blister risk',
                      air_temp: 'Air temperature impacting engine cooling and reliability',
                      humidity: 'Humidity level — high humidity reduces grip',
                      rainfall: 'Rain on track — dramatically increases incident probability',
                    };
                    const key = f.feature.toLowerCase().replace(/[^a-z_]/g, '');
                    const match = Object.entries(desc).find(([k]) => key.includes(k));
                    if (!match) return null;
                    return (
                      <div key={f.feature} className="flex items-start gap-2 text-[10px]">
                        <span className="text-yellow-500/60 shrink-0 mt-px">*</span>
                        <span className="text-muted-foreground">
                          <span className="text-foreground/70">{f.feature}</span> — {match[1]}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </>
            )}
          </div>
        )}

        {/* SC Rates by Circuit */}
        {scRateData.length > 0 && (
          <div className="bg-card border border-border rounded-lg p-4">
            <div className="flex items-center gap-2 mb-3">
              <Zap className="w-4 h-4 text-yellow-500" />
              <h3 className="text-foreground font-semibold text-sm">SC Rate by Circuit</h3>
              {(() => {
                const currentCircuit = session?.circuit;
                const currentRate = currentCircuit ? scData?.circuit_sc_rates?.[currentCircuit] : null;
                const avgRate = scRateData.length > 0 ? scRateData.reduce((s, d) => s + d.rate, 0) / scRateData.length : 0;
                if (currentRate != null) {
                  const level = currentRate >= 1 ? 'HIGH' : currentRate >= 0.5 ? 'MED' : 'LOW';
                  const color = currentRate >= 1 ? 'text-red-400 bg-red-400/10 border-red-400/20' : currentRate >= 0.5 ? 'text-yellow-400 bg-yellow-400/10 border-yellow-400/20' : 'text-green-400 bg-green-400/10 border-green-400/20';
                  return (
                    <span className={`text-[10px] font-mono px-2 py-0.5 rounded-full border ${color}`}>
                      {currentCircuit}: {currentRate.toFixed(2)} SC/race — {level} {currentRate > avgRate ? '(above avg)' : '(below avg)'}
                    </span>
                  );
                }
                return null;
              })()}
              <span className="ml-auto text-[11px] text-muted-foreground">avg SC deployments per race</span>
            </div>
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={scRateData} layout="vertical" margin={{ left: 10, right: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
                <XAxis type="number" tick={{ fill: '#8b949e', fontSize: 10 }} />
                <YAxis type="category" dataKey="circuit" tick={{ fill: '#8b949e', fontSize: 10 }} width={120} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="rate" name="SC/Race" radius={[0, 4, 4, 0]}>
                  {scRateData.map((d, i) => (
                    <Cell key={i} fill={d.rate >= 1 ? '#ef4444' : d.rate >= 0.5 ? '#eab308' : '#22c55e'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* ── Lap Prediction Models ── */}
      {(xgbData?.metadata || bilstmData?.metadata) && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">

          {/* XGBoost Lap Predictor */}
          {xgbData?.metadata && (
            <div className="bg-card border border-border rounded-lg p-4">
              <div className="flex items-center gap-2 mb-3">
                <Brain className="w-4 h-4 text-primary" />
                <h3 className="text-foreground font-semibold text-sm">XGBoost Lap Predictor</h3>
                <span className="ml-auto text-[10px] text-muted-foreground font-mono">
                  v{xgbData.metadata.model_version} &middot; {xgbData.metadata.n_estimators} trees
                </span>
              </div>

              {/* Validation metrics */}
              <div className="grid grid-cols-4 gap-2 mb-3">
                {[
                  { label: 'R\u00B2', value: xgbData.metadata.validation?.r2?.toFixed(4) },
                  { label: 'MAE', value: `${xgbData.metadata.validation?.mae?.toFixed(3)}s` },
                  { label: 'RMSE', value: `${xgbData.metadata.validation?.rmse?.toFixed(3)}s` },
                  { label: 'Test', value: `${(xgbData.metadata.validation?.n_test / 1000).toFixed(1)}K laps` },
                ].map(m => (
                  <div key={m.label} className="bg-background rounded-lg p-2 text-center">
                    <div className="text-[10px] text-muted-foreground">{m.label}</div>
                    <div className="text-sm font-mono text-foreground">{m.value}</div>
                  </div>
                ))}
              </div>

              {/* Feature importances */}
              {xgbFeatureData.length > 0 && (
                <>
                  <div className="text-[11px] text-muted-foreground mb-2">Top Feature Importances</div>
                  <ResponsiveContainer width="100%" height={240}>
                    <BarChart data={xgbFeatureData} layout="vertical" margin={{ left: 10, right: 10 }}>
                      <XAxis type="number" tick={{ fill: '#8b949e', fontSize: 10 }} />
                      <YAxis type="category" dataKey="feature" tick={{ fill: '#8b949e', fontSize: 10 }} width={120} />
                      <Tooltip content={<CustomTooltip />} />
                      <Bar dataKey="importance" name="Importance" fill="#FF8000" radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </>
              )}
            </div>
          )}

          {/* BiLSTM Lap Predictor */}
          {bilstmData?.metadata && (
            <div className="bg-card border border-border rounded-lg p-4">
              <div className="flex items-center gap-2 mb-3">
                <Cpu className="w-4 h-4 text-purple-400" />
                <h3 className="text-foreground font-semibold text-sm">BiLSTM Temporal Predictor</h3>
                <span className="ml-auto text-[10px] text-muted-foreground font-mono">
                  v{bilstmData.metadata.model_version} &middot; {(bilstmData.metadata.architecture?.total_params / 1000).toFixed(0)}K params
                </span>
              </div>

              {/* Architecture */}
              <div className="grid grid-cols-4 gap-2 mb-3">
                {[
                  { label: 'R\u00B2', value: bilstmData.metadata.validation?.r2?.toFixed(4) },
                  { label: 'MAE', value: `${bilstmData.metadata.validation?.mae?.toFixed(3)}s` },
                  { label: 'RMSE', value: `${bilstmData.metadata.validation?.rmse?.toFixed(3)}s` },
                  { label: 'Test', value: `${(bilstmData.metadata.validation?.n_test / 1000).toFixed(1)}K laps` },
                ].map(m => (
                  <div key={m.label} className="bg-background rounded-lg p-2 text-center">
                    <div className="text-[10px] text-muted-foreground">{m.label}</div>
                    <div className="text-sm font-mono text-foreground">{m.value}</div>
                  </div>
                ))}
              </div>

              {/* Architecture details */}
              <div className="text-[11px] text-muted-foreground mb-2">Architecture</div>
              <div className="grid grid-cols-2 gap-2 mb-3">
                {[
                  { label: 'Type', value: bilstmData.metadata.architecture?.type },
                  { label: 'Hidden', value: bilstmData.metadata.architecture?.hidden_size },
                  { label: 'Layers', value: bilstmData.metadata.architecture?.num_layers },
                  { label: 'Window', value: `${bilstmData.metadata.architecture?.window_size} laps` },
                  { label: 'Dropout', value: bilstmData.metadata.architecture?.dropout },
                  { label: 'Optimizer', value: bilstmData.metadata.training?.optimizer },
                ].map(m => (
                  <div key={m.label} className="flex items-center justify-between bg-background rounded-lg px-3 py-1.5">
                    <span className="text-[10px] text-muted-foreground">{m.label}</span>
                    <span className="text-[11px] font-mono text-foreground">{m.value}</span>
                  </div>
                ))}
              </div>

              {/* Model comparison */}
              <div className="text-[11px] text-muted-foreground mb-2">vs XGBoost</div>
              <div className="flex items-center gap-3">
                {xgbData?.metadata?.validation && bilstmData.metadata.validation && (() => {
                  const xgbMae = xgbData.metadata.validation.mae;
                  const bilstmMae = bilstmData.metadata.validation.mae;
                  const diff = bilstmMae - xgbMae;
                  const pct = ((diff / xgbMae) * 100).toFixed(1);
                  return (
                    <>
                      <div className={`text-[11px] px-2 py-0.5 rounded-full border ${diff > 0
                        ? 'text-yellow-400 bg-yellow-400/10 border-yellow-400/20'
                        : 'text-green-400 bg-green-400/10 border-green-400/20'}`}>
                        MAE {diff > 0 ? '+' : ''}{diff.toFixed(3)}s ({diff > 0 ? '+' : ''}{pct}%)
                      </div>
                      <span className="text-[10px] text-muted-foreground">
                        XGBoost wins on accuracy, BiLSTM captures temporal patterns
                      </span>
                    </>
                  );
                })()}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Lap Time Predictor ── */}
      {xgbData?.metadata && (
        <div className="bg-card border border-border rounded-lg p-4">
          <div className="flex items-center gap-2 mb-4">
            <Brain className="w-4 h-4 text-primary" />
            <h3 className="text-foreground font-semibold text-sm">Lap Time Predictor</h3>
            <span className="ml-auto text-[10px] text-muted-foreground">
              XGBoost MAE {xgbData.metadata.validation?.mae?.toFixed(2)}s
              {bilstmData?.metadata?.validation && <> &middot; BiLSTM MAE {bilstmData.metadata.validation.mae?.toFixed(2)}s</>}
            </span>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-6 gap-3 mb-4">
            <div>
              <label className="text-[10px] text-muted-foreground block mb-1">Circuit</label>
              <select value={predCircuit} onChange={e => setPredCircuit(e.target.value)} title="Circuit"
                className="w-full bg-background border border-[rgba(255,128,0,0.15)] rounded-lg px-2 py-1.5 text-[12px] text-foreground">
                {Object.keys(xgbData.metadata.encodings?.circuit_rank || {}).sort().map(c => (
                  <option key={c} value={c}>{c.replace(' Grand Prix', '')}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-[10px] text-muted-foreground block mb-1">Driver</label>
              <select value={predDriver} onChange={e => setPredDriver(e.target.value)} title="Driver"
                className="w-full bg-background border border-[rgba(255,128,0,0.15)] rounded-lg px-2 py-1.5 text-[12px] text-foreground">
                {Object.keys(xgbData.metadata.encodings?.driver_rank || {}).sort().map(d => (
                  <option key={d} value={d}>{d}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-[10px] text-muted-foreground block mb-1">Compound</label>
              <select value={predCompound} onChange={e => setPredCompound(e.target.value)} title="Compound"
                className="w-full bg-background border border-[rgba(255,128,0,0.15)] rounded-lg px-2 py-1.5 text-[12px] text-foreground">
                {['SOFT', 'MEDIUM', 'HARD'].map(c => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            </div>
            <div className="flex gap-2">
              <div className="flex-1">
                <label className="text-[10px] text-muted-foreground block mb-1">Lap Start</label>
                <input type="number" value={predLapStart} onChange={e => setPredLapStart(+e.target.value)} min={1} max={80} title="Lap Start"
                  className="w-full bg-background border border-[rgba(255,128,0,0.15)] rounded-lg px-2 py-1.5 text-[12px] text-foreground font-mono" />
              </div>
              <div className="flex-1">
                <label className="text-[10px] text-muted-foreground block mb-1">Lap End</label>
                <input type="number" value={predLapEnd} onChange={e => setPredLapEnd(+e.target.value)} min={1} max={80} title="Lap End"
                  className="w-full bg-background border border-[rgba(255,128,0,0.15)] rounded-lg px-2 py-1.5 text-[12px] text-foreground font-mono" />
              </div>
            </div>
            <div>
              <label className="text-[10px] text-muted-foreground block mb-1">Model</label>
              <select value={predModel} onChange={e => setPredModel(e.target.value as any)} title="Model"
                className="w-full bg-background border border-[rgba(255,128,0,0.15)] rounded-lg px-2 py-1.5 text-[12px] text-foreground">
                <option value="both">Both (Compare)</option>
                <option value="xgboost">XGBoost</option>
                <option value="bilstm">BiLSTM</option>
              </select>
            </div>
            <div className="flex items-end">
              <button
                type="button"
                onClick={() => {
                  setPredLoading(true);
                  const params = {
                    circuit: predCircuit, driver_code: predDriver, compound: predCompound,
                    lap_start: predLapStart, lap_end: predLapEnd,
                  };
                  if (predModel === 'xgboost') {
                    strategy.predictLap(params).then(r => { setPredResult(r); setBilstmResult(null); })
                      .catch(() => { setPredResult(null); setBilstmResult(null); })
                      .finally(() => setPredLoading(false));
                  } else if (predModel === 'bilstm') {
                    strategy.predictLapBilstm(params).then(r => { setBilstmResult(r); setPredResult(null); })
                      .catch(() => { setBilstmResult(null); setPredResult(null); })
                      .finally(() => setPredLoading(false));
                  } else {
                    Promise.all([
                      strategy.predictLap(params).catch(() => null),
                      strategy.predictLapBilstm(params).catch(() => null),
                    ]).then(([xgb, bilstm]) => { setPredResult(xgb); setBilstmResult(bilstm); })
                      .finally(() => setPredLoading(false));
                  }
                }}
                disabled={predLoading}
                className="w-full bg-primary hover:bg-primary/80 text-black font-semibold text-[12px] rounded-lg px-3 py-1.5 transition-colors disabled:opacity-50"
              >
                {predLoading ? 'Predicting...' : 'Predict'}
              </button>
            </div>
          </div>

          {/* Prediction results — merged chart for XGBoost + BiLSTM */}
          {(predResult?.predictions || bilstmResult?.predictions) && (() => {
            const meta = predResult || bilstmResult;
            // Merge both model predictions into one dataset keyed by lap
            const lapMap: Record<number, any> = {};
            if (predResult?.predictions) {
              for (const p of predResult.predictions) {
                lapMap[p.lap] = { lap: p.lap, tyre_life: p.tyre_life, deg_delta: p.deg_delta, xgboost: p.predicted_s };
              }
            }
            if (bilstmResult?.predictions) {
              for (const p of bilstmResult.predictions) {
                if (!lapMap[p.lap]) lapMap[p.lap] = { lap: p.lap, tyre_life: p.tyre_life, deg_delta: p.deg_delta };
                lapMap[p.lap].bilstm = p.predicted_s;
              }
            }
            const chartData = Object.values(lapMap).sort((a: any, b: any) => a.lap - b.lap);
            const hasXgb = predResult?.predictions;
            const hasBilstm = bilstmResult?.predictions;

            return (
              <div>
                <div className="flex items-center gap-3 mb-2">
                  <span className="text-[11px] text-muted-foreground">
                    {meta.driver} at {meta.circuit?.replace(' Grand Prix', '')} &middot; {meta.compound} &middot; Baseline: {meta.baseline_pace_s}s
                  </span>
                  <span className="text-[10px] text-muted-foreground">
                    Weather: {meta.weather?.air_temp_c}°C air / {meta.weather?.track_temp_c}°C track / {meta.weather?.humidity_pct}% hum
                  </span>
                  {hasXgb && hasBilstm && (
                    <span className="ml-auto text-[10px] text-muted-foreground flex items-center gap-3">
                      <span className="flex items-center gap-1"><span className="inline-block w-3 h-[2px] bg-primary" /> XGBoost</span>
                      <span className="flex items-center gap-1"><span className="inline-block w-3 h-[2px] bg-[#06B6D4]" /> BiLSTM</span>
                    </span>
                  )}
                </div>
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart data={chartData} margin={{ left: 5, right: 10, top: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="lap" tick={{ fill: '#8b949e', fontSize: 11 }} label={{ value: 'Lap', position: 'insideBottom', offset: -2, fill: '#8b949e', fontSize: 10 }} />
                    <YAxis tick={{ fill: '#8b949e', fontSize: 11 }} domain={['auto', 'auto']} label={{ value: 'Lap Time (s)', angle: -90, position: 'insideLeft', fill: '#8b949e', fontSize: 10 }} />
                    <Tooltip content={({ active, payload }: any) => {
                      if (!active || !payload?.length) return null;
                      const d = payload[0]?.payload;
                      return (
                        <div className="bg-background border border-[rgba(255,128,0,0.2)] rounded-lg p-2 text-[12px]">
                          <div className="text-foreground font-semibold">Lap {d?.lap}</div>
                          {d?.xgboost != null && <div className="text-primary">XGBoost: <span className="font-mono">{d.xgboost.toFixed(3)}s</span></div>}
                          {d?.bilstm != null && <div className="text-[#06B6D4]">BiLSTM: <span className="font-mono">{d.bilstm.toFixed(3)}s</span></div>}
                          {d?.xgboost != null && d?.bilstm != null && (
                            <div className="text-muted-foreground mt-1">Delta: <span className="font-mono">{(d.xgboost - d.bilstm).toFixed(3)}s</span></div>
                          )}
                          <div className="text-muted-foreground">Tyre Life: <span className="font-mono">{d?.tyre_life}</span></div>
                          <div className="text-muted-foreground">Deg: <span className="font-mono">{d?.deg_delta?.toFixed(3)}s</span></div>
                        </div>
                      );
                    }} />
                    <ReferenceLine y={meta.baseline_pace_s} stroke="#8b949e" strokeDasharray="3 3" label={{ value: 'Baseline', fill: '#8b949e', fontSize: 10 }} />
                    {hasXgb && <Line type="monotone" dataKey="xgboost" stroke="#FF8000" strokeWidth={2} dot={false} name="XGBoost" />}
                    {hasBilstm && <Line type="monotone" dataKey="bilstm" stroke="#06B6D4" strokeWidth={2} dot={false} name="BiLSTM" strokeDasharray={hasXgb ? "5 3" : undefined} />}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            );
          })()}
        </div>
      )}

      {/* XGBoost Circuit Accuracy */}
      {xgbCircuitData.length > 0 && (
        <div className="bg-card border border-border rounded-lg p-4">
          <div className="flex items-center gap-2 mb-3">
            <Flag className="w-4 h-4 text-primary" />
            <h3 className="text-foreground font-semibold text-sm">XGBoost Accuracy by Circuit</h3>
            <span className="ml-auto text-[11px] text-muted-foreground">
              MAE in seconds &middot; lower is better
            </span>
          </div>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={xgbCircuitData} layout="vertical" margin={{ left: 10, right: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
              <XAxis type="number" tick={{ fill: '#8b949e', fontSize: 10 }} domain={[0, 'auto']} />
              <YAxis type="category" dataKey="circuit" tick={{ fill: '#8b949e', fontSize: 10 }} width={120} />
              <Tooltip content={({ active, payload, label }: any) => {
                if (!active || !payload?.length) return null;
                const d = payload[0]?.payload;
                return (
                  <div className="bg-background border border-[rgba(255,128,0,0.2)] rounded-lg p-2 text-[12px]">
                    <div className="text-foreground font-semibold mb-1">{label}</div>
                    <div className="text-muted-foreground">MAE: <span className="text-foreground font-mono">{d?.mae?.toFixed(3)}s</span></div>
                    <div className="text-muted-foreground">Median AE: <span className="text-foreground font-mono">{d?.median_ae?.toFixed(3)}s</span></div>
                    <div className="text-muted-foreground">Sample: <span className="text-foreground font-mono">{d?.n_laps?.toLocaleString()} laps</span></div>
                  </div>
                );
              }} />
              <Bar dataKey="mae" name="MAE (s)" radius={[0, 4, 4, 0]}>
                {xgbCircuitData.map((d, i) => (
                  <Cell key={i} fill={d.mae < 0.3 ? '#22c55e' : d.mae < 0.4 ? '#FF8000' : d.mae < 0.5 ? '#eab308' : '#ef4444'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Model Confidence */}
      {modelHealth && (
        <div className="bg-card border border-border rounded-lg p-4">
          <div className="flex items-center gap-2 mb-3">
            <Activity className="w-4 h-4 text-cyan-400" />
            <h3 className="text-foreground font-semibold text-sm">Model Confidence</h3>
            <span className="ml-auto text-[11px] text-muted-foreground">OmniDapt drift detection</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {(modelHealth.models || Object.entries(modelHealth).filter(([k]) => k !== 'status' && k !== 'timestamp'))
              .map((item: any) => {
                const name = item.name || item[0];
                const data = item.status ? item : item[1];
                const status = (typeof data === 'object' ? data.status : data) || 'unknown';
                const isOk = status === 'ok' || status === 'healthy' || status === 'active';
                const isDrift = status === 'drift' || status === 'degraded' || status === 'warning';
                const color = isOk ? 'text-green-400 bg-green-400/10 border-green-400/20' :
                  isDrift ? 'text-yellow-400 bg-yellow-400/10 border-yellow-400/20' :
                  'text-red-400 bg-red-400/10 border-red-400/20';
                const label = isOk ? 'OK' : isDrift ? 'DRIFT' : 'DOWN';
                const lastTrained = typeof data === 'object' ? (data.last_trained || data.updated_at || data.timestamp) : null;
                return (
                  <div key={name} className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border ${color}`}>
                    <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: isOk ? '#22c55e' : isDrift ? '#eab308' : '#ef4444' }} />
                    <span className="text-[11px] font-mono text-foreground">{name}</span>
                    <span className="text-[10px] font-semibold">{label}</span>
                    {lastTrained && (
                      <span className="text-[10px] text-muted-foreground">
                        {new Date(lastTrained).toLocaleDateString()}
                      </span>
                    )}
                  </div>
                );
              })}
          </div>
        </div>
      )}

      {/* ELT Driver Deltas */}
      {driverDeltaData.length > 0 && (
        <div className="bg-card border border-border rounded-lg p-4">
          <div className="flex items-center gap-2 mb-3">
            <TrendingUp className="w-4 h-4 text-primary" />
            <h3 className="text-foreground font-semibold text-sm">Driver Pace Deltas (ELT Model)</h3>
            <span className="ml-auto text-[11px] text-muted-foreground">
              Negative = faster than field median — {eltData?.driver_count} drivers, {eltData?.baseline_count} baselines
            </span>
          </div>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={driverDeltaData} margin={{ left: 5, right: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="driver" tick={{ fill: '#8b949e', fontSize: 11 }} />
              <YAxis
                tick={{ fill: '#8b949e', fontSize: 11 }}
                label={{ value: 'Delta (s)', angle: -90, position: 'insideLeft', fill: '#8b949e', fontSize: 11 }}
              />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine y={0} stroke="#8b949e" strokeDasharray="3 3" />
              <Bar dataKey="delta" name="Pace Delta" radius={[4, 4, 0, 0]}>
                {driverDeltaData.map((d: any, i: number) => (
                  <Cell key={i} fill={d.delta < 0 ? '#22c55e' : '#ef4444'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
