import { useState, useEffect, useMemo } from 'react';
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, ReferenceLine,
} from 'recharts';
import {
  Loader2, Flag, Timer, Users, TrendingUp, AlertTriangle, Disc, Gauge, Zap,
  ChevronDown, ChevronUp, Shield, Activity, Brain, Cpu,
} from 'lucide-react';
import { strategy } from '../api/local';
import { getOpponentDrivers } from '../api/driverIntel';

// ─── Constants ──────────────────────────────────────────────────────
const compoundColors: Record<string, string> = {
  SOFT: '#ef4444', MEDIUM: '#f59e0b', HARD: '#e8e8f0',
  INTERMEDIATE: '#22c55e', WET: '#3b82f6', UNKNOWN: '#6b7280',
};

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
      <span className="text-[10px] tracking-[0.25em] text-[#FF8000]/60 font-semibold">{label}</span>
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
  const [predLoading, setPredLoading] = useState(false);

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
        <Loader2 className="w-6 h-6 animate-spin text-[#FF8000]" />
        <span className="ml-2 text-muted-foreground">Loading strategy data...</span>
      </div>
    );
  }

  return (
    <div className="space-y-4 pb-6">

      {/* ── Section 1: Session KPIs ── */}
      {session && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-3">
            <KPI
              icon={<Flag className="w-4 h-4 text-[#FF8000]" />}
              label="RACE"
              value={session.meeting}
              detail={`${session.circuit} — ${session.year}`}
            />
            <KPI
              icon={<Gauge className="w-4 h-4 text-[#FF8000]" />}
              label="TOTAL LAPS"
              value={String(session.total_laps)}
              detail={`Session ${session.session_key}`}
            />
            <KPI
              icon={<Users className="w-4 h-4 text-[#FF8000]" />}
              label="DRIVERS"
              value={String(drivers.length)}
              detail={`${simMeta?.n_strategies_evaluated || '—'} strategies evaluated`}
            />
            <KPI
              icon={<Timer className="w-4 h-4 text-[#FF8000]" />}
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
                  isMcL ? 'border-l-2 border-l-[#FF8000]' : ''
                } ${isExpanded ? 'bg-secondary' : 'hover:bg-[#1e2433]'}`}
                onClick={() => setExpandedDriver(isExpanded ? null : d.driver)}
              >
                <div className="flex items-center gap-3 px-3 py-1.5">
                  {/* Driver name */}
                  <div className="w-12 shrink-0">
                    <span className={`text-sm font-mono ${isMcL ? 'text-[#FF8000] font-semibold' : 'text-foreground'}`}>
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
                          className="h-full flex items-center justify-center text-[9px] font-mono"
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
                            <div className="text-[#FF8000]">Pit window: L{s.predicted_pit_window}</div>
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
                        isMcLaren(b.team) ? 'border-l-2 border-l-[#FF8000]' : ''
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <span className={`text-sm font-mono ${isMcLaren(b.team) ? 'text-[#FF8000] font-semibold' : 'text-foreground'}`}>
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
                      isMcLaren(b.team) ? 'border-l-2 border-l-[#FF8000]' : ''
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <span className={`text-[12px] font-mono w-10 ${isMcLaren(b.team) ? 'text-[#FF8000] font-semibold' : 'text-foreground'}`}>
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
                          isMcLaren(s.team) ? 'bg-[#FF8000]/5' : ''
                        }`}
                      >
                        <td className="py-1.5 pr-2 font-mono text-muted-foreground">{i + 1}</td>
                        <td className={`py-1.5 pr-2 font-mono ${isMcLaren(s.team) ? 'text-[#FF8000] font-semibold' : 'text-foreground'}`}>
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
                <Brain className="w-4 h-4 text-[#FF8000]" />
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
            <Brain className="w-4 h-4 text-[#FF8000]" />
            <h3 className="text-foreground font-semibold text-sm">Lap Time Predictor</h3>
            <span className="ml-auto text-[10px] text-muted-foreground">XGBoost inference &middot; MAE {xgbData.metadata.validation?.mae?.toFixed(2)}s</span>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-4">
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
            <div className="flex items-end">
              <button
                type="button"
                onClick={() => {
                  setPredLoading(true);
                  strategy.predictLap({
                    circuit: predCircuit, driver_code: predDriver, compound: predCompound,
                    lap_start: predLapStart, lap_end: predLapEnd,
                  }).then(setPredResult).catch(() => setPredResult(null)).finally(() => setPredLoading(false));
                }}
                disabled={predLoading}
                className="w-full bg-[#FF8000] hover:bg-[#FF8000]/80 text-black font-semibold text-[12px] rounded-lg px-3 py-1.5 transition-colors disabled:opacity-50"
              >
                {predLoading ? 'Predicting...' : 'Predict'}
              </button>
            </div>
          </div>

          {/* Prediction results */}
          {predResult?.predictions && (
            <div>
              <div className="flex items-center gap-3 mb-2">
                <span className="text-[11px] text-muted-foreground">
                  {predResult.driver} at {predResult.circuit.replace(' Grand Prix', '')} &middot; {predResult.compound} &middot; Baseline: {predResult.baseline_pace_s}s
                </span>
                <span className="text-[10px] text-muted-foreground">
                  Weather: {predResult.weather?.air_temp_c}°C air / {predResult.weather?.track_temp_c}°C track / {predResult.weather?.humidity_pct}% hum
                </span>
              </div>
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={predResult.predictions} margin={{ left: 5, right: 10, top: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="lap" tick={{ fill: '#8b949e', fontSize: 11 }} label={{ value: 'Lap', position: 'insideBottom', offset: -2, fill: '#8b949e', fontSize: 10 }} />
                  <YAxis tick={{ fill: '#8b949e', fontSize: 11 }} domain={['auto', 'auto']} label={{ value: 'Lap Time (s)', angle: -90, position: 'insideLeft', fill: '#8b949e', fontSize: 10 }} />
                  <Tooltip content={({ active, payload }: any) => {
                    if (!active || !payload?.length) return null;
                    const d = payload[0]?.payload;
                    return (
                      <div className="bg-background border border-[rgba(255,128,0,0.2)] rounded-lg p-2 text-[12px]">
                        <div className="text-foreground font-semibold">Lap {d?.lap}</div>
                        <div className="text-muted-foreground">Predicted: <span className="text-foreground font-mono">{d?.predicted_s?.toFixed(3)}s</span></div>
                        <div className="text-muted-foreground">Tyre Life: <span className="text-foreground font-mono">{d?.tyre_life}</span></div>
                        <div className="text-muted-foreground">Deg Delta: <span className="text-foreground font-mono">{d?.deg_delta?.toFixed(3)}s</span></div>
                      </div>
                    );
                  }} />
                  <ReferenceLine y={predResult.baseline_pace_s} stroke="#8b949e" strokeDasharray="3 3" label={{ value: 'Baseline', fill: '#8b949e', fontSize: 10 }} />
                  <Line type="monotone" dataKey="predicted_s" stroke="#FF8000" strokeWidth={2} dot={false} name="Predicted" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}

      {/* XGBoost Circuit Accuracy */}
      {xgbCircuitData.length > 0 && (
        <div className="bg-card border border-border rounded-lg p-4">
          <div className="flex items-center gap-2 mb-3">
            <Flag className="w-4 h-4 text-[#FF8000]" />
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
                      <span className="text-[9px] text-muted-foreground">
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
            <TrendingUp className="w-4 h-4 text-[#FF8000]" />
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
