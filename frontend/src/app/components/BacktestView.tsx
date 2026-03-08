import { useState, useEffect } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
} from 'recharts';
import {
  FlaskConical, Loader2, Target, AlertTriangle,
  ChevronDown, ChevronUp, Play, Shield,
  Zap, Timer, TrendingDown, Layers, RotateCcw,
} from 'lucide-react';
import KexBriefingCard from './KexBriefingCard';

/* ─── Types ──────────────────────────────────────────────────────── */

interface SystemHealth {
  health: number;
  level: string;
  score_mean: number;
  vote_count: number;
  total_models: number;
}

interface BacktestEntry {
  round: number;
  race_name: string;
  driver_code: string;
  constructor_name: string;
  constructor_id: string;
  predicted_overall_health: number;
  predicted_systems: Record<string, SystemHealth>;
  predicted_trends: Record<string, string>;
  predicted_risk: boolean;
  flagged_systems: string[];
  degrading_systems: string[];
  races_in_training: number;
  train_seasons: number[];
  train_accuracy: number | null;
  cv_std: number | null;
  // Multi-model fields
  elt_predicted_pace: number | null;
  elt_baseline: number | null;
  elt_driver_advantage: number | null;
  elt_confidence_std: number | null;
  strategy_predicted: string | null;
  strategy_time_delta_s: number | null;
  strategy_match: boolean | null;
  cliff_warnings: number;
  composite_risk: number | null;
  composite_risk_level: string | null;
  composite_signals: Record<string, number>;
  composite_weights: Record<string, number>;
  // Actual
  actual_grid: number;
  actual_position: number;
  actual_positions_gained: number;
  actual_status: string;
  actual_points: number;
  actual_outcome: string;
  actual_is_dnf: boolean;
  prediction_correct: boolean;
}

interface CaseStudy {
  round: number;
  race_name: string;
  driver_code: string;
  constructor_name: string;
  score: number;
  reasons: string[];
  predicted_health: number;
  predicted_risk: boolean;
  flagged_systems: string[];
  degrading_systems: string[];
  actual_grid: number;
  actual_position: number;
  actual_positions_gained: number;
  actual_status: string;
  actual_outcome: string;
  actual_points: number;
  actual_is_dnf: boolean;
  // Multi-model signals
  composite_risk: number | null;
  composite_risk_level: string | null;
  composite_signals: Record<string, number>;
  elt_predicted_pace: number | null;
  elt_driver_advantage: number | null;
  strategy_predicted: string | null;
  strategy_match: boolean | null;
  strategy_time_delta_s: number | null;
  cliff_warnings: number;
  predicted_systems: Record<string, SystemHealth>;
}

interface ConfusionMatrix {
  true_positive: number;
  false_positive: number;
  false_negative: number;
  true_negative: number;
}

interface BacktestMetrics {
  total_predictions: number;
  correct: number;
  accuracy: number;
  confusion_matrix: ConfusionMatrix;
  precision: number;
  recall: number;
  f1_score: number;
  outcome_distribution: Record<string, number>;
  team_accuracy: Record<string, number>;
  // Multi-model
  strategy_match_rate: number | null;
  avg_strategy_delta_s: number | null;
  composite_risk_distribution: Record<string, number>;
  avg_composite_risk: number | null;
  cliff_warnings_total: number;
  elt_coverage: number;
  models_active: Record<string, boolean>;
}

interface BacktestData {
  season: number;
  races_evaluated: number;
  total_predictions: number;
  metrics: BacktestMetrics;
  case_studies: CaseStudy[];
  system_correlations: Record<string, { flag_precision: number; flagged_count: number }>;
  generated_at: string;
  results?: BacktestEntry[];
}

/* ─── Constants ──────────────────────────────────────────────────── */

const teamColors: Record<string, string> = {
  'red_bull': '#3671C6', 'mclaren': '#FF8000', 'ferrari': '#E8002D',
  'mercedes': '#27F4D2', 'aston_martin': '#229971', 'alpine': '#FF87BC',
  'williams': '#64C4FF', 'rb': '#6692FF', 'sauber': '#52E252', 'haas': '#B6BABD',
};

const teamDisplayNames: Record<string, string> = {
  'red_bull': 'Red Bull', 'mclaren': 'McLaren', 'ferrari': 'Ferrari',
  'mercedes': 'Mercedes', 'aston_martin': 'Aston Martin', 'alpine': 'Alpine',
  'williams': 'Williams', 'rb': 'RB', 'sauber': 'Kick Sauber', 'haas': 'Haas',
};

const outcomeColors: Record<string, string> = {
  normal: '#888', underperformance: '#f59e0b', major_underperformance: '#ef4444',
  outperformance: '#22c55e', major_outperformance: '#05DF72',
  lapped: '#ef4444', dnf_mechanical: '#dc2626', dnf_other: '#fb923c',
};

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-[#0D1117] border border-[rgba(255,128,0,0.2)] rounded-lg p-2 text-[12px]">
      <div className="text-muted-foreground mb-1">{label}</div>
      {payload.map((e: any, i: number) => (
        <div key={i} className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full" style={{ backgroundColor: e.color }} />
          <span className="text-muted-foreground">{e.name}:</span>
          <span className="text-foreground font-mono">
            {typeof e.value === 'number' ? e.value.toFixed(1) : e.value}
          </span>
        </div>
      ))}
    </div>
  );
};

/* ─── Metric Card ────────────────────────────────────────────────── */

function MetricCard({ label, value, sub, color }: { label: string; value: string; sub?: string; color?: string }) {
  return (
    <div className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.08)] rounded-xl p-4 text-center">
      <div className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">{label}</div>
      <div className="text-2xl font-bold font-mono" style={{ color: color || '#FF8000' }}>{value}</div>
      {sub && <div className="text-[10px] text-muted-foreground mt-0.5">{sub}</div>}
    </div>
  );
}

/* ─── Confusion Matrix ───────────────────────────────────────────── */

function ConfusionMatrixDisplay({ cm }: { cm: ConfusionMatrix }) {
  const total = cm.true_positive + cm.false_positive + cm.false_negative + cm.true_negative;
  const cell = (val: number, label: string, good: boolean) => (
    <div className={`rounded-lg p-3 text-center border ${good ? 'border-green-500/20 bg-green-500/5' : 'border-red-500/20 bg-red-500/5'}`}>
      <div className={`text-xl font-bold font-mono ${good ? 'text-green-400' : 'text-red-400'}`}>{val}</div>
      <div className="text-[9px] text-muted-foreground mt-0.5">{label}</div>
      <div className="text-[9px] text-muted-foreground/60">{total > 0 ? ((val / total) * 100).toFixed(0) : 0}%</div>
    </div>
  );
  return (
    <div className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.08)] rounded-xl p-4">
      <div className="text-[10px] text-muted-foreground uppercase tracking-wider mb-3">Confusion Matrix</div>
      <div className="grid grid-cols-2 gap-2">
        {cell(cm.true_positive, 'True Positive', true)}
        {cell(cm.false_positive, 'False Positive', false)}
        {cell(cm.false_negative, 'False Negative', false)}
        {cell(cm.true_negative, 'True Negative', true)}
      </div>
      <div className="grid grid-cols-2 gap-2 mt-2 text-[9px] text-muted-foreground/60">
        <div className="text-center">Flagged risk, bad outcome</div>
        <div className="text-center">Flagged risk, normal outcome</div>
        <div className="text-center">Missed, bad outcome</div>
        <div className="text-center">No flag, normal outcome</div>
      </div>
    </div>
  );
}

/* ─── Case Study Card ────────────────────────────────────────────── */

const BAD_OUTCOMES = new Set(['dnf_mechanical', 'dnf_other', 'lapped', 'major_underperformance', 'underperformance']);

function CaseStudyCard({ cs, rank, insight }: { cs: CaseStudy; rank: number; insight?: string }) {
  const [expanded, setExpanded] = useState(false);
  const teamColor = teamColors[cs.constructor_name?.toLowerCase().replace(/\s/g, '_')] || '#888';
  const isHit = cs.predicted_risk && BAD_OUTCOMES.has(cs.actual_outcome);
  const isMiss = !cs.predicted_risk && BAD_OUTCOMES.has(cs.actual_outcome);
  const gained = cs.actual_positions_gained ?? 0;

  const riskColor = (cs.composite_risk_level === 'critical' || cs.composite_risk_level === 'high')
    ? '#ef4444' : cs.composite_risk_level === 'medium' ? '#f59e0b' : '#05DF72';

  return (
    <div className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.08)] rounded-xl p-4">
      {/* Header row */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono text-muted-foreground/50">#{rank}</span>
          <span className="text-sm font-medium text-foreground">R{cs.round} {cs.race_name}</span>
        </div>
        <div className="flex items-center gap-1.5">
          {isHit && <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded bg-green-500/15 text-green-400">HIT</span>}
          {isMiss && <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded bg-red-500/15 text-red-400">MISS</span>}
          <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-[#FF8000]/10 text-[#FF8000]">
            Score: {cs.score}
          </span>
        </div>
      </div>

      {/* Driver + team + result summary */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-3">
          <span className="text-sm font-bold" style={{ color: teamColor }}>{cs.driver_code}</span>
          <span className="text-[11px] text-muted-foreground">{teamDisplayNames[cs.constructor_name?.toLowerCase().replace(/\s/g, '_')] || cs.constructor_name}</span>
        </div>
        <div className="flex items-center gap-2">
          {cs.actual_points > 0 && (
            <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-[#05DF72]/10 text-[#05DF72] border border-[#05DF72]/20">
              {cs.actual_points} pts
            </span>
          )}
          {cs.actual_is_dnf && (
            <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-red-500/10 text-red-400 border border-red-500/20">
              DNF
            </span>
          )}
        </div>
      </div>

      {/* Prediction vs Actual */}
      <div className="grid grid-cols-2 gap-3 text-[11px]">
        <div>
          <div className="text-muted-foreground/60 text-[9px] uppercase mb-0.5">Prediction</div>
          <div className="flex items-center gap-1.5">
            <Shield className="w-3 h-3 text-muted-foreground" />
            <span className="text-muted-foreground">Health:</span>
            <span className={`font-mono font-medium ${cs.predicted_health >= 70 ? 'text-green-400' : cs.predicted_health >= 50 ? 'text-amber-400' : 'text-red-400'}`}>
              {cs.predicted_health}%
            </span>
          </div>
          <div className="flex items-center gap-1.5 mt-0.5">
            <AlertTriangle className="w-3 h-3 text-muted-foreground" />
            <span className="text-muted-foreground">Risk:</span>
            <span className={cs.predicted_risk ? 'text-red-400 font-medium' : 'text-green-400'}>
              {cs.predicted_risk ? 'YES' : 'No'}
            </span>
          </div>
        </div>
        <div>
          <div className="text-muted-foreground/60 text-[9px] uppercase mb-0.5">Actual</div>
          <div className="flex items-center gap-1.5">
            <Target className="w-3 h-3 text-muted-foreground" />
            <span className="text-muted-foreground">Grid {cs.actual_grid}</span>
            <span className="text-muted-foreground/40">&rarr;</span>
            <span className="font-mono font-medium text-foreground">P{cs.actual_position}</span>
            <span className={`text-[9px] font-mono ${gained > 0 ? 'text-green-400' : gained < 0 ? 'text-red-400' : 'text-muted-foreground/50'}`}>
              {gained > 0 ? `+${gained}` : gained === 0 ? '=' : gained}
            </span>
          </div>
          <div className="flex items-center gap-1.5 mt-0.5">
            <span className="text-[9px] px-1 py-0.5 rounded font-mono" style={{
              color: outcomeColors[cs.actual_outcome] || '#888',
              background: `${outcomeColors[cs.actual_outcome] || '#888'}15`,
            }}>
              {cs.actual_outcome.replace(/_/g, ' ')}
            </span>
          </div>
        </div>
      </div>

      {/* Composite Risk Bar */}
      {cs.composite_risk != null && (
        <div className="mt-2 bg-[#0D1117] rounded-lg p-2">
          <div className="flex items-center justify-between mb-1">
            <span className="text-[9px] text-muted-foreground/60 uppercase tracking-wider flex items-center gap-1">
              <Layers className="w-2.5 h-2.5" /> Composite Risk
            </span>
            <span className="text-[10px] font-mono font-medium" style={{ color: riskColor }}>
              {cs.composite_risk.toFixed(0)} <span className="text-[8px] capitalize">({cs.composite_risk_level})</span>
            </span>
          </div>
          <div className="h-1.5 rounded-full bg-[#1A1F2E] overflow-hidden">
            <div
              className="h-full rounded-full transition-all"
              style={{ width: `${Math.min(cs.composite_risk, 100)}%`, backgroundColor: riskColor }}
            />
          </div>
          {/* Signal breakdown */}
          {cs.composite_signals && Object.keys(cs.composite_signals).length > 0 && (
            <div className="flex gap-2 mt-1.5 flex-wrap">
              {Object.entries(cs.composite_signals).map(([signal, value]) => {
                const sColor = value >= 60 ? '#ef4444' : value >= 30 ? '#f59e0b' : '#05DF72';
                return (
                  <span key={signal} className="text-[8px] font-mono" style={{ color: sColor }}>
                    {signal}: {typeof value === 'number' ? value.toFixed(0) : value}
                  </span>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* Multi-model details row */}
      <div className="flex flex-wrap gap-1.5 mt-2">
        {cs.strategy_predicted && (
          <span className={`text-[8px] font-mono px-1.5 py-0.5 rounded border ${
            cs.strategy_match ? 'bg-green-500/10 text-green-400 border-green-500/20' : 'bg-amber-500/10 text-amber-400 border-amber-500/20'
          }`}>
            Strategy: {cs.strategy_predicted}{cs.strategy_match ? ' ✓' : ' ✗'}
            {cs.strategy_time_delta_s != null && ` (${cs.strategy_time_delta_s > 0 ? '+' : ''}${cs.strategy_time_delta_s.toFixed(1)}s)`}
          </span>
        )}
        {cs.elt_predicted_pace != null && (
          <span className="text-[8px] font-mono px-1.5 py-0.5 rounded bg-[#64C4FF]/10 text-[#64C4FF] border border-[#64C4FF]/20">
            ELT Pace: {cs.elt_predicted_pace.toFixed(2)}s
            {cs.elt_driver_advantage != null && ` (${cs.elt_driver_advantage > 0 ? '+' : ''}${cs.elt_driver_advantage.toFixed(2)}s adv)`}
          </span>
        )}
        {(cs.cliff_warnings ?? 0) > 0 && (
          <span className="text-[8px] font-mono px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20">
            <TrendingDown className="w-2.5 h-2.5 inline mr-0.5" />{cs.cliff_warnings} cliff warn
          </span>
        )}
      </div>

      {/* System flags */}
      {(cs.flagged_systems.length > 0 || cs.degrading_systems.length > 0) && (
        <div className="flex flex-wrap gap-1 mt-2">
          {cs.flagged_systems.map(s => (
            <span key={s} className="text-[9px] px-1.5 py-0.5 rounded bg-red-500/10 text-red-400 border border-red-500/20">
              {s}
            </span>
          ))}
          {cs.degrading_systems.map(s => (
            <span key={s} className="text-[9px] px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20">
              {s} (degrading)
            </span>
          ))}
        </div>
      )}

      {/* Expand reasons */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1 text-[10px] text-[#FF8000] hover:text-[#FF9933] mt-2 transition-colors"
      >
        {expanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
        {expanded ? 'Hide details' : `${cs.reasons.length} reason${cs.reasons.length !== 1 ? 's' : ''} + system detail`}
      </button>

      {expanded && (
        <div className="mt-1.5 space-y-2">
          {/* Reasons */}
          <div className="pl-3 border-l border-[rgba(255,128,0,0.1)] space-y-0.5">
            {cs.reasons.map((r, i) => (
              <div key={i} className="text-[10px] text-muted-foreground">
                <span className="text-[#FF8000] mr-1">&rarr;</span> {r}
              </div>
            ))}
          </div>

          {/* System health breakdown */}
          {cs.predicted_systems && Object.keys(cs.predicted_systems).length > 0 && (
            <div className="bg-[#0D1117] rounded-lg p-2">
              <div className="text-[9px] text-muted-foreground/60 uppercase tracking-wider mb-1.5">System Health</div>
              <div className="grid grid-cols-2 gap-x-3 gap-y-1">
                {Object.entries(cs.predicted_systems).map(([sys, sh]) => {
                  const h = sh?.health ?? 0;
                  const sysColor = h >= 70 ? '#05DF72' : h >= 50 ? '#f59e0b' : '#ef4444';
                  return (
                    <div key={sys} className="flex items-center justify-between">
                      <span className="text-[9px] text-muted-foreground truncate">{sys}</span>
                      <span className="text-[9px] font-mono font-medium" style={{ color: sysColor }}>{h}%</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* KeX per-case insight */}
          {insight && (
            <div className="bg-[#0D1117] rounded-lg p-2 border-l-2 border-[#FF8000]/30">
              <div className="text-[9px] text-muted-foreground/60 uppercase tracking-wider mb-1 flex items-center gap-1">
                KeX Insight
              </div>
              <p className="text-[11px] text-muted-foreground leading-relaxed whitespace-pre-line">{insight}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ─── Main Component ─────────────────────────────────────────────── */

export function BacktestView() {
  const [data, setData] = useState<BacktestData | null>(null);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [kex, setKex] = useState<{ text: string; scores?: Record<string, number>; summary?: string; model_used?: string; provider_used?: string; generated_at?: number; case_insights?: { round: number; driver_code: string; race_name: string; insight: string }[] } | null>(null);
  const [kexLoading, setKexLoading] = useState(false);

  // Load latest results on mount
  useEffect(() => {
    fetch('/api/local/backtest/results')
      .then(r => r.ok ? r.json() : null)
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  }, []);

  // Load KeX briefing when data is available
  useEffect(() => {
    if (!data) return;
    setKexLoading(true);
    fetch('/api/local/backtest/kex', { method: 'POST' })
      .then(r => r.ok ? r.json() : null)
      .then(setKex)
      .catch(() => setKex(null))
      .finally(() => setKexLoading(false));
  }, [data]);

  const runBacktest = async (force = false) => {
    setRunning(true);
    setError(null);
    try {
      const params = new URLSearchParams({ team: 'McLaren' });
      if (force) params.set('force', 'true');
      const r = await fetch(`/api/local/backtest/run?${params}`, { method: 'POST' });
      if (!r.ok) throw new Error(`Backtest failed: ${r.status}`);
      const result = await r.json();
      setData(result);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setRunning(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20 gap-3">
        <Loader2 className="w-5 h-5 text-[#FF8000] animate-spin" />
        <span className="text-muted-foreground text-sm">Loading backtest results...</span>
      </div>
    );
  }

  const m = data?.metrics;

  // Prepare chart data
  const outcomeData = m?.outcome_distribution
    ? Object.entries(m.outcome_distribution).map(([name, value]) => ({
        name: name.replace(/_/g, ' '),
        value,
        fill: outcomeColors[name] || '#888',
      }))
    : [];

  const teamAccData = m?.team_accuracy
    ? Object.entries(m.team_accuracy)
        .sort(([, a], [, b]) => b - a)
        .map(([team, acc]) => ({
          team: teamDisplayNames[team.toLowerCase().replace(/\s/g, '_')] || team,
          accuracy: acc,
          fill: teamColors[team.toLowerCase().replace(/\s/g, '_')] || '#888',
        }))
    : [];

  const systemData = data?.system_correlations
    ? Object.entries(data.system_correlations)
        .filter(([, v]) => v.flagged_count > 0)
        .map(([sys, v]) => ({
          system: sys,
          precision: v.flag_precision,
          flags: v.flagged_count,
        }))
    : [];

  return (
    <div className="space-y-4">
      {/* ── Header + Controls ── */}
      <div className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded-xl p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <FlaskConical className="w-5 h-5 text-[#FF8000]" />
            <div>
              <h2 className="text-sm font-semibold text-foreground">McLaren Backtest Framework</h2>
              <p className="text-[11px] text-muted-foreground">
                {data?.results?.[0]?.train_seasons?.length
                  ? `Trained on ${data.results[0].train_seasons[0]}–${data.results[0].train_seasons[data.results[0].train_seasons.length - 1]} telemetry — replayed against ${data.season} race outcomes`
                  : `Replayed against ${data?.season || 2024} race outcomes`}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => runBacktest(false)}
              disabled={running}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[11px] font-medium bg-[#FF8000] text-[#0D1117] hover:bg-[#FF9933] disabled:opacity-50 transition-colors"
            >
              {running ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
              {running ? 'Running...' : 'Run Backtest'}
            </button>
            <button
              onClick={() => runBacktest(true)}
              disabled={running}
              title="Force retrain all models from scratch"
              className="flex items-center gap-1.5 px-2 py-1.5 rounded-lg text-[11px] font-medium bg-zinc-800 text-zinc-400 hover:bg-zinc-700 hover:text-zinc-200 disabled:opacity-50 transition-colors border border-zinc-700"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
        {error && (
          <div className="mt-2 text-[11px] text-red-400 bg-red-500/10 px-3 py-1.5 rounded-lg">{error}</div>
        )}
        {data?.generated_at && (
          <div className="mt-2 flex items-center gap-3 text-[9px] text-muted-foreground/50 font-mono">
            <span>Last run: {new Date(data.generated_at).toLocaleString()}</span>
            <span className="px-1.5 py-0.5 rounded bg-[#FF8000]/10 text-[#FF8000]">
              Train: {data.results?.[0]?.train_seasons?.length
                ? `${data.results[0].train_seasons[0]}–${data.results[0].train_seasons[data.results[0].train_seasons.length - 1]}`
                : '2018–2023'} → Test: {data.season}
            </span>
            {data.results?.[0]?.races_in_training != null && (
              <span>{data.results[0].races_in_training} races in training set</span>
            )}
            {data.results?.[0]?.train_accuracy != null && (
              <span>CV accuracy: {data.results[0].train_accuracy}%{data.results[0]?.cv_std ? ` ±${data.results[0].cv_std}%` : ''}</span>
            )}
            <span className="text-[#FF8000]">XGBoost + SHAP</span>
          </div>
        )}
      </div>

      {!data && !running && (
        <div className="text-center py-16">
          <FlaskConical className="w-10 h-10 text-muted-foreground/30 mx-auto mb-3" />
          <p className="text-sm text-muted-foreground">No backtest results yet</p>
          <p className="text-[11px] text-muted-foreground/60 mt-1">Run a backtest to replay 2024 races and validate predictions</p>
        </div>
      )}

      {m && data && (
        <>
          {/* ── Key Metrics ── */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <MetricCard
              label="Accuracy"
              value={`${m.accuracy}%`}
              sub={`${m.correct}/${m.total_predictions} correct`}
              color={m.accuracy >= 60 ? '#05DF72' : m.accuracy >= 40 ? '#FF8000' : '#ef4444'}
            />
            <MetricCard
              label="Precision"
              value={`${m.precision}%`}
              sub="Risk flag accuracy"
              color={m.precision >= 50 ? '#05DF72' : '#FF8000'}
            />
            <MetricCard
              label="Recall"
              value={`${m.recall}%`}
              sub="Bad outcomes caught"
              color={m.recall >= 50 ? '#05DF72' : '#FF8000'}
            />
            <MetricCard
              label="F1 Score"
              value={`${m.f1_score}%`}
              sub={`${data.races_evaluated} races evaluated`}
            />
          </div>

          {/* ── Multi-Model Signals ── */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <MetricCard
              label="Avg Composite Risk"
              value={m.avg_composite_risk != null ? `${m.avg_composite_risk}` : '—'}
              sub="0 = safe, 100 = critical"
              color={
                (m.avg_composite_risk ?? 0) >= 50 ? '#ef4444'
                : (m.avg_composite_risk ?? 0) >= 30 ? '#FF8000'
                : '#05DF72'
              }
            />
            <MetricCard
              label="Strategy Match"
              value={m.strategy_match_rate != null ? `${m.strategy_match_rate}%` : '—'}
              sub={m.avg_strategy_delta_s != null ? `Avg delta: ${m.avg_strategy_delta_s > 0 ? '+' : ''}${m.avg_strategy_delta_s}s` : 'No data'}
            />
            <MetricCard
              label="ELT Coverage"
              value={`${m.elt_coverage}%`}
              sub="Drivers with pace model"
              color="#64C4FF"
            />
            <MetricCard
              label="Cliff Warnings"
              value={`${m.cliff_warnings_total}`}
              sub="Stints past tyre cliff"
              color={m.cliff_warnings_total > 0 ? '#f59e0b' : '#05DF72'}
            />
          </div>

          {/* ── Composite Risk Distribution ── */}
          {m.composite_risk_distribution && Object.keys(m.composite_risk_distribution).length > 0 && (
            <div className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.08)] rounded-xl p-4">
              <div className="text-[10px] text-muted-foreground uppercase tracking-wider mb-3 flex items-center gap-2">
                <Layers className="w-3.5 h-3.5 text-[#FF8000]" />
                Composite Risk Distribution
              </div>
              <div className="flex items-center gap-3">
                {(['normal', 'low', 'medium', 'high', 'critical'] as const).map(level => {
                  const count = m.composite_risk_distribution[level] || 0;
                  const colors: Record<string, string> = {
                    normal: '#05DF72', low: '#22c55e', medium: '#f59e0b', high: '#ef4444', critical: '#dc2626',
                  };
                  const total = m.total_predictions;
                  const pct = total > 0 ? (count / total) * 100 : 0;
                  return (
                    <div key={level} className="flex-1 text-center">
                      <div className="text-lg font-bold font-mono" style={{ color: colors[level] }}>{count}</div>
                      <div className="text-[9px] text-muted-foreground capitalize">{level}</div>
                      <div className="mt-1 h-1.5 rounded-full bg-[#0D1117] overflow-hidden">
                        <div className="h-full rounded-full transition-all" style={{ width: `${pct}%`, backgroundColor: colors[level] }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* ── Active Models ── */}
          {m.models_active && (
            <div className="flex items-center gap-2 px-1">
              <span className="text-[9px] text-muted-foreground/50 uppercase tracking-wider">Models:</span>
              {Object.entries(m.models_active).map(([model, active]) => (
                <span
                  key={model}
                  className={`text-[9px] px-2 py-0.5 rounded-full font-mono ${
                    active ? 'bg-green-500/10 text-green-400 border border-green-500/20' : 'bg-[#1A1F2E] text-muted-foreground/40 border border-[rgba(255,128,0,0.05)]'
                  }`}
                >
                  {model === 'anomaly' && <Zap className="w-2.5 h-2.5 inline mr-1" />}
                  {model === 'elt' && <Timer className="w-2.5 h-2.5 inline mr-1" />}
                  {model === 'strategy' && <Target className="w-2.5 h-2.5 inline mr-1" />}
                  {model === 'cliff' && <TrendingDown className="w-2.5 h-2.5 inline mr-1" />}
                  {model}
                </span>
              ))}
            </div>
          )}

          {/* ── Confusion Matrix + Outcome Distribution ── */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <ConfusionMatrixDisplay cm={m.confusion_matrix} />

            <div className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.08)] rounded-xl p-4">
              <div className="text-[10px] text-muted-foreground uppercase tracking-wider mb-3">Outcome Distribution</div>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={outcomeData} layout="vertical" margin={{ left: 10, right: 10 }}>
                  <XAxis type="number" tick={{ fontSize: 9, fill: '#555' }} />
                  <YAxis type="category" dataKey="name" tick={{ fontSize: 9, fill: '#888' }} width={120} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                    {outcomeData.map((d, i) => <Cell key={i} fill={d.fill} fillOpacity={0.7} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* ── Team Accuracy + System Prediction Value ── */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {teamAccData.length > 0 && (
              <div className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.08)] rounded-xl p-4">
                <div className="text-[10px] text-muted-foreground uppercase tracking-wider mb-3">Per-Team Accuracy</div>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={teamAccData} margin={{ left: 0, right: 10 }}>
                    <XAxis dataKey="team" tick={{ fontSize: 8, fill: '#888' }} interval={0} angle={-30} textAnchor="end" height={60} />
                    <YAxis domain={[0, 100]} tick={{ fontSize: 9, fill: '#555' }} />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey="accuracy" name="Accuracy %" radius={[4, 4, 0, 0]}>
                      {teamAccData.map((d, i) => <Cell key={i} fill={d.fill} fillOpacity={0.7} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            {systemData.length > 0 && (
              <div className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.08)] rounded-xl p-4">
                <div className="text-[10px] text-muted-foreground uppercase tracking-wider mb-3">System Prediction Value</div>
                <ResponsiveContainer width="100%" height={220}>
                  <RadarChart data={systemData} cx="50%" cy="50%" outerRadius="68%">
                    <PolarGrid stroke="rgba(255,128,0,0.1)" />
                    <PolarAngleAxis dataKey="system" tick={{ fontSize: 9, fill: '#888' }} />
                    <PolarRadiusAxis domain={[0, 100]} tick={{ fontSize: 8, fill: '#555' }} />
                    <Tooltip content={<CustomTooltip />} />
                    <Radar dataKey="precision" name="Flag Precision %" stroke="#FF8000" fill="#FF8000" fillOpacity={0.2} strokeWidth={1.5} dot={{ r: 3, fill: '#FF8000' }} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>

          {/* ── Case Studies ── */}
          {data.case_studies.length > 0 && (
            <div>
              <div className="flex items-center gap-2 mb-3">
                <Target className="w-4 h-4 text-[#FF8000]" />
                <h3 className="text-sm font-semibold text-foreground">Case Study Candidates</h3>
                <span className="text-[9px] text-muted-foreground bg-[#1A1F2E] px-2 py-0.5 rounded-full">
                  Top {data.case_studies.length}
                </span>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {data.case_studies.map((cs, i) => (
                  <CaseStudyCard
                    key={`${cs.round}-${cs.driver_code}`}
                    cs={cs}
                    rank={i + 1}
                    insight={kex?.case_insights?.find(ci => ci.round === cs.round && ci.driver_code === cs.driver_code)?.insight}
                  />
                ))}
              </div>
            </div>
          )}

          {/* ── KeX Intelligence Briefing ── */}
          <KexBriefingCard
            title="Backtest Intelligence Briefing"
            icon="brain"
            kex={kex}
            loading={kexLoading}
            loadingText="Generating backtest intelligence..."
          />
        </>
      )}
    </div>
  );
}
