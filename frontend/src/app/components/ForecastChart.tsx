import { useState, useEffect } from 'react';
import {
  ResponsiveContainer, ComposedChart, Line, XAxis, YAxis, Tooltip, ReferenceLine,
} from 'recharts';
import { TrendingUp, TrendingDown, Minus, Loader2, AlertTriangle } from 'lucide-react';
import type { FeatureForecast } from './anomalyHelpers';
import KexBriefingCard from './KexBriefingCard';

/* ── Default features per system (always forecastable) ─────────────── */
const DEFAULT_SYSTEM_FEATURES: Record<string, string[]> = {
  'Performance':      ['LapTime_mean', 'Sector1Time_mean', 'Sector2Time_mean', 'Sector3Time_mean'],
  'Speed Traps':      ['SpeedI1_mean', 'SpeedI2_mean', 'SpeedFL_mean', 'SpeedST_mean'],
  'Tyre Management':  ['TyreLife_mean'],
};

/* Map feature column back to its system label */
function featureToSystem(col: string): string {
  const norm = col.toLowerCase();
  for (const [sys, feats] of Object.entries(DEFAULT_SYSTEM_FEATURES)) {
    if (feats.some(f => f.toLowerCase() === norm)) return sys;
  }
  return '';
}

/* Pretty-print a feature column name */
function prettyLabel(col: string): string {
  return col
    .replace(/_mean$/, '')
    .replace(/([A-Z])/g, ' $1')
    .replace(/^[\s]/, '')
    .replace(/I(\d)/, '(I$1)')
    .replace(/_/g, ' ')
    .trim();
}

/* ── Severity cascade order ────────────────────────────────────────── */
const SEVERITY_CASCADE = ['critical', 'high', 'medium', 'low'];

interface ForecastChartProps {
  driverCode: string;
  features?: string[];
}

export function ForecastChart({ driverCode, features }: ForecastChartProps) {
  const [forecasts, setForecasts] = useState<FeatureForecast[]>([]);
  const [loading, setLoading] = useState(!!driverCode);
  const [kex, setKex] = useState<{ text: string; scores?: Record<string, number>; summary?: string; model_used?: string; provider_used?: string; generated_at?: number; grounding_score?: number } | null>(null);
  const [kexLoading, setKexLoading] = useState(false);

  useEffect(() => {
    if (!driverCode) return;

    setForecasts([]);
    setLoading(true);
    fetch('/api/pipeline/anomaly')
      .then(r => r.json())
      .then(async (data) => {
        const drivers = data?.drivers ?? data ?? [];
        const driverData = Array.isArray(drivers)
          ? drivers.find((d: any) => d.code === driverCode || d.driver_code === driverCode)
          : null;

        let featureList = features ?? [];

        if (featureList.length === 0 && driverData) {
          const races = driverData.races ?? [];
          const lastRace = races[races.length - 1];

          if (lastRace?.systems) {
            // ── Severity cascade: try each level, stop when we get features ──
            for (const severity of SEVERITY_CASCADE) {
              for (const sysData of Object.values(lastRace.systems) as any[]) {
                if (sysData.classifier_severity === severity) {
                  const raw = sysData.features;
                  let sysFeatures: string[];
                  if (Array.isArray(raw)) {
                    sysFeatures = raw.slice(0, 2).map((f: any) => f.feature ?? f.label ?? f);
                  } else if (raw && typeof raw === 'object') {
                    sysFeatures = Object.keys(raw).slice(0, 2);
                  } else {
                    sysFeatures = [];
                  }
                  featureList.push(...sysFeatures);
                }
              }
              if (featureList.length > 0) break;
            }
          }
        }

        // ── Fallback: default feature set if nothing from anomaly data ──
        if (featureList.length === 0) {
          featureList = Object.values(DEFAULT_SYSTEM_FEATURES).flat();
        }

        featureList = [...new Set(featureList.map(f => f.includes('_') ? f : `${f}_mean`))];

        const results = await Promise.all(
          featureList.map(col =>
            fetch(`/api/omni/analytics/forecast/${driverCode}?column=${encodeURIComponent(col)}&horizon=12&method=auto`, { method: 'POST' })
              .then(r => r.ok ? r.json() : null)
              .catch(() => null)
          )
        );

        const fcs: FeatureForecast[] = [];
        for (const r of results) {
          if (!r?.values) continue;
          fcs.push({
            column: r.column,
            method: r.method,
            mae: r.mae,
            rmse: r.rmse,
            trend_direction: r.trend_direction,
            trend_pct: r.trend_pct,
            volatility: r.volatility,
            risk_flag: r.risk_flag,
            history: r.history,
            history_timestamps: r.history_timestamps,
            data: r.values.map((val: number, i: number) => ({
              step: r.timestamps?.[i] ?? `+${i + 1}`,
              value: val,
              lower: r.lower_bound?.[i] ?? val,
              upper: r.upper_bound?.[i] ?? val,
            })),
          });
        }
        setForecasts(fcs);
      })
      .catch(() => setForecasts([]))
      .finally(() => setLoading(false));
  }, [driverCode, features]);

  /* ── Fetch KeX forecast briefing ─────────────────────────────────── */
  useEffect(() => {
    if (!driverCode) return;
    setKexLoading(true);
    fetch(`/api/forecast/kex/${driverCode}`, { method: 'POST' })
      .then(r => r.ok ? r.json() : null)
      .then(data => setKex(data?.text ? data : null))
      .catch(() => setKex(null))
      .finally(() => setKexLoading(false));
  }, [driverCode]);

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-[12px] text-muted-foreground py-8 justify-center">
        <Loader2 className="w-4 h-4 animate-spin text-primary" />
        Forecasting features for {driverCode}...
      </div>
    );
  }

  if (forecasts.length === 0) {
    return (
      <div className="text-center py-8 text-sm text-muted-foreground">
        No forecast data available for {driverCode}.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* ── Forecast card grid ──────────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {forecasts.map(fc => {
          const system = featureToSystem(fc.column);
          const label = prettyLabel(fc.column);
          const displayLabel = system ? `${system} (${label})` : label;

          const TrendIcon = fc.trend_direction === 'rising' ? TrendingUp
            : fc.trend_direction === 'falling' ? TrendingDown : Minus;
          const trendColor = fc.trend_direction === 'rising' ? '#3FB950'
            : fc.trend_direction === 'falling' ? '#F85149' : '#8B949E';

          const historyData = (fc.history ?? []).map((v, i) => ({
            step: fc.history_timestamps?.[i] ?? `R${i + 1}`,
            actual: v, value: null as number | null,
            lower: null as number | null, upper: null as number | null,
          }));
          const bridgeStep = fc.history?.length
            ? { step: 'Now', actual: fc.history[fc.history.length - 1], value: fc.data[0]?.value ?? null, lower: null as number | null, upper: null as number | null }
            : null;
          const forecastData = fc.data.map(d => ({
            step: d.step, actual: null as number | null,
            value: d.value, lower: d.lower, upper: d.upper,
          }));
          const chartData = [...historyData, ...(bridgeStep ? [bridgeStep] : []), ...forecastData];

          // Compute Y domain from all values with 10% padding
          const allVals = chartData.flatMap(d => [d.actual, d.value, d.lower, d.upper].filter((v): v is number => v != null));
          const yMin = Math.min(...allVals);
          const yMax = Math.max(...allVals);
          const yPad = (yMax - yMin) * 0.15 || 1;
          const yDomain: [number, number] = [Math.floor(yMin - yPad), Math.ceil(yMax + yPad)];

          // Show ~10-12 evenly spaced X ticks for dense data
          const totalPts = chartData.length;
          const tickInterval = totalPts > 15 ? Math.ceil(totalPts / 10) : 0;

          return (
            <div key={fc.column} className="bg-[#161B22] rounded-lg p-4 border border-[#30363D]">
              {/* Header */}
              <div className="flex items-center justify-between mb-2">
                <span className="text-[14px] font-semibold text-[#E6EDF3]">{displayLabel}</span>
                <div className="flex items-center gap-2">
                  {fc.risk_flag && (
                    <AlertTriangle className="w-3.5 h-3.5 text-primary" />
                  )}
                  <span
                    className="flex items-center gap-1 text-[12px] font-mono font-semibold px-2 py-0.5 rounded-full"
                    style={{ color: trendColor, background: `${trendColor}18` }}
                  >
                    <TrendIcon className="w-3 h-3" />
                    {fc.trend_pct != null ? `${fc.trend_pct > 0 ? '+' : ''}${fc.trend_pct.toFixed(1)}%` : '—'}
                  </span>
                </div>
              </div>

              {/* Chart */}
              <ResponsiveContainer width="100%" height={240}>
                <ComposedChart data={chartData} margin={{ top: 8, right: 12, bottom: 4, left: 8 }}>
                  <XAxis dataKey="step" tick={{ fontSize: 9, fill: '#8B949E' }} axisLine={false} tickLine={false} interval={tickInterval} />
                  <YAxis domain={yDomain} tick={{ fontSize: 9, fill: '#8B949E' }} axisLine={false} tickLine={false} width={50} />
                  <Tooltip
                    contentStyle={{ background: '#161B22', border: '1px solid #30363D', borderRadius: 8, fontSize: 12, color: '#E6EDF3' }}
                    labelStyle={{ color: '#8B949E' }}
                    formatter={(v: any, name: string) => v != null ? [Number(v).toFixed(2), name] : ['-', name]}
                  />
                  <ReferenceLine
                    x="Now"
                    stroke="rgba(255,255,255,0.4)"
                    strokeDasharray="4 3"
                    label={{ value: 'Now', position: 'top', fontSize: 10, fill: '#E6EDF3' }}
                  />
                  <Line type="monotone" dataKey="upper" stroke="rgba(255,128,0,0.35)" strokeWidth={1} strokeDasharray="4 3" dot={false} name="Upper 95%" connectNulls={false} legendType="none" />
                  <Line type="monotone" dataKey="lower" stroke="rgba(255,128,0,0.35)" strokeWidth={1} strokeDasharray="4 3" dot={false} name="Lower 95%" connectNulls={false} legendType="none" />
                  <Line type="monotone" dataKey="value" stroke="#FF8000" strokeWidth={2.5} dot={{ r: 3, fill: '#FF8000' }} name="Forecast" connectNulls={false} />
                  <Line type="monotone" dataKey="actual" stroke="#8B949E" strokeWidth={1.5} dot={false} name="History" connectNulls />
                </ComposedChart>
              </ResponsiveContainer>

              {/* Footer — MAE / RMSE */}
              <div className="flex items-center gap-4 mt-2 text-[11px] text-[#8B949E] font-mono">
                {fc.mae != null && <span>MAE: {fc.mae.toFixed(2)}</span>}
                {fc.rmse != null && <span>RMSE: {fc.rmse.toFixed(2)}</span>}
              </div>
            </div>
          );
        })}
      </div>

      {/* ── KeX Forecast Briefing (Gen UI) ─────────────────────────── */}
      <KexBriefingCard
        title="WISE Forecast Briefing"
        icon="sparkles"
        kex={kex}
        loading={kexLoading}
        loadingText="Generating forecast intelligence…"
      />
    </div>
  );
}
