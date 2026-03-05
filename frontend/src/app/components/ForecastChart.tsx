import { useState, useEffect } from 'react';
import {
  ResponsiveContainer, ComposedChart, Area, Line, XAxis, YAxis, Tooltip, ReferenceLine,
} from 'recharts';
import { TrendingUp, TrendingDown, Minus, Loader2 } from 'lucide-react';
import type { FeatureForecast } from './anomalyHelpers';

interface ForecastChartProps {
  driverCode: string;
  /** Pre-loaded forecasts (if available from parent). If empty, will fetch on-demand. */
  prefetchedForecasts?: FeatureForecast[];
  /** Features to forecast — if not provided, fetches critical/warning features from anomaly data. */
  features?: string[];
}

export function ForecastChart({ driverCode, prefetchedForecasts, features }: ForecastChartProps) {
  const [forecasts, setForecasts] = useState<FeatureForecast[]>(prefetchedForecasts ?? []);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (prefetchedForecasts && prefetchedForecasts.length > 0) {
      setForecasts(prefetchedForecasts);
      return;
    }
    if (!driverCode) return;

    // Fetch anomaly data to find critical/warning features, then forecast them
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
          // Extract top features from critical/warning systems
          const races = driverData.races ?? [];
          const lastRace = races[races.length - 1];
          if (lastRace?.systems) {
            for (const sysData of Object.values(lastRace.systems) as any[]) {
              if (sysData.classifier_severity === 'high' || sysData.classifier_severity === 'critical' ||
                  sysData.classifier_severity === 'medium') {
                const sysFeatures = (sysData.features ?? []).slice(0, 2).map((f: any) => f.feature ?? f.label ?? f);
                featureList.push(...sysFeatures);
              }
            }
          }
        }

        featureList = [...new Set(featureList.map(f => f.includes('_') ? f : `${f}_mean`))];
        if (featureList.length === 0) {
          setForecasts([]);
          setLoading(false);
          return;
        }

        const results = await Promise.all(
          featureList.map(col =>
            fetch(`/api/omni/analytics/forecast/${driverCode}?column=${encodeURIComponent(col)}&horizon=5&method=ets`, { method: 'POST' })
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
  }, [driverCode, prefetchedForecasts, features]);

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-[12px] text-muted-foreground py-8 justify-center">
        <Loader2 className="w-4 h-4 animate-spin text-[#FF8000]" />
        Forecasting anomaly features for {driverCode}...
      </div>
    );
  }

  if (forecasts.length === 0) {
    return (
      <div className="text-center py-8 text-sm text-muted-foreground">
        No critical or warning features to forecast for {driverCode}.
      </div>
    );
  }

  return (
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
  );
}
