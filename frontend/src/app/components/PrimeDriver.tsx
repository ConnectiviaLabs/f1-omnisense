import { useState, useEffect, useMemo } from 'react';
import { Loader2, Search, TrendingUp, Brain } from 'lucide-react';
import { DriverIntel } from './DriverIntel';
import { ForecastChart } from './ForecastChart';
import { HealthGauge } from './HealthGauge';
import { StatusBadge } from './StatusBadge';
import {
  type VehicleData,
  parseAnomalyDrivers, levelColor,
  MAINTENANCE_LABELS, SEVERITY_COLORS,
} from './anomalyHelpers';
import type { FeatureForecast } from './anomalyHelpers';
import type { Pillar } from './Sidebar';

interface PrimeDriverProps {
  prefetchedVehicles?: VehicleData[];
  prefetchedForecasts?: Record<string, FeatureForecast[]>;
  activePillar: Pillar;
}

export function PrimeDriver({ prefetchedVehicles, prefetchedForecasts, activePillar }: PrimeDriverProps) {
  const [vehicles, setVehicles] = useState<VehicleData[]>(prefetchedVehicles ?? []);
  const [loading, setLoading] = useState(!prefetchedVehicles);
  const [selectedDriver, setSelectedDriver] = useState<string | null>(null);
  const [search, setSearch] = useState('');

  useEffect(() => {
    if (prefetchedVehicles && prefetchedVehicles.length > 0) {
      setVehicles(prefetchedVehicles);
      setLoading(false);
      return;
    }
    fetch('/api/pipeline/anomaly')
      .then(r => r.json())
      .then(data => setVehicles(parseAnomalyDrivers(data)))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [prefetchedVehicles]);

  return (
    <div className="space-y-4">
      {activePillar === 'telemetry' && <DriverIntel showTabBar={false} />}
      {activePillar === 'anomaly' && <DriverAnomalyView vehicles={vehicles} loading={loading} search={search} setSearch={setSearch} selectedDriver={selectedDriver} setSelectedDriver={setSelectedDriver} />}
      {activePillar === 'forecast' && <DriverForecastView vehicles={vehicles} loading={loading} selectedDriver={selectedDriver} setSelectedDriver={setSelectedDriver} prefetchedForecasts={prefetchedForecasts} />}
    </div>
  );
}

/* ─── Anomaly Detection View ─── */

function DriverAnomalyView({ vehicles, loading, search, setSearch, selectedDriver, setSelectedDriver }: {
  vehicles: VehicleData[];
  loading: boolean;
  search: string;
  setSearch: (s: string) => void;
  selectedDriver: string | null;
  setSelectedDriver: (d: string | null) => void;
}) {
  const filtered = vehicles.filter(v =>
    !search || v.driver.toLowerCase().includes(search.toLowerCase()) || v.code.toLowerCase().includes(search.toLowerCase())
  );
  const sorted = [...filtered].sort((a, b) => {
    const order = { critical: 0, warning: 1, nominal: 2 };
    return (order[a.level] ?? 3) - (order[b.level] ?? 3);
  });
  const selected = selectedDriver ? vehicles.find(v => v.code === selectedDriver) : null;

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-sm text-muted-foreground py-8 justify-center">
        <Loader2 className="w-4 h-4 animate-spin text-[#FF8000]" />
        Loading anomaly data...
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Summary Bar */}
      <div className="flex items-center gap-4 text-[12px]">
        <span className="text-muted-foreground">{vehicles.length} drivers monitored</span>
        <span className="text-red-400">{vehicles.filter(v => v.level === 'critical').length} critical</span>
        <span className="text-[#FF8000]">{vehicles.filter(v => v.level === 'warning').length} warning</span>
        <span className="text-green-400">{vehicles.filter(v => v.level === 'nominal').length} nominal</span>
      </div>

      <div className="flex gap-4">
        {/* Driver List */}
        <div className="w-[280px] shrink-0 space-y-2">
          <div className="relative">
            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground" />
            <input
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder="Search drivers..."
              className="w-full bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded-lg pl-8 pr-3 py-1.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:border-[#FF8000]/30"
            />
          </div>
          <div className="space-y-1 max-h-[60vh] overflow-y-auto">
            {sorted.map(v => (
              <button
                key={v.code}
                onClick={() => setSelectedDriver(v.code === selectedDriver ? null : v.code)}
                className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-left transition-all ${
                  v.code === selectedDriver
                    ? 'bg-[#FF8000]/10 border border-[#FF8000]/30'
                    : 'bg-[#1A1F2E] border border-transparent hover:border-[rgba(255,128,0,0.12)]'
                }`}
              >
                <div className="w-2 h-2 rounded-full shrink-0" style={{ background: levelColor(v.level) }} />
                <div className="flex-1 min-w-0">
                  <div className="text-[12px] font-medium text-foreground truncate">{v.code} — {v.driver}</div>
                  <div className="text-[10px] text-muted-foreground">{v.team}</div>
                </div>
                <span className="text-[11px] font-mono tabular-nums" style={{ color: levelColor(v.level) }}>
                  {v.overallHealth.toFixed(0)}%
                </span>
              </button>
            ))}
          </div>
        </div>

        {/* Detail Panel */}
        <div className="flex-1 min-w-0">
          {selected ? (
            <DriverAnomalyDetail vehicle={selected} />
          ) : (
            <div className="flex items-center justify-center h-full text-sm text-muted-foreground">
              Select a driver to view anomaly details
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

interface AnomalyKex {
  driver_code: string;
  text: string;
  model_used: string;
  provider_used: string;
  sentiment: { label: string; score: number };
  topics: string[];
  generated_at: number;
}

function DriverAnomalyDetail({ vehicle }: { vehicle: VehicleData }) {
  const [kex, setKex] = useState<AnomalyKex | null>(null);
  const [kexLoading, setKexLoading] = useState(false);

  useEffect(() => {
    setKex(null);
    setKexLoading(true);
    fetch(`/api/anomaly/kex/${encodeURIComponent(vehicle.code)}`, { method: 'POST' })
      .then(r => r.ok ? r.json() : null)
      .then(d => setKex(d))
      .catch(() => setKex(null))
      .finally(() => setKexLoading(false));
  }, [vehicle.code]);

  const recentRaces = useMemo(() => vehicle.races.slice(-10), [vehicle.races]);

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="bg-[#1A1F2E] rounded-xl border border-[rgba(255,128,0,0.12)] p-4">
        <div className="flex items-center gap-4">
          <HealthGauge value={vehicle.overallHealth} size={64} />
          <div>
            <div className="text-lg font-semibold text-foreground">{vehicle.code} — {vehicle.driver}</div>
            <div className="text-sm text-muted-foreground">{vehicle.team}</div>
            <div className="flex items-center gap-2 mt-1">
              <StatusBadge status={vehicle.level} />
              <span className="text-[11px] text-muted-foreground">Last race: {vehicle.lastRace}</span>
            </div>
          </div>
        </div>
      </div>

      {/* System Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {vehicle.systems.map(sys => {
          const MaintenanceInfo = MAINTENANCE_LABELS[sys.maintenanceAction ?? 'none'] ?? MAINTENANCE_LABELS.none;
          const MIcon = MaintenanceInfo.icon;
          return (
            <div key={sys.name} className="bg-[#1A1F2E] rounded-xl border border-[rgba(255,128,0,0.12)] p-3">
              <div className="flex items-center justify-between mb-2">
                <span className="text-[12px] font-medium text-foreground">{sys.name}</span>
                <span className="text-[11px] font-mono" style={{ color: levelColor(sys.level) }}>
                  {sys.health.toFixed(0)}%
                </span>
              </div>
              {/* Health bar */}
              <div className="w-full h-1.5 rounded-full bg-[#0D1117] mb-2">
                <div className="h-full rounded-full transition-all" style={{ width: `${sys.health}%`, background: levelColor(sys.level) }} />
              </div>
              {/* Severity distribution */}
              {sys.severityProbabilities && (
                <div className="flex h-1 rounded-full overflow-hidden mb-2">
                  {Object.entries(sys.severityProbabilities).map(([sev, pct]) => {
                    const pctNum = typeof pct === 'number' ? pct : 0;
                    return <div key={sev} style={{ width: `${pctNum * 100}%`, background: SEVERITY_COLORS[sev as keyof typeof SEVERITY_COLORS] }} />;
                  })}
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
              <div className="mt-2 text-[9px] text-muted-foreground">
                {sys.voteCount}/{sys.totalModels} models agree
              </div>
            </div>
          );
        })}
      </div>

      {/* Race History — last 10 */}
      {recentRaces.length > 1 && (
        <div className="bg-[#1A1F2E] rounded-xl border border-[rgba(255,128,0,0.12)] p-3">
          <h3 className="text-[12px] font-medium text-foreground mb-2">Race-by-Race Health <span className="text-muted-foreground font-normal">(last {recentRaces.length})</span></h3>
          <div className="overflow-x-auto">
            <table className="w-full text-[11px]">
              <thead>
                <tr className="text-muted-foreground border-b border-[rgba(255,128,0,0.08)]">
                  <th className="text-left py-1.5 pr-3">Race</th>
                  {vehicle.systems.map(s => (
                    <th key={s.name} className="text-center px-2 py-1.5">{s.name}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {recentRaces.map(race => (
                  <tr key={race.race} className="border-b border-[rgba(255,128,0,0.04)]">
                    <td className="py-1.5 pr-3 text-foreground font-medium">{race.race}</td>
                    {vehicle.systems.map(s => {
                      const raceSystem = race.systems[s.name];
                      if (!raceSystem) return <td key={s.name} className="text-center px-2 text-muted-foreground">—</td>;
                      return (
                        <td key={s.name} className="text-center px-2">
                          <span className="font-mono" style={{ color: levelColor(raceSystem.level as any) }}>
                            {raceSystem.health.toFixed(0)}%
                          </span>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── WISE Anomaly Briefing ── */}
      <div className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded-xl p-4">
        <h3 className="text-sm text-muted-foreground flex items-center gap-2 mb-3">
          <Brain className="w-4 h-4" /> WISE Anomaly Briefing
        </h3>
        {kexLoading && (
          <div className="flex items-center justify-center gap-2 py-6">
            <Loader2 className="w-4 h-4 text-[#FF8000] animate-spin" />
            <span className="text-[11px] text-muted-foreground">Extracting anomaly intelligence…</span>
          </div>
        )}
        {kex && !kexLoading && (
          <div className="space-y-3">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-[9px] font-semibold tracking-wider px-1.5 py-0.5 rounded bg-[#3b82f6]/20 text-[#3b82f6]">REALTIME</span>
              {kex.sentiment && (
                <span className={`text-[8px] font-semibold px-1 py-0.5 rounded ${
                  kex.sentiment.label === 'positive' ? 'bg-green-500/15 text-green-400' :
                  kex.sentiment.label === 'negative' ? 'bg-red-500/15 text-red-400' :
                  'bg-zinc-500/15 text-zinc-400'
                }`}>
                  {kex.sentiment.label === 'positive' ? '\u25B2' : kex.sentiment.label === 'negative' ? '\u25BC' : '\u25CF'} {kex.sentiment.score}
                </span>
              )}
              {kex.topics?.length > 0 && kex.topics.map(t => (
                <span key={t} className="text-[8px] px-1.5 py-0.5 rounded bg-[#FF8000]/10 text-[#FF8000]">{t}</span>
              ))}
            </div>
            <div className="text-[12px] text-muted-foreground leading-relaxed whitespace-pre-line">{kex.text}</div>
            <div className="flex items-center justify-between pt-2 border-t border-[rgba(255,128,0,0.06)]">
              <span className="text-[9px] text-muted-foreground/50 font-mono">via {kex.model_used} ({kex.provider_used})</span>
              <span className="text-[9px] text-muted-foreground/50">{new Date(kex.generated_at * 1000).toLocaleString()}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/* ─── Forecast View ─── */

function DriverForecastView({ vehicles, loading, selectedDriver, setSelectedDriver, prefetchedForecasts }: {
  vehicles: VehicleData[];
  loading: boolean;
  selectedDriver: string | null;
  setSelectedDriver: (d: string | null) => void;
  prefetchedForecasts?: Record<string, FeatureForecast[]>;
}) {
  if (loading) {
    return (
      <div className="flex items-center gap-2 text-sm text-muted-foreground py-8 justify-center">
        <Loader2 className="w-4 h-4 animate-spin text-[#FF8000]" />
        Loading...
      </div>
    );
  }

  const driverOptions = vehicles.map(v => v.code).sort();
  const effectiveDriver = selectedDriver ?? driverOptions[0] ?? null;

  return (
    <div className="space-y-4">
      {/* Driver selector */}
      <div className="flex items-center gap-3">
        <label className="text-[12px] text-muted-foreground">Driver</label>
        <select
          title="Select driver"
          value={effectiveDriver ?? ''}
          onChange={e => setSelectedDriver(e.target.value)}
          className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded-lg px-3 py-1.5 text-sm text-foreground focus:outline-none focus:border-[#FF8000]/30"
        >
          {driverOptions.map(code => (
            <option key={code} value={code}>{code}</option>
          ))}
        </select>
      </div>

      {effectiveDriver && (
        <div className="bg-[#1A1F2E] rounded-xl border border-[rgba(255,128,0,0.12)] p-3">
          <h3 className="text-[12px] font-medium text-foreground mb-3 flex items-center gap-1.5">
            <TrendingUp className="w-3 h-3 text-[#FF8000]" />
            Feature Forecasts — {effectiveDriver}
          </h3>
          <ForecastChart
            driverCode={effectiveDriver}
            prefetchedForecasts={prefetchedForecasts?.[effectiveDriver]}
          />
        </div>
      )}
    </div>
  );
}
