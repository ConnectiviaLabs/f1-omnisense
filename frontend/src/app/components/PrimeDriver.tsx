import { useState, useEffect, useMemo } from 'react';
import { Loader2, Search } from 'lucide-react';
import KexBriefingCard from './KexBriefingCard';
import { DriverIntel } from './DriverIntel';
import { ForecastChart } from './ForecastChart';
import { HealthGauge } from './HealthGauge';
import { StatusBadge } from './StatusBadge';
import {
  type VehicleData,
  parseAnomalyDrivers, levelColor,
  MAINTENANCE_LABELS, SEVERITY_COLORS,
} from './anomalyHelpers';
import { TEAM_COLORS_BY_ID as TEAM_COLORS, teamIdFromName } from '../constants/teams';
import type { Pillar } from './Sidebar';

interface PrimeDriverProps {
  prefetchedVehicles?: VehicleData[];
  activePillar: Pillar;
}

export function PrimeDriver({ prefetchedVehicles, activePillar }: PrimeDriverProps) {
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

  // Auto-select first McLaren driver when data loads
  useEffect(() => {
    if (selectedDriver || vehicles.length === 0) return;
    const mcl = vehicles.find(v => v.team === 'McLaren');
    if (mcl) setSelectedDriver(mcl.code);
  }, [vehicles, selectedDriver]);

  return (
    <div className="space-y-4">
      {activePillar === 'telemetry' && <DriverIntel showTabBar={false} prefetchedVehicles={vehicles} />}
      {activePillar === 'anomaly' && <DriverAnomalyView vehicles={vehicles} loading={loading} search={search} setSearch={setSearch} selectedDriver={selectedDriver} setSelectedDriver={setSelectedDriver} />}
      {activePillar === 'forecast' && <DriverForecastView vehicles={vehicles} loading={loading} selectedDriver={selectedDriver} setSelectedDriver={setSelectedDriver} />}
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
  const isMcLaren = (v: VehicleData) => v.team === 'McLaren';
  const sorted = [...filtered].sort((a, b) => {
    // McLaren always first, then by severity
    if (isMcLaren(a) && !isMcLaren(b)) return -1;
    if (!isMcLaren(a) && isMcLaren(b)) return 1;
    const order = { critical: 0, warning: 1, nominal: 2 };
    return (order[a.level] ?? 3) - (order[b.level] ?? 3);
  });
  const selected = selectedDriver ? vehicles.find(v => v.code === selectedDriver) : null;

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-sm text-muted-foreground py-8 justify-center">
        <Loader2 className="w-4 h-4 animate-spin text-primary" />
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
        <span className="text-primary">{vehicles.filter(v => v.level === 'warning').length} warning</span>
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
              className="w-full bg-card border border-border rounded-lg pl-8 pr-3 py-1.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:border-primary/30"
            />
          </div>
          <div className="space-y-1 max-h-[60vh] overflow-y-auto">
            {sorted.map((v, i) => {
              const mcl = isMcLaren(v);
              const vColor = TEAM_COLORS[teamIdFromName(v.team)] ?? '#888';
              const isSelected = v.code === selectedDriver;
              // Insert divider between McLaren and rest
              const showDivider = mcl === false && i > 0 && isMcLaren(sorted[i - 1]);
              return (
                <div key={v.code}>
                  {showDivider && (
                    <div className="flex items-center gap-2 py-1.5 px-1">
                      <div className="h-px flex-1 bg-border" />
                      <span className="text-[9px] text-muted-foreground uppercase tracking-wider">Field</span>
                      <div className="h-px flex-1 bg-border" />
                    </div>
                  )}
                  <button
                    type="button"
                    onClick={() => setSelectedDriver(v.code === selectedDriver ? null : v.code)}
                    className={`w-full flex items-center gap-2.5 rounded-lg text-left transition-all ${
                      isSelected
                        ? 'bg-primary/10 border border-primary/30'
                        : mcl
                          ? 'bg-primary/[0.04] border border-primary/10 hover:border-primary/25'
                          : 'bg-card border border-transparent hover:border-border'
                    } ${mcl ? 'px-3 py-2.5' : 'px-3 py-2'}`}
                  >
                    <div
                      className={`rounded-full shrink-0 ${mcl ? 'w-2.5 h-2.5' : 'w-2 h-2'}`}
                      style={{ background: mcl ? vColor : levelColor(v.level) }}
                    />
                    <div className="flex-1 min-w-0">
                      <div className={`font-medium text-foreground truncate ${mcl ? 'text-[13px]' : 'text-[12px]'}`}>
                        {v.code} — {v.driver}
                      </div>
                      <div className="text-[10px] text-muted-foreground">{v.team}</div>
                    </div>
                    {mcl && <HealthGauge value={v.overallHealth} size={28} />}
                    <span className="text-[11px] font-mono tabular-nums" style={{ color: levelColor(v.level) }}>
                      {v.overallHealth.toFixed(0)}%
                    </span>
                  </button>
                </div>
              );
            })}
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
  scores: Record<string, number>;
  summary: string;
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
  const mcl = vehicle.team === 'McLaren';
  const detailColor = TEAM_COLORS[teamIdFromName(vehicle.team)] ?? '#888';

  return (
    <div className="space-y-4">
      {/* Header */}
      <div
        className={`bg-card rounded-lg border p-4 ${mcl ? 'border-primary/20' : 'border-border'}`}
        style={{ borderTop: `3px solid ${detailColor}` }}
      >
        <div className="flex items-center gap-4">
          <HealthGauge value={vehicle.overallHealth} size={mcl ? 72 : 64} />
          <div>
            <div className={`font-semibold text-foreground ${mcl ? 'text-xl' : 'text-lg'}`}>{vehicle.code} — {vehicle.driver}</div>
            <div className="text-sm" style={{ color: detailColor }}>{vehicle.team}</div>
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
              <div className="mt-2 text-[10px] text-muted-foreground">
                {sys.voteCount}/{sys.totalModels} models agree
              </div>
            </div>
          );
        })}
      </div>

      {/* Race History — last 10 */}
      {recentRaces.length > 1 && (
        <div className="bg-card rounded-lg border border-border p-3">
          <h3 className="text-[12px] font-medium text-foreground mb-2">Race-by-Race Health <span className="text-muted-foreground font-normal">(last {recentRaces.length})</span></h3>
          <div className="overflow-x-auto">
            <table className="w-full text-[11px]">
              <thead>
                <tr className="text-muted-foreground border-b border-border">
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
      <KexBriefingCard
        title="WISE Anomaly Briefing"
        kex={kex}
        loading={kexLoading}
        loadingText="Extracting anomaly intelligence\u2026"
      />
    </div>
  );
}

/* ─── Forecast View ─── */

function DriverForecastView({ vehicles, loading }: {
  vehicles: VehicleData[];
  loading: boolean;
  selectedDriver: string | null;
  setSelectedDriver: (d: string | null) => void;
}) {
  const [expandedField, setExpandedField] = useState<Record<string, boolean>>({});

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-sm text-muted-foreground py-8 justify-center">
        <Loader2 className="w-4 h-4 animate-spin text-primary" />
        Loading...
      </div>
    );
  }

  const mclarenDrivers = vehicles.filter(v => v.team === 'McLaren');
  const fieldDrivers = vehicles.filter(v => v.team !== 'McLaren').sort((a, b) => {
    const order = { critical: 0, warning: 1, nominal: 2 };
    return (order[a.level] ?? 3) - (order[b.level] ?? 3);
  });

  const toggleField = (code: string) =>
    setExpandedField(prev => ({ ...prev, [code]: !prev[code] }));

  return (
    <div className="space-y-5">
      {/* ── McLaren drivers — always visible, full forecast ── */}
      {mclarenDrivers.map(v => {
        const teamColor = TEAM_COLORS[teamIdFromName(v.team)] ?? '#FF8000';
        return (
          <div key={v.code} className="space-y-3">
            {/* Driver header card */}
            <div
              className="bg-card rounded-lg border border-primary/20 p-4"
              style={{ borderLeft: `4px solid ${teamColor}` }}
            >
              <div className="flex items-center gap-4">
                <HealthGauge value={v.overallHealth} size={56} />
                <div className="flex-1 min-w-0">
                  <div className="text-[16px] font-bold text-foreground">{v.driver}</div>
                  <div className="text-[11px]" style={{ color: teamColor }}>{v.code} · {v.team}</div>
                </div>
                <StatusBadge status={v.level} />
              </div>
            </div>
            {/* Forecast charts + KeX briefing */}
            <ForecastChart driverCode={v.code} />
          </div>
        );
      })}

      {/* ── Field drivers — collapsible ── */}
      {fieldDrivers.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center gap-2 py-1">
            <div className="h-px flex-1 bg-border" />
            <span className="text-[9px] text-muted-foreground uppercase tracking-wider">Field — {fieldDrivers.length} drivers</span>
            <div className="h-px flex-1 bg-border" />
          </div>

          {fieldDrivers.map(v => {
            const isOpen = expandedField[v.code] ?? false;
            const vColor = TEAM_COLORS[teamIdFromName(v.team)] ?? '#888';
            return (
              <div key={v.code} className="bg-card rounded-lg border border-border overflow-hidden">
                <button
                  type="button"
                  onClick={() => toggleField(v.code)}
                  className="w-full flex items-center gap-3 px-3 py-2 hover:bg-background/40 transition-colors"
                >
                  <div className="w-1 h-6 rounded-full shrink-0" style={{ background: vColor }} />
                  <HealthGauge value={v.overallHealth} size={24} />
                  <div className="flex-1 min-w-0 text-left">
                    <div className="text-[12px] font-medium text-foreground">{v.driver}</div>
                    <div className="text-[10px] text-muted-foreground">{v.code} · {v.team}</div>
                  </div>
                  <span className="text-[11px] font-mono tabular-nums" style={{ color: levelColor(v.level) }}>
                    {v.overallHealth.toFixed(0)}%
                  </span>
                  <span className="text-[10px] text-muted-foreground">{isOpen ? '▾' : '▸'}</span>
                </button>
                {isOpen && (
                  <div className="px-3 pb-3">
                    <ForecastChart driverCode={v.code} hideKex />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
