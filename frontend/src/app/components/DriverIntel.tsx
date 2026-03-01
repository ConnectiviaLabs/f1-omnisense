import { useState, useEffect, useMemo, useCallback } from 'react';
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
} from 'recharts';
import { Users, Search, Loader2, Target, Zap, Shield, ChevronRight } from 'lucide-react';
import type { DriverPerformanceMarker, DriverOvertakeProfile, DriverTelemetryProfile } from '../types';
import * as api from '../api/driverIntel';

const teamColors: Record<string, string> = {
  'Red Bull': '#3671C6', 'McLaren': '#FF8000', 'Ferrari': '#E8002D',
  'Mercedes': '#27F4D2', 'Aston Martin': '#229971', 'Alpine': '#FF87BC',
  'Williams': '#64C4FF', 'RB': '#6692FF', 'Kick Sauber': '#52E252',
  'Haas F1 Team': '#B6BABD', 'AlphaTauri': '#6692FF', 'Alfa Romeo': '#C92D4B',
  'Racing Point': '#F596C8', 'Renault': '#FFF500', 'Toro Rosso': '#469BFF',
  'Force India': '#F596C8',
};

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload?.length) {
    return (
      <div className="bg-[#0D1117] border border-[rgba(255,128,0,0.2)] rounded-lg p-2 text-[12px]">
        <div className="text-muted-foreground mb-1">{label}</div>
        {payload.map((entry: any, i: number) => (
          <div key={i} className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full" style={{ backgroundColor: entry.color }} />
            <span className="text-muted-foreground">{entry.name}:</span>
            <span className="text-foreground font-mono">{typeof entry.value === 'number' ? entry.value.toFixed(2) : entry.value}</span>
          </div>
        ))}
      </div>
    );
  }
  return null;
};

type Tab = 'overview' | 'performance' | 'compare';

// Normalize a value to 0-100 scale (invert if lower is better)
function normalize(val: number | null, min: number, max: number, invert = false): number {
  if (val === null || val === undefined) return 0;
  const clamped = Math.max(min, Math.min(max, val));
  const norm = ((clamped - min) / (max - min)) * 100;
  return invert ? 100 - norm : norm;
}

export function DriverIntel() {
  const [tab, setTab] = useState<Tab>('overview');
  const [selectedDriver, setSelectedDriver] = useState<string | null>(null);

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-1 bg-[#1A1F2E] rounded-lg p-0.5 border border-[rgba(255,128,0,0.12)] w-fit">
        {([
          { id: 'overview' as Tab, label: 'Driver Grid', icon: Users },
          { id: 'performance' as Tab, label: 'Performance Profile', icon: Target },
          { id: 'compare' as Tab, label: 'Compare Drivers', icon: Zap },
        ]).map(t => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`text-sm px-4 py-1.5 rounded-md transition-all flex items-center gap-2 ${
              tab === t.id ? 'bg-[#FF8000]/10 text-[#FF8000]' : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            <t.icon className="w-3.5 h-3.5" />
            {t.label}
          </button>
        ))}
      </div>

      {tab === 'overview' && <DriverGrid onSelect={(d) => { setSelectedDriver(d); setTab('performance'); }} />}
      {tab === 'performance' && <PerformanceProfile driverCode={selectedDriver} onSelect={setSelectedDriver} />}
      {tab === 'compare' && <CompareDrivers />}
    </div>
  );
}

/* ─── Driver Grid ─── */

function DriverGrid({ onSelect }: { onSelect: (code: string) => void }) {
  const [drivers, setDrivers] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');

  useEffect(() => {
    api.getOpponentDrivers().then(data => {
      setDrivers(data.drivers || []);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  const filtered = useMemo(() => {
    if (!search) return drivers;
    const q = search.toLowerCase();
    return drivers.filter((d: any) =>
      (d.driver_id || '').toLowerCase().includes(q) ||
      (d.team || '').toLowerCase().includes(q) ||
      (d.nationality || '').toLowerCase().includes(q)
    );
  }, [drivers, search]);

  if (loading) {
    return <div className="flex items-center justify-center py-20"><Loader2 className="w-6 h-6 text-[#FF8000] animate-spin" /></div>;
  }

  return (
    <div className="space-y-4">
      <div className="relative w-80">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
        <input
          type="text"
          value={search}
          onChange={e => setSearch(e.target.value)}
          placeholder="Search drivers, teams..."
          className="w-full pl-10 pr-4 py-2 text-sm bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded-lg text-foreground placeholder:text-muted-foreground focus:outline-none focus:border-[#FF8000]/40"
        />
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
        {filtered.map((d: any) => {
          const team = d.team || d.constructor || '';
          const color = teamColors[team] || '#666';
          return (
            <button
              key={d.driver_id}
              onClick={() => onSelect(d.driver_id)}
              className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded-xl p-4 text-left hover:border-[#FF8000]/40 transition-all group"
            >
              <div className="flex items-center gap-2 mb-2">
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: color }} />
                <span className="text-xs text-muted-foreground font-mono">{team}</span>
              </div>
              <div className="text-foreground font-semibold text-sm group-hover:text-[#FF8000] transition-colors">
                {d.driver_id?.replace(/_/g, ' ').replace(/\b\w/g, (c: string) => c.toUpperCase())}
              </div>
              <div className="flex items-center gap-3 mt-2 text-xs text-muted-foreground">
                {d.nationality && <span>{d.nationality}</span>}
                {d.total_races != null && <span>{d.total_races} races</span>}
                {d.wins != null && d.wins > 0 && <span className="text-[#f59e0b]">{d.wins} wins</span>}
              </div>
              <ChevronRight className="w-4 h-4 text-muted-foreground mt-2 opacity-0 group-hover:opacity-100 transition-opacity" />
            </button>
          );
        })}
      </div>

      {filtered.length === 0 && (
        <div className="text-center py-10 text-muted-foreground text-sm">No drivers match your search.</div>
      )}
    </div>
  );
}

/* ─── Performance Profile ─── */

function PerformanceProfile({ driverCode, onSelect }: { driverCode: string | null; onSelect: (code: string) => void }) {
  const [markers, setMarkers] = useState<DriverPerformanceMarker[]>([]);
  const [overtakes, setOvertakes] = useState<DriverOvertakeProfile[]>([]);
  const [telemetry, setTelemetry] = useState<DriverTelemetryProfile[]>([]);
  const [allMarkers, setAllMarkers] = useState<DriverPerformanceMarker[]>([]);
  const [allOvertakes, setAllOvertakes] = useState<DriverOvertakeProfile[]>([]);
  const [drivers, setDrivers] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  // Load all data on mount
  useEffect(() => {
    Promise.all([
      api.getPerformanceMarkers(),
      api.getOvertakeProfiles(),
      api.getTelemetryProfiles(),
      api.getOpponentDrivers(),
    ]).then(([m, o, t, d]) => {
      setAllMarkers(m);
      setAllOvertakes(o);
      setTelemetry(t);
      setDrivers(d.drivers || []);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  // Filter for selected driver
  useEffect(() => {
    if (!driverCode) return;
    // Find the driver code (3-letter) from driver_id
    const driver = drivers.find((d: any) => d.driver_id === driverCode);
    const code = driver?.code || driverCode.slice(0, 3).toUpperCase();

    setMarkers(allMarkers.filter(m => m.Driver === code));
    setOvertakes(allOvertakes.filter(o => o.driver_code === code));
  }, [driverCode, drivers, allMarkers, allOvertakes]);

  const driverTelemetry = useMemo(() => {
    if (!driverCode || !drivers.length) return null;
    const driver = drivers.find((d: any) => d.driver_id === driverCode);
    const code = driver?.code || driverCode.slice(0, 3).toUpperCase();
    return telemetry.find(t => t.driver_code === code) || null;
  }, [driverCode, drivers, telemetry]);

  const radarData = useMemo(() => {
    const m = markers[0];
    const o = overtakes[0];
    const t = driverTelemetry;
    if (!m && !o && !t) return [];

    return [
      { metric: 'Consistency', value: normalize(m?.lap_time_consistency_std ?? null, 0, 30, true), fullMark: 100 },
      { metric: 'Tyre Mgmt', value: normalize(m?.degradation_slope_s_per_lap ?? null, -0.3, 0.1, true), fullMark: 100 },
      { metric: 'Overtaking', value: normalize(o?.overtake_ratio ?? null, 0.5, 1.5, false), fullMark: 100 },
      { metric: 'Late Race', value: normalize(m?.late_race_delta_s ?? null, -30, 5, true), fullMark: 100 },
      { metric: 'Top Speed', value: normalize(t?.avg_race_speed_kmh ?? null, 170, 220, false), fullMark: 100 },
      { metric: 'Braking', value: normalize(t?.avg_braking_g ?? null, 2, 5, false), fullMark: 100 },
    ];
  }, [markers, overtakes, driverTelemetry]);

  // Overtake bar data
  const overtakeBarData = useMemo(() => {
    const o = overtakes[0];
    if (!o) return [];
    const avgMade = allOvertakes.reduce((s, x) => s + x.overtakes_per_race, 0) / (allOvertakes.length || 1);
    const avgLost = allOvertakes.reduce((s, x) => s + x.times_overtaken_per_race, 0) / (allOvertakes.length || 1);
    return [
      { metric: 'OT Made/Race', driver: o.overtakes_per_race, avg: +avgMade.toFixed(1) },
      { metric: 'OT Lost/Race', driver: o.times_overtaken_per_race, avg: +avgLost.toFixed(1) },
      { metric: 'OT Ratio', driver: o.overtake_ratio, avg: +(allOvertakes.reduce((s, x) => s + x.overtake_ratio, 0) / (allOvertakes.length || 1)).toFixed(2) },
    ];
  }, [overtakes, allOvertakes]);

  if (loading) {
    return <div className="flex items-center justify-center py-20"><Loader2 className="w-6 h-6 text-[#FF8000] animate-spin" /></div>;
  }

  if (!driverCode) {
    return (
      <div className="text-center py-20 text-muted-foreground">
        <Target className="w-8 h-8 mx-auto mb-3 opacity-50" />
        <p className="text-sm">Select a driver from the Driver Grid tab to view their performance profile.</p>
      </div>
    );
  }

  const driverName = driverCode.replace(/_/g, ' ').replace(/\b\w/g, (c: string) => c.toUpperCase());
  const m = markers[0];
  const t = driverTelemetry;

  return (
    <div className="space-y-4">
      {/* Driver selector dropdown */}
      <div className="flex items-center gap-3">
        <select
          value={driverCode}
          onChange={e => onSelect(e.target.value)}
          aria-label="Select driver"
          className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:border-[#FF8000]/40"
        >
          {drivers.map((d: any) => (
            <option key={d.driver_id} value={d.driver_id}>
              {d.driver_id?.replace(/_/g, ' ').replace(/\b\w/g, (c: string) => c.toUpperCase())} — {d.team || d.constructor || ''}
            </option>
          ))}
        </select>
        <h2 className="text-foreground text-lg font-semibold">{driverName}</h2>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Radar Chart */}
        <div className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded-xl p-4">
          <h3 className="text-sm text-muted-foreground mb-3 flex items-center gap-2"><Target className="w-4 h-4" /> Performance Radar</h3>
          {radarData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="rgba(255,128,0,0.15)" />
                <PolarAngleAxis dataKey="metric" tick={{ fill: '#888', fontSize: 11 }} />
                <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                <Radar name={driverName} dataKey="value" stroke="#FF8000" fill="#FF8000" fillOpacity={0.25} />
              </RadarChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[300px] flex items-center justify-center text-muted-foreground text-sm">No data available</div>
          )}
        </div>

        {/* Stat Cards */}
        <div className="space-y-3">
          <h3 className="text-sm text-muted-foreground flex items-center gap-2"><Zap className="w-4 h-4" /> Key Metrics</h3>
          <div className="grid grid-cols-2 gap-3">
            <StatCard label="Tyre Degradation" value={m?.degradation_slope_s_per_lap} unit="s/lap" precision={3} />
            <StatCard label="Late Race Delta" value={m?.late_race_delta_s} unit="s" precision={2} />
            <StatCard label="Consistency (std)" value={m?.lap_time_consistency_std} unit="s" precision={2} />
            <StatCard label="Avg Stint Length" value={m?.avg_stint_length} unit="laps" precision={1} />
            <StatCard label="Sector 1 CV" value={m?.sector1_cv} unit="%" precision={2} />
            <StatCard label="Sector 2 CV" value={m?.sector2_cv} unit="%" precision={2} />
            <StatCard label="Heat Sensitivity" value={m?.heat_lap_delta_s} unit="s" precision={2} />
            <StatCard label="Humidity Effect" value={m?.humidity_lap_delta_s} unit="s" precision={2} />
          </div>

          {t && (
            <>
              <h3 className="text-sm text-muted-foreground flex items-center gap-2 mt-4"><Shield className="w-4 h-4" /> Telemetry Style</h3>
              <div className="grid grid-cols-2 gap-3">
                <StatCard label="Avg Speed" value={t.avg_race_speed_kmh} unit="km/h" precision={1} />
                <StatCard label="Avg Braking G" value={t.avg_braking_g} unit="G" precision={2} />
                <StatCard label="Full Throttle" value={t.full_throttle_ratio * 100} unit="%" precision={1} />
                <StatCard label="DRS Gain" value={t.drs_speed_gain_kmh} unit="km/h" precision={1} />
                <StatCard label="Brake→Throttle" value={t.brake_to_throttle_avg_s * 1000} unit="ms" precision={0} />
                <StatCard label="Braking Consistency" value={t.braking_consistency} unit="std" precision={2} />
              </div>
            </>
          )}
        </div>
      </div>

      {/* Overtake Bar Chart */}
      {overtakeBarData.length > 0 && (
        <div className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded-xl p-4">
          <h3 className="text-sm text-muted-foreground mb-3 flex items-center gap-2"><Shield className="w-4 h-4" /> Overtaking vs Grid Average</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={overtakeBarData} layout="vertical">
              <XAxis type="number" tick={{ fill: '#888', fontSize: 11 }} />
              <YAxis type="category" dataKey="metric" tick={{ fill: '#888', fontSize: 11 }} width={100} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="driver" name={driverName} fill="#FF8000" radius={[0, 4, 4, 0]} />
              <Bar dataKey="avg" name="Grid Avg" fill="#444" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

/* ─── Compare Drivers ─── */

function CompareDrivers() {
  const [allMarkers, setAllMarkers] = useState<DriverPerformanceMarker[]>([]);
  const [allOvertakes, setAllOvertakes] = useState<DriverOvertakeProfile[]>([]);
  const [allTelemetry, setAllTelemetry] = useState<DriverTelemetryProfile[]>([]);
  const [drivers, setDrivers] = useState<any[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);

  const COMPARE_COLORS = ['#FF8000', '#3671C6', '#E8002D', '#27F4D2'];

  useEffect(() => {
    Promise.all([
      api.getPerformanceMarkers(),
      api.getOvertakeProfiles(),
      api.getTelemetryProfiles(),
      api.getOpponentDrivers(),
    ]).then(([m, o, t, d]) => {
      setAllMarkers(m);
      setAllOvertakes(o);
      setAllTelemetry(t);
      setDrivers(d.drivers || []);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  const toggleDriver = useCallback((id: string) => {
    setSelected(prev => {
      if (prev.includes(id)) return prev.filter(x => x !== id);
      if (prev.length >= 4) return prev;
      return [...prev, id];
    });
  }, []);

  const compareRadarData = useMemo(() => {
    if (selected.length < 2) return [];
    const metrics = ['Consistency', 'Tyre Mgmt', 'Overtaking', 'Late Race', 'Top Speed', 'Braking'];
    return metrics.map(metric => {
      const row: any = { metric };
      selected.forEach(driverId => {
        const driver = drivers.find((d: any) => d.driver_id === driverId);
        const code = driver?.code || driverId.slice(0, 3).toUpperCase();
        const m = allMarkers.find(x => x.Driver === code);
        const o = allOvertakes.find(x => x.driver_code === code);
        const t = allTelemetry.find(x => x.driver_code === code);

        let val = 0;
        switch (metric) {
          case 'Consistency': val = normalize(m?.lap_time_consistency_std ?? null, 0, 30, true); break;
          case 'Tyre Mgmt': val = normalize(m?.degradation_slope_s_per_lap ?? null, -0.3, 0.1, true); break;
          case 'Overtaking': val = normalize(o?.overtake_ratio ?? null, 0.5, 1.5, false); break;
          case 'Late Race': val = normalize(m?.late_race_delta_s ?? null, -30, 5, true); break;
          case 'Top Speed': val = normalize(t?.avg_race_speed_kmh ?? null, 170, 220, false); break;
          case 'Braking': val = normalize(t?.avg_braking_g ?? null, 2, 5, false); break;
        }
        row[code] = val;
      });
      return row;
    });
  }, [selected, drivers, allMarkers, allOvertakes, allTelemetry]);

  if (loading) {
    return <div className="flex items-center justify-center py-20"><Loader2 className="w-6 h-6 text-[#FF8000] animate-spin" /></div>;
  }

  return (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">Select 2-4 drivers to compare. ({selected.length}/4 selected)</p>

      {/* Driver picker chips */}
      <div className="flex flex-wrap gap-2">
        {drivers.map((d: any) => {
          const isSelected = selected.includes(d.driver_id);
          const idx = selected.indexOf(d.driver_id);
          return (
            <button
              key={d.driver_id}
              onClick={() => toggleDriver(d.driver_id)}
              className={`text-xs px-3 py-1.5 rounded-full border transition-all ${
                isSelected
                  ? 'border-[#FF8000] text-[#FF8000] bg-[#FF8000]/10'
                  : 'border-[rgba(255,128,0,0.12)] text-muted-foreground hover:text-foreground hover:border-[rgba(255,128,0,0.3)]'
              }`}
              disabled={!isSelected && selected.length >= 4}
            >
              {isSelected && <span className="inline-block w-2 h-2 rounded-full mr-1.5" style={{ backgroundColor: COMPARE_COLORS[idx] }} />}
              {d.code || d.driver_id?.slice(0, 3).toUpperCase()}
            </button>
          );
        })}
      </div>

      {/* Overlaid Radar */}
      {compareRadarData.length > 0 && (
        <div className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded-xl p-4">
          <h3 className="text-sm text-muted-foreground mb-3">Performance Comparison</h3>
          <ResponsiveContainer width="100%" height={400}>
            <RadarChart data={compareRadarData}>
              <PolarGrid stroke="rgba(255,128,0,0.15)" />
              <PolarAngleAxis dataKey="metric" tick={{ fill: '#888', fontSize: 11 }} />
              <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
              {selected.map((driverId, i) => {
                const driver = drivers.find((d: any) => d.driver_id === driverId);
                const code = driver?.code || driverId.slice(0, 3).toUpperCase();
                return (
                  <Radar
                    key={driverId}
                    name={code}
                    dataKey={code}
                    stroke={COMPARE_COLORS[i]}
                    fill={COMPARE_COLORS[i]}
                    fillOpacity={0.1}
                  />
                );
              })}
            </RadarChart>
          </ResponsiveContainer>

          {/* Legend */}
          <div className="flex items-center justify-center gap-6 mt-2">
            {selected.map((driverId, i) => {
              const driver = drivers.find((d: any) => d.driver_id === driverId);
              const code = driver?.code || driverId.slice(0, 3).toUpperCase();
              const name = driverId.replace(/_/g, ' ').replace(/\b\w/g, (c: string) => c.toUpperCase());
              return (
                <div key={driverId} className="flex items-center gap-2 text-xs">
                  <span className="w-3 h-3 rounded-full" style={{ backgroundColor: COMPARE_COLORS[i] }} />
                  <span className="text-foreground">{code}</span>
                  <span className="text-muted-foreground">{name}</span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {selected.length < 2 && (
        <div className="text-center py-10 text-muted-foreground text-sm">Select at least 2 drivers to see the comparison radar.</div>
      )}
    </div>
  );
}

/* ─── Stat Card ─── */

function StatCard({ label, value, unit, precision = 2 }: {
  label: string; value: number | null | undefined; unit: string; precision?: number;
}) {
  const display = value != null ? value.toFixed(precision) : '—';

  return (
    <div className="bg-[#0D1117] border border-[rgba(255,128,0,0.08)] rounded-lg p-3">
      <div className="text-[11px] text-muted-foreground mb-1">{label}</div>
      <div className="flex items-baseline gap-1">
        <span className="text-foreground font-mono text-lg">{display}</span>
        <span className="text-xs text-muted-foreground">{unit}</span>
      </div>
    </div>
  );
}
