import { useState, useEffect, useMemo } from 'react';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, BarChart, Bar, Cell,
} from 'recharts';
import { MapPin, Loader2, Wind, Thermometer, Droplets, Timer, ChevronRight, Trophy, Flag, TrendingUp, Users } from 'lucide-react';
import type { CircuitIntelligence, CircuitPitLoss, RaceAirDensity } from '../types';
import * as api from '../api/circuitIntel';
import type { CircuitHistory, CircuitKex } from '../api/circuitIntel';
import KexBriefingCard from './KexBriefingCard';
import { TrackMap } from './TrackMap';
import { CIRCUITS } from '../data/circuits';

// Map circuit_slug (e.g. "abu_dhabi") → CIRCUITS entry via geojsonPath
const SLUG_TO_CIRCUIT = new Map(
  CIRCUITS.map(c => {
    const slug = c.geojsonPath.replace('/circuits/', '').replace('.geojson', '');
    return [slug, c] as const;
  })
);
// Map MongoDB circuit_slug → CIRCUITS geojson slug
const NAME_ALIASES: Record<string, string> = {
  'albert_park': 'australian',
  'americas': 'usa',
  'baku': 'azerbaijan',
  'catalunya': 'spanish',
  'hungaroring': 'hungarian',
  'imola': 'emilia_romagna',
  'interlagos': 'brazilian',
  'jeddah': 'saudi_arabian',
  'losail': 'qatar',
  'marina_bay': 'singapore',
  'monza': 'italian',
  'red_bull_ring': 'austrian',
  'rodriguez': 'mexico_city',
  'shanghai': 'chinese',
  'silverstone': 'british',
  'spa': 'belgian',
  'suzuka': 'japanese',
  'vegas': 'las_vegas',
  'villeneuve': 'canadian',
  'yas_marina': 'abu_dhabi',
  'zandvoort': 'dutch',
};

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload?.length) {
    return (
      <div className="bg-background border border-[rgba(255,128,0,0.2)] rounded-lg p-2 text-[12px]">
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

export function CircuitIntel() {
  const [circuits, setCircuits] = useState<CircuitIntelligence[]>([]);
  const [pitLoss, setPitLoss] = useState<CircuitPitLoss[]>([]);
  const [airDensity, setAirDensity] = useState<RaceAirDensity[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [history, setHistory] = useState<CircuitHistory | null>(null);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [kex, setKex] = useState<CircuitKex | null>(null);
  const [kexLoading, setKexLoading] = useState(false);

  useEffect(() => {
    Promise.all([
      api.getCircuits(),
      api.getPitLoss(),
      api.getAirDensity(),
    ]).then(([c, p, a]) => {
      setCircuits(c);
      setPitLoss(p);
      setAirDensity(a);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  const selectedCircuit = useMemo(() =>
    circuits.find(c => c.circuit_slug === selected) || null,
    [circuits, selected]
  );

  const selectedPitLoss = useMemo(() =>
    pitLoss.find(p => p.circuit === selected) || null,
    [pitLoss, selected]
  );

  const selectedAirData = useMemo(() =>
    airDensity
      .filter(a => a.circuit_slug === selected)
      .sort((a, b) => a.year - b.year),
    [airDensity, selected]
  );

  // Auto-generate KeX when circuit changes (backend auto-regenerates if data changed)
  useEffect(() => {
    setKex(null);
    if (!selected) return;
    setKexLoading(true);
    api.getCircuitKex(selected)
      .then(setKex)
      .catch(() => setKex(null))
      .finally(() => setKexLoading(false));
  }, [selected]);

  // Fetch history when circuit changes
  useEffect(() => {
    if (!selected) { setHistory(null); return; }
    setHistoryLoading(true);
    api.getCircuitHistory(selected)
      .then(h => setHistory(h.seasons?.length ? h : null))
      .catch(() => setHistory(null))
      .finally(() => setHistoryLoading(false));
  }, [selected]);

  // Pit loss ranking sorted by estimated pit lane loss
  const pitLossRanking = useMemo(() =>
    [...pitLoss].sort((a, b) => b.est_pit_lane_loss_s - a.est_pit_lane_loss_s),
    [pitLoss]
  );


  if (loading) {
    return <div className="flex items-center justify-center py-20"><Loader2 className="w-6 h-6 text-[#FF8000] animate-spin" /></div>;
  }

  return (
    <div className="flex gap-4 h-[calc(100vh-200px)]">
      {/* Left: Circuit List */}
      <div className="w-72 shrink-0 bg-card border border-border rounded-xl overflow-y-auto">
        <div className="p-3 border-b border-border">
          <h3 className="text-sm text-muted-foreground flex items-center gap-2">
            <MapPin className="w-4 h-4" />
            {circuits.length} Circuits
          </h3>
        </div>
        <div className="p-1">
          {circuits.sort((a, b) => a.circuit_name.localeCompare(b.circuit_name)).map(c => (
            <button
              key={c.circuit_slug}
              onClick={() => setSelected(c.circuit_slug)}
              className={`w-full text-left px-3 py-2.5 rounded-lg text-sm transition-all flex items-center justify-between ${
                selected === c.circuit_slug
                  ? 'bg-[#FF8000]/10 text-[#FF8000]'
                  : 'text-muted-foreground hover:bg-secondary hover:text-foreground'
              }`}
            >
              <div>
                <div className="font-medium">{c.circuit_name}</div>
                <div className="text-[11px] opacity-70 mt-0.5">
                  {(c.computed_length_m / 1000).toFixed(2)} km &middot; {c.estimated_corners} corners &middot; {c.drs_zones} DRS
                </div>
              </div>
              <ChevronRight className="w-3.5 h-3.5 shrink-0 opacity-40" />
            </button>
          ))}
        </div>
      </div>

      {/* Right: Circuit Detail */}
      <div className="flex-1 overflow-y-auto space-y-4">
        {!selected ? (
          <div className="flex items-center justify-center h-full text-muted-foreground text-sm">
            <div className="text-center">
              <MapPin className="w-8 h-8 mx-auto mb-3 opacity-50" />
              <p>Select a circuit to view intelligence data.</p>
            </div>
          </div>
        ) : (
          <>
            {/* Header */}
            <div className="bg-card border border-border rounded-xl p-4">
              <h2 className="text-foreground text-lg font-semibold">{selectedCircuit?.circuit_name}</h2>
              <div className="flex flex-wrap gap-4 mt-3">
                <InfoChip icon={<MapPin className="w-3.5 h-3.5" />} label="Length" value={`${((selectedCircuit?.computed_length_m || 0) / 1000).toFixed(2)} km`} />
                <InfoChip icon={<span className="text-xs">&#x27F0;</span>} label="Corners" value={String(selectedCircuit?.estimated_corners || '—')} />
                <InfoChip icon={<span className="text-xs font-bold">DRS</span>} label="DRS Zones" value={String(selectedCircuit?.drs_zones || '—')} />
                <InfoChip icon={<span className="text-xs">&#x2191;</span>} label="Elevation" value={
                  selectedCircuit?.elevation_gain_m != null
                    ? `${selectedCircuit.elevation_gain_m.toFixed(0)}m gain`
                    : '—'
                } />
                <InfoChip icon={<span className="text-xs">S</span>} label="Sectors" value={String(selectedCircuit?.sectors || '—')} />
                <InfoChip icon={<span className="text-xs">#</span>} label="Coordinates" value={String(selectedCircuit?.coordinate_count || '—')} />
              </div>
            </div>

            {/* Track Layout Map */}
            {(() => {
              const slug = selected!;
              const ci = SLUG_TO_CIRCUIT.get(slug) ?? SLUG_TO_CIRCUIT.get(NAME_ALIASES[slug] ?? '');
              if (!ci) return null;
              return (
                <div className="bg-card border border-border rounded-xl overflow-hidden">
                  <TrackMap
                    geojsonPath={ci.geojsonPath}
                    circuitName={ci.circuitName}
                    locality={ci.locality}
                    country={ci.country}
                    lengthKm={ci.lengthKm}
                    turns={ci.turns}
                    drsZones={ci.drsZones}
                    colorMode="speed"
                    height={400}
                  />
                </div>
              );
            })()}

            {/* Pit Loss */}
            {selectedPitLoss && (
              <div className="bg-card border border-border rounded-xl p-4">
                <h3 className="text-sm text-muted-foreground mb-3 flex items-center gap-2"><Timer className="w-4 h-4" /> Pit Stop Analysis</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  <PitCard label="Pit Lane Loss" value={selectedPitLoss.est_pit_lane_loss_s} unit="s" />
                  <PitCard label="Avg Total Pit" value={selectedPitLoss.avg_total_pit_s} unit="s" />
                  <PitCard label="Median Pit" value={selectedPitLoss.median_total_pit_s} unit="s" />
                  <PitCard label="Samples" value={selectedPitLoss.sample_count} unit="" precision={0} />
                </div>
                {selectedPitLoss.jolpica_avg_pit_duration_s && (
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mt-3">
                    <PitCard label="Jolpica Avg" value={selectedPitLoss.jolpica_avg_pit_duration_s} unit="s" />
                    <PitCard label="Jolpica Median" value={selectedPitLoss.jolpica_median_pit_duration_s} unit="s" />
                    <PitCard label="Jolpica Samples" value={selectedPitLoss.jolpica_pit_sample_count} unit="" precision={0} />
                  </div>
                )}
              </div>
            )}

            {/* Air Density History */}
            {selectedAirData.length > 0 && (
              <div className="bg-card border border-border rounded-xl p-4">
                <h3 className="text-sm text-muted-foreground mb-3 flex items-center gap-2"><Wind className="w-4 h-4" /> Environmental Conditions by Year</h3>
                <ResponsiveContainer width="100%" height={250}>
                  <LineChart data={selectedAirData}>
                    <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="year" tick={{ fill: '#888', fontSize: 11 }} />
                    <YAxis yAxisId="temp" tick={{ fill: '#888', fontSize: 11 }} />
                    <YAxis yAxisId="density" orientation="right" tick={{ fill: '#888', fontSize: 11 }} />
                    <Tooltip content={<CustomTooltip />} />
                    <Line yAxisId="temp" type="monotone" dataKey="avg_temp_c" name="Temp (C)" stroke="#ef4444" strokeWidth={2} dot={{ r: 4 }} />
                    <Line yAxisId="temp" type="monotone" dataKey="avg_humidity_pct" name="Humidity (%)" stroke="#3b82f6" strokeWidth={2} dot={{ r: 4 }} />
                    <Line yAxisId="density" type="monotone" dataKey="air_density_kg_m3" name="Air Density" stroke="#FF8000" strokeWidth={2} dot={{ r: 4 }} />
                  </LineChart>
                </ResponsiveContainer>
                <div className="flex items-center justify-center gap-6 mt-2 text-xs text-muted-foreground">
                  <span className="flex items-center gap-1"><Thermometer className="w-3 h-3 text-red-400" /> Temperature</span>
                  <span className="flex items-center gap-1"><Droplets className="w-3 h-3 text-blue-400" /> Humidity</span>
                  <span className="flex items-center gap-1"><Wind className="w-3 h-3 text-[#FF8000]" /> Air Density</span>
                </div>
              </div>
            )}

            {/* Pit Loss Rank */}
            {selectedPitLoss && pitLossRanking.length > 0 && (() => {
              const rank = pitLossRanking.findIndex(p => p.circuit === selected) + 1;
              const total = pitLossRanking.length;
              const val = selectedPitLoss.est_pit_lane_loss_s;
              const fastest = pitLossRanking[pitLossRanking.length - 1].est_pit_lane_loss_s;
              const slowest = pitLossRanking[0].est_pit_lane_loss_s;
              const pct = ((val - fastest) / (slowest - fastest)) * 100;
              return (
                <div className="bg-card border border-border rounded-xl p-4">
                  <h3 className="text-sm text-muted-foreground mb-3 flex items-center gap-2">
                    <Timer className="w-4 h-4" /> Pit Lane Loss Rank
                  </h3>
                  <div className="flex items-center gap-4 mb-3">
                    <div className="text-center">
                      <div className="text-2xl font-mono font-bold text-[#FF8000]">{rank}<span className="text-sm text-muted-foreground font-normal">/{total}</span></div>
                      <div className="text-[10px] text-muted-foreground">Slowest pit lane</div>
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center justify-between text-[10px] text-muted-foreground mb-1">
                        <span>Fastest ({fastest.toFixed(1)}s)</span>
                        <span>Slowest ({slowest.toFixed(1)}s)</span>
                      </div>
                      <div className="h-2.5 bg-background rounded-full overflow-hidden relative">
                        <div className="h-full rounded-full" style={{ width: '100%', background: 'linear-gradient(90deg, #22c55e, #eab308, #ef4444)' , opacity: 0.3 }} />
                        <div
                          className="absolute top-0 w-3 h-2.5 bg-[#FF8000] rounded-full border-2 border-[#1A1F2E]"
                          style={{ left: `calc(${pct}% - 6px)` }}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              );
            })()}

            {/* ── Historical Race Performance ─────────────────────────── */}
            {historyLoading && (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="w-5 h-5 text-[#FF8000] animate-spin" />
                <span className="text-muted-foreground text-sm ml-2">Loading race history…</span>
              </div>
            )}

            {history && (
              <>
                {/* KPI row */}
                <div className="bg-card border border-border rounded-xl p-4">
                  <h3 className="text-sm text-muted-foreground mb-3 flex items-center gap-2">
                    <Trophy className="w-4 h-4" /> Race History Overview
                  </h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    <PitCard label="Races in DB" value={history.total_races} unit="" precision={0} />
                    <PitCard label="Pole → Win" value={history.pole_stats.rate * 100} unit="%" precision={0} />
                    <PitCard label="Avg Pos. Gained" value={history.positions_gained.avg} unit="pos" />
                    <PitCard label="DNF Rate" value={history.dnf_rate * 100} unit="%" precision={1} />
                  </div>
                </div>

                {/* Winners Table */}
                {history.winners.length > 0 && (
                  <div className="bg-card border border-border rounded-xl p-4">
                    <h3 className="text-sm text-muted-foreground mb-3 flex items-center gap-2">
                      <Flag className="w-4 h-4" /> Race Winners
                    </h3>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="text-muted-foreground text-[11px] border-b border-border">
                            <th className="text-left py-2 px-2">Season</th>
                            <th className="text-left py-2 px-2">Winner</th>
                            <th className="text-left py-2 px-2">Constructor</th>
                            <th className="text-right py-2 px-2">Grid</th>
                          </tr>
                        </thead>
                        <tbody>
                          {[...history.winners].reverse().map(w => (
                            <tr key={w.season} className="border-b border-[rgba(255,128,0,0.04)] hover:bg-secondary/50">
                              <td className="py-2 px-2 font-mono text-muted-foreground">{w.season}</td>
                              <td className="py-2 px-2 text-foreground font-semibold">{w.driver_code}</td>
                              <td className="py-2 px-2 text-muted-foreground">{w.constructor}</td>
                              <td className="py-2 px-2 text-right font-mono">
                                <span className={w.grid === 1 ? 'text-[#FF8000]' : 'text-foreground'}>
                                  P{w.grid}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* Constructor Dominance */}
                {history.top_constructors.length > 0 && (
                  <div className="bg-card border border-border rounded-xl p-4">
                    <h3 className="text-sm text-muted-foreground mb-3 flex items-center gap-2">
                      <Users className="w-4 h-4" /> Constructor Points at This Circuit
                    </h3>
                    <ResponsiveContainer width="100%" height={Math.max(200, history.top_constructors.length * 32)}>
                      <BarChart data={history.top_constructors} layout="vertical" margin={{ left: 80, right: 20 }}>
                        <CartesianGrid stroke="rgba(255,255,255,0.05)" horizontal={false} />
                        <XAxis type="number" tick={{ fill: '#888', fontSize: 11 }} />
                        <YAxis type="category" dataKey="name" tick={{ fill: '#ccc', fontSize: 11 }} width={75} />
                        <Tooltip content={<CustomTooltip />} />
                        <Bar dataKey="points" name="Points" radius={[0, 4, 4, 0]}>
                          {history.top_constructors.map((c, i) => (
                            <Cell key={c.id} fill={i === 0 ? '#FF8000' : i < 3 ? '#FF8000aa' : '#FF800055'} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}

                {/* Podium Kings */}
                {history.top_podiums.length > 0 && (
                  <div className="bg-card border border-border rounded-xl p-4">
                    <h3 className="text-sm text-muted-foreground mb-3 flex items-center gap-2">
                      <TrendingUp className="w-4 h-4" /> Most Podiums at This Circuit
                    </h3>
                    <div className="flex flex-wrap gap-2">
                      {history.top_podiums.map((p, i) => (
                        <div
                          key={p.driver}
                          className="flex items-center gap-2 bg-background border border-border rounded-lg px-3 py-2"
                        >
                          <span className={`font-mono text-sm font-bold ${i === 0 ? 'text-[#FFD700]' : i === 1 ? 'text-[#C0C0C0]' : i === 2 ? 'text-[#CD7F32]' : 'text-muted-foreground'}`}>
                            {p.count}
                          </span>
                          <span className="text-foreground text-sm">{p.driver}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}

            {/* ── WISE Circuit Intelligence Briefing ──────────────────── */}
            {selected && (
              <KexBriefingCard
                title="WISE Circuit Briefing"
                kex={kex}
                loading={kexLoading}
                loadingText="Extracting circuit intelligence\u2026"
              />
            )}
          </>
        )}
      </div>
    </div>
  );
}

function InfoChip({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="flex items-center gap-2 bg-background border border-border rounded-lg px-3 py-2">
      <span className="text-[#FF8000]">{icon}</span>
      <div>
        <div className="text-[10px] text-muted-foreground">{label}</div>
        <div className="text-sm text-foreground font-mono">{value}</div>
      </div>
    </div>
  );
}

function PitCard({ label, value, unit, precision = 2 }: { label: string; value: number | null | undefined; unit: string; precision?: number }) {
  return (
    <div className="bg-background border border-border rounded-lg p-3">
      <div className="text-[11px] text-muted-foreground mb-1">{label}</div>
      <div className="flex items-baseline gap-1">
        <span className="text-foreground font-mono text-lg">{value != null ? value.toFixed(precision) : '—'}</span>
        {unit && <span className="text-xs text-muted-foreground">{unit}</span>}
      </div>
    </div>
  );
}
