import { useState, useEffect, useMemo } from 'react';
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid,
} from 'recharts';
import { MapPin, Loader2, Wind, Thermometer, Droplets, Timer, ChevronRight } from 'lucide-react';
import type { CircuitIntelligence, CircuitPitLoss, RaceAirDensity } from '../types';
import * as api from '../api/circuitIntel';

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

export function CircuitIntel() {
  const [circuits, setCircuits] = useState<CircuitIntelligence[]>([]);
  const [pitLoss, setPitLoss] = useState<CircuitPitLoss[]>([]);
  const [airDensity, setAirDensity] = useState<RaceAirDensity[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

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
      <div className="w-72 shrink-0 bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded-xl overflow-y-auto">
        <div className="p-3 border-b border-[rgba(255,128,0,0.08)]">
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
                  : 'text-muted-foreground hover:bg-[#222838] hover:text-foreground'
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
            <div className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded-xl p-4">
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

            {/* Pit Loss */}
            {selectedPitLoss && (
              <div className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded-xl p-4">
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
              <div className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded-xl p-4">
                <h3 className="text-sm text-muted-foreground mb-3 flex items-center gap-2"><Wind className="w-4 h-4" /> Environmental Conditions by Year</h3>
                <ResponsiveContainer width="100%" height={250}>
                  <LineChart data={selectedAirData}>
                    <CartesianGrid stroke="rgba(255,128,0,0.08)" />
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

            {/* Pit Loss Ranking (all circuits) */}
            <div className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded-xl p-4">
              <h3 className="text-sm text-muted-foreground mb-3 flex items-center gap-2"><Timer className="w-4 h-4" /> Pit Lane Loss Ranking (All Circuits)</h3>
              <ResponsiveContainer width="100%" height={Math.max(300, pitLossRanking.length * 28)}>
                <BarChart data={pitLossRanking} layout="vertical" margin={{ left: 10 }}>
                  <XAxis type="number" tick={{ fill: '#888', fontSize: 11 }} unit="s" />
                  <YAxis type="category" dataKey="circuit" tick={{ fill: '#888', fontSize: 10 }} width={120} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="est_pit_lane_loss_s" name="Pit Loss (s)" radius={[0, 4, 4, 0]}>
                    {pitLossRanking.map((entry) => (
                      <rect
                        key={entry.circuit}
                        fill={entry.circuit === selected ? '#FF8000' : '#444'}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function InfoChip({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="flex items-center gap-2 bg-[#0D1117] border border-[rgba(255,128,0,0.08)] rounded-lg px-3 py-2">
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
    <div className="bg-[#0D1117] border border-[rgba(255,128,0,0.08)] rounded-lg p-3">
      <div className="text-[11px] text-muted-foreground mb-1">{label}</div>
      <div className="flex items-baseline gap-1">
        <span className="text-foreground font-mono text-lg">{value != null ? value.toFixed(precision) : '—'}</span>
        {unit && <span className="text-xs text-muted-foreground">{unit}</span>}
      </div>
    </div>
  );
}
