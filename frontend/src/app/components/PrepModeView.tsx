import { useState, useEffect } from 'react';
import {
  Loader2, Play, AlertTriangle,
} from 'lucide-react';

/* ─── Types ──────────────────────────────────────────────────────── */

interface SystemHealth {
  health: number;
  level: string;
}

interface AnomalyData {
  overall_health: number;
  overall_level: string;
  last_race: string;
  race_count: number;
  systems: Record<string, SystemHealth>;
}

interface EltData {
  baseline_pace_s: number;
  driver_advantage_s: number;
  predicted_pace_s: number;
}

interface DegCurve {
  compound: string;
  temp_band: string;
  coefficients: number[];
  deg_per_lap_s: number | null;
  r_squared: number | null;
  n_stints: number | null;
}

interface CircuitData {
  pit_loss_s?: number;
  air_density?: {
    density_kg_m3: number;
    temperature_c: number;
    humidity_pct: number;
    pressure_hpa: number;
  };
  intelligence?: Record<string, unknown>;
}

interface PrepReport {
  report_type: string;
  race_name: string;
  season: number;
  team: string;
  drivers: string[];
  generated_at: string;
  generation_time_s: number;
  pillars: {
    anomaly: Record<string, AnomalyData>;
    elt_pace: Record<string, EltData>;
    degradation_curves: DegCurve[];
    strategy_simulation: Record<string, unknown> | null;
    circuit: CircuitData;
  };
  briefing: string;
  model_used: string;
}

/* ─── Available races (2024 season) ──────────────────────────────── */

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

/* ─── Shared helpers (match RaceStrategy / CircuitIntel) ─────────── */

const healthColor = (h: number) =>
  h >= 80 ? '#05DF72' : h >= 60 ? '#f59e0b' : '#ef4444';

const levelBg = (level: string) => {
  switch (level) {
    case 'normal': return 'bg-green-500/10 text-green-400 border-green-500/20';
    case 'medium': return 'bg-amber-500/10 text-amber-400 border-amber-500/20';
    case 'high': return 'bg-red-500/10 text-red-400 border-red-500/20';
    case 'critical': return 'bg-red-500/20 text-red-400 border-red-500/30';
    default: return 'bg-muted text-muted-foreground border-border';
  }
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

/* ─── Component ──────────────────────────────────────────────────── */

export function PrepModeView() {
  const [selectedRace, setSelectedRace] = useState('Austrian Grand Prix');
  const [report, setReport] = useState<PrepReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchLatest();
  }, []);

  const fetchLatest = async () => {
    try {
      const res = await fetch(`/api/advantage/prep-mode/latest?season=2024&team=McLaren`);
      if (res.ok) {
        const data = await res.json();
        setReport(data);
        setSelectedRace(data.race_name);
      }
    } catch { /* no cached report */ }
  };

  const generate = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/advantage/prep-mode/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          race_name: selectedRace,
          season: 2024,
          team: 'McLaren',
          force: true,
        }),
      });
      if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
      const data = await res.json();
      setReport(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Generation failed');
    } finally {
      setLoading(false);
    }
  };

  const p = report?.pillars;

  return (
    <div className="space-y-4">
      {/* Controls — inline selector row matching CircuitIntel / RaceStrategy */}
      <div className="flex items-center gap-3">
        <select
          title="Select race"
          value={selectedRace}
          onChange={(e) => setSelectedRace(e.target.value)}
          className="bg-card border border-border rounded-lg px-3 py-1.5 text-sm text-foreground focus:outline-none focus:border-primary/30"
        >
          {RACES_2024.map(r => (
            <option key={r} value={r}>{r}</option>
          ))}
        </select>
        <button
          type="button"
          onClick={generate}
          disabled={loading}
          className="flex items-center gap-2 px-4 py-1.5 bg-primary text-primary-foreground rounded-lg text-[12px] font-medium hover:bg-primary/90 disabled:opacity-50 transition-colors"
        >
          {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
          {loading ? 'Generating...' : 'Generate'}
        </button>
        {report && (
          <span className="text-[10px] text-muted-foreground font-mono">
            {report.generation_time_s}s | {report.model_used} | {report.team} · {report.drivers.join(', ')}
          </span>
        )}
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-3 text-[12px] text-red-400 flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          {error}
        </div>
      )}

      {loading && !report && (
        <div className="bg-card border border-border rounded-lg p-12 text-center">
          <Loader2 className="w-8 h-8 animate-spin text-primary mx-auto mb-3" />
          <div className="text-[12px] text-muted-foreground">Generating pre-race intelligence...</div>
          <div className="text-[10px] text-muted-foreground/60 mt-1">Anomaly + ELT + Degradation + LLM synthesis</div>
        </div>
      )}

      {report && (
        <>
          {/* ── LLM Briefing ── */}
          <Divider label="PRE-RACE BRIEFING" />
          <div className="bg-card border border-border rounded-lg p-4">
            <div className="text-[12px] text-foreground/90 whitespace-pre-line leading-relaxed">
              {report.briefing}
            </div>
          </div>

          {/* ── System Health ── */}
          {p && Object.keys(p.anomaly).length > 0 && (
            <>
              <Divider label="SYSTEM HEALTH" />
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {Object.entries(p.anomaly).map(([driver, data]) => (
                  <div key={driver} className="bg-card border border-border rounded-lg p-3">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-[12px] font-medium">{driver}</span>
                      <div className="flex items-center gap-2">
                        <span
                          className="text-lg font-mono font-bold"
                          style={{ color: healthColor(data.overall_health) }}
                        >
                          {data.overall_health}%
                        </span>
                        <span className={`text-[9px] px-1.5 py-0.5 rounded border ${levelBg(data.overall_level)}`}>
                          {data.overall_level}
                        </span>
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-1.5">
                      {Object.entries(data.systems).map(([sys, sh]) => (
                        <div key={sys} className="flex items-center justify-between text-[10px] px-2 py-1 rounded bg-background border border-border/50">
                          <span className="text-muted-foreground truncate mr-2">{sys}</span>
                          <span className="font-mono" style={{ color: healthColor(sh.health) }}>
                            {sh.health}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}

          {/* ── ELT Pace ── */}
          {p && Object.keys(p.elt_pace).length > 0 && (
            <>
              <Divider label="PACE PREDICTION" />
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                {Object.entries(p.elt_pace).map(([driver, data]) => (
                  <div key={driver} className="bg-card border border-border rounded-lg p-3 text-center">
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

          {/* ── Degradation Curves ── */}
          {p && p.degradation_curves.length > 0 && (
            <>
              <Divider label="TYRE DEGRADATION" />
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                {p.degradation_curves
                  .filter(c => c.temp_band === 'all')
                  .map((c, i) => {
                    const compoundColor: Record<string, string> = {
                      SOFT: '#ef4444', MEDIUM: '#f59e0b', HARD: '#888', INTERMEDIATE: '#22c55e', WET: '#3b82f6',
                    };
                    return (
                      <div key={i} className="bg-card border border-border rounded-lg p-3 text-center">
                        <div
                          className="text-[10px] font-mono font-bold mb-1"
                          style={{ color: compoundColor[c.compound] || '#888' }}
                        >
                          {c.compound}
                        </div>
                        <div className="text-sm font-mono text-foreground">
                          {c.deg_per_lap_s != null ? `${c.deg_per_lap_s.toFixed(4)} s/lap` : '—'}
                        </div>
                        {c.r_squared != null && (
                          <div className="text-[9px] text-muted-foreground">R²={c.r_squared.toFixed(3)}</div>
                        )}
                        {c.n_stints != null && (
                          <div className="text-[9px] text-muted-foreground">{c.n_stints} stints</div>
                        )}
                      </div>
                    );
                  })}
              </div>
            </>
          )}

          {/* ── Circuit Conditions ── */}
          {p && (p.circuit.pit_loss_s || p.circuit.air_density) && (
            <>
              <Divider label="CIRCUIT CONDITIONS" />
              <div className="flex gap-3">
                {p.circuit.pit_loss_s != null && (
                  <div className="bg-card border border-border rounded-lg px-4 py-2 text-center">
                    <div className="text-[9px] text-muted-foreground">Pit Loss</div>
                    <div className="text-sm font-mono font-bold text-primary">{p.circuit.pit_loss_s.toFixed(1)}s</div>
                  </div>
                )}
                {p.circuit.air_density && (
                  <>
                    <div className="bg-card border border-border rounded-lg px-4 py-2 text-center">
                      <div className="text-[9px] text-muted-foreground">Temperature</div>
                      <div className="text-sm font-mono font-bold">{p.circuit.air_density.temperature_c}°C</div>
                    </div>
                    <div className="bg-card border border-border rounded-lg px-4 py-2 text-center">
                      <div className="text-[9px] text-muted-foreground">Air Density</div>
                      <div className="text-sm font-mono font-bold">{p.circuit.air_density.density_kg_m3} kg/m³</div>
                    </div>
                    <div className="bg-card border border-border rounded-lg px-4 py-2 text-center">
                      <div className="text-[9px] text-muted-foreground">Humidity</div>
                      <div className="text-sm font-mono font-bold">{p.circuit.air_density.humidity_pct}%</div>
                    </div>
                  </>
                )}
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}
