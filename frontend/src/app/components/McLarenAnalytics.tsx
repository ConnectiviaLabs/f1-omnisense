import { useState, useEffect, useMemo } from 'react';
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, AreaChart, Area, ReferenceLine,
} from 'recharts';
import {
  Trophy, TrendingUp, Loader2, Flag, Users, Timer, Gauge, Activity, Disc, Shield,
} from 'lucide-react';
import { HealthGauge } from './HealthGauge';
import {
  type VehicleData, parseAnomalyDrivers, levelColor, MAINTENANCE_LABELS,
} from './anomalyHelpers';

// ─── Types ──────────────────────────────────────────────────────────
interface CarSummary {
  race: string; avgSpeed: number; topSpeed: number; avgRPM: number;
  maxRPM: number; avgThrottle: number; brakePct: number; drsPct: number;
  compounds: string[]; samples: number;
}

// ─── Shared helpers ─────────────────────────────────────────────────
const teamColors: Record<string, string> = {
  'Red Bull': '#3671C6', 'McLaren': '#FF8000', 'Ferrari': '#E8002D',
  'Mercedes': '#27F4D2', 'Aston Martin': '#229971', 'Alpine': '#FF87BC',
  'Williams': '#64C4FF', 'RB': '#6692FF', 'Kick Sauber': '#52E252',
  'Haas F1 Team': '#B6BABD',
};

const compoundColors: Record<string, string> = {
  SOFT: '#ef4444', MEDIUM: '#f59e0b', HARD: '#e8e8f0', INTERMEDIATE: '#22c55e', WET: '#3b82f6',
};

// Helper: match season regardless of string/number type in MongoDB
const matchSeason = (docSeason: any, year: string) => String(docSeason) === year;

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-[#0D1117] border border-[rgba(255,128,0,0.2)] rounded-lg p-2 text-[12px]">
      <div className="text-muted-foreground mb-1">{label}</div>
      {payload.map((e: any, i: number) => (
        <div key={i} className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full" style={{ backgroundColor: e.color }} />
          <span className="text-muted-foreground">{e.name}:</span>
          <span className="text-foreground font-mono">{typeof e.value === 'number' ? e.value.toFixed(1) : e.value}</span>
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
    <div className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded-xl p-3">
      <div className="flex items-center gap-2 mb-1.5">
        {icon}
        <span className="text-[12px] text-muted-foreground tracking-wider">{label}</span>
      </div>
      <div className="text-lg font-mono text-foreground">{value}</div>
      <div className="text-[12px] text-muted-foreground mt-0.5">{detail}</div>
    </div>
  );
}

const SYSTEM_ICONS: Record<string, React.ElementType> = { 'Speed': Gauge, 'Lap Pace': Activity, 'Tyre Management': Disc };

// ─── Component ──────────────────────────────────────────────────────
export function McLarenAnalytics() {
  const [year, setYear] = useState('2024');
  const [availableYears, setAvailableYears] = useState<string[]>(['2024', '2023']);

  // ── Fetch available seasons from MongoDB ──────────────────────────
  useEffect(() => {
    fetch('/api/jolpica/seasons')
      .then(r => r.ok ? r.json() : [])
      .then((seasons: any[]) => {
        const years = seasons.map(s => String(s)).filter(s => /^\d{4}$/.test(s)).sort().reverse();
        if (years.length > 0) setAvailableYears(years);
      })
      .catch(() => {});
  }, []);

  // ── MongoDB: race_results (raw) — derive standings + results ──────
  const [allRaceResults, setAllRaceResults] = useState<any[]>([]);
  const [dataLoading, setDataLoading] = useState(true);

  useEffect(() => {
    setDataLoading(true);
    fetch('/api/jolpica/race_results')
      .then(r => r.ok ? r.json() : [])
      .then(d => setAllRaceResults(d))
      .catch(() => {})
      .finally(() => setDataLoading(false));
  }, []);

  // Filter race results for selected year (handle string/number season)
  const raceResults = useMemo(() =>
    allRaceResults.filter(r => matchSeason(r.season, year)).sort((a, b) => Number(a.round) - Number(b.round)),
    [allRaceResults, year]);

  // ── MongoDB: constructor_standings ─────────────────────────────────
  const [allConstructorStandings, setAllConstructorStandings] = useState<any[]>([]);
  useEffect(() => {
    fetch('/api/jolpica/constructor_standings')
      .then(r => r.ok ? r.json() : [])
      .then(d => setAllConstructorStandings(d))
      .catch(() => {});
  }, []);

  const constructorStandings = useMemo(() =>
    allConstructorStandings.filter(s => matchSeason(s.season, year)),
    [allConstructorStandings, year]);

  // ── MongoDB: driver_standings ──────────────────────────────────────
  const [allDriverStandings, setAllDriverStandings] = useState<any[]>([]);
  useEffect(() => {
    fetch('/api/jolpica/driver_standings')
      .then(r => r.ok ? r.json() : [])
      .then(d => setAllDriverStandings(d))
      .catch(() => {});
  }, []);

  const driverStandings = useMemo(() =>
    allDriverStandings.filter(s => matchSeason(s.season, year)),
    [allDriverStandings, year]);

  // ── MongoDB: Telemetry race summaries ─────────────────────────────
  const [norSummary, setNorSummary] = useState<CarSummary[]>([]);
  const [piaSummary, setPiaSummary] = useState<CarSummary[]>([]);
  const [telLoading, setTelLoading] = useState(true);

  useEffect(() => {
    setTelLoading(true);
    Promise.all([
      fetch(`/api/local/mccar-summary/${year}/NOR`).then(r => r.ok ? r.json() : []),
      fetch(`/api/local/mccar-summary/${year}/PIA`).then(r => r.ok ? r.json() : []),
    ]).then(([n, p]) => { setNorSummary(n); setPiaSummary(p); })
      .catch(() => {})
      .finally(() => setTelLoading(false));
  }, [year]);

  // ── MongoDB: Pit stops from jolpica_pit_stops collection ──────────
  const [pitStopsRaw, setPitStopsRaw] = useState<any[]>([]);
  useEffect(() => {
    fetch(`/api/jolpica/pit_stops?season=${year}`)
      .then(r => r.ok ? r.json() : [])
      .then(d => setPitStopsRaw(d))
      .catch(() => {});
  }, [year]);

  // ── MongoDB: Stints + sessions from openf1 collections ───────────
  const [stintsRaw, setStintsRaw] = useState<any[]>([]);
  const [sessionsRaw, setSessionsRaw] = useState<any[]>([]);
  useEffect(() => {
    Promise.all([
      fetch('/api/openf1/stints').then(r => r.ok ? r.json() : []),
      fetch('/api/openf1/sessions').then(r => r.ok ? r.json() : []),
    ]).then(([st, sess]) => { setStintsRaw(st); setSessionsRaw(sess); })
      .catch(() => {});
  }, []);

  // ── MongoDB: Anomaly scores ───────────────────────────────────────
  const [anomalyVehicles, setAnomalyVehicles] = useState<VehicleData[]>([]);
  useEffect(() => {
    fetch('/api/pipeline/anomaly').then(r => r.json())
      .then(d => setAnomalyVehicles(parseAnomalyDrivers(d)))
      .catch(() => {});
  }, []);

  // ── MongoDB: Rivals via omni analytics ────────────────────────────
  const [rivalsData, setRivalsData] = useState<any>(null);
  useEffect(() => {
    fetch('/api/omni/analytics/rivals/NOR')
      .then(r => r.ok ? r.json() : null)
      .then(d => setRivalsData(d))
      .catch(() => {});
  }, []);

  // ─── Derived data ─────────────────────────────────────────────────

  const norVehicle = anomalyVehicles.find(v => v.code === 'NOR');
  const piaVehicle = anomalyVehicles.find(v => v.code === 'PIA');

  // KPIs from MongoDB standings
  const kpiStats = useMemo(() => {
    const norS = driverStandings.find(s => s.Driver?.code === 'NOR');
    const piaS = driverStandings.find(s => s.Driver?.code === 'PIA');
    const mcS = constructorStandings.find(s =>
      s.Constructor?.name?.toLowerCase().includes('mclaren'));

    // Last race points gained for McLaren
    const lastRace = raceResults.slice(-1)[0];
    const lastGained = lastRace
      ? (lastRace.Results || [])
          .filter((r: any) => r.Constructor?.name?.toLowerCase().includes('mclaren'))
          .reduce((sum: number, r: any) => sum + Number(r.points || 0), 0)
      : 0;

    return {
      norPts: norS?.points ?? '—', norPos: norS?.position ?? '—',
      piaPts: piaS?.points ?? '—', piaPos: piaS?.position ?? '—',
      teamPts: mcS?.points ?? '—', teamPos: mcS?.position ?? '—',
      lastGained: String(lastGained),
    };
  }, [driverStandings, constructorStandings, raceResults]);

  // Points progression from MongoDB race_results
  const pointsProgression = useMemo(() => {
    if (!raceResults.length) return [];
    let norCum = 0, piaCum = 0, teamCum = 0;
    return raceResults.map(race => {
      const gpName = (race.raceName || '').replace(' Grand Prix', '');
      let raceTeamPts = 0;
      for (const result of (race.Results || [])) {
        const code = result.Driver?.code;
        const pts = Number(result.points || 0);
        const isMcLaren = result.Constructor?.name?.toLowerCase().includes('mclaren');
        if (code === 'NOR') norCum += pts;
        if (code === 'PIA') piaCum += pts;
        if (isMcLaren) raceTeamPts += pts;
      }
      teamCum += raceTeamPts;
      return { race: gpName, NOR: norCum, PIA: piaCum, Team: teamCum };
    });
  }, [raceResults]);

  // Telemetry merged
  const telemetryData = useMemo(() => {
    const raceMap = new Map<string, any>();
    norSummary.forEach(r => {
      raceMap.set(r.race, { race: r.race.slice(0, 12), norTop: r.topSpeed, norThrottle: r.avgThrottle, norBrake: r.brakePct, norDrs: r.drsPct });
    });
    piaSummary.forEach(r => {
      const e = raceMap.get(r.race) ?? { race: r.race.slice(0, 12) };
      e.piaTop = r.topSpeed; e.piaThrottle = r.avgThrottle; e.piaBrake = r.brakePct; e.piaDrs = r.drsPct;
      raceMap.set(r.race, e);
    });
    return Array.from(raceMap.values());
  }, [norSummary, piaSummary]);

  // Pit stop data from MongoDB jolpica_pit_stops
  const pitStopData = useMemo(() => {
    if (!pitStopsRaw.length) return [];
    const raceStops = new Map<string, { race: string; NOR: number[]; PIA: number[] }>();
    pitStopsRaw.forEach(stop => {
      const race = (stop.raceName || '').replace(' Grand Prix', '').slice(0, 12);
      const driver = stop.driverId || '';
      const dur = parseFloat(stop.duration || '0');
      if (dur <= 0 || dur > 60 || !race) return;
      if (!raceStops.has(race)) raceStops.set(race, { race, NOR: [], PIA: [] });
      const entry = raceStops.get(race)!;
      if (driver.includes('norris')) entry.NOR.push(dur);
      else if (driver.includes('piastri')) entry.PIA.push(dur);
    });
    return Array.from(raceStops.values()).map(r => ({
      race: r.race,
      NOR: r.NOR.length ? +(r.NOR.reduce((a, b) => a + b, 0) / r.NOR.length).toFixed(1) : null,
      PIA: r.PIA.length ? +(r.PIA.reduce((a, b) => a + b, 0) / r.PIA.length).toFixed(1) : null,
    }));
  }, [pitStopsRaw]);

  const avgPitDuration = useMemo(() => {
    const all = pitStopData.flatMap(r => [r.NOR, r.PIA].filter(Boolean) as number[]);
    return all.length ? +(all.reduce((a, b) => a + b, 0) / all.length).toFixed(1) : 0;
  }, [pitStopData]);

  // Tire compound grid from MongoDB openf1_stints + openf1_sessions
  const tireGrid = useMemo(() => {
    if (!stintsRaw.length || !sessionsRaw.length) return { races: [] as string[], NOR: {} as Record<string, string[]>, PIA: {} as Record<string, string[]> };
    // Filter sessions for selected year
    const yearSessions = sessionsRaw.filter(s => s.session_type === 'Race' && matchSeason(s.year, year));
    const sessionMap = new Map<number, string>();
    yearSessions.forEach(s => {
      sessionMap.set(s.session_key, (s.circuit_short_name || s.meeting_name || '').slice(0, 10));
    });
    const yearSessionKeys = new Set(yearSessions.map(s => s.session_key));
    // Driver numbers: NOR=4, PIA=81
    const driverMap: Record<number, 'NOR' | 'PIA'> = { 4: 'NOR', 81: 'PIA' };
    const races = new Set<string>();
    const nor: Record<string, string[]> = {};
    const pia: Record<string, string[]> = {};
    stintsRaw.forEach(stint => {
      if (!yearSessionKeys.has(stint.session_key)) return;
      const race = sessionMap.get(stint.session_key);
      if (!race) return;
      const code = driverMap[stint.driver_number];
      if (!code) return;
      races.add(race);
      const target = code === 'NOR' ? nor : pia;
      if (!target[race]) target[race] = [];
      const compound = (stint.compound || '').toUpperCase();
      if (compound && !target[race].includes(compound)) target[race].push(compound);
    });
    return { races: Array.from(races), NOR: nor, PIA: pia };
  }, [stintsRaw, sessionsRaw, year]);

  // NOR health trend
  const norHealthTrend = useMemo(() => {
    if (!norVehicle?.races?.length) return [];
    return norVehicle.races.slice(-8).map(r => ({
      race: r.race?.slice(0, 8) ?? '',
      speed: r.systems['Speed']?.health ?? 0,
      pace: r.systems['Lap Pace']?.health ?? 0,
      tyre: r.systems['Tyre Management']?.health ?? 0,
    }));
  }, [norVehicle]);

  // Rivals chart data
  const rivalsChart = useMemo(() => {
    if (!rivalsData?.rivals) return [];
    return rivalsData.rivals.slice(0, 6).map((r: any) => ({
      code: r.code, anomalyPct: r.anomaly_pct,
      fill: r.code === 'NOR' ? '#FF8000' : '#6b7280',
    }));
  }, [rivalsData]);

  // Race results table
  const raceResultsTable = useMemo(() => {
    if (!raceResults.length) return [];
    return raceResults.slice(-8).reverse().map(race => {
      const results = race.Results || [];
      const norResult = results.find((r: any) => r.Constructor?.name?.toLowerCase().includes('mclaren') && r.Driver?.code === 'NOR');
      const piaResult = results.find((r: any) => r.Constructor?.name?.toLowerCase().includes('mclaren') && r.Driver?.code === 'PIA');
      const winner = results[0];
      const gpName = (race.raceName || '').replace(' Grand Prix', '');
      const gained = results
        .filter((r: any) => r.Constructor?.name?.toLowerCase().includes('mclaren'))
        .reduce((sum: number, r: any) => sum + Number(r.points || 0), 0);
      return {
        round: race.round, gp: gpName,
        norPos: norResult?.position ?? '—', piaPos: piaResult?.position ?? '—',
        winner: winner?.Driver?.code ?? '—', winnerTeam: winner?.Constructor?.name ?? '',
        gained: String(gained),
      };
    });
  }, [raceResults]);

  // ─── Render ───────────────────────────────────────────────────────
  return (
    <div className="space-y-4">
      {/* Header + Year Selector */}
      <div className="flex items-center justify-between">
        <div>
          <p className="text-[10px] text-[#FF8000]/60 tracking-[0.25em] font-semibold">MCLAREN F1 TEAM</p>
          <h2 className="text-sm text-foreground">Competitive Intelligence</h2>
        </div>
        <div className="flex items-center gap-1 bg-[#1A1F2E] rounded-lg p-0.5 border border-[rgba(255,128,0,0.12)]">
          {availableYears.map(y => (
            <button key={y} onClick={() => setYear(y)}
              className={`text-sm px-3 py-1.5 rounded-md transition-all ${year === y ? 'bg-[#FF8000]/10 text-[#FF8000]' : 'text-muted-foreground hover:text-foreground'}`}>
              {y}
            </button>
          ))}
        </div>
      </div>

      {/* Section 1: KPI Strip */}
      {dataLoading ? (
        <div className="flex items-center justify-center py-6"><Loader2 className="w-5 h-5 text-[#FF8000] animate-spin" /></div>
      ) : (
        <div className="grid grid-cols-4 gap-3">
          <KPI icon={<Users className="w-4 h-4 text-[#FF8000]" />} label="NOR" value={`P${kpiStats.norPos}`} detail={`${kpiStats.norPts} pts`} />
          <KPI icon={<Users className="w-4 h-4 text-cyan-400" />} label="PIA" value={`P${kpiStats.piaPts} pts`} detail={`P${kpiStats.piaPos}`} />
          <KPI icon={<Trophy className="w-4 h-4 text-green-400" />} label="Constructors" value={`P${kpiStats.teamPos}`} detail={`${kpiStats.teamPts} pts`} />
          <KPI icon={<Flag className="w-4 h-4 text-amber-400" />} label="Last Race" value={`+${kpiStats.lastGained}`} detail="pts gained" />
        </div>
      )}

      {/* Section 2: System Health Strip */}
      {(norVehicle || piaVehicle) && (
        <>
          <Divider label="SYSTEM HEALTH" />
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {[{ v: norVehicle, label: 'NOR', color: '#FF8000' }, { v: piaVehicle, label: 'PIA', color: '#00d4ff' }].map(({ v, label, color }) => {
              if (!v) return null;
              return (
                <div key={label} className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded-xl p-4">
                  <div className="flex items-center gap-4">
                    <HealthGauge value={v.overallHealth} size={60} strokeWidth={5} showLabel={false} />
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-sm font-medium" style={{ color }}>{label}</span>
                        <span className="text-[11px] font-mono" style={{ color: levelColor(v.level) }}>
                          {v.overallHealth}% — {v.level.toUpperCase()}
                        </span>
                      </div>
                      <div className="space-y-1.5">
                        {v.systems.map(sys => {
                          const Icon = SYSTEM_ICONS[sys.name] ?? Gauge;
                          const maint = MAINTENANCE_LABELS[sys.maintenanceAction ?? 'none'] ?? MAINTENANCE_LABELS.none;
                          return (
                            <div key={sys.name} className="flex items-center gap-2">
                              <Icon className="w-3 h-3 shrink-0" style={{ color: levelColor(sys.level) }} />
                              <span className="text-[11px] text-muted-foreground w-24 truncate">{sys.name}</span>
                              <div className="flex-1 h-1.5 bg-[#222838] rounded-full overflow-hidden">
                                <div className="h-full rounded-full transition-all duration-700" style={{ width: `${sys.health}%`, background: levelColor(sys.level) }} />
                              </div>
                              <span className="text-[11px] font-mono w-8 text-right" style={{ color: levelColor(sys.level) }}>{sys.health}%</span>
                              {sys.maintenanceAction && sys.maintenanceAction !== 'none' && (
                                <span className="text-[9px] px-1.5 py-0.5 rounded" style={{ background: `${maint.color}15`, color: maint.color }}>{maint.label}</span>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </>
      )}

      {/* Section 3: Championship Trajectory */}
      <Divider label="CHAMPIONSHIP TRAJECTORY" />
      <div className="grid grid-cols-12 gap-4">
        <div className="col-span-8 bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded-xl p-4">
          <h3 className="text-sm text-foreground mb-1">Points Progression</h3>
          <p className="text-[12px] text-muted-foreground mb-3">NOR + PIA + Constructors cumulative</p>
          <div className="h-[260px]">
            {pointsProgression.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={pointsProgression}>
                  <defs>
                    <linearGradient id="norGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#FF8000" stopOpacity={0.3} />
                      <stop offset="100%" stopColor="#FF8000" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="piaGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#00d4ff" stopOpacity={0.3} />
                      <stop offset="100%" stopColor="#00d4ff" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="teamGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#22c55e" stopOpacity={0.2} />
                      <stop offset="100%" stopColor="#22c55e" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,128,0,0.08)" />
                  <XAxis dataKey="race" stroke="#8888a0" fontSize={9} angle={-30} textAnchor="end" height={50} />
                  <YAxis stroke="#8888a0" fontSize={10} />
                  <Tooltip content={<CustomTooltip />} />
                  <Area type="monotone" dataKey="Team" stroke="#22c55e" fill="url(#teamGrad)" strokeWidth={1.5} dot={false} name="Constructors" strokeDasharray="4 2" />
                  <Area type="monotone" dataKey="NOR" stroke="#FF8000" fill="url(#norGrad)" strokeWidth={2} dot={false} name="NOR" />
                  <Area type="monotone" dataKey="PIA" stroke="#00d4ff" fill="url(#piaGrad)" strokeWidth={2} dot={false} name="PIA" />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-full text-muted-foreground text-sm">No data</div>
            )}
          </div>
        </div>

        <div className="col-span-4 bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded-xl p-4">
          <h3 className="text-sm text-foreground mb-1">Constructors Gap</h3>
          <p className="text-[12px] text-muted-foreground mb-3">Points to nearest rivals</p>
          <div className="space-y-2">
            {constructorStandings.slice(0, 5).map((team, i) => {
              const pts = Number(team.points);
              const leaderPts = Number(constructorStandings[0]?.points ?? 0);
              const isMcLaren = team.Constructor?.name?.toLowerCase().includes('mclaren');
              return (
                <div key={team.Constructor?.name || i} className={`flex items-center gap-2 p-2 rounded-lg ${isMcLaren ? 'bg-[#FF8000]/10 border border-[#FF8000]/20' : ''}`}>
                  <span className="text-[12px] text-muted-foreground w-5 font-mono">P{i + 1}</span>
                  <div className="w-2 h-5 rounded-full" style={{ backgroundColor: teamColors[team.Constructor?.name] ?? '#555' }} />
                  <span className={`text-sm flex-1 truncate ${isMcLaren ? 'text-[#FF8000] font-medium' : 'text-foreground'}`}>
                    {team.Constructor?.name}
                  </span>
                  <span className="text-sm font-mono text-foreground">{pts}</span>
                  {i > 0 && (
                    <span className="text-[11px] font-mono text-muted-foreground w-12 text-right">
                      {pts - leaderPts}
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Section 4: Car Telemetry Intelligence */}
      <Divider label="CAR TELEMETRY INTELLIGENCE" />
      {telLoading ? (
        <div className="flex items-center justify-center py-8"><Loader2 className="w-5 h-5 text-[#FF8000] animate-spin" /></div>
      ) : telemetryData.length > 0 ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded-xl p-4">
            <h3 className="text-sm text-foreground mb-1">Top Speed</h3>
            <p className="text-[12px] text-muted-foreground mb-3">NOR vs PIA — km/h per race</p>
            <div className="h-[220px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={telemetryData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,128,0,0.08)" />
                  <XAxis dataKey="race" stroke="#8888a0" fontSize={9} angle={-30} textAnchor="end" height={50} />
                  <YAxis stroke="#8888a0" fontSize={10} domain={['dataMin - 5', 'dataMax + 5']} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="norTop" fill="#FF8000" opacity={0.8} radius={[3, 3, 0, 0]} name="NOR" />
                  <Bar dataKey="piaTop" fill="#00d4ff" opacity={0.8} radius={[3, 3, 0, 0]} name="PIA" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded-xl p-4">
            <h3 className="text-sm text-foreground mb-1">Driving Style</h3>
            <p className="text-[12px] text-muted-foreground mb-3">Season avg throttle / brake / DRS %</p>
            <div className="h-[220px]">
              {(() => {
                const avg = (arr: CarSummary[], key: keyof CarSummary) => {
                  const vals = arr.map(r => Number(r[key])).filter(v => v > 0);
                  return vals.length ? +(vals.reduce((a, b) => a + b, 0) / vals.length).toFixed(1) : 0;
                };
                const data = [
                  { metric: 'Throttle %', NOR: avg(norSummary, 'avgThrottle'), PIA: avg(piaSummary, 'avgThrottle') },
                  { metric: 'Brake %', NOR: avg(norSummary, 'brakePct'), PIA: avg(piaSummary, 'brakePct') },
                  { metric: 'DRS %', NOR: avg(norSummary, 'drsPct'), PIA: avg(piaSummary, 'drsPct') },
                ];
                return (
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={data} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,128,0,0.08)" />
                      <XAxis type="number" stroke="#8888a0" fontSize={10} unit="%" />
                      <YAxis dataKey="metric" type="category" stroke="#8888a0" fontSize={11} width={80} />
                      <Tooltip content={<CustomTooltip />} />
                      <Bar dataKey="NOR" fill="#FF8000" opacity={0.8} radius={[0, 4, 4, 0]} name="NOR" />
                      <Bar dataKey="PIA" fill="#00d4ff" opacity={0.8} radius={[0, 4, 4, 0]} name="PIA" />
                    </BarChart>
                  </ResponsiveContainer>
                );
              })()}
            </div>
          </div>
        </div>
      ) : (
        <div className="text-center py-6 text-muted-foreground text-sm">No telemetry data for {year}</div>
      )}

      {/* Section 5: Pit Wall Strategy */}
      <Divider label="PIT WALL STRATEGY" />
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded-xl p-4">
          <h3 className="text-sm text-foreground mb-1 flex items-center gap-2"><Timer className="w-3.5 h-3.5 text-[#FF8000]" />Pit Stop Duration</h3>
          <p className="text-[12px] text-muted-foreground mb-3">Average per race (seconds)</p>
          <div className="h-[220px]">
            {pitStopData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={pitStopData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,128,0,0.08)" />
                  <XAxis dataKey="race" stroke="#8888a0" fontSize={9} angle={-30} textAnchor="end" height={50} />
                  <YAxis stroke="#8888a0" fontSize={10} />
                  <Tooltip content={<CustomTooltip />} />
                  <ReferenceLine y={avgPitDuration} stroke="rgba(255,128,0,0.3)" strokeDasharray="4 2" />
                  <Line type="monotone" dataKey="NOR" stroke="#FF8000" strokeWidth={2} dot={{ r: 3, fill: '#FF8000' }} name="NOR" connectNulls />
                  <Line type="monotone" dataKey="PIA" stroke="#00d4ff" strokeWidth={2} dot={{ r: 3, fill: '#00d4ff' }} name="PIA" connectNulls />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-full text-muted-foreground text-sm">No pit data</div>
            )}
          </div>
        </div>

        <div className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded-xl p-4">
          <h3 className="text-sm text-foreground mb-1 flex items-center gap-2"><Shield className="w-3.5 h-3.5 text-[#FF8000]" />Tire Strategy</h3>
          <p className="text-[12px] text-muted-foreground mb-3">Compound selection per race</p>
          {tireGrid.races.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full text-[11px]">
                <thead>
                  <tr className="border-b border-[rgba(255,128,0,0.08)]">
                    <th className="text-left py-1 px-1 text-muted-foreground font-normal w-10">Driver</th>
                    {tireGrid.races.map(r => (
                      <th key={r} className="text-center py-1 px-0.5 text-muted-foreground font-normal">{r.slice(0, 5)}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {[{ label: 'NOR', data: tireGrid.NOR, color: '#FF8000' }, { label: 'PIA', data: tireGrid.PIA, color: '#00d4ff' }].map(({ label, data, color }) => (
                    <tr key={label} className="border-b border-[rgba(255,128,0,0.04)]">
                      <td className="py-1.5 px-1 font-mono font-medium" style={{ color }}>{label}</td>
                      {tireGrid.races.map(race => (
                        <td key={race} className="py-1 px-0.5 text-center">
                          <div className="flex gap-0.5 justify-center">
                            {(data[race] ?? []).map((c, i) => (
                              <span key={i} className="w-2.5 h-2.5 rounded-full inline-block" style={{ background: compoundColors[c] ?? '#555' }} title={c} />
                            ))}
                          </div>
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="flex items-center justify-center h-[180px] text-muted-foreground text-sm">No tire data</div>
          )}
        </div>
      </div>

      {/* Section 6: Competitive Intelligence */}
      <Divider label="COMPETITIVE INTELLIGENCE" />
      <div className="grid grid-cols-12 gap-4">
        <div className="col-span-5 bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded-xl p-4">
          <h3 className="text-sm text-foreground mb-3 flex items-center gap-2"><TrendingUp className="w-3.5 h-3.5 text-[#FF8000]" />NOR Health Trend</h3>
          {norHealthTrend.length > 0 ? (
            <div className="space-y-3">
              {[{ key: 'speed', label: 'Speed', color: '#FF8000' }, { key: 'pace', label: 'Lap Pace', color: '#00d4ff' }, { key: 'tyre', label: 'Tyre Mgmt', color: '#22c55e' }].map(sys => (
                <div key={sys.key}>
                  <span className="text-[10px] text-muted-foreground">{sys.label}</span>
                  <ResponsiveContainer width="100%" height={60}>
                    <AreaChart data={norHealthTrend} margin={{ top: 2, right: 2, bottom: 0, left: 2 }}>
                      <defs>
                        <linearGradient id={`ht-${sys.key}`} x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor={sys.color} stopOpacity={0.3} />
                          <stop offset="100%" stopColor={sys.color} stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <XAxis dataKey="race" tick={false} axisLine={false} />
                      <YAxis domain={[0, 100]} tick={false} axisLine={false} width={0} />
                      <Area type="monotone" dataKey={sys.key} stroke={sys.color} fill={`url(#ht-${sys.key})`} strokeWidth={1.5} dot={false} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-muted-foreground text-sm">No health data</div>
          )}
        </div>

        <div className="col-span-7 bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded-xl p-4">
          <h3 className="text-sm text-foreground mb-1">NOR vs Field</h3>
          <p className="text-[12px] text-muted-foreground mb-3">Anomaly rate comparison — higher = more anomalies</p>
          {rivalsChart.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={[
                ...(rivalsData?.target ? [{ code: rivalsData.target.code || 'NOR', anomalyPct: rivalsData.target.anomaly_pct }] : []),
                ...rivalsChart,
              ].sort((a, b) => b.anomalyPct - a.anomalyPct)} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,128,0,0.08)" />
                <XAxis type="number" stroke="#8888a0" fontSize={10} unit="%" />
                <YAxis dataKey="code" type="category" stroke="#8888a0" fontSize={11} width={40} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="anomalyPct" radius={[0, 4, 4, 0]} name="Anomaly %">
                  {[
                    ...(rivalsData?.target ? [{ code: rivalsData.target.code || 'NOR', anomalyPct: rivalsData.target.anomaly_pct }] : []),
                    ...rivalsChart,
                  ].sort((a, b) => b.anomalyPct - a.anomalyPct).map((e, i) => (
                    <Cell key={i} fill={e.code === 'NOR' ? '#FF8000' : '#6b7280'} opacity={0.8} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-[220px] text-muted-foreground text-sm">Loading rival data...</div>
          )}
        </div>
      </div>

      {/* Section 7: Race Results */}
      <Divider label="RACE RESULTS" />
      <div className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded-xl p-4">
        <div className="space-y-1">
          <div className="grid grid-cols-[40px_160px_60px_60px_60px_100px_60px] gap-2 px-2 py-1 text-[11px] text-muted-foreground tracking-wider">
            <span>RND</span><span>GRAND PRIX</span><span>NOR</span><span>PIA</span><span>WIN</span><span>TEAM</span><span>+PTS</span>
          </div>
          {raceResultsTable.map(r => (
            <div key={r.round} className="grid grid-cols-[40px_160px_60px_60px_60px_100px_60px] gap-2 px-2 py-1.5 rounded-lg hover:bg-[#222838] transition-colors text-sm items-center">
              <span className="text-[#FF8000] font-mono">R{r.round}</span>
              <span className="text-foreground truncate">{r.gp}</span>
              <span className="font-mono" style={{ color: '#FF8000' }}>P{r.norPos}</span>
              <span className="font-mono" style={{ color: '#00d4ff' }}>P{r.piaPos}</span>
              <span className="text-foreground font-mono">{r.winner}</span>
              <span className="text-muted-foreground truncate">{r.winnerTeam}</span>
              <span className={`font-mono ${Number(r.gained) > 0 ? 'text-green-400' : 'text-muted-foreground'}`}>
                {Number(r.gained) > 0 ? `+${r.gained}` : r.gained}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
