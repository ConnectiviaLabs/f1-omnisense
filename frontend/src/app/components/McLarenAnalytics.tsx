import { useState, useEffect, useMemo } from 'react';
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, AreaChart, Area, ReferenceLine,
} from 'recharts';
import {
  Trophy, TrendingUp, Loader2, Flag, Users, Timer, Gauge, Activity, Disc, Shield,
  Zap, AlertTriangle, ChevronUp, ChevronDown, Minus, Brain, CheckCircle, XCircle,
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
  'red_bull': '#3671C6', 'mclaren': '#FF8000', 'ferrari': '#E8002D',
  'mercedes': '#27F4D2', 'aston_martin': '#229971', 'alpine': '#FF87BC',
  'williams': '#64C4FF', 'rb': '#6692FF', 'sauber': '#52E252',
  'haas': '#B6BABD',
};

const teamDisplayNames: Record<string, string> = {
  'red_bull': 'Red Bull', 'mclaren': 'McLaren', 'ferrari': 'Ferrari',
  'mercedes': 'Mercedes', 'aston_martin': 'Aston Martin', 'alpine': 'Alpine',
  'williams': 'Williams', 'rb': 'RB', 'sauber': 'Kick Sauber',
  'haas': 'Haas', 'alphatauri': 'AlphaTauri', 'alfa': 'Alfa Romeo',
  'toro_rosso': 'Toro Rosso', 'renault': 'Renault', 'racing_point': 'Racing Point',
};

const compoundColors: Record<string, string> = {
  SOFT: '#ef4444', MEDIUM: '#f59e0b', HARD: '#e8e8f0', INTERMEDIATE: '#22c55e', WET: '#3b82f6',
};

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
  const [availableYears, setAvailableYears] = useState<string[]>(['2024']);

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

  // ── MongoDB: jolpica_race_results (flat rows) ─────────────────────
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

  // ── MongoDB: jolpica_driver_standings ──────────────────────────────
  const [allDriverStandings, setAllDriverStandings] = useState<any[]>([]);
  useEffect(() => {
    fetch('/api/jolpica/driver_standings')
      .then(r => r.ok ? r.json() : [])
      .then(d => setAllDriverStandings(d))
      .catch(() => {});
  }, []);

  // ── MongoDB: jolpica_constructor_standings ─────────────────────────
  const [allConstructorStandings, setAllConstructorStandings] = useState<any[]>([]);
  useEffect(() => {
    fetch('/api/jolpica/constructor_standings')
      .then(r => r.ok ? r.json() : [])
      .then(d => setAllConstructorStandings(d))
      .catch(() => {});
  }, []);

  // ── MongoDB: Telemetry race summaries ─────────────────────────────
  const [norSummary, setNorSummary] = useState<CarSummary[]>([]);
  const [piaSummary, setPiaSummary] = useState<CarSummary[]>([]);
  const [telLoading, setTelLoading] = useState(true);

  // ── MongoDB: Pit stops ────────────────────────────────────────────
  const [pitStopsRaw, setPitStopsRaw] = useState<any[]>([]);

  // ── MongoDB: Tire strategy (year-filtered server join) ────────────
  const [tireStrategyRaw, setTireStrategyRaw] = useState<any[]>([]);

  // ── MongoDB: Anomaly scores ───────────────────────────────────────
  const [anomalyVehicles, setAnomalyVehicles] = useState<VehicleData[]>([]);
  useEffect(() => {
    fetch('/api/pipeline/anomaly').then(r => r.json())
      .then(d => setAnomalyVehicles(parseAnomalyDrivers(d)))
      .catch(() => {});
  }, []);

  // Tire strategy + pit stops fetched per year (alongside telemetry)

  // ── Detect McLaren drivers for selected year ──────────────────────
  const mcLarenDrivers = useMemo(() => {
    const drivers = new Set<string>();
    allRaceResults.forEach(r => {
      if (matchSeason(r.season, year) && (r.constructor_id || '').toLowerCase().includes('mclaren')) {
        if (r.driver_code) drivers.add(r.driver_code);
      }
    });
    return Array.from(drivers).sort();
  }, [allRaceResults, year]);

  // Fetch telemetry + pit stops per year + McLaren drivers
  useEffect(() => {
    if (!mcLarenDrivers.length) { setTelLoading(false); return; }
    setTelLoading(true);
    Promise.all(
      mcLarenDrivers.map(code =>
        fetch(`/api/local/mccar-summary/${year}/${code}`).then(r => r.ok ? r.json() : [])
      )
    ).then(results => {
      // Store first two drivers' telemetry (primary + secondary)
      setNorSummary(results[0] || []);
      setPiaSummary(results[1] || []);
    }).catch(() => {}).finally(() => setTelLoading(false));

    fetch(`/api/jolpica/pit_stops?season=${year}`)
      .then(r => r.ok ? r.json() : [])
      .then(d => setPitStopsRaw(d))
      .catch(() => {});

    fetch(`/api/mclaren/tire-strategy/${year}`)
      .then(r => r.ok ? r.json() : [])
      .then(d => setTireStrategyRaw(d))
      .catch(() => {});
  }, [year, mcLarenDrivers.join(',')]); // eslint-disable-line react-hooks/exhaustive-deps

  // ─── Derived data ─────────────────────────────────────────────────

  const norVehicle = anomalyVehicles.find(v => mcLarenDrivers.includes(v.code));
  const piaVehicle = anomalyVehicles.find(v => mcLarenDrivers.includes(v.code) && v.code !== norVehicle?.code);

  // Filter standings for selected year
  const driverStandings = useMemo(() =>
    allDriverStandings.filter(s => matchSeason(s.season, year)),
    [allDriverStandings, year]);

  const constructorStandings = useMemo(() =>
    allConstructorStandings.filter(s => matchSeason(s.season, year)).sort((a, b) => Number(a.position) - Number(b.position)),
    [allConstructorStandings, year]);

  // KPIs — fields: season, position, points, driver_code, constructor_name
  const kpiStats = useMemo(() => {
    const d1 = driverStandings.find(s => s.driver_code === mcLarenDrivers[0]);
    const d2 = driverStandings.find(s => s.driver_code === mcLarenDrivers[1]);
    const mcS = constructorStandings.find(s => (s.constructor_id || '').toLowerCase().includes('mclaren'));

    // Last race points gained
    const yearResults = allRaceResults.filter(r => matchSeason(r.season, year));
    const maxRound = Math.max(0, ...yearResults.map(r => Number(r.round || 0)));
    const lastRacePts = yearResults
      .filter(r => Number(r.round) === maxRound && (r.constructor_id || '').toLowerCase().includes('mclaren'))
      .reduce((sum, r) => sum + Number(r.points || 0), 0);

    return {
      d1Code: mcLarenDrivers[0] || '—', d1Pts: d1?.points ?? '—', d1Pos: d1?.position ?? '—',
      d2Code: mcLarenDrivers[1] || '—', d2Pts: d2?.points ?? '—', d2Pos: d2?.position ?? '—',
      teamPts: mcS?.points ?? '—', teamPos: mcS?.position ?? '—',
      lastGained: String(lastRacePts),
    };
  }, [driverStandings, constructorStandings, allRaceResults, mcLarenDrivers, year]);

  // Points progression — flat rows: season, round, race_name, driver_code, constructor_name, points
  const pointsProgression = useMemo(() => {
    const yearResults = allRaceResults.filter(r => matchSeason(r.season, year));
    if (!yearResults.length) return [];
    // Group by round
    const rounds = new Map<number, { race: string; results: any[] }>();
    yearResults.forEach(r => {
      const rnd = Number(r.round || 0);
      if (!rounds.has(rnd)) rounds.set(rnd, { race: (r.race_name || '').replace(' Grand Prix', ''), results: [] });
      rounds.get(rnd)!.results.push(r);
    });
    const sorted = Array.from(rounds.entries()).sort((a, b) => a[0] - b[0]);
    const cum: Record<string, number> = {};
    mcLarenDrivers.forEach(d => { cum[d] = 0; });
    let teamCum = 0;
    return sorted.map(([, { race, results }]) => {
      let raceTeam = 0;
      results.forEach(r => {
        const pts = Number(r.points || 0);
        const isMc = (r.constructor_id || '').toLowerCase().includes('mclaren');
        if (mcLarenDrivers.includes(r.driver_code)) cum[r.driver_code] = (cum[r.driver_code] || 0) + pts;
        if (isMc) raceTeam += pts;
      });
      teamCum += raceTeam;
      const point: any = { race, Team: teamCum };
      mcLarenDrivers.forEach(d => { point[d] = cum[d] || 0; });
      return point;
    });
  }, [allRaceResults, mcLarenDrivers, year]);

  // Telemetry merged
  const telemetryData = useMemo(() => {
    const raceMap = new Map<string, any>();
    const d1 = mcLarenDrivers[0] || 'D1';
    const d2 = mcLarenDrivers[1] || 'D2';
    norSummary.forEach(r => {
      raceMap.set(r.race, { race: r.race.slice(0, 12), [`${d1}Top`]: r.topSpeed, [`${d1}Throttle`]: r.avgThrottle, [`${d1}Brake`]: r.brakePct, [`${d1}Drs`]: r.drsPct });
    });
    piaSummary.forEach(r => {
      const e = raceMap.get(r.race) ?? { race: r.race.slice(0, 12) };
      e[`${d2}Top`] = r.topSpeed; e[`${d2}Throttle`] = r.avgThrottle; e[`${d2}Brake`] = r.brakePct; e[`${d2}Drs`] = r.drsPct;
      raceMap.set(r.race, e);
    });
    return Array.from(raceMap.values());
  }, [norSummary, piaSummary, mcLarenDrivers]);

  // Pit stops — fields: season, race_name, driver_id (e.g. "norris"), duration_s
  const pitStopData = useMemo(() => {
    if (!pitStopsRaw.length || !mcLarenDrivers.length) return [];
    // Map driver_id to driver_code
    const driverIdMap: Record<string, string> = {};
    allRaceResults.filter(r => matchSeason(r.season, year) && mcLarenDrivers.includes(r.driver_code)).forEach(r => {
      if (r.driver_id) driverIdMap[r.driver_id] = r.driver_code;
    });
    const raceStops = new Map<string, Record<string, number[]>>();
    pitStopsRaw.forEach(stop => {
      const race = (stop.race_name || '').replace(' Grand Prix', '').slice(0, 12);
      const code = driverIdMap[stop.driver_id || ''];
      const dur = Number(stop.duration_s || 0);
      if (!code || dur <= 0 || dur > 60 || !race) return;
      if (!raceStops.has(race)) raceStops.set(race, {});
      const entry = raceStops.get(race)!;
      if (!entry[code]) entry[code] = [];
      entry[code].push(dur);
    });
    return Array.from(raceStops.entries()).map(([race, drivers]) => {
      const row: any = { race };
      mcLarenDrivers.forEach(d => {
        const stops = drivers[d];
        row[d] = stops?.length ? +(stops.reduce((a, b) => a + b, 0) / stops.length).toFixed(1) : null;
      });
      return row;
    });
  }, [pitStopsRaw, mcLarenDrivers, allRaceResults, year]);

  const avgPitDuration = useMemo(() => {
    const all = pitStopData.flatMap(r => mcLarenDrivers.map(d => r[d]).filter(Boolean) as number[]);
    return all.length ? +(all.reduce((a, b) => a + b, 0) / all.length).toFixed(1) : 0;
  }, [pitStopData, mcLarenDrivers]);

  // Tire compound grid from server-side join endpoint
  const numToCode: Record<number, string> = { 1: 'VER', 3: 'RIC', 4: 'NOR', 11: 'PER', 14: 'ALO', 16: 'LEC', 55: 'SAI', 81: 'PIA', 2: 'SAR', 63: 'RUS', 44: 'HAM', 10: 'GAS', 31: 'OCO', 23: 'ALB', 22: 'TSU', 27: 'HUL', 20: 'MAG', 77: 'BOT', 24: 'ZHO', 18: 'STR', 21: 'DEV' };
  const tireGrid = useMemo(() => {
    if (!tireStrategyRaw.length) return { races: [] as string[], drivers: {} as Record<string, Record<string, string[]>> };
    const races = new Set<string>();
    const drivers: Record<string, Record<string, string[]>> = {};
    mcLarenDrivers.forEach(d => { drivers[d] = {}; });
    tireStrategyRaw.forEach((stint: any) => {
      const race = (stint.circuit || '').slice(0, 10);
      const code = numToCode[stint.driver_number];
      if (!code || !mcLarenDrivers.includes(code) || !race) return;
      races.add(race);
      if (!drivers[code][race]) drivers[code][race] = [];
      if (stint.compound && !drivers[code][race].includes(stint.compound)) drivers[code][race].push(stint.compound);
    });
    return { races: Array.from(races), drivers };
  }, [tireStrategyRaw, mcLarenDrivers]);

  // McLaren vs The World — avg finish + points/race for top drivers
  const worldComparison = useMemo(() => {
    const yearResults = allRaceResults.filter(r => matchSeason(r.season, year));
    if (!yearResults.length) return { avgFinish: [] as any[], pointsPerRace: [] as any[] };
    // Group by driver
    const driverStats = new Map<string, { code: string; teamId: string; positions: number[]; points: number[]; races: number }>();
    yearResults.forEach(r => {
      const code = r.driver_code;
      if (!code) return;
      const pos = Number(r.position);
      const pts = Number(r.points || 0);
      if (!driverStats.has(code)) driverStats.set(code, { code, teamId: r.constructor_id || '', positions: [], points: [], races: 0 });
      const s = driverStats.get(code)!;
      if (pos > 0 && pos <= 20) s.positions.push(pos);
      s.points.push(pts);
      s.races++;
    });
    // Sort by total points desc, take top 12
    const sorted = Array.from(driverStats.values())
      .map(s => ({
        code: s.code,
        teamId: s.teamId,
        avgFinish: s.positions.length ? +(s.positions.reduce((a, b) => a + b, 0) / s.positions.length).toFixed(1) : 20,
        pointsPerRace: s.races ? +(s.points.reduce((a, b) => a + b, 0) / s.races).toFixed(1) : 0,
        totalPts: s.points.reduce((a, b) => a + b, 0),
        podiums: s.positions.filter(p => p <= 3).length,
        races: s.races,
      }))
      .sort((a, b) => b.totalPts - a.totalPts)
      .slice(0, 12);
    return {
      avgFinish: [...sorted].sort((a, b) => a.avgFinish - b.avgFinish),
      pointsPerRace: sorted,
    };
  }, [allRaceResults, year]);

  // Race results table — flat rows: round, race_name, driver_code, constructor_id, position, points
  const raceResultsTable = useMemo(() => {
    const yearResults = allRaceResults.filter(r => matchSeason(r.season, year));
    if (!yearResults.length) return [];
    // Group by round
    const rounds = new Map<number, { race: string; results: any[] }>();
    yearResults.forEach(r => {
      const rnd = Number(r.round || 0);
      if (!rounds.has(rnd)) rounds.set(rnd, { race: (r.race_name || '').replace(' Grand Prix', ''), results: [] });
      rounds.get(rnd)!.results.push(r);
    });
    return Array.from(rounds.entries())
      .sort((a, b) => b[0] - a[0]) // most recent first
      .slice(0, 8)
      .map(([rnd, { race, results }]) => {
        const winner = results.find(r => String(r.position) === '1');
        const mcPts = results
          .filter(r => (r.constructor_id || '').toLowerCase().includes('mclaren'))
          .reduce((sum, r) => sum + Number(r.points || 0), 0);
        const row: any = {
          round: rnd, gp: race,
          winner: winner?.driver_code ?? '—',
          winnerTeam: teamDisplayNames[winner?.constructor_id] || winner?.constructor_name || '',
          gained: String(mcPts),
        };
        mcLarenDrivers.forEach(d => {
          const dr = results.find(r => r.driver_code === d);
          row[d] = dr?.position ?? '—';
        });
        return row;
      });
  }, [allRaceResults, mcLarenDrivers, year]);

  // Driver colors
  const driverColors = ['#FF8000', '#00d4ff', '#22c55e', '#f59e0b'];

  // ─── KeX WISE Insights (LLM-generated) ─────────────────────────────
  interface KexInsight {
    pillar: string;
    driver: string;
    text: string;
    grounding_score: number;
    model_used: string;
    sentiment?: { label: string; score: number };
    entities?: { text: string; label: string }[];
    topics?: string[];
  }
  const [kexInsights, setKexInsights] = useState<KexInsight[]>([]);
  const [kexLoading, setKexLoading] = useState(false);
  useEffect(() => {
    setKexLoading(true);
    setKexInsights([]);
    fetch(`/api/omni/kex/mclaren-briefing/${year}`)
      .then(r => r.ok ? r.json() : null)
      .then(d => { if (d?.insights) setKexInsights(d.insights); })
      .catch(() => {})
      .finally(() => setKexLoading(false));
  }, [year]);

  // ─── Intelligence Briefing — auto-extracted insights ──────────────
  interface Insight {
    icon: React.ElementType;
    title: string;
    body: string;
    sentiment: 'positive' | 'negative' | 'neutral';
  }

  const insights = useMemo(() => {
    const out: Insight[] = [];
    const yearResults = allRaceResults.filter(r => matchSeason(r.season, year));
    if (!yearResults.length) return out;

    // Group results by round for McLaren
    const rounds = new Map<number, any[]>();
    yearResults.forEach(r => {
      const rnd = Number(r.round || 0);
      if (!rounds.has(rnd)) rounds.set(rnd, []);
      rounds.get(rnd)!.push(r);
    });
    const sortedRounds = Array.from(rounds.entries()).sort((a, b) => a[0] - b[0]);
    const totalRaces = sortedRounds.length;

    // --- Championship momentum (last 3 vs previous 3) ---
    if (totalRaces >= 6) {
      const mcPtsPerRound = sortedRounds.map(([, results]) =>
        results.filter(r => (r.constructor_id || '').toLowerCase().includes('mclaren'))
          .reduce((s, r) => s + Number(r.points || 0), 0)
      );
      const recent3 = mcPtsPerRound.slice(-3).reduce((a, b) => a + b, 0);
      const prev3 = mcPtsPerRound.slice(-6, -3).reduce((a, b) => a + b, 0);
      const diff = recent3 - prev3;
      if (diff > 10) {
        out.push({ icon: ChevronUp, title: 'Momentum Surge', body: `McLaren scored ${recent3} pts in the last 3 races vs ${prev3} in the prior 3 (+${diff}). Championship push is accelerating.`, sentiment: 'positive' });
      } else if (diff < -10) {
        out.push({ icon: ChevronDown, title: 'Momentum Drop', body: `McLaren scored ${recent3} pts in the last 3 races vs ${prev3} in the prior 3 (${diff}). Scoring rate is declining.`, sentiment: 'negative' });
      } else {
        out.push({ icon: Minus, title: 'Steady Form', body: `McLaren scored ${recent3} pts in the last 3 races vs ${prev3} prior. Consistent pace maintained.`, sentiment: 'neutral' });
      }
    }

    // --- Driver head-to-head ---
    if (mcLarenDrivers.length >= 2) {
      let d1Wins = 0, d2Wins = 0;
      sortedRounds.forEach(([, results]) => {
        const p1 = results.find(r => r.driver_code === mcLarenDrivers[0]);
        const p2 = results.find(r => r.driver_code === mcLarenDrivers[1]);
        if (p1 && p2) {
          const pos1 = Number(p1.position), pos2 = Number(p2.position);
          if (pos1 < pos2) d1Wins++;
          else if (pos2 < pos1) d2Wins++;
        }
      });
      const leader = d1Wins >= d2Wins ? mcLarenDrivers[0] : mcLarenDrivers[1];
      const leaderWins = Math.max(d1Wins, d2Wins);
      const trailingWins = Math.min(d1Wins, d2Wins);
      out.push({
        icon: Users,
        title: 'Intra-Team Battle',
        body: `${leader} leads the head-to-head ${leaderWins}-${trailingWins} in qualifying-style race finishes across ${totalRaces} races.`,
        sentiment: leaderWins - trailingWins > 4 ? 'negative' : 'neutral',
      });
    }

    // --- Anomaly health alerts ---
    [norVehicle, piaVehicle].filter(Boolean).forEach(v => {
      if (!v) return;
      const criticalSystems = v.systems.filter(s => s.health < 60);
      if (criticalSystems.length > 0) {
        out.push({
          icon: AlertTriangle,
          title: `${v.code} System Alert`,
          body: `${criticalSystems.map(s => `${s.name} at ${s.health}%`).join(', ')}. Overall health: ${v.overallHealth}% (${v.level}).`,
          sentiment: 'negative',
        });
      } else if (v.overallHealth >= 85) {
        out.push({
          icon: Zap,
          title: `${v.code} Peak Condition`,
          body: `All systems nominal at ${v.overallHealth}% overall health. No maintenance actions required.`,
          sentiment: 'positive',
        });
      }
    });

    // --- Telemetry speed trend ---
    if (norSummary.length >= 4) {
      const firstHalf = norSummary.slice(0, Math.floor(norSummary.length / 2));
      const secondHalf = norSummary.slice(Math.floor(norSummary.length / 2));
      const avgFirst = firstHalf.reduce((s, r) => s + r.topSpeed, 0) / firstHalf.length;
      const avgSecond = secondHalf.reduce((s, r) => s + r.topSpeed, 0) / secondHalf.length;
      const diff = avgSecond - avgFirst;
      if (Math.abs(diff) > 1) {
        out.push({
          icon: Gauge,
          title: `Speed ${diff > 0 ? 'Gain' : 'Loss'} Detected`,
          body: `${mcLarenDrivers[0]}'s avg top speed ${diff > 0 ? 'increased' : 'decreased'} by ${Math.abs(diff).toFixed(1)} km/h in the second half of the season vs the first.`,
          sentiment: diff > 0 ? 'positive' : 'negative',
        });
      }
    }

    // --- Pit stop trend ---
    if (pitStopData.length >= 4) {
      const firstHalf = pitStopData.slice(0, Math.floor(pitStopData.length / 2));
      const secondHalf = pitStopData.slice(Math.floor(pitStopData.length / 2));
      const avgDur = (data: any[]) => {
        const vals = data.flatMap(r => mcLarenDrivers.map(d => r[d]).filter(Boolean) as number[]);
        return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
      };
      const earlyAvg = avgDur(firstHalf);
      const lateAvg = avgDur(secondHalf);
      const diff = lateAvg - earlyAvg;
      if (Math.abs(diff) > 0.5) {
        out.push({
          icon: Timer,
          title: `Pit Crew ${diff < 0 ? 'Improving' : 'Slower'}`,
          body: `Average pit stop ${diff < 0 ? 'improved' : 'worsened'} by ${Math.abs(diff).toFixed(1)}s in the second half (${lateAvg.toFixed(1)}s vs ${earlyAvg.toFixed(1)}s).`,
          sentiment: diff < 0 ? 'positive' : 'negative',
        });
      }
    }

    // --- Podium rate ---
    const mcPodiums = yearResults.filter(r =>
      mcLarenDrivers.includes(r.driver_code) && Number(r.position) <= 3
    ).length;
    const mcStarts = yearResults.filter(r => mcLarenDrivers.includes(r.driver_code)).length;
    if (mcStarts > 0) {
      const rate = (mcPodiums / mcStarts * 100).toFixed(0);
      out.push({
        icon: Trophy,
        title: 'Podium Rate',
        body: `McLaren achieved ${mcPodiums} podiums from ${mcStarts} starts (${rate}% conversion rate) in ${year}.`,
        sentiment: Number(rate) >= 40 ? 'positive' : Number(rate) >= 20 ? 'neutral' : 'negative',
      });
    }

    return out;
  }, [allRaceResults, year, mcLarenDrivers, norVehicle, piaVehicle, norSummary, pitStopData]);

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
          <KPI icon={<Users className="w-4 h-4 text-[#FF8000]" />} label={kpiStats.d1Code} value={`P${kpiStats.d1Pos}`} detail={`${kpiStats.d1Pts} pts`} />
          <KPI icon={<Users className="w-4 h-4 text-cyan-400" />} label={kpiStats.d2Code} value={`P${kpiStats.d2Pos}`} detail={`${kpiStats.d2Pts} pts`} />
          <KPI icon={<Trophy className="w-4 h-4 text-green-400" />} label="Constructors" value={`P${kpiStats.teamPos}`} detail={`${kpiStats.teamPts} pts`} />
          <KPI icon={<Flag className="w-4 h-4 text-amber-400" />} label="Last Race" value={`+${kpiStats.lastGained}`} detail="pts gained" />
        </div>
      )}

      {/* Intelligence Briefing — auto-extracted knowledge */}
      {insights.length > 0 && (
        <>
          <Divider label="INTELLIGENCE BRIEFING" />
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {insights.map((ins, i) => {
              const sentimentColors = {
                positive: { border: 'rgba(5,223,114,0.2)', bg: 'rgba(5,223,114,0.05)', icon: '#05DF72' },
                negative: { border: 'rgba(251,44,54,0.2)', bg: 'rgba(251,44,54,0.05)', icon: '#FB2C36' },
                neutral: { border: 'rgba(255,128,0,0.15)', bg: 'rgba(255,128,0,0.03)', icon: '#FF8000' },
              };
              const sc = sentimentColors[ins.sentiment];
              const Icon = ins.icon;
              return (
                <div key={i} className="rounded-xl p-3" style={{ border: `1px solid ${sc.border}`, background: sc.bg }}>
                  <div className="flex items-center gap-2 mb-1.5">
                    <Icon className="w-3.5 h-3.5 shrink-0" style={{ color: sc.icon }} />
                    <span className="text-[12px] font-medium text-foreground">{ins.title}</span>
                  </div>
                  <p className="text-[11px] text-muted-foreground leading-relaxed">{ins.body}</p>
                </div>
              );
            })}
          </div>
        </>
      )}

      {/* KeX WISE — LLM-generated insights */}
      {(kexLoading || kexInsights.length > 0) && (
        <>
          <Divider label="WISE ANALYSIS" />
          {kexLoading ? (
            <div className="flex items-center justify-center gap-2 py-4">
              <Loader2 className="w-4 h-4 text-[#FF8000] animate-spin" />
              <span className="text-[11px] text-muted-foreground">Generating WISE insights…</span>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {kexInsights.map((ins, i) => {
                const pillarColors: Record<string, string> = {
                  realtime: '#3b82f6', anomaly: '#f59e0b', forecast: '#a855f7',
                };
                const pillarColor = pillarColors[ins.pillar] || '#FF8000';
                const verified = ins.grounding_score >= 0.5;

                // Build inline-annotated text fragments
                const entityColors: Record<string, { bg: string; text: string }> = {
                  GPE: { bg: 'rgba(59,130,246,0.15)', text: '#60a5fa' },
                  LOC: { bg: 'rgba(59,130,246,0.15)', text: '#60a5fa' },
                  ORG: { bg: 'rgba(168,85,247,0.15)', text: '#c084fc' },
                  PERSON: { bg: 'rgba(34,197,94,0.15)', text: '#4ade80' },
                  DATE: { bg: 'rgba(245,158,11,0.12)', text: '#fbbf24' },
                  TIME: { bg: 'rgba(245,158,11,0.12)', text: '#fbbf24' },
                  CARDINAL: { bg: 'rgba(236,72,153,0.12)', text: '#f472b6' },
                  ORDINAL: { bg: 'rgba(236,72,153,0.12)', text: '#f472b6' },
                  QUANTITY: { bg: 'rgba(236,72,153,0.12)', text: '#f472b6' },
                  PERCENT: { bg: 'rgba(236,72,153,0.12)', text: '#f472b6' },
                };
                const defaultEntityStyle = { bg: 'rgba(148,163,184,0.12)', text: '#94a3b8' };

                // Build annotation spans sorted longest-first for greedy matching
                const annotations: { text: string; type: 'entity' | 'topic'; label: string; style: { bg: string; text: string } }[] = [];
                for (const e of (ins.entities || [])) {
                  if (e.text.length > 1) {
                    annotations.push({
                      text: e.text,
                      type: 'entity',
                      label: e.label,
                      style: entityColors[e.label] || defaultEntityStyle,
                    });
                  }
                }
                for (const t of (ins.topics || [])) {
                  if (!annotations.some(a => a.text.toLowerCase() === t.toLowerCase())) {
                    annotations.push({
                      text: t,
                      type: 'topic',
                      label: 'TOPIC',
                      style: { bg: 'rgba(255,128,0,0.12)', text: '#FF8000' },
                    });
                  }
                }
                annotations.sort((a, b) => b.text.length - a.text.length);

                // Annotate text: find and wrap matches
                const annotateText = (raw: string) => {
                  if (!annotations.length) return [<span key="raw">{raw}</span>];

                  type Frag = { start: number; end: number; ann: typeof annotations[0] };
                  const frags: Frag[] = [];
                  const taken = new Uint8Array(raw.length);

                  for (const ann of annotations) {
                    const regex = new RegExp(ann.text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi');
                    let m: RegExpExecArray | null;
                    while ((m = regex.exec(raw)) !== null) {
                      const s = m.index, e = s + m[0].length;
                      let overlap = false;
                      for (let k = s; k < e; k++) { if (taken[k]) { overlap = true; break; } }
                      if (!overlap) {
                        frags.push({ start: s, end: e, ann });
                        for (let k = s; k < e; k++) taken[k] = 1;
                      }
                    }
                  }
                  frags.sort((a, b) => a.start - b.start);

                  const result: React.ReactNode[] = [];
                  let cursor = 0;
                  for (const f of frags) {
                    if (f.start > cursor) result.push(<span key={`t${cursor}`}>{raw.slice(cursor, f.start)}</span>);
                    result.push(
                      <span
                        key={`a${f.start}`}
                        className="relative group/ann cursor-default rounded-[2px] px-[1px]"
                        style={{ background: f.ann.style.bg, color: f.ann.style.text }}
                        title={`${f.ann.label}`}
                      >
                        {raw.slice(f.start, f.end)}
                        <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 hidden group-hover/ann:block
                          text-[7px] font-mono px-1 py-0.5 rounded bg-[#0D1117] border border-[rgba(255,128,0,0.2)]
                          text-muted-foreground whitespace-nowrap z-10 pointer-events-none">
                          {f.ann.label}
                        </span>
                      </span>
                    );
                    cursor = f.end;
                  }
                  if (cursor < raw.length) result.push(<span key={`t${cursor}`}>{raw.slice(cursor)}</span>);
                  return result;
                };

                return (
                  <div key={i} className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded-xl p-3">
                    <div className="flex items-center gap-2 mb-2">
                      <Brain className="w-3.5 h-3.5 shrink-0 text-[#FF8000]" />
                      <span className="text-[9px] font-semibold tracking-wider px-1.5 py-0.5 rounded"
                        style={{ background: `${pillarColor}20`, color: pillarColor }}>
                        {ins.pillar.toUpperCase()}
                      </span>
                      <span className="text-[10px] font-mono text-[#FF8000]">{ins.driver}</span>
                      {ins.sentiment && (
                        <span className={`text-[8px] font-semibold px-1 py-0.5 rounded ${
                          ins.sentiment.label === 'positive' ? 'bg-green-500/15 text-green-400' :
                          ins.sentiment.label === 'negative' ? 'bg-red-500/15 text-red-400' :
                          'bg-zinc-500/15 text-zinc-400'
                        }`}>
                          {ins.sentiment.label === 'positive' ? '▲' : ins.sentiment.label === 'negative' ? '▼' : '●'} {ins.sentiment.score}
                        </span>
                      )}
                      <div className="flex-1" />
                      {verified
                        ? <CheckCircle className="w-3 h-3 text-green-400" />
                        : <XCircle className="w-3 h-3 text-red-400/50" />}
                      <span className="text-[9px] text-muted-foreground font-mono">{ins.grounding_score.toFixed(2)}</span>
                    </div>
                    <p className="text-[11px] text-muted-foreground leading-relaxed whitespace-pre-line">
                      {annotateText(ins.text)}
                    </p>
                    {ins.model_used && (
                      <p className="text-[9px] text-muted-foreground/50 mt-1.5 font-mono">via {ins.model_used}</p>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </>
      )}

      {/* Section 2: System Health Strip */}
      {(norVehicle || piaVehicle) && (
        <>
          <Divider label="SYSTEM HEALTH" />
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {[norVehicle, piaVehicle].filter(Boolean).map((v, idx) => {
              if (!v) return null;
              const color = driverColors[idx];
              return (
                <div key={v.code} className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded-xl p-4">
                  <div className="flex items-center gap-4">
                    <HealthGauge value={v.overallHealth} size={60} strokeWidth={5} showLabel={false} />
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-sm font-medium" style={{ color }}>{v.code}</span>
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
          <p className="text-[12px] text-muted-foreground mb-3">{mcLarenDrivers.join(' + ')} + Constructors cumulative</p>
          <div className="h-[260px]">
            {pointsProgression.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={pointsProgression}>
                  <defs>
                    {mcLarenDrivers.map((d, i) => (
                      <linearGradient key={d} id={`grad-${d}`} x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor={driverColors[i]} stopOpacity={0.3} />
                        <stop offset="100%" stopColor={driverColors[i]} stopOpacity={0} />
                      </linearGradient>
                    ))}
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
                  {mcLarenDrivers.map((d, i) => (
                    <Area key={d} type="monotone" dataKey={d} stroke={driverColors[i]} fill={`url(#grad-${d})`} strokeWidth={2} dot={false} name={d} />
                  ))}
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
              const isMcLaren = (team.constructor_id || '').toLowerCase().includes('mclaren');
              const name = teamDisplayNames[team.constructor_id] || team.constructor_name || team.constructor_id;
              return (
                <div key={team.constructor_id || i} className={`flex items-center gap-2 p-2 rounded-lg ${isMcLaren ? 'bg-[#FF8000]/10 border border-[#FF8000]/20' : ''}`}>
                  <span className="text-[12px] text-muted-foreground w-5 font-mono">P{team.position}</span>
                  <div className="w-2 h-5 rounded-full" style={{ backgroundColor: teamColors[team.constructor_id] ?? '#555' }} />
                  <span className={`text-sm flex-1 truncate ${isMcLaren ? 'text-[#FF8000] font-medium' : 'text-foreground'}`}>{name}</span>
                  <span className="text-sm font-mono text-foreground">{pts}</span>
                  {i > 0 && <span className="text-[11px] font-mono text-muted-foreground w-12 text-right">{pts - leaderPts}</span>}
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
      ) : telemetryData.length > 0 && mcLarenDrivers.length >= 2 ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded-xl p-4">
            <h3 className="text-sm text-foreground mb-1">Top Speed</h3>
            <p className="text-[12px] text-muted-foreground mb-3">{mcLarenDrivers[0]} vs {mcLarenDrivers[1]} — km/h per race</p>
            <div className="h-[220px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={telemetryData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,128,0,0.08)" />
                  <XAxis dataKey="race" stroke="#8888a0" fontSize={9} angle={-30} textAnchor="end" height={50} />
                  <YAxis stroke="#8888a0" fontSize={10} domain={['dataMin - 5', 'dataMax + 5']} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey={`${mcLarenDrivers[0]}Top`} fill={driverColors[0]} opacity={0.8} radius={[3, 3, 0, 0]} name={mcLarenDrivers[0]} />
                  <Bar dataKey={`${mcLarenDrivers[1]}Top`} fill={driverColors[1]} opacity={0.8} radius={[3, 3, 0, 0]} name={mcLarenDrivers[1]} />
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
                  { metric: 'Throttle %', [mcLarenDrivers[0]]: avg(norSummary, 'avgThrottle'), [mcLarenDrivers[1]]: avg(piaSummary, 'avgThrottle') },
                  { metric: 'Brake %', [mcLarenDrivers[0]]: avg(norSummary, 'brakePct'), [mcLarenDrivers[1]]: avg(piaSummary, 'brakePct') },
                  { metric: 'DRS %', [mcLarenDrivers[0]]: avg(norSummary, 'drsPct'), [mcLarenDrivers[1]]: avg(piaSummary, 'drsPct') },
                ];
                return (
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={data} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,128,0,0.08)" />
                      <XAxis type="number" stroke="#8888a0" fontSize={10} unit="%" />
                      <YAxis dataKey="metric" type="category" stroke="#8888a0" fontSize={11} width={80} />
                      <Tooltip content={<CustomTooltip />} />
                      <Bar dataKey={mcLarenDrivers[0]} fill={driverColors[0]} opacity={0.8} radius={[0, 4, 4, 0]} name={mcLarenDrivers[0]} />
                      <Bar dataKey={mcLarenDrivers[1]} fill={driverColors[1]} opacity={0.8} radius={[0, 4, 4, 0]} name={mcLarenDrivers[1]} />
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
                  {mcLarenDrivers.map((d, i) => (
                    <Line key={d} type="monotone" dataKey={d} stroke={driverColors[i]} strokeWidth={2} dot={{ r: 3, fill: driverColors[i] }} name={d} connectNulls />
                  ))}
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
                  {mcLarenDrivers.map((d, i) => (
                    <tr key={d} className="border-b border-[rgba(255,128,0,0.04)]">
                      <td className="py-1.5 px-1 font-mono font-medium" style={{ color: driverColors[i] }}>{d}</td>
                      {tireGrid.races.map(race => (
                        <td key={race} className="py-1 px-0.5 text-center">
                          <div className="flex gap-0.5 justify-center">
                            {(tireGrid.drivers[d]?.[race] ?? []).map((c, j) => (
                              <span key={j} className="w-2.5 h-2.5 rounded-full inline-block" style={{ background: compoundColors[c] ?? '#555' }} title={c} />
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

      {/* Section 6: McLaren vs The World */}
      <Divider label="MCLAREN VS THE WORLD" />
      {worldComparison.pointsPerRace.length > 0 ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded-xl p-4">
            <h3 className="text-sm text-foreground mb-1 flex items-center gap-2"><TrendingUp className="w-3.5 h-3.5 text-[#FF8000]" />Points Per Race</h3>
            <p className="text-[12px] text-muted-foreground mb-3">Top 12 drivers — avg points per GP</p>
            <div className="h-[320px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={worldComparison.pointsPerRace} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,128,0,0.08)" />
                  <XAxis type="number" stroke="#8888a0" fontSize={10} />
                  <YAxis dataKey="code" type="category" stroke="#8888a0" fontSize={11} width={40} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="pointsPerRace" radius={[0, 4, 4, 0]} name="Pts/Race">
                    {worldComparison.pointsPerRace.map((e, i) => (
                      <Cell key={i} fill={mcLarenDrivers.includes(e.code) ? '#FF8000' : teamColors[e.teamId] ?? '#6b7280'} opacity={mcLarenDrivers.includes(e.code) ? 1 : 0.6} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded-xl p-4">
            <h3 className="text-sm text-foreground mb-1 flex items-center gap-2"><Users className="w-3.5 h-3.5 text-[#FF8000]" />Average Finish Position</h3>
            <p className="text-[12px] text-muted-foreground mb-3">Top 12 drivers — lower is better</p>
            <div className="h-[320px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={worldComparison.avgFinish} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,128,0,0.08)" />
                  <XAxis type="number" stroke="#8888a0" fontSize={10} reversed />
                  <YAxis dataKey="code" type="category" stroke="#8888a0" fontSize={11} width={40} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="avgFinish" radius={[0, 4, 4, 0]} name="Avg Finish">
                    {worldComparison.avgFinish.map((e, i) => (
                      <Cell key={i} fill={mcLarenDrivers.includes(e.code) ? '#FF8000' : teamColors[e.teamId] ?? '#6b7280'} opacity={mcLarenDrivers.includes(e.code) ? 1 : 0.6} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      ) : (
        <div className="text-center py-6 text-muted-foreground text-sm">No comparison data for {year}</div>
      )}

      {/* Section 7: Race Results */}
      <Divider label="RACE RESULTS" />
      <div className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded-xl p-4">
        <div className="space-y-1">
          <div className={`grid gap-2 px-2 py-1 text-[11px] text-muted-foreground tracking-wider`}
            style={{ gridTemplateColumns: `40px 160px ${mcLarenDrivers.map(() => '60px').join(' ')} 60px 100px 60px` }}>
            <span>RND</span><span>GRAND PRIX</span>
            {mcLarenDrivers.map(d => <span key={d}>{d}</span>)}
            <span>WIN</span><span>TEAM</span><span>+PTS</span>
          </div>
          {raceResultsTable.map(r => (
            <div key={r.round} className="grid gap-2 px-2 py-1.5 rounded-lg hover:bg-[#222838] transition-colors text-sm items-center"
              style={{ gridTemplateColumns: `40px 160px ${mcLarenDrivers.map(() => '60px').join(' ')} 60px 100px 60px` }}>
              <span className="text-[#FF8000] font-mono">R{r.round}</span>
              <span className="text-foreground truncate">{r.gp}</span>
              {mcLarenDrivers.map((d, i) => (
                <span key={d} className="font-mono" style={{ color: driverColors[i] }}>P{r[d]}</span>
              ))}
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
