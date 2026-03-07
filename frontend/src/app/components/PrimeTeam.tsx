import { useState, useEffect, useMemo } from 'react';
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, AreaChart, Area,
} from 'recharts';
import { Loader2, TrendingUp, Trophy, Users, Timer, Gauge, Flag, ArrowLeft, GitCompare } from 'lucide-react';
import { ForecastChart } from './ForecastChart';
import KexBriefingCard from './KexBriefingCard';
import { HealthGauge } from './HealthGauge';
import { StatusBadge } from './StatusBadge';
import {
  type VehicleData, parseAnomalyDrivers, levelColor, MAINTENANCE_LABELS,
} from './anomalyHelpers';
import type { Pillar } from './Sidebar';

/* ─── Team Constants ─── */

const TEAMS = [
  { id: 'red_bull', name: 'Red Bull' },
  { id: 'mclaren', name: 'McLaren' },
  { id: 'ferrari', name: 'Ferrari' },
  { id: 'mercedes', name: 'Mercedes' },
  { id: 'aston_martin', name: 'Aston Martin' },
  { id: 'alpine', name: 'Alpine' },
  { id: 'williams', name: 'Williams' },
  { id: 'rb', name: 'RB' },
  { id: 'sauber', name: 'Kick Sauber' },
  { id: 'haas', name: 'Haas' },
];

const TEAM_COLORS: Record<string, string> = {
  red_bull: '#3671C6', mclaren: '#FF8000', ferrari: '#E8002D',
  mercedes: '#27F4D2', aston_martin: '#229971', alpine: '#FF87BC',
  williams: '#64C4FF', rb: '#6692FF', sauber: '#52E252', haas: '#B6BABD',
};

const F1_CDN = 'https://media.formula1.com/image/upload/c_lfill,w_96/q_auto/v1740000000/common/f1/2026';
const TEAM_LOGOS: Record<string, string> = {
  red_bull: `${F1_CDN}/redbullracing/2026redbullracinglogowhite.webp`,
  mclaren: `${F1_CDN}/mclaren/2026mclarenlogowhite.webp`,
  ferrari: `${F1_CDN}/ferrari/2026ferrarilogowhite.webp`,
  mercedes: `${F1_CDN}/mercedes/2026mercedeslogowhite.webp`,
  aston_martin: `${F1_CDN}/astonmartin/2026astonmartinlogowhite.webp`,
  alpine: `${F1_CDN}/alpine/2026alpinelogowhite.webp`,
  williams: `${F1_CDN}/williams/2026williamslogowhite.webp`,
  rb: `${F1_CDN}/racingbulls/2026racingbullslogowhite.webp`,
  sauber: `${F1_CDN}/audi/2026audilogowhite.webp`,
  haas: `${F1_CDN}/haasf1team/2026haasf1teamlogowhite.webp`,
};

// Map anomaly data team names → constructor_id
const TEAM_NAME_MAP: Record<string, string> = {
  'red bull racing': 'red_bull', 'mclaren': 'mclaren', 'ferrari': 'ferrari',
  'mercedes': 'mercedes', 'aston martin': 'aston_martin', 'alpine': 'alpine',
  'williams': 'williams', 'rb': 'rb', 'kick sauber': 'sauber', 'haas f1 team': 'haas',
  'haas': 'haas', 'sauber': 'sauber', 'racing bulls': 'rb', 'visa cash app rb': 'rb',
};

// Map constructor_id → VectorProfiles team name (for similarity API)
const TEAM_VP_NAME: Record<string, string> = {
  red_bull: 'Red Bull', mclaren: 'McLaren', ferrari: 'Ferrari',
  mercedes: 'Mercedes', aston_martin: 'Aston Martin', alpine: 'Alpine F1 Team',
  williams: 'Williams', rb: 'RB F1 Team', sauber: 'Sauber', haas: 'Haas F1 Team',
};

interface SimilarTeam { team: string; score: number; drivers: string[] }
interface IntraTeamPair { driver_a: string; driver_b: string; score: number }

function teamIdFromName(teamName: string): string {
  const lower = teamName.toLowerCase();
  if (TEAM_NAME_MAP[lower]) return TEAM_NAME_MAP[lower];
  for (const [key, id] of Object.entries(TEAM_NAME_MAP)) {
    if (lower.includes(key) || key.includes(lower)) return id;
  }
  return lower.replace(/\s+/g, '_');
}

/* ─── Props ─── */

interface PrimeTeamProps {
  prefetchedVehicles?: VehicleData[];
  activePillar: Pillar;
}

export function PrimeTeam({ prefetchedVehicles, activePillar }: PrimeTeamProps) {
  const [selectedTeam, setSelectedTeam] = useState<string | null>(null);
  const [selectedSeason, setSelectedSeason] = useState(2024);
  const [vehicles, setVehicles] = useState<VehicleData[]>(prefetchedVehicles ?? []);
  const [loading, setLoading] = useState(!prefetchedVehicles);

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

  const teamColor = selectedTeam ? (TEAM_COLORS[selectedTeam] ?? '#FF8000') : '#FF8000';
  const teamName = selectedTeam ? (TEAMS.find(t => t.id === selectedTeam)?.name ?? selectedTeam) : '';

  // Show grid overview when no team selected (telemetry pillar)
  if (activePillar === 'telemetry' && !selectedTeam) {
    return <TeamGrid vehicles={vehicles} loading={loading} onSelect={setSelectedTeam} />;
  }

  // For anomaly/forecast without selection, default to mclaren
  const effectiveTeam = selectedTeam ?? 'mclaren';
  const effectiveColor = TEAM_COLORS[effectiveTeam] ?? '#FF8000';
  const effectiveName = TEAMS.find(t => t.id === effectiveTeam)?.name ?? effectiveTeam;

  return (
    <div className="space-y-4">
      {/* Selectors */}
      <div className="flex items-center gap-4">
        {activePillar === 'telemetry' && (
          <button
            type="button"
            onClick={() => setSelectedTeam(null)}
            className="flex items-center gap-1.5 text-[12px] text-muted-foreground hover:text-[#FF8000] transition-colors"
          >
            <ArrowLeft className="w-3.5 h-3.5" /> All Teams
          </button>
        )}
        <div className="flex items-center gap-2">
          <label className="text-[12px] text-muted-foreground">Team</label>
          <select
            title="Select team"
            value={effectiveTeam}
            onChange={e => setSelectedTeam(e.target.value)}
            className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded-lg px-3 py-1.5 text-sm text-foreground focus:outline-none focus:border-[#FF8000]/30"
          >
            {TEAMS.map(t => (
              <option key={t.id} value={t.id}>{t.name}</option>
            ))}
          </select>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-[12px] text-muted-foreground">Season</label>
          <select
            title="Select season"
            value={selectedSeason}
            onChange={e => setSelectedSeason(Number(e.target.value))}
            className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded-lg px-3 py-1.5 text-sm text-foreground focus:outline-none focus:border-[#FF8000]/30"
          >
            {[2024, 2023, 2022, 2021, 2020].map(y => (
              <option key={y} value={y}>{y}</option>
            ))}
          </select>
        </div>
        {TEAM_LOGOS[effectiveTeam] && (
          <img src={TEAM_LOGOS[effectiveTeam]} alt={effectiveName} className="h-6 object-contain" />
        )}
      </div>

      {activePillar === 'telemetry' && (
        <TeamTelemetryView teamId={effectiveTeam} teamName={effectiveName} teamColor={effectiveColor} season={selectedSeason} vehicles={vehicles} />
      )}
      {activePillar === 'anomaly' && (
        <TeamAnomalyView teamId={effectiveTeam} teamName={effectiveName} teamColor={effectiveColor} vehicles={vehicles} loading={loading} />
      )}
      {activePillar === 'forecast' && (
        <TeamForecastView teamId={effectiveTeam} teamName={effectiveName} teamColor={effectiveColor} vehicles={vehicles} season={selectedSeason} />
      )}
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════════════
   TEAM GRID OVERVIEW — matches Driver Grid design
   ══════════════════════════════════════════════════════════════════════ */

function TeamGrid({ vehicles, loading, onSelect }: {
  vehicles: VehicleData[]; loading: boolean; onSelect: (teamId: string) => void;
}) {
  // Aggregate vehicles by team
  const teamData = useMemo(() => {
    const byTeam: Record<string, { drivers: VehicleData[]; teamId: string; name: string }> = {};
    for (const v of vehicles) {
      const teamId = teamIdFromName(v.team);
      const teamDef = TEAMS.find(t => t.id === teamId);
      if (!teamDef) continue;
      if (!byTeam[teamId]) byTeam[teamId] = { drivers: [], teamId, name: teamDef.name };
      byTeam[teamId].drivers.push(v);
    }
    return Object.values(byTeam).sort((a, b) => {
      const aHealth = a.drivers.reduce((s, d) => s + d.overallHealth, 0) / a.drivers.length;
      const bHealth = b.drivers.reduce((s, d) => s + d.overallHealth, 0) / b.drivers.length;
      return bHealth - aHealth;
    });
  }, [vehicles]);

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-sm text-muted-foreground py-8 justify-center">
        <Loader2 className="w-4 h-4 animate-spin text-[#FF8000]" /> Loading teams...
      </div>
    );
  }

  const mclarenTeam = teamData.find(t => t.teamId === 'mclaren');
  const otherTeams = teamData.filter(t => t.teamId !== 'mclaren');

  const renderTeamCard = (teamId: string, name: string, drivers: VehicleData[], isMcLaren: boolean) => {
    const color = TEAM_COLORS[teamId] ?? '#FF8000';
    const avgHealth = drivers.reduce((s, d) => s + d.overallHealth, 0) / drivers.length;
    const worstLevel = drivers.some(d => d.level === 'critical') ? 'critical'
      : drivers.some(d => d.level === 'warning') ? 'warning' : 'nominal';

    const systemMap: Record<string, number[]> = {};
    for (const d of drivers) {
      for (const sys of d.systems) {
        if (!systemMap[sys.name]) systemMap[sys.name] = [];
        systemMap[sys.name].push(sys.health);
      }
    }
    const systems = Object.entries(systemMap).map(([sysName, vals]) => ({
      name: sysName,
      health: vals.reduce((a, b) => a + b, 0) / vals.length,
    }));

    return (
      <button
        type="button"
        key={teamId}
        onClick={() => onSelect(teamId)}
        className={`bg-[#1A1F2E] rounded-xl border text-left transition-all group cursor-pointer relative overflow-hidden ${
          isMcLaren
            ? 'border-[#FF8000]/30 hover:border-[#FF8000]/50 p-5'
            : 'border-[rgba(255,128,0,0.12)] hover:border-[rgba(255,128,0,0.3)] p-4'
        }`}
      >
        {/* Team color accent stripe */}
        <div className="absolute top-0 left-0 w-1 h-full rounded-l-xl" style={{ background: color }} />

        {/* Header */}
        <div className={`flex items-center gap-3 ${isMcLaren ? 'mb-4' : 'mb-3'} ml-2`}>
          <div
            className={`rounded-lg flex items-center justify-center ${isMcLaren ? 'w-10 h-10' : 'w-8 h-8'}`}
            style={{ background: `${color}20`, border: `1px solid ${color}40` }}
          >
            {TEAM_LOGOS[teamId] ? (
              <img src={TEAM_LOGOS[teamId]} alt={name} className={isMcLaren ? 'w-7 h-7 object-contain' : 'w-5 h-5 object-contain'} />
            ) : (
              <Trophy className={isMcLaren ? 'w-5 h-5' : 'w-4 h-4'} style={{ color }} />
            )}
          </div>
          <div className="flex-1 min-w-0">
            <div className={`font-semibold text-foreground transition-colors truncate ${
              isMcLaren ? 'text-base group-hover:text-[#FF8000]' : 'text-sm group-hover:text-[#FF8000]'
            }`}>{name}</div>
            <div className={`text-muted-foreground ${isMcLaren ? 'text-[11px]' : 'text-[10px]'}`}>
              {drivers.map(d => d.code).join(', ')} — {drivers.length} driver{drivers.length > 1 ? 's' : ''}
            </div>
          </div>
          <HealthGauge value={avgHealth} size={isMcLaren ? 48 : 40} />
        </div>

        {/* System Health Bars */}
        <div className={`space-y-1.5 ml-2 ${isMcLaren ? 'space-y-2' : ''}`}>
          {systems.map(sys => (
            <div key={sys.name} className="flex items-center gap-2">
              <span className={`text-muted-foreground truncate ${isMcLaren ? 'text-[11px] w-24' : 'text-[10px] w-20'}`}>{sys.name}</span>
              <div className={`flex-1 rounded-full bg-[#0D1117] ${isMcLaren ? 'h-2' : 'h-1.5'}`}>
                <div
                  className="h-full rounded-full transition-all"
                  style={{
                    width: `${sys.health}%`,
                    background: sys.health >= 80 ? '#22c55e' : sys.health >= 60 ? color : '#ef4444',
                  }}
                />
              </div>
              <span className={`font-mono text-right text-muted-foreground ${isMcLaren ? 'text-[11px] w-10' : 'text-[10px] w-8'}`}>{sys.health.toFixed(0)}%</span>
            </div>
          ))}
        </div>

        {/* Status */}
        {worstLevel !== 'nominal' && (
          <div className={`flex items-center gap-1 ml-2 ${isMcLaren ? 'mt-3 text-[10px]' : 'mt-2 text-[9px]'}`} style={{ color: worstLevel === 'critical' ? '#ef4444' : '#f59e0b' }}>
            <span>⚠</span>
            <span>{drivers.filter(d => d.level === 'critical').length} critical</span>
          </div>
        )}
      </button>
    );
  };

  return (
    <div className="space-y-4">
      {/* McLaren — prominent row */}
      {mclarenTeam && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {renderTeamCard(mclarenTeam.teamId, mclarenTeam.name, mclarenTeam.drivers, true)}
        </div>
      )}

      {/* Other teams grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
        {otherTeams.map(({ teamId, name, drivers }) =>
          renderTeamCard(teamId, name, drivers, false)
        )}
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════════════
   TELEMETRY PILLAR
   ══════════════════════════════════════════════════════════════════════ */

interface ConstructorProfile {
  constructor_id: string;
  season: number;
  championship_position?: number;
  championship_points?: number;
  championship_wins?: number;
  total_podiums?: number;
  total_points?: number;
  total_wins?: number;
  dnf_count?: number;
  total_races?: number;
  avg_grid_position?: number;
  avg_finish_position?: number;
  points_per_race?: number;
  avg_qual_position?: number;
  q3_rate?: number;
  pole_count?: number;
  front_row_count?: number;
  fleet_avg_speed?: number;
  fleet_avg_throttle?: number;
  fleet_avg_brake_pct?: number;
  fleet_avg_drs_pct?: number;
  fleet_top_speed?: number;
  drivers?: { driver_code: string; driver_name: string; position?: number; points?: number; wins?: number }[];
}

function TeamTelemetryView({ teamId, teamName, teamColor, season, vehicles: _vehicles }: {
  teamId: string; teamName: string; teamColor: string; season: number; vehicles?: VehicleData[];
}) {
  const [profile, setProfile] = useState<ConstructorProfile | null>(null);
  const [raceResults, setRaceResults] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [similarTeams, setSimilarTeams] = useState<SimilarTeam[]>([]);
  const [intraPairs, setIntraPairs] = useState<IntraTeamPair[]>([]);
  const [telKex, setTelKex] = useState<Record<string, any>>({});
  const [telKexLoading, setTelKexLoading] = useState<Record<string, boolean>>({});

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`/api/constructor_profiles?season=${season}&constructor_id=${teamId}`).then(r => r.json()),
      fetch('/api/jolpica/race_results').then(r => r.json()),
    ])
      .then(([profiles, results]) => {
        const p = Array.isArray(profiles) ? profiles[0] : profiles;
        setProfile(p ?? null);
        setRaceResults(Array.isArray(results) ? results : []);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [teamId, season]);

  // Fetch team similarity data (season-aware for 2025+)
  useEffect(() => {
    const vpName = TEAM_VP_NAME[teamId];
    if (!vpName) return;
    const seasonParam = season >= 2025 ? `&season=${season}` : '';
    fetch(`/api/local/team_intel/similar/${encodeURIComponent(vpName)}?k=5${seasonParam}`)
      .then(r => r.ok ? r.json() : []).then(setSimilarTeams).catch(() => setSimilarTeams([]));
    fetch(`/api/local/team_intel/intra/${encodeURIComponent(vpName)}?${season >= 2025 ? `season=${season}` : ''}`)
      .then(r => r.ok ? r.json() : []).then(setIntraPairs).catch(() => setIntraPairs([]));
  }, [teamId, season]);

  // Auto-generate telemetry KeX briefings when profile loads
  useEffect(() => {
    const codes = (profile?.drivers ?? []).map(d => d.driver_code).filter(Boolean);
    if (codes.length === 0) return;
    for (const code of codes) {
      if (telKex[code] || telKexLoading[code]) continue;
      setTelKexLoading(prev => ({ ...prev, [code]: true }));
      fetch(`/api/mccar-telemetry/kex/${code}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ year: season }),
      })
        .then(r => r.ok ? r.json() : null)
        .then(data => { if (data) setTelKex(prev => ({ ...prev, [code]: data })); })
        .catch(() => {})
        .finally(() => setTelKexLoading(prev => ({ ...prev, [code]: false })));
    }
  }, [profile, season]);

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-sm text-muted-foreground py-8 justify-center">
        <Loader2 className="w-4 h-4 animate-spin text-[#FF8000]" /> Loading team profile...
      </div>
    );
  }

  if (!profile) {
    return <div className="text-sm text-muted-foreground py-8 text-center">No profile data for {teamName} ({season})</div>;
  }

  // Points progression from race results
  const teamRaces = raceResults
    .filter((r: any) => r.constructor_id === teamId && r.season === season)
    .sort((a: any, b: any) => (a.round ?? 0) - (b.round ?? 0));

  const pointsByRound: Record<number, number> = {};
  for (const r of teamRaces) {
    const round = r.round ?? 0;
    pointsByRound[round] = (pointsByRound[round] ?? 0) + (r.points ?? 0);
  }
  let cumulative = 0;
  const progression = Object.entries(pointsByRound)
    .sort(([a], [b]) => Number(a) - Number(b))
    .map(([round, pts]) => {
      cumulative += pts;
      return { round: `R${round}`, points: cumulative };
    });

  const kpi = (label: string, value: string | number | undefined, icon: React.ElementType) => {
    const Icon = icon;
    return (
      <div className="bg-[#1A1F2E] rounded-xl border border-[rgba(255,128,0,0.12)] p-3">
        <div className="flex items-center gap-2 mb-1">
          <Icon className="w-3.5 h-3.5 text-muted-foreground" />
          <span className="text-[10px] text-muted-foreground uppercase tracking-wider">{label}</span>
        </div>
        <div className="text-lg font-semibold text-foreground">{value ?? '—'}</div>
      </div>
    );
  };

  return (
    <div className="space-y-4">
      {/* Team Header */}
      <div className="bg-[#1A1F2E] rounded-xl border border-[rgba(255,128,0,0.12)] p-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg flex items-center justify-center" style={{ background: `${teamColor}20`, border: `1px solid ${teamColor}40` }}>
            {TEAM_LOGOS[teamId] ? (
              <img src={TEAM_LOGOS[teamId]} alt={teamName} className="w-7 h-7 object-contain" />
            ) : (
              <Trophy className="w-5 h-5" style={{ color: teamColor }} />
            )}
          </div>
          <div>
            <div className="text-lg font-semibold text-foreground">{teamName}</div>
            <div className="text-sm text-muted-foreground">
              {season} Season — P{profile.championship_position ?? '?'} — {profile.championship_points ?? profile.total_points ?? 0} pts
            </div>
          </div>
        </div>
      </div>

      {/* KPI Grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        {kpi('Position', profile.championship_position ? `P${profile.championship_position}` : '—', Trophy)}
        {kpi('Points', profile.championship_points ?? profile.total_points, Flag)}
        {kpi('Wins', profile.championship_wins ?? profile.total_wins ?? 0, Trophy)}
        {kpi('Podiums', profile.total_podiums ?? 0, TrendingUp)}
        {kpi('Avg Grid', profile.avg_grid_position?.toFixed(1), Gauge)}
        {kpi('Avg Finish', profile.avg_finish_position?.toFixed(1), Flag)}
      </div>

      {/* Driver Lineup + Qualifying + Pit Stops */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {/* Driver Lineup */}
        <div className="bg-[#1A1F2E] rounded-xl border border-[rgba(255,128,0,0.12)] p-3">
          <h3 className="text-[12px] font-medium text-foreground mb-2 flex items-center gap-1.5">
            <Users className="w-3.5 h-3.5" style={{ color: teamColor }} /> Driver Lineup
          </h3>
          <div className="space-y-2">
            {(profile.drivers ?? []).map(d => (
              <div key={d.driver_code} className="flex items-center justify-between text-[12px]">
                <span className="text-foreground font-medium">{d.driver_code ?? d.driver_name}</span>
                <span className="text-muted-foreground">
                  {d.position ? `P${d.position}` : ''} {d.points != null ? `• ${d.points} pts` : ''}
                </span>
              </div>
            ))}
            {(!profile.drivers || profile.drivers.length === 0) && (
              <span className="text-[11px] text-muted-foreground">No driver data</span>
            )}
          </div>
        </div>

        {/* Qualifying */}
        <div className="bg-[#1A1F2E] rounded-xl border border-[rgba(255,128,0,0.12)] p-3">
          <h3 className="text-[12px] font-medium text-foreground mb-2 flex items-center gap-1.5">
            <Gauge className="w-3.5 h-3.5" style={{ color: teamColor }} /> Qualifying
          </h3>
          <div className="space-y-1.5 text-[12px]">
            <div className="flex justify-between"><span className="text-muted-foreground">Avg Qual Position</span><span className="text-foreground font-mono">{profile.avg_qual_position?.toFixed(1) ?? '—'}</span></div>
            <div className="flex justify-between"><span className="text-muted-foreground">Q3 Rate</span><span className="text-foreground font-mono">{profile.q3_rate != null ? `${(profile.q3_rate * 100).toFixed(0)}%` : '—'}</span></div>
            <div className="flex justify-between"><span className="text-muted-foreground">Poles</span><span className="text-foreground font-mono">{profile.pole_count ?? 0}</span></div>
            <div className="flex justify-between"><span className="text-muted-foreground">Front Rows</span><span className="text-foreground font-mono">{profile.front_row_count ?? 0}</span></div>
          </div>
        </div>

        {/* Race Stats */}
        <div className="bg-[#1A1F2E] rounded-xl border border-[rgba(255,128,0,0.12)] p-3">
          <h3 className="text-[12px] font-medium text-foreground mb-2 flex items-center gap-1.5">
            <Timer className="w-3.5 h-3.5" style={{ color: teamColor }} /> Race Stats
          </h3>
          <div className="space-y-1.5 text-[12px]">
            <div className="flex justify-between"><span className="text-muted-foreground">Races</span><span className="text-foreground font-mono">{profile.total_races ?? '—'}</span></div>
            <div className="flex justify-between"><span className="text-muted-foreground">DNFs</span><span className="text-foreground font-mono">{profile.dnf_count ?? 0}</span></div>
            <div className="flex justify-between"><span className="text-muted-foreground">Pts/Race</span><span className="text-foreground font-mono">{profile.points_per_race?.toFixed(1) ?? '—'}</span></div>
            <div className="flex justify-between"><span className="text-muted-foreground">Top Speed</span><span className="text-foreground font-mono">{profile.fleet_top_speed ?? '—'} km/h</span></div>
          </div>
        </div>
      </div>

      {/* Fleet Telemetry Bars */}
      {(profile.fleet_avg_speed || profile.fleet_avg_throttle || profile.fleet_avg_brake_pct) && (
        <div className="bg-[#1A1F2E] rounded-xl border border-[rgba(255,128,0,0.12)] p-3">
          <h3 className="text-[12px] font-medium text-foreground mb-3">Fleet Telemetry Averages</h3>
          <div className="space-y-2">
            {[
              { label: 'Speed', value: profile.fleet_avg_speed, max: 250, unit: 'km/h' },
              { label: 'Throttle', value: profile.fleet_avg_throttle, max: 100, unit: '%' },
              { label: 'Brake', value: profile.fleet_avg_brake_pct, max: 100, unit: '%' },
              { label: 'DRS', value: profile.fleet_avg_drs_pct, max: 100, unit: '%' },
            ].filter(m => m.value != null).map(m => (
              <div key={m.label} className="flex items-center gap-3 text-[11px]">
                <span className="w-16 text-muted-foreground">{m.label}</span>
                <div className="flex-1 h-2 rounded-full bg-[#0D1117]">
                  <div className="h-full rounded-full" style={{ width: `${((m.value ?? 0) / m.max) * 100}%`, background: teamColor }} />
                </div>
                <span className="w-16 text-right font-mono text-foreground">{m.value?.toFixed(1)} {m.unit}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Points Progression */}
      {progression.length > 1 && (
        <div className="bg-[#1A1F2E] rounded-xl border border-[rgba(255,128,0,0.12)] p-3">
          <h3 className="text-[12px] font-medium text-foreground mb-3">Points Progression</h3>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={progression}>
              <XAxis dataKey="round" tick={{ fill: '#6b7280', fontSize: 10 }} tickLine={false} axisLine={false} />
              <YAxis tick={{ fill: '#6b7280', fontSize: 10 }} tickLine={false} axisLine={false} width={35} />
              <Tooltip contentStyle={{ background: '#1A1F2E', border: '1px solid rgba(255,128,0,0.2)', borderRadius: 8, fontSize: 11 }} />
              <Line type="monotone" dataKey="points" stroke={teamColor} strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* ── Intra-Team Driver Similarity ── */}
      {intraPairs.length > 0 && (
        <div className="bg-[#1A1F2E] rounded-xl border border-[rgba(255,128,0,0.12)] p-4">
          <h3 className="text-[12px] font-medium text-foreground mb-1 flex items-center gap-1.5">
            <GitCompare className="w-3.5 h-3.5" style={{ color: teamColor }} /> Intra-Team Driver Similarity
          </h3>
          <p className="text-[10px] text-muted-foreground/60 mb-3">How similar are {teamName}'s drivers based on vector profiles</p>
          <div className="space-y-2">
            {intraPairs.map(p => {
              const pct = Math.round(p.score * 100);
              return (
                <div key={`${p.driver_a}-${p.driver_b}`} className="flex items-center gap-3 text-[11px]">
                  <span className="font-semibold text-foreground w-8">{p.driver_a}</span>
                  <span className="text-muted-foreground text-[10px]">vs</span>
                  <span className="font-semibold text-foreground w-8">{p.driver_b}</span>
                  <div className="flex-1 h-2 rounded-full bg-[#0D1117]">
                    <div className="h-full rounded-full transition-all" style={{ width: `${pct}%`, background: teamColor }} />
                  </div>
                  <span className="font-mono w-10 text-right" style={{ color: teamColor }}>{pct}%</span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ── Similar Teams ── */}
      {similarTeams.length > 0 && (
        <div className="bg-[#1A1F2E] rounded-xl border border-[rgba(255,128,0,0.12)] p-4">
          <h3 className="text-[12px] font-medium text-foreground mb-1 flex items-center gap-1.5">
            <Users className="w-3.5 h-3.5" style={{ color: teamColor }} /> Similar Teams
          </h3>
          <p className="text-[10px] text-muted-foreground/60 mb-3">Teams with the most similar combined driver profiles</p>
          <div className="space-y-2">
            {similarTeams.map((st, i) => {
              const pct = Math.round(st.score * 100);
              // Find team color by matching VP name back to constructor id
              const matchedId = Object.entries(TEAM_VP_NAME).find(([, v]) => v === st.team)?.[0];
              const stColor = matchedId ? (TEAM_COLORS[matchedId] ?? '#6b7280') : '#6b7280';
              return (
                <div key={st.team} className="flex items-center gap-3 text-[11px]">
                  <span className="text-muted-foreground font-mono w-4">{i + 1}</span>
                  <span className="w-2 h-5 rounded-sm" style={{ backgroundColor: stColor }} />
                  <div className="flex-1 min-w-0">
                    <span className="font-semibold text-foreground">{st.team}</span>
                    <span className="text-[10px] text-muted-foreground ml-2">{st.drivers.join(', ')}</span>
                  </div>
                  <div className="w-20 h-1.5 rounded-full bg-[#0D1117]">
                    <div className="h-full rounded-full transition-all" style={{ width: `${pct}%`, backgroundColor: stColor }} />
                  </div>
                  <span className="font-mono w-10 text-right" style={{ color: stColor }}>{pct}%</span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ── KeX Car Telemetry Briefings ── */}
      {(profile.drivers ?? []).some(d => d.driver_code && (telKex[d.driver_code] || telKexLoading[d.driver_code])) && (
        <div className="space-y-3">
          <h3 className="text-[12px] font-medium text-foreground flex items-center gap-1.5">
            {teamName} Car Telemetry Intelligence
          </h3>
          {(profile.drivers ?? []).map(d => {
            const code = d.driver_code;
            if (!code) return null;
            const kex = telKex[code] ?? null;
            const kexLoading = telKexLoading[code] ?? false;
            if (!kex && !kexLoading) return null;
            return (
              <KexBriefingCard
                key={code}
                title={`${code} — ${teamName} Car Telemetry`}
                icon="sparkles"
                kex={kex}
                loading={kexLoading}
                loadingText={`Generating ${code} telemetry briefing…`}
              />
            );
          })}
        </div>
      )}
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════════════
   ANOMALY PILLAR
   ══════════════════════════════════════════════════════════════════════ */

function TeamAnomalyView({ teamId, teamName, teamColor, vehicles, loading }: {
  teamId: string; teamName: string; teamColor: string; vehicles: VehicleData[]; loading: boolean;
}) {
  const teamDrivers = useMemo(() =>
    vehicles.filter(v => teamIdFromName(v.team) === teamId),
    [vehicles, teamId]
  );
  const [anomKex, setAnomKex] = useState<Record<string, any>>({});
  const [anomKexLoading, setAnomKexLoading] = useState<Record<string, boolean>>({});

  // Auto-generate anomaly KeX briefings when team drivers load
  useEffect(() => {
    if (loading || teamDrivers.length === 0) return;
    for (const v of teamDrivers) {
      if (anomKex[v.code] || anomKexLoading[v.code]) continue;
      setAnomKexLoading(prev => ({ ...prev, [v.code]: true }));
      fetch(`/api/anomaly/kex/${v.code}`, { method: 'POST' })
        .then(r => r.ok ? r.json() : null)
        .then(data => { if (data) setAnomKex(prev => ({ ...prev, [v.code]: data })); })
        .catch(() => {})
        .finally(() => setAnomKexLoading(prev => ({ ...prev, [v.code]: false })));
    }
  }, [teamDrivers, loading]);

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-sm text-muted-foreground py-8 justify-center">
        <Loader2 className="w-4 h-4 animate-spin text-[#FF8000]" /> Loading anomaly data...
      </div>
    );
  }

  if (teamDrivers.length === 0) {
    return <div className="text-sm text-muted-foreground py-8 text-center">No anomaly data for {teamName}</div>;
  }

  const avgHealth = teamDrivers.reduce((s, v) => s + v.overallHealth, 0) / teamDrivers.length;
  const worstLevel = teamDrivers.some(v => v.level === 'critical') ? 'critical'
    : teamDrivers.some(v => v.level === 'warning') ? 'warning' : 'nominal';

  return (
    <div className="space-y-4">
      {/* Team Health Summary */}
      <div className="bg-[#1A1F2E] rounded-xl border border-[rgba(255,128,0,0.12)] p-4">
        <div className="flex items-center gap-4">
          <HealthGauge value={avgHealth} size={64} />
          {TEAM_LOGOS[teamId] && <img src={TEAM_LOGOS[teamId]} alt={teamName} className="h-8 object-contain" />}
          <div>
            <div className="text-lg font-semibold text-foreground">{teamName} Fleet Health</div>
            <div className="flex items-center gap-2 mt-1">
              <StatusBadge status={worstLevel} />
              <span className="text-[11px] text-muted-foreground">
                {teamDrivers.length} driver{teamDrivers.length > 1 ? 's' : ''} monitored
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Driver Comparison */}
      <div className={`grid gap-3 ${teamDrivers.length >= 2 ? 'grid-cols-1 md:grid-cols-2' : 'grid-cols-1'}`}>
        {teamDrivers.map(v => (
          <div key={v.code} className="bg-[#1A1F2E] rounded-xl border border-[rgba(255,128,0,0.12)] p-4">
            <div className="flex items-center gap-3 mb-3">
              <HealthGauge value={v.overallHealth} size={48} />
              <div>
                <div className="text-sm font-semibold text-foreground">{v.code} — {v.driver}</div>
                <div className="flex items-center gap-2 mt-0.5">
                  <StatusBadge status={v.level} />
                  <span className="text-[10px] text-muted-foreground">Last: {v.lastRace}</span>
                </div>
              </div>
            </div>

            {/* Systems */}
            <div className="space-y-2">
              {v.systems.map(sys => {
                const MI = MAINTENANCE_LABELS[sys.maintenanceAction ?? 'none'] ?? MAINTENANCE_LABELS.none;
                const MIcon = MI.icon;
                return (
                  <div key={sys.name} className="flex items-center gap-2">
                    <span className="text-[11px] text-muted-foreground w-24 shrink-0">{sys.name}</span>
                    <div className="flex-1 h-1.5 rounded-full bg-[#0D1117]">
                      <div className="h-full rounded-full" style={{ width: `${sys.health}%`, background: levelColor(sys.level) }} />
                    </div>
                    <span className="text-[10px] font-mono w-10 text-right" style={{ color: levelColor(sys.level) }}>
                      {sys.health.toFixed(0)}%
                    </span>
                    <div className="flex items-center gap-0.5 text-[9px] w-20"
                      style={{ color: MI.color }}>
                      <MIcon className="w-2.5 h-2.5" />
                      <span>{MI.label}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      {/* System-by-System Comparison Table */}
      {teamDrivers.length >= 2 && (
        <div className="bg-[#1A1F2E] rounded-xl border border-[rgba(255,128,0,0.12)] p-3">
          <h3 className="text-[12px] font-medium text-foreground mb-2">System Comparison</h3>
          <table className="w-full text-[11px]">
            <thead>
              <tr className="text-muted-foreground border-b border-[rgba(255,128,0,0.08)]">
                <th className="text-left py-1.5 pr-3">System</th>
                {teamDrivers.map(v => (
                  <th key={v.code} className="text-center px-2 py-1.5">{v.code}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(teamDrivers[0]?.systems ?? []).map(sys => (
                <tr key={sys.name} className="border-b border-[rgba(255,128,0,0.04)]">
                  <td className="py-1.5 pr-3 text-foreground">{sys.name}</td>
                  {teamDrivers.map(v => {
                    const dSys = v.systems.find(s => s.name === sys.name);
                    return (
                      <td key={v.code} className="text-center px-2">
                        <span className="font-mono" style={{ color: dSys ? levelColor(dSys.level) : '#6b7280' }}>
                          {dSys ? `${dSys.health.toFixed(0)}%` : '—'}
                        </span>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Race-by-Race Health Trend */}
      {teamDrivers.length > 0 && teamDrivers[0].races.length > 1 && (() => {
        const raceNames = teamDrivers[0].races.slice(-10).map(r => r.race);
        const trendData = raceNames.map(race => {
          const entry: Record<string, any> = { race };
          for (const v of teamDrivers) {
            const raceData = v.races.find(r => r.race === race);
            if (raceData) {
              const sysValues = Object.values(raceData.systems).map((s: any) => s.health);
              entry[v.code] = sysValues.length > 0 ? sysValues.reduce((a: number, b: number) => a + b, 0) / sysValues.length : null;
            }
          }
          return entry;
        });

        return (
          <div className="bg-[#1A1F2E] rounded-xl border border-[rgba(255,128,0,0.12)] p-3">
            <h3 className="text-[12px] font-medium text-foreground mb-3">Race-by-Race Health Trend</h3>
            <ResponsiveContainer width="100%" height={160}>
              <AreaChart data={trendData}>
                <XAxis dataKey="race" tick={{ fill: '#6b7280', fontSize: 9 }} tickLine={false} axisLine={false} />
                <YAxis domain={[0, 100]} tick={{ fill: '#6b7280', fontSize: 10 }} tickLine={false} axisLine={false} width={30} />
                <Tooltip contentStyle={{ background: '#1A1F2E', border: '1px solid rgba(255,128,0,0.2)', borderRadius: 8, fontSize: 11 }} />
                {teamDrivers.map((v, i) => (
                  <Area
                    key={v.code}
                    type="monotone"
                    dataKey={v.code}
                    stroke={i === 0 ? teamColor : `${teamColor}88`}
                    fill={i === 0 ? `${teamColor}15` : `${teamColor}08`}
                    strokeWidth={1.5}
                    dot={false}
                  />
                ))}
              </AreaChart>
            </ResponsiveContainer>
          </div>
        );
      })()}

      {/* ── KeX Anomaly Briefings ── */}
      {teamDrivers.some(v => anomKex[v.code] || anomKexLoading[v.code]) && (
        <div className="space-y-3">
          <h3 className="text-[12px] font-medium text-foreground flex items-center gap-1.5">
            {teamName} Health &amp; Reliability Intelligence
          </h3>
          {teamDrivers.map(v => {
            const kex = anomKex[v.code] ?? null;
            const kexLoading = anomKexLoading[v.code] ?? false;
            if (!kex && !kexLoading) return null;
            return (
              <KexBriefingCard
                key={v.code}
                title={`${v.code} — ${teamName} Health & Reliability`}
                icon="brain"
                kex={kex}
                loading={kexLoading}
                loadingText={`Generating ${v.code} health briefing…`}
              />
            );
          })}
        </div>
      )}
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════════════
   FORECAST PILLAR
   ══════════════════════════════════════════════════════════════════════ */

function TeamForecastView({ teamId, teamName, teamColor, vehicles, season }: {
  teamId: string; teamName: string; teamColor: string; vehicles: VehicleData[];
  season: number;
}) {
  const [profile, setProfile] = useState<ConstructorProfile | null>(null);

  useEffect(() => {
    fetch(`/api/constructor_profiles?season=${season}&constructor_id=${teamId}`)
      .then(r => r.json())
      .then(data => setProfile(Array.isArray(data) ? data[0] : data))
      .catch(() => {});
  }, [teamId, season]);

  // Resolve driver codes from profile or anomaly data
  const driverCodes = useMemo(() => {
    if (profile?.drivers?.length) {
      return profile.drivers.map(d => d.driver_code).filter(Boolean);
    }
    return vehicles.filter(v => teamIdFromName(v.team) === teamId).map(v => v.code);
  }, [profile, vehicles, teamId]);

  if (driverCodes.length === 0) {
    return <div className="text-sm text-muted-foreground py-8 text-center">No driver data for {teamName} to forecast</div>;
  }

  return (
    <div className="space-y-4">
      {/* Forecast panels per driver */}
      {driverCodes.map(code => (
        <div key={code} className="bg-[#1A1F2E] rounded-xl border border-[rgba(255,128,0,0.12)] p-3">
          <h3 className="text-[12px] font-medium text-foreground mb-3 flex items-center gap-1.5">
            <TrendingUp className="w-3 h-3" style={{ color: teamColor }} />
            Feature Forecasts — {code}
          </h3>
          <ForecastChart driverCode={code} />
        </div>
      ))}
    </div>
  );
}
