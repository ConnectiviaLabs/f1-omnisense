/* ═══════════════════════════════════════════════════════════════════════
 *  F1 Team Constants — Single Source of Truth
 *  Used across DriverIntel, FleetOverview, PrimeTeam, McLarenAnalytics,
 *  BacktestView, LiveDashboard, CarTelemetry, RaceStrategy, etc.
 * ═══════════════════════════════════════════════════════════════════════ */

/* ── Team metadata ── */

export const TEAMS = [
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
] as const;

/* ── Team colors — keyed by constructor_id ── */

export const TEAM_COLORS_BY_ID: Record<string, string> = {
  red_bull: '#3671C6', mclaren: '#FF8000', ferrari: '#E8002D',
  mercedes: '#27F4D2', aston_martin: '#229971', alpine: '#FF87BC',
  williams: '#64C4FF', rb: '#6692FF', sauber: '#52E252', haas: '#B6BABD',
};

/* ── Team colors — keyed by display name (for APIs returning team names) ── */

export const TEAM_COLORS_BY_NAME: Record<string, string> = {
  'Red Bull': '#3671C6', 'Red Bull Racing': '#3671C6',
  'McLaren': '#FF8000',
  'Ferrari': '#E8002D',
  'Mercedes': '#27F4D2',
  'Aston Martin': '#229971',
  'Alpine': '#FF87BC', 'Alpine F1 Team': '#FF87BC',
  'Williams': '#64C4FF',
  'RB': '#6692FF', 'RB F1 Team': '#6692FF', 'Racing Bulls': '#6692FF',
  'Kick Sauber': '#52E252', 'Sauber': '#52E252',
  'Haas F1 Team': '#B6BABD', 'Haas': '#B6BABD',
  'AlphaTauri': '#6692FF', 'Toro Rosso': '#469BFF',
  'Alfa Romeo': '#C92D4B',
  'Racing Point': '#F596C8', 'Force India': '#F596C8',
  'Renault': '#FFF500',
};

/* ── Team display names ── */

export const TEAM_DISPLAY_NAMES: Record<string, string> = {
  red_bull: 'Red Bull', mclaren: 'McLaren', ferrari: 'Ferrari',
  mercedes: 'Mercedes', aston_martin: 'Aston Martin', alpine: 'Alpine',
  williams: 'Williams', rb: 'RB', sauber: 'Kick Sauber', haas: 'Haas',
  alphatauri: 'AlphaTauri', alfa: 'Alfa Romeo',
  toro_rosso: 'Toro Rosso', renault: 'Renault', racing_point: 'Racing Point',
};

/* ── Team logos (F1 CDN) ── */

export const F1_CDN = 'https://media.formula1.com/image/upload/c_lfill,w_96/q_auto/v1740000000/common/f1/2026';

export const TEAM_LOGOS: Record<string, string> = {
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

/* ── Map display name → constructor_id (for logo/color lookups) ── */

export const TEAM_NAME_TO_ID: Record<string, string> = {
  'Red Bull': 'red_bull', 'Red Bull Racing': 'red_bull',
  'McLaren': 'mclaren', 'Ferrari': 'ferrari',
  'Mercedes': 'mercedes', 'Aston Martin': 'aston_martin',
  'Alpine': 'alpine', 'Alpine F1 Team': 'alpine',
  'Williams': 'williams',
  'RB': 'rb', 'RB F1 Team': 'rb', 'Racing Bulls': 'rb',
  'Kick Sauber': 'sauber', 'Sauber': 'sauber', 'Alfa Romeo': 'sauber',
  'Haas F1 Team': 'haas', 'Haas': 'haas',
  'AlphaTauri': 'rb', 'Toro Rosso': 'rb',
  'Visa Cash App RB': 'rb',
};

/* ── Map constructor_id → VectorProfiles team name ── */

export const TEAM_VP_NAME: Record<string, string> = {
  red_bull: 'Red Bull', mclaren: 'McLaren', ferrari: 'Ferrari',
  mercedes: 'Mercedes', aston_martin: 'Aston Martin', alpine: 'Alpine F1 Team',
  williams: 'Williams', rb: 'RB F1 Team', sauber: 'Sauber', haas: 'Haas F1 Team',
};

/* ── Tyre compound colors (standardized) ── */

export const COMPOUND_COLORS: Record<string, string> = {
  SOFT: '#ef4444',
  MEDIUM: '#f59e0b',
  HARD: '#e8e8f0',
  INTERMEDIATE: '#22c55e',
  WET: '#3b82f6',
  UNKNOWN: '#6b7280',
};

export const COMPOUND_TEXT_COLORS: Record<string, string> = {
  SOFT: '#ef4444',
  MEDIUM: '#f59e0b',
  HARD: '#d4d4d8',
  INTERMEDIATE: '#22c55e',
  WET: '#3b82f6',
  UNKNOWN: '#6b7280',
};

/* ── Nationality flags ── */

export const NATIONALITY_FLAGS: Record<string, string> = {
  'British': '\u{1F1EC}\u{1F1E7}', 'Dutch': '\u{1F1F3}\u{1F1F1}', 'Spanish': '\u{1F1EA}\u{1F1F8}',
  'Monegasque': '\u{1F1F2}\u{1F1E8}', 'Mexican': '\u{1F1F2}\u{1F1FD}', 'Australian': '\u{1F1E6}\u{1F1FA}',
  'Canadian': '\u{1F1E8}\u{1F1E6}', 'German': '\u{1F1E9}\u{1F1EA}', 'French': '\u{1F1EB}\u{1F1F7}',
  'Finnish': '\u{1F1EB}\u{1F1EE}', 'Thai': '\u{1F1F9}\u{1F1ED}', 'Japanese': '\u{1F1EF}\u{1F1F5}',
  'Chinese': '\u{1F1E8}\u{1F1F3}', 'Danish': '\u{1F1E9}\u{1F1F0}', 'American': '\u{1F1FA}\u{1F1F8}',
  'Italian': '\u{1F1EE}\u{1F1F9}', 'Brazilian': '\u{1F1E7}\u{1F1F7}', 'New Zealander': '\u{1F1F3}\u{1F1FF}',
  'Argentine': '\u{1F1E6}\u{1F1F7}', 'Polish': '\u{1F1F5}\u{1F1F1}', 'Indonesian': '\u{1F1EE}\u{1F1E9}',
  'Russian': '\u{1F1F7}\u{1F1FA}', 'Indian': '\u{1F1EE}\u{1F1F3}', 'Belgian': '\u{1F1E7}\u{1F1EA}',
  'Colombian': '\u{1F1E8}\u{1F1F4}', 'Venezuelan': '\u{1F1FB}\u{1F1EA}', 'Swedish': '\u{1F1F8}\u{1F1EA}',
  'Malaysian': '\u{1F1F2}\u{1F1FE}', 'Austrian': '\u{1F1E6}\u{1F1F9}', 'Swiss': '\u{1F1E8}\u{1F1ED}',
  'South African': '\u{1F1FF}\u{1F1E6}', 'Irish': '\u{1F1EE}\u{1F1EA}', 'Portuguese': '\u{1F1F5}\u{1F1F9}',
  'Hungarian': '\u{1F1ED}\u{1F1FA}', 'Czech': '\u{1F1E8}\u{1F1FF}', 'Korean': '\u{1F1F0}\u{1F1F7}',
  'Chilean': '\u{1F1E8}\u{1F1F1}', 'Uruguayan': '\u{1F1FA}\u{1F1FE}', 'Peruvian': '\u{1F1F5}\u{1F1EA}',
  'Romanian': '\u{1F1F7}\u{1F1F4}', 'Norwegian': '\u{1F1F3}\u{1F1F4}', 'Emirati': '\u{1F1E6}\u{1F1EA}',
  'Saudi': '\u{1F1F8}\u{1F1E6}', 'Qatari': '\u{1F1F6}\u{1F1E6}', 'Bahraini': '\u{1F1E7}\u{1F1ED}',
};

/* ── Helper: resolve team color from any format ── */

export function getTeamColor(team: string, fallback = '#666'): string {
  return TEAM_COLORS_BY_NAME[team] ?? TEAM_COLORS_BY_ID[team] ?? TEAM_COLORS_BY_ID[team.toLowerCase().replace(/\s+/g, '_')] ?? fallback;
}

/* ── Helper: resolve team logo URL from any format ── */

export function getTeamLogoUrl(team: string): string | null {
  const id = TEAM_NAME_TO_ID[team] ?? team;
  return TEAM_LOGOS[id] ?? null;
}

/* ── Helper: resolve constructor_id from team name ── */

export function teamIdFromName(teamName: string): string {
  const lower = teamName.toLowerCase();
  const direct = TEAM_NAME_TO_ID[teamName];
  if (direct) return direct;
  for (const [key, id] of Object.entries(TEAM_NAME_TO_ID)) {
    if (lower === key.toLowerCase()) return id;
    if (lower.includes(key.toLowerCase()) || key.toLowerCase().includes(lower)) return id;
  }
  return lower.replace(/\s+/g, '_');
}
