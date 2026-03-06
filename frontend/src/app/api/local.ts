// Local data API — reads pre-fetched JSON from /f1/data/other/ via Vite dev server

const LOCAL_BASE = '/api';

async function fetchLocal<T>(route: string): Promise<T> {
  const res = await fetch(`${LOCAL_BASE}/${route}`);
  if (!res.ok) throw new Error(`Local data ${route}: ${res.status}`);
  return res.json();
}

// OpenF1 local data
export const openf1 = {
  sessions: () => fetchLocal<any[]>('openf1/sessions'),
  laps: () => fetchLocal<any[]>('openf1/laps'),
  position: () => fetchLocal<any[]>('openf1/position'),
  weather: () => fetchLocal<any[]>('openf1/weather'),
  intervals: () => fetchLocal<any[]>('openf1/intervals'),
  pit: () => fetchLocal<any[]>('openf1/pit'),
  stints: () => fetchLocal<any[]>('openf1/stints'),
  drivers: () => fetchLocal<any[]>('openf1/drivers'),
  overtakes: () => fetchLocal<any[]>('openf1/overtakes'),
  raceControl: () => fetchLocal<any[]>('openf1/race_control'),
  championshipDrivers: () => fetchLocal<any[]>('openf1/championship_drivers'),
  championshipTeams: () => fetchLocal<any[]>('openf1/championship_teams'),
};

// Jolpica local data
export const jolpica = {
  driverStandings: () => fetchLocal<any>('jolpica/driver_standings'),
  constructorStandings: () => fetchLocal<any>('jolpica/constructor_standings'),
  raceResults: () => fetchLocal<any>('jolpica/race_results'),
  qualifying: () => fetchLocal<any>('jolpica/qualifying'),
  circuits: () => fetchLocal<any>('jolpica/circuits'),
  pitStops: () => fetchLocal<any>('jolpica/pit_stops'),
  lapTimes: () => fetchLocal<any>('jolpica/lap_times'),
  drivers: () => fetchLocal<any>('jolpica/drivers'),
  seasons: () => fetchLocal<any>('jolpica/seasons'),
};

// Pipeline results
export const pipeline = {
  gdino: () => fetchLocal<any>('pipeline/gdino'),
  fused: () => fetchLocal<any>('pipeline/fused'),
  minicpm: () => fetchLocal<any>('pipeline/minicpm'),
  videomae: () => fetchLocal<any>('pipeline/videomae'),
  timesformer: () => fetchLocal<any>('pipeline/timesformer'),
  videos: () => fetchLocal<any>('pipeline/videos'),
};

// Strategy & model data
export const strategy = {
  tracker: (sessionKey?: number, year?: number) => {
    const p = sessionKey ? `?session_key=${sessionKey}` : year ? `?year=${year}` : '';
    return fetchLocal<any>(`local/strategy/tracker${p}`);
  },
  simulations: (sessionKey?: number) => {
    const p = sessionKey ? `?session_key=${sessionKey}` : '';
    return fetchLocal<any>(`local/strategy/simulations${p}`);
  },
  degradation: (circuit?: string) => {
    const p = circuit ? `?circuit=${encodeURIComponent(circuit)}` : '';
    return fetchLocal<any>(`local/strategy/degradation${p}`);
  },
  elt: (year?: number, circuit?: string) => {
    const params = new URLSearchParams();
    if (year) params.set('year', String(year));
    if (circuit) params.set('circuit', circuit);
    const qs = params.toString();
    return fetchLocal<any>(`local/strategy/elt${qs ? '?' + qs : ''}`);
  },
  scProbability: () => fetchLocal<any>('local/strategy/sc-probability'),
  xgboost: () => fetchLocal<any>('local/strategy/xgboost'),
  bilstm: () => fetchLocal<any>('local/strategy/bilstm'),
  predictLap: (params: {
    circuit: string; driver_code: string; compound: string;
    lap_start?: number; lap_end?: number; tyre_life_start?: number;
    position?: number; stint?: number; baseline_pace_s?: number | null;
  }) => fetch(`/api/local/strategy/predict-lap`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  }).then(r => { if (!r.ok) throw new Error(`predict-lap: ${r.status}`); return r.json(); }),
  battleIntel: (sessionKey?: number) => {
    const p = sessionKey ? `?session_key=${sessionKey}` : '';
    return fetchLocal<any>(`local/strategy/battle-intel${p}`);
  },
};

// CSV data fetch helper
export async function fetchCSV(path: string): Promise<string> {
  const res = await fetch(`${LOCAL_BASE}/${path}`);
  if (!res.ok) throw new Error(`CSV ${path}: ${res.status}`);
  return res.text();
}

// Parse CSV string into array of objects
export function parseCSV(csv: string): Record<string, string>[] {
  const lines = csv.trim().split('\n');
  if (lines.length < 2) return [];
  const headers = lines[0].split(',');
  return lines.slice(1).map(line => {
    const values = line.split(',');
    const obj: Record<string, string> = {};
    headers.forEach((h, i) => { obj[h] = values[i] ?? ''; });
    return obj;
  });
}
