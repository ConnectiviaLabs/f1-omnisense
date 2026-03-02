import type { JolpicaDriverStanding, JolpicaConstructorStanding, JolpicaRaceResult } from '../types';

const LOCAL_BASE = '/api/jolpica';

async function fetchLocal<T>(route: string): Promise<T> {
  const res = await fetch(`${LOCAL_BASE}/${route}`);
  if (!res.ok) throw new Error(`Jolpica ${route}: ${res.status}`);
  return res.json();
}

export async function getDriverStandings(season: string = 'current'): Promise<JolpicaDriverStanding[]> {
  const data = await fetchLocal<JolpicaDriverStanding[]>('driver_standings');
  if (!data || data.length === 0) return [];
  if (season === 'current') {
    const maxSeason = Math.max(...data.map((d: any) => Number(d.season || 0)));
    return data.filter((d: any) => Number(d.season || 0) === maxSeason);
  }
  return data.filter((d: any) => d.season === season);
}

export async function getConstructorStandings(season: string = 'current'): Promise<JolpicaConstructorStanding[]> {
  const data = await fetchLocal<JolpicaConstructorStanding[]>('constructor_standings');
  if (!data || data.length === 0) return [];
  if (season === 'current') {
    const maxSeason = Math.max(...data.map((d: any) => Number(d.season || 0)));
    return data.filter((d: any) => Number(d.season || 0) === maxSeason);
  }
  return data.filter((d: any) => d.season === season);
}

export async function getRaceResults(season: string = 'current'): Promise<JolpicaRaceResult[]> {
  const data = await fetchLocal<JolpicaRaceResult[]>('race_results');
  if (!data || data.length === 0) return [];
  if (season === 'current') {
    const maxSeason = Math.max(...data.map((d: any) => Number(d.season || 0)));
    return data.filter((d: any) => Number(d.season || 0) === maxSeason);
  }
  return data.filter((d: any) => d.season === season);
}

export async function getQualifyingResults(season: string = 'current'): Promise<any[]> {
  const data = await fetchLocal<any[]>('qualifying');
  if (!data || data.length === 0) return [];
  if (season === 'current') {
    const maxSeason = Math.max(...data.map((d: any) => Number(d.season || 0)));
    return data.filter((d: any) => Number(d.season || 0) === maxSeason);
  }
  return data;
}

export async function getCircuits(_season: string = 'current'): Promise<any[]> {
  return fetchLocal<any[]>('circuits');
}

export async function getSeasons(): Promise<any[]> {
  return fetchLocal<any[]>('seasons');
}
