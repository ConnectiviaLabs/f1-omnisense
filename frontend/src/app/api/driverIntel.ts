import type { DriverPerformanceMarker, DriverOvertakeProfile, DriverTelemetryProfile } from '../types';

const BASE = '/api';

export async function getPerformanceMarkers(driver?: string): Promise<DriverPerformanceMarker[]> {
  const qs = driver ? `?driver=${driver}` : '';
  const res = await fetch(`${BASE}/driver_intel/performance_markers${qs}`);
  return res.json();
}

export async function getOvertakeProfiles(driver?: string): Promise<DriverOvertakeProfile[]> {
  const qs = driver ? `?driver=${driver}` : '';
  const res = await fetch(`${BASE}/driver_intel/overtake_profiles${qs}`);
  return res.json();
}

export async function getTelemetryProfiles(driver?: string): Promise<DriverTelemetryProfile[]> {
  const qs = driver ? `?driver=${driver}` : '';
  const res = await fetch(`${BASE}/driver_intel/telemetry_profiles${qs}`);
  return res.json();
}

export async function getOpponentDrivers(): Promise<{ drivers: any[]; count: number }> {
  const res = await fetch(`${BASE}/opponents/drivers`);
  return res.json();
}

export async function getOpponentDriver(id: string): Promise<any> {
  const res = await fetch(`${BASE}/opponents/drivers/${id}`);
  return res.json();
}

export async function compareDrivers(ids: string[]): Promise<any> {
  const qs = ids.map(id => `ids=${id}`).join('&');
  const res = await fetch(`${BASE}/opponents/compare?${qs}`);
  return res.json();
}

export async function getLeaderboard(metric: string, topN = 10): Promise<any> {
  const res = await fetch(`${BASE}/opponents/leaderboard/${metric}?top_n=${topN}`);
  return res.json();
}
