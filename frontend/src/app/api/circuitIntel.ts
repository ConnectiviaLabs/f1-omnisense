import type { CircuitIntelligence, CircuitPitLoss, RaceAirDensity } from '../types';

const BASE = '/api';

export async function getCircuits(circuit?: string): Promise<CircuitIntelligence[]> {
  const qs = circuit ? `?circuit=${circuit}` : '';
  const res = await fetch(`${BASE}/circuit_intel/circuits${qs}`);
  return res.json();
}

export async function getPitLoss(circuit?: string): Promise<CircuitPitLoss[]> {
  const qs = circuit ? `?circuit=${circuit}` : '';
  const res = await fetch(`${BASE}/circuit_intel/pit_loss${qs}`);
  return res.json();
}

export async function getAirDensity(circuit?: string, year?: number): Promise<RaceAirDensity[]> {
  const params = new URLSearchParams();
  if (circuit) params.set('circuit', circuit);
  if (year) params.set('year', String(year));
  const qs = params.toString();
  const res = await fetch(`${BASE}/circuit_intel/air_density${qs ? '?' + qs : ''}`);
  return res.json();
}

export async function getCircuitDrivers(circuit: string): Promise<any[]> {
  const res = await fetch(`${BASE}/opponents/circuits/${circuit}/drivers`);
  return res.json();
}

export interface CircuitHistoryWinner {
  season: number;
  driver_code: string;
  constructor: string;
  grid: number | null;
}

export interface CircuitHistory {
  circuit_id: string;
  race_name: string;
  seasons: number[];
  winners: CircuitHistoryWinner[];
  pole_stats: { total: number; wins: number; rate: number };
  positions_gained: { avg: number; median: number };
  top_constructors: { id: string; name: string; points: number; seasons: number }[];
  top_podiums: { driver: string; count: number }[];
  dnf_rate: number;
  total_races: number;
}

export async function getCircuitHistory(circuitId: string): Promise<CircuitHistory> {
  const res = await fetch(`${BASE}/circuit_intel/history/${encodeURIComponent(circuitId)}`);
  return res.json();
}

export interface CircuitKex {
  circuit_id: string;
  circuit_name: string;
  text: string;
  model_used: string;
  provider_used: string;
  sentiment: { label: string; score: number };
  entities: { text: string; label: string }[];
  topics: string[];
  generated_at: number;
}

export async function getCircuitKex(circuitId: string, force = false): Promise<CircuitKex> {
  const qs = force ? '?force=true' : '';
  const res = await fetch(`${BASE}/circuit_intel/kex/${encodeURIComponent(circuitId)}${qs}`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error(`KeX generation failed: ${res.status}`);
  return res.json();
}
