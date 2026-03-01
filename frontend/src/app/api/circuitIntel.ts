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
