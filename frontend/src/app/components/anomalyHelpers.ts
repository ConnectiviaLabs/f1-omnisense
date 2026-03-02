// Shared anomaly/health types and helpers — used by FleetOverview & LiveDashboard
import {
  Gauge, Activity, Disc, Cpu,
  ShieldAlert, Bell, Eye, FileText, CheckCircle2,
} from 'lucide-react';

// ─── Types (aligned with anomaly_scores.json) ──────────────────────
export type HealthLevel = 'nominal' | 'warning' | 'critical';

export interface SystemHealth {
  name: string;
  icon: React.ElementType;
  health: number;
  level: HealthLevel;
  details: string;
  voteCount: number;
  totalModels: number;
  metrics: { label: string; value: string }[];
  maintenanceAction?: string;
  severityProbabilities?: Record<string, number>;
}

export interface RaceHealth {
  race: string;
  systems: Record<string, {
    health: number;
    level: string;
    vote_severity: string;
    score_mean: number;
    voting_score: number;
    vote_count: number;
    total_models: number;
    top_model: string;
    features: Record<string, number>;
    classifier_severity?: string;
    classifier_confidence?: number;
    severity_probabilities?: Record<string, number>;
    maintenance_action?: string;
  }>;
}

export interface VehicleData {
  driver: string;
  number: number;
  code: string;
  overallHealth: number;
  level: HealthLevel;
  lastRace: string;
  systems: SystemHealth[];
  races: RaceHealth[];
}

// ─── Constants ─────────────────────────────────────────────────────
export const SYSTEM_ICONS: Record<string, React.ElementType> = {
  'Speed': Gauge,
  'Lap Pace': Activity,
  'Tyre Management': Disc,
};

export const MAINTENANCE_LABELS: Record<string, { label: string; icon: React.ElementType; color: string }> = {
  alert_and_remediate: { label: 'Immediate Action', icon: ShieldAlert, color: '#ef4444' },
  alert:              { label: 'Schedule Review',   icon: Bell,        color: '#FF8000' },
  log_and_monitor:    { label: 'Monitor',           icon: Eye,         color: '#eab308' },
  log:                { label: 'Logged',            icon: FileText,    color: '#6b7280' },
  none:               { label: 'No Action',         icon: CheckCircle2, color: '#22c55e' },
};

export const SEVERITY_COLORS: Record<string, string> = {
  normal: '#22c55e', low: '#6b7280', medium: '#eab308', high: '#FF8000', critical: '#ef4444',
};

// ─── Helpers ───────────────────────────────────────────────────────
export function mapLevel(anomalyLevel: string): HealthLevel {
  switch (anomalyLevel) {
    case 'critical':
    case 'high':
      return 'critical';
    case 'medium':
      return 'warning';
    default:
      return 'nominal';
  }
}

export function buildSystemHealth(
  sysName: string,
  data: RaceHealth['systems'][string],
): SystemHealth {
  const level = data.classifier_severity ? mapLevel(data.classifier_severity) : mapLevel(data.level);
  const featureEntries = Object.entries(data.features);
  const details = featureEntries.map(([k, v]) => `${k}: ${v}`).join(', ') || `Score: ${data.score_mean.toFixed(3)}`;

  return {
    name: sysName,
    icon: SYSTEM_ICONS[sysName] ?? Cpu,
    health: data.health,
    level,
    details,
    voteCount: data.vote_count,
    totalModels: data.total_models,
    metrics: featureEntries.map(([k, v]) => ({ label: k, value: String(v) })),
    maintenanceAction: data.maintenance_action,
    severityProbabilities: data.severity_probabilities,
  };
}

export const levelColor = (l: HealthLevel) =>
  l === 'nominal' ? '#22c55e' : l === 'warning' ? '#FF8000' : '#ef4444';

export const levelBg = (l: HealthLevel) =>
  l === 'nominal' ? 'rgba(34,197,94,0.08)' : l === 'warning' ? 'rgba(255,128,0,0.12)' : 'rgba(239,68,68,0.08)';

export function parseAnomalyDrivers(data: any): VehicleData[] {
  return (data.drivers ?? []).map((d: any) => {
    const latestRace: RaceHealth | undefined = d.races?.[d.races.length - 1];
    const systems: SystemHealth[] = latestRace
      ? Object.entries(latestRace.systems).map(([name, sysData]) =>
          buildSystemHealth(name, sysData as RaceHealth['systems'][string])
        )
      : [];
    return {
      driver: d.driver,
      number: d.number,
      code: d.code,
      overallHealth: d.overall_health,
      level: mapLevel(d.overall_level),
      lastRace: d.last_race,
      systems,
      races: d.races ?? [],
    };
  });
}
