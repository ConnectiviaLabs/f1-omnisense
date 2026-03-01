// OpenF1 API Types
export interface OpenF1Session {
  session_key: number;
  session_name: string;
  session_type: string;
  date_start: string;
  date_end: string;
  year: number;
  circuit_key: number;
  circuit_short_name: string;
  country_name: string;
  country_key: number;
  location: string;
  meeting_key: number;
  meeting_name: string;
}

export interface OpenF1Position {
  session_key: number;
  meeting_key: number;
  driver_number: number;
  date: string;
  position: number;
}

export interface OpenF1Lap {
  session_key: number;
  meeting_key: number;
  driver_number: number;
  lap_number: number;
  lap_duration: number | null;
  duration_sector_1: number | null;
  duration_sector_2: number | null;
  duration_sector_3: number | null;
  is_pit_out_lap: boolean;
  date_start: string;
  st_speed: number | null;
}

export interface OpenF1Weather {
  session_key: number;
  meeting_key: number;
  date: string;
  air_temperature: number;
  track_temperature: number;
  humidity: number;
  pressure: number;
  rainfall: boolean;
  wind_direction: number;
  wind_speed: number;
}

export interface OpenF1Interval {
  session_key: number;
  meeting_key: number;
  driver_number: number;
  date: string;
  gap_to_leader: number | null;
  interval: number | null;
}

export interface OpenF1Pit {
  session_key: number;
  meeting_key: number;
  driver_number: number;
  date: string;
  lap_number: number;
  pit_duration: number | null;
}

export interface OpenF1Stint {
  session_key: number;
  meeting_key: number;
  driver_number: number;
  stint_number: number;
  lap_start: number;
  lap_end: number;
  compound: string;
  tyre_age_at_start: number;
}

export interface OpenF1Driver {
  session_key: number;
  meeting_key: number;
  driver_number: number;
  broadcast_name: string;
  full_name: string;
  name_acronym: string;
  team_name: string;
  team_colour: string;
  first_name: string;
  last_name: string;
  country_code: string;
  headshot_url: string;
}

export interface OpenF1RaceControl {
  date: string;
  session_key: number;
  meeting_key: number;
  driver_number: number | null;
  lap_number: number | null;
  category: string;
  flag: string | null;
  scope: string | null;
  message: string;
}

// Jolpica API Types
export interface JolpicaDriverStanding {
  position: string;
  positionText: string;
  points: string;
  wins: string;
  Driver: {
    driverId: string;
    permanentNumber: string;
    code: string;
    givenName: string;
    familyName: string;
    nationality: string;
  };
  Constructors: {
    constructorId: string;
    name: string;
    nationality: string;
  }[];
}

export interface JolpicaConstructorStanding {
  position: string;
  positionText: string;
  points: string;
  wins: string;
  Constructor: {
    constructorId: string;
    name: string;
    nationality: string;
  };
}

export interface JolpicaRaceResult {
  season: string;
  round: string;
  raceName: string;
  Circuit: {
    circuitId: string;
    circuitName: string;
    Location: {
      lat: string;
      long: string;
      locality: string;
      country: string;
    };
  };
  date: string;
  Results: {
    number: string;
    position: string;
    positionText: string;
    points: string;
    Driver: {
      driverId: string;
      code: string;
      givenName: string;
      familyName: string;
    };
    Constructor: {
      constructorId: string;
      name: string;
    };
    grid: string;
    laps: string;
    status: string;
    Time?: { millis: string; time: string };
    FastestLap?: {
      rank: string;
      lap: string;
      Time: { time: string };
      AverageSpeed: { units: string; speed: string };
    };
  }[];
}

// Pipeline Extraction Types
export interface RegulationRule {
  id: string;
  category: string;
  description: string | null;
  value: string | null;
  unit: string | null;
  condition: string | null;
  reference: string | null;
  severity: string;
  source_standard: string | null;
  _source: string;
  _page: number;
}

export interface EquipmentItem {
  tag: string;
  type: string;
  description: string;
  kks: string;
  specs: Record<string, string>;
  location_description: string | null;
  _source: string;
  _page: number;
}

export interface DimensionalData {
  component: string;
  dimension: string;
  value: number | string | null;
  unit: string;
  _source: string;
  _page: number;
}

export interface MaterialSpec {
  material: string;
  application: string;
  properties: Record<string, string>;
  _source: string;
  _page: number;
}

export interface DocumentPass {
  number: number;
  name: string;
  pages_processed: number;
  items_found: number;
  tokens_in: number;
  tokens_out: number;
  cost_usd: number;
  latency_s: number;
}

export interface DocumentMeta {
  name: string;
  title: string;
  document_type: string;
  revision: string;
  date: string;
  total_pages: number;
  passes: DocumentPass[];
  sections: { number: string; title: string }[];
  standards: string[];
  topics: string[];
}

export interface PipelineStats {
  total_pages: number;
  total_rules: number;
  total_equipment: number;
  total_dimensions: number;
  total_materials: number;
  total_tokens_in: number;
  total_tokens_out: number;
  total_cost_usd: number;
  total_latency_s: number;
}

export interface IntelligenceData {
  documents: DocumentMeta[];
  rules: RegulationRule[];
  equipment: EquipmentItem[];
  dimensional_data: DimensionalData[];
  material_specs: MaterialSpec[];
  stats: PipelineStats;
}

// Driver Intelligence Types
export interface DriverPerformanceMarker {
  Driver: string;
  degradation_slope_s_per_lap: number | null;
  late_race_delta_s: number | null;
  lap_time_consistency_std: number | null;
  sector1_cv: number | null;
  sector2_cv: number | null;
  sector3_cv: number | null;
  heat_lap_delta_s: number | null;
  humidity_lap_delta_s: number | null;
  throttle_smoothness: number | null;
  brake_overlap_rate: number | null;
  avg_top_speed_kmh: number | null;
  avg_throttle_pct: number | null;
  late_race_speed_drop_kmh: number | null;
  avg_stint_length: number | null;
  long_stint_lap_delta: number | null;
}

export interface DriverOvertakeProfile {
  driver_code: string;
  driver_number: number;
  total_overtakes_made: number;
  total_times_overtaken: number;
  overtake_ratio: number;
  overtakes_per_race: number;
  times_overtaken_per_race: number;
  overtake_net: number;
  races_analysed: number;
}

export interface DriverTelemetryProfile {
  driver_code: string;
  avg_race_speed_kmh: number;
  avg_throttle_pct: number;
  avg_braking_g: number;
  max_braking_g: number;
  braking_consistency: number;
  full_throttle_ratio: number;
  drs_usage_ratio: number;
  drs_speed_gain_kmh: number;
  avg_gear: number;
  avg_upshift_rpm: number;
  avg_downshift_rpm: number;
  brake_to_throttle_avg_s: number;
  late_race_speed_drop_kmh: number;
  late_race_braking_delta: number;
  late_race_throttle_delta: number;
}

// Circuit Intelligence Types
export interface CircuitIntelligence {
  circuit_slug: string;
  circuit_name: string;
  centroid: [number, number];
  computed_length_m: number;
  coordinate_count: number;
  coordinates: [number, number][];
  drs_zones: number;
  estimated_corners: number;
  elevation_min_m: number | null;
  elevation_max_m: number | null;
  elevation_gain_m: number | null;
  sectors: number;
}

export interface CircuitPitLoss {
  circuit: string;
  avg_total_pit_s: number;
  est_pit_lane_loss_s: number;
  median_total_pit_s: number;
  sample_count: number;
  jolpica_avg_pit_duration_s: number | null;
  jolpica_median_pit_duration_s: number | null;
  jolpica_pit_sample_count: number | null;
}

export interface RaceAirDensity {
  race: string;
  year: number;
  circuit_slug: string;
  air_density_kg_m3: number;
  avg_temp_c: number;
  avg_humidity_pct: number;
  avg_surface_pressure_hpa: number;
  density_loss_pct: number;
  downforce_loss_pct: number;
  elevation_m: number;
  race_date: string;
}

export interface JolpicaQualifying {
  season: number;
  round: number;
  race_name: string;
  circuit_id: string;
  driver_id: string;
  driver_code: string;
  constructor_id: string;
  position: number;
  q1: string | null;
  q2: string | null;
  q3: string | null;
}

export interface JolpicaPitStop {
  season: number;
  round: number;
  race_name: string;
  circuit_id: string;
  driver_id: string;
  stop: number;
  lap: number;
  duration_s: number;
}

export interface JolpicaSprintResult {
  season: number;
  round: number;
  race_name: string;
  circuit_id: string;
  driver_id: string;
  driver_code: string;
  constructor_id: string;
  constructor_name: string;
  grid: number;
  position: number;
  points: number;
  laps: number;
  status: string;
  positions_gained: number;
}

// View types
export type ViewType = 'dashboard' | 'mclaren-analytics' | 'car' | 'driver' | 'ai-insights' | 'regulations' | 'media' | 'chat' | 'fleet-overview' | 'driver-intel' | 'circuit-intel';
