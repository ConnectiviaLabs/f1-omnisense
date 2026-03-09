import { useState, useCallback, useEffect, useRef, useMemo } from 'react';
import {
  Users,
  Shield,
  Box,
  Loader2,
  Sparkles,
  Grid3X3,
  ScatterChart as ScatterIcon,
  Brain,
  GitCompareArrows,
  X,
  Send,
  ArrowRightLeft,
  ChevronDown,
  ChevronUp,
  Search,
  SlidersHorizontal,
  TriangleRight,
} from 'lucide-react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';

// ── Types ───────────────────────────────────────────────────────────────

interface MatrixResult {
  entity_type: string;
  entities: string[];
  teams: string[];
  matrix: number[][];
  count: number;
  source: string;
}

interface ClusterEntity {
  code: string;
  team: string;
  x: number;
  y: number;
  z: number;
  cluster: number;
}

interface ClusterInfo {
  id: number;
  members: string[];
  centroid: number[];
}

interface Discriminator {
  metric: string;
  spread: number;
  cluster_values: Record<string, number>;
}

interface ClusterResult {
  entity_type: string;
  entities: ClusterEntity[];
  clusters: ClusterInfo[];
  cluster_profiles: Record<string, number>[];
  discriminators: Discriminator[];
  explained_variance: number[];
  source: string;
}

interface InsightResult {
  insight: string;
  model_used: string;
  entity_type: string;
  count: number;
  pairs: {
    most_similar: { a: string; b: string; score: number }[];
    most_dissimilar: { a: string; b: string; score: number }[];
  };
}

interface ComparePair {
  a: string;
  b: string;
  similarity: number;
  a_metrics: Record<string, number>;
  b_metrics: Record<string, number>;
}

interface CompareResult {
  pairs: ComparePair[];
  statistics: { avg: number; max: number; min: number };
  highest_pair: { a: string; b: string; similarity: number };
  lowest_pair: { a: string; b: string; similarity: number };
  entities: string[];
  suggested_questions: string[];
}

interface CrossInsightResult {
  insight: string;
  model_used: string;
  entities: string[];
  query: string;
  correlations_found: { pair: string[]; converging: string[]; diverging: string[] }[];
  suggested_questions: string[];
}

interface AvailableEntity {
  code: string;
  team: string;
}

// ── Constants ───────────────────────────────────────────────────────────

const ENTITY_TABS = [
  { value: 'driver', label: 'Driver', icon: Users, hint: 'Compare driver performance and behavioral profiles' },
  { value: 'team', label: 'Team', icon: Shield, hint: 'Compare team-level operational and strategic profiles' },
  { value: 'car', label: 'Car', icon: Box, hint: 'Compare car telemetry and mechanical profiles' },
];

const SOURCE_OPTIONS: { value: string; label: string; entities: string[] }[] = [
  { value: 'VectorProfiles', label: 'Vector Profiles', entities: ['driver', 'team', 'car'] },
  { value: 'victory_driver_profiles', label: 'Victory Driver', entities: ['driver'] },
  { value: 'victory_car_profiles', label: 'Victory Car', entities: ['car'] },
  { value: 'victory_team_kb', label: 'Victory Team', entities: ['team'] },
];

// Preferred source per entity type (Victory-specific first, VectorProfiles as fallback)
const PREFERRED_SOURCE: Record<string, string> = {
  driver: 'victory_driver_profiles',
  car: 'victory_car_profiles',
  team: 'victory_team_kb',
};

const CLUSTER_COLORS = ['#FF8000', '#3B82F6', '#05DF72', '#FB2C36', '#A855F7', '#F59E0B'];

// ── Helpers ─────────────────────────────────────────────────────────────

function simColor(score: number): string {
  // Low → dim gray, High → bright orange
  if (score >= 0.95) return 'rgba(255,128,0,0.85)';
  if (score >= 0.85) return 'rgba(255,128,0,0.55)';
  if (score >= 0.75) return 'rgba(255,128,0,0.35)';
  if (score >= 0.60) return 'rgba(255,128,0,0.18)';
  return 'rgba(255,128,0,0.06)';
}

function simText(score: number): string {
  // High contrast: white on bright cells, warm on mid, dim on low
  if (score >= 0.90) return '#FFFFFF';
  if (score >= 0.80) return 'rgba(255,255,255,0.95)';
  if (score >= 0.70) return 'rgba(255,220,180,0.9)';
  return 'rgba(200,200,215,0.7)';
}

function simLabel(score: number): { label: string; color: string } {
  if (score >= 0.8) return { label: 'High', color: '#05DF72' };
  if (score >= 0.5) return { label: 'Medium', color: '#FF8000' };
  return { label: 'Low', color: '#FB2C36' };
}

/* ── Animated Circular Progress ──────────────────────────────────────── */

function CircularProgress({ value, size = 140 }: { value: number; size?: number }) {
  const r = (size - 20) / 2;
  const circ = 2 * Math.PI * r;
  const { label, color } = simLabel(value);
  const pct = Math.round(value * 100);

  return (
    <div className="flex flex-col items-center gap-2">
      <svg width={size} height={size} className="transform -rotate-90">
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="rgba(255,128,0,0.08)" strokeWidth={8} />
        <circle
          cx={size / 2} cy={size / 2} r={r} fill="none"
          stroke={color} strokeWidth={8} strokeLinecap="round"
          strokeDasharray={circ}
          strokeDashoffset={circ * (1 - value)}
          style={{ transition: 'stroke-dashoffset 1s ease-out, stroke 0.3s ease' }}
        />
      </svg>
      <div className="absolute flex flex-col items-center justify-center" style={{ width: size, height: size }}>
        <span className="text-2xl font-bold text-foreground">{pct}%</span>
        <span className="text-[11px] font-medium" style={{ color }}>{label}</span>
      </div>
    </div>
  );
}

/* ── Entity Metric Card ──────────────────────────────────────────────── */

function EntityMetricCard({ code, metrics }: { code: string; metrics: Record<string, number> }) {
  const entries = Object.entries(metrics).slice(0, 8);
  return (
    <div className="rounded-lg border border-border bg-card/30 p-4">
      <div className="text-sm font-semibold text-primary mb-3">{code}</div>
      <div className="space-y-1.5">
        {entries.map(([k, v]) => (
          <div key={k} className="flex items-center justify-between text-[11px]">
            <span className="text-muted-foreground">{k.replace(/_/g, ' ')}</span>
            <span className="text-foreground font-mono">{typeof v === 'number' ? v.toFixed(2) : v}</span>
          </div>
        ))}
        {entries.length === 0 && <span className="text-[11px] text-muted-foreground/50">No metrics</span>}
      </div>
    </div>
  );
}

// ── Component ───────────────────────────────────────────────────────────

export function AdvantageCrossover() {
  const [entityType, setEntityType] = useState('driver');
  const [source, setSource] = useState(() => PREFERRED_SOURCE['driver'] ?? 'VectorProfiles');
  const [activeView, setActiveView] = useState<'matrix' | 'cluster' | 'insight' | 'compare'>('matrix');

  const [matrixData, setMatrixData] = useState<MatrixResult | null>(null);
  const [clusterData, setClusterData] = useState<ClusterResult | null>(null);
  const [insightData, setInsightData] = useState<InsightResult | null>(null);

  const [matrixLoading, setMatrixLoading] = useState(false);
  const [clusterLoading, setClusterLoading] = useState(false);
  const [insightLoading, setInsightLoading] = useState(false);
  const [error, setError] = useState('');
  const [nClusters, setNClusters] = useState(4);
  const [building, setBuilding] = useState(false);


  // Matrix enhancements
  const [matrixFilter, setMatrixFilter] = useState('');
  const [hoveredCell, setHoveredCell] = useState<{ row: number; col: number } | null>(null);
  const [minCorr, setMinCorr] = useState(0);
  const [triangleMode, setTriangleMode] = useState<'full' | 'upper' | 'lower'>('full');

  // Compare state
  const [availableEntities, setAvailableEntities] = useState<AvailableEntity[]>([]);
  const [selectedEntities, setSelectedEntities] = useState<string[]>([]);
  const [compareData, setCompareData] = useState<CompareResult | null>(null);
  const [compareLoading, setCompareLoading] = useState(false);
  const [crossInsight, setCrossInsight] = useState<CrossInsightResult | null>(null);
  const [crossInsightLoading, setCrossInsightLoading] = useState(false);
  const [crossQuery, setCrossQuery] = useState('');
  const [entitiesLoading, setEntitiesLoading] = useState(false);
  const [insightExpanded, setInsightExpanded] = useState(false);
  const crossInputRef = useRef<HTMLInputElement>(null);

  const buildProfiles = useCallback(async () => {
    setBuilding(true);
    setError('');
    try {
      const res = await fetch('/api/victory/build', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rebuild: true }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      const vecCount = data.vector?.count ?? 0;
      if (vecCount > 0) {
        setError('');
      } else {
        setError('Build completed but no profiles were generated');
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Build failed');
    } finally {
      setBuilding(false);
    }
  }, []);

  const fetchMatrix = useCallback(async () => {
    setMatrixLoading(true);
    setError('');
    try {
      const res = await fetch('/api/advantage/crossover/matrix', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ entity_type: entityType, source }),
      });
      if (!res.ok) {
        // Fallback to VectorProfiles if preferred source fails
        if (source !== 'VectorProfiles') {
          const fallback = await fetch('/api/advantage/crossover/matrix', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ entity_type: entityType, source: 'VectorProfiles' }),
          });
          if (fallback.ok) {
            setSource('VectorProfiles');
            setMatrixData(await fallback.json());
            return;
          }
        }
        throw new Error(await res.text());
      }
      setMatrixData(await res.json());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to compute matrix');
    } finally {
      setMatrixLoading(false);
    }
  }, [entityType, source]);

  const fetchCluster = useCallback(async () => {
    setClusterLoading(true);
    setError('');
    try {
      const res = await fetch('/api/advantage/crossover/cluster', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ entity_type: entityType, source, n_clusters: nClusters }),
      });
      if (!res.ok) {
        if (source !== 'VectorProfiles') {
          const fallback = await fetch('/api/advantage/crossover/cluster', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ entity_type: entityType, source: 'VectorProfiles', n_clusters: nClusters }),
          });
          if (fallback.ok) {
            setSource('VectorProfiles');
            setClusterData(await fallback.json());
            return;
          }
        }
        throw new Error(await res.text());
      }
      setClusterData(await res.json());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to compute clusters');
    } finally {
      setClusterLoading(false);
    }
  }, [entityType, source, nClusters]);

  const fetchInsight = useCallback(async () => {
    setInsightLoading(true);
    setError('');
    try {
      const res = await fetch('/api/advantage/crossover/insight', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ entity_type: entityType }),
      });
      if (!res.ok) throw new Error(await res.text());
      setInsightData(await res.json());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to generate insight');
    } finally {
      setInsightLoading(false);
    }
  }, [entityType]);

  const fetchAvailableEntities = useCallback(async () => {
    setEntitiesLoading(true);
    try {
      const res = await fetch(`/api/advantage/crossover/entities?entity_type=${entityType}&source=${source}`);
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setAvailableEntities(data.entities ?? []);
    } catch {
      setAvailableEntities([]);
    } finally {
      setEntitiesLoading(false);
    }
  }, [entityType, source]);

  const fetchCompare = useCallback(async () => {
    if (selectedEntities.length < 2) return;
    setCompareLoading(true);
    setError('');
    setCrossInsight(null);
    setCrossQuery('');
    try {
      const res = await fetch('/api/advantage/crossover/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ entity_type: entityType, entities: selectedEntities, source }),
      });
      if (!res.ok) throw new Error(await res.text());
      setCompareData(await res.json());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Comparison failed');
    } finally {
      setCompareLoading(false);
    }
  }, [entityType, selectedEntities, source]);

  const fetchCrossInsight = useCallback(async (query: string) => {
    if (!query.trim() || selectedEntities.length < 2) return;
    setCrossInsightLoading(true);
    setInsightExpanded(false);
    try {
      const res = await fetch('/api/advantage/crossover/cross_insights', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ entity_type: entityType, entities: selectedEntities, query: query.trim(), source }),
      });
      if (!res.ok) throw new Error(await res.text());
      setCrossInsight(await res.json());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Insight generation failed');
    } finally {
      setCrossInsightLoading(false);
    }
  }, [entityType, selectedEntities, source]);

  // Fetch entities when switching to compare tab
  useEffect(() => {
    if (activeView === 'compare' && availableEntities.length === 0 && !entitiesLoading) {
      fetchAvailableEntities();
    }
  }, [activeView, availableEntities.length, entitiesLoading, fetchAvailableEntities]);

  const toggleEntity = (code: string) => {
    setSelectedEntities(prev =>
      prev.includes(code)
        ? prev.filter(e => e !== code)
        : prev.length >= 8 ? prev : [...prev, code]
    );
  };

  const handleViewChange = (view: 'matrix' | 'cluster' | 'insight' | 'compare') => {
    setActiveView(view);
    if (view === 'matrix' && !matrixData && !matrixLoading) fetchMatrix();
    if (view === 'cluster' && !clusterData && !clusterLoading) fetchCluster();
  };

  // ── Matrix computed stats ──
  const matrixStats = useMemo(() => {
    if (!matrixData) return null;
    const { entities, matrix } = matrixData;
    let sum = 0, count = 0;
    let maxScore = -1, minScore = 2;
    let maxPair = { a: '', b: '' }, minPair = { a: '', b: '' };
    for (let i = 0; i < entities.length; i++) {
      for (let j = i + 1; j < entities.length; j++) {
        const s = matrix[i][j];
        if (s == null) continue;
        sum += s; count++;
        if (s > maxScore) { maxScore = s; maxPair = { a: entities[i], b: entities[j] }; }
        if (s < minScore) { minScore = s; minPair = { a: entities[i], b: entities[j] }; }
      }
    }
    return {
      avg: count > 0 ? sum / count : 0,
      max: maxScore, min: minScore,
      maxPair, minPair,
    };
  }, [matrixData]);

  // ── Filtered matrix indices ──
  const filteredIndices = useMemo(() => {
    if (!matrixData) return [];
    const filter = matrixFilter.toLowerCase().trim();
    if (!filter) return matrixData.entities.map((_, i) => i);
    return matrixData.entities
      .map((e, i) => ({ e, i }))
      .filter(({ e }) => e.toLowerCase().includes(filter))
      .map(({ i }) => i);
  }, [matrixData, matrixFilter]);

  return (
    <div className="pt-4 space-y-4">
      {/* Controls Bar */}
      <div className="flex items-center justify-between gap-4 flex-wrap">
        {/* Entity Type Tabs */}
        <div className="flex items-center gap-1 bg-card rounded-lg p-0.5">
          {ENTITY_TABS.map(({ value, label, icon: Icon, hint }) => (
            <button
              key={value}
              title={hint}
              onClick={() => {
                setEntityType(value);
                setMatrixData(null); setClusterData(null); setInsightData(null);
                setCompareData(null); setSelectedEntities([]); setAvailableEntities([]); setCrossInsight(null);
                setMatrixFilter(''); setMinCorr(0); setTriangleMode('full');
                setSource(PREFERRED_SOURCE[value] ?? 'VectorProfiles');
              }}
              className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm transition-all ${
                entityType === value
                  ? 'bg-primary/15 text-primary font-medium'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              {label}
              {matrixData && matrixData.entity_type === value && (
                <span className="text-[10px] font-mono bg-primary/10 px-1.5 py-0.5 rounded">{matrixData.count}</span>
              )}
            </button>
          ))}
        </div>

        {/* Source selector — always visible, grouped by category */}
        <select
          title="Data source for embeddings — Victory Profiles are curated multi-source profiles; Base Profiles use raw Vector Profiles"
          value={source}
          onChange={e => {
            setSource(e.target.value);
            setMatrixData(null); setClusterData(null); setCompareData(null);
            setSelectedEntities([]); setAvailableEntities([]); setCrossInsight(null);
            setError('');
          }}
          className="bg-card border border-border rounded-lg px-3 py-2 text-[11px] text-foreground focus:outline-none focus:border-primary/40 transition-colors"
        >
          {(() => {
            const available = SOURCE_OPTIONS.filter(o => o.entities.includes(entityType));
            const victory = available.filter(o => o.value.startsWith('victory_'));
            const base = available.filter(o => !o.value.startsWith('victory_'));
            return (
              <>
                {victory.length > 0 && (
                  <optgroup label="Victory Profiles">
                    {victory.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                  </optgroup>
                )}
                {base.length > 0 && (
                  <optgroup label="Base Profiles">
                    {base.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                  </optgroup>
                )}
              </>
            );
          })()}
        </select>
      </div>

      {/* View Tabs */}
      <div className="flex items-center gap-2">
        <button
          onClick={() => handleViewChange('matrix')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-[12px] tracking-wide transition-all border ${
            activeView === 'matrix'
              ? 'border-primary/30 bg-primary/8 text-primary'
              : 'border-transparent bg-card text-muted-foreground hover:text-foreground'
          }`}
        >
          <Grid3X3 className="w-3.5 h-3.5" />
          <span title="Pairwise cosine similarity heatmap — shows how alike each pair of entities is based on their embedding vectors">Similarity Matrix</span>
        </button>
        <button
          onClick={() => handleViewChange('cluster')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-[12px] tracking-wide transition-all border ${
            activeView === 'cluster'
              ? 'border-primary/30 bg-primary/8 text-primary'
              : 'border-transparent bg-card text-muted-foreground hover:text-foreground'
          }`}
        >
          <ScatterIcon className="w-3.5 h-3.5" />
          <span title="PCA dimensionality reduction + KMeans clustering — groups similar entities and shows what metrics differentiate the clusters">Cluster Map</span>
        </button>
        <button
          onClick={() => handleViewChange('insight')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-[12px] tracking-wide transition-all border ${
            activeView === 'insight'
              ? 'border-primary/30 bg-primary/8 text-primary'
              : 'border-transparent bg-card text-muted-foreground hover:text-foreground'
          }`}
        >
          <Brain className="w-3.5 h-3.5" />
          <span title="LLM-generated analysis of the most/least similar pairs, identifying behavioral patterns and strategic implications">AI Insights</span>
        </button>
        <button
          onClick={() => handleViewChange('compare')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-[12px] tracking-wide transition-all border ${
            activeView === 'compare'
              ? 'border-primary/30 bg-primary/8 text-primary'
              : 'border-transparent bg-card text-muted-foreground hover:text-foreground'
          }`}
        >
          <GitCompareArrows className="w-3.5 h-3.5" />
          <span title="Select 2-8 entities for direct pairwise comparison with metric breakdowns and cross-entity LLM intelligence">Compare</span>
        </button>
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/20 rounded-lg px-4 py-3 text-sm text-red-400 flex items-center justify-between">
          <span>{error}</span>
          {error.toLowerCase().includes('embedding') && (
            <button
              onClick={buildProfiles}
              disabled={building}
              className="ml-4 px-4 py-1.5 rounded-lg text-[12px] font-medium bg-primary text-[#0D1117] hover:bg-[#FF9A33] disabled:opacity-50 transition-all flex items-center gap-2"
            >
              {building ? <Loader2 className="w-3 h-3 animate-spin" /> : null}
              {building ? 'Building Profiles...' : 'Build VictoryProfiles'}
            </button>
          )}
        </div>
      )}

      {/* ── Matrix View ─────────────────────────────────── */}
      {activeView === 'matrix' && (
        <div className="space-y-4">
          {!matrixData && !matrixLoading && (
            <div className="flex flex-col items-center justify-center py-16">
              <button
                onClick={fetchMatrix}
                className="flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium bg-gradient-to-r from-primary to-[#FF9A33] text-[#0D1117] hover:shadow-[0_0_20px_rgba(255,128,0,0.3)] active:scale-[0.98] transition-all"
              >
                <Grid3X3 className="w-4 h-4" />
                Compute Similarity Matrix
              </button>
              <p className="text-[12px] text-muted-foreground/50 mt-3">
Computes cosine similarity between all {entityType} embedding vectors — scores range from 0 (completely different) to 1 (identical)
              </p>
            </div>
          )}

          {matrixLoading && (
            <div className="flex items-center justify-center py-16">
              <Loader2 className="w-6 h-6 animate-spin text-primary/50" />
              <span className="ml-3 text-sm text-muted-foreground">Computing similarity matrix...</span>
            </div>
          )}

          {matrixData && matrixStats && (
            <>
              {/* ── Summary Stats Bar ── */}
              <div className="grid grid-cols-3 gap-3">
                <div className="rounded-lg border border-border bg-card/30 p-3" title="Mean cosine similarity across all unique pairs — higher means entities are more alike overall">
                  <div className="text-[10px] text-muted-foreground tracking-widest uppercase mb-1">Avg Similarity</div>
                  <div className="text-xl font-bold font-mono text-foreground">{matrixStats.avg.toFixed(2)}</div>
                </div>
                <div className="rounded-lg border border-border bg-card/30 p-3" title="The pair with the highest cosine similarity — these entities share the most similar profile characteristics">
                  <div className="text-[10px] text-muted-foreground tracking-widest uppercase mb-1">Most Similar Pair</div>
                  <div className="flex items-baseline gap-2">
                    <span className="text-[12px] text-foreground font-medium">{matrixStats.maxPair.a}–{matrixStats.maxPair.b}</span>
                    <span className="text-xl font-bold font-mono text-success">{matrixStats.max.toFixed(2)}</span>
                  </div>
                </div>
                <div className="rounded-lg border border-border bg-card/30 p-3" title="The pair with the lowest cosine similarity — these entities have the most different profiles">
                  <div className="text-[10px] text-muted-foreground tracking-widest uppercase mb-1">Least Similar Pair</div>
                  <div className="flex items-baseline gap-2">
                    <span className="text-[12px] text-foreground font-medium">{matrixStats.minPair.a}–{matrixStats.minPair.b}</span>
                    <span className="text-xl font-bold font-mono text-danger">{matrixStats.min.toFixed(2)}</span>
                  </div>
                </div>
              </div>

              {/* ── Filter + Controls row ── */}
              <div className="flex items-center gap-3 flex-wrap">
                {/* Entity name filter */}
                <div className="relative flex-1 min-w-[140px] max-w-[200px]">
                  <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground/50" />
                  <input
                    type="text"
                    value={matrixFilter}
                    onChange={e => setMatrixFilter(e.target.value)}
                    placeholder="Filter entities..."
                    className="w-full bg-card border border-border rounded-lg pl-8 pr-3 py-1.5 text-[12px] text-foreground placeholder:text-muted-foreground/40 focus:outline-none focus:border-primary/40 transition-colors"
                  />
                </div>

                {/* Separator */}
                <div className="w-px h-6 bg-border" />

                {/* Correlation threshold */}
                <div className="flex items-center gap-2">
                  <SlidersHorizontal className="w-3.5 h-3.5 text-muted-foreground/60" />
                  <span className="text-[10px] text-muted-foreground whitespace-nowrap" title="Minimum cosine similarity threshold — cells below this value are dimmed to highlight strong relationships">Min Corr</span>
                  <input
                    type="range"
                    min={0}
                    max={0.99}
                    step={0.01}
                    value={minCorr}
                    onChange={e => setMinCorr(parseFloat(e.target.value))}
                    title="Minimum correlation threshold"
                    className="w-24 h-1.5 accent-primary cursor-pointer"
                  />
                  <span className="text-[11px] font-mono text-primary w-10 text-right">{minCorr > 0 ? minCorr.toFixed(2) : 'Off'}</span>
                </div>

                {/* Separator */}
                <div className="w-px h-6 bg-border" />

                {/* Triangle slice toggle */}
                <div className="flex items-center gap-1 bg-card rounded-lg p-0.5">
                  {(['full', 'upper', 'lower'] as const).map(mode => (
                    <button
                      type="button"
                      key={mode}
                      onClick={() => setTriangleMode(mode)}
                      className={`flex items-center gap-1 px-2 py-1 rounded-md text-[10px] tracking-wide transition-all ${
                        triangleMode === mode
                          ? 'bg-primary/15 text-primary font-medium'
                          : 'text-muted-foreground hover:text-foreground'
                      }`}
                      title={mode === 'full' ? 'Full matrix' : mode === 'upper' ? 'Upper triangle only' : 'Lower triangle only'}
                    >
                      {mode === 'full' ? (
                        <Grid3X3 className="w-3 h-3" />
                      ) : (
                        <TriangleRight className={`w-3 h-3 ${mode === 'lower' ? 'rotate-90' : ''}`} />
                      )}
                      {mode === 'full' ? 'Full' : mode === 'upper' ? 'Upper' : 'Lower'}
                    </button>
                  ))}
                </div>

                {/* Separator */}
                <div className="w-px h-6 bg-border" />

                {/* Color scale legend */}
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-muted-foreground font-mono">Low</span>
                  <div
                    className="w-24 h-2 rounded-full"
                    style={{ background: 'linear-gradient(to right, rgba(255,128,0,0.06), rgba(255,128,0,0.4), rgba(255,128,0,0.7), #FF8000)' }}
                  />
                  <span className="text-[10px] text-muted-foreground font-mono">High</span>
                </div>

                <button
                  onClick={fetchMatrix}
                  className="text-[11px] text-muted-foreground hover:text-foreground transition-colors ml-auto"
                >
                  Refresh
                </button>
              </div>

              {/* ── Matrix Table ── */}
              <div className="rounded-lg border border-border bg-card/30 overflow-hidden">
                <div className="overflow-auto max-h-[520px]">
                  <table className="w-full text-[11px] border-collapse">
                    <thead>
                      <tr>
                        <th className="sticky top-0 left-0 z-20 bg-card px-2 py-1.5 text-left text-muted-foreground font-medium" />
                        {filteredIndices.map(j => (
                          <th
                            key={j}
                            className={`sticky top-0 z-10 bg-card px-1.5 py-1.5 text-center font-medium whitespace-nowrap transition-colors ${
                              hoveredCell?.col === j ? 'text-primary' : 'text-muted-foreground'
                            }`}
                            style={{ writingMode: filteredIndices.length > 15 ? 'vertical-rl' : undefined, minWidth: filteredIndices.length > 15 ? '28px' : '48px' }}
                          >
                            {matrixData.entities[j]}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {filteredIndices.map(i => (
                        <tr key={i}>
                          <td className={`sticky left-0 z-10 bg-card px-2 py-1 font-medium whitespace-nowrap border-r border-border transition-colors ${
                            hoveredCell?.row === i ? 'text-primary' : 'text-foreground'
                          }`}>
                            {matrixData.entities[i]}
                          </td>
                          {filteredIndices.map(j => {
                            const score = matrixData.matrix[i][j];
                            const isDiag = i === j;
                            const isHoveredRow = hoveredCell?.row === i;
                            const isHoveredCol = hoveredCell?.col === j;
                            const isHoveredCell = isHoveredRow && isHoveredCol;
                            // Triangle slicing: hide the mirrored half
                            const isSliced = (triangleMode === 'upper' && i > j) || (triangleMode === 'lower' && i < j);
                            // Correlation threshold: dim cells below min
                            const isBelowThreshold = !isDiag && !isSliced && minCorr > 0 && score < minCorr;
                            return (
                              <td
                                key={j}
                                className={`px-1 py-1 text-center font-mono cursor-default transition-colors ${
                                  isHoveredCell ? 'ring-1 ring-primary ring-inset' : ''
                                }`}
                                style={{
                                  background: isDiag
                                    ? 'rgba(255,128,0,0.03)'
                                    : isSliced
                                      ? 'transparent'
                                      : isBelowThreshold
                                        ? 'rgba(255,128,0,0.02)'
                                        : (isHoveredRow || isHoveredCol)
                                          ? simColor(score)
                                          : simColor(score),
                                  color: isDiag
                                    ? 'rgba(144,144,168,0.3)'
                                    : isSliced
                                      ? 'transparent'
                                      : isBelowThreshold
                                        ? 'rgba(144,144,168,0.15)'
                                        : simText(score),
                                  outline: (isHoveredRow || isHoveredCol) && !isDiag && !isHoveredCell && !isSliced ? '1px solid rgba(255,128,0,0.08)' : undefined,
                                }}
                                title={isDiag ? matrixData.entities[i] : isSliced ? '' : `${matrixData.entities[i]} ↔ ${matrixData.entities[j]}: ${score?.toFixed(4)}`}
                                onMouseEnter={() => !isSliced && setHoveredCell({ row: i, col: j })}
                                onMouseLeave={() => setHoveredCell(null)}
                              >
                                {isDiag ? '—' : isSliced ? '' : score?.toFixed(2)}
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* ── Matrix Footer ── */}
                <div className="px-4 py-2 border-t border-border flex items-center justify-between text-[11px] text-muted-foreground/60">
                  <span>
                    {filteredIndices.length} × {filteredIndices.length} matrix · {filteredIndices.length} entities shown
                    {triangleMode !== 'full' && <> · {triangleMode} triangle</>}
                    {minCorr > 0 && <> · ≥{minCorr.toFixed(2)} corr</>}
                  </span>
                  <span className="font-mono">{matrixData.source}</span>
                </div>
              </div>
            </>
          )}
        </div>
      )}

      {/* ── Cluster View ────────────────────────────────── */}
      {activeView === 'cluster' && (
        <div>
          {!clusterData && !clusterLoading && (
            <div className="flex flex-col items-center justify-center py-16">
              <div className="flex items-center gap-3">
                <button
                  onClick={fetchCluster}
                  className="flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium bg-gradient-to-r from-primary to-[#FF9A33] text-[#0D1117] hover:shadow-[0_0_20px_rgba(255,128,0,0.3)] active:scale-[0.98] transition-all"
                >
                  <ScatterIcon className="w-4 h-4" />
                  Compute Clusters
                </button>
                <div className="flex items-center gap-2">
                  <span className="text-[11px] text-muted-foreground">Clusters:</span>
                  <input
                    type="number"
                    min={2}
                    max={8}
                    value={nClusters}
                    onChange={e => setNClusters(Math.max(2, Math.min(8, parseInt(e.target.value) || 4)))}
                    title="Number of clusters"
                    className="w-14 bg-card border border-border rounded px-2 py-1 text-sm text-foreground text-center focus:outline-none focus:border-primary/40"
                  />
                </div>
              </div>
              <p className="text-[12px] text-muted-foreground/50 mt-3">
PCA reduces profile dimensions for visualization, KMeans groups similar entities into clusters
              </p>
            </div>
          )}

          {clusterLoading && (
            <div className="flex items-center justify-center py-16">
              <Loader2 className="w-6 h-6 animate-spin text-primary/50" />
              <span className="ml-3 text-sm text-muted-foreground">Computing clusters...</span>
            </div>
          )}

          {clusterData && (
            <div className="space-y-4">
              {/* Scatter Plot */}
              <div className="rounded-lg border border-border bg-card/30 p-4">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-sm text-foreground font-medium cursor-help" title="Principal Component Analysis reduces high-dimensional embedding vectors into 2D/3D coordinates for visualization — nearby points share similar characteristics">PCA Projection</span>
                  <div className="flex items-center gap-3">
                    <div className="flex items-center gap-2 mr-2">
                      <span className="text-[11px] text-muted-foreground">Clusters:</span>
                      <input
                        type="number"
                        min={2}
                        max={8}
                        value={nClusters}
                        onChange={e => setNClusters(Math.max(2, Math.min(8, parseInt(e.target.value) || 4)))}
                        title="Number of clusters"
                        className="w-14 bg-card border border-border rounded px-2 py-1 text-sm text-foreground text-center focus:outline-none focus:border-primary/40"
                      />
                      <button
                        type="button"
                        onClick={fetchCluster}
                        className="text-[11px] text-muted-foreground hover:text-foreground transition-colors"
                      >
                        Refresh
                      </button>
                    </div>
                    {clusterData.clusters.map(c => (
                      <div key={c.id} className="flex items-center gap-1.5 text-[11px]">
                        <div className="w-2.5 h-2.5 rounded-full" style={{ background: CLUSTER_COLORS[c.id % CLUSTER_COLORS.length] }} />
                        <span className="text-muted-foreground">
                          C{c.id} ({c.members.length})
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
                <ResponsiveContainer width="100%" height={400}>
                  <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 20 }}>
                    <XAxis
                      type="number"
                      dataKey="x"
                      name="PC1"
                      tick={{ fill: '#9090A8', fontSize: 10 }}
                      axisLine={{ stroke: 'rgba(255,255,255,0.08)' }}
                      tickLine={false}
                      label={{ value: `PC1 (${(clusterData.explained_variance[0] * 100).toFixed(1)}%)`, position: 'bottom', fill: '#9090A8', fontSize: 11 }}
                    />
                    <YAxis
                      type="number"
                      dataKey="y"
                      name="PC2"
                      tick={{ fill: '#9090A8', fontSize: 10 }}
                      axisLine={{ stroke: 'rgba(255,255,255,0.08)' }}
                      tickLine={false}
                      label={{ value: `PC2 (${(clusterData.explained_variance[1] * 100).toFixed(1)}%)`, angle: -90, position: 'left', fill: '#9090A8', fontSize: 11 }}
                    />
                    <Tooltip
                      cursor={false}
                      content={({ active, payload }) => {
                        if (!active || !payload?.length) return null;
                        const d = payload[0].payload as ClusterEntity;
                        return (
                          <div className="bg-card border border-[rgba(255,128,0,0.2)] rounded-lg px-3 py-2 text-[12px]">
                            <div className="text-foreground font-medium">{d.code}</div>
                            <div className="text-muted-foreground">{d.team}</div>
                            <div className="text-muted-foreground/60 mt-1">Cluster {d.cluster}</div>
                          </div>
                        );
                      }}
                    />
                    <Scatter data={clusterData.entities} shape="circle">
                      {clusterData.entities.map((entry, idx) => (
                        <Cell
                          key={idx}
                          fill={CLUSTER_COLORS[entry.cluster % CLUSTER_COLORS.length]}
                          r={Math.max(5, Math.abs(entry.z) * 8 + 5)}
                          fillOpacity={0.75}
                          stroke={CLUSTER_COLORS[entry.cluster % CLUSTER_COLORS.length]}
                          strokeWidth={1}
                          strokeOpacity={0.3}
                        />
                      ))}
                    </Scatter>
                  </ScatterChart>
                </ResponsiveContainer>

                {/* Variance info */}
                <div className="flex items-center gap-4 mt-2 text-[11px] text-muted-foreground/50">
                  <span title="How much of the total data variation each principal component captures — higher % means that axis explains more of the differences between entities">Explained variance: {clusterData.explained_variance.map((v, i) => `PC${i + 1}: ${(v * 100).toFixed(1)}%`).join(', ')}</span>
                </div>
              </div>

              {/* Cluster Members + Avg Metrics */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
                {clusterData.clusters.map((c, ci) => {
                  const profile = clusterData.cluster_profiles?.[ci] || {};
                  const profileEntries = Object.entries(profile).slice(0, 6);
                  return (
                    <div
                      key={c.id}
                      className="rounded-lg border border-border bg-card/30 p-3"
                    >
                      <div className="flex items-center gap-2 mb-2">
                        <div
                          className="w-3 h-3 rounded-full"
                          style={{ background: CLUSTER_COLORS[c.id % CLUSTER_COLORS.length] }}
                        />
                        <span className="text-[12px] font-medium text-foreground">
                          Cluster {c.id}
                        </span>
                        <span className="text-[11px] text-muted-foreground ml-auto">
                          {c.members.length} members
                        </span>
                      </div>
                      <div className="flex flex-wrap gap-1.5 mb-2">
                        {c.members.map(m => (
                          <span
                            key={m}
                            className="text-[11px] px-2 py-0.5 rounded-md"
                            style={{
                              background: `${CLUSTER_COLORS[c.id % CLUSTER_COLORS.length]}15`,
                              color: CLUSTER_COLORS[c.id % CLUSTER_COLORS.length],
                            }}
                          >
                            {m}
                          </span>
                        ))}
                      </div>
                      {profileEntries.length > 0 && (
                        <div className="border-t border-border pt-2 mt-1 space-y-1">
                          {profileEntries.map(([k, v]) => (
                            <div key={k} className="flex items-center justify-between text-[10px]">
                              <span className="text-muted-foreground/70">{k.replace(/_/g, ' ')}</span>
                              <span className="text-foreground/80 font-mono">{typeof v === 'number' ? v.toFixed(1) : v}</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>

              {/* Discriminating Features — what separates these clusters */}
              {clusterData.discriminators && clusterData.discriminators.length > 0 && (
                <div className="rounded-lg border border-border bg-card/30 p-4">
                  <div className="text-sm font-medium text-foreground mb-3 cursor-help" title="Metrics with the highest variance between cluster averages — these are the characteristics that most differentiate the groups">
                    What Separates These Clusters
                  </div>
                  <div className="space-y-2">
                    {clusterData.discriminators.slice(0, 6).map(d => (
                      <div key={d.metric} className="flex items-center gap-3">
                        <span className="text-[12px] text-foreground/80 w-36 shrink-0">{d.metric.replace(/_/g, ' ')}</span>
                        <div className="flex items-center gap-2 flex-1">
                          {Object.entries(d.cluster_values).map(([cLabel, val]) => {
                            const cIdx = parseInt(cLabel.replace('C', ''));
                            return (
                              <div key={cLabel} className="flex items-center gap-1">
                                <div className="w-2 h-2 rounded-full" style={{ background: CLUSTER_COLORS[cIdx % CLUSTER_COLORS.length] }} />
                                <span className="text-[11px] font-mono text-foreground/70">{val}</span>
                              </div>
                            );
                          })}
                        </div>
                        <div className="w-16 bg-background/60 rounded-full h-1.5 overflow-hidden">
                          <div
                            className="h-full rounded-full bg-primary"
                            style={{ width: `${Math.min(100, d.spread * 100)}%` }}
                          />
                        </div>
                        <span className="text-[10px] text-muted-foreground/50 w-12 text-right font-mono">
                          {(d.spread * 100).toFixed(0)}%
                        </span>
                      </div>
                    ))}
                  </div>
                  <div className="text-[10px] text-muted-foreground/40 mt-2">
                    Spread = relative variance of cluster means — higher = more discriminating
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* ── Insight View ────────────────────────────────── */}
      {activeView === 'insight' && (
        <div>
          <div className="flex items-center gap-3 mb-4">
            <button
              onClick={fetchInsight}
              disabled={insightLoading}
              className="flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium transition-all disabled:opacity-40 bg-gradient-to-r from-primary to-[#FF9A33] text-[#0D1117] hover:shadow-[0_0_20px_rgba(255,128,0,0.3)] active:scale-[0.98]"
            >
              {insightLoading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Sparkles className="w-4 h-4" />
              )}
              {insightLoading ? 'Analyzing...' : 'Analyze Patterns'}
            </button>
          </div>

          {insightData && (
            <div className="space-y-4">
              {/* Pairs */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="rounded-lg border border-[rgba(5,223,114,0.15)] bg-[rgba(5,223,114,0.04)] p-4">
                  <div className="text-[12px] font-medium text-[#05DF72] mb-3 tracking-wide cursor-help" title="Entity pairs with the highest cosine similarity scores — their embedding vectors point in nearly the same direction, indicating shared characteristics">MOST SIMILAR</div>
                  <div className="space-y-2">
                    {insightData.pairs.most_similar.map((p, i) => (
                      <div key={i} className="flex items-center justify-between text-sm">
                        <span className="text-foreground">
                          {p.a} <span className="text-muted-foreground/40 mx-1">↔</span> {p.b}
                        </span>
                        <span className="font-mono text-[#05DF72] text-[12px]">{p.score.toFixed(4)}</span>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="rounded-lg border border-[rgba(251,44,54,0.15)] bg-[rgba(251,44,54,0.04)] p-4">
                  <div className="text-[12px] font-medium text-[#FB2C36] mb-3 tracking-wide cursor-help" title="Entity pairs with the lowest cosine similarity scores — their profiles diverge the most, suggesting fundamentally different characteristics">MOST DISSIMILAR</div>
                  <div className="space-y-2">
                    {insightData.pairs.most_dissimilar.map((p, i) => (
                      <div key={i} className="flex items-center justify-between text-sm">
                        <span className="text-foreground">
                          {p.a} <span className="text-muted-foreground/40 mx-1">↔</span> {p.b}
                        </span>
                        <span className="font-mono text-[#FB2C36] text-[12px]">{p.score.toFixed(4)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* LLM Insight */}
              <div className="rounded-lg border border-border bg-[rgba(255,128,0,0.04)] p-4">
                <div className="flex items-center gap-2 mb-3">
                  <Brain className="w-4 h-4 text-primary" />
                  <span className="text-sm font-medium text-primary cursor-help" title="LLM-synthesized analysis of embedding similarity patterns, identifying strategic and behavioral implications">Pattern Analysis</span>
                  <span className="text-[11px] text-muted-foreground ml-auto" title="The LLM model used to generate this analysis">{insightData.model_used}</span>
                </div>
                <div className="text-[13px] text-foreground/85 leading-relaxed whitespace-pre-line">
                  {insightData.insight}
                </div>
              </div>
            </div>
          )}

          {!insightData && !insightLoading && (
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <div className="w-14 h-14 rounded-lg bg-primary/8 flex items-center justify-center mb-4">
                <Sparkles className="w-6 h-6 text-primary/40" />
              </div>
              <p className="text-sm text-muted-foreground">
Click "Analyze Patterns" to generate LLM-synthesized insights about {entityType} similarity groupings, identifying the most/least similar pairs and their strategic implications
              </p>
            </div>
          )}
        </div>
      )}

      {/* ── Compare View ──────────────────────────────────── */}
      {activeView === 'compare' && (
        <div className="space-y-4">
          {/* Entity Selector */}
          <div className="rounded-lg border border-border bg-card/30 p-4">
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm font-medium text-foreground">
                Select {entityType}s to compare <span className="text-muted-foreground/50 text-[11px]">(2-8)</span>
              </span>
              {selectedEntities.length > 0 && (
                <button onClick={() => setSelectedEntities([])} className="text-[11px] text-muted-foreground hover:text-foreground transition-colors">
                  Clear all
                </button>
              )}
            </div>

            {entitiesLoading ? (
              <div className="flex items-center gap-2 py-4 justify-center">
                <Loader2 className="w-4 h-4 animate-spin text-primary/50" />
                <span className="text-[12px] text-muted-foreground">Loading entities...</span>
              </div>
            ) : (
              <div className="flex flex-wrap gap-2">
                {availableEntities.map(ent => {
                  const code = typeof ent === 'string' ? ent : ent.code;
                  const team = typeof ent === 'string' ? '' : ent.team;
                  const selected = selectedEntities.includes(code);
                  return (
                    <button
                      key={code}
                      onClick={() => toggleEntity(code)}
                      className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[12px] transition-all border ${
                        selected
                          ? 'border-primary/40 bg-primary/12 text-primary font-medium'
                          : 'border-border bg-background/40 text-muted-foreground hover:text-foreground hover:border-[rgba(255,128,0,0.15)]'
                      }`}
                    >
                      {code}
                      {team && <span className="text-[10px] text-muted-foreground/50">{team}</span>}
                      {selected && <X className="w-3 h-3 ml-0.5 opacity-60" />}
                    </button>
                  );
                })}
                {availableEntities.length === 0 && !entitiesLoading && (
                  <span className="text-[12px] text-muted-foreground/50">No entities found for this source</span>
                )}
              </div>
            )}

            {/* Selected chips summary + Compare button */}
            <div className="flex items-center justify-between mt-4 pt-3 border-t border-border">
              <span className="text-[11px] text-muted-foreground">
                {selectedEntities.length} selected{selectedEntities.length >= 8 && ' (max)'}
              </span>
              <button
                onClick={fetchCompare}
                disabled={selectedEntities.length < 2 || compareLoading}
                className="flex items-center gap-2 px-5 py-2 rounded-lg text-[12px] font-medium transition-all disabled:opacity-30 bg-gradient-to-r from-primary to-[#FF9A33] text-[#0D1117] hover:shadow-[0_0_20px_rgba(255,128,0,0.3)] active:scale-[0.98]"
              >
                {compareLoading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <ArrowRightLeft className="w-3.5 h-3.5" />}
                {compareLoading ? 'Comparing...' : 'Compare'}
              </button>
            </div>
          </div>

          {/* ── Two-Entity Layout ─────────────────────────── */}
          {compareData && selectedEntities.length === 2 && compareData.pairs.length === 1 && (
            <div className="grid grid-cols-[1fr_auto_1fr] gap-6 items-start">
              <EntityMetricCard code={compareData.pairs[0].a} metrics={compareData.pairs[0].a_metrics} />
              <div className="relative flex items-center justify-center pt-4">
                <CircularProgress value={compareData.pairs[0].similarity} />
              </div>
              <EntityMetricCard code={compareData.pairs[0].b} metrics={compareData.pairs[0].b_metrics} />
            </div>
          )}

          {/* ── Multi-Entity Layout (3+) ─────────────────── */}
          {compareData && selectedEntities.length > 2 && (
            <div className="space-y-4">
              {/* Stats bar */}
              <div className="grid grid-cols-3 gap-3">
                {[
                  { label: 'Average', value: compareData.statistics.avg, pair: null },
                  { label: 'Highest', value: compareData.statistics.max, pair: compareData.highest_pair },
                  { label: 'Lowest', value: compareData.statistics.min, pair: compareData.lowest_pair },
                ].map(stat => {
                  const { color } = simLabel(stat.value);
                  return (
                    <div key={stat.label} className="rounded-lg border border-border bg-card/30 p-3 text-center">
                      <div className="text-[11px] text-muted-foreground mb-1">{stat.label}</div>
                      <div className="text-xl font-bold font-mono" style={{ color }}>{(stat.value * 100).toFixed(1)}%</div>
                      {stat.pair && (
                        <div className="text-[10px] text-muted-foreground/60 mt-1">
                          {stat.pair.a} <span className="opacity-40">↔</span> {stat.pair.b}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>

              {/* Pairwise cards */}
              <div className="rounded-lg border border-border bg-card/30 p-4">
                <div className="text-sm font-medium text-foreground mb-3 cursor-help" title="Cosine similarity between each pair of selected entities — 100% means identical profiles, lower values indicate divergence">Pairwise Similarity</div>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2 max-h-[400px] overflow-y-auto">
                  {compareData.pairs.map((p, i) => {
                    const { color } = simLabel(p.similarity);
                    return (
                      <div key={i} className="rounded-lg border border-border bg-background/40 p-3 flex flex-col items-center gap-1.5">
                        <div className="flex items-center gap-2 text-[12px]">
                          <span className="text-foreground font-medium">{p.a}</span>
                          <span className="text-muted-foreground/30">↔</span>
                          <span className="text-foreground font-medium">{p.b}</span>
                        </div>
                        <span className="text-lg font-bold font-mono" style={{ color }}>
                          {(p.similarity * 100).toFixed(1)}%
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}

          {/* ── Cross-Entity Intelligence Panel ──────────── */}
          {compareData && (
            <div className="rounded-lg border border-border bg-[rgba(255,128,0,0.02)] p-4 space-y-4">
              <div className="flex items-center gap-2">
                <Brain className="w-4 h-4 text-primary" />
                <span className="text-sm font-medium text-primary cursor-help" title="Ask questions about the compared entities — the LLM uses their metrics, similarity scores, and metric correlations as context">Cross-Entity Intelligence</span>
              </div>

              {/* Suggested questions */}
              {compareData.suggested_questions && compareData.suggested_questions.length > 0 && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                  {compareData.suggested_questions.map((q, i) => (
                    <button
                      key={i}
                      onClick={() => { setCrossQuery(q); fetchCrossInsight(q); }}
                      disabled={crossInsightLoading}
                      className="text-left px-3 py-2 rounded-lg text-[11px] text-muted-foreground bg-background/40 border border-border hover:border-[rgba(255,128,0,0.2)] hover:text-foreground transition-all disabled:opacity-40"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              )}

              {/* Custom query input */}
              <div className="flex items-center gap-2">
                <input
                  ref={crossInputRef}
                  type="text"
                  value={crossQuery}
                  onChange={e => setCrossQuery(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter' && crossQuery.trim()) fetchCrossInsight(crossQuery); }}
                  placeholder={`Ask about ${selectedEntities.join(', ')}...`}
                  className="flex-1 bg-background/60 border border-border rounded-lg px-3 py-2 text-[12px] text-foreground placeholder:text-muted-foreground/40 focus:outline-none focus:border-primary/40 transition-colors"
                />
                <button
                  onClick={() => fetchCrossInsight(crossQuery)}
                  disabled={!crossQuery.trim() || crossInsightLoading}
                  className="flex items-center gap-1.5 px-4 py-2 rounded-lg text-[12px] font-medium transition-all disabled:opacity-30 bg-primary/15 text-primary hover:bg-primary/25"
                >
                  {crossInsightLoading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Send className="w-3.5 h-3.5" />}
                </button>
              </div>

              {/* Insight display — KeX briefing style */}
              {crossInsight && (
                <div className="bg-card border border-border rounded-lg p-4 space-y-3">
                  {/* Correlation pills */}
                  {crossInsight.correlations_found && crossInsight.correlations_found.length > 0 && (
                    <div className="flex flex-wrap gap-1.5">
                      {crossInsight.correlations_found.slice(0, 4).map((c, i) => (
                        <div key={i} className="text-[10px] font-mono px-2 py-1 rounded-md bg-[rgba(255,128,0,0.06)] border border-border">
                          <span className="text-foreground/70">{c.pair[0]} ↔ {c.pair[1]}</span>
                          {c.converging.length > 0 && (
                            <span className="text-[#05DF72]/80 ml-1.5" title={`Metrics within 10% of each other: ${c.converging.join(', ')}`}>+{c.converging.length} converging</span>
                          )}
                          {c.diverging.length > 0 && (
                            <span className="text-[#FB2C36]/80 ml-1.5" title={`Metrics differing by more than 50%: ${c.diverging.join(', ')}`}>-{c.diverging.length} diverging</span>
                          )}
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Summary preview (first ~200 chars) */}
                  {!insightExpanded && (
                    <p className="text-[11px] text-muted-foreground leading-relaxed">
                      {crossInsight.insight.length > 200
                        ? crossInsight.insight.slice(0, 200).trimEnd() + '…'
                        : crossInsight.insight}
                    </p>
                  )}

                  {/* Model + provider attribution */}
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-[10px] text-muted-foreground/50 font-mono" title="The LLM model used to generate this cross-entity intelligence">
                      via {crossInsight.model_used || 'auto'}
                    </span>
                  </div>

                  {/* View Full Briefing toggle */}
                  <button
                    onClick={() => setInsightExpanded(!insightExpanded)}
                    className="flex items-center gap-1.5 text-[11px] text-primary hover:text-[#FF9933] transition-colors w-full justify-center py-1.5 rounded-lg hover:bg-primary/5"
                  >
                    {insightExpanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                    {insightExpanded ? 'Hide Full Briefing' : 'View Full Briefing'}
                  </button>

                  {/* Full text (collapsible) */}
                  {insightExpanded && (
                    <div className="pt-1 border-t border-border">
                      <div className="text-[12px] text-muted-foreground leading-relaxed whitespace-pre-line">
                        {crossInsight.insight}
                      </div>
                      {/* Detailed correlations in expanded view */}
                      {crossInsight.correlations_found && crossInsight.correlations_found.length > 0 && (
                        <div className="border-t border-border pt-3 mt-3 space-y-2">
                          <div className="text-[11px] text-muted-foreground font-medium cursor-help" title="Converging metrics are within 10% of each other (similar values), diverging metrics differ by more than 50% (opposite behavior)">Metric Correlations</div>
                          {crossInsight.correlations_found.slice(0, 6).map((c, i) => (
                            <div key={i} className="text-[11px] space-y-0.5">
                              <span className="text-foreground/70">{c.pair[0]} ↔ {c.pair[1]}</span>
                              {c.converging.length > 0 && (
                                <div className="text-[#05DF72]/80 ml-3">Converging: {c.converging.join(', ')}</div>
                              )}
                              {c.diverging.length > 0 && (
                                <div className="text-[#FB2C36]/80 ml-3">Diverging: {c.diverging.join(', ')}</div>
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
