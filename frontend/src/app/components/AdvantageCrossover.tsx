import { useState, useCallback } from 'react';
import {
  Users,
  Shield,
  Box,
  Loader2,
  Sparkles,
  Grid3X3,
  ScatterChart as ScatterIcon,
  Brain,
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

// ── Constants ───────────────────────────────────────────────────────────

const ENTITY_TABS = [
  { value: 'driver', label: 'Driver', icon: Users },
  { value: 'team', label: 'Team', icon: Shield },
  { value: 'car', label: 'Car', icon: Box },
];

const SOURCE_OPTIONS = [
  { value: 'VectorProfiles', label: 'Vector Profiles' },
  { value: 'victory_driver_profiles', label: 'Victory Driver' },
  { value: 'victory_car_profiles', label: 'Victory Car' },
  { value: 'victory_team_kb', label: 'Victory Team' },
];

const CLUSTER_COLORS = ['#FF8000', '#3B82F6', '#05DF72', '#FB2C36', '#A855F7', '#F59E0B'];

// ── Helpers ─────────────────────────────────────────────────────────────

function simColor(score: number): string {
  // Low → dim gray, High → bright orange
  if (score >= 0.95) return '#FF8000';
  if (score >= 0.85) return 'rgba(255,128,0,0.7)';
  if (score >= 0.75) return 'rgba(255,128,0,0.4)';
  if (score >= 0.60) return 'rgba(255,128,0,0.2)';
  return 'rgba(255,128,0,0.06)';
}

function simText(score: number): string {
  if (score >= 0.95) return '#FF8000';
  if (score >= 0.80) return '#FF8000';
  if (score >= 0.65) return 'rgba(255,128,0,0.7)';
  return 'rgba(144,144,168,0.6)';
}

// ── Component ───────────────────────────────────────────────────────────

export function AdvantageCrossover() {
  const [entityType, setEntityType] = useState('driver');
  const [source, setSource] = useState('VectorProfiles');
  const [activeView, setActiveView] = useState<'matrix' | 'cluster' | 'insight'>('matrix');

  const [matrixData, setMatrixData] = useState<MatrixResult | null>(null);
  const [clusterData, setClusterData] = useState<ClusterResult | null>(null);
  const [insightData, setInsightData] = useState<InsightResult | null>(null);

  const [matrixLoading, setMatrixLoading] = useState(false);
  const [clusterLoading, setClusterLoading] = useState(false);
  const [insightLoading, setInsightLoading] = useState(false);
  const [error, setError] = useState('');
  const [nClusters, setNClusters] = useState(4);
  const [building, setBuilding] = useState(false);

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
      if (!res.ok) throw new Error(await res.text());
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
      if (!res.ok) throw new Error(await res.text());
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

  const handleViewChange = (view: 'matrix' | 'cluster' | 'insight') => {
    setActiveView(view);
    if (view === 'matrix' && !matrixData && !matrixLoading) fetchMatrix();
    if (view === 'cluster' && !clusterData && !clusterLoading) fetchCluster();
  };

  return (
    <div className="pt-4 space-y-4">
      {/* Controls Bar */}
      <div className="flex items-center justify-between gap-4 flex-wrap">
        {/* Entity Type Tabs */}
        <div className="flex items-center gap-1 bg-[#1A1F2E] rounded-lg p-0.5">
          {ENTITY_TABS.map(({ value, label, icon: Icon }) => (
            <button
              key={value}
              onClick={() => { setEntityType(value); setMatrixData(null); setClusterData(null); setInsightData(null); }}
              className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm transition-all ${
                entityType === value
                  ? 'bg-[#FF8000]/15 text-[#FF8000] font-medium'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              {label}
            </button>
          ))}
        </div>

        {/* Source Selector */}
        <select
          value={source}
          onChange={e => { setSource(e.target.value); setMatrixData(null); setClusterData(null); }}
          className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:border-[#FF8000]/40 transition-colors"
        >
          {SOURCE_OPTIONS.map(o => (
            <option key={o.value} value={o.value}>{o.label}</option>
          ))}
        </select>
      </div>

      {/* View Tabs */}
      <div className="flex items-center gap-2">
        <button
          onClick={() => handleViewChange('matrix')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-[12px] tracking-wide transition-all border ${
            activeView === 'matrix'
              ? 'border-[#FF8000]/30 bg-[#FF8000]/8 text-[#FF8000]'
              : 'border-transparent bg-[#1A1F2E] text-muted-foreground hover:text-foreground'
          }`}
        >
          <Grid3X3 className="w-3.5 h-3.5" />
          Similarity Matrix
        </button>
        <button
          onClick={() => handleViewChange('cluster')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-[12px] tracking-wide transition-all border ${
            activeView === 'cluster'
              ? 'border-[#FF8000]/30 bg-[#FF8000]/8 text-[#FF8000]'
              : 'border-transparent bg-[#1A1F2E] text-muted-foreground hover:text-foreground'
          }`}
        >
          <ScatterIcon className="w-3.5 h-3.5" />
          Cluster Map
        </button>
        <button
          onClick={() => handleViewChange('insight')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-[12px] tracking-wide transition-all border ${
            activeView === 'insight'
              ? 'border-[#FF8000]/30 bg-[#FF8000]/8 text-[#FF8000]'
              : 'border-transparent bg-[#1A1F2E] text-muted-foreground hover:text-foreground'
          }`}
        >
          <Brain className="w-3.5 h-3.5" />
          AI Insights
        </button>
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/20 rounded-lg px-4 py-3 text-sm text-red-400 flex items-center justify-between">
          <span>{error}</span>
          {error.toLowerCase().includes('embedding') && (
            <button
              onClick={buildProfiles}
              disabled={building}
              className="ml-4 px-4 py-1.5 rounded-lg text-[12px] font-medium bg-[#FF8000] text-[#0D1117] hover:bg-[#FF9A33] disabled:opacity-50 transition-all flex items-center gap-2"
            >
              {building ? <Loader2 className="w-3 h-3 animate-spin" /> : null}
              {building ? 'Building Profiles...' : 'Build VictoryProfiles'}
            </button>
          )}
        </div>
      )}

      {/* ── Matrix View ─────────────────────────────────── */}
      {activeView === 'matrix' && (
        <div>
          {!matrixData && !matrixLoading && (
            <div className="flex flex-col items-center justify-center py-16">
              <button
                onClick={fetchMatrix}
                className="flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium bg-gradient-to-r from-[#FF8000] to-[#FF9A33] text-[#0D1117] hover:shadow-[0_0_20px_rgba(255,128,0,0.3)] active:scale-[0.98] transition-all"
              >
                <Grid3X3 className="w-4 h-4" />
                Compute Similarity Matrix
              </button>
              <p className="text-[12px] text-muted-foreground/50 mt-3">
                Pairwise cosine similarity across all {entityType}s
              </p>
            </div>
          )}

          {matrixLoading && (
            <div className="flex items-center justify-center py-16">
              <Loader2 className="w-6 h-6 animate-spin text-[#FF8000]/50" />
              <span className="ml-3 text-sm text-muted-foreground">Computing similarity matrix...</span>
            </div>
          )}

          {matrixData && (
            <div className="rounded-xl border border-[rgba(255,128,0,0.08)] bg-[#1A1F2E]/30 overflow-hidden">
              <div className="px-4 py-3 border-b border-[rgba(255,128,0,0.08)] flex items-center justify-between">
                <span className="text-sm text-foreground font-medium">
                  {matrixData.count} {entityType}s — {matrixData.source}
                </span>
                <button
                  onClick={fetchMatrix}
                  className="text-[11px] text-muted-foreground hover:text-foreground transition-colors"
                >
                  Refresh
                </button>
              </div>
              <div className="overflow-auto max-h-[500px]">
                <table className="w-full text-[11px]">
                  <thead>
                    <tr>
                      <th className="sticky top-0 left-0 z-20 bg-[#1A1F2E] px-2 py-1.5 text-left text-muted-foreground font-medium" />
                      {matrixData.entities.map(e => (
                        <th
                          key={e}
                          className="sticky top-0 z-10 bg-[#1A1F2E] px-1.5 py-1.5 text-center text-muted-foreground font-medium whitespace-nowrap"
                          style={{ writingMode: matrixData.count > 15 ? 'vertical-rl' : undefined, minWidth: matrixData.count > 15 ? '28px' : '48px' }}
                        >
                          {e}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {matrixData.entities.map((rowEntity, i) => (
                      <tr key={rowEntity}>
                        <td className="sticky left-0 z-10 bg-[#1A1F2E] px-2 py-1 text-foreground font-medium whitespace-nowrap border-r border-[rgba(255,128,0,0.06)]">
                          {rowEntity}
                        </td>
                        {matrixData.matrix[i].map((score, j) => (
                          <td
                            key={j}
                            className="px-1 py-1 text-center font-mono cursor-default transition-colors"
                            style={{
                              background: i === j ? 'rgba(255,128,0,0.03)' : simColor(score),
                              color: i === j ? 'rgba(144,144,168,0.3)' : simText(score),
                            }}
                            title={`${rowEntity} ↔ ${matrixData.entities[j]}: ${score?.toFixed(4)}`}
                          >
                            {i === j ? '—' : score?.toFixed(2)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
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
                  className="flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium bg-gradient-to-r from-[#FF8000] to-[#FF9A33] text-[#0D1117] hover:shadow-[0_0_20px_rgba(255,128,0,0.3)] active:scale-[0.98] transition-all"
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
                    className="w-14 bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded px-2 py-1 text-sm text-foreground text-center focus:outline-none focus:border-[#FF8000]/40"
                  />
                </div>
              </div>
              <p className="text-[12px] text-muted-foreground/50 mt-3">
                PCA 3D projection + KMeans clustering
              </p>
            </div>
          )}

          {clusterLoading && (
            <div className="flex items-center justify-center py-16">
              <Loader2 className="w-6 h-6 animate-spin text-[#FF8000]/50" />
              <span className="ml-3 text-sm text-muted-foreground">Computing clusters...</span>
            </div>
          )}

          {clusterData && (
            <div className="space-y-4">
              {/* Scatter Plot */}
              <div className="rounded-xl border border-[rgba(255,128,0,0.08)] bg-[#1A1F2E]/30 p-4">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-sm text-foreground font-medium">PCA Projection</span>
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
                        className="w-14 bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded px-2 py-1 text-sm text-foreground text-center focus:outline-none focus:border-[#FF8000]/40"
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
                      axisLine={{ stroke: 'rgba(255,128,0,0.1)' }}
                      tickLine={false}
                      label={{ value: `PC1 (${(clusterData.explained_variance[0] * 100).toFixed(1)}%)`, position: 'bottom', fill: '#9090A8', fontSize: 11 }}
                    />
                    <YAxis
                      type="number"
                      dataKey="y"
                      name="PC2"
                      tick={{ fill: '#9090A8', fontSize: 10 }}
                      axisLine={{ stroke: 'rgba(255,128,0,0.1)' }}
                      tickLine={false}
                      label={{ value: `PC2 (${(clusterData.explained_variance[1] * 100).toFixed(1)}%)`, angle: -90, position: 'left', fill: '#9090A8', fontSize: 11 }}
                    />
                    <Tooltip
                      cursor={false}
                      content={({ active, payload }) => {
                        if (!active || !payload?.length) return null;
                        const d = payload[0].payload as ClusterEntity;
                        return (
                          <div className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.2)] rounded-lg px-3 py-2 text-[12px]">
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
                  <span>Explained variance: {clusterData.explained_variance.map((v, i) => `PC${i + 1}: ${(v * 100).toFixed(1)}%`).join(', ')}</span>
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
                      className="rounded-lg border border-[rgba(255,128,0,0.08)] bg-[#1A1F2E]/30 p-3"
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
                        <div className="border-t border-[rgba(255,128,0,0.06)] pt-2 mt-1 space-y-1">
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
                <div className="rounded-xl border border-[rgba(255,128,0,0.08)] bg-[#1A1F2E]/30 p-4">
                  <div className="text-sm font-medium text-foreground mb-3">
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
                        <div className="w-16 bg-[#0D1117]/60 rounded-full h-1.5 overflow-hidden">
                          <div
                            className="h-full rounded-full bg-[#FF8000]"
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
              className="flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium transition-all disabled:opacity-40 bg-gradient-to-r from-[#FF8000] to-[#FF9A33] text-[#0D1117] hover:shadow-[0_0_20px_rgba(255,128,0,0.3)] active:scale-[0.98]"
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
                  <div className="text-[12px] font-medium text-[#05DF72] mb-3 tracking-wide">MOST SIMILAR</div>
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
                  <div className="text-[12px] font-medium text-[#FB2C36] mb-3 tracking-wide">MOST DISSIMILAR</div>
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
              <div className="rounded-xl border border-[rgba(255,128,0,0.12)] bg-[rgba(255,128,0,0.04)] p-4">
                <div className="flex items-center gap-2 mb-3">
                  <Brain className="w-4 h-4 text-[#FF8000]" />
                  <span className="text-sm font-medium text-[#FF8000]">Pattern Analysis</span>
                  <span className="text-[11px] text-muted-foreground ml-auto">{insightData.model_used}</span>
                </div>
                <div className="text-[13px] text-foreground/85 leading-relaxed whitespace-pre-line">
                  {insightData.insight}
                </div>
              </div>
            </div>
          )}

          {!insightData && !insightLoading && (
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <div className="w-14 h-14 rounded-2xl bg-[#FF8000]/8 flex items-center justify-center mb-4">
                <Sparkles className="w-6 h-6 text-[#FF8000]/40" />
              </div>
              <p className="text-sm text-muted-foreground">
                Click "Analyze Patterns" to get LLM-synthesized insights about {entityType} similarity groupings
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
