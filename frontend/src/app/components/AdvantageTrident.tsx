import { useState, useEffect, useMemo } from 'react';
import {
  Layers,
  RefreshCw,
  Clock,
  Lightbulb,
  AlertTriangle,
  TrendingUp,
  Target,
  ChevronDown,
  ChevronRight,
  Loader2,
  Zap,
  History,
  Users,
  Shield,
  LayoutGrid,
} from 'lucide-react';

// ── Types ───────────────────────────────────────────────────────────────

interface TridentSection {
  title: string;
  content: string;
  source_count?: number;
}

interface TridentReport {
  report_id: string;
  scope: string;
  entity: string | null;
  generated_at: number;
  stale_after: number;
  from_cache?: boolean;
  sections: {
    key_insights: TridentSection;
    recommendations: TridentSection;
    anomaly_patterns: TridentSection;
    forecast_signals: TridentSection;
  };
  metadata: {
    model_used: string;
    generation_time_s: number;
  };
}

interface HistoryEntry {
  report_id: string;
  scope: string;
  entity: string | null;
  generated_at: number;
  metadata: { model_used: string; generation_time_s: number };
}

// ── Constants ───────────────────────────────────────────────────────────

const SECTION_CONFIG = [
  { key: 'key_insights' as const, label: 'Key Insights', icon: Lightbulb, accent: '#3B82F6', bg: 'rgba(59,130,246,0.06)' },
  { key: 'recommendations' as const, label: 'Recommendations', icon: Target, accent: '#FF8000', bg: 'rgba(255,128,0,0.06)' },
  { key: 'anomaly_patterns' as const, label: 'Anomaly Patterns', icon: AlertTriangle, accent: '#FB2C36', bg: 'rgba(251,44,54,0.06)' },
  { key: 'forecast_signals' as const, label: 'Forecast Signals', icon: TrendingUp, accent: '#05DF72', bg: 'rgba(5,223,114,0.06)' },
];

const SCOPE_OPTIONS = [
  { value: 'grid', label: 'Full Grid', icon: LayoutGrid },
  { value: 'driver', label: 'Driver', icon: Users },
  { value: 'team', label: 'Team', icon: Shield },
];

// ── Component ───────────────────────────────────────────────────────────

export function AdvantageTrident() {
  const [activeTab, setActiveTab] = useState<'agent' | 'history'>('agent');
  const [scope, setScope] = useState('grid');
  const [entity, setEntity] = useState('');
  const [report, setReport] = useState<TridentReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [expandedHistoryId, setExpandedHistoryId] = useState<string | null>(null);
  const [expandedReport, setExpandedReport] = useState<TridentReport | null>(null);
  const [availableEntities, setAvailableEntities] = useState<string[]>([]);
  const [entitiesLoading, setEntitiesLoading] = useState(false);

  // Fetch available entities when scope changes to driver/team
  useEffect(() => {
    if (scope === 'grid') { setAvailableEntities([]); return; }
    setEntitiesLoading(true);
    fetch(`/api/advantage/crossover/entities?entity_type=${scope}&source=VectorProfiles`)
      .then(r => { if (r.ok) return r.json(); throw new Error('fetch failed'); })
      .then(data => setAvailableEntities(data.entities || []))
      .catch(() => setAvailableEntities([]))
      .finally(() => setEntitiesLoading(false));
  }, [scope]);

  // Load latest report on mount
  useEffect(() => {
    fetch(`/api/advantage/trident/latest?scope=grid`)
      .then(r => { if (r.ok) return r.json(); throw new Error('none'); })
      .then(setReport)
      .catch(err => { console.error('Failed to load latest report:', err); });
  }, []);

  const generate = async (force = false) => {
    setLoading(true);
    setError('');
    try {
      const res = await fetch('/api/advantage/trident/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scope,
          entity: scope === 'grid' ? null : entity || null,
          force,
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setReport(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Generation failed');
    } finally {
      setLoading(false);
    }
  };

  const loadHistory = async () => {
    setHistoryLoading(true);
    try {
      const res = await fetch(`/api/advantage/trident/history?scope=${scope}&limit=20`);
      if (res.ok) setHistory(await res.json());
    } catch (err) { console.error('Failed to load history:', err); }
    setHistoryLoading(false);
  };

  useEffect(() => {
    if (activeTab === 'history') loadHistory();
  }, [activeTab, scope]);

  const loadHistoryReport = async (id: string) => {
    if (expandedHistoryId === id) {
      setExpandedHistoryId(null);
      return;
    }
    setExpandedHistoryId(id);
    try {
      const res = await fetch(`/api/advantage/trident/report/${id}`);
      if (res.ok) setExpandedReport(await res.json());
    } catch (err) { console.error('Failed to load report:', err); }
  };

  const staleness = useMemo(() => {
    if (!report) return null;
    const now = Date.now() / 1000;
    const age = now - report.generated_at;
    if (age < 60) return { label: 'Just now', fresh: true };
    if (age < 3600) return { label: `${Math.floor(age / 60)}m ago`, fresh: age < 1800 };
    return { label: `${Math.floor(age / 3600)}h ago`, fresh: false };
  }, [report]);

  return (
    <div className="pt-4 space-y-4">
      {/* Tab Bar + Controls */}
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-1 bg-[#1A1F2E] rounded-lg p-0.5">
          <button
            onClick={() => setActiveTab('agent')}
            className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm transition-all ${
              activeTab === 'agent'
                ? 'bg-[#FF8000]/15 text-[#FF8000] font-medium'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            <Zap className="w-3.5 h-3.5" />
            Convergence Agent
          </button>
          <button
            onClick={() => setActiveTab('history')}
            className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm transition-all ${
              activeTab === 'history'
                ? 'bg-[#FF8000]/15 text-[#FF8000] font-medium'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            <History className="w-3.5 h-3.5" />
            Prompt History
          </button>
        </div>

        {/* Scope Selector */}
        <div className="flex items-center gap-2">
          {SCOPE_OPTIONS.map(({ value, label, icon: Icon }) => (
            <button
              key={value}
              onClick={() => { setScope(value); setEntity(''); }}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[12px] tracking-wide transition-all border ${
                scope === value
                  ? 'border-[#FF8000]/30 bg-[#FF8000]/8 text-[#FF8000]'
                  : 'border-transparent bg-[#1A1F2E] text-muted-foreground hover:text-foreground'
              }`}
            >
              <Icon className="w-3 h-3" />
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Entity Selector (when scoped to driver/team) */}
      {scope !== 'grid' && (
        <div className="space-y-2">
          <span className="text-[11px] text-muted-foreground/60 tracking-wide">
            Select {scope === 'driver' ? 'a driver' : 'a team'}
          </span>
          {entitiesLoading ? (
            <div className="flex items-center gap-2 py-2">
              <Loader2 className="w-3.5 h-3.5 animate-spin text-[#FF8000]/50" />
              <span className="text-[11px] text-muted-foreground">Loading entities…</span>
            </div>
          ) : availableEntities.length === 0 ? (
            <p className="text-[11px] text-muted-foreground/50 py-2">No entities available</p>
          ) : (
            <div className="flex flex-wrap gap-2">
              {availableEntities.map(ent => {
                const selected = entity === ent;
                return (
                  <button
                    key={ent}
                    onClick={() => setEntity(selected ? '' : ent)}
                    className={`px-3 py-1.5 rounded-lg text-[12px] tracking-wide transition-all border ${
                      selected
                        ? 'border-[#FF8000]/40 bg-[#FF8000]/12 text-[#FF8000] font-medium'
                        : 'border-[rgba(255,128,0,0.08)] bg-[#1A1F2E] text-muted-foreground hover:text-foreground hover:border-[rgba(255,128,0,0.2)]'
                    }`}
                  >
                    {ent}
                  </button>
                );
              })}
            </div>
          )}
        </div>
      )}

      {activeTab === 'agent' ? (
        <>
          {/* Generate Button + Status */}
          <div className="flex items-center gap-3">
            <button
              onClick={() => generate(false)}
              disabled={loading || (scope !== 'grid' && !entity)}
              className="flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium transition-all disabled:opacity-40 disabled:cursor-not-allowed bg-gradient-to-r from-[#FF8000] to-[#FF9A33] text-[#0D1117] hover:shadow-[0_0_20px_rgba(255,128,0,0.3)] active:scale-[0.98]"
            >
              {loading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Layers className="w-4 h-4" />
              )}
              {loading ? 'Synthesizing...' : 'Generate Report'}
            </button>

            {report && !loading && (
              <button
                onClick={() => generate(true)}
                className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-[12px] text-muted-foreground hover:text-foreground transition-colors bg-[#1A1F2E] hover:bg-[#222838]"
              >
                <RefreshCw className="w-3 h-3" />
                Force Refresh
              </button>
            )}

            {staleness && (
              <div className="flex items-center gap-1.5 text-[11px]">
                <Clock className="w-3 h-3" />
                <span className={staleness.fresh ? 'text-green-400' : 'text-muted-foreground'}>
                  {staleness.label}
                </span>
                {report?.from_cache && (
                  <span className="text-muted-foreground/50 ml-1">(cached)</span>
                )}
              </div>
            )}
          </div>

          {error && (
            <div className="bg-red-500/10 border border-red-500/20 rounded-lg px-4 py-3 text-sm text-red-400">
              {error}
            </div>
          )}

          {/* Report Cards — 2x2 Grid */}
          {report && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {SECTION_CONFIG.map(({ key, label, icon: Icon, accent, bg }) => {
                const section = report.sections[key];
                return (
                  <div
                    key={key}
                    className="rounded-xl border border-[rgba(255,128,0,0.08)] overflow-hidden transition-all hover:border-[rgba(255,128,0,0.15)]"
                    style={{ background: bg }}
                  >
                    {/* Card Header */}
                    <div
                      className="flex items-center gap-2.5 px-4 py-3 border-b"
                      style={{ borderColor: `${accent}20` }}
                    >
                      <div
                        className="w-7 h-7 rounded-lg flex items-center justify-center"
                        style={{ background: `${accent}18` }}
                      >
                        <Icon className="w-3.5 h-3.5" style={{ color: accent }} />
                      </div>
                      <span className="text-sm font-medium tracking-wide" style={{ color: accent }}>
                        {label}
                      </span>
                    </div>

                    {/* Card Content */}
                    <div className="px-4 py-3 max-h-[300px] overflow-y-auto">
                      <div className="text-[13px] text-foreground/85 leading-relaxed whitespace-pre-line">
                        {section?.content || 'No data available'}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {/* Metadata Footer */}
          {report && (
            <div className="flex items-center gap-4 text-[11px] text-muted-foreground/60 pt-1">
              <span>Model: {report.metadata.model_used}</span>
              <span>Generated in {report.metadata.generation_time_s}s</span>
              <span>Scope: {report.scope}{report.entity ? ` / ${report.entity}` : ''}</span>
            </div>
          )}

          {/* Empty State */}
          {!report && !loading && !error && (
            <div className="flex flex-col items-center justify-center py-16 text-center">
              <div className="w-16 h-16 rounded-2xl bg-[#FF8000]/8 flex items-center justify-center mb-4">
                <Layers className="w-7 h-7 text-[#FF8000]/40" />
              </div>
              <p className="text-sm text-muted-foreground mb-1">No convergence report yet</p>
              <p className="text-[12px] text-muted-foreground/50">
                Click "Generate Report" to synthesize insights from KeX, anomaly, and forecast data
              </p>
            </div>
          )}
        </>
      ) : (
        /* ── History Tab ───────────────────────────────────── */
        <div className="space-y-2">
          {historyLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="w-5 h-5 animate-spin text-[#FF8000]/50" />
            </div>
          ) : history.length === 0 ? (
            <div className="text-center py-12 text-sm text-muted-foreground/60">
              No reports in history for this scope
            </div>
          ) : (
            history.map(entry => (
              <div key={entry.report_id} className="rounded-lg border border-[rgba(255,128,0,0.08)] bg-[#1A1F2E]/50 overflow-hidden">
                <button
                  onClick={() => loadHistoryReport(entry.report_id)}
                  className="w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-[#222838]/50 transition-colors"
                >
                  {expandedHistoryId === entry.report_id ? (
                    <ChevronDown className="w-3.5 h-3.5 text-[#FF8000] shrink-0" />
                  ) : (
                    <ChevronRight className="w-3.5 h-3.5 text-muted-foreground shrink-0" />
                  )}
                  <span className="text-[11px] tracking-widest uppercase px-2 py-0.5 rounded bg-[#FF8000]/10 text-[#FF8000]">
                    {entry.scope}
                  </span>
                  {entry.entity && (
                    <span className="text-sm text-foreground font-medium">{entry.entity}</span>
                  )}
                  <span className="ml-auto text-[11px] text-muted-foreground font-mono">
                    {new Date(entry.generated_at * 1000).toLocaleString('en-GB', {
                      day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit',
                    })}
                  </span>
                  <span className="text-[10px] text-muted-foreground/50">
                    {entry.metadata.generation_time_s}s
                  </span>
                </button>

                {expandedHistoryId === entry.report_id && expandedReport && (
                  <div className="px-4 pb-4 border-t border-[rgba(255,128,0,0.06)]">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-3">
                      {SECTION_CONFIG.map(({ key, label, icon: Icon, accent }) => {
                        const section = expandedReport.sections[key];
                        return (
                          <div key={key} className="rounded-lg bg-[#0D1117]/60 p-3">
                            <div className="flex items-center gap-2 mb-2">
                              <Icon className="w-3 h-3" style={{ color: accent }} />
                              <span className="text-[12px] font-medium" style={{ color: accent }}>{label}</span>
                            </div>
                            <div className="text-[12px] text-foreground/75 leading-relaxed whitespace-pre-line max-h-[200px] overflow-y-auto">
                              {section?.content || 'N/A'}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}
