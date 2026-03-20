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
  ThumbsUp,
  ThumbsDown,
  Database,
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
  { key: 'key_insights' as const, label: 'Key Insights', icon: Lightbulb, accent: '#3B82F6', bg: 'rgba(59,130,246,0.06)', hint: 'High-level patterns synthesized from Knowledge Exchange (KeX) briefings and cross-entity data' },
  { key: 'recommendations' as const, label: 'Recommendations', icon: Target, accent: '#FF8000', bg: 'rgba(255,128,0,0.06)', hint: 'Actionable strategy suggestions derived from anomaly trends and forecast convergence' },
  { key: 'anomaly_patterns' as const, label: 'Anomaly Patterns', icon: AlertTriangle, accent: '#FB2C36', bg: 'rgba(251,44,54,0.06)', hint: 'Recurring deviations detected by the anomaly detection pipeline across car systems' },
  { key: 'forecast_signals' as const, label: 'Forecast Signals', icon: TrendingUp, accent: '#05DF72', bg: 'rgba(5,223,114,0.06)', hint: 'Forward-looking indicators from time-series forecasting models (ELT, degradation curves)' },
];

const SCOPE_OPTIONS = [
  { value: 'grid', label: 'Full Grid', icon: LayoutGrid, hint: 'Analyze all drivers and teams across the entire grid' },
  { value: 'driver', label: 'Driver', icon: Users, hint: 'Focus analysis on a single driver\'s performance profile' },
  { value: 'team', label: 'Team', icon: Shield, hint: 'Focus analysis on a single team\'s operational profile' },
  { value: 'database', label: 'Database', icon: Database, hint: 'Cross-collection synthesis across the entire intelligence platform' },
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
  const [feedbackSent, setFeedbackSent] = useState<Record<string, string>>({});

  // Fetch available entities when scope changes to driver/team
  useEffect(() => {
    if (scope === 'grid') { setAvailableEntities([]); return; }
    setEntitiesLoading(true);
    fetch(`/api/advantage/crossover/entities?entity_type=${scope}&source=VectorProfiles`)
      .then(r => { if (r.ok) return r.json(); throw new Error('fetch failed'); })
      .then(data => {
        const raw = data.entities || [];
        let ents: string[];
        if (scope === 'team') {
          // Extract unique team names from driver entities
          const teams = new Set<string>();
          raw.forEach((e: { code: string; team?: string } | string) => {
            const t = typeof e === 'string' ? '' : e.team || '';
            if (t) teams.add(t);
          });
          ents = [...teams].sort();
        } else {
          ents = raw.map((e: { code: string } | string) =>
            typeof e === 'string' ? e : e.code
          );
        }
        setAvailableEntities(ents);
      })
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

  // SSE auto-refresh: listen for trident:report:generated events
  useEffect(() => {
    let eventSource: EventSource | null = null;
    try {
      eventSource = new EventSource('/api/omni/agents/stream');
      eventSource.addEventListener('agent_event', (e) => {
        try {
          const data = JSON.parse(e.data);
          if (data.topic === 'trident:report:generated') {
            fetch(`/api/advantage/trident/latest?scope=${scope}${entity ? `&entity=${entity}` : ''}`)
              .then(r => { if (r.ok) return r.json(); throw new Error('fetch failed'); })
              .then(setReport)
              .catch(() => {});
          }
        } catch { /* ignore parse errors */ }
      });
    } catch { /* ignore SSE errors */ }
    return () => { eventSource?.close(); };
  }, [scope, entity]);

  const generate = async (force = false) => {
    setLoading(true);
    setError('');
    try {
      let res: Response;
      if (scope === 'database') {
        res = await fetch('/api/advantage/trident/database-synthesis', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ force }),
        });
      } else {
        res = await fetch('/api/advantage/trident/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            scope,
            entity: scope === 'grid' ? null : entity || null,
            force,
          }),
        });
      }
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
        <div className="flex items-center gap-1 bg-card rounded-lg p-0.5">
          <button
            onClick={() => setActiveTab('agent')}
            className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm transition-all ${
              activeTab === 'agent'
                ? 'bg-primary/15 text-primary font-medium'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            <Zap className="w-3.5 h-3.5" />
            <span title="Synthesizes KeX briefings, anomaly detection, and forecast data into a unified intelligence report via LLM">Convergence Agent</span>
          </button>
          <button
            onClick={() => setActiveTab('history')}
            className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm transition-all ${
              activeTab === 'history'
                ? 'bg-primary/15 text-primary font-medium'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            <History className="w-3.5 h-3.5" />
            <span title="Previously generated convergence reports, stored for comparison and trend tracking">Prompt History</span>
          </button>
        </div>

        {/* Scope Selector */}
        <div className="flex items-center gap-2">
          {SCOPE_OPTIONS.map(({ value, label, icon: Icon, hint }) => (
            <button
              key={value}
              onClick={() => { setScope(value); setEntity(''); }}
              title={hint}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[12px] tracking-wide transition-all border ${
                scope === value
                  ? 'border-primary/30 bg-primary/8 text-primary'
                  : 'border-transparent bg-card text-muted-foreground hover:text-foreground'
              }`}
            >
              <Icon className="w-3 h-3" />
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Entity Selector (when scoped to driver/team) */}
      {(scope === 'driver' || scope === 'team') && (
        <div className="space-y-2">
          <span className="text-[11px] text-muted-foreground/60 tracking-wide">
            Select {scope === 'driver' ? 'a driver' : 'a team'}
          </span>
          {entitiesLoading ? (
            <div className="flex items-center gap-2 py-2">
              <Loader2 className="w-3.5 h-3.5 animate-spin text-primary/50" />
              <span className="text-[11px] text-muted-foreground">Loading entities…</span>
            </div>
          ) : availableEntities.length === 0 ? (
            <p className="text-[11px] text-muted-foreground/50 py-2">No entities available</p>
          ) : (
            <div className="flex flex-wrap gap-2">
              {(() => {
                const MCLAREN_DRIVERS = ['NOR', 'PIA'];
                const sorted = [...availableEntities].sort((a, b) => {
                  const aM = MCLAREN_DRIVERS.includes(a) ? 0 : 1;
                  const bM = MCLAREN_DRIVERS.includes(b) ? 0 : 1;
                  return aM - bM || a.localeCompare(b);
                });
                return sorted.map(ent => {
                  const selected = entity === ent;
                  const isMcLaren = MCLAREN_DRIVERS.includes(ent);
                  return (
                    <button
                      key={ent}
                      onClick={() => setEntity(selected ? '' : ent)}
                      className={`px-3 py-1.5 rounded-lg text-[12px] tracking-wide transition-all border ${
                        selected
                          ? 'border-primary/40 bg-primary/12 text-primary font-medium'
                          : isMcLaren
                            ? 'border-primary/20 bg-primary/5 text-primary hover:bg-primary/10'
                            : 'border-border bg-card text-muted-foreground hover:text-foreground hover:border-[rgba(255,128,0,0.2)]'
                      }`}
                    >
                      {ent}
                    </button>
                  );
                });
              })()}
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
              disabled={loading || (scope !== 'grid' && scope !== 'database' && !entity)}
              className="flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium transition-all disabled:opacity-40 disabled:cursor-not-allowed bg-gradient-to-r from-primary to-[#FF9A33] text-[#0D1117] hover:shadow-[0_0_20px_rgba(255,128,0,0.3)] active:scale-[0.98]"
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
                className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-[12px] text-muted-foreground hover:text-foreground transition-colors bg-card hover:bg-secondary"
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
                  <span className="text-muted-foreground/50 ml-1" title="This report was served from cache rather than freshly generated">(cached)</span>
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
              {SECTION_CONFIG.map(({ key, label, icon: Icon, accent, bg, hint }) => {
                const section = report.sections[key];
                return (
                  <div
                    key={key}
                    className="rounded-lg border border-border overflow-hidden transition-all hover:border-[rgba(255,128,0,0.15)]"
                    style={{ background: bg }}
                  >
                    {/* Card Header */}
                    <div
                      className="flex items-center gap-2.5 px-4 py-3 border-b cursor-help"
                      style={{ borderColor: `${accent}20` }}
                      title={hint}
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

                    {/* Feedback */}
                    <div className="flex items-center gap-2 px-4 py-2 border-t" style={{ borderColor: `${accent}15` }}>
                      {feedbackSent[`${report.report_id}:${key}`] ? (
                        <span className="text-[11px] text-muted-foreground/60">
                          {feedbackSent[`${report.report_id}:${key}`] === 'up' ? 'Helpful' : 'Not helpful'} — thanks
                        </span>
                      ) : (
                        <>
                          <button
                            onClick={async () => {
                              await fetch('/api/advantage/trident/feedback', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ report_id: report.report_id, section: key, rating: 'up' }),
                              });
                              setFeedbackSent(prev => ({ ...prev, [`${report.report_id}:${key}`]: 'up' }));
                            }}
                            className="p-1.5 rounded hover:bg-green-500/10 transition-colors"
                            title="This section was helpful"
                          >
                            <ThumbsUp className="w-3 h-3 text-muted-foreground hover:text-green-400" />
                          </button>
                          <button
                            onClick={async () => {
                              await fetch('/api/advantage/trident/feedback', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ report_id: report.report_id, section: key, rating: 'down' }),
                              });
                              setFeedbackSent(prev => ({ ...prev, [`${report.report_id}:${key}`]: 'down' }));
                            }}
                            className="p-1.5 rounded hover:bg-red-500/10 transition-colors"
                            title="This section was not helpful"
                          >
                            <ThumbsDown className="w-3 h-3 text-muted-foreground hover:text-red-400" />
                          </button>
                        </>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {/* Metadata Footer */}
          {report && (
            <div className="flex items-center gap-4 text-[11px] text-muted-foreground/60 pt-1">
              <span title="The LLM used to synthesize this intelligence report">{report.metadata.model_used}</span>
              <span title="Time taken to gather data from all pipelines and generate the LLM synthesis">Generated in {report.metadata.generation_time_s}s</span>
              <span title="The analysis scope — grid covers all entities, driver/team focuses on one">Scope: {report.scope}{report.entity ? ` / ${report.entity}` : ''}</span>
            </div>
          )}

          {/* Empty State */}
          {!report && !loading && !error && (
            <div className="flex flex-col items-center justify-center py-16 text-center">
              <div className="w-16 h-16 rounded-lg bg-primary/8 flex items-center justify-center mb-4">
                <Layers className="w-7 h-7 text-primary/40" />
              </div>
              <p className="text-sm text-muted-foreground mb-1">No convergence report yet</p>
              <p className="text-[12px] text-muted-foreground/50">
                Click "Generate Report" to synthesize insights from Knowledge Exchange (KeX) briefings, anomaly detection, and forecast data
              </p>
            </div>
          )}
        </>
      ) : (
        /* ── History Tab ───────────────────────────────────── */
        <div className="space-y-2">
          {historyLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="w-5 h-5 animate-spin text-primary/50" />
            </div>
          ) : history.length === 0 ? (
            <div className="text-center py-12 text-sm text-muted-foreground/60">
              No reports in history for this scope
            </div>
          ) : (
            history.map(entry => (
              <div key={entry.report_id} className="rounded-lg border border-border bg-card/50 overflow-hidden">
                <button
                  onClick={() => loadHistoryReport(entry.report_id)}
                  className="w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-secondary/50 transition-colors"
                >
                  {expandedHistoryId === entry.report_id ? (
                    <ChevronDown className="w-3.5 h-3.5 text-primary shrink-0" />
                  ) : (
                    <ChevronRight className="w-3.5 h-3.5 text-muted-foreground shrink-0" />
                  )}
                  <span className="text-[11px] tracking-widest uppercase px-2 py-0.5 rounded bg-primary/10 text-primary">
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
                  <div className="px-4 pb-4 border-t border-border">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-3">
                      {SECTION_CONFIG.map(({ key, label, icon: Icon, accent }) => {
                        const section = expandedReport.sections[key];
                        return (
                          <div key={key} className="rounded-lg bg-background/60 p-3">
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
