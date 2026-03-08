import { useState, useEffect, useRef, useCallback } from 'react';
import { Activity, Play, Loader2, CheckCircle, XCircle, Radio, ChevronDown, ChevronUp } from 'lucide-react';

// ── Types ──────────────────────────────────────────────────────────────────

interface AgentStatus {
  name: string;
  description: string;
  status: 'idle' | 'running' | 'completed' | 'error';
  last_run: number | null;
  events_published: number;
  events_consumed: number;
  last_error: string | null;
}

interface AgentEvent {
  topic: string;
  agent: string;
  timestamp: number;
  severity: 'info' | 'low' | 'medium' | 'high' | 'critical';
  session_key?: number;
  driver_number?: number;
  payload: Record<string, unknown>;
}

interface AgentActivityProps {
  sessionKey?: number;
  driverNumber?: number;
  className?: string;
}

// ── Severity styling ───────────────────────────────────────────────────────

const SEVERITY_STYLES: Record<string, { dot: string; bg: string; text: string }> = {
  critical: { dot: 'bg-[#FB2C36]', bg: 'bg-[#FB2C36]/8', text: 'text-[#FCA5A5]' },
  high:     { dot: 'bg-primary',    bg: 'bg-primary/8',    text: 'text-primary' },
  medium:   { dot: 'bg-[#F59E0B]', bg: 'bg-[#F59E0B]/8', text: 'text-[#FCD34D]' },
  low:      { dot: 'bg-[#3B82F6]', bg: 'bg-[#3B82F6]/8', text: 'text-[#93C5FD]' },
  info:     { dot: 'bg-muted-foreground/40', bg: 'bg-secondary', text: 'text-muted-foreground' },
};

const STATUS_ICON: Record<string, React.ReactNode> = {
  idle:      <div className="w-2 h-2 rounded-full bg-muted-foreground/30" />,
  running:   <Loader2 className="w-3 h-3 text-primary animate-spin" />,
  completed: <CheckCircle className="w-3 h-3 text-[#05DF72]" />,
  error:     <XCircle className="w-3 h-3 text-[#FB2C36]" />,
};

const AGENT_LABELS: Record<string, string> = {
  telemetry_anomaly: 'Telemetry Anomaly',
  weather_adapt: 'Weather Adapt',
  pit_window: 'Pit Window',
  predictive_maintenance: 'Predictive Maintenance',
  knowledge_convergence: 'Knowledge Convergence',
  visual_inspect: 'Visual Inspection',
};

// ── Component ──────────────────────────────────────────────────────────────

export function AgentActivity({ sessionKey, driverNumber, className = '' }: AgentActivityProps) {
  const [agents, setAgents] = useState<AgentStatus[]>([]);
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [expandedEvent, setExpandedEvent] = useState<number | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const eventsEndRef = useRef<HTMLDivElement>(null);

  // Fetch agent status
  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch('/api/omni/agents/status');
      if (res.ok) {
        const data = await res.json();
        setAgents(data.agents || []);
      }
    } catch { /* ignore */ }
  }, []);

  // Connect SSE stream
  useEffect(() => {
    const es = new EventSource('/api/omni/agents/stream');
    eventSourceRef.current = es;

    es.addEventListener('connected', () => setIsConnected(true));

    es.addEventListener('agent_event', (e) => {
      try {
        const event = JSON.parse(e.data) as AgentEvent;
        setEvents(prev => [event, ...prev].slice(0, 100));
        // Refresh status when we get status events
        if (event.topic?.includes(':status')) {
          fetchStatus();
        }
      } catch { /* ignore parse errors */ }
    });

    es.addEventListener('keepalive', () => { /* connection alive */ });

    es.onerror = () => {
      setIsConnected(false);
      // Reconnect after 3s
      setTimeout(() => {
        if (eventSourceRef.current === es) {
          es.close();
          // Will reconnect on next render
        }
      }, 3000);
    };

    fetchStatus();

    return () => {
      es.close();
      eventSourceRef.current = null;
    };
  }, [fetchStatus]);

  // Auto-scroll events
  useEffect(() => {
    eventsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [events.length]);

  // Run all agents
  const handleRunAll = async () => {
    if (!sessionKey) return;
    setIsRunning(true);
    setEvents([]);

    try {
      await fetch('/api/omni/agents/run-all', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_key: sessionKey,
          driver_number: driverNumber,
        }),
      });
      fetchStatus();
    } catch (err) {
      console.error('Failed to start agents:', err);
    }

    // Reset running state after timeout (agents run async)
    setTimeout(() => setIsRunning(false), 2000);
  };

  const anyRunning = agents.some(a => a.status === 'running');

  return (
    <div className={`flex flex-col rounded-lg border border-border bg-card overflow-hidden ${className}`}>

      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border bg-secondary/30">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-primary" />
          <span className="text-[11px] font-semibold tracking-wide uppercase text-foreground">
            Agent Activity
          </span>
          <div className={`w-1.5 h-1.5 rounded-full ${isConnected ? 'bg-[#05DF72]' : 'bg-muted-foreground/30'}`} />
        </div>

        <button
          onClick={handleRunAll}
          disabled={!sessionKey || isRunning || anyRunning}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[10px] font-semibold tracking-wide uppercase transition-all cursor-pointer
            ${isRunning || anyRunning
              ? 'bg-primary/10 text-primary/50 cursor-not-allowed'
              : 'bg-primary/15 text-primary hover:bg-primary hover:text-background border border-primary/20'
            }`}
        >
          {isRunning || anyRunning
            ? <><Loader2 className="w-3 h-3 animate-spin" /> Running</>
            : <><Play className="w-3 h-3" /> Run Analysis</>
          }
        </button>
      </div>

      {/* Agent Status Row */}
      <div className="flex items-center gap-3 px-4 py-2.5 border-b border-border bg-secondary/15">
        {agents.map(agent => (
          <div
            key={agent.name}
            className="flex items-center gap-1.5"
            title={`${AGENT_LABELS[agent.name] || agent.name}: ${agent.status}`}
          >
            {STATUS_ICON[agent.status] || STATUS_ICON.idle}
            <span className="text-[10px] text-muted-foreground font-mono">
              {(AGENT_LABELS[agent.name] || agent.name).split(' ')[0]}
            </span>
          </div>
        ))}
        {agents.length === 0 && (
          <span className="text-[10px] text-muted-foreground/50 font-mono">No agents loaded</span>
        )}
      </div>

      {/* Event Feed */}
      <div className="flex-1 overflow-y-auto max-h-[400px] min-h-[120px]">
        {events.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-8 text-muted-foreground/40">
            <Radio className="w-5 h-5 mb-2" />
            <span className="text-[10px] font-mono tracking-wide uppercase">
              {sessionKey ? 'Ready — click Run Analysis' : 'Select a session to begin'}
            </span>
          </div>
        ) : (
          <div className="divide-y divide-border">
            {events.map((event, i) => {
              const style = SEVERITY_STYLES[event.severity] || SEVERITY_STYLES.info;
              const isExpanded = expandedEvent === i;
              const summaryText = (event.payload?.summary as string) || event.topic;

              return (
                <div key={`${event.timestamp}-${i}`} className={`px-4 py-2.5 ${style.bg} transition-colors`}>
                  <div
                    className="flex items-start gap-2 cursor-pointer"
                    onClick={() => setExpandedEvent(isExpanded ? null : i)}
                  >
                    <div className={`w-1.5 h-1.5 rounded-full mt-1.5 flex-shrink-0 ${style.dot}`} />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-[10px] font-mono text-muted-foreground/60">
                          {AGENT_LABELS[event.agent] || event.agent}
                        </span>
                        <span className={`text-[9px] font-mono uppercase tracking-wider ${style.text}`}>
                          {event.severity !== 'info' ? event.severity : ''}
                        </span>
                      </div>
                      <p className={`text-[11px] leading-relaxed mt-0.5 ${style.text} line-clamp-2`}>
                        {summaryText}
                      </p>
                    </div>
                    {isExpanded
                      ? <ChevronUp className="w-3 h-3 text-muted-foreground/40 flex-shrink-0 mt-1" />
                      : <ChevronDown className="w-3 h-3 text-muted-foreground/40 flex-shrink-0 mt-1" />
                    }
                  </div>

                  {isExpanded && (
                    <div className="mt-2 ml-3.5 p-2 rounded bg-background/50 border border-border">
                      <pre className="text-[10px] font-mono text-muted-foreground overflow-x-auto whitespace-pre-wrap max-h-[200px] overflow-y-auto">
                        {JSON.stringify(event.payload, null, 2)}
                      </pre>
                    </div>
                  )}
                </div>
              );
            })}
            <div ref={eventsEndRef} />
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between px-4 py-2 border-t border-border bg-secondary/15">
        <span className="text-[10px] text-muted-foreground/40 font-mono">
          {events.length} events
        </span>
        <span className="text-[10px] text-muted-foreground/40 font-mono">
          {sessionKey ? `Session ${sessionKey}` : 'No session'}
        </span>
      </div>
    </div>
  );
}
