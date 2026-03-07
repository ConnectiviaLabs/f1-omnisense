import { useRef, useEffect, useState, useCallback } from 'react';
import {
  Send, Bot, User, Loader2, Sparkles, RotateCcw,
  TrendingUp, TrendingDown, Minus, Info, AlertTriangle, XCircle,
  RefreshCw,
} from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine,
  BarChart, Bar, Cell,
} from 'recharts';

/* ── Lightweight UIMessage type (replaces @ai-sdk/react dependency) ── */
interface UIMessagePart { type: string; text?: string; [k: string]: any }
interface UIMessage { id: string; role: 'user' | 'assistant'; parts: UIMessagePart[] }

/* ── Simple chat hook that talks to our /api/chat endpoint ────────── */
function useSimpleChat() {
  const [messages, setMessages] = useState<UIMessage[]>([]);
  const [status, setStatus] = useState<'ready' | 'submitted' | 'streaming'>('ready');
  const [error, setError] = useState<Error | null>(null);

  const clearError = useCallback(() => setError(null), []);

  const sendMessage = useCallback(async ({ text }: { text: string }) => {
    const userMsg: UIMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      parts: [{ type: 'text', text }],
    };
    setMessages(prev => [...prev, userMsg]);
    setStatus('submitted');
    setError(null);

    try {
      // Try Gen UI endpoint first (Vite dev middleware handles streaming + tools)
      const genRes = await fetch('/api/fleet/diagnose', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          id: crypto.randomUUID(),
          messages: [...messages, userMsg].map(m => ({
            id: m.id, role: m.role, parts: m.parts,
          })),
        }),
      });

      if (genRes.ok) {
        // Gen UI available — parse AI SDK data stream (SSE format)
        setStatus('streaming');
        const reader = genRes.body?.getReader();
        const decoder = new TextDecoder();
        let textParts: string[] = [];
        let toolParts: UIMessagePart[] = [];
        if (reader) {
          let buffer = '';
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() ?? '';
            for (const line of lines) {
              const trimmed = line.trim();
              // AI SDK v4 SSE format: "data: {...}"
              if (trimmed.startsWith('data: ')) {
                const payload = trimmed.slice(6);
                if (payload === '[DONE]') continue;
                try {
                  const evt = JSON.parse(payload);
                  if (evt.type === 'tool-input-available') {
                    toolParts.push({
                      type: `tool-${evt.toolName}`,
                      toolName: evt.toolName,
                      input: evt.input ?? {},
                      state: 'result',
                    });
                  } else if (evt.type === 'text-delta') {
                    textParts.push(evt.textDelta ?? '');
                  }
                } catch { /* skip non-JSON lines */ }
              }
              // AI SDK v3 format fallback: "0:..." text, "9:..." tool results
              else if (trimmed.startsWith('0:')) {
                try { textParts.push(JSON.parse(trimmed.slice(2))); } catch { /* skip */ }
              } else if (trimmed.startsWith('9:')) {
                try {
                  const arr = JSON.parse(trimmed.slice(2));
                  for (const tc of arr) {
                    toolParts.push({
                      type: `tool-${tc.toolName}`,
                      toolName: tc.toolName,
                      input: tc.args,
                      output: tc.result,
                      state: 'result',
                    });
                  }
                } catch { /* skip */ }
              }
            }
          }
        }
        const parts: UIMessagePart[] = [
          ...toolParts,
          ...(textParts.length ? [{ type: 'text', text: textParts.join('') }] : []),
        ];
        if (parts.length) {
          setMessages(prev => [...prev, {
            id: crypto.randomUUID(), role: 'assistant', parts,
          }]);
        }
        setStatus('ready');
        return;
      }

      // Fallback: plain /api/chat (Python backend — no Gen UI)
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      });
      if (!res.ok) {
        const body = await res.text();
        throw new Error(body || `Chat failed (${res.status})`);
      }
      const data = await res.json();
      setMessages(prev => [...prev, {
        id: crypto.randomUUID(), role: 'assistant',
        parts: [{ type: 'text', text: data.answer ?? data.text ?? '' }],
      }]);
    } catch (e: any) {
      setError(e instanceof Error ? e : new Error(String(e)));
    } finally {
      setStatus('ready');
    }
  }, [messages]);

  return { messages, setMessages, sendMessage, status, error, clearError };
}

/* ── Suggestion prompts (diagnostic-focused) ─────────────────────── */
const SUGGESTIONS = [
  "How is Norris's car health trending this season?",
  'Compare brake system health across the last 5 races',
  'What are the biggest reliability risks for the McLaren fleet?',
  'Show me engine performance trends for Piastri',
  'Which systems need maintenance attention before the next race?',
  'Analyze tyre management across the McLaren cars',
];

/* ── Severity / trend color maps ─────────────────────────────────── */
const SEV_COLORS: Record<string, string> = {
  nominal: '#22c55e', warning: '#f59e0b', critical: '#ef4444', info: '#3b82f6',
};
const SEV_BG: Record<string, string> = {
  nominal: 'rgba(34,197,94,0.08)', warning: 'rgba(245,158,11,0.08)',
  critical: 'rgba(239,68,68,0.08)', info: 'rgba(59,130,246,0.08)',
};
const TREND_ICON: Record<string, React.ReactNode> = {
  up: <TrendingUp className="w-3.5 h-3.5" />,
  down: <TrendingDown className="w-3.5 h-3.5" />,
  stable: <Minus className="w-3.5 h-3.5" />,
};
const SEV_ICON: Record<string, React.ReactNode> = {
  info: <Info className="w-4 h-4" />,
  warning: <AlertTriangle className="w-4 h-4" />,
  critical: <XCircle className="w-4 h-4" />,
};

/* ── Fade-in animation wrapper ───────────────────────────────────── */
function FadeIn({ children, className = '' }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={`animate-[fadeSlideIn_0.3s_ease-out] ${className}`}>
      {children}
      <style>{`
        @keyframes fadeSlideIn {
          from { opacity: 0; transform: translateY(6px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   GenUI Widget Renderers
   ═══════════════════════════════════════════════════════════════════ */

function MetricCard({ title, value, trend, severity, subtitle }: {
  title: string; value: string; trend: string; severity: string; subtitle?: string;
}) {
  const color = SEV_COLORS[severity] ?? '#666';
  return (
    <FadeIn>
      <div
        className="rounded-lg px-3 py-2.5 border transition-all hover:scale-[1.02]"
        style={{
          borderColor: color,
          borderLeftWidth: 3,
          background: SEV_BG[severity] ?? 'rgba(255,255,255,0.03)',
        }}
      >
        <div className="flex items-center justify-between mb-1">
          <span className="text-[10px] uppercase tracking-wider text-muted-foreground">{title}</span>
          <span style={{ color }}>{TREND_ICON[trend]}</span>
        </div>
        <div className="text-lg font-semibold text-foreground">{value}</div>
        {subtitle && <div className="text-[10px] text-muted-foreground mt-0.5">{subtitle}</div>}
      </div>
    </FadeIn>
  );
}

function SparklineChart({ title, data, unit, thresholds }: {
  title: string;
  data: { race: string; value: number }[];
  unit: string;
  thresholds?: { warning: number; critical: number };
}) {
  // Shorten race names for x-axis
  const shortData = data.map(d => ({
    ...d,
    short: d.race.replace(/ Grand Prix$/i, '').replace(/^20\d{2} /, ''),
  }));
  return (
    <FadeIn>
      <div className="rounded-lg bg-[#0D1117] border border-[rgba(255,128,0,0.12)] p-3">
        <div className="text-[10px] uppercase tracking-wider text-muted-foreground mb-2">{title}</div>
        <ResponsiveContainer width="100%" height={150}>
          <LineChart data={shortData} margin={{ top: 4, right: 8, bottom: 0, left: -20 }}>
            <XAxis dataKey="short" tick={{ fontSize: 8, fill: '#666' }} interval={0} angle={-35} textAnchor="end" height={50} />
            <YAxis tick={{ fontSize: 9, fill: '#666' }} domain={['dataMin - 5', 'dataMax + 5']} />
            <Tooltip
              contentStyle={{ background: '#1A1F2E', border: '1px solid rgba(255,128,0,0.2)', borderRadius: 8, fontSize: 11 }}
              formatter={(v: number) => [`${v} ${unit}`, '']}
              labelFormatter={(label) => shortData.find(d => d.short === label)?.race ?? label}
            />
            <Line type="monotone" dataKey="value" stroke="#FF8000" strokeWidth={2} dot={{ r: 3, fill: '#FF8000', strokeWidth: 0 }} activeDot={{ r: 5, fill: '#FF8000' }} />
            {thresholds && (
              <>
                <ReferenceLine y={thresholds.warning} stroke="#f59e0b" strokeDasharray="4 4" label={{ value: 'Warning', fontSize: 8, fill: '#f59e0b', position: 'right' }} />
                <ReferenceLine y={thresholds.critical} stroke="#ef4444" strokeDasharray="4 4" label={{ value: 'Critical', fontSize: 8, fill: '#ef4444', position: 'right' }} />
              </>
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </FadeIn>
  );
}

function ComparisonBar({ title, items }: {
  title: string; items: { label: string; value: number; max: number }[];
}) {
  const chartData = items.map(it => ({ name: it.label, value: it.value, max: it.max }));
  return (
    <FadeIn>
      <div className="rounded-lg bg-[#0D1117] border border-[rgba(255,128,0,0.12)] p-3">
        <div className="text-[10px] uppercase tracking-wider text-muted-foreground mb-2">{title}</div>
        <ResponsiveContainer width="100%" height={items.length * 36 + 20}>
          <BarChart data={chartData} layout="vertical" margin={{ top: 0, right: 8, bottom: 0, left: 60 }}>
            <XAxis type="number" tick={{ fontSize: 9, fill: '#666' }} />
            <YAxis type="category" dataKey="name" tick={{ fontSize: 10, fill: '#999' }} width={55} />
            <Tooltip
              contentStyle={{ background: '#1A1F2E', border: '1px solid rgba(255,128,0,0.2)', borderRadius: 8, fontSize: 11 }}
            />
            <Bar dataKey="value" radius={[0, 4, 4, 0]}>
              {chartData.map((entry, i) => {
                const pct = entry.max > 0 ? entry.value / entry.max : 0;
                const fill = pct > 0.7 ? '#22c55e' : pct > 0.4 ? '#f59e0b' : '#ef4444';
                return <Cell key={i} fill={fill} />;
              })}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </FadeIn>
  );
}

function Recommendation({ title, description, severity, action }: {
  title: string; description: string; severity: string; action: string;
}) {
  const color = SEV_COLORS[severity] ?? '#666';
  return (
    <FadeIn>
      <div
        className="rounded-lg px-3 py-2.5 border"
        style={{
          borderColor: color,
          borderLeftWidth: 3,
          background: SEV_BG[severity] ?? 'rgba(255,255,255,0.03)',
        }}
      >
        <div className="flex items-center gap-2 mb-1">
          <span style={{ color }}>{SEV_ICON[severity] ?? SEV_ICON.info}</span>
          <span className="text-xs font-medium text-foreground">{title}</span>
        </div>
        <p className="text-[11px] text-muted-foreground leading-relaxed mb-1.5">{description}</p>
        <span
          className="inline-block text-[10px] px-2 py-0.5 rounded-full font-medium"
          style={{ background: color + '22', color }}
        >
          {action}
        </span>
      </div>
    </FadeIn>
  );
}

function AnalysisText({ content }: { content: string }) {
  return (
    <FadeIn>
      <div className="border-l-2 border-[#FF8000]/40 pl-3 py-1">
        <p className="text-[11px] text-muted-foreground leading-relaxed">{content}</p>
      </div>
    </FadeIn>
  );
}

/* ── Similar Drivers widget ────────────────────────────────────── */
function SimilarDriversWidget({ driver_code, similar }: { driver_code: string; similar: { driver_code: string; team: string; score: number }[] }) {
  if (!similar?.length) return null;
  return (
    <FadeIn>
      <div className="rounded-lg border border-[rgba(255,128,0,0.15)] bg-[#0D1117] p-3">
        <div className="text-[10px] font-semibold tracking-wider text-[#FF8000]/70 mb-2">
          SIMILAR TO {driver_code}
        </div>
        <div className="space-y-1.5">
          {similar.map((s, i) => {
            const pct = Math.round(s.score * 100);
            return (
              <div key={s.driver_code} className="flex items-center gap-2 text-[11px]">
                <span className="text-muted-foreground w-3 font-mono">{i + 1}</span>
                <span className="font-semibold text-foreground w-8">{s.driver_code}</span>
                <span className="text-muted-foreground flex-1 truncate">{s.team || '—'}</span>
                <div className="w-16 h-1 bg-[#222838] rounded-full overflow-hidden">
                  <div className="h-full rounded-full bg-[#FF8000]" style={{ width: `${pct}%` }} />
                </div>
                <span className="font-mono text-[#FF8000] w-8 text-right">{pct}%</span>
              </div>
            );
          })}
        </div>
      </div>
    </FadeIn>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   Tool Part Renderer — maps tool name → widget
   ═══════════════════════════════════════════════════════════════════ */

function renderToolPart(part: any) {
  // Skip tools still streaming input
  if (part.state === 'input-streaming') return null;

  const name = part.type?.startsWith('tool-')
    ? part.type.slice(5)
    : part.toolName;
  const args = part.input ?? {};

  switch (name) {
    case 'metric_card':
      return <MetricCard {...args} />;
    case 'sparkline':
      return <SparklineChart {...args} />;
    case 'comparison':
      return <ComparisonBar {...args} />;
    case 'recommendation':
      return <Recommendation {...args} />;
    case 'text':
      return <AnalysisText {...args} />;
    case 'similar_drivers':
      return <SimilarDriversWidget {...(part.output ?? args)} />;
    default:
      return null;
  }
}

/* ═══════════════════════════════════════════════════════════════════
   Message Renderers
   ═══════════════════════════════════════════════════════════════════ */

function AssistantMessage({ message }: { message: UIMessage }) {
  const parts = message.parts ?? [];
  const hasToolParts = parts.some(p => p.type.startsWith('tool-') || p.type === 'dynamic-tool');

  // Collect metric_card parts to render as a grid
  const metricCards: any[] = [];
  const otherParts: any[] = [];

  parts.forEach((part, i) => {
    const name = part.type?.startsWith('tool-') ? part.type.slice(5) : (part as any).toolName;
    if (name === 'metric_card') {
      metricCards.push({ part, key: i });
    } else {
      otherParts.push({ part, key: i });
    }
  });

  return (
    <div className="flex items-start gap-3">
      <div className="w-7 h-7 rounded-lg bg-[#FF8000]/10 flex items-center justify-center shrink-0 mt-0.5">
        <Bot className="w-4 h-4 text-[#FF8000]" />
      </div>
      <div className={`space-y-2 ${hasToolParts ? 'w-full max-w-full' : 'max-w-[75%]'}`}>
        {/* Metric cards in a responsive grid */}
        {metricCards.length > 0 && (
          <div className={`grid gap-2 ${metricCards.length >= 3 ? 'grid-cols-3' : 'grid-cols-2'}`}>
            {metricCards.map(({ part, key }) => (
              <div key={key}>{renderToolPart(part)}</div>
            ))}
          </div>
        )}

        {/* Other parts sequentially */}
        {otherParts.map(({ part, key }) => {
          if (part.type === 'text') {
            if (!(part as any).text?.trim()) return null;
            return (
              <FadeIn key={key}>
                <div className="rounded-xl px-4 py-3 text-[11px] leading-relaxed whitespace-pre-wrap bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] text-foreground">
                  {(part as any).text}
                </div>
              </FadeIn>
            );
          }
          if (part.type.startsWith('tool-') || part.type === 'dynamic-tool') {
            const rendered = renderToolPart(part);
            return rendered ? <div key={key}>{rendered}</div> : null;
          }
          return null;
        })}
      </div>
    </div>
  );
}

function UserMessage({ message }: { message: UIMessage }) {
  const text = message.parts?.find(p => p.type === 'text') as any;
  return (
    <div className="flex items-start gap-3 flex-row-reverse">
      <div className="w-7 h-7 rounded-lg bg-blue-500/10 flex items-center justify-center shrink-0 mt-0.5">
        <User className="w-4 h-4 text-blue-400" />
      </div>
      <div className="max-w-[75%]">
        <div className="rounded-xl px-4 py-3 text-[11px] leading-relaxed whitespace-pre-wrap bg-blue-500/10 border border-blue-500/20 text-foreground">
          {text?.text ?? ''}
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   Error Display
   ═══════════════════════════════════════════════════════════════════ */

function ErrorBanner({ error, onRetry }: { error: Error; onRetry: () => void }) {
  return (
    <div className="flex items-start gap-3">
      <div className="w-7 h-7 rounded-lg bg-red-500/10 flex items-center justify-center shrink-0 mt-0.5">
        <XCircle className="w-4 h-4 text-red-400" />
      </div>
      <div className="bg-red-500/5 border border-red-500/20 rounded-xl px-4 py-3 max-w-[75%]">
        <p className="text-[11px] text-red-400 mb-2">
          {error.message.includes('GROQ_API_KEY')
            ? 'Groq API key not configured. Add GROQ_API_KEY to your .env file.'
            : `Connection error: ${error.message}`}
        </p>
        <button
          type="button"
          onClick={onRetry}
          className="flex items-center gap-1.5 text-[10px] text-red-400 hover:text-red-300 transition-colors"
        >
          <RefreshCw className="w-3 h-3" />
          Retry
        </button>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   Streaming Skeleton
   ═══════════════════════════════════════════════════════════════════ */

function StreamingSkeleton() {
  return (
    <div className="flex items-start gap-3">
      <div className="w-7 h-7 rounded-lg bg-[#FF8000]/10 flex items-center justify-center shrink-0 mt-0.5">
        <Bot className="w-4 h-4 text-[#FF8000]" />
      </div>
      <div className="space-y-2 w-full">
        {/* Metric card skeletons */}
        <div className="grid grid-cols-3 gap-2">
          {[1, 2, 3].map(i => (
            <div key={i} className="rounded-lg bg-[#1A1F2E] border border-[rgba(255,128,0,0.08)] px-3 py-2.5 animate-pulse">
              <div className="h-2 w-16 bg-[#FF8000]/10 rounded mb-2" />
              <div className="h-5 w-12 bg-[#FF8000]/15 rounded mb-1" />
              <div className="h-2 w-20 bg-[#FF8000]/5 rounded" />
            </div>
          ))}
        </div>
        {/* Chart skeleton */}
        <div className="rounded-lg bg-[#0D1117] border border-[rgba(255,128,0,0.08)] p-3 animate-pulse">
          <div className="h-2 w-24 bg-[#FF8000]/10 rounded mb-3" />
          <div className="h-[100px] bg-[#FF8000]/5 rounded" />
        </div>
        {/* Text skeleton */}
        <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
          <Loader2 className="w-3 h-3 animate-spin text-[#FF8000]" />
          Analyzing diagnostics...
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   Main Chatbot Component
   ═══════════════════════════════════════════════════════════════════ */

export function Chatbot() {
  const {
    messages, setMessages, sendMessage, status, error, clearError,
  } = useSimpleChat();

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const formRef = useRef<HTMLFormElement>(null);
  const [lastInput, setLastInput] = useState('');

  const isLoading = status === 'streaming' || status === 'submitted';
  // Show skeleton only when waiting for first content (submitted but no assistant message yet)
  const showSkeleton = status === 'submitted';
  // Show streaming indicator when we have content coming in
  const isStreaming = status === 'streaming';

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleSend = (text: string) => {
    if (!text.trim() || isLoading) return;
    setLastInput(text.trim());
    clearError();
    sendMessage({ text: text.trim() });
  };

  const handleRetry = () => {
    if (lastInput) {
      clearError();
      sendMessage({ text: lastInput });
    }
  };

  return (
    <div className="flex flex-col h-[calc(100vh-140px)]">
      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto space-y-4 pb-4">
        {messages.length === 0 && !error ? (
          <EmptyState onSelect={handleSend} />
        ) : (
          messages.map((msg) =>
            msg.role === 'user'
              ? <UserMessage key={msg.id} message={msg} />
              : msg.role === 'assistant'
                ? <AssistantMessage key={msg.id} message={msg} />
                : null
          )
        )}

        {/* Error display */}
        {error && <ErrorBanner error={error} onRetry={handleRetry} />}

        {/* Loading skeleton before first content arrives */}
        {showSkeleton && <StreamingSkeleton />}

        {/* Streaming indicator when content is flowing */}
        {isStreaming && (
          <div className="flex items-center gap-2 text-[10px] text-muted-foreground pl-10">
            <div className="flex gap-0.5">
              <span className="w-1 h-1 rounded-full bg-[#FF8000] animate-pulse" />
              <span className="w-1 h-1 rounded-full bg-[#FF8000] animate-pulse [animation-delay:150ms]" />
              <span className="w-1 h-1 rounded-full bg-[#FF8000] animate-pulse [animation-delay:300ms]" />
            </div>
            Generating...
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="shrink-0 pt-3 border-t border-[rgba(255,128,0,0.12)]">
        <form
          ref={formRef}
          onSubmit={(e) => {
            e.preventDefault();
            const input = inputRef.current;
            if (input) {
              handleSend(input.value);
              input.value = '';
            }
          }}
          className="flex items-center gap-2 bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)] px-4 py-2 focus-within:border-[#FF8000]/40 transition-colors"
        >
          <input
            ref={inputRef}
            type="text"
            placeholder="Ask about car diagnostics, system health, reliability..."
            className="flex-1 bg-transparent text-sm text-foreground placeholder:text-muted-foreground focus:outline-none"
            disabled={isLoading}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                formRef.current?.requestSubmit();
              }
            }}
          />
          <button
            type="submit"
            disabled={isLoading}
            title="Send message"
            className="w-7 h-7 rounded-lg bg-[#FF8000] flex items-center justify-center disabled:opacity-30 disabled:cursor-not-allowed hover:bg-[#FF8000]/80 transition-colors"
          >
            <Send className="w-3.5 h-3.5 text-[#0D1117]" />
          </button>
        </form>
        <div className="flex items-center justify-between mt-2 px-1">
          <span className="text-[11px] text-muted-foreground">
            Powered by Groq Llama 3.3 70B · GenUI
          </span>
          <div className="flex items-center gap-2">
            <span className="text-[11px] text-muted-foreground">
              {messages.filter(m => m.role === 'user').length} queries
            </span>
            {messages.length > 0 && (
              <button
                type="button"
                onClick={() => { setMessages([]); clearError(); }}
                className="flex items-center gap-1 text-[11px] text-muted-foreground hover:text-[#FF8000] transition-colors"
                title="New conversation"
              >
                <RotateCcw className="w-3 h-3" />
                New
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── Empty State ─────────────────────────────────────────────────── */

function EmptyState({ onSelect }: { onSelect: (q: string) => void }) {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center px-8">
      <div className="w-14 h-14 rounded-2xl bg-[#FF8000]/10 flex items-center justify-center mb-4">
        <Sparkles className="w-7 h-7 text-[#FF8000]" />
      </div>
      <h3 className="text-sm text-foreground mb-1">F1 Diagnostic AI</h3>
      <p className="text-[11px] text-muted-foreground mb-6 max-w-md">
        Ask questions about car health, system diagnostics, and reliability trends.
        Responses include interactive charts and visual metrics powered by real telemetry data.
      </p>
      <div className="grid grid-cols-2 gap-2 w-full max-w-lg">
        {SUGGESTIONS.map((q) => (
          <button
            type="button"
            key={q}
            onClick={() => onSelect(q)}
            className="text-left text-[11px] text-muted-foreground bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)] px-3 py-2.5 hover:border-[#FF8000]/30 hover:text-foreground transition-all"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}
