import { useState, useRef, useEffect } from 'react';
import {
  Send, Bot, User, Loader2, Sparkles,
  FileText, BookOpen, Cog, Ruler, Database, RotateCcw, Filter,
} from 'lucide-react';

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
}

interface Source {
  content: string;
  data_type: string;
  category: string;
  source: string;
  page: number;
}

const SUGGESTIONS = [
  'What are the minimum car weight requirements?',
  'Explain the front wing dimensional constraints',
  'What tire compounds are allowed in 2024?',
  'List the safety equipment regulations',
  'What are the power unit restrictions?',
  'Describe the floor body regulations',
];

const typeIcons: Record<string, React.ReactNode> = {
  regulation: <BookOpen className="w-3 h-3" />,
  equipment: <Cog className="w-3 h-3" />,
  dimension: <Ruler className="w-3 h-3" />,
  material: <Database className="w-3 h-3" />,
  document_metadata: <FileText className="w-3 h-3" />,
};

const FILTER_LABELS: Record<string, string> = {
  regulation: 'Regulations',
  equipment: 'Equipment',
  dimension: 'Dimensions',
  material: 'Materials',
  document_metadata: 'Documents',
};

export function Chatbot() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [expandedSources, setExpandedSources] = useState<number | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(() => localStorage.getItem('omni_chat_session'));
  const [modelUsed, setModelUsed] = useState<string | null>(null);
  const [activeFilters, setActiveFilters] = useState<Set<string>>(new Set());
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const sendMessage = async (text: string) => {
    if (!text.trim() || loading) return;

    const userMsg: ChatMessage = { role: 'user', content: text.trim() };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      // Try OmniRAG endpoint first (server-side session), fall back to legacy /api/chat
      let data: any;
      try {
        const omniRes = await fetch('/api/omni/rag/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: text.trim(),
            session_id: sessionId,
            ...(activeFilters.size > 0 && { data_types: [...activeFilters] }),
          }),
        });
        if (omniRes.ok) {
          data = await omniRes.json();
          if (data.session_id) {
            setSessionId(data.session_id);
            localStorage.setItem('omni_chat_session', data.session_id);
          }
          if (data.model_used) setModelUsed(data.model_used);
        }
      } catch { /* OmniRAG unavailable, fall through */ }

      if (!data) {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: text.trim(),
            history: messages.slice(-10).map(m => ({ role: m.role, content: m.content })),
            ...(activeFilters.size > 0 && { data_types: [...activeFilters] }),
          }),
        });
        if (!res.ok) throw new Error(`Server error: ${res.status}`);
        data = await res.json();
        setModelUsed(null);
      }

      const assistantMsg: ChatMessage = {
        role: 'assistant',
        content: data.answer,
        sources: data.sources,
      };
      setMessages(prev => [...prev, assistantMsg]);
    } catch (err) {
      const errorMsg: ChatMessage = {
        role: 'assistant',
        content: 'Failed to reach the knowledge agent. Make sure the chat server is running:\n```\npython pipeline/chat_server.py\n```',
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  };

  return (
    <div className="flex flex-col h-[calc(100vh-140px)]">
      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto space-y-4 pb-4">
        {messages.length === 0 ? (
          <EmptyState onSelect={sendMessage} />
        ) : (
          messages.map((msg, i) => (
            <MessageBubble
              key={i}
              message={msg}
              index={i}
              expandedSources={expandedSources}
              onToggleSources={() => setExpandedSources(expandedSources === i ? null : i)}
            />
          ))
        )}

        {loading && (
          <div className="flex items-start gap-3">
            <div className="w-7 h-7 rounded-lg bg-[#FF8000]/10 flex items-center justify-center shrink-0 mt-0.5">
              <Bot className="w-4 h-4 text-[#FF8000]" />
            </div>
            <div className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)] px-4 py-3">
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Loader2 className="w-3 h-3 animate-spin text-[#FF8000]" />
                Searching knowledge base...
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="shrink-0 pt-3 border-t border-[rgba(255,128,0,0.12)]">
        {/* Topic Filters */}
        <div className="flex items-center gap-1.5 mb-2 px-1 flex-wrap">
          <Filter className="w-3 h-3 text-muted-foreground shrink-0" />
          {Object.entries(FILTER_LABELS).map(([key, label]) => {
            const isActive = activeFilters.has(key);
            return (
              <button
                key={key}
                type="button"
                onClick={() => {
                  setActiveFilters(prev => {
                    const next = new Set(prev);
                    if (next.has(key)) next.delete(key);
                    else next.add(key);
                    return next;
                  });
                }}
                className={`flex items-center gap-1 px-2 py-1 rounded-lg text-[11px] border transition-all ${
                  isActive
                    ? 'bg-[#FF8000]/15 border-[#FF8000]/40 text-[#FF8000]'
                    : 'bg-[#1A1F2E] border-[rgba(255,128,0,0.08)] text-muted-foreground hover:border-[#FF8000]/20 hover:text-foreground'
                }`}
              >
                {typeIcons[key]}
                {label}
              </button>
            );
          })}
          {activeFilters.size > 0 && (
            <button
              type="button"
              onClick={() => setActiveFilters(new Set())}
              className="text-[10px] text-muted-foreground hover:text-[#FF8000] transition-colors ml-1"
            >
              Clear
            </button>
          )}
        </div>
        <div className="flex items-center gap-2 bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)] px-4 py-2 focus-within:border-[#FF8000]/40 transition-colors">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about F1 regulations, specs, equipment..."
            className="flex-1 bg-transparent text-sm text-foreground placeholder:text-muted-foreground focus:outline-none"
            disabled={loading}
          />
          <button
            onClick={() => sendMessage(input)}
            disabled={!input.trim() || loading}
            className="w-7 h-7 rounded-lg bg-[#FF8000] flex items-center justify-center disabled:opacity-30 disabled:cursor-not-allowed hover:bg-[#FF8000]/80 transition-colors"
          >
            <Send className="w-3.5 h-3.5 text-[#0D1117]" />
          </button>
        </div>
        <div className="flex items-center justify-between mt-2 px-1">
          <span className="text-[11px] text-muted-foreground">
            {modelUsed ? `Model: ${modelUsed}` : 'Powered by Groq Llama 3.3 70B'} + Atlas Vector Search
          </span>
          <div className="flex items-center gap-2">
            <span className="text-[11px] text-muted-foreground">
              {messages.filter(m => m.role === 'user').length} queries
            </span>
            {messages.length > 0 && (
              <button
                type="button"
                onClick={() => {
                  setMessages([]);
                  setSessionId(null);
                  setModelUsed(null);
                  localStorage.removeItem('omni_chat_session');
                }}
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

function EmptyState({ onSelect }: { onSelect: (q: string) => void }) {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center px-8">
      <div className="w-14 h-14 rounded-2xl bg-[#FF8000]/10 flex items-center justify-center mb-4">
        <Sparkles className="w-7 h-7 text-[#FF8000]" />
      </div>
      <h3 className="text-sm text-foreground mb-1">F1 Knowledge Agent</h3>
      <p className="text-[11px] text-muted-foreground mb-6 max-w-md">
        Ask questions about FIA technical regulations, car dimensions,
        equipment specs, and materials. Powered by RAG over 2,449 extracted documents.
      </p>
      <div className="grid grid-cols-2 gap-2 w-full max-w-lg">
        {SUGGESTIONS.map((q) => (
          <button
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

function MessageBubble({
  message, index, expandedSources, onToggleSources,
}: {
  message: ChatMessage;
  index: number;
  expandedSources: number | null;
  onToggleSources: () => void;
}) {
  const isUser = message.role === 'user';

  return (
    <div className={`flex items-start gap-3 ${isUser ? 'flex-row-reverse' : ''}`}>
      <div className={`w-7 h-7 rounded-lg flex items-center justify-center shrink-0 mt-0.5 ${
        isUser ? 'bg-blue-500/10' : 'bg-[#FF8000]/10'
      }`}>
        {isUser
          ? <User className="w-4 h-4 text-blue-400" />
          : <Bot className="w-4 h-4 text-[#FF8000]" />
        }
      </div>

      <div className={`max-w-[75%] space-y-2 ${isUser ? 'items-end' : ''}`}>
        <div className={`rounded-xl px-4 py-3 text-[11px] leading-relaxed whitespace-pre-wrap ${
          isUser
            ? 'bg-blue-500/10 border border-blue-500/20 text-foreground'
            : 'bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] text-foreground'
        }`}>
          {message.content}
        </div>

        {/* Sources */}
        {message.sources && message.sources.length > 0 && (
          <div>
            <button
              onClick={onToggleSources}
              className="text-[12px] text-[#FF8000] hover:text-[#FF8000]/80 flex items-center gap-1"
            >
              <FileText className="w-3 h-3" />
              {expandedSources === index ? 'Hide' : 'Show'} {message.sources.length} sources
            </button>

            {expandedSources === index && (
              <div className="mt-2 space-y-1.5">
                {message.sources.map((src, si) => (
                  <div
                    key={si}
                    className="bg-[#0D1117] border border-[rgba(255,128,0,0.12)] rounded-lg px-3 py-2"
                  >
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-[#FF8000]">
                        {typeIcons[src.data_type] || <FileText className="w-3 h-3" />}
                      </span>
                      <span className="text-[11px] text-[#FF8000] font-mono">{src.data_type}</span>
                      <span className="text-[11px] text-muted-foreground">/ {src.category}</span>
                      <span className="text-[11px] text-muted-foreground ml-auto">p.{src.page}</span>
                    </div>
                    <p className="text-[12px] text-muted-foreground line-clamp-3">{src.content}</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
