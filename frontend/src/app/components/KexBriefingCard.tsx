import { useState } from 'react';
import { Brain, Sparkles, ChevronDown, ChevronUp, Loader2 } from 'lucide-react';
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer, Tooltip,
} from 'recharts';

/* ─── Types ───────────────────────────────────────────────────────────── */

interface KexData {
  text: string;
  scores?: Record<string, number>;
  summary?: string;
  model_used?: string;
  provider_used?: string;
  generated_at?: number;
  grounding_score?: number;
}

interface Props {
  title: string;
  icon?: 'brain' | 'sparkles';
  kex: KexData | null;
  loading: boolean;
  loadingText?: string;
}

/* ─── Main component ──────────────────────────────────────────────────── */

export default function KexBriefingCard({ title, icon = 'brain', kex, loading, loadingText }: Props) {
  const [expanded, setExpanded] = useState(false);
  const IconComponent = icon === 'sparkles' ? Sparkles : Brain;

  // Detect scale: if any value > 10, it's 0-100 scale; otherwise 0-10
  const maxVal = kex?.scores
    ? Math.max(...Object.values(kex.scores).filter(v => typeof v === 'number'))
    : 0;
  const scale = maxVal > 10 ? 100 : 10;

  const radarData = kex?.scores
    ? Object.entries(kex.scores).map(([key, value]) => ({
        dimension: key,
        value: typeof value === 'number' ? value : 0,
        fullMark: scale,
      }))
    : [];

  return (
    <div className="bg-card border border-border rounded-xl p-4">
      <h3 className="text-sm text-muted-foreground flex items-center gap-2 mb-3">
        <IconComponent className="w-4 h-4" /> {title}
      </h3>

      {loading && (
        <div className="flex items-center justify-center gap-2 py-6">
          <Loader2 className="w-4 h-4 text-[#FF8000] animate-spin" />
          <span className="text-[11px] text-muted-foreground">{loadingText || 'Generating intelligence\u2026'}</span>
        </div>
      )}

      {kex && !loading && (
        <div className="space-y-3">
          {/* ── Radar chart from real data scores ── */}
          {radarData.length > 0 && (
            <ResponsiveContainer width="100%" height={220}>
              <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="72%">
                <PolarGrid stroke="rgba(255,255,255,0.05)" />
                <PolarAngleAxis
                  dataKey="dimension"
                  tick={{ fontSize: 9, fill: '#888' }}
                />
                <PolarRadiusAxis
                  domain={[0, scale]}
                  tickCount={6}
                  tick={{ fontSize: 8, fill: '#555' }}
                  axisLine={false}
                />
                <Tooltip
                  contentStyle={{
                    background: '#1A1F2E',
                    border: '1px solid rgba(255,255,255,0.1)',
                    fontSize: 11,
                    borderRadius: 8,
                  }}
                  formatter={(value: number) => [`${value}/${scale}`, '']}
                />
                <Radar
                  dataKey="value"
                  stroke="#FF8000"
                  fill="#FF8000"
                  fillOpacity={0.2}
                  strokeWidth={1.5}
                  dot={{ r: 3, fill: '#FF8000' }}
                />
              </RadarChart>
            </ResponsiveContainer>
          )}

          {/* ── Score pills ── */}
          {radarData.length > 0 && (
            <div className="flex flex-wrap gap-1.5 justify-center">
              {radarData.map(d => {
                const pct = d.value / scale;
                const color = pct >= 0.7 ? '#05DF72' : pct >= 0.4 ? '#FF8000' : '#FB2C36';
                return (
                  <span
                    key={d.dimension}
                    className="text-[9px] font-mono px-1.5 py-0.5 rounded"
                    style={{ color, background: `${color}15`, border: `1px solid ${color}25` }}
                  >
                    {d.dimension} {d.value}
                  </span>
                );
              })}
            </div>
          )}

          {/* ── Summary preview ── */}
          {kex.summary && !expanded && (
            <p className="text-[11px] text-muted-foreground leading-relaxed">
              {kex.summary}
            </p>
          )}

          {/* ── Grounding + model info ── */}
          <div className="flex items-center gap-2 flex-wrap">
            {kex.grounding_score != null && (
              <span className={`text-[8px] font-semibold px-1 py-0.5 rounded ${
                kex.grounding_score >= 0.7 ? 'bg-green-500/15 text-green-400' : 'bg-amber-500/15 text-amber-400'
              }`}>
                {Math.round(kex.grounding_score * 100)}% grounded
              </span>
            )}
            <span className="text-[9px] text-muted-foreground/50 font-mono">
              via {kex.model_used || 'auto'}{kex.provider_used ? ` (${kex.provider_used})` : ''}
            </span>
            {kex.generated_at && (
              <span className="text-[9px] text-muted-foreground/50">
                {new Date(kex.generated_at * 1000).toLocaleString()}
              </span>
            )}
          </div>

          {/* ── Expand toggle ── */}
          <button
            onClick={() => setExpanded(!expanded)}
            className="flex items-center gap-1.5 text-[11px] text-[#FF8000] hover:text-[#FF9933] transition-colors w-full justify-center py-1.5 rounded-lg hover:bg-[#FF8000]/5"
          >
            {expanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
            {expanded ? 'Hide Full Briefing' : 'View Full Briefing'}
          </button>

          {/* ── Full text (collapsible) ── */}
          {expanded && (
            <div className="pt-1 border-t border-border">
              <div className="text-[12px] text-muted-foreground leading-relaxed whitespace-pre-line">
                {kex.text}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
