import { useState, useEffect, useMemo } from 'react';
import {
  Search, Filter, BookOpen, ChevronDown, ChevronRight,
  Loader2, Ruler, Cog, Hash,
} from 'lucide-react';
import type { IntelligenceData, RegulationRule } from '../types';

type TabType = 'rules' | 'dimensions' | 'equipment';

export function Regulations() {
  const [data, setData] = useState<IntelligenceData | null>(null);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [expandedRule, setExpandedRule] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<TabType>('rules');

  useEffect(() => {
    fetch('/api/pipeline/intelligence')
      .then((r) => r.json())
      .then((d) => { setData(d); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  const categories = useMemo(() => {
    if (!data?.rules) return [];
    const cats = new Set<string>();
    data.rules.forEach((r) => { if (r.category) cats.add(r.category); });
    return ['all', ...Array.from(cats).sort()];
  }, [data]);

  const filteredRules = useMemo(() => {
    if (!data?.rules) return [];
    let rules = data.rules;
    if (selectedCategory !== 'all') {
      rules = rules.filter((r) => r.category === selectedCategory);
    }
    if (search) {
      const q = search.toLowerCase();
      rules = rules.filter((r) =>
        (r.id || '').toLowerCase().includes(q) ||
        (r.description || '').toLowerCase().includes(q) ||
        (r.category || '').toLowerCase().includes(q)
      );
    }
    return rules;
  }, [data, selectedCategory, search]);

  const filteredDimensions = useMemo(() => {
    if (!data?.dimensional_data) return [];
    if (!search) return data.dimensional_data;
    const q = search.toLowerCase();
    return data.dimensional_data.filter((d) =>
      (d.component || '').toLowerCase().includes(q) ||
      (d.dimension || '').toLowerCase().includes(q)
    );
  }, [data, search]);

  const filteredEquipment = useMemo(() => {
    if (!data?.equipment) return [];
    if (!search) return data.equipment;
    const q = search.toLowerCase();
    return data.equipment.filter((e) =>
      (e.tag || '').toLowerCase().includes(q) ||
      (e.type || '').toLowerCase().includes(q) ||
      (e.description || '').toLowerCase().includes(q)
    );
  }, [data, search]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-6 h-6 text-[#FF8000] animate-spin" />
        <span className="ml-3 text-muted-foreground text-sm">Loading regulations data...</span>
      </div>
    );
  }

  if (!data?.rules) {
    return (
      <div className="flex items-center justify-center h-64">
        <BookOpen className="w-6 h-6 text-amber-400" />
        <span className="ml-3 text-muted-foreground text-sm">
          No regulations data. Run the extraction pipeline first.
        </span>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Search + Filters */}
      <div className="flex items-center gap-3">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search regulations, dimensions, equipment..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full bg-card border border-border rounded-lg pl-10 pr-4 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:border-[#FF8000]/40"
          />
        </div>
        {/* Tabs */}
        <div className="flex items-center bg-card border border-border rounded-lg overflow-hidden">
          {([
            { id: 'rules' as TabType, label: 'Rules', icon: <BookOpen className="w-3 h-3" />, count: data.stats?.total_rules ?? 0 },
            { id: 'dimensions' as TabType, label: 'Dimensions', icon: <Ruler className="w-3 h-3" />, count: data.stats?.total_dimensions ?? 0 },
            { id: 'equipment' as TabType, label: 'Equipment', icon: <Cog className="w-3 h-3" />, count: data.stats?.total_equipment ?? 0 },
          ]).map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-1.5 px-3 py-2 text-[12px] transition-colors ${
                activeTab === tab.id
                  ? 'bg-[#FF8000]/10 text-[#FF8000]'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              {tab.icon}
              {tab.label}
              <span className="font-mono text-[11px]">({tab.count})</span>
            </button>
          ))}
        </div>
      </div>

      {/* Category Filter (rules tab only) */}
      {activeTab === 'rules' && (
        <div className="flex items-center gap-2 overflow-x-auto pb-1">
          <Filter className="w-3 h-3 text-muted-foreground shrink-0" />
          {categories.slice(0, 15).map((cat) => (
            <button
              key={cat}
              onClick={() => setSelectedCategory(cat)}
              className={`text-[12px] px-2 py-1 rounded-lg whitespace-nowrap transition-all ${
                selectedCategory === cat
                  ? 'bg-[#FF8000]/10 text-[#FF8000]'
                  : 'text-muted-foreground hover:bg-secondary hover:text-foreground'
              }`}
            >
              {cat === 'all' ? 'All Categories' : cat}
            </button>
          ))}
        </div>
      )}

      {/* Results count */}
      <div className="text-[12px] text-muted-foreground">
        {activeTab === 'rules' && `${filteredRules.length} regulation${filteredRules.length !== 1 ? 's' : ''}`}
        {activeTab === 'dimensions' && `${filteredDimensions.length} dimensional spec${filteredDimensions.length !== 1 ? 's' : ''}`}
        {activeTab === 'equipment' && `${filteredEquipment.length} equipment item${filteredEquipment.length !== 1 ? 's' : ''}`}
        {search && ` matching "${search}"`}
      </div>

      {/* Rules Tab */}
      {activeTab === 'rules' && (
        <div className="space-y-1.5">
          {filteredRules.slice(0, 100).map((rule, idx) => (
            <RuleCard
              key={`${rule.id}-${idx}`}
              rule={rule}
              expanded={expandedRule === `${rule.id}-${idx}`}
              onToggle={() => setExpandedRule(expandedRule === `${rule.id}-${idx}` ? null : `${rule.id}-${idx}`)}
            />
          ))}
          {filteredRules.length > 100 && (
            <div className="text-center text-[12px] text-muted-foreground py-2">
              Showing 100 of {filteredRules.length} — refine your search to see more
            </div>
          )}
          {filteredRules.length === 0 && (
            <div className="text-center text-muted-foreground text-sm py-8">No rules match your filters</div>
          )}
        </div>
      )}

      {/* Dimensions Tab */}
      {activeTab === 'dimensions' && (
        <div className="bg-card border border-border rounded-lg shadow-[0_1px_3px_rgba(0,0,0,0.3)] overflow-hidden">
          <table className="w-full text-[11px]">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left text-muted-foreground font-normal px-4 py-2 tracking-wider text-[11px]">COMPONENT</th>
                <th className="text-left text-muted-foreground font-normal px-4 py-2 tracking-wider text-[11px]">DIMENSION</th>
                <th className="text-right text-muted-foreground font-normal px-4 py-2 tracking-wider text-[11px]">VALUE</th>
                <th className="text-right text-muted-foreground font-normal px-4 py-2 tracking-wider text-[11px]">PAGE</th>
              </tr>
            </thead>
            <tbody>
              {filteredDimensions.slice(0, 80).map((dim, i) => (
                <tr key={i} className="border-b border-[rgba(255,128,0,0.04)] hover:bg-secondary transition-colors">
                  <td className="px-4 py-1.5 text-foreground">{dim.component}</td>
                  <td className="px-4 py-1.5 text-muted-foreground truncate max-w-[300px]">{dim.dimension}</td>
                  <td className="px-4 py-1.5 text-right font-mono text-[#FF8000]">
                    {dim.value != null ? `${dim.value} ${dim.unit || ''}` : '—'}
                  </td>
                  <td className="px-4 py-1.5 text-right font-mono text-muted-foreground">{dim._page}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {filteredDimensions.length > 80 && (
            <div className="text-center text-[12px] text-muted-foreground py-2">
              Showing 80 of {filteredDimensions.length}
            </div>
          )}
        </div>
      )}

      {/* Equipment Tab */}
      {activeTab === 'equipment' && (
        <div className="grid grid-cols-2 gap-2">
          {filteredEquipment.slice(0, 60).map((eq, i) => (
            <div key={i} className="bg-card border border-border rounded-lg shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-3">
              <div className="flex items-center gap-2 mb-1">
                <Hash className="w-3 h-3 text-[#FF8000]" />
                <span className="text-sm font-mono text-[#FF8000]">{eq.tag}</span>
              </div>
              <div className="text-[11px] text-foreground">{eq.type}</div>
              {eq.description && (
                <div className="text-[12px] text-muted-foreground mt-1 line-clamp-2">{eq.description}</div>
              )}
              {eq.location_description && (
                <div className="text-[12px] text-cyan-400 mt-1">{eq.location_description}</div>
              )}
            </div>
          ))}
          {filteredEquipment.length === 0 && (
            <div className="col-span-2 text-center text-muted-foreground text-sm py-8">No equipment matches</div>
          )}
        </div>
      )}
    </div>
  );
}

function RuleCard({ rule, expanded, onToggle }: {
  rule: RegulationRule;
  expanded: boolean;
  onToggle: () => void;
}) {
  const severityColor = rule.severity === 'violation'
    ? 'text-red-400 bg-red-500/10'
    : rule.severity === 'warning'
    ? 'text-amber-400 bg-amber-500/10'
    : 'text-cyan-400 bg-cyan-500/10';

  return (
    <div className="bg-card border border-border rounded-lg shadow-[0_1px_3px_rgba(0,0,0,0.3)] overflow-hidden">
      <button
        onClick={onToggle}
        className="w-full flex items-center gap-3 px-4 py-2.5 text-left hover:bg-secondary transition-colors"
      >
        {expanded
          ? <ChevronDown className="w-3 h-3 text-[#FF8000] shrink-0" />
          : <ChevronRight className="w-3 h-3 text-muted-foreground shrink-0" />
        }
        <span className="text-[12px] font-mono text-[#FF8000] w-14 shrink-0">{rule.id}</span>
        <span className="text-[12px] px-1.5 py-0.5 rounded-full shrink-0 tracking-wider text-[11px] uppercase ${severityColor}">
          <span className={`${severityColor} px-1.5 py-0.5 rounded-full`}>{rule.severity}</span>
        </span>
        <span className="text-[11px] text-foreground flex-1 truncate">
          {rule.description || rule.category}
        </span>
        <span className="text-[12px] text-muted-foreground shrink-0">{rule.category}</span>
      </button>

      {expanded && (
        <div className="px-4 pb-3 pt-1 border-t border-border space-y-2">
          {rule.description && (
            <p className="text-[11px] text-muted-foreground leading-relaxed">{rule.description}</p>
          )}
          <div className="flex items-center gap-4 text-[12px]">
            {rule.value && (
              <div>
                <span className="text-muted-foreground">Value: </span>
                <span className="font-mono text-[#FF8000]">{rule.value} {rule.unit || ''}</span>
              </div>
            )}
            {rule.condition && (
              <div>
                <span className="text-muted-foreground">Condition: </span>
                <span className="text-foreground">{rule.condition}</span>
              </div>
            )}
            {rule.reference && (
              <div>
                <span className="text-muted-foreground">Ref: </span>
                <span className="font-mono text-foreground">{rule.reference}</span>
              </div>
            )}
            <div>
              <span className="text-muted-foreground">Page: </span>
              <span className="font-mono text-foreground">{rule._page}</span>
            </div>
            {rule.source_standard && (
              <div>
                <span className="text-muted-foreground">Standard: </span>
                <span className="text-cyan-400">{rule.source_standard}</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
