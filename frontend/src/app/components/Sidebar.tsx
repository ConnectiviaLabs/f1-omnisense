import {
  LayoutDashboard,
  Activity,
  Video,
  MessageCircle,
  Box,
  Shield,
  Users,
  Gauge,
  Flag,
  Brain,
  BookOpen,
  Heart,
  Home,
  AlertTriangle,
  TrendingUp,
  MapPin,
  BarChart3,
  Calendar,
  Layers,
  GitCompare,
  FlaskConical,
} from 'lucide-react';
import type { ViewType } from '../types';

export type Pillar = 'telemetry' | 'anomaly' | 'forecast';
export type StrategyTab = 'race-strategy' | 'circuit-intel' | 'season-analytics' | 'backtest';

interface SidebarProps {
  activeView: ViewType;
  onViewChange: (view: ViewType) => void;
  onGoHome: () => void;
  platform: 'race-day' | 'prime';
  anomalyCount?: number;
  activePillar: Pillar;
  onPillarChange: (p: Pillar) => void;
  activeStrategyTab: StrategyTab;
  onStrategyTabChange: (t: StrategyTab) => void;
}

type NavItem = { id: ViewType; label: string; icon: React.ElementType; badge?: number };
type SubItem = { id: string; label: string; icon: React.ElementType };

const RACE_DAY_ITEMS: NavItem[] = [
  { id: 'dashboard', label: 'Live Dashboard', icon: LayoutDashboard },
  { id: 'car', label: 'Car Telemetry', icon: Gauge },
  { id: 'driver', label: 'Driver Biometrics', icon: Heart },
  { id: 'schedule', label: '2026 Calendar', icon: Calendar },
];

const PRIME_INTEL_ITEMS: (anomalyCount?: number) => NavItem[] = (anomalyCount) => [
  { id: 'prime-driver', label: 'Driver', icon: Users },
  { id: 'prime-car', label: 'Car', icon: Box, badge: anomalyCount },
  { id: 'prime-team', label: 'Team', icon: Shield },
];

const PILLAR_ITEMS: SubItem[] = [
  { id: 'telemetry', label: 'Telemetry', icon: Gauge },
  { id: 'anomaly', label: 'Anomaly Detection', icon: AlertTriangle },
  { id: 'forecast', label: 'Forecasting', icon: TrendingUp },
];

const STRATEGY_ITEMS: SubItem[] = [
  { id: 'race-strategy', label: 'Race Strategy', icon: Flag },
  { id: 'circuit-intel', label: 'Circuit Intel', icon: MapPin },
  { id: 'season-analytics', label: 'Season Analytics', icon: BarChart3 },
  { id: 'backtest', label: 'Backtest', icon: FlaskConical },
];

const ADVANTAGE_ITEMS: NavItem[] = [
  { id: 'advantage-trident', label: 'Trident', icon: Layers },
  { id: 'advantage-crossover', label: 'Crossover', icon: GitCompare },
];

const KNOWLEDGE_ITEMS: NavItem[] = [
  { id: 'ai-insights', label: 'Knowledge Base', icon: Brain },
  { id: 'regulations', label: 'Regulations', icon: BookOpen },
  { id: 'chat', label: 'Knowledge Agent', icon: MessageCircle },
  { id: 'media', label: 'Media Intel', icon: Video },
];


export function Sidebar({ activeView, onViewChange, onGoHome, platform, anomalyCount, activePillar, onPillarChange, activeStrategyTab, onStrategyTabChange }: SidebarProps) {
  const isPrime = platform === 'prime';

  return (
    <aside className="w-[220px] min-h-full bg-background border-r border-border flex flex-col">
      {/* Logo Area — click to go home */}
      <button onClick={onGoHome} className="p-4 border-b border-border text-left hover:bg-card/50 transition-colors">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-[#FF8000] flex items-center justify-center">
            <Activity className="w-4 h-4 text-[#0D1117]" />
          </div>
          <div>
            <div className="text-[#FF8000] text-sm tracking-[0.2em]">F1 OMNISENSE</div>
            <div className="text-[12px] text-foreground tracking-wider">DATASENSE</div>
          </div>
        </div>
        <div className="flex items-center gap-1.5 mt-2">
          <Home className="w-3 h-3 text-muted-foreground" />
          <span className="text-[11px] text-muted-foreground tracking-wide">Back to platforms</span>
        </div>
      </button>

      {/* Navigation */}
      <nav className="flex-1 p-2 overflow-y-auto">
        {/* Race Day items */}
        {!isPrime && (
          <div className="mb-1">
            <div className="text-[10px] text-muted-foreground tracking-widest uppercase px-3 pt-3 pb-1">RACE DAY</div>
            {RACE_DAY_ITEMS.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => onViewChange(id)}
                className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-all border-l-[1.5px] ${
                  activeView === id
                    ? 'bg-[#FF8000]/10 text-[#FF8000] border-l-[#FF8000] font-medium'
                    : 'text-muted-foreground hover:bg-secondary hover:text-foreground border-l-transparent'
                }`}
              >
                <Icon className="w-4 h-4 shrink-0" />
                <span className="tracking-wide flex-1 text-left">{label}</span>
              </button>
            ))}
          </div>
        )}

        {/* Prime — Intelligence (Driver & Car with pillar sub-items) */}
        {isPrime && (
          <div className="mb-1">
            <div className="text-[10px] text-muted-foreground tracking-widest uppercase px-3 pt-3 pb-1">INTELLIGENCE</div>
            {PRIME_INTEL_ITEMS(anomalyCount).map(({ id, label, icon: Icon, badge }) => {
              const isActive = activeView === id;
              return (
                <div key={id}>
                  <button
                    onClick={() => onViewChange(id)}
                    className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-all border-l-[1.5px] ${
                      isActive
                        ? 'bg-[#FF8000]/10 text-[#FF8000] border-l-[#FF8000] font-medium'
                        : 'text-muted-foreground hover:bg-secondary hover:text-foreground border-l-transparent'
                    }`}
                  >
                    <Icon className="w-4 h-4 shrink-0" />
                    <span className="tracking-wide flex-1 text-left">{label}</span>
                    {badge != null && badge > 0 && (
                      <span className="ml-auto w-5 h-5 rounded-full bg-red-500 text-white text-[10px] font-bold flex items-center justify-center">
                        {badge}
                      </span>
                    )}
                  </button>
                  {isActive && (
                    <div className="ml-4 border-l border-border pl-1 my-0.5">
                      {PILLAR_ITEMS.map(({ id: pid, label: pLabel, icon: PIcon }) => (
                        <button
                          key={pid}
                          onClick={() => onPillarChange(pid as Pillar)}
                          className={`w-full flex items-center gap-2.5 px-3 py-1.5 rounded-md text-[13px] transition-all ${
                            activePillar === pid
                              ? 'text-[#FF8000] font-medium'
                              : 'text-muted-foreground hover:text-foreground'
                          }`}
                        >
                          <PIcon className="w-3.5 h-3.5 shrink-0" />
                          <span className="tracking-wide">{pLabel}</span>
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {/* Prime — Strategy (separate section, sub-items shown directly) */}
        {isPrime && (
          <div className="mb-1">
            <div className="text-[10px] text-muted-foreground tracking-widest uppercase px-3 pt-3 pb-1">STRATEGY</div>
            {STRATEGY_ITEMS.map(({ id: sid, label: sLabel, icon: SIcon }) => (
              <button
                key={sid}
                onClick={() => { onViewChange('prime-strategy'); onStrategyTabChange(sid as StrategyTab); }}
                className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-all border-l-[1.5px] ${
                  activeView === 'prime-strategy' && activeStrategyTab === sid
                    ? 'bg-[#FF8000]/10 text-[#FF8000] border-l-[#FF8000] font-medium'
                    : 'text-muted-foreground hover:bg-secondary hover:text-foreground border-l-transparent'
                }`}
              >
                <SIcon className="w-4 h-4 shrink-0" />
                <span className="tracking-wide flex-1 text-left">{sLabel}</span>
              </button>
            ))}
          </div>
        )}

        {/* Advantage — Prime only */}
        {isPrime && (
          <div className="mb-1">
            <div className="text-[10px] text-muted-foreground tracking-widest uppercase px-3 pt-3 pb-1">ADVANTAGE</div>
            {ADVANTAGE_ITEMS.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => onViewChange(id)}
                className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-all border-l-[1.5px] ${
                  activeView === id
                    ? 'bg-[#FF8000]/10 text-[#FF8000] border-l-[#FF8000] font-medium'
                    : 'text-muted-foreground hover:bg-secondary hover:text-foreground border-l-transparent'
                }`}
              >
                <Icon className="w-4 h-4 shrink-0" />
                <span className="tracking-wide flex-1 text-left">{label}</span>
              </button>
            ))}
          </div>
        )}

        {/* Knowledge — always visible */}
        <div className="mb-1">
          <div className="text-[10px] text-muted-foreground tracking-widest uppercase px-3 pt-3 pb-1">KNOWLEDGE</div>
          {KNOWLEDGE_ITEMS.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => onViewChange(id)}
              className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-all border-l-[1.5px] ${
                activeView === id
                  ? 'bg-[#FF8000]/10 text-[#FF8000] border-l-[#FF8000] font-medium'
                  : 'text-muted-foreground hover:bg-secondary hover:text-foreground border-l-transparent'
              }`}
            >
              <Icon className="w-4 h-4 shrink-0" />
              <span className="tracking-wide flex-1 text-left">{label}</span>
            </button>
          ))}
        </div>
      </nav>
    </aside>
  );
}
