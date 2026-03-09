import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
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
  ClipboardList,
  ChevronLeft,
  ChevronRight,
  ChevronDown,
  Wifi,
  Cpu,
  Zap,
  User,
  Settings,
  LogOut,
  HelpCircle,
  Radio,
} from 'lucide-react';
import type { ViewType } from '../types';
import { useSidebar } from '../hooks/useSidebar';

export type Pillar = 'telemetry' | 'anomaly' | 'forecast';
export type StrategyTab = 'race-strategy' | 'circuit-intel' | 'season-analytics' | 'backtest' | 'prep-mode';

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
  { id: 'prep-mode', label: 'Prep Mode', icon: ClipboardList },
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
  { id: 'radio' as ViewType, label: 'Team Radio', icon: Radio },
];

/* ─── Animation variants ─── */
const sidebarVariants = {
  expanded: { width: 220, transition: { duration: 0.25, ease: [0.4, 0, 0.2, 1] as const } },
  collapsed: { width: 56, transition: { duration: 0.25, ease: [0.4, 0, 0.2, 1] as const } },
};

const labelVariants = {
  show: { opacity: 1, x: 0, transition: { duration: 0.15, delay: 0.1 } },
  hide: { opacity: 0, x: -8, transition: { duration: 0.1 } },
};

const subItemVariants = {
  open: { height: 'auto' as const, opacity: 1, transition: { duration: 0.2, ease: 'easeOut' as const } },
  closed: { height: 0, opacity: 0, transition: { duration: 0.15, ease: 'easeIn' as const } },
};

/* ─── System Status Indicator ─── */
function SystemStatus({ collapsed }: { collapsed: boolean }) {
  const [latency, setLatency] = useState<number | null>(null);

  useEffect(() => {
    const measure = async () => {
      try {
        const start = performance.now();
        await fetch('/api/pipeline/health', { method: 'HEAD' });
        setLatency(Math.round(performance.now() - start));
      } catch {
        setLatency(null);
      }
    };
    measure();
    const id = setInterval(measure, 30_000);
    return () => clearInterval(id);
  }, []);

  const latencyColor = latency === null ? 'text-danger' : latency < 200 ? 'text-success' : latency < 500 ? 'text-warning' : 'text-danger';
  const latencyDot = latency === null ? 'bg-danger' : latency < 200 ? 'bg-success' : latency < 500 ? 'bg-warning' : 'bg-danger';

  if (collapsed) {
    return (
      <div className="px-2 py-2 border-t border-border flex flex-col items-center gap-2">
        <div className={`w-2 h-2 rounded-full ${latencyDot} animate-pulse`} />
      </div>
    );
  }

  return (
    <div className="px-3 py-3 border-t border-border space-y-2">
      <div className="text-[10px] text-muted-foreground tracking-widest uppercase">SYSTEM</div>
      <div className="space-y-1.5">
        <div className="flex items-center gap-2">
          <Wifi className="w-3 h-3 text-success shrink-0" />
          <span className="text-[11px] text-muted-foreground flex-1">Sensors</span>
          <span className="text-[10px] text-success font-mono">ONLINE</span>
        </div>
        <div className="flex items-center gap-2">
          <Cpu className="w-3 h-3 text-success shrink-0" />
          <span className="text-[11px] text-muted-foreground flex-1">AI Models</span>
          <span className="text-[10px] text-success font-mono">READY</span>
        </div>
        <div className="flex items-center gap-2">
          <Zap className={`w-3 h-3 ${latencyColor} shrink-0`} />
          <span className="text-[11px] text-muted-foreground flex-1">Latency</span>
          <span className={`text-[10px] ${latencyColor} font-mono`}>
            {latency !== null ? `${latency}ms` : 'ERR'}
          </span>
        </div>
      </div>
    </div>
  );
}

/* ─── User Profile ─── */
function UserProfile({ collapsed }: { collapsed: boolean }) {
  const [open, setOpen] = useState(false);

  if (collapsed) {
    return (
      <div className="px-2 py-2 border-t border-border flex justify-center">
        <button title="User profile" className="w-8 h-8 rounded-full bg-primary/15 border border-primary/20 flex items-center justify-center hover:bg-primary/25 transition-colors">
          <User className="w-3.5 h-3.5 text-primary" />
        </button>
      </div>
    );
  }

  return (
    <div className="px-3 py-3 border-t border-border relative">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-2.5 px-2 py-1.5 rounded-lg hover:bg-secondary transition-colors"
      >
        <div className="w-7 h-7 rounded-full bg-primary/15 border border-primary/20 flex items-center justify-center shrink-0">
          <User className="w-3.5 h-3.5 text-primary" />
        </div>
        <div className="flex-1 text-left min-w-0">
          <div className="text-[12px] text-foreground font-medium truncate">Race Engineer</div>
          <div className="text-[10px] text-muted-foreground truncate">McLaren Racing</div>
        </div>
        <ChevronDown className={`w-3 h-3 text-muted-foreground transition-transform duration-200 ${open ? 'rotate-180' : ''}`} />
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: 4, scale: 0.96 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 4, scale: 0.96 }}
            transition={{ duration: 0.15 }}
            className="absolute bottom-full left-3 right-3 mb-1 bg-card border border-border rounded-lg shadow-lg overflow-hidden z-50"
          >
            {[
              { icon: Settings, label: 'Settings', action: () => {} },
              { icon: HelpCircle, label: 'Help & Support', action: () => {} },
              { icon: LogOut, label: 'Sign Out', action: () => {} },
            ].map(({ icon: Icon, label, action }) => (
              <button
                key={label}
                onClick={() => { action(); setOpen(false); }}
                className="w-full flex items-center gap-2.5 px-3 py-2 text-[12px] text-muted-foreground hover:bg-secondary hover:text-foreground transition-colors"
              >
                <Icon className="w-3.5 h-3.5 shrink-0" />
                {label}
              </button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

/* ─── Main Sidebar ─── */
export function Sidebar({ activeView, onViewChange, onGoHome, platform, anomalyCount, activePillar, onPillarChange, activeStrategyTab, onStrategyTabChange }: SidebarProps) {
  const isPrime = platform === 'prime';
  const { collapsed, toggle } = useSidebar();

  const renderNavItem = (id: ViewType, label: string, Icon: React.ElementType, badge?: number) => {
    const isActive = activeView === id;
    return (
      <button
        key={id}
        onClick={() => onViewChange(id)}
        title={collapsed ? label : undefined}
        className={`w-full flex items-center gap-3 rounded-lg text-sm transition-all border-l-[1.5px] ${
          collapsed ? 'px-0 py-2 justify-center' : 'px-3 py-2'
        } ${
          isActive
            ? 'bg-primary/10 text-primary border-l-primary font-medium'
            : 'text-muted-foreground hover:bg-secondary hover:text-foreground border-l-transparent'
        }`}
      >
        <Icon className="w-4 h-4 shrink-0" />
        <AnimatePresence mode="wait">
          {!collapsed && (
            <motion.span
              key="label"
              variants={labelVariants}
              initial="hide"
              animate="show"
              exit="hide"
              className="tracking-wide flex-1 text-left whitespace-nowrap overflow-hidden"
            >
              {label}
            </motion.span>
          )}
        </AnimatePresence>
        {!collapsed && badge != null && badge > 0 && (
          <span className="ml-auto w-5 h-5 rounded-full bg-red-500 text-white text-[10px] font-bold flex items-center justify-center">
            {badge}
          </span>
        )}
      </button>
    );
  };

  const renderSectionLabel = (text: string) => (
    <AnimatePresence mode="wait">
      {!collapsed ? (
        <motion.div
          key="label"
          variants={labelVariants}
          initial="hide"
          animate="show"
          exit="hide"
          className="text-[10px] text-muted-foreground tracking-widest uppercase px-3 pb-1.5"
        >
          {text}
        </motion.div>
      ) : (
        <div className="border-t border-border mx-2 my-1" />
      )}
    </AnimatePresence>
  );

  return (
    <motion.aside
      variants={sidebarVariants}
      animate={collapsed ? 'collapsed' : 'expanded'}
      className="min-h-full bg-background border-r border-border flex flex-col overflow-hidden shrink-0"
    >
      {/* Logo Area */}
      <button onClick={onGoHome} className={`border-b border-border text-left hover:bg-card/50 transition-colors ${collapsed ? 'p-2 flex justify-center' : 'p-4'}`}>
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center shrink-0">
            <Activity className="w-4 h-4 text-[#0D1117]" />
          </div>
          <AnimatePresence mode="wait">
            {!collapsed && (
              <motion.div key="branding" variants={labelVariants} initial="hide" animate="show" exit="hide">
                <div className="text-primary text-sm tracking-[0.2em] whitespace-nowrap">F1 OMNISENSE</div>
                <div className="text-[12px] text-foreground tracking-wider">DATASENSE</div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
        <AnimatePresence mode="wait">
          {!collapsed && (
            <motion.div key="back" variants={labelVariants} initial="hide" animate="show" exit="hide" className="flex items-center gap-1.5 mt-2">
              <Home className="w-3 h-3 text-muted-foreground" />
              <span className="text-[11px] text-muted-foreground tracking-wide">Back to platforms</span>
            </motion.div>
          )}
        </AnimatePresence>
      </button>

      {/* Collapse Toggle */}
      <div className={`flex ${collapsed ? 'justify-center' : 'justify-end'} px-2 py-1.5`}>
        <button
          onClick={toggle}
          className="w-6 h-6 rounded-md flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-secondary transition-colors"
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {collapsed ? <ChevronRight className="w-3.5 h-3.5" /> : <ChevronLeft className="w-3.5 h-3.5" />}
        </button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-2 py-1 overflow-y-auto space-y-3">
        {/* Race Day */}
        {!isPrime && (
          <div>
            {renderSectionLabel('RACE DAY')}
            {RACE_DAY_ITEMS.map(({ id, label, icon }) => renderNavItem(id, label, icon))}
          </div>
        )}

        {/* Prime — Intelligence with pillar sub-items */}
        {isPrime && (
          <div>
            {renderSectionLabel('INTELLIGENCE')}
            {PRIME_INTEL_ITEMS(anomalyCount).map(({ id, label, icon: Icon, badge }) => {
              const isActive = activeView === id;
              return (
                <div key={id}>
                  {renderNavItem(id, label, Icon, badge)}
                  {/* Pillar sub-items — only when expanded and active */}
                  <AnimatePresence>
                    {isActive && !collapsed && (
                      <motion.div
                        variants={subItemVariants}
                        initial="closed"
                        animate="open"
                        exit="closed"
                        className="ml-4 border-l border-border pl-1 my-0.5 overflow-hidden"
                      >
                        {PILLAR_ITEMS.map(({ id: pid, label: pLabel, icon: PIcon }) => (
                          <button
                            key={pid}
                            onClick={() => onPillarChange(pid as Pillar)}
                            className={`w-full flex items-center gap-2.5 px-3 py-1.5 rounded-md text-[13px] transition-all ${
                              activePillar === pid
                                ? 'text-primary font-medium'
                                : 'text-muted-foreground hover:text-foreground'
                            }`}
                          >
                            <PIcon className="w-3.5 h-3.5 shrink-0" />
                            <span className="tracking-wide">{pLabel}</span>
                          </button>
                        ))}
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              );
            })}
          </div>
        )}

        {/* Prime — Strategy */}
        {isPrime && (
          <div>
            {renderSectionLabel('STRATEGY')}
            {STRATEGY_ITEMS.map(({ id: sid, label: sLabel, icon: SIcon }) => (
              <button
                key={sid}
                onClick={() => { onViewChange('prime-strategy'); onStrategyTabChange(sid as StrategyTab); }}
                title={collapsed ? sLabel : undefined}
                className={`w-full flex items-center gap-3 rounded-lg text-sm transition-all border-l-[1.5px] ${
                  collapsed ? 'px-0 py-2 justify-center' : 'px-3 py-2'
                } ${
                  activeView === 'prime-strategy' && activeStrategyTab === sid
                    ? 'bg-primary/10 text-primary border-l-primary font-medium'
                    : 'text-muted-foreground hover:bg-secondary hover:text-foreground border-l-transparent'
                }`}
              >
                <SIcon className="w-4 h-4 shrink-0" />
                <AnimatePresence mode="wait">
                  {!collapsed && (
                    <motion.span key="label" variants={labelVariants} initial="hide" animate="show" exit="hide" className="tracking-wide flex-1 text-left whitespace-nowrap overflow-hidden">
                      {sLabel}
                    </motion.span>
                  )}
                </AnimatePresence>
              </button>
            ))}
          </div>
        )}

        {/* Advantage */}
        {isPrime && (
          <div>
            {renderSectionLabel('ADVANTAGE')}
            {ADVANTAGE_ITEMS.map(({ id, label, icon }) => renderNavItem(id, label, icon))}
          </div>
        )}

        {/* Knowledge */}
        <div>
          {renderSectionLabel('KNOWLEDGE')}
          {KNOWLEDGE_ITEMS.map(({ id, label, icon }) => renderNavItem(id, label, icon))}
        </div>
      </nav>

      {/* System Status */}
      <SystemStatus collapsed={collapsed} />

      {/* User Profile */}
      <UserProfile collapsed={collapsed} />
    </motion.aside>
  );
}
