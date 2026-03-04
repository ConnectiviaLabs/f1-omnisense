import {
  LayoutDashboard,
  BarChart3,
  Brain,
  Activity,
  Box,
  Users,
  Gauge,
  AlertTriangle,
  TrendingUp,
  Combine,
  GitMerge,
  Bot,
  Wrench,
} from 'lucide-react';
import type { ViewType } from '../types';

interface SidebarProps {
  activeView: ViewType;
  onViewChange: (view: ViewType) => void;
  anomalyCount?: number;
}

type NavItem = { id: ViewType; label: string; icon: React.ElementType; badge?: number };

export function Sidebar({ activeView, onViewChange, anomalyCount }: SidebarProps) {
  const navGroups: { label: string; items: NavItem[] }[] = [
    {
      label: 'OPERATIONS',
      items: [
        { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
        { id: 'fleet-overview', label: 'Fleet Overview', icon: Box },
        { id: 'race-strategy', label: 'Repair Priority', icon: Wrench },
        { id: 'mclaren-analytics', label: 'Analytics', icon: BarChart3 },
        { id: 'ai-insights', label: 'AI Insights', icon: Brain },
        { id: 'driver', label: 'Driver Biodata', icon: Users },
      ],
    },
    {
      label: 'INTELLIGENCE PILLARS',
      items: [
        { id: 'car', label: 'Telemetry', icon: Gauge },
        { id: 'anomaly-detection', label: 'Anomaly Detection', icon: AlertTriangle, badge: anomalyCount },
        { id: 'forecasting', label: 'Forecasting', icon: TrendingUp },
      ],
    },
    {
      label: 'INTELLIGENCE INTERSECTIONS',
      items: [
        { id: 'driver-intel', label: 'Crossover Intel \u2014 A', icon: Combine },
        { id: 'circuit-intel', label: 'Crossover Intel \u2014 B', icon: Combine },
        { id: 'regulations', label: 'Convergence', icon: GitMerge },
        { id: 'chat', label: 'AI Agents', icon: Bot },
      ],
    },
  ];

  return (
    <aside className="w-[220px] min-h-full bg-[#0D1117] border-r border-[rgba(255,128,0,0.12)] flex flex-col">
      {/* Logo Area */}
      <div className="p-4 border-b border-[rgba(255,128,0,0.12)]">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-[#FF8000] flex items-center justify-center">
            <Activity className="w-4 h-4 text-[#0D1117]" />
          </div>
          <div>
            <div className="text-[#FF8000] text-sm tracking-[0.2em]">F1 OMNISENSE</div>
            <div className="text-[12px] text-foreground tracking-wider">DATASENSE</div>
          </div>
        </div>
        <div className="text-[11px] text-muted-foreground mt-2 tracking-wide">Powered by Connectivia Labs</div>
      </div>


      {/* Navigation */}
      <nav className="flex-1 p-2 overflow-y-auto">
        {navGroups.map(({ label: groupLabel, items }) => (
          <div key={groupLabel} className="mb-1">
            <div className="text-[10px] text-muted-foreground tracking-widest uppercase px-3 pt-3 pb-1">{groupLabel}</div>
            {items.map(({ id, label, icon: Icon, badge }) => (
              <button
                key={id}
                onClick={() => onViewChange(id)}
                className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-all border-l-[1.5px] ${
                  activeView === id
                    ? 'bg-[#FF8000]/10 text-[#FF8000] border-l-[#FF8000] font-medium'
                    : 'text-muted-foreground hover:bg-[#222838] hover:text-foreground border-l-transparent'
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
            ))}
          </div>
        ))}
      </nav>

    </aside>
  );
}
