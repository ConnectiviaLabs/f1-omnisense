import {
  LayoutDashboard,
  BarChart3,
  Brain,
  BookOpen,
  Activity,
  Video,
  MessageCircle,
  Box,
  Users,
  MapPin,
  Gauge,
  Flag,
  AlertTriangle,
  TrendingUp,
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
      label: 'DRIVER & CAR',
      items: [
        { id: 'driver-intel', label: 'Driver Intel', icon: Users },
        { id: 'car' as ViewType, label: 'Car Telemetry', icon: Gauge },
      ],
    },
    {
      label: 'FIELD & TEAM',
      items: [
        { id: 'dashboard', label: 'Live Dashboard', icon: LayoutDashboard },
        { id: 'circuit-intel', label: 'Circuit Intel', icon: MapPin },
        { id: 'race-strategy', label: 'Race Strategy', icon: Flag },
        { id: 'mclaren-analytics', label: 'McLaren Analytics', icon: BarChart3 },
        { id: 'fleet-overview', label: 'Fleet Overview', icon: Box },
      ],
    },
    {
      label: 'INTELLIGENCE PILLARS',
      items: [
        { id: 'car' as ViewType, label: 'Telemetry', icon: Gauge },
        { id: 'anomaly-detection' as ViewType, label: 'Anomaly Detection', icon: AlertTriangle, badge: anomalyCount },
        { id: 'forecasting' as ViewType, label: 'Forecasting', icon: TrendingUp },
      ],
    },
    {
      label: 'KNOWLEDGE',
      items: [
        { id: 'ai-insights', label: 'Knowledge Base', icon: Brain },
        { id: 'regulations', label: 'Regulations', icon: BookOpen },
        { id: 'chat', label: 'Knowledge Agent', icon: MessageCircle },
      ],
    },
    {
      label: 'MEDIA',
      items: [
        { id: 'media', label: 'Media Intel', icon: Video },
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
                key={`${groupLabel}-${id}`}
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
