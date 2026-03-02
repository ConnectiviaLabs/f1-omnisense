import {
  LayoutDashboard,
  BarChart3,
  Brain,
  BookOpen,
  Activity,
  Radio,
  Video,
  MessageCircle,
  Car,
  User,
  Box,
  Users,
  MapPin,
} from 'lucide-react';
import type { ViewType } from '../types';

interface SidebarProps {
  activeView: ViewType;
  onViewChange: (view: ViewType) => void;
}

type NavItem = { id: ViewType; label: string; icon: React.ElementType };

const navGroups: { label: string; items: NavItem[] }[] = [
  {
    label: 'DRIVER & CAR',
    items: [
      { id: 'driver-intel', label: 'Driver Intel', icon: Users },
    ],
  },
  {
    label: 'FIELD & TEAM',
    items: [
      { id: 'dashboard', label: 'Live Dashboard', icon: LayoutDashboard },
      { id: 'circuit-intel', label: 'Circuit Intel', icon: MapPin },
      { id: 'mclaren-analytics', label: 'McLaren Analytics', icon: BarChart3 },
      { id: 'fleet-overview', label: 'Fleet Overview', icon: Box },
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

export function Sidebar({ activeView, onViewChange }: SidebarProps) {
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

      {/* Live Status */}
      <div className="px-4 py-3 border-b border-[rgba(255,128,0,0.12)]">
        <div className="flex items-center gap-2">
          <Radio className="w-3 h-3 text-green-400 animate-pulse" />
          <span className="text-[12px] text-green-400 tracking-wider">LIVE DATA FEED</span>
        </div>
        <div className="text-[12px] text-muted-foreground mt-1 font-mono">
          OpenF1 + Jolpica APIs
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-2 overflow-y-auto">
        {navGroups.map(({ label: groupLabel, items }) => (
          <div key={groupLabel} className="mb-1">
            <div className="text-[10px] text-muted-foreground tracking-widest uppercase px-3 pt-3 pb-1">{groupLabel}</div>
            {items.map(({ id, label, icon: Icon }) => (
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
                <span className="tracking-wide">{label}</span>
              </button>
            ))}
          </div>
        ))}
      </nav>

    </aside>
  );
}
