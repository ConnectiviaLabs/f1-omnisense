import {
  LayoutDashboard,
  BarChart3,
  Brain,
  BookOpen,
  Activity,
  Settings,
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

const navItems: { id: ViewType; label: string; icon: React.ElementType }[] = [
  { id: 'dashboard', label: 'Live Dashboard', icon: LayoutDashboard },
  { id: 'mclaren-analytics', label: 'McLaren Analytics', icon: BarChart3 },
  { id: 'driver-intel', label: 'Driver Intel', icon: Users },
  { id: 'circuit-intel', label: 'Circuit Intel', icon: MapPin },
  { id: 'car', label: 'Car Telemetry', icon: Car },
  { id: 'driver', label: 'Driver Bio', icon: User },
  { id: 'media', label: 'Media Intel', icon: Video },
  { id: 'ai-insights', label: 'Knowledge Base', icon: Brain },
  { id: 'regulations', label: 'Regulations', icon: BookOpen },
  { id: 'chat', label: 'Knowledge Agent', icon: MessageCircle },
  { id: 'fleet-overview', label: 'Fleet Overview', icon: Box },
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
      <nav className="flex-1 p-2 space-y-0.5">
        {navItems.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => onViewChange(id)}
            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all border-l-[1.5px] ${
              activeView === id
                ? 'bg-[#FF8000]/10 text-[#FF8000] border-l-[#FF8000] font-medium'
                : 'text-muted-foreground hover:bg-[#222838] hover:text-foreground border-l-transparent'
            }`}
          >
            <Icon className="w-4 h-4 shrink-0" />
            <span className="tracking-wide">{label}</span>
          </button>
        ))}
      </nav>

      {/* System Status */}
      <div className="p-4 border-t border-[rgba(255,128,0,0.12)]">
        <button className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm text-muted-foreground hover:bg-[#222838] hover:text-foreground transition-all">
          <Settings className="w-4 h-4" />
          <span className="tracking-wide">Settings</span>
        </button>
        <div className="mt-3 px-3 space-y-1.5">
          <div className="flex items-center justify-between text-[12px]">
            <span className="text-muted-foreground">OpenF1</span>
            <span className="text-green-400">Connected</span>
          </div>
          <div className="flex items-center justify-between text-[12px]">
            <span className="text-muted-foreground">Jolpica</span>
            <span className="text-green-400">Connected</span>
          </div>
          <div className="flex items-center justify-between text-[12px]">
            <span className="text-muted-foreground">Pipeline</span>
            <span className="text-green-400">Ready</span>
          </div>
          <div className="flex items-center justify-between text-[12px]">
            <span className="text-muted-foreground">Local Data</span>
            <span className="text-green-400">Loaded</span>
          </div>
        </div>
      </div>
    </aside>
  );
}
