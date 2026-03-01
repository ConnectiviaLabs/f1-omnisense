import { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { LiveDashboard } from './components/LiveDashboard';
import { McLarenAnalytics } from './components/McLarenAnalytics';
import { CarTelemetry } from './components/CarTelemetry';
import { DriverBiometrics } from './components/DriverBiometrics';
import { AIInsights } from './components/AIInsights';
import { Regulations } from './components/Regulations';
import { MediaIntelligence } from './components/MediaIntelligence';
import { Chatbot } from './components/Chatbot';
import { FleetOverview } from './components/FleetOverview';
import { DriverIntel } from './components/DriverIntel';
import { CircuitIntel } from './components/CircuitIntel';
import { ChevronRight, Wifi, Signal, Clock } from 'lucide-react';
import type { ViewType } from './types';

const viewTitles: Record<ViewType, { title: string; subtitle: string }> = {
  dashboard: { title: 'Live Race Dashboard', subtitle: 'Real-time F1 telemetry from OpenF1 API' },
  'mclaren-analytics': { title: 'McLaren Analytics', subtitle: 'Season standings, race strategy, tire stints & pit stops' },
  'driver-intel': { title: 'Driver Intelligence', subtitle: 'Performance markers, overtaking profiles & telemetry style for all 40 drivers' },
  'circuit-intel': { title: 'Circuit Intelligence', subtitle: 'Track layouts, pit loss times, air density & environmental conditions' },
  car: { title: 'Car Telemetry', subtitle: 'RPM, speed, throttle, brake, DRS & tire data from McLaren MCL60/MCL38' },
  driver: { title: 'Driver Biometrics', subtitle: 'Heart rate, cockpit temperature & physiological data for NOR & PIA' },
  'ai-insights': { title: 'Knowledge Base', subtitle: 'Pipeline intelligence & extraction stats' },
  regulations: { title: 'Regulations Browser', subtitle: 'FIA technical regulations, specs & equipment extracted via Groq' },
  media: { title: 'Media Intelligence', subtitle: 'GroundingDINO, SAM2, VideoMAE, TimeSformer, Gemma 3 & CLIP results' },
  chat: { title: 'Knowledge Agent', subtitle: 'RAG chatbot over FIA regulations & technical specs' },
  'fleet-overview': { title: 'Fleet Overview', subtitle: 'McLaren predictive maintenance & vehicle health monitoring' },
};

export default function App() {
  const [activeView, setActiveView] = useState<ViewType>('dashboard');
  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const renderView = () => {
    switch (activeView) {
      case 'dashboard': return <LiveDashboard />;
      case 'mclaren-analytics': return <McLarenAnalytics />;
      case 'driver-intel': return <DriverIntel />;
      case 'circuit-intel': return <CircuitIntel />;
      case 'car': return <CarTelemetry />;
      case 'driver': return <DriverBiometrics />;
      case 'ai-insights': return <AIInsights />;
      case 'regulations': return <Regulations />;
      case 'media': return <MediaIntelligence />;
      case 'chat': return <Chatbot />;
      case 'fleet-overview': return <FleetOverview />;
      default: return <LiveDashboard />;
    }
  };

  return (
    <div className="h-full flex bg-[#0D1117] font-['Inter',sans-serif] overflow-hidden">
      <Sidebar activeView={activeView} onViewChange={setActiveView} />

      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Top Bar */}
        <header className="h-12 border-b border-[rgba(255,128,0,0.12)] bg-[#0D1117] flex items-center justify-between px-4 shrink-0">
          <nav className="flex items-center gap-1.5 text-[11px] tracking-wide leading-none">
            <span className="text-muted-foreground">F1 OmniSense</span>
            <ChevronRight className="w-3 h-3 text-[rgba(255,128,0,0.3)] shrink-0" />
            <span className="text-foreground">{viewTitles[activeView].title}</span>
          </nav>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-3 text-[10px]">
              <div className="flex items-center gap-1.5">
                <Wifi className="w-3 h-3 text-green-400" />
                <span className="text-green-400">Live</span>
              </div>
              <div className="flex items-center gap-1.5">
                <Signal className="w-3 h-3 text-green-400" />
                <span className="text-muted-foreground">OpenF1 + Jolpica</span>
              </div>
            </div>
            <div className="flex items-center gap-1.5 bg-[#222838] rounded-lg px-3 py-1">
              <Clock className="w-3 h-3 text-[#FF8000]" />
              <span className="text-[10px] font-mono text-foreground">
                {currentTime.toLocaleTimeString('en-GB', { hour12: false })}
              </span>
            </div>
          </div>
        </header>

        {/* Page Header */}
        <div className="px-5 pt-5 pb-3 shrink-0 border-b border-[rgba(255,128,0,0.08)]">
          <div className="flex items-center gap-3">
            <div className="w-1 h-7 rounded-full bg-[#FF8000]" />
            <div>
              <h1 className="text-foreground text-xl font-semibold tracking-tight">{viewTitles[activeView].title}</h1>
              <p className="text-sm text-muted-foreground mt-0.5">{viewTitles[activeView].subtitle}</p>
            </div>
          </div>
        </div>

        {/* Content Area */}
        <main className="flex-1 overflow-y-auto px-5 pb-5">
          {renderView()}
        </main>
      </div>
    </div>
  );
}
