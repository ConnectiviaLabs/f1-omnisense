import { useState, useEffect } from 'react';
import { SidebarProvider } from './hooks/useSidebar';
import { Sidebar } from './components/Sidebar';
import { Home_v3 as Home } from './components/Home_v3';
import { LiveDashboard } from './components/LiveDashboard';
import { CarTelemetry } from './components/CarTelemetry';
import { DriverBiometrics } from './components/DriverBiometrics';
import { RaceSchedule } from './components/RaceSchedule';
import { PrimeDriver } from './components/PrimeDriver';
import { PrimeCar } from './components/PrimeCar';
import { PrimeTeam } from './components/PrimeTeam';
import { PrimeStrategy } from './components/PrimeStrategy';
import { AIInsights } from './components/AIInsights';
import { Regulations } from './components/Regulations';
import { MediaIntelligence } from './components/MediaIntelligence';
import { Chatbot } from './components/Chatbot';
import { AdvantageTrident } from './components/AdvantageTrident';
import { AdvantageCrossover } from './components/AdvantageCrossover';
import { TeamRadioTimeline } from './components/TeamRadioTimeline';
import { RealRacing } from './components/RealRacing';
import { ChevronRight, Wifi, Signal, Clock } from 'lucide-react';
import type { ViewType } from './types';
import type { Pillar, StrategyTab } from './components/Sidebar';
import { parseAnomalyDrivers } from './components/anomalyHelpers';
import type { VehicleData } from './components/anomalyHelpers';

const RACE_DAY_VIEWS = new Set<ViewType>(['dashboard', 'car', 'driver', 'schedule']);

const viewTitles: Record<ViewType, { title: string; subtitle: string }> = {
  home: { title: 'Home', subtitle: '' },
  dashboard: { title: 'Live Race Dashboard', subtitle: 'Real-time F1 telemetry from OpenF1 API' },
  car: { title: 'Car Telemetry', subtitle: 'RPM, speed, throttle, brake, DRS & tire data across all drivers' },
  driver: { title: 'Driver Biometrics', subtitle: 'Heart rate, cockpit temperature & physiological data for NOR & PIA' },
  schedule: { title: '2026 Race Calendar', subtitle: '24-round FIA Formula 1 World Championship season schedule' },
  'prime-driver': { title: 'Driver Intelligence', subtitle: 'Performance markers, overtaking profiles & telemetry style for all 40 drivers' },
  'prime-car': { title: 'Car Intelligence', subtitle: 'Predictive maintenance, anomaly detection & fleet health monitoring' },
  'prime-team': { title: 'Team Intelligence', subtitle: 'Constructor performance, fleet anomalies & forecasting across all 10 teams' },
  'prime-strategy': { title: 'Strategy Intelligence', subtitle: 'Race strategy, circuit analysis & season analytics' },
  'advantage-trident': { title: 'Trident', subtitle: 'Convergence reports synthesized from KeX, anomalies & forecasts' },
  'advantage-crossover': { title: 'Crossover', subtitle: 'Entity similarity across drivers, teams & cars' },
  'ai-insights': { title: 'Knowledge Base', subtitle: 'Pipeline intelligence & extraction stats' },
  regulations: { title: 'Regulations Browser', subtitle: 'FIA technical regulations, specs & equipment extracted via Groq' },
  media: { title: 'Media Intelligence', subtitle: 'GroundingDINO, SAM2, VideoMAE, TimeSformer, Gemma 3 & CLIP results' },
  radio: { title: 'Team Radio', subtitle: 'Signal extraction, driver communication profiles & race radio timeline' },
  chat: { title: 'Knowledge Agent', subtitle: 'RAG chatbot over FIA regulations & technical specs' },
  'real-racing': { title: 'Real Racing Data', subtitle: 'AiM RaceStudio 3 telemetry, track maps, health & anomaly analysis' },
};

export default function App() {
  const [activeView, setActiveView] = useState<ViewType>('home');
  const [currentTime, setCurrentTime] = useState(new Date());
  const [lastPlatform, setLastPlatform] = useState<'race-day' | 'prime'>('race-day');
  const [activePillar, setActivePillar] = useState<Pillar>('telemetry');
  const [activeStrategyTab, setActiveStrategyTab] = useState<StrategyTab>('race-strategy');

  // Track which platform we're in based on view changes
  const platform: 'race-day' | 'prime' = RACE_DAY_VIEWS.has(activeView) ? 'race-day' : 'prime';

  // Remember last platform for Knowledge views (so sidebar stays correct)
  const isPrimeView = (v: ViewType) => v.startsWith('prime-') || v.startsWith('advantage-') || v === 'real-racing';

  useEffect(() => {
    if (RACE_DAY_VIEWS.has(activeView)) setLastPlatform('race-day');
    else if (isPrimeView(activeView)) setLastPlatform('prime');
  }, [activeView]);

  const effectivePlatform = RACE_DAY_VIEWS.has(activeView) || isPrimeView(activeView)
    ? platform
    : lastPlatform;

  // ── Pre-fetch fleet anomaly data on app startup ──────────
  const [fleetVehicles, setFleetVehicles] = useState<VehicleData[]>([]);
  const [fleetLoading, setFleetLoading] = useState(true);

  useEffect(() => {
    fetch('/api/pipeline/anomaly')
      .then(r => r.json())
      .then(data => setFleetVehicles(parseAnomalyDrivers(data)))
      .catch(err => console.error('Fleet prefetch error:', err))
      .finally(() => setFleetLoading(false));
  }, []);

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const renderView = () => {
    switch (activeView) {
      case 'dashboard': return <LiveDashboard />;
      case 'car': return <CarTelemetry />;
      case 'driver': return <DriverBiometrics />;
      case 'schedule': return <RaceSchedule />;
      case 'prime-driver': return <PrimeDriver prefetchedVehicles={fleetVehicles} activePillar={activePillar} />;
      case 'prime-car': return <PrimeCar prefetchedVehicles={fleetVehicles} prefetchLoading={fleetLoading} activePillar={activePillar} />;
      case 'prime-team': return <PrimeTeam prefetchedVehicles={fleetVehicles} activePillar={activePillar} />;
      case 'prime-strategy': return <PrimeStrategy activeTab={activeStrategyTab} />;
      case 'advantage-trident': return <AdvantageTrident />;
      case 'advantage-crossover': return <AdvantageCrossover />;
      case 'ai-insights': return <AIInsights />;
      case 'regulations': return <Regulations />;
      case 'media': return <MediaIntelligence />;
      case 'radio': return <TeamRadioTimeline />;
      case 'chat': return <Chatbot />;
      case 'real-racing': return <RealRacing />;
      default: return <LiveDashboard />;
    }
  };

  // Home — full-width, no sidebar
  if (activeView === 'home') {
    return (
      <div className="h-full bg-background font-['Inter',sans-serif]">
        <Home onSelectPlatform={setActiveView} />
      </div>
    );
  }

  return (
    <SidebarProvider>
    <div className="h-full flex bg-background font-['Inter',sans-serif] overflow-hidden">
      <Sidebar
        activeView={activeView}
        onViewChange={setActiveView}
        onGoHome={() => setActiveView('home')}
        platform={effectivePlatform}
        anomalyCount={fleetVehicles.filter(v => v.systems.some(s => s.level === 'critical')).length || undefined}
        activePillar={activePillar}
        onPillarChange={setActivePillar}
        activeStrategyTab={activeStrategyTab}
        onStrategyTabChange={setActiveStrategyTab}
      />

      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Top Bar */}
        <header className="h-12 border-b border-border bg-background flex items-center justify-between px-4 shrink-0 shadow-sm">
          <nav className="flex items-center gap-1.5 text-[11px] tracking-wide leading-none">
            <span className="text-muted-foreground">F1 OmniSense</span>
            <ChevronRight className="w-3 h-3 text-primary/30 shrink-0" />
            <span className="text-muted-foreground">{effectivePlatform === 'race-day' ? 'Race Day' : 'Prime'}</span>
            <ChevronRight className="w-3 h-3 text-primary/30 shrink-0" />
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
            <div className="flex items-center gap-1.5 bg-secondary rounded-lg px-3 py-1">
              <Clock className="w-3 h-3 text-primary" />
              <span className="text-[10px] font-mono text-foreground">
                {currentTime.toLocaleTimeString('en-GB', { hour12: false })}
              </span>
            </div>
          </div>
        </header>

        {/* Page Header */}
        <div className="px-5 pt-5 pb-3 shrink-0 border-b border-border">
          <div className="flex items-center gap-3">
            <div className="w-1 h-7 rounded-full bg-primary" />
            <div>
              <h1 className="text-foreground text-xl font-semibold tracking-tight">{viewTitles[activeView].title}</h1>
              <p className="text-[13px] text-muted-foreground mt-px">{viewTitles[activeView].subtitle}</p>
            </div>
          </div>
        </div>

        {/* Content Area */}
        <main className="flex-1 overflow-y-auto px-5 pb-5">
          {renderView()}
        </main>
      </div>
    </div>
    </SidebarProvider>
  );
}
