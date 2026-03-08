import { useState } from 'react';
import { Radio, FlaskConical, ArrowRight, Check, Activity } from 'lucide-react';
import type { ViewType } from '../types';

interface HomeProps {
  onSelectPlatform: (view: ViewType) => void;
}

export function Home_v3({ onSelectPlatform }: HomeProps) {
  const [hovered, setHovered] = useState<'raceday' | 'prime' | null>(null);

  return (
    <div className="relative h-full flex flex-col items-center justify-center bg-background overflow-hidden px-12 py-10">

      {/* Header */}
      <div className="relative z-10 flex flex-col items-center mb-10">
        <div className="flex items-center gap-3 mb-3">
          <div className="w-8 h-8 rounded-lg bg-[#FF8000] flex items-center justify-center">
            <Activity className="w-4 h-4 text-background" />
          </div>
          <span className="text-foreground text-lg font-semibold tracking-[0.2em] uppercase">McLaren Intelligence Platform</span>
        </div>
        <p className="text-[11px] text-muted-foreground tracking-[0.15em] uppercase">Operational Mode Selection</p>
      </div>

      {/* Cards */}
      <div className="relative z-10 flex gap-5 w-full max-w-4xl">

        {/* ── Race Day ── */}
        <div
          onMouseEnter={() => setHovered('raceday')}
          onMouseLeave={() => setHovered(null)}
          className={`flex-1 flex flex-col rounded-lg overflow-hidden transition-all duration-300 border ${
            hovered === 'raceday' ? 'border-[#FF8000]/40 shadow-[0_0_40px_rgba(255,128,0,0.08)]' : 'border-border'
          } bg-card`}
        >
          {/* Hero band */}
          <div className={`relative flex flex-col items-center justify-center py-10 border-b border-border transition-colors duration-300 ${
            hovered === 'raceday' ? 'bg-[#FF8000]/6' : ''
          }`}>
            <div className={`w-16 h-16 rounded-lg flex items-center justify-center mb-4 border border-border transition-all duration-300 ${
              hovered === 'raceday' ? 'bg-[#FF8000]/15 border-[#FF8000]/20' : 'bg-secondary'
            }`}>
              <Radio className="w-7 h-7 text-[#FF8000]" />
            </div>

            <div className="flex items-center gap-2 mb-1">
              <span className="w-1.5 h-1.5 rounded-full bg-[#FF8000]" />
              <span className="text-[9px] text-[#FF8000]/60 tracking-[0.25em] uppercase font-medium">Live Session</span>
            </div>

            <h2 className="text-xl font-semibold text-foreground tracking-tight">Race Day</h2>
          </div>

          {/* Body */}
          <div className="flex flex-col flex-1 p-6">
            <p className="text-xs text-muted-foreground leading-relaxed mb-6">
              Real-time telemetry, live timing, driver biometrics & on-track intelligence for every session.
            </p>

            <div className="flex flex-col gap-2.5 flex-1">
              {[
                'Live race positions & gap data',
                'Car telemetry — speed, RPM, DRS',
                'Driver biometrics & cockpit data',
                'Circuit maps & DRS zone analysis',
              ].map(item => (
                <div key={item} className="flex items-center gap-2.5">
                  <div className="w-4 h-4 rounded-md bg-[#FF8000]/10 border border-[#FF8000]/20 flex items-center justify-center flex-shrink-0">
                    <Check className="w-2.5 h-2.5 text-[#FF8000]/70" />
                  </div>
                  <span className="text-[11px] text-muted-foreground">{item}</span>
                </div>
              ))}
            </div>

            <button
              onClick={() => onSelectPlatform('dashboard')}
              className={`mt-6 w-full flex items-center justify-center gap-2 py-3 rounded-lg text-xs font-semibold tracking-wide transition-all duration-300 cursor-pointer border border-[#FF8000]/15 ${
                hovered === 'raceday'
                  ? 'bg-[#FF8000] text-background'
                  : 'bg-[#FF8000]/8 text-[#FF8000]/50'
              }`}
            >
              Enter Race Day
              <ArrowRight
                className="w-3.5 h-3.5 transition-transform duration-300"
                style={{ transform: hovered === 'raceday' ? 'translateX(3px)' : 'translateX(0)' }}
              />
            </button>
          </div>
        </div>

        {/* ── Prime ── */}
        <div
          onMouseEnter={() => setHovered('prime')}
          onMouseLeave={() => setHovered(null)}
          className={`flex-1 flex flex-col rounded-lg overflow-hidden transition-all duration-300 border ${
            hovered === 'prime' ? 'border-[#FF8000]/40 shadow-[0_0_40px_rgba(255,128,0,0.08)]' : 'border-border'
          } bg-card`}
        >
          {/* Hero band */}
          <div className={`relative flex flex-col items-center justify-center py-10 border-b border-border transition-colors duration-300 ${
            hovered === 'prime' ? 'bg-[#FF8000]/6' : ''
          }`}>
            <div className={`w-16 h-16 rounded-lg flex items-center justify-center mb-4 border border-border transition-all duration-300 ${
              hovered === 'prime' ? 'bg-[#FF8000]/15 border-[#FF8000]/20' : 'bg-secondary'
            }`}>
              <FlaskConical className="w-7 h-7 text-[#FF8000]" />
            </div>

            <div className="flex items-center gap-2 mb-1">
              <div className="w-1.5 h-1.5 rounded-sm bg-[#FF8000]/50" />
              <span className="text-[9px] text-[#FF8000]/60 tracking-[0.25em] uppercase font-medium">Deep Analytics</span>
            </div>

            <h2 className="text-xl font-semibold text-foreground tracking-tight">Prime</h2>
          </div>

          {/* Body */}
          <div className="flex flex-col flex-1 p-6">
            <p className="text-xs text-muted-foreground leading-relaxed mb-6">
              Offline driver analysis, car health monitoring, predictive maintenance & race strategy simulation.
            </p>

            <div className="flex flex-col gap-2.5 flex-1">
              {[
                'Driver intelligence & head-to-head',
                'Fleet health & anomaly detection',
                'Tyre strategy & pit window models',
                'McLaren constructor deep-dives',
              ].map(item => (
                <div key={item} className="flex items-center gap-2.5">
                  <div className="w-4 h-4 rounded-md bg-[#FF8000]/10 border border-[#FF8000]/20 flex items-center justify-center flex-shrink-0">
                    <Check className="w-2.5 h-2.5 text-[#FF8000]/70" />
                  </div>
                  <span className="text-[11px] text-muted-foreground">{item}</span>
                </div>
              ))}
            </div>

            <button
              onClick={() => onSelectPlatform('prime-driver')}
              className={`mt-6 w-full flex items-center justify-center gap-2 py-3 rounded-lg text-xs font-semibold tracking-wide transition-all duration-300 cursor-pointer border border-[#FF8000]/15 ${
                hovered === 'prime'
                  ? 'bg-[#FF8000] text-background'
                  : 'bg-[#FF8000]/8 text-[#FF8000]/50'
              }`}
            >
              Enter Prime
              <ArrowRight
                className="w-3.5 h-3.5 transition-transform duration-300"
                style={{ transform: hovered === 'prime' ? 'translateX(3px)' : 'translateX(0)' }}
              />
            </button>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="relative z-10 mt-8">
        <span className="text-[10px] text-muted-foreground/40 tracking-[0.2em] uppercase">McLaren Intelligence Platform</span>
      </div>
    </div>
  );
}
