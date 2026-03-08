import { useState } from 'react';
import { Activity, Radio, FlaskConical, ArrowRight, Zap, BarChart3, Brain, Shield, MapPin, Users, Check } from 'lucide-react';
import type { ViewType } from '../types';

interface HomeProps {
  onSelectPlatform: (view: ViewType) => void;
}

export function Home_v3({ onSelectPlatform }: HomeProps) {
  const [hovered, setHovered] = useState<'raceday' | 'prime' | null>(null);

  return (
    <div className="relative h-full flex flex-col items-center justify-center bg-background overflow-hidden px-12 py-10">

      {/* Grid background */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          backgroundImage: `
            linear-gradient(rgba(255,128,0,0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,128,0,0.03) 1px, transparent 1px)
          `,
          backgroundSize: '64px 64px',
        }}
      />

      {/* Ambient glows */}
      <div className="absolute top-1/2 left-1/4 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-[#FF8000]/4 rounded-full blur-[120px] pointer-events-none" />
      <div className="absolute top-1/2 right-1/4 translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-[#FF8000]/4 rounded-full blur-[120px] pointer-events-none" />

      {/* Header */}
      <div className="relative z-10 flex flex-col items-center mb-10">
        <div className="flex items-center gap-3 mb-2">
          <div className="w-10 h-10 rounded-xl bg-[#FF8000] flex items-center justify-center shadow-[0_0_20px_rgba(255,128,0,0.5)]">
            <Activity className="w-5 h-5 text-[#0D1117]" />
          </div>
          <span className="text-[#FF8000] text-xl font-bold tracking-[0.2em]">F1 OMNISENSE</span>
        </div>
        <p className="text-[11px] text-[#FF8000]/40 tracking-[0.15em] uppercase">Powered by Connectivia Labs</p>
        <p className="text-[10px] text-white/20 tracking-[0.3em] uppercase mt-1">Choose your platform to continue</p>
      </div>

      {/* Cards */}
      <div className="relative z-10 flex gap-5 w-full max-w-4xl">

        {/* ── Race Day ── */}
        <div
          onMouseEnter={() => setHovered('raceday')}
          onMouseLeave={() => setHovered(null)}
          className="flex-1 flex flex-col rounded-2xl overflow-hidden transition-all duration-500 border"
          style={{
            borderColor: hovered === 'raceday' ? 'rgba(255,128,0,0.4)' : 'rgba(255,128,0,0.1)',
            boxShadow: hovered === 'raceday' ? '0 0 60px rgba(255,128,0,0.1), inset 0 1px 0 rgba(255,128,0,0.1)' : 'none',
            background: '#111827',
          }}
        >
          {/* Hero band */}
          <div
            className="relative flex flex-col items-center justify-center py-10 transition-all duration-500"
            style={{
              background: hovered === 'raceday'
                ? 'linear-gradient(180deg, rgba(255,128,0,0.12) 0%, rgba(255,128,0,0.04) 100%)'
                : 'linear-gradient(180deg, rgba(255,128,0,0.06) 0%, transparent 100%)',
              borderBottom: '1px solid rgba(255,128,0,0.08)',
            }}
          >
            <div
              className="w-20 h-20 rounded-2xl flex items-center justify-center mb-4 transition-all duration-500"
              style={{
                background: hovered === 'raceday' ? 'rgba(255,128,0,0.2)' : 'rgba(255,128,0,0.08)',
                boxShadow: hovered === 'raceday' ? '0 0 40px rgba(255,128,0,0.2)' : 'none',
                border: '1px solid rgba(255,128,0,0.15)',
              }}
            >
              <Radio className="w-9 h-9 text-[#FF8000]" />
            </div>

            <div className="flex items-center gap-2 mb-1">
              <span className="relative flex h-1.5 w-1.5">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#FF8000] opacity-75" />
                <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-[#FF8000]" />
              </span>
              <span className="text-[9px] text-[#FF8000]/60 tracking-[0.25em] uppercase font-medium">Live Session</span>
            </div>

            <h2 className="text-2xl font-bold text-white tracking-tight">Race Day</h2>
          </div>

          {/* Body */}
          <div className="flex flex-col flex-1 p-6">
            <p className="text-xs text-white/35 leading-relaxed mb-6">
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
                  <div className="w-4 h-4 rounded-full bg-[#FF8000]/10 border border-[#FF8000]/20 flex items-center justify-center flex-shrink-0">
                    <Check className="w-2.5 h-2.5 text-[#FF8000]/70" />
                  </div>
                  <span className="text-[11px] text-white/40">{item}</span>
                </div>
              ))}
            </div>

            {/* Launch button */}
            <button
              onClick={() => onSelectPlatform('dashboard')}
              className="mt-6 w-full flex items-center justify-center gap-2 py-3 rounded-xl text-xs font-semibold tracking-wide transition-all duration-300 cursor-pointer hover:brightness-110"
              style={{
                background: hovered === 'raceday' ? '#FF8000' : 'rgba(255,128,0,0.08)',
                color: hovered === 'raceday' ? '#0D1117' : 'rgba(255,128,0,0.5)',
                border: '1px solid rgba(255,128,0,0.15)',
              }}
            >
              Launch Race Day
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
          className="flex-1 flex flex-col rounded-2xl overflow-hidden transition-all duration-500 border"
          style={{
            borderColor: hovered === 'prime' ? 'rgba(255,128,0,0.4)' : 'rgba(255,128,0,0.1)',
            boxShadow: hovered === 'prime' ? '0 0 60px rgba(255,128,0,0.1), inset 0 1px 0 rgba(255,128,0,0.1)' : 'none',
            background: '#111827',
          }}
        >
          {/* Hero band */}
          <div
            className="relative flex flex-col items-center justify-center py-10 transition-all duration-500"
            style={{
              background: hovered === 'prime'
                ? 'linear-gradient(180deg, rgba(255,128,0,0.12) 0%, rgba(255,128,0,0.04) 100%)'
                : 'linear-gradient(180deg, rgba(255,128,0,0.06) 0%, transparent 100%)',
              borderBottom: '1px solid rgba(255,128,0,0.08)',
            }}
          >
            <div
              className="w-20 h-20 rounded-2xl flex items-center justify-center mb-4 transition-all duration-500"
              style={{
                background: hovered === 'prime' ? 'rgba(255,128,0,0.2)' : 'rgba(255,128,0,0.08)',
                boxShadow: hovered === 'prime' ? '0 0 40px rgba(255,128,0,0.2)' : 'none',
                border: '1px solid rgba(255,128,0,0.15)',
              }}
            >
              <FlaskConical className="w-9 h-9 text-[#FF8000]" />
            </div>

            <div className="flex items-center gap-2 mb-1">
              <div className="w-1.5 h-1.5 rounded-sm bg-[#FF8000]/50" />
              <span className="text-[9px] text-[#FF8000]/60 tracking-[0.25em] uppercase font-medium">Deep Analytics</span>
            </div>

            <h2 className="text-2xl font-bold text-white tracking-tight">Prime</h2>
          </div>

          {/* Body */}
          <div className="flex flex-col flex-1 p-6">
            <p className="text-xs text-white/35 leading-relaxed mb-6">
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
                  <div className="w-4 h-4 rounded-full bg-[#FF8000]/10 border border-[#FF8000]/20 flex items-center justify-center flex-shrink-0">
                    <Check className="w-2.5 h-2.5 text-[#FF8000]/70" />
                  </div>
                  <span className="text-[11px] text-white/40">{item}</span>
                </div>
              ))}
            </div>

            {/* Launch button */}
            <button
              onClick={() => onSelectPlatform('prime-driver')}
              className="mt-6 w-full flex items-center justify-center gap-2 py-3 rounded-xl text-xs font-semibold tracking-wide transition-all duration-300 cursor-pointer hover:brightness-110"
              style={{
                background: hovered === 'prime' ? '#FF8000' : 'rgba(255,128,0,0.08)',
                color: hovered === 'prime' ? '#0D1117' : 'rgba(255,128,0,0.5)',
                border: '1px solid rgba(255,128,0,0.15)',
              }}
            >
              Launch Prime
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
        <span className="text-[10px] text-white/10 tracking-[0.2em] uppercase">F1 OmniSense · Powered by Connectivia Labs</span>
      </div>
    </div>
  );
}
