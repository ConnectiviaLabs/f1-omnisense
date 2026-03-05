import { useState } from 'react';
import { Activity, Radio, FlaskConical, ArrowRight, Zap, BarChart3, Brain, Shield } from 'lucide-react';
import type { ViewType } from '../types';

interface HomeProps {
  onSelectPlatform: (view: ViewType) => void;
}

export function Home_v1({ onSelectPlatform }: HomeProps) {
  const [hovered, setHovered] = useState<'raceday' | 'prime' | null>(null);

  return (
    <div className="relative h-full flex flex-col bg-[#0D1117] overflow-hidden">

      {/* Subtle grid background */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          backgroundImage: `
            linear-gradient(rgba(255,128,0,0.04) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,128,0,0.04) 1px, transparent 1px)
          `,
          backgroundSize: '48px 48px',
        }}
      />

      {/* Top glow */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[600px] h-[200px] bg-[#FF8000]/5 rounded-full blur-3xl pointer-events-none" />

      {/* Header */}
      <div className="relative z-10 flex flex-col items-center pt-12 pb-8">
        <div className="flex items-center gap-4 mb-3">
          <div className="w-11 h-11 rounded-xl bg-[#FF8000] flex items-center justify-center shadow-[0_0_24px_rgba(255,128,0,0.4)]">
            <Activity className="w-5 h-5 text-[#0D1117]" />
          </div>
          <div>
            <div className="text-[#FF8000] text-2xl font-bold tracking-[0.2em]">F1 OMNISENSE</div>
            <div className="text-xs text-[#FF8000]/40 tracking-[0.15em] uppercase">Powered by Connectivia Labs</div>
          </div>
        </div>
        <div className="flex items-center gap-2 mt-1">
          <div className="h-px w-12 bg-gradient-to-r from-transparent to-[#FF8000]/30" />
          <span className="text-xs text-white/20 tracking-[0.25em] uppercase">Select Platform</span>
          <div className="h-px w-12 bg-gradient-to-l from-transparent to-[#FF8000]/30" />
        </div>
      </div>

      {/* Split panels */}
      <div className="relative z-10 flex flex-1 gap-px mx-8 mb-8 rounded-2xl overflow-hidden border border-[#FF8000]/10">

        {/* Race Day */}
        <button
          onClick={() => onSelectPlatform('dashboard')}
          onMouseEnter={() => setHovered('raceday')}
          onMouseLeave={() => setHovered(null)}
          className="relative flex-1 flex flex-col justify-between p-10 text-left bg-[#111827] transition-all duration-500 overflow-hidden group"
          style={{
            background: hovered === 'raceday'
              ? 'linear-gradient(135deg, #131C2E 0%, #0D1117 100%)'
              : '#111827',
          }}
        >
          <div
            className="absolute inset-0 pointer-events-none transition-opacity duration-500"
            style={{ opacity: hovered === 'raceday' ? 1 : 0, background: 'radial-gradient(ellipse at 20% 50%, rgba(255,128,0,0.07) 0%, transparent 70%)' }}
          />
          <Radio
            className="absolute -bottom-8 -right-8 text-[#FF8000]/5 transition-all duration-500 group-hover:text-[#FF8000]/10 group-hover:-bottom-4 group-hover:-right-4"
            style={{ width: 180, height: 180 }}
          />

          <div className="relative">
            <div className="inline-flex items-center gap-1.5 mb-8">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#FF8000] opacity-75" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-[#FF8000]" />
              </span>
              <span className="text-[10px] text-[#FF8000]/70 tracking-[0.2em] uppercase font-medium">Live</span>
            </div>

            <div className="w-16 h-16 rounded-2xl bg-[#FF8000]/10 border border-[#FF8000]/15 flex items-center justify-center mb-6 transition-all duration-300 group-hover:bg-[#FF8000]/20 group-hover:border-[#FF8000]/30 group-hover:shadow-[0_0_32px_rgba(255,128,0,0.15)]">
              <Radio className="w-7 h-7 text-[#FF8000]" />
            </div>

            <h2 className="text-3xl font-bold text-white mb-3 tracking-tight">Race Day</h2>
            <p className="text-sm text-white/40 leading-relaxed max-w-xs">
              Real-time telemetry, live timing, driver biometrics & on-track intelligence for every session.
            </p>
          </div>

          <div className="relative">
            <div className="h-px bg-[#FF8000]/8 mb-6" />
            <div className="flex flex-col gap-3 mb-8">
              {[
                { icon: Zap, label: 'Live Dashboard', sub: 'Real-time race positions & gaps' },
                { icon: BarChart3, label: 'Car Telemetry', sub: 'Speed, RPM, throttle & DRS' },
                { icon: Activity, label: 'Driver Biometrics', sub: 'Heart rate & cockpit data' },
              ].map(({ icon: Icon, label, sub }) => (
                <div key={label} className="flex items-center gap-3">
                  <div className="w-7 h-7 rounded-lg bg-[#FF8000]/8 flex items-center justify-center flex-shrink-0">
                    <Icon className="w-3.5 h-3.5 text-[#FF8000]/60" />
                  </div>
                  <div>
                    <div className="text-xs font-medium text-white/70">{label}</div>
                    <div className="text-[10px] text-white/25">{sub}</div>
                  </div>
                </div>
              ))}
            </div>

            <div className="flex items-center gap-2 text-[#FF8000]/60 text-xs font-medium tracking-wide group-hover:text-[#FF8000] transition-colors duration-300">
              <span>Enter Race Day</span>
              <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform duration-300" />
            </div>
          </div>
        </button>

        <div className="w-px bg-[#FF8000]/10 flex-shrink-0" />

        {/* Prime */}
        <button
          onClick={() => onSelectPlatform('prime-driver')}
          onMouseEnter={() => setHovered('prime')}
          onMouseLeave={() => setHovered(null)}
          className="relative flex-1 flex flex-col justify-between p-10 text-left bg-[#111827] transition-all duration-500 overflow-hidden group"
          style={{
            background: hovered === 'prime'
              ? 'linear-gradient(135deg, #131C2E 0%, #0D1117 100%)'
              : '#111827',
          }}
        >
          <div
            className="absolute inset-0 pointer-events-none transition-opacity duration-500"
            style={{ opacity: hovered === 'prime' ? 1 : 0, background: 'radial-gradient(ellipse at 80% 50%, rgba(255,128,0,0.07) 0%, transparent 70%)' }}
          />
          <FlaskConical
            className="absolute -bottom-8 -right-8 text-[#FF8000]/5 transition-all duration-500 group-hover:text-[#FF8000]/10 group-hover:-bottom-4 group-hover:-right-4"
            style={{ width: 180, height: 180 }}
          />

          <div className="relative">
            <div className="inline-flex items-center gap-1.5 mb-8">
              <div className="w-2 h-2 rounded-sm bg-[#FF8000]/50" />
              <span className="text-[10px] text-[#FF8000]/70 tracking-[0.2em] uppercase font-medium">Analytics</span>
            </div>

            <div className="w-16 h-16 rounded-2xl bg-[#FF8000]/10 border border-[#FF8000]/15 flex items-center justify-center mb-6 transition-all duration-300 group-hover:bg-[#FF8000]/20 group-hover:border-[#FF8000]/30 group-hover:shadow-[0_0_32px_rgba(255,128,0,0.15)]">
              <FlaskConical className="w-7 h-7 text-[#FF8000]" />
            </div>

            <h2 className="text-3xl font-bold text-white mb-3 tracking-tight">Prime</h2>
            <p className="text-sm text-white/40 leading-relaxed max-w-xs">
              Offline driver analysis, car health monitoring, predictive maintenance & race strategy simulation.
            </p>
          </div>

          <div className="relative">
            <div className="h-px bg-[#FF8000]/8 mb-6" />
            <div className="flex flex-col gap-3 mb-8">
              {[
                { icon: Brain, label: 'Driver Intelligence', sub: 'Career stats & head-to-head' },
                { icon: Shield, label: 'Fleet Health', sub: 'Predictive maintenance & risk' },
                { icon: BarChart3, label: 'Race Strategy', sub: 'Tyre deg & pit windows' },
              ].map(({ icon: Icon, label, sub }) => (
                <div key={label} className="flex items-center gap-3">
                  <div className="w-7 h-7 rounded-lg bg-[#FF8000]/8 flex items-center justify-center flex-shrink-0">
                    <Icon className="w-3.5 h-3.5 text-[#FF8000]/60" />
                  </div>
                  <div>
                    <div className="text-xs font-medium text-white/70">{label}</div>
                    <div className="text-[10px] text-white/25">{sub}</div>
                  </div>
                </div>
              ))}
            </div>

            <div className="flex items-center gap-2 text-[#FF8000]/60 text-xs font-medium tracking-wide group-hover:text-[#FF8000] transition-colors duration-300">
              <span>Enter Prime</span>
              <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform duration-300" />
            </div>
          </div>
        </button>
      </div>

      <div className="relative z-10 flex justify-center pb-6">
        <span className="text-[10px] text-white/10 tracking-[0.2em] uppercase">F1 OmniSense · All rights reserved</span>
      </div>
    </div>
  );
}
