import { useState } from 'react';
import { Activity, Radio, FlaskConical, ArrowRight, Zap, BarChart3, Brain, Shield, MapPin, Users } from 'lucide-react';
import type { ViewType } from '../types';

interface HomeProps {
  onSelectPlatform: (view: ViewType) => void;
}

const platforms = [
  {
    id: 'raceday' as const,
    view: 'dashboard' as ViewType,
    icon: Radio,
    badge: 'Live',
    badgeLive: true,
    title: 'Race Day',
    description: 'Real-time telemetry, live timing, driver biometrics & on-track intelligence for every session.',
    features: [
      { icon: Zap, label: 'Live Dashboard' },
      { icon: BarChart3, label: 'Car Telemetry' },
      { icon: Activity, label: 'Biometrics' },
      { icon: MapPin, label: 'Circuit Intel' },
    ],
    cta: 'Enter Race Day',
  },
  {
    id: 'prime' as const,
    view: 'prime-driver' as ViewType,
    icon: FlaskConical,
    badge: 'Analytics',
    badgeLive: false,
    title: 'Prime',
    description: 'Offline driver analysis, car health monitoring, predictive maintenance & race strategy simulation.',
    features: [
      { icon: Brain, label: 'Driver Intel' },
      { icon: Shield, label: 'Fleet Health' },
      { icon: BarChart3, label: 'Strategy' },
      { icon: Users, label: 'McLaren' },
    ],
    cta: 'Enter Prime',
  },
];

export function Home_v4({ onSelectPlatform }: HomeProps) {
  const [hovered, setHovered] = useState<'raceday' | 'prime' | null>(null);

  return (
    <div className="relative h-full flex flex-col items-center bg-background overflow-hidden px-16 pt-12">

      {/* Horizontal scan lines */}
      <div
        className="absolute inset-0 pointer-events-none opacity-40"
        style={{
          backgroundImage: 'repeating-linear-gradient(0deg, transparent, transparent 3px, rgba(255,128,0,0.012) 3px, rgba(255,128,0,0.012) 4px)',
        }}
      />

      {/* Side glows */}
      <div className="absolute left-0 top-1/2 -translate-y-1/2 w-64 h-[600px] bg-[#FF8000]/5 blur-[80px] pointer-events-none rounded-full" />
      <div className="absolute right-0 top-1/2 -translate-y-1/2 w-64 h-[600px] bg-[#FF8000]/5 blur-[80px] pointer-events-none rounded-full" />

      {/* Header */}
      <div className="relative z-10 flex flex-col items-center mb-12">
        <div className="flex items-center gap-3 mb-1.5">
          <div className="w-10 h-10 rounded-lg bg-[#FF8000] flex items-center justify-center shadow-[0_0_24px_rgba(255,128,0,0.45)]">
            <Activity className="w-5 h-5 text-[#0D1117]" />
          </div>
          <span className="text-[#FF8000] text-2xl font-bold tracking-[0.2em]">F1 OMNISENSE</span>
        </div>
        <p className="text-[11px] text-[#FF8000]/40 tracking-[0.15em] uppercase mb-1">Powered by Connectivia Labs</p>
        <p className="text-[10px] text-white/15 tracking-[0.3em] uppercase">Select your platform</p>
      </div>

      {/* Horizontal strips */}
      <div className="relative z-10 flex flex-col gap-3 w-full max-w-5xl">
        {platforms.map(({ id, view, icon: Icon, badge, badgeLive, title, description, features, cta }) => {
          const isHovered = hovered === id;
          return (
            <button
              key={id}
              onClick={() => onSelectPlatform(view)}
              onMouseEnter={() => setHovered(id)}
              onMouseLeave={() => setHovered(null)}
              className="group relative w-full flex items-center gap-8 px-8 py-6 rounded-2xl text-left overflow-hidden transition-all duration-400 border"
              style={{
                background: isHovered
                  ? 'linear-gradient(90deg, #131C2E 0%, #111827 60%, #0F1520 100%)'
                  : '#111827',
                borderColor: isHovered ? 'rgba(255,128,0,0.3)' : 'rgba(255,128,0,0.08)',
                boxShadow: isHovered ? '0 0 50px rgba(255,128,0,0.08)' : 'none',
              }}
            >
              {/* Hover left-edge accent */}
              <div
                className="absolute left-0 top-0 bottom-0 w-0.5 rounded-full transition-opacity duration-300"
                style={{
                  background: 'linear-gradient(180deg, transparent, #FF8000, transparent)',
                  opacity: isHovered ? 1 : 0,
                }}
              />

              {/* Icon */}
              <div
                className="flex-shrink-0 w-14 h-14 rounded-lg flex items-center justify-center transition-all duration-300"
                style={{
                  background: isHovered ? 'rgba(255,128,0,0.18)' : 'rgba(255,128,0,0.08)',
                  border: '1px solid rgba(255,128,0,0.15)',
                  boxShadow: isHovered ? '0 0 28px rgba(255,128,0,0.2)' : 'none',
                }}
              >
                <Icon className="w-6 h-6 text-[#FF8000]" />
              </div>

              {/* Title + description */}
              <div className="flex-shrink-0 w-64">
                <div className="flex items-center gap-2 mb-1.5">
                  {badgeLive ? (
                    <div className="flex items-center gap-1.5">
                      <span className="relative flex h-1.5 w-1.5">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#FF8000] opacity-75" />
                        <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-[#FF8000]" />
                      </span>
                      <span className="text-[9px] text-[#FF8000]/60 tracking-[0.2em] uppercase font-medium">{badge}</span>
                    </div>
                  ) : (
                    <div className="flex items-center gap-1.5">
                      <div className="w-1.5 h-1.5 rounded-sm bg-[#FF8000]/40" />
                      <span className="text-[9px] text-[#FF8000]/60 tracking-[0.2em] uppercase font-medium">{badge}</span>
                    </div>
                  )}
                </div>
                <h2 className="text-xl font-bold text-white tracking-tight mb-1.5">{title}</h2>
                <p className="text-[11px] text-white/30 leading-relaxed">{description}</p>
              </div>

              {/* Vertical divider */}
              <div className="flex-shrink-0 w-px self-stretch bg-[#FF8000]/8 mx-2" />

              {/* Feature chips */}
              <div className="flex-1 flex items-center gap-3 flex-wrap">
                {features.map(({ icon: FIcon, label }) => (
                  <div
                    key={label}
                    className="flex items-center gap-2 px-3 py-2 rounded-lg transition-all duration-300"
                    style={{
                      background: isHovered ? 'rgba(255,128,0,0.1)' : 'rgba(255,128,0,0.05)',
                      border: '1px solid rgba(255,128,0,0.1)',
                    }}
                  >
                    <FIcon className="w-3 h-3 text-[#FF8000]/60 flex-shrink-0" />
                    <span className="text-[11px] text-white/50 font-medium">{label}</span>
                  </div>
                ))}
              </div>

              {/* CTA */}
              <div
                className="flex-shrink-0 flex items-center gap-2 text-xs font-semibold tracking-wide transition-all duration-300 pl-4"
                style={{ color: isHovered ? '#FF8000' : 'rgba(255,128,0,0.35)' }}
              >
                <span className="whitespace-nowrap">{cta}</span>
                <ArrowRight
                  className="w-3.5 h-3.5 transition-transform duration-300"
                  style={{ transform: isHovered ? 'translateX(4px)' : 'translateX(0)' }}
                />
              </div>
            </button>
          );
        })}
      </div>

      {/* Footer */}
      <div className="relative z-10 mt-10">
        <span className="text-[10px] text-white/10 tracking-[0.2em] uppercase">F1 OmniSense · All rights reserved</span>
      </div>
    </div>
  );
}
