import { Activity, Radio, FlaskConical } from 'lucide-react';
import type { ViewType } from '../types';

interface HomeProps {
  onSelectPlatform: (view: ViewType) => void;
}

export function Home({ onSelectPlatform }: HomeProps) {
  return (
    <div className="h-full flex flex-col items-center justify-center bg-[#0D1117] px-8">
      {/* Branding */}
      <div className="flex items-center gap-3 mb-2">
        <div className="w-12 h-12 rounded-xl bg-[#FF8000] flex items-center justify-center">
          <Activity className="w-6 h-6 text-[#0D1117]" />
        </div>
        <div>
          <div className="text-[#FF8000] text-2xl font-semibold tracking-[0.15em]">F1 OMNISENSE</div>
          <div className="text-sm text-muted-foreground tracking-wider">Powered by Connectivia Labs</div>
        </div>
      </div>

      <p className="text-muted-foreground text-sm mb-12 tracking-wide">Select your platform</p>

      {/* Platform Cards */}
      <div className="flex gap-6 max-w-3xl w-full">
        {/* Race Day */}
        <button
          onClick={() => onSelectPlatform('dashboard')}
          className="flex-1 group rounded-2xl border border-[rgba(255,128,0,0.15)] bg-[#1A1F2E] p-8 text-left transition-all hover:border-[#FF8000]/40 hover:bg-[#1A1F2E]/80 hover:shadow-[0_0_40px_rgba(255,128,0,0.08)]"
        >
          <div className="w-14 h-14 rounded-xl bg-[#FF8000]/10 flex items-center justify-center mb-5 group-hover:bg-[#FF8000]/20 transition-colors">
            <Radio className="w-7 h-7 text-[#FF8000]" />
          </div>
          <h2 className="text-xl font-semibold text-foreground mb-2 tracking-tight">Race Day</h2>
          <p className="text-sm text-muted-foreground leading-relaxed">
            Real-time telemetry, live timing, driver biometrics & on-track intelligence
          </p>
          <div className="mt-6 flex flex-wrap gap-2">
            {['Live Dashboard', 'Car Telemetry', 'Biometrics'].map(tag => (
              <span key={tag} className="text-[10px] px-2 py-0.5 rounded-full bg-[#FF8000]/8 text-[#FF8000]/70 border border-[#FF8000]/10">
                {tag}
              </span>
            ))}
          </div>
        </button>

        {/* Prime */}
        <button
          onClick={() => onSelectPlatform('prime-driver')}
          className="flex-1 group rounded-2xl border border-[rgba(255,128,0,0.15)] bg-[#1A1F2E] p-8 text-left transition-all hover:border-[#FF8000]/40 hover:bg-[#1A1F2E]/80 hover:shadow-[0_0_40px_rgba(255,128,0,0.08)]"
        >
          <div className="w-14 h-14 rounded-xl bg-[#FF8000]/10 flex items-center justify-center mb-5 group-hover:bg-[#FF8000]/20 transition-colors">
            <FlaskConical className="w-7 h-7 text-[#FF8000]" />
          </div>
          <h2 className="text-xl font-semibold text-foreground mb-2 tracking-tight">Prime</h2>
          <p className="text-sm text-muted-foreground leading-relaxed">
            Offline driver analysis, car health monitoring, predictive maintenance & race strategy
          </p>
          <div className="mt-6 flex flex-wrap gap-2">
            {['Driver Intel', 'Fleet Health', 'Strategy'].map(tag => (
              <span key={tag} className="text-[10px] px-2 py-0.5 rounded-full bg-[#FF8000]/8 text-[#FF8000]/70 border border-[#FF8000]/10">
                {tag}
              </span>
            ))}
          </div>
        </button>
      </div>
    </div>
  );
}
