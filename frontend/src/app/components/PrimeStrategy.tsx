import { RaceStrategy } from './RaceStrategy';
import { CircuitIntel } from './CircuitIntel';
import { McLarenAnalytics } from './McLarenAnalytics';
import type { StrategyTab } from './Sidebar';

interface PrimeStrategyProps {
  activeTab: StrategyTab;
}

export function PrimeStrategy({ activeTab }: PrimeStrategyProps) {
  return (
    <div className="space-y-4">
      {activeTab === 'race-strategy' && <RaceStrategy />}
      {activeTab === 'circuit-intel' && <CircuitIntel />}
      {activeTab === 'season-analytics' && <McLarenAnalytics />}
    </div>
  );
}
