import { FleetOverview } from './FleetOverview';
import type { VehicleData, FeatureForecast } from './anomalyHelpers';
import type { Pillar } from './Sidebar';

interface PrimeCarProps {
  prefetchedVehicles: VehicleData[];
  prefetchedForecasts: Record<string, FeatureForecast[]>;
  prefetchLoading: boolean;
  activePillar: Pillar;
}

export function PrimeCar({ prefetchedVehicles, prefetchedForecasts, prefetchLoading, activePillar }: PrimeCarProps) {
  return (
    <div className="space-y-4">
      {/* Phase 2 will break FleetOverview into separate focused components per pillar. */}
      <FleetOverview
        prefetchedVehicles={prefetchedVehicles}
        prefetchedForecasts={prefetchedForecasts}
        prefetchLoading={prefetchLoading}
        defaultSection={activePillar}
      />
    </div>
  );
}
