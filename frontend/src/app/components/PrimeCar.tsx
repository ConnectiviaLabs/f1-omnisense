import { FleetOverview } from './FleetOverview';
import type { VehicleData } from './anomalyHelpers';
import type { Pillar } from './Sidebar';

interface PrimeCarProps {
  prefetchedVehicles: VehicleData[];
  prefetchLoading: boolean;
  activePillar: Pillar;
}

export function PrimeCar({ prefetchedVehicles, prefetchLoading, activePillar }: PrimeCarProps) {
  return (
    <div className="space-y-4">
      <FleetOverview
        prefetchedVehicles={prefetchedVehicles}
        prefetchLoading={prefetchLoading}
        defaultSection={activePillar}
      />
    </div>
  );
}
