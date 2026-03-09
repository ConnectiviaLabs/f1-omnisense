import { useMemo } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { ChevronUp, ChevronDown, Radio } from 'lucide-react';
import { ScrollArea } from './ui/scroll-area';
import type { OpenF1Driver, OpenF1Stint, OpenF1Position } from '../types';
import { COMPOUND_COLORS } from '../constants/teams';

// ─── Types ──────────────────────────────────────────────────────────
interface SectorTimeRow {
  driver_number: number;
  acronym: string;
  team_colour: string;
  s1: number | null;
  s2: number | null;
  s3: number | null;
  total: number | null;
}

interface SectorBests {
  overallS1: number;
  overallS2: number;
  overallS3: number;
  personal: Map<number, { s1: number; s2: number; s3: number }>;
}

interface RaceControlEvent {
  date: string;
  category: string;
  flag: string | null;
  message: string;
  lap_number?: number | null;
}

interface GapDataPoint {
  lap: number;
  [driverKey: string]: number | null;
}

interface LiveAnalyticsSidebarProps {
  positions: OpenF1Position[] | null;
  sectorTimesData: SectorTimeRow[];
  sectorBests: SectorBests;
  stints: OpenF1Stint[] | null;
  raceControlEvents: RaceControlEvent[];
  gapChartData: GapDataPoint[];
  fastestLap: { driver_number: number; lap_number: number; lap_duration: number } | null;
  driverMap: Map<number, OpenF1Driver>;
  positionDeltas: Map<number, number>;
  currentMaxLap: number;
  replayActive: boolean;
  replayTime: number;
  height?: number;
}

function sectorColor(value: number | null, overallBest: number, personalBest: number): string {
  if (!value) return 'text-muted-foreground';
  if (value <= overallBest + 0.001) return 'text-purple-400';
  if (value <= personalBest + 0.001) return 'text-green-400';
  return 'text-foreground';
}

function flagDotColor(flag: string | null): string {
  if (flag === 'RED') return '#ef4444';
  if (flag === 'YELLOW' || flag === 'DOUBLE YELLOW') return '#f59e0b';
  if (flag === 'GREEN') return '#22c55e';
  return '#8888a0';
}

export function LiveAnalyticsSidebar({
  positions,
  sectorTimesData,
  sectorBests,
  stints,
  raceControlEvents,
  gapChartData,
  fastestLap,
  driverMap,
  positionDeltas,
  currentMaxLap,
  replayActive,
  replayTime,
  height = 550,
}: LiveAnalyticsSidebarProps) {
  // Current tire compound per driver (latest stint)
  const currentCompounds = useMemo(() => {
    const map = new Map<number, string>();
    if (!stints) return map;
    for (const s of stints) {
      map.set(s.driver_number, s.compound);
    }
    return map;
  }, [stints]);

  // Gap to leader values from latest lap of gapChartData
  const latestGaps = useMemo(() => {
    if (!gapChartData.length || !positions) return [];
    const latest = gapChartData[gapChartData.length - 1];
    const gaps: { code: string; gap: number; color: string }[] = [];
    for (const pos of positions.slice(0, 10)) {
      const driver = driverMap.get(pos.driver_number);
      const code = driver?.name_acronym || `D${pos.driver_number}`;
      const gapVal = latest[code];
      if (typeof gapVal === 'number') {
        gaps.push({ code, gap: gapVal, color: `#${driver?.team_colour ?? '555555'}` });
      }
    }
    return gaps;
  }, [gapChartData, positions, driverMap]);

  const maxGap = useMemo(() => Math.max(...latestGaps.map(g => g.gap), 1), [latestGaps]);

  return (
    <div
      className="bg-card border border-[rgba(255,128,0,0.20)] rounded-lg shadow-[0_4px_12px_rgba(0,0,0,0.4)] overflow-hidden flex flex-col"
      style={{ height }}
    >
      {/* Header */}
      <div className="px-3 py-2 border-b border-border flex items-center gap-2 shrink-0">
        <Radio className="w-3 h-3 text-primary" />
        <span className="text-[12px] tracking-widest text-muted-foreground font-bold">LIVE ANALYTICS</span>
        {currentMaxLap > 0 && (
          <span className="text-[12px] text-primary font-mono ml-auto">LAP {currentMaxLap}</span>
        )}
      </div>

      <ScrollArea className="flex-1">
        <div className="p-2 space-y-1">

          {/* ── LIVE POSITIONS ── */}
          <div className="mb-2">
            <div className="text-[10px] tracking-widest text-muted-foreground font-bold px-1 mb-1">POSITIONS</div>
            <div className="space-y-0">
              <AnimatePresence mode="popLayout">
                {positions?.slice(0, 20).map((pos) => {
                  const driver = driverMap.get(pos.driver_number);
                  const delta = positionDeltas.get(pos.driver_number) ?? 0;
                  const hasFastestLap = fastestLap?.driver_number === pos.driver_number;
                  const compound = currentCompounds.get(pos.driver_number);
                  return (
                    <motion.div
                      key={pos.driver_number}
                      layout
                      initial={{ opacity: 0, x: -8 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ layout: { duration: 0.3 } }}
                      className="flex items-center gap-1.5 px-1 py-[3px] rounded hover:bg-secondary transition-colors"
                    >
                      <span className="text-[12px] font-mono text-foreground w-5 text-right">P{pos.position}</span>
                      <div className="w-0.5 h-3.5 rounded-full" style={{ backgroundColor: driver ? `#${driver.team_colour}` : '#555' }} />
                      <span className="text-[12px] font-mono text-foreground w-8">{driver?.name_acronym ?? `D${pos.driver_number}`}</span>
                      {/* Delta arrow */}
                      <span className="w-3 flex items-center justify-center">
                        {delta > 0 && <ChevronUp className="w-2.5 h-2.5 text-green-400" />}
                        {delta < 0 && <ChevronDown className="w-2.5 h-2.5 text-red-400" />}
                      </span>
                      {/* Fastest lap badge */}
                      {hasFastestLap && (
                        <span className="text-[10px] font-bold text-purple-400 bg-purple-400/10 px-1 rounded">FL</span>
                      )}
                      {/* Tire compound dot */}
                      {compound && (
                        <div
                          className="w-2 h-2 rounded-full ml-auto shrink-0"
                          style={{ backgroundColor: COMPOUND_COLORS[compound] ?? '#888' }}
                          title={compound}
                        />
                      )}
                    </motion.div>
                  );
                })}
              </AnimatePresence>
              {(!positions || positions.length === 0) && (
                <div className="text-center text-muted-foreground text-[12px] py-3">
                  {replayActive && replayTime === 0 ? 'Press play' : 'No data'}
                </div>
              )}
            </div>
          </div>

          {/* ── Divider ── */}
          <div className="border-t border-border" />

          {/* ── SECTOR TIMES ── */}
          <div className="mb-2">
            <div className="flex items-center justify-between px-1 mb-1">
              <span className="text-[10px] tracking-widest text-muted-foreground font-bold">SECTOR TIMES</span>
              <span className="text-[10px] text-muted-foreground font-mono">L{currentMaxLap}</span>
            </div>
            {sectorTimesData.length > 0 ? (
              <div className="space-y-0">
                {/* Header */}
                <div className="grid grid-cols-[40px_1fr_1fr_1fr] gap-0.5 text-[10px] text-muted-foreground px-1 pb-0.5 border-b border-[rgba(255,128,0,0.04)]">
                  <span>DRV</span>
                  <span className="text-right">S1</span>
                  <span className="text-right">S2</span>
                  <span className="text-right">S3</span>
                </div>
                {sectorTimesData.slice(0, 10).map((row) => {
                  const pb = sectorBests.personal.get(row.driver_number) ?? { s1: Infinity, s2: Infinity, s3: Infinity };
                  return (
                    <div key={row.driver_number} className="grid grid-cols-[40px_1fr_1fr_1fr] gap-0.5 text-[11px] font-mono px-1 py-[2px] hover:bg-secondary rounded transition-colors">
                      <span className="flex items-center gap-1">
                        <span className="w-0.5 h-2.5 rounded-full" style={{ backgroundColor: `#${row.team_colour}` }} />
                        <span className="text-foreground">{row.acronym}</span>
                      </span>
                      <span className={`text-right ${sectorColor(row.s1, sectorBests.overallS1, pb.s1)}`}>
                        {row.s1?.toFixed(3) ?? '—'}
                      </span>
                      <span className={`text-right ${sectorColor(row.s2, sectorBests.overallS2, pb.s2)}`}>
                        {row.s2?.toFixed(3) ?? '—'}
                      </span>
                      <span className={`text-right ${sectorColor(row.s3, sectorBests.overallS3, pb.s3)}`}>
                        {row.s3?.toFixed(3) ?? '—'}
                      </span>
                    </div>
                  );
                })}
                {/* Legend */}
                <div className="flex items-center gap-3 px-1 pt-1 text-[10px]">
                  <span className="flex items-center gap-0.5"><span className="w-1.5 h-1.5 rounded-full bg-purple-400" /> Best</span>
                  <span className="flex items-center gap-0.5"><span className="w-1.5 h-1.5 rounded-full bg-green-400" /> PB</span>
                </div>
              </div>
            ) : (
              <div className="text-center text-muted-foreground text-[11px] py-2">No sector data</div>
            )}
          </div>

          {/* ── Divider ── */}
          <div className="border-t border-border" />

          {/* ── GAP TO LEADER ── */}
          <div className="mb-2">
            <div className="text-[10px] tracking-widest text-muted-foreground font-bold px-1 mb-1">GAP TO LEADER</div>
            {latestGaps.length > 0 ? (
              <div className="space-y-0.5 px-1">
                {latestGaps.slice(0, 8).map((g) => (
                  <div key={g.code} className="flex items-center gap-1.5">
                    <span className="text-[11px] font-mono text-foreground w-7">{g.code}</span>
                    <div className="flex-1 h-2 bg-secondary rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all duration-500"
                        style={{
                          width: `${Math.max(2, (g.gap / maxGap) * 100)}%`,
                          backgroundColor: g.color,
                          opacity: 0.7,
                        }}
                      />
                    </div>
                    <span className="text-[10px] font-mono text-muted-foreground w-12 text-right">
                      {g.gap === 0 ? 'LEADER' : `+${g.gap.toFixed(1)}s`}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center text-muted-foreground text-[11px] py-2">No gap data</div>
            )}
          </div>

          {/* ── Divider ── */}
          <div className="border-t border-border" />

          {/* ── TIRE COMPOUNDS ── */}
          <div className="mb-2">
            <div className="text-[10px] tracking-widest text-muted-foreground font-bold px-1 mb-1">TIRES</div>
            {currentCompounds.size > 0 && positions ? (
              <div className="grid grid-cols-4 gap-x-1 gap-y-0.5 px-1">
                {positions.slice(0, 20).map((pos) => {
                  const driver = driverMap.get(pos.driver_number);
                  const compound = currentCompounds.get(pos.driver_number);
                  return (
                    <div key={pos.driver_number} className="flex items-center gap-1">
                      <div
                        className="w-2 h-2 rounded-full shrink-0"
                        style={{ backgroundColor: COMPOUND_COLORS[compound ?? ''] ?? '#555' }}
                      />
                      <span className="text-[10px] font-mono text-muted-foreground truncate">
                        {driver?.name_acronym ?? `D${pos.driver_number}`}
                      </span>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="text-center text-muted-foreground text-[11px] py-2">No tire data</div>
            )}
          </div>

          {/* ── Divider ── */}
          <div className="border-t border-border" />

          {/* ── RACE CONTROL ── */}
          <div>
            <div className="text-[10px] tracking-widest text-muted-foreground font-bold px-1 mb-1">RACE CONTROL</div>
            {raceControlEvents.length > 0 ? (
              <div className="space-y-0.5 px-1">
                {raceControlEvents.slice(-6).reverse().map((evt, i) => (
                  <div key={i} className="flex items-start gap-1.5 py-[2px]">
                    <div
                      className="w-1.5 h-1.5 rounded-full mt-1 shrink-0"
                      style={{ backgroundColor: flagDotColor(evt.flag) }}
                    />
                    <div className="min-w-0 flex-1">
                      <span className="text-[10px] text-foreground leading-tight line-clamp-1">{evt.message}</span>
                      {evt.lap_number && (
                        <span className="text-[10px] text-muted-foreground ml-1">L{evt.lap_number}</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center text-muted-foreground text-[11px] py-2">No events</div>
            )}
          </div>

        </div>
      </ScrollArea>
    </div>
  );
}
