import { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { MapPin, Maximize2 } from 'lucide-react';
import type { TurnInfo, DRSZone } from '../data/circuits';

// ─── Types ──────────────────────────────────────────────────────────
interface GeoJSONFeature {
  type: string;
  properties: { Name: string; [k: string]: any };
  geometry: { type: string; coordinates: [number, number][] };
}

export interface CarPosition {
  driverNumber: number;
  code: string;
  x: number; // normalised 0-1
  y: number; // normalised 0-1
  color: string;
}

type ColorMode = 'accent' | 'sectors' | 'speed';

interface TrackMapProps {
  geojsonPath: string;
  circuitName: string;
  locality: string;
  country: string;
  lengthKm: number;
  cars?: CarPosition[];
  accentColor?: string;
  className?: string;
  height?: number;
  turns?: TurnInfo[];
  drsZones?: DRSZone[];
  colorMode?: ColorMode;
}

interface Point { x: number; y: number; }

const SECTOR_COLORS = ['#e74c3c', '#3498db', '#f1c40f'];
const DRS_COLOR = '#00ff88';
const KERB_RED = '#e74c3c';
const KERB_WHITE = '#f0f0f0';

// ─── Helpers ────────────────────────────────────────────────────────
function projectCoordinates(
  coords: [number, number][],
  padding: number,
  width: number,
  height: number,
): Point[] {
  const lngs = coords.map(c => c[0]);
  const lats = coords.map(c => c[1]);
  const minLng = Math.min(...lngs), maxLng = Math.max(...lngs);
  const minLat = Math.min(...lats), maxLat = Math.max(...lats);

  const geoW = maxLng - minLng || 0.001;
  const geoH = maxLat - minLat || 0.001;
  const drawW = width - padding * 2;
  const drawH = height - padding * 2;

  const scale = Math.min(drawW / geoW, drawH / geoH);
  const offsetX = padding + (drawW - geoW * scale) / 2;
  const offsetY = padding + (drawH - geoH * scale) / 2;

  return coords.map(([lng, lat]) => ({
    x: offsetX + (lng - minLng) * scale,
    y: offsetY + (maxLat - lat) * scale,
  }));
}

function buildPathD(points: Point[], close = true): string {
  if (points.length === 0) return '';
  return points
    .map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x.toFixed(1)},${p.y.toFixed(1)}`)
    .join(' ') + (close ? ' Z' : '');
}

/** Split projected points into 3 equal sectors */
function splitSectors(points: Point[]): Point[][] {
  const n = points.length;
  const third = Math.floor(n / 3);
  return [
    points.slice(0, third + 1),
    points.slice(third, third * 2 + 1),
    points.slice(third * 2),
  ];
}

/** Compute curvature at each point (angle change) for speed gradient */
function computeCurvature(points: Point[]): number[] {
  const curvatures: number[] = new Array(points.length).fill(0);
  for (let i = 1; i < points.length - 1; i++) {
    const prev = points[i - 1];
    const curr = points[i];
    const next = points[i + 1];
    const a1 = Math.atan2(curr.y - prev.y, curr.x - prev.x);
    const a2 = Math.atan2(next.y - curr.y, next.x - curr.x);
    let diff = Math.abs(a2 - a1);
    if (diff > Math.PI) diff = 2 * Math.PI - diff;
    curvatures[i] = diff;
  }
  return curvatures;
}

/** Map curvature to color: low (green/fast) → high (red/slow) */
function curvatureToColor(curvature: number, maxCurv: number): string {
  const t = Math.min(curvature / (maxCurv * 0.5), 1);
  if (t < 0.5) {
    const f = t * 2;
    const r = Math.round(f * 255);
    const g = 255;
    return `rgb(${r},${g},60)`;
  }
  const f = (t - 0.5) * 2;
  const r = 255;
  const g = Math.round((1 - f) * 255);
  return `rgb(${r},${g},60)`;
}

/** Compute perpendicular kerb markers at turn positions */
function computeKerbs(
  points: Point[],
  turnPositions: number[],    // normalised 0-1
  kerbLength: number = 10,
): { x1: number; y1: number; x2: number; y2: number; position: number }[] {
  const n = points.length;
  const kerbs: { x1: number; y1: number; x2: number; y2: number; position: number }[] = [];
  for (const pos of turnPositions) {
    const idx = Math.round(pos * (n - 1));
    const pt = points[Math.min(idx, n - 1)];
    const prev = points[Math.max(idx - 2, 0)];
    const next = points[Math.min(idx + 2, n - 1)];
    // Direction tangent
    const dx = next.x - prev.x;
    const dy = next.y - prev.y;
    const len = Math.sqrt(dx * dx + dy * dy) || 1;
    // Perpendicular
    const nx = -dy / len;
    const ny = dx / len;
    kerbs.push({
      x1: pt.x + nx * kerbLength,
      y1: pt.y + ny * kerbLength,
      x2: pt.x - nx * kerbLength,
      y2: pt.y - ny * kerbLength,
      position: pos,
    });
  }
  return kerbs;
}

/** Map speed (km/h) to color: slow = red, medium = amber, fast = green */
function speedToColor(speedKph: number): string {
  const t = Math.min(Math.max((speedKph - 50) / 270, 0), 1); // 50-320 range
  if (t < 0.33) {
    return '#ef4444'; // slow — red
  } else if (t < 0.66) {
    return '#f59e0b'; // medium — amber
  }
  return '#22c55e'; // fast — green
}

/** Extract a slice of points for a DRS zone (handles wrap-around) */
function extractDRSPath(points: Point[], start: number, end: number): Point[] {
  const n = points.length;
  const si = Math.round(start * (n - 1));
  const ei = Math.round(end * (n - 1));
  if (si <= ei) return points.slice(si, ei + 1);
  return [...points.slice(si), ...points.slice(0, ei + 1)];
}

// ─── Component ──────────────────────────────────────────────────────
export function TrackMap({
  geojsonPath,
  circuitName,
  locality,
  country,
  lengthKm,
  cars,
  accentColor = '#FF8000',
  className = '',
  height = 420,
  turns,
  drsZones,
  colorMode = 'sectors',
}: TrackMapProps) {
  const [coords, setCoords] = useState<[number, number][]>([]);
  const [loading, setLoading] = useState(true);
  const [hoveredTurn, setHoveredTurn] = useState<TurnInfo | null>(null);
  const [tooltipPos, setTooltipPos] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  // Zoom & pan state
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const panStart = useRef<{ x: number; y: number; panX: number; panY: number }>({ x: 0, y: 0, panX: 0, panY: 0 });
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setLoading(true);
    fetch(geojsonPath)
      .then(r => r.json())
      .then((data: { features: GeoJSONFeature[] }) => {
        const feature = data.features?.[0];
        if (feature?.geometry?.coordinates) {
          setCoords(feature.geometry.coordinates as [number, number][]);
        }
      })
      .catch(() => setCoords([]))
      .finally(() => setLoading(false));
  }, [geojsonPath]);

  // Reset zoom/pan when circuit changes
  useEffect(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, [geojsonPath]);

  const WIDTH = 500;
  const projected = useMemo(
    () => (coords.length > 0 ? projectCoordinates(coords, 16, WIDTH, height) : []),
    [coords, height],
  );

  const fullPathD = useMemo(() => buildPathD(projected), [projected]);

  // Sector paths
  const sectorPaths = useMemo(() => {
    if (projected.length === 0) return [];
    return splitSectors(projected).map(pts => buildPathD(pts, false));
  }, [projected]);

  // Sector midpoints (for labels)
  const sectorMidpoints = useMemo(() => {
    if (projected.length === 0) return [];
    return splitSectors(projected).map(pts => pts[Math.floor(pts.length / 2)]);
  }, [projected]);

  // Speed gradient segments (finer step for smoother rendering)
  const speedSegments = useMemo(() => {
    if (colorMode !== 'speed' || projected.length < 3) return [];
    const curvatures = computeCurvature(projected);
    const maxCurv = Math.max(...curvatures.filter(c => c > 0), 0.01);
    const segments: { d: string; color: string }[] = [];
    const step = 2; // finer steps for smoother gradient
    for (let i = 0; i < projected.length - step; i += step) {
      const slice = projected.slice(i, i + step + 1);
      // Smooth curvature with wider window
      const windowStart = Math.max(0, i - 2);
      const windowEnd = Math.min(curvatures.length, i + step + 3);
      const window = curvatures.slice(windowStart, windowEnd);
      const avgCurv = window.reduce((a, b) => a + b, 0) / window.length;
      segments.push({
        d: buildPathD(slice, false),
        color: curvatureToColor(avgCurv, maxCurv),
      });
    }
    return segments;
  }, [colorMode, projected]);

  // DRS zone paths
  const drsPaths = useMemo(() => {
    if (!drsZones || drsZones.length === 0 || projected.length === 0) return [];
    return drsZones.map(zone => {
      const pts = extractDRSPath(projected, zone.start, zone.end);
      const mid = pts[Math.floor(pts.length / 2)];
      return { d: buildPathD(pts, false), mid };
    });
  }, [drsZones, projected]);

  // Turn positions on track
  const turnPositions = useMemo(() => {
    if (!turns || turns.length === 0 || projected.length === 0) return [];
    return turns.map(t => {
      const idx = Math.round(t.position * (projected.length - 1));
      const pt = projected[Math.min(idx, projected.length - 1)];
      return { ...t, px: pt.x, py: pt.y };
    });
  }, [turns, projected]);

  // Kerb markers at turn apexes
  const kerbMarkers = useMemo(() => {
    if (!turns || turns.length === 0 || projected.length === 0) return [];
    return computeKerbs(projected, turns.map(t => t.position), 9);
  }, [turns, projected]);

  // Car position dots
  const carDots = useMemo(() => {
    if (!cars || cars.length === 0 || projected.length === 0) return [];
    return cars.map(car => {
      const idx = Math.round(car.x * (projected.length - 1));
      const pt = projected[Math.min(idx, projected.length - 1)];
      return { ...car, px: pt.x, py: pt.y };
    });
  }, [cars, projected]);

  // Zoom handlers
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    setZoom(z => Math.max(0.5, Math.min(5, z - e.deltaY * 0.002)));
  }, []);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0) return;
    setIsPanning(true);
    panStart.current = { x: e.clientX, y: e.clientY, panX: pan.x, panY: pan.y };
  }, [pan]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isPanning) return;
    setPan({
      x: panStart.current.panX + (e.clientX - panStart.current.x),
      y: panStart.current.panY + (e.clientY - panStart.current.y),
    });
  }, [isPanning]);

  const handleMouseUp = useCallback(() => setIsPanning(false), []);

  const resetView = useCallback(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, []);

  if (loading) {
    return (
      <div className={`flex items-center justify-center bg-[#0D1117] rounded-lg ${className}`} style={{ height }}>
        <span className="text-[12px] text-muted-foreground animate-pulse">Loading track...</span>
      </div>
    );
  }

  if (projected.length === 0) {
    return (
      <div className={`flex items-center justify-center bg-[#0D1117] rounded-lg ${className}`} style={{ height }}>
        <span className="text-[12px] text-muted-foreground">No track data</span>
      </div>
    );
  }

  return (
    <div className={`bg-[#0D1117] rounded-lg border border-[rgba(255,128,0,0.12)] overflow-hidden relative ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-[rgba(255,128,0,0.12)]">
        <div className="flex items-center gap-2">
          <MapPin className="w-3.5 h-3.5" style={{ color: accentColor }} />
          <span className="text-[11px] font-semibold text-foreground tracking-wide">{circuitName}</span>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
            <span>{locality}, {country}</span>
            <span className="font-mono">{lengthKm} km</span>
          </div>
          {/* Sector legend */}
          {colorMode === 'sectors' && (
            <div className="flex items-center gap-1.5 ml-2">
              {['S1', 'S2', 'S3'].map((s, i) => (
                <div key={s} className="flex items-center gap-0.5">
                  <div className="w-2 h-2 rounded-full" style={{ backgroundColor: SECTOR_COLORS[i] }} />
                  <span className="text-[10px] text-muted-foreground">{s}</span>
                </div>
              ))}
            </div>
          )}
          {/* Reset zoom button */}
          {zoom !== 1 && (
            <button
              onClick={resetView}
              className="flex items-center gap-1 px-1.5 py-0.5 text-[10px] text-muted-foreground bg-[rgba(255,255,255,0.05)] rounded hover:bg-[rgba(255,255,255,0.1)] transition-colors"
            >
              <Maximize2 className="w-2.5 h-2.5" />
              Reset
            </button>
          )}
        </div>
      </div>

      {/* SVG Track with zoom/pan */}
      <div
        ref={containerRef}
        className="overflow-hidden cursor-grab active:cursor-grabbing"
        style={{ height: height - 36 }}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        <svg
          viewBox={`0 0 ${WIDTH} ${height}`}
          className="w-full"
          style={{
            height: height - 36,
            transform: `scale(${zoom}) translate(${pan.x / zoom}px, ${pan.y / zoom}px)`,
            transformOrigin: 'center center',
            transition: isPanning ? 'none' : 'transform 0.15s ease-out',
          }}
        >
          <defs>
            <filter id="trackGlow">
              <feGaussianBlur stdDeviation="4" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
            {/* Checkered pattern for start/finish */}
            <pattern id="checkered" width="4" height="4" patternUnits="userSpaceOnUse">
              <rect width="2" height="2" fill="white" />
              <rect x="2" y="2" width="2" height="2" fill="white" />
              <rect x="2" width="2" height="2" fill="#333" />
              <rect y="2" width="2" height="2" fill="#333" />
            </pattern>
          </defs>

          {/* Asphalt base (dark road) */}
          <path
            d={fullPathD}
            fill="none"
            stroke="#222838"
            strokeWidth="14"
            strokeLinecap="round"
            strokeLinejoin="round"
          />

          {/* Track rendering based on color mode */}
          {colorMode === 'speed' ? (
            // Speed gradient mode
            <>
              {speedSegments.map((seg, i) => (
                <path
                  key={i}
                  d={seg.d}
                  fill="none"
                  stroke={seg.color}
                  strokeWidth="5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeOpacity="0.85"
                />
              ))}
            </>
          ) : colorMode === 'sectors' ? (
            // Sector coloring mode
            <>
              {/* Glow per sector */}
              {sectorPaths.map((d, i) => (
                <path
                  key={`glow-${i}`}
                  d={d}
                  fill="none"
                  stroke={SECTOR_COLORS[i]}
                  strokeWidth="10"
                  strokeOpacity="0.12"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              ))}
              {/* Main sector strokes */}
              {sectorPaths.map((d, i) => (
                <path
                  key={`sector-${i}`}
                  d={d}
                  fill="none"
                  stroke={SECTOR_COLORS[i]}
                  strokeWidth="4.5"
                  strokeOpacity="0.8"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              ))}
              {/* Sector labels */}
              {sectorMidpoints.map((pt, i) => (
                <g key={`label-${i}`}>
                  <rect
                    x={pt.x - 8}
                    y={pt.y - 16}
                    width="16"
                    height="10"
                    rx="2"
                    fill={SECTOR_COLORS[i]}
                    fillOpacity="0.2"
                  />
                  <text
                    x={pt.x}
                    y={pt.y - 8.5}
                    fill={SECTOR_COLORS[i]}
                    fontSize="7"
                    fontWeight="700"
                    fontFamily="monospace"
                    textAnchor="middle"
                  >
                    S{i + 1}
                  </text>
                </g>
              ))}
            </>
          ) : (
            // Accent (original) mode
            <>
              <path
                d={fullPathD}
                fill="none"
                stroke={accentColor}
                strokeWidth="10"
                strokeOpacity="0.12"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <path
                d={fullPathD}
                fill="none"
                stroke={accentColor}
                strokeWidth="4.5"
                strokeOpacity="0.8"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </>
          )}

          {/* Road center dashes */}
          <path
            d={fullPathD}
            fill="none"
            stroke="rgba(255,255,255,0.1)"
            strokeWidth="1"
            strokeDasharray="8 6"
            strokeLinecap="round"
            strokeLinejoin="round"
          />

          {/* Road edge lines (track boundary markers) */}
          <path
            d={fullPathD}
            fill="none"
            stroke="rgba(255,255,255,0.06)"
            strokeWidth="16"
            strokeLinecap="round"
            strokeLinejoin="round"
          />

          {/* Kerb markers at corners */}
          {kerbMarkers.map((k, i) => (
            <g key={`kerb-${i}`}>
              {/* Alternating red/white stripes perpendicular to track */}
              <line
                x1={k.x1} y1={k.y1} x2={k.x2} y2={k.y2}
                stroke={KERB_RED}
                strokeWidth="3"
                strokeOpacity="0.6"
                strokeDasharray="3 2"
              />
              <line
                x1={k.x1} y1={k.y1} x2={k.x2} y2={k.y2}
                stroke={KERB_WHITE}
                strokeWidth="1.5"
                strokeOpacity="0.4"
                strokeDasharray="2 3"
                strokeDashoffset="3"
              />
            </g>
          ))}

          {/* DRS Zones */}
          {drsPaths.map((drs, i) => (
            <g key={`drs-${i}`}>
              <path
                d={drs.d}
                fill="none"
                stroke={DRS_COLOR}
                strokeWidth="8"
                strokeOpacity="0.35"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <path
                d={drs.d}
                fill="none"
                stroke={DRS_COLOR}
                strokeWidth="3"
                strokeOpacity="0.9"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <rect
                x={drs.mid.x - 10}
                y={drs.mid.y - 14}
                width="20"
                height="10"
                rx="2"
                fill="#0D1117"
                fillOpacity="0.8"
                stroke={DRS_COLOR}
                strokeWidth="0.5"
                strokeOpacity="0.5"
              />
              <text
                x={drs.mid.x}
                y={drs.mid.y - 6.5}
                fill={DRS_COLOR}
                fontSize="6"
                fontWeight="700"
                fontFamily="monospace"
                textAnchor="middle"
              >
                DRS
              </text>
            </g>
          ))}

          {/* Start/finish line — checkered flag */}
          {projected.length > 0 && (() => {
            const sf = projected[0];
            const next = projected[Math.min(3, projected.length - 1)];
            const angle = Math.atan2(next.y - sf.y, next.x - sf.x) * (180 / Math.PI);
            return (
              <g>
                <rect
                  x={sf.x - 5}
                  y={sf.y - 3}
                  width="10"
                  height="6"
                  fill="url(#checkered)"
                  transform={`rotate(${angle}, ${sf.x}, ${sf.y})`}
                  rx="1"
                />
                <circle cx={sf.x} cy={sf.y} r="7" fill={accentColor} fillOpacity="0.2" />
                <circle cx={sf.x} cy={sf.y} r="3" fill={accentColor} />
                <text
                  x={sf.x}
                  y={sf.y + 14}
                  fill="rgba(255,255,255,0.5)"
                  fontSize="6"
                  fontWeight="600"
                  fontFamily="monospace"
                  textAnchor="middle"
                >
                  START
                </text>
              </g>
            );
          })()}

          {/* Turn markers with speed badges */}
          {turnPositions.map(turn => {
            const color = turn.speedKph ? speedToColor(turn.speedKph) : 'rgba(255,255,255,0.5)';
            return (
              <g
                key={`turn-${turn.number}`}
                style={{ cursor: 'pointer' }}
                onMouseEnter={(e) => {
                  setHoveredTurn(turn);
                  const rect = containerRef.current?.getBoundingClientRect();
                  if (rect) {
                    setTooltipPos({
                      x: e.clientX - rect.left,
                      y: e.clientY - rect.top - 60,
                    });
                  }
                }}
                onMouseLeave={() => setHoveredTurn(null)}
              >
                {/* Glow ring color-coded by speed */}
                <circle cx={turn.px} cy={turn.py} r="8" fill={color} fillOpacity="0.1" />
                {/* Background */}
                <circle cx={turn.px} cy={turn.py} r="6" fill="#0D1117" fillOpacity="0.85" />
                {/* Border ring */}
                <circle cx={turn.px} cy={turn.py} r="5.5" fill="none" stroke={color} strokeWidth="0.8" strokeOpacity="0.6" />
                {/* Turn number */}
                <text
                  x={turn.px}
                  y={turn.py + 2.5}
                  fill={color}
                  fontSize="5.5"
                  fontWeight="700"
                  fontFamily="monospace"
                  textAnchor="middle"
                >
                  {turn.number}
                </text>
                {/* Speed badge (small text below for named turns) */}
                {turn.name && turn.name !== `Turn ${turn.number}` && (
                  <text
                    x={turn.px}
                    y={turn.py + 13}
                    fill="rgba(255,255,255,0.4)"
                    fontSize="4"
                    fontWeight="600"
                    fontFamily="monospace"
                    textAnchor="middle"
                  >
                    {turn.name}
                  </text>
                )}
              </g>
            );
          })}

          {/* Car position dots */}
          {carDots.map((car) => (
            <g key={car.driverNumber}>
              {/* Glow */}
              <circle cx={car.px} cy={car.py} r="8" fill={car.color} fillOpacity="0.15" />
              {/* Dot */}
              <circle
                cx={car.px}
                cy={car.py}
                r="4.5"
                fill={car.color}
                stroke="#0D1117"
                strokeWidth="1.2"
                style={{ transition: 'cx 0.5s ease-out, cy 0.5s ease-out' }}
              />
              {/* Position badge */}
              <rect
                x={car.px + 6}
                y={car.py - 5}
                width={car.code.length * 5 + 8}
                height="10"
                rx="2"
                fill="#0D1117"
                fillOpacity="0.75"
              />
              {/* Label */}
              <text
                x={car.px + 10}
                y={car.py + 2.5}
                fill={car.color}
                fontSize="7"
                fontWeight="700"
                fontFamily="monospace"
              >
                {car.code}
              </text>
            </g>
          ))}
        </svg>
      </div>

      {/* Turn tooltip */}
      {hoveredTurn && (
        <div
          className="absolute pointer-events-none z-50"
          style={{
            left: tooltipPos.x,
            top: tooltipPos.y,
            transform: 'translateX(-50%)',
          }}
        >
          <div className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.2)] rounded-lg px-3 py-2 shadow-xl">
            <div className="text-[12px] font-semibold text-foreground">
              T{hoveredTurn.number} — {hoveredTurn.name || `Turn ${hoveredTurn.number}`}
            </div>
            <div className="flex gap-3 mt-1 text-[11px] text-muted-foreground">
              {hoveredTurn.gear && (
                <span>Gear: <span className="text-cyan-400 font-mono">{hoveredTurn.gear}</span></span>
              )}
              {hoveredTurn.speedKph && (
                <span>Speed: <span className="text-[#FF8000] font-mono">{hoveredTurn.speedKph} km/h</span></span>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Color mode toggle (bottom-right) */}
      <div className="absolute bottom-2 right-2 flex gap-1">
        {(['sectors', 'speed', 'accent'] as ColorMode[]).map(mode => (
          <button
            key={mode}
            onClick={() => {
              // We need a parent-controlled colorMode, but for now we'll make it local
              const evt = new CustomEvent('trackmap-colormode', { detail: mode });
              window.dispatchEvent(evt);
            }}
            className={`px-1.5 py-0.5 text-[9px] rounded font-mono transition-colors ${
              colorMode === mode
                ? 'bg-[rgba(255,128,0,0.15)] text-[#FF8000]'
                : 'bg-[rgba(255,255,255,0.03)] text-muted-foreground hover:bg-[rgba(255,255,255,0.06)]'
            }`}
          >
            {mode.toUpperCase()}
          </button>
        ))}
      </div>

      {/* Zoom indicator */}
      {zoom !== 1 && (
        <div className="absolute bottom-2 left-2 text-[10px] text-muted-foreground font-mono bg-[rgba(0,0,0,0.5)] px-1.5 py-0.5 rounded">
          {Math.round(zoom * 100)}%
        </div>
      )}
    </div>
  );
}
