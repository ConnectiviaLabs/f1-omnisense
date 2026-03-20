import { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import 'maplibre-gl/dist/maplibre-gl.css';
import Map, { Source, Layer, Marker, NavigationControl } from 'react-map-gl/maplibre';
import maplibregl from 'maplibre-gl';
import { MapPin } from 'lucide-react';
import type { TurnInfo, DRSZone } from '../data/circuits';
import {
  buildCumulativeDistances,
  interpolatePosition,
  extractSubLine,
  splitIntoSectors,
  buildSpeedGradientStops,
  getBounds,
} from '../utils/trackGeometry';

// ─── Types ──────────────────────────────────────────────────────────
export interface CarPosition {
  driverNumber: number;
  code: string;
  x: number; // normalised 0-1 along track
  y: number; // unused (always 0)
  color: string;
}

type ColorMode = 'accent' | 'sectors' | 'speed';

interface TrackMapGLProps {
  geojsonPath: string;
  circuitName: string;
  locality: string;
  country: string;
  lengthKm: number;
  lat: number;
  lng: number;
  cars?: CarPosition[];
  accentColor?: string;
  className?: string;
  height?: number;
  turns?: TurnInfo[];
  drsZones?: DRSZone[];
  colorMode?: ColorMode;
}

const SECTOR_COLORS = ['#e74c3c', '#3498db', '#f1c40f'];
const DRS_COLOR = '#00ff88';
const MAP_STYLE = 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json';

// ─── Component ──────────────────────────────────────────────────────
export function TrackMapGL({
  geojsonPath,
  circuitName,
  locality,
  country,
  lengthKm,
  lat,
  lng,
  cars,
  accentColor = '#FF8000',
  className = '',
  height = 550,
  turns,
  drsZones,
  colorMode: initialColorMode = 'sectors',
}: TrackMapGLProps) {
  const [coords, setCoords] = useState<[number, number][] | null>(null);
  const [loading, setLoading] = useState(true);
  const [colorMode, setColorMode] = useState<ColorMode>(initialColorMode);
  const [hoveredTurn, setHoveredTurn] = useState<TurnInfo | null>(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });
  const [mapReady, setMapReady] = useState(false);
  const mapRef = useRef<any>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // ─── Fetch GeoJSON ──────────────────────────────────────────────
  useEffect(() => {
    setLoading(true);
    setMapReady(false);
    fetch(geojsonPath)
      .then((r) => r.json())
      .then((data) => {
        const feature = data.features?.[0];
        if (feature?.geometry?.coordinates) {
          setCoords(feature.geometry.coordinates as [number, number][]);
        }
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [geojsonPath]);

  // ─── Precomputed geometry ───────────────────────────────────────
  const cumDist = useMemo(
    () => (coords ? buildCumulativeDistances(coords) : []),
    [coords],
  );

  const bounds = useMemo(
    () => (coords ? getBounds(coords) : null),
    [coords],
  );

  // ─── Track GeoJSON sources ─────────────────────────────────────
  const fullTrackGeoJSON = useMemo(() => {
    if (!coords) return null;
    return {
      type: 'Feature' as const,
      properties: {},
      geometry: { type: 'LineString' as const, coordinates: coords },
    };
  }, [coords]);

  const sectorGeoJSON = useMemo(() => {
    if (!coords) return null;
    const [s1, s2, s3] = splitIntoSectors(coords);
    return {
      type: 'FeatureCollection' as const,
      features: [
        { type: 'Feature' as const, properties: { sector: 1 }, geometry: { type: 'LineString' as const, coordinates: s1 } },
        { type: 'Feature' as const, properties: { sector: 2 }, geometry: { type: 'LineString' as const, coordinates: s2 } },
        { type: 'Feature' as const, properties: { sector: 3 }, geometry: { type: 'LineString' as const, coordinates: s3 } },
      ],
    };
  }, [coords]);

  // ─── DRS zone GeoJSON ──────────────────────────────────────────
  const drsGeoJSON = useMemo(() => {
    if (!coords || !drsZones?.length) return null;
    const features = drsZones.map((zone, i) => {
      const line = extractSubLine(coords, cumDist, zone.start, zone.end, 30);
      return {
        type: 'Feature' as const,
        properties: { id: i },
        geometry: { type: 'LineString' as const, coordinates: line },
      };
    });
    return { type: 'FeatureCollection' as const, features };
  }, [coords, cumDist, drsZones]);

  // ─── DRS label positions ───────────────────────────────────────
  const drsLabels = useMemo(() => {
    if (!coords || !drsZones?.length || !cumDist.length) return [];
    return drsZones.map((zone) => {
      const mid = zone.start <= zone.end
        ? (zone.start + zone.end) / 2
        : ((zone.start + zone.end + 1) / 2) % 1;
      return interpolatePosition(coords, cumDist, mid);
    });
  }, [coords, cumDist, drsZones]);

  // ─── Turn positions ────────────────────────────────────────────
  const turnPositions = useMemo(() => {
    if (!coords || !turns?.length || !cumDist.length) return [];
    return turns.map((turn) => ({
      ...turn,
      ...interpolatePosition(coords, cumDist, turn.position),
    }));
  }, [coords, cumDist, turns]);

  // ─── Car positions ─────────────────────────────────────────────
  const carMarkers = useMemo(() => {
    if (!coords || !cars?.length || !cumDist.length) return [];
    return cars.map((car) => ({
      ...car,
      ...interpolatePosition(coords, cumDist, car.x),
    }));
  }, [coords, cumDist, cars]);

  // ─── Start/Finish position ─────────────────────────────────────
  const sfPosition = useMemo(() => {
    if (!coords?.length) return null;
    return { lng: coords[0][0], lat: coords[0][1] };
  }, [coords]);

  // ─── Speed gradient source (single LineString with lineMetrics) ─
  const speedGradientStops = useMemo(() => {
    if (!coords || !cumDist.length) return [];
    return buildSpeedGradientStops(coords, cumDist);
  }, [coords, cumDist]);

  // ─── Fit bounds on load ────────────────────────────────────────
  const onMapLoad = useCallback(() => {
    setMapReady(true);
    if (bounds && mapRef.current) {
      mapRef.current.fitBounds(
        [[bounds[0], bounds[1]], [bounds[2], bounds[3]]],
        { padding: 60, duration: 800 },
      );
    }
  }, [bounds]);

  // Re-fit when circuit changes
  useEffect(() => {
    if (bounds && mapRef.current) {
      mapRef.current.fitBounds(
        [[bounds[0], bounds[1]], [bounds[2], bounds[3]]],
        { padding: 60, duration: 800 },
      );
    }
  }, [bounds]);

  // ─── Render ────────────────────────────────────────────────────
  if (loading) {
    return (
      <div className={`bg-card rounded-lg border border-border flex items-center justify-center ${className}`} style={{ height }}>
        <div className="text-muted-foreground text-sm animate-pulse">Loading track map...</div>
      </div>
    );
  }

  return (
    <div ref={containerRef} className={`relative bg-card rounded-lg border border-border overflow-hidden ${className}`} style={{ height }}>
      {/* Header overlay */}
      <div className="absolute top-0 left-0 right-0 z-10 p-3 bg-gradient-to-b from-[#0D1117ee] to-transparent pointer-events-none">
        <div className="flex items-center gap-2">
          <MapPin className="w-3.5 h-3.5 text-primary" />
          <span className="text-sm font-medium text-foreground">{circuitName}</span>
          <span className="text-[12px] text-muted-foreground">{locality}, {country}</span>
          <span className="text-[12px] text-muted-foreground ml-auto">{lengthKm} km</span>
        </div>
      </div>

      {/* MapLibre Map */}
      <Map
        ref={mapRef}
        mapLib={maplibregl}
        mapStyle={MAP_STYLE}
        initialViewState={{
          longitude: lng,
          latitude: lat,
          zoom: 14,
        }}
        style={{ width: '100%', height: '100%' }}
        onLoad={onMapLoad}
        attributionControl={false}
        dragRotate={false}
        pitchWithRotate={false}
      >
        <NavigationControl position="bottom-left" showCompass={false} />

        {/* ── Asphalt base layer ── */}
        {mapReady && fullTrackGeoJSON && (
          <Source id="track-base" type="geojson" data={fullTrackGeoJSON}>
            <Layer
              id="track-asphalt-outer"
              type="line"
              paint={{
                'line-color': '#2A3142',
                'line-width': 16,
                'line-opacity': 0.95,
              }}
              layout={{ 'line-cap': 'round', 'line-join': 'round' }}
            />
            <Layer
              id="track-asphalt"
              type="line"
              paint={{
                'line-color': '#111122',
                'line-width': 12,
                'line-opacity': 0.95,
              }}
              layout={{ 'line-cap': 'round', 'line-join': 'round' }}
            />
          </Source>
        )}

        {/* ── Sector coloring mode ── */}
        {mapReady && colorMode === 'sectors' && sectorGeoJSON && (
          <Source id="track-sectors" type="geojson" data={sectorGeoJSON}>
            {/* Glow layer */}
            <Layer
              id="sector-glow"
              type="line"
              paint={{
                'line-color': [
                  'match', ['get', 'sector'],
                  1, SECTOR_COLORS[0],
                  2, SECTOR_COLORS[1],
                  3, SECTOR_COLORS[2],
                  '#fff',
                ],
                'line-width': 12,
                'line-opacity': 0.35,
                'line-blur': 6,
              }}
              layout={{ 'line-cap': 'round', 'line-join': 'round' }}
            />
            {/* Main track stroke */}
            <Layer
              id="sector-main"
              type="line"
              paint={{
                'line-color': [
                  'match', ['get', 'sector'],
                  1, SECTOR_COLORS[0],
                  2, SECTOR_COLORS[1],
                  3, SECTOR_COLORS[2],
                  '#fff',
                ],
                'line-width': 5,
                'line-opacity': 1,
              }}
              layout={{ 'line-cap': 'round', 'line-join': 'round' }}
            />
          </Source>
        )}

        {/* ── Speed gradient mode ── */}
        {mapReady && colorMode === 'speed' && fullTrackGeoJSON && (
          <Source id="track-speed" type="geojson" data={fullTrackGeoJSON} lineMetrics>
            <Layer
              id="speed-gradient"
              type="line"
              paint={{
                'line-width': 6,
                'line-opacity': 1,
                'line-gradient': [
                  'interpolate',
                  ['linear'],
                  ['line-progress'],
                  ...speedGradientStops.flatMap(([progress, color]) => [progress, color]),
                ],
              }}
              layout={{ 'line-cap': 'round', 'line-join': 'round' }}
            />
          </Source>
        )}

        {/* ── Accent mode ── */}
        {mapReady && colorMode === 'accent' && fullTrackGeoJSON && (
          <Source id="track-accent" type="geojson" data={fullTrackGeoJSON}>
            <Layer
              id="accent-glow"
              type="line"
              paint={{
                'line-color': accentColor,
                'line-width': 12,
                'line-opacity': 0.3,
                'line-blur': 6,
              }}
              layout={{ 'line-cap': 'round', 'line-join': 'round' }}
            />
            <Layer
              id="accent-main"
              type="line"
              paint={{
                'line-color': accentColor,
                'line-width': 5,
                'line-opacity': 1,
              }}
              layout={{ 'line-cap': 'round', 'line-join': 'round' }}
            />
          </Source>
        )}

        {/* ── Center dashes ── */}
        {mapReady && fullTrackGeoJSON && (
          <Source id="track-center" type="geojson" data={fullTrackGeoJSON}>
            <Layer
              id="center-dash"
              type="line"
              paint={{
                'line-color': '#ffffff',
                'line-width': 1,
                'line-opacity': 0.3,
                'line-dasharray': [2, 4],
              }}
              layout={{ 'line-cap': 'round' }}
            />
          </Source>
        )}

        {/* ── DRS zones ── */}
        {mapReady && drsGeoJSON && (
          <Source id="drs-zones" type="geojson" data={drsGeoJSON}>
            <Layer
              id="drs-glow"
              type="line"
              paint={{
                'line-color': DRS_COLOR,
                'line-width': 10,
                'line-opacity': 0.3,
                'line-blur': 3,
              }}
              layout={{ 'line-cap': 'round', 'line-join': 'round' }}
            />
            <Layer
              id="drs-core"
              type="line"
              paint={{
                'line-color': DRS_COLOR,
                'line-width': 3,
                'line-opacity': 0.8,
              }}
              layout={{ 'line-cap': 'round', 'line-join': 'round' }}
            />
          </Source>
        )}

        {/* ── DRS labels ── */}
        {drsLabels.map((pos, i) => (
          <Marker key={`drs-${i}`} longitude={pos.lng} latitude={pos.lat} anchor="center">
            <div className="bg-[#00ff8833] border border-[#00ff8866] rounded px-1 py-0.5 text-[10px] font-bold text-[#00ff88] tracking-wider">
              DRS
            </div>
          </Marker>
        ))}

        {/* ── Start/Finish marker ── */}
        {sfPosition && (
          <Marker longitude={sfPosition.lng} latitude={sfPosition.lat} anchor="center">
            <div className="flex flex-col items-center">
              <div
                className="w-5 h-3 rounded-sm border border-white/50"
                style={{
                  background: `repeating-conic-gradient(#fff 0% 25%, #111 0% 50%) 50%/6px 6px`,
                }}
              />
              <span className="text-[10px] text-white/60 mt-0.5 tracking-wider font-bold">S/F</span>
            </div>
          </Marker>
        )}

        {/* ── Turn markers ── */}
        {turnPositions.map((turn) => (
          <Marker
            key={`turn-${turn.number}`}
            longitude={turn.lng}
            latitude={turn.lat}
            anchor="center"
          >
            <div
              className="relative cursor-pointer"
              onMouseEnter={(e) => {
                setHoveredTurn(turn);
                const rect = containerRef.current?.getBoundingClientRect();
                if (rect) setTooltipPos({ x: e.clientX - rect.left, y: e.clientY - rect.top });
              }}
              onMouseLeave={() => setHoveredTurn(null)}
            >
              <div className="w-4 h-4 rounded-full bg-background border border-white/40 flex items-center justify-center">
                <span className="text-[10px] text-white font-bold">{turn.number}</span>
              </div>
            </div>
          </Marker>
        ))}

        {/* ── Car markers ── */}
        {carMarkers.map((car) => (
          <Marker
            key={`car-${car.driverNumber}`}
            longitude={car.lng}
            latitude={car.lat}
            anchor="center"
          >
            <div className="flex flex-col items-center" style={{ transition: 'all 0.5s ease-out' }}>
              {/* Glow */}
              <div
                className="absolute w-6 h-6 rounded-full"
                style={{
                  backgroundColor: car.color,
                  opacity: 0.25,
                  filter: 'blur(4px)',
                }}
              />
              {/* Dot */}
              <div
                className="w-3 h-3 rounded-full border border-white/50 z-10"
                style={{ backgroundColor: car.color }}
              />
              {/* Label */}
              <div
                className="text-[10px] font-bold mt-0.5 px-1 rounded z-10"
                style={{
                  backgroundColor: `${car.color}cc`,
                  color: '#fff',
                  textShadow: '0 0 2px rgba(0,0,0,0.8)',
                }}
              >
                {car.code}
              </div>
            </div>
          </Marker>
        ))}
      </Map>

      {/* ── Turn tooltip ── */}
      {hoveredTurn && (
        <div
          className="absolute z-20 bg-[#0D1117ee] border border-[rgba(255,128,0,0.25)] rounded-lg p-2 pointer-events-none"
          style={{ left: tooltipPos.x + 12, top: tooltipPos.y - 40 }}
        >
          <div className="text-[12px] font-bold text-foreground">
            Turn {hoveredTurn.number}
            {hoveredTurn.name && <span className="text-primary ml-1">{hoveredTurn.name}</span>}
          </div>
          {hoveredTurn.gear && (
            <div className="text-[11px] text-muted-foreground">
              Gear {hoveredTurn.gear} &middot; {hoveredTurn.speedKph} km/h
            </div>
          )}
        </div>
      )}

      {/* ── Color mode toggle ── */}
      <div className="absolute bottom-2 right-2 z-10 flex gap-1">
        {(['sectors', 'speed', 'accent'] as ColorMode[]).map((mode) => (
          <button
            key={mode}
            onClick={() => setColorMode(mode)}
            className={`px-2 py-0.5 rounded text-[10px] font-bold tracking-wider transition-all ${
              colorMode === mode
                ? 'bg-primary text-white'
                : 'bg-secondary text-muted-foreground hover:text-foreground'
            }`}
          >
            {mode.toUpperCase()}
          </button>
        ))}
      </div>

      {/* ── Sector legend ── */}
      {colorMode === 'sectors' && (
        <div className="absolute top-10 right-3 z-10 flex gap-2">
          {['S1', 'S2', 'S3'].map((label, i) => (
            <div key={label} className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-full" style={{ backgroundColor: SECTOR_COLORS[i] }} />
              <span className="text-[10px] text-muted-foreground font-mono">{label}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
