"""
Circuit Intelligence Fetcher
─────────────────────────────
Fetches circuit GeoJSON layouts from bacinger/f1-circuits,
elevation from Open-Meteo, and corner data from existing DB.

Collections created/updated:
  - circuit_intelligence : per-circuit layout, elevation, corners, metadata

Usage:
    python -m pipeline.enrichment.fetch_circuits
"""

from __future__ import annotations

import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from pymongo import UpdateOne

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from updater._db import get_db

GEOJSON_URL = "https://raw.githubusercontent.com/bacinger/f1-circuits/master/f1-circuits.geojson"
ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"

# Map circuit slugs from GeoJSON → our opponent_circuit_profiles slugs
GEOJSON_TO_SLUG = {
    "Albert Park Circuit": "albert_park",
    "Circuit of the Americas": "americas",
    "Bahrain International Circuit": "bahrain",
    "Baku City Circuit": "baku",
    "Circuit de Barcelona-Catalunya": "catalunya",
    "Hockenheimring": "hockenheimring",
    "Hungaroring": "hungaroring",
    "Autodromo Enzo e Dino Ferrari": "imola",
    "Autodromo Jose Carlos Pace": "interlagos",
    "Istanbul Park": "istanbul",
    "Jeddah Corniche Circuit": "jeddah",
    "Jeddah Street Circuit": "jeddah",
    "Losail International Circuit": "losail",
    "Marina Bay Street Circuit": "marina_bay",
    "Miami International Autodrome": "miami",
    "Circuit de Monaco": "monaco",
    "Autodromo Nazionale di Monza": "monza",
    "Autodromo Internazionale del Mugello": "mugello",
    "Nürburgring": "nurburgring",
    "Autodromo Internacional do Algarve": "portimao",
    "Red Bull Ring": "red_bull_ring",
    "Circuit Paul Ricard": "ricard",
    "Autodromo Hermanos Rodriguez": "rodriguez",
    "Shanghai International Circuit": "shanghai",
    "Silverstone Circuit": "silverstone",
    "Sochi Autodrom": "sochi",
    "Circuit de Spa-Francorchamps": "spa",
    "Suzuka International Racing Course": "suzuka",
    "Las Vegas Strip Circuit": "vegas",
    "Las Vegas Street Circuit": "vegas",
    "Circuit Gilles Villeneuve": "villeneuve",
    "Yas Marina Circuit": "yas_marina",
    "Circuit Zandvoort": "zandvoort",
    "Circuit Park Zandvoort": "zandvoort",
}

# Map openf1_sessions circuit_short_name → slug
OF_CIRCUIT_TO_SLUG = {
    "Austin": "americas", "Baku": "baku", "Catalunya": "catalunya",
    "Hungaroring": "hungaroring", "Imola": "imola", "Interlagos": "interlagos",
    "Jeddah": "jeddah", "Las Vegas": "vegas", "Lusail": "losail",
    "Marina Bay": "marina_bay", "Miami": "miami", "Monaco": "monaco",
    "Monza": "monza", "Melbourne": "albert_park", "Red Bull Ring": "red_bull_ring",
    "Bahrain": "bahrain", "Sakhir": "bahrain", "Shanghai": "shanghai",
    "Silverstone": "silverstone", "Sochi": "sochi",
    "Spa-Francorchamps": "spa", "Suzuka": "suzuka",
    "Yas Marina": "yas_marina", "Zandvoort": "zandvoort",
    "Mexico City": "rodriguez", "Montréal": "villeneuve",
    "Singapore": "marina_bay",
}


def fetch_geojson() -> list[dict]:
    """Download circuit GeoJSON from GitHub."""
    print("  Fetching GeoJSON from bacinger/f1-circuits...")
    resp = requests.get(GEOJSON_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("features", [])


def get_elevation(lat: float, lon: float) -> float | None:
    """Get elevation in meters from Open-Meteo."""
    try:
        resp = requests.get(ELEVATION_URL, params={
            "latitude": lat, "longitude": lon,
        }, timeout=10)
        resp.raise_for_status()
        elev = resp.json().get("elevation", [None])
        return elev[0] if elev else None
    except Exception:
        return None


def compute_track_stats(coords: list) -> dict:
    """Compute track length, bounding box, and centroid from coordinates."""
    if not coords or len(coords) < 2:
        return {}

    total_length_m = 0
    lats, lons = [], []

    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i][0], coords[i][1]
        lon2, lat2 = coords[i + 1][0], coords[i + 1][1]
        lats.extend([lat1, lat2])
        lons.extend([lon1, lon2])

        # Haversine distance
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        total_length_m += 6371000 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    centroid_lat = sum(lats) / len(lats)
    centroid_lon = sum(lons) / len(lons)

    return {
        "computed_length_m": round(total_length_m),
        "centroid": [round(centroid_lon, 6), round(centroid_lat, 6)],
        "bbox": [
            round(min(lons), 6), round(min(lats), 6),
            round(max(lons), 6), round(max(lats), 6),
        ],
        "coordinate_count": len(coords),
    }


def air_density(temp_c: float, rh_pct: float, pressure_hpa: float) -> float:
    """Calculate air density in kg/m3 from weather conditions."""
    T = temp_c + 273.15
    P = pressure_hpa * 100
    Rd = 287.058
    Rv = 461.495
    es = 611.2 * math.exp(17.67 * temp_c / (temp_c + 243.5))
    e = (rh_pct / 100.0) * es
    Pd = P - e
    return (Pd / (Rd * T)) + (e / (Rv * T))


def estimate_air_density_from_elevation(elevation_m: float) -> dict:
    """Estimate standard atmosphere air density from elevation."""
    # International Standard Atmosphere model
    T0 = 288.15  # sea level temp K
    P0 = 101325  # sea level pressure Pa
    L = 0.0065   # lapse rate K/m
    g = 9.80665
    M = 0.0289644  # molar mass dry air
    R = 8.31447

    T = T0 - L * elevation_m
    P = P0 * (T / T0) ** (g * M / (R * L))

    rho = P / (287.058 * T)
    rho_sea = P0 / (287.058 * T0)

    return {
        "std_atm_density_kg_m3": round(rho, 4),
        "density_loss_pct": round((1 - rho / rho_sea) * 100, 1),
        "std_atm_pressure_hpa": round(P / 100, 1),
        "std_atm_temp_c": round(T - 273.15, 1),
    }


def main():
    db = get_db()
    print("✅ Connected to MongoDB")
    print("\nBuilding circuit intelligence...")

    features = fetch_geojson()
    print(f"  Downloaded {len(features)} circuit geometries")

    ops = []
    now = datetime.now(timezone.utc)

    for feat in features:
        props = feat.get("properties", {})
        geom = feat.get("geometry", {})
        name = props.get("Name", "")
        slug = GEOJSON_TO_SLUG.get(name)

        if not slug:
            continue

        coords = geom.get("coordinates", [])
        stats = compute_track_stats(coords)

        # Get elevation from centroid
        elevation = props.get("altitude")
        if elevation is None and stats.get("centroid"):
            elevation = get_elevation(stats["centroid"][1], stats["centroid"][0])
            time.sleep(0.3)

        # Compute air density estimates
        atm = estimate_air_density_from_elevation(elevation) if elevation else {}

        doc = {
            "circuit_slug": slug,
            "circuit_name": name,
            "location": props.get("Location", ""),
            "country": props.get("Location", "").split(",")[-1].strip() if props.get("Location") else "",
            "opened": props.get("opened"),
            "first_gp": props.get("firstgp"),
            "official_length_m": props.get("length"),
            "elevation_m": elevation,
            "geometry_type": geom.get("type"),
            "coordinates": coords,
            **stats,
            **atm,
            "updated_at": now,
        }

        ops.append(UpdateOne(
            {"circuit_slug": slug},
            {"$set": doc},
            upsert=True,
        ))

    # Also add circuits from openf1_sessions not in GeoJSON
    existing_slugs = {GEOJSON_TO_SLUG.get(f.get("properties", {}).get("Name", "")) for f in features}
    for session in db["openf1_sessions"].find({}, {"circuit_short_name": 1, "country_name": 1, "_id": 0}):
        short_name = session.get("circuit_short_name", "")
        slug = OF_CIRCUIT_TO_SLUG.get(short_name)
        if slug and slug not in existing_slugs:
            existing_slugs.add(slug)

    if ops:
        db["circuit_intelligence"].create_index("circuit_slug", unique=True)
        result = db["circuit_intelligence"].bulk_write(ops, ordered=False)
        count = result.upserted_count + result.modified_count
        print(f"  ✅ Upserted {count} circuit profiles")
    else:
        print("  ⚠ No circuits to upsert")

    # Verify
    total = db["circuit_intelligence"].count_documents({})
    print(f"\n  circuit_intelligence: {total} circuits")

    sample = db["circuit_intelligence"].find_one(
        {"circuit_slug": "rodriguez"},
        {"_id": 0, "coordinates": 0, "updated_at": 0},
    )
    if sample:
        print(f"\n  Sample (Mexico City):")
        for k, v in sample.items():
            print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
