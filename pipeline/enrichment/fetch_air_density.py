"""
Air Density Fetcher
───────────────────
Fetches historical race-day air density data from Open-Meteo archive API
for every race in the database. Computes air density impact on performance.

Collections created/updated:
  - race_air_density : per-race air density, pressure, elevation data

Usage:
    python -m pipeline.enrichment.fetch_air_density
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

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"

# Circuit GPS coordinates (lat, lon)
CIRCUIT_COORDS = {
    "albert_park": (-37.8497, 144.9680),
    "americas": (30.1328, -97.6411),
    "bahrain": (26.0325, 50.5106),
    "baku": (40.3725, 49.8533),
    "catalunya": (41.5700, 2.2611),
    "hockenheimring": (49.3278, 8.5656),
    "hungaroring": (47.5789, 19.2486),
    "imola": (44.3439, 11.7167),
    "interlagos": (-23.7014, -46.6969),
    "istanbul": (40.9517, 29.4050),
    "jeddah": (21.6319, 39.1044),
    "losail": (25.4900, 51.4542),
    "marina_bay": (1.2914, 103.8640),
    "miami": (25.9581, -80.2389),
    "monaco": (43.7347, 7.4206),
    "monza": (45.6156, 9.2811),
    "mugello": (43.9975, 11.3719),
    "nurburgring": (50.3356, 6.9475),
    "portimao": (37.2270, -8.6267),
    "red_bull_ring": (47.2197, 14.7647),
    "ricard": (43.2506, 5.7917),
    "rodriguez": (19.4042, -99.0907),
    "shanghai": (31.3389, 121.2197),
    "silverstone": (52.0786, -1.0169),
    "sochi": (43.4057, 39.9578),
    "spa": (50.4372, 5.9714),
    "suzuka": (34.8431, 136.5406),
    "vegas": (36.1147, -115.1728),
    "villeneuve": (45.5000, -73.5228),
    "yas_marina": (24.4672, 54.6031),
    "zandvoort": (52.3888, 4.5409),
}

# Map Race names from fastf1_laps → circuit slugs
RACE_TO_SLUG = {
    "Australian Grand Prix": "albert_park",
    "United States Grand Prix": "americas",
    "70th Anniversary Grand Prix": "silverstone",
    "Bahrain Grand Prix": "bahrain",
    "Sakhir Grand Prix": "bahrain",
    "Azerbaijan Grand Prix": "baku",
    "Spanish Grand Prix": "catalunya",
    "German Grand Prix": "hockenheimring",
    "Hungarian Grand Prix": "hungaroring",
    "Emilia Romagna Grand Prix": "imola",
    "Brazilian Grand Prix": "interlagos",
    "São Paulo Grand Prix": "interlagos",
    "Turkish Grand Prix": "istanbul",
    "Saudi Arabian Grand Prix": "jeddah",
    "Qatar Grand Prix": "losail",
    "Singapore Grand Prix": "marina_bay",
    "Miami Grand Prix": "miami",
    "Monaco Grand Prix": "monaco",
    "Italian Grand Prix": "monza",
    "Tuscan Grand Prix": "mugello",
    "Eifel Grand Prix": "nurburgring",
    "Portuguese Grand Prix": "portimao",
    "Austrian Grand Prix": "red_bull_ring",
    "Styrian Grand Prix": "red_bull_ring",
    "French Grand Prix": "ricard",
    "Mexican Grand Prix": "rodriguez",
    "Mexico City Grand Prix": "rodriguez",
    "Chinese Grand Prix": "shanghai",
    "British Grand Prix": "silverstone",
    "Russian Grand Prix": "sochi",
    "Belgian Grand Prix": "spa",
    "Japanese Grand Prix": "suzuka",
    "Las Vegas Grand Prix": "vegas",
    "Canadian Grand Prix": "villeneuve",
    "Abu Dhabi Grand Prix": "yas_marina",
    "Dutch Grand Prix": "zandvoort",
}


def air_density(temp_c: float, rh_pct: float, pressure_hpa: float) -> float:
    """Calculate air density in kg/m3."""
    T = temp_c + 273.15
    P = pressure_hpa * 100
    Rd = 287.058
    Rv = 461.495
    es = 611.2 * math.exp(17.67 * temp_c / (temp_c + 243.5))
    e = (rh_pct / 100.0) * es
    Pd = P - e
    return (Pd / (Rd * T)) + (e / (Rv * T))


SEA_LEVEL_DENSITY = 1.225  # kg/m3 standard


def fetch_race_day_weather(lat: float, lon: float, date: str) -> dict | None:
    """Fetch hourly weather from Open-Meteo archive for a specific date."""
    try:
        resp = requests.get(ARCHIVE_URL, params={
            "latitude": lat,
            "longitude": lon,
            "start_date": date,
            "end_date": date,
            "hourly": "temperature_2m,relative_humidity_2m,surface_pressure",
        }, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        hourly = data.get("hourly", {})
        temps = hourly.get("temperature_2m", [])
        rhs = hourly.get("relative_humidity_2m", [])
        pressures = hourly.get("surface_pressure", [])

        if not temps or not pressures:
            return None

        # Use afternoon hours (12:00-16:00 local) as race window approximation
        # Open-Meteo returns UTC, so we take the middle of the day
        mid_indices = list(range(10, 17))  # 10:00-16:00 UTC
        valid_indices = [i for i in mid_indices if i < len(temps) and temps[i] is not None]

        if not valid_indices:
            valid_indices = [i for i in range(len(temps)) if temps[i] is not None]

        avg_temp = sum(temps[i] for i in valid_indices) / len(valid_indices)
        avg_rh = sum(rhs[i] for i in valid_indices) / len(valid_indices) if rhs else 50
        avg_pressure = sum(pressures[i] for i in valid_indices) / len(valid_indices)

        rho = air_density(avg_temp, avg_rh, avg_pressure)

        return {
            "avg_temp_c": round(avg_temp, 1),
            "avg_humidity_pct": round(avg_rh, 1),
            "avg_surface_pressure_hpa": round(avg_pressure, 1),
            "air_density_kg_m3": round(rho, 4),
            "density_loss_pct": round((1 - rho / SEA_LEVEL_DENSITY) * 100, 1),
            "downforce_loss_pct": round((1 - rho / SEA_LEVEL_DENSITY) * 100, 1),
        }
    except Exception as e:
        return None


def get_elevation(lat: float, lon: float) -> float | None:
    """Get elevation from Open-Meteo."""
    try:
        resp = requests.get(ELEVATION_URL, params={"latitude": lat, "longitude": lon}, timeout=10)
        resp.raise_for_status()
        elev = resp.json().get("elevation", [None])
        return elev[0] if elev else None
    except Exception:
        return None


def main():
    db = get_db()
    print("✅ Connected to MongoDB")
    print("\nFetching race-day air density data...")

    # Get unique (Year, Race) from fastf1_laps to find race dates
    # Use fastf1_weather for actual race dates (has Time field)
    race_dates_pipeline = [
        {"$match": {"SessionType": "R"}},
        {"$group": {
            "_id": {"Year": "$Year", "Race": "$Race"},
        }},
        {"$sort": {"_id.Year": 1, "_id.Race": 1}},
    ]
    race_combos = list(db["fastf1_laps"].aggregate(race_dates_pipeline, allowDiskUse=True))
    print(f"  Found {len(race_combos)} unique (Year, Race) combos")

    # For race dates, use openf1_sessions where available
    session_dates = {}
    for s in db["openf1_sessions"].find(
        {"session_type": "Race"},
        {"circuit_short_name": 1, "year": 1, "date_start": 1, "_id": 0},
    ):
        key = (s.get("year"), s.get("circuit_short_name"))
        if s.get("date_start"):
            session_dates[key] = s["date_start"][:10]  # YYYY-MM-DD

    # Check existing
    existing = set()
    for doc in db["race_air_density"].find({}, {"year": 1, "race": 1, "_id": 0}):
        existing.add((doc.get("year"), doc.get("race")))

    # Fetch elevation per circuit (once)
    elevations = {}
    for slug, (lat, lon) in CIRCUIT_COORDS.items():
        elev = get_elevation(lat, lon)
        if elev is not None:
            elevations[slug] = elev
        time.sleep(0.2)
    print(f"  Fetched elevations for {len(elevations)} circuits")

    ops = []
    now = datetime.now(timezone.utc)
    fetched = 0

    for combo in race_combos:
        year = combo["_id"]["Year"]
        race = combo["_id"]["Race"]

        if (year, race) in existing:
            continue

        slug = RACE_TO_SLUG.get(race)
        if not slug or slug not in CIRCUIT_COORDS:
            continue

        lat, lon = CIRCUIT_COORDS[slug]

        # Try to find race date
        # Check openf1_sessions first (2023+)
        date_str = None
        from enrichment.fetch_circuits import OF_CIRCUIT_TO_SLUG
        for of_name, of_slug in OF_CIRCUIT_TO_SLUG.items():
            if of_slug == slug:
                date_str = session_dates.get((year, of_name))
                if date_str:
                    break

        # Fallback: use jolpica_race_results for historical dates (2018-2022)
        if not date_str:
            year_int = int(year) if not isinstance(year, int) else year
            jolpica_doc = db["jolpica_race_results"].find_one(
                {"season": year_int, "race_name": race},
                {"date": 1, "_id": 0},
            )
            if jolpica_doc and jolpica_doc.get("date"):
                date_str = jolpica_doc["date"][:10]

        if not date_str:
            continue

        weather = fetch_race_day_weather(lat, lon, date_str)
        if not weather:
            continue

        doc = {
            "year": year,
            "race": race,
            "circuit_slug": slug,
            "race_date": date_str,
            "latitude": lat,
            "longitude": lon,
            "elevation_m": elevations.get(slug),
            **weather,
            "ingested_at": now,
        }

        ops.append(UpdateOne(
            {"year": year, "race": race},
            {"$set": doc},
            upsert=True,
        ))
        fetched += 1

        if fetched % 10 == 0:
            print(f"  Fetched {fetched} race-day conditions...")
            time.sleep(1)  # Rate limit
        else:
            time.sleep(0.3)

    if ops:
        db["race_air_density"].create_index([("year", 1), ("race", 1)], unique=True)
        db["race_air_density"].create_index("circuit_slug")
        result = db["race_air_density"].bulk_write(ops, ordered=False)
        count = result.upserted_count + result.modified_count
        print(f"\n  ✅ Upserted {count} race-day air density records")
    else:
        print("\n  ⚠ No new records to insert (all already exist or no dates found)")

    # Verify
    total = db["race_air_density"].count_documents({})
    print(f"\n  race_air_density: {total} records")

    # Show extremes
    highest = db["race_air_density"].find_one(
        sort=[("density_loss_pct", -1)],
    )
    lowest = db["race_air_density"].find_one(
        sort=[("density_loss_pct", 1)],
    )
    if highest:
        print(f"\n  Thinnest air: {highest.get('year')} {highest.get('race')}")
        print(f"    Elevation: {highest.get('elevation_m')}m, Density: {highest.get('air_density_kg_m3')} kg/m3, Loss: {highest.get('density_loss_pct')}%")
    if lowest:
        print(f"  Densest air: {lowest.get('year')} {lowest.get('race')}")
        print(f"    Elevation: {lowest.get('elevation_m')}m, Density: {lowest.get('air_density_kg_m3')} kg/m3, Loss: {lowest.get('density_loss_pct')}%")


if __name__ == "__main__":
    main()
