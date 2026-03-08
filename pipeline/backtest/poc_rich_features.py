"""
PoC: Rich Feature Engineering from McLaren Telemetry CSVs

Tests whether richer telemetry features (throttle consistency, brake patterns,
gear shifts, DRS efficiency, tyre-aware degradation) discriminate between
good and bad race outcomes — something the current race-level averages cannot do.

Usage:
    PYTHONPATH=pipeline:. python3 -m pipeline.backtest.poc_rich_features
"""

import os
import numpy as np
import pandas as pd
from pymongo import MongoClient

CSV_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "Car", "mclaren_data")

BAD_OUTCOMES_POSITIONS = {
    # McLaren 2024 "bad" races — lapped, big position losses, or major underperformance
    # Using positions_gained <= -5 OR lapped as "bad outcome"
}


def load_race_csv(filepath: str) -> pd.DataFrame:
    """Load a single race CSV."""
    df = pd.read_csv(filepath)
    # Convert types
    for col in ["RPM", "Speed", "Throttle", "TyreLife", "LapNumber"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Brake"] = df["Brake"].map({"True": 1, "False": 0, True: 1, False: 0}).fillna(0).astype(int)
    df["DRS"] = pd.to_numeric(df["DRS"], errors="coerce").fillna(0).astype(int)
    df["nGear"] = pd.to_numeric(df["nGear"], errors="coerce").fillna(0).astype(int)
    return df


def compute_rich_features(df: pd.DataFrame) -> dict:
    """
    Compute per-race rich telemetry features for a single driver.
    Input: DataFrame with columns [RPM, Speed, nGear, Throttle, Brake, DRS, LapNumber, TyreLife, Compound]
    """
    features = {}

    # --- Throttle features ---
    features["throttle_mean"] = df["Throttle"].mean()
    features["throttle_std"] = df["Throttle"].std()
    features["throttle_consistency"] = 1.0 - (df["Throttle"].std() / max(df["Throttle"].mean(), 1))

    # Per-lap throttle variance (how consistent within each lap)
    lap_throttle_std = df.groupby("LapNumber")["Throttle"].std()
    features["throttle_intra_lap_std_mean"] = lap_throttle_std.mean()
    features["throttle_intra_lap_std_std"] = lap_throttle_std.std()  # variance of variance = instability

    # Full throttle ratio (throttle > 95%)
    features["full_throttle_ratio"] = (df["Throttle"] > 95).mean()
    # Lift-and-coast indicator (throttle between 20-80% — partial application)
    features["partial_throttle_ratio"] = ((df["Throttle"] > 20) & (df["Throttle"] < 80)).mean()

    # --- Brake features ---
    features["brake_ratio"] = df["Brake"].mean()  # fraction of time braking

    # Brake transitions (on/off switches per lap) — stressed driver brakes more erratically
    brake_transitions = (df["Brake"].diff().abs()).sum()
    n_laps = df["LapNumber"].nunique()
    features["brake_transitions_per_lap"] = brake_transitions / max(n_laps, 1)

    # Brake-while-throttle (simultaneous — unusual, indicates car issues or driver compensation)
    features["brake_throttle_overlap"] = ((df["Brake"] == 1) & (df["Throttle"] > 20)).mean()

    # --- RPM features ---
    features["rpm_mean"] = df["RPM"].mean()
    features["rpm_std"] = df["RPM"].std()
    features["rpm_max"] = df["RPM"].max()

    # RPM range utilization
    rpm_range = df["RPM"].max() - df["RPM"].min()
    features["rpm_range"] = rpm_range
    features["rpm_cv"] = df["RPM"].std() / max(df["RPM"].mean(), 1)  # coefficient of variation

    # Per-lap RPM consistency
    lap_rpm_std = df.groupby("LapNumber")["RPM"].std()
    features["rpm_intra_lap_std_mean"] = lap_rpm_std.mean()

    # --- Speed features ---
    features["speed_mean"] = df["Speed"].mean()
    features["speed_std"] = df["Speed"].std()
    features["speed_max"] = df["Speed"].max()

    # Speed consistency across laps
    lap_speed_mean = df.groupby("LapNumber")["Speed"].mean()
    features["speed_lap_to_lap_std"] = lap_speed_mean.std()  # how much avg speed varies between laps

    # --- Gear features ---
    features["gear_mean"] = df["nGear"].mean()
    features["gear_std"] = df["nGear"].std()

    # Gear shift count per lap
    gear_shifts = (df["nGear"].diff().abs() > 0).sum()
    features["gear_shifts_per_lap"] = gear_shifts / max(n_laps, 1)

    # Time in top gear (gear >= 7) — higher = more straight-line speed
    features["top_gear_ratio"] = (df["nGear"] >= 7).mean()

    # Gear distribution entropy (uniform = diverse usage, peaked = predictable)
    gear_counts = df["nGear"].value_counts(normalize=True)
    features["gear_entropy"] = -(gear_counts * np.log2(gear_counts + 1e-10)).sum()

    # --- DRS features ---
    features["drs_active_ratio"] = (df["DRS"] >= 10).mean()  # DRS values 10-14 = open
    features["drs_available_ratio"] = (df["DRS"] >= 8).mean()  # DRS values 8+ = available or open

    # --- Tyre-aware features ---
    # How features degrade with tyre life
    if "TyreLife" in df.columns and df["TyreLife"].nunique() > 3:
        # Split into fresh tyres (life <= 10) vs worn (life > 20)
        fresh = df[df["TyreLife"] <= 10]
        worn = df[df["TyreLife"] > 20]

        if len(fresh) > 100 and len(worn) > 100:
            features["speed_degradation"] = fresh["Speed"].mean() - worn["Speed"].mean()
            features["throttle_degradation"] = fresh["Throttle"].mean() - worn["Throttle"].mean()
            features["rpm_degradation"] = fresh["RPM"].mean() - worn["RPM"].mean()
            features["brake_increase_with_wear"] = worn["Brake"].mean() - fresh["Brake"].mean()
        else:
            features["speed_degradation"] = 0
            features["throttle_degradation"] = 0
            features["rpm_degradation"] = 0
            features["brake_increase_with_wear"] = 0

        # Tyre life correlation with speed (negative = degrading)
        valid = df.dropna(subset=["TyreLife", "Speed"])
        if len(valid) > 50:
            features["tyre_speed_correlation"] = valid["TyreLife"].corr(valid["Speed"])
        else:
            features["tyre_speed_correlation"] = 0
    else:
        features["speed_degradation"] = 0
        features["throttle_degradation"] = 0
        features["rpm_degradation"] = 0
        features["brake_increase_with_wear"] = 0
        features["tyre_speed_correlation"] = 0

    # --- Lap-to-lap consistency (overall car stability) ---
    lap_features = df.groupby("LapNumber").agg({
        "Speed": "mean",
        "RPM": "mean",
        "Throttle": "mean",
    })
    if len(lap_features) > 3:
        # Rolling std of lap-level metrics (instability signal)
        for col in ["Speed", "RPM", "Throttle"]:
            rolling_std = lap_features[col].rolling(3, min_periods=2).std()
            features[f"{col.lower()}_rolling_instability"] = rolling_std.mean()

        # Late-race degradation (last 25% vs first 25%)
        n = len(lap_features)
        q1 = lap_features.iloc[:n // 4]
        q4 = lap_features.iloc[-(n // 4):]
        if len(q1) > 0 and len(q4) > 0:
            features["late_race_speed_drop"] = q1["Speed"].mean() - q4["Speed"].mean()
            features["late_race_rpm_drop"] = q1["RPM"].mean() - q4["RPM"].mean()
    else:
        features["speed_rolling_instability"] = 0
        features["rpm_rolling_instability"] = 0
        features["throttle_rolling_instability"] = 0
        features["late_race_speed_drop"] = 0
        features["late_race_rpm_drop"] = 0

    return features


def classify_outcome(row: dict) -> str:
    """Classify a race outcome as good/bad based on position change and status."""
    status = row.get("status", "Finished")
    pos = int(row.get("position", 0))
    grid = int(row.get("grid", 0))
    gained = grid - pos

    if "Retired" in str(status) or "Accident" in str(status) or "Collision" in str(status):
        return "dnf"
    if "Lapped" in str(status) or (pos > 15 and gained < -10):
        return "bad"
    if gained <= -5:
        return "bad"
    if gained <= -3 and pos > 10:
        return "bad"
    return "good"


def _normalize_race_name(name: str) -> str:
    """Normalize race name for fuzzy matching."""
    return (name.lower()
            .replace("são paulo", "sao paulo")
            .replace("é", "e").replace("ã", "a")
            .replace("grand prix", "").strip())


def _match_race(csv_race: str, db_races: dict) -> str | None:
    """Find the best matching race name from DB outcomes."""
    csv_norm = _normalize_race_name(csv_race)
    for db_name in db_races:
        db_norm = _normalize_race_name(db_name)
        # Exact match after normalization
        if csv_norm == db_norm:
            return db_name
        # Substring match (e.g. "Emilia Romagna" in "Emilia Romagna")
        if csv_norm in db_norm or db_norm in csv_norm:
            return db_name
        # First two words match
        csv_words = csv_norm.split()[:2]
        db_words = db_norm.split()[:2]
        if csv_words == db_words:
            return db_name
    return None


def main():
    # Load race outcomes from MongoDB — both 2023 and 2024
    client = MongoClient("mongodb://localhost:27017")
    db = client["marip_f1"]

    outcomes = {}
    for season in [2023, 2024]:
        results = list(db.jolpica_race_results.find(
            {"season": season, "constructor_id": "mclaren"},
            {"round": 1, "driver_code": 1, "position": 1, "grid": 1, "status": 1,
             "race_name": 1, "points": 1, "laps": 1, "_id": 0}
        ).sort("round", 1))

        for r in results:
            key = (season, r["race_name"], r["driver_code"])
            outcome = classify_outcome(r)
            outcomes[key] = {
                "outcome": outcome,
                "grid": r.get("grid"),
                "position": r.get("position"),
                "gained": int(r.get("grid", 0)) - int(r.get("position", 0)),
                "status": r.get("status"),
                "season": season,
                "race_name": r["race_name"],
                "driver": r["driver_code"],
            }

    print(f"\n{'='*80}")
    print("McLaren 2023-2024 — Rich Telemetry Feature PoC")
    print(f"{'='*80}")

    good = sum(1 for v in outcomes.values() if v["outcome"] == "good")
    bad = sum(1 for v in outcomes.values() if v["outcome"] in ("bad", "dnf"))
    print(f"\nOutcomes: {good} good, {bad} bad/dnf")
    print("\nBad races:")
    for k, v in outcomes.items():
        if v["outcome"] in ("bad", "dnf"):
            print(f"  {v['season']} {v['race_name']:32s} {v['driver']} G{v['grid']}->P{v['position']} ({v['gained']:+d}) {v['status']}")

    # Build per-season race name lookup for matching
    db_races_by_season = {}
    for (season, race_name, driver), v in outcomes.items():
        db_races_by_season.setdefault(season, set()).add(race_name)

    # Process all race CSVs (2023 + 2024)
    all_features = []
    csv_files = sorted([f for f in os.listdir(CSV_DIR)
                        if (f.startswith("2023") or f.startswith("2024"))
                        and f.endswith(".csv") and "ALL_COMBINED" not in f])

    matched_count = 0
    unmatched = []
    for csv_file in csv_files:
        # Extract season and race name from filename
        season = int(csv_file[:4])
        race_name = csv_file[5:].replace("_Race.csv", "").replace("_", " ")
        filepath = os.path.join(CSV_DIR, csv_file)

        # Match to DB race name
        db_race_name = _match_race(race_name, db_races_by_season.get(season, set()))

        df = load_race_csv(filepath)
        drivers = df["Driver"].unique()

        for driver in drivers:
            driver_df = df[df["Driver"] == driver].copy()
            if len(driver_df) < 100:
                continue

            feats = compute_rich_features(driver_df)
            feats["race_name"] = race_name
            feats["driver"] = driver
            feats["csv_file"] = csv_file
            feats["season"] = season

            key = (season, db_race_name, driver) if db_race_name else None
            if key and key in outcomes:
                feats["outcome"] = outcomes[key]["outcome"]
                feats["grid"] = outcomes[key]["grid"]
                feats["position"] = outcomes[key]["position"]
                feats["gained"] = outcomes[key]["gained"]
                matched_count += 1
            else:
                feats["outcome"] = "unknown"
                feats["grid"] = None
                feats["position"] = None
                feats["gained"] = None
                if db_race_name is None:
                    unmatched.append(f"{season} {race_name}")

            all_features.append(feats)

    if unmatched:
        print(f"\nUnmatched CSVs: {set(unmatched)}")
    print(f"Matched {matched_count} driver-race entries")

    feat_df = pd.DataFrame(all_features)
    feat_df = feat_df[feat_df["outcome"] != "unknown"]

    print(f"\nProcessed {len(feat_df)} driver-races with matched outcomes")
    print(f"  Good: {(feat_df['outcome'] == 'good').sum()}")
    print(f"  Bad:  {(feat_df['outcome'].isin(['bad', 'dnf'])).sum()}")

    # --- Analyze discrimination ---
    numeric_cols = [c for c in feat_df.columns if feat_df[c].dtype in (np.float64, np.int64, float, int)
                    and c not in ("grid", "position", "gained")]

    good_df = feat_df[feat_df["outcome"] == "good"]
    bad_df = feat_df[feat_df["outcome"].isin(["bad", "dnf"])]

    print(f"\n{'='*80}")
    print("FEATURE DISCRIMINATION ANALYSIS")
    print(f"{'='*80}")
    print(f"\n{'Feature':<40s} {'Good Mean':>12s} {'Bad Mean':>12s} {'Diff':>10s} {'Effect':>8s} {'Disc?':>6s}")
    print("-" * 90)

    discriminating = []
    for col in sorted(numeric_cols):
        g_mean = good_df[col].mean()
        b_mean = bad_df[col].mean()
        pooled_std = feat_df[col].std()
        if pooled_std > 0:
            effect_size = abs(g_mean - b_mean) / pooled_std  # Cohen's d
        else:
            effect_size = 0

        diff = b_mean - g_mean
        disc = "YES" if effect_size > 0.3 else ""  # medium+ effect size
        if disc:
            discriminating.append((col, effect_size, g_mean, b_mean))

        print(f"{col:<40s} {g_mean:>12.3f} {b_mean:>12.3f} {diff:>+10.3f} {effect_size:>8.3f} {disc:>6s}")

    print(f"\n{'='*80}")
    print(f"DISCRIMINATING FEATURES (Cohen's d > 0.3)")
    print(f"{'='*80}")
    discriminating.sort(key=lambda x: -x[1])
    for col, d, gm, bm in discriminating:
        direction = "BAD higher" if bm > gm else "GOOD higher"
        print(f"  d={d:.3f}  {col:<40s}  ({direction})")

    if not discriminating:
        print("  NONE — these features don't discriminate either.")
        print("  The McLaren CSVs have the same limitation as openf1_car_data.")
    else:
        print(f"\n  Found {len(discriminating)} discriminating features!")
        print("  These can replace the generic anomaly health signal.")

    # --- Show per-race feature profile for bad races ---
    print(f"\n{'='*80}")
    print("BAD RACE FEATURE PROFILES")
    print(f"{'='*80}")

    top_features = [col for col, d, _, _ in discriminating[:8]] if discriminating else [
        "throttle_consistency", "brake_transitions_per_lap", "speed_lap_to_lap_std",
        "gear_entropy", "late_race_speed_drop", "speed_degradation",
        "throttle_intra_lap_std_std", "rpm_cv"
    ]

    for _, row in bad_df.iterrows():
        print(f"\n  {row['race_name']} — {row['driver']} (G{row['grid']}->P{row['position']}, {row['gained']:+d})")
        for col in top_features:
            if col in row:
                g_mean = good_df[col].mean()
                val = row[col]
                pct = (val - g_mean) / max(abs(g_mean), 1e-6) * 100
                marker = " ***" if abs(pct) > 20 else ""
                print(f"    {col:<40s} {val:>10.3f}  (vs good avg {g_mean:>10.3f}, {pct:+.0f}%){marker}")

    print(f"\n{'='*80}")
    print("DONE")


if __name__ == "__main__":
    main()
