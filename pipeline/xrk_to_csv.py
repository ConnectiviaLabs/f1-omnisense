#!/usr/bin/env python3
"""
AiM RaceStudio 3 XRK Binary File → CSV Converter
Parses proprietary .xrk telemetry files and exports to CSV.

Supports:
  - Single-sample (S) records: float16-encoded sensor data
  - Multi-sample (M) records: batched float16 IMU data
  - GPS data from GNFO/GPS sections
  - LAP markers
  - Session metadata (driver, track, date, vehicle)
"""
import struct
import sys
import os
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


# ── Channel name map (from CHS headers) ────────────────────────────────
CHANNEL_NAMES = {
    0: "MasterClock",
    1: "LapTime",
    2: "PredictiveTime",
    3: "BestRunDiff",
    4: "BestTodayDiff",
    5: "PrevLapDiff",
    6: "RefLapDiff",
    7: "RollTime",
    8: "BestTime",
    9: "LoggerTemp_C",
    10: "ExternalVoltage_V",
    11: "TotalOdometer",
    12: "ResetOdometer1",
    13: "ResetOdometer2",
    14: "ResetOdometer3",
    15: "ResetOdometer4",
    16: "OilPressure_bar",
    17: "FuelPressure_bar",
    18: "WaterTemp_C",
    19: "FrontBrake_bar",
    20: "RearBrake_bar",
    21: "OilTemp_C",
    22: "Lambda",
    23: "TPS_raw",
    24: "InlineAcc_g",
    25: "LateralAcc_g",
    26: "VerticalAcc_g",
    27: "RollRate_dps",
    28: "PitchRate_dps",
    29: "YawRate_dps",
    30: "Luminosity",
    31: "Throttle_pct",
    32: "AFR",
    33: "SV_OilPressure",
    34: "SV_FuelPressure",
    35: "SV_WaterTemp",
    36: "LowBattery",
    37: "OilPressureAlert",
    38: "WaterTempAlert",
    39: "OilTempAlert",
    40: "WaterLowTempAlert",
    41: "RPM",
    42: "TPS",
    43: "Gear",
    44: "BatteryVoltage_V",
}

# Channels with known-good float16 encoding
FLOAT16_CHANNELS = {9, 10, 16, 17, 18, 19, 20, 21, 22, 23, 30}
# Channels with 4-byte uint32 values
UINT32_CHANNELS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15}
# Channels with 4-byte float32 values
FLOAT32_CHANNELS = {31, 32}
# IMU channels in (M) records
IMU_CHANNELS = {24, 25, 26, 27, 28, 29}

# Post-processing scaling for specific channels
CHANNEL_SCALE = {
    10: 1.0 / 1000.0,  # VBat: float16 gives millivolts, convert to volts
}


def parse_metadata(data: bytes) -> dict:
    """Extract session metadata from XRK file."""
    meta = {}

    # Driver name from <hRCR section
    m = re.search(b'<hRCR[\\s\\S]{7}([A-Za-z0-9 _.]+)', data[:10000])
    if m:
        meta["driver"] = m.group(1).decode("ascii", errors="replace").strip()

    # Track name from <hTRK section
    m = re.search(b'<hTRK[\\s\\S]{7}([A-Za-z0-9 _.]+)', data[:10000])
    if m:
        meta["track"] = m.group(1).decode("ascii", errors="replace").strip()

    # Date from <hTMD section
    m = re.search(b'<hTMD[\\s\\S]{5}([0-9/]+)', data[:10000])
    if m:
        meta["date"] = m.group(1).decode("ascii", errors="replace").strip()

    # Time from <hTMT section
    m = re.search(b'<hTMT[\\s\\S]{5}([0-9:]+)', data[:10000])
    if m:
        meta["time"] = m.group(1).decode("ascii", errors="replace").strip()

    # Manufacturer from <hMANL section
    m = re.search(b'<hMANL[\\s\\S]{5}([A-Za-z0-9 _]+)', data[:10000])
    if m:
        meta["manufacturer"] = m.group(1).decode("ascii", errors="replace").strip()

    # Model from <hMODL section
    m = re.search(b'<hMODL[\\s\\S]{5}([A-Za-z0-9 _]+)', data[:10000])
    if m:
        meta["model"] = m.group(1).decode("ascii", errors="replace").strip()

    return meta


def find_record_positions(data: bytes) -> list:
    """Find all data record start positions: (S, (G, (MI, (M."""
    positions = []
    for m in re.finditer(b"\\((?:MI|S|G|M)", data):
        positions.append((m.start(), m.group()))
    return positions


def parse_s_records(data: bytes, record_positions: list) -> dict:
    """Parse all (S) single-sample records. Returns {channel: [(ts, value), ...]}."""
    channels = defaultdict(list)

    for i, (pos, tag) in enumerate(record_positions):
        if tag != b"(S":
            continue

        payload_start = pos + 2
        if i + 1 < len(record_positions):
            next_pos = record_positions[i + 1][0]
            payload = data[payload_start : next_pos - 1]
        else:
            end = data.find(b")", payload_start)
            if end < 0:
                continue
            payload = data[payload_start:end]

        if len(payload) < 6:
            continue

        ts = struct.unpack_from("<I", payload, 0)[0]
        ch = struct.unpack_from("<H", payload, 4)[0]
        val_bytes = payload[6:]

        if ch > 50:
            continue  # spurious channel ID

        if len(val_bytes) == 2:
            val = float(np.frombuffer(val_bytes, dtype=np.float16)[0])
        elif len(val_bytes) == 4:
            if ch in UINT32_CHANNELS:
                val = float(struct.unpack_from("<I", val_bytes)[0])
            elif ch in FLOAT32_CHANNELS:
                val = float(struct.unpack_from("<f", val_bytes)[0])
            else:
                # Try float32 first; fall back to float16 of first 2 bytes
                val = float(struct.unpack_from("<f", val_bytes)[0])
        elif len(val_bytes) == 0:
            continue
        else:
            continue

        if not np.isfinite(val):
            continue

        # Apply per-channel scaling
        if ch in CHANNEL_SCALE:
            val *= CHANNEL_SCALE[ch]

        channels[ch].append((ts, val))

    return channels


def parse_m_records(data: bytes, record_positions: list) -> dict:
    """Parse all (M) multi-sample records for IMU channels."""
    channels = defaultdict(list)

    for i, (pos, tag) in enumerate(record_positions):
        if tag != b"(M":
            continue

        payload_start = pos + 2
        if i + 1 < len(record_positions):
            next_pos = record_positions[i + 1][0]
            payload = data[payload_start : next_pos - 1]
        else:
            end = data.find(b")", payload_start)
            if end < 0:
                continue
            payload = data[payload_start:end]

        if len(payload) < 8:
            continue

        ts_lo = struct.unpack_from("<H", payload, 0)[0]
        ts_hi = struct.unpack_from("<H", payload, 2)[0]
        ch = struct.unpack_from("<H", payload, 4)[0]
        count = struct.unpack_from("<H", payload, 6)[0]

        if ch > 50 or ch not in IMU_CHANNELS:
            continue

        ts_base = ts_lo | (ts_hi << 16)

        values = []
        for j in range(8, len(payload) - 1, 2):
            if j + 2 <= len(payload):
                val = float(np.frombuffer(payload[j : j + 2], dtype=np.float16)[0])
                if np.isfinite(val) and abs(val) < 1000:
                    values.append(val)

        if not values:
            continue

        n = len(values)
        # Distribute samples evenly across the batch time window
        # AiM batches IMU at ~200Hz in groups of 10-20
        sample_interval = max(1, count // n) if count > 0 else 1
        for k, v in enumerate(values):
            channels[ch].append((ts_base + k * sample_interval, v))

    return channels


def parse_gps_data(data: bytes) -> list:
    """Parse GPS data from GNFO/GPS sections.

    GPS data layout (56 bytes, 14 × int32 fields):
      [0]  timestamp_ms
      [4]  monotonic counter (not a coordinate)
      [8]  distance/altitude field
      [12] variable field
      [16] latitude  (signed int32, degrees × 1e-7)
      [20] longitude (signed int32, degrees × 1e-7)
      [24] altitude-related field
      [28] HDOP or heading
      [32] velocity north component (signed, cm/s)
      [36] velocity east component (signed, cm/s)
      [40] velocity down component (signed, cm/s)
      [44] satellite count
    """
    gps_records = []

    for m in re.finditer(b"<hGNFO", data):
        gnfo_pos = m.start()

        # Find the GPS data section
        gps_pos = data.find(b"<hGPS", gnfo_pos + 6)
        if gps_pos < 0 or gps_pos > gnfo_pos + 2000:
            continue

        gps_d = data[gps_pos + 12 : gps_pos + 80]
        if len(gps_d) < 48:
            continue

        ts = struct.unpack_from("<I", gps_d, 0)[0]

        # Coordinates (signed int32, degrees × 1e-7)
        lat_raw = struct.unpack_from("<i", gps_d, 16)[0]
        lon_raw = struct.unpack_from("<i", gps_d, 20)[0]

        # Velocity components (signed int32, cm/s)
        v_north = struct.unpack_from("<i", gps_d, 32)[0]
        v_east = struct.unpack_from("<i", gps_d, 36)[0]

        # Compute speed from velocity components
        speed_cms = np.sqrt(v_north**2 + v_east**2)
        speed_kmh = speed_cms / 100.0 * 3.6  # cm/s → km/h

        # Compute heading from velocity (0=North, 90=East)
        heading = np.degrees(np.arctan2(v_east, v_north)) % 360

        lat = lat_raw / 1e7
        lon = lon_raw / 1e7

        # Satellite count
        sats = struct.unpack_from("<I", gps_d, 44)[0]

        gps_records.append({
            "timestamp_ms": ts,
            "GPS_Lat": lat,
            "GPS_Lon": lon,
            "GPS_Speed_kmh": round(speed_kmh, 2),
            "GPS_Heading_deg": round(heading, 1),
            "GPS_Satellites": min(sats, 30),  # cap unreasonable values
        })

    return gps_records


def parse_laps(data: bytes) -> list:
    """Parse LAP markers."""
    laps = []
    for m in re.finditer(b"<hLAP", data):
        pos = m.start()
        hdr = data[pos + 12 : pos + 32]
        if len(hdr) < 12:
            continue
        lap_num = struct.unpack_from("<H", hdr, 2)[0]
        lap_time_raw = struct.unpack_from("<I", hdr, 4)[0]
        laps.append({
            "lap_number": lap_num,
            "lap_time_ms": lap_time_raw,
            "lap_time_s": lap_time_raw / 1000.0,
        })
    return laps


def build_telemetry_dataframe(s_channels, m_channels, gps_data):
    """Merge all channels into a single DataFrame on timestamp."""
    all_dfs = []

    # Process (S) record channels
    for ch, records in s_channels.items():
        name = CHANNEL_NAMES.get(ch, f"ch{ch}")
        if ch == 0:
            continue  # skip MasterClock (it's redundant with timestamp)
        ts_vals = [(ts, val) for ts, val in records]
        if ts_vals:
            df = pd.DataFrame(ts_vals, columns=["timestamp_ms", name])
            df = df.drop_duplicates(subset="timestamp_ms").sort_values("timestamp_ms")
            all_dfs.append(df)

    # Process (M) record channels (IMU)
    for ch, records in m_channels.items():
        name = CHANNEL_NAMES.get(ch, f"ch{ch}")
        ts_vals = [(ts, val) for ts, val in records]
        if ts_vals:
            df = pd.DataFrame(ts_vals, columns=["timestamp_ms", name])
            df = df.drop_duplicates(subset="timestamp_ms").sort_values("timestamp_ms")
            all_dfs.append(df)

    # Process GPS data
    if gps_data:
        gps_df = pd.DataFrame(gps_data)
        gps_df = gps_df.drop_duplicates(subset="timestamp_ms").sort_values("timestamp_ms")
        all_dfs.append(gps_df)

    if not all_dfs:
        return pd.DataFrame()

    # Merge all channels on timestamp using outer join
    result = all_dfs[0]
    for df in all_dfs[1:]:
        result = result.merge(df, on="timestamp_ms", how="outer")

    result = result.sort_values("timestamp_ms").reset_index(drop=True)

    # Add time in seconds column
    result.insert(1, "time_s", result["timestamp_ms"] / 1000.0)

    return result


def convert_xrk_to_csv(input_path: str, output_dir: str = None):
    """Main conversion function."""
    input_path = Path(input_path)
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem

    print(f"Reading {input_path.name} ({input_path.stat().st_size / 1024:.0f} KB)...")
    with open(input_path, "rb") as f:
        data = f.read()

    # ── Extract metadata ──
    meta = parse_metadata(data)
    print(f"  Driver: {meta.get('driver', 'Unknown')}")
    print(f"  Track:  {meta.get('track', 'Unknown')}")
    print(f"  Date:   {meta.get('date', 'Unknown')} {meta.get('time', '')}")
    print(f"  Vehicle: {meta.get('manufacturer', '')} {meta.get('model', '')}")

    # ── Find all record positions ──
    print("Scanning records...")
    record_positions = find_record_positions(data)
    print(f"  Found {len(record_positions)} data records")

    # ── Parse channels ──
    print("Parsing sensor data (S records)...")
    s_channels = parse_s_records(data, record_positions)
    active_s = {ch: len(recs) for ch, recs in s_channels.items() if recs}
    for ch, count in sorted(active_s.items()):
        name = CHANNEL_NAMES.get(ch, f"ch{ch}")
        vals = [v for _, v in s_channels[ch][:100]]
        print(f"    {name:25s}: {count:7d} samples  range=[{min(vals):.3f}, {max(vals):.3f}]")

    print("Parsing IMU data (M records)...")
    m_channels = parse_m_records(data, record_positions)
    active_m = {ch: len(recs) for ch, recs in m_channels.items() if recs}
    for ch, count in sorted(active_m.items()):
        name = CHANNEL_NAMES.get(ch, f"ch{ch}")
        vals = [v for _, v in m_channels[ch][:100]]
        print(f"    {name:25s}: {count:7d} samples  range=[{min(vals):.3f}, {max(vals):.3f}]")

    print("Parsing GPS data...")
    gps_data = parse_gps_data(data)
    print(f"    GPS fixes: {len(gps_data)}")

    print("Parsing lap data...")
    laps = parse_laps(data)
    for lap in laps:
        print(f"    Lap {lap['lap_number']}: {lap['lap_time_s']:.3f}s")

    # ── Build merged DataFrame ──
    print("Building merged telemetry DataFrame...")
    df = build_telemetry_dataframe(s_channels, m_channels, gps_data)
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # ── Add lap number column ──
    if laps and not df.empty:
        lap_starts = [0]  # session start
        cumulative = 0
        for lap in laps:
            cumulative += lap["lap_time_ms"]
            lap_starts.append(cumulative)
        df["lap"] = 0
        for i, start in enumerate(lap_starts):
            df.loc[df["timestamp_ms"] >= start, "lap"] = i + 1

    # ── Write outputs ──
    # 1. Main telemetry CSV
    telemetry_path = output_dir / f"{stem}_telemetry.csv"
    df.to_csv(telemetry_path, index=False, float_format="%.4f")
    print(f"\n  Telemetry CSV: {telemetry_path}")
    print(f"    {df.shape[0]} rows × {df.shape[1]} columns")

    # 2. Laps CSV
    if laps:
        laps_path = output_dir / f"{stem}_laps.csv"
        laps_df = pd.DataFrame(laps)
        laps_df.to_csv(laps_path, index=False)
        print(f"  Laps CSV:      {laps_path}")

    # 3. Metadata CSV
    meta_path = output_dir / f"{stem}_metadata.csv"
    meta_df = pd.DataFrame([meta])
    meta_df.to_csv(meta_path, index=False)
    print(f"  Metadata CSV:  {meta_path}")

    # 4. Summary
    print(f"\nActive channels in output:")
    for col in df.columns:
        if col not in ("timestamp_ms", "time_s", "lap"):
            valid = df[col].dropna()
            if len(valid) > 0:
                print(f"    {col:30s}: {len(valid):7d} points  [{valid.min():.3f} → {valid.max():.3f}]")

    return df, laps, meta


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default to the file in the same directory
        xrk_files = list(Path(__file__).parent.glob("*.xrk"))
        if xrk_files:
            input_file = str(xrk_files[0])
        else:
            print("Usage: python xrk_to_csv.py <file.xrk> [output_dir]")
            sys.exit(1)
    else:
        input_file = sys.argv[1]

    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    convert_xrk_to_csv(input_file, output_dir)
