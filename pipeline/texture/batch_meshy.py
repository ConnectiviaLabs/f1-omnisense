"""Batch 3D generation for all non-McLaren F1 2025 cars via Meshy.ai.

Usage:
    python pipeline/texture/batch_meshy.py [--copy-to-public]

Generates GLBs from image_1.jpg in each team's output/3d_models/<team>/ folder.
Use --copy-to-public to also copy the resulting GLBs to frontend/public/models/.
"""

import logging
import shutil
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "output" / "3d_models"
PUBLIC_MODELS = PROJECT_ROOT / "frontend" / "public" / "models"

# Load .env for MESHY_API_KEY
load_dotenv(PROJECT_ROOT / ".env")

TEAMS = {
    "red_bull":     {"name": "Red Bull RB21",            "public": "red_bull.glb"},
    "ferrari":      {"name": "Ferrari SF-25",            "public": "ferrari.glb"},
    "mercedes":     {"name": "Mercedes W16",             "public": "mercedes.glb"},
    "aston_martin": {"name": "Aston Martin AMR25",       "public": "aston_martin.glb"},
    "alpine":       {"name": "Alpine A525",              "public": "alpine.glb"},
    "williams":     {"name": "Williams FW47",            "public": "williams.glb"},
    "rb":           {"name": "Racing Bulls VCARB 02",    "public": "rb.glb"},
    "sauber":       {"name": "Sauber C45",               "public": "sauber.glb"},
    "haas":         {"name": "Haas VF-25",               "public": "haas.glb"},
}

TEAM_ORDER = list(TEAMS.keys())

# Texture prompts per team for better results
TEXTURE_PROMPTS = {
    "red_bull":     "Formula 1 car, Red Bull Racing dark blue livery with yellow accents",
    "ferrari":      "Formula 1 car, Ferrari red livery with black and yellow details",
    "mercedes":     "Formula 1 car, Mercedes silver and black livery with teal accents",
    "aston_martin": "Formula 1 car, Aston Martin British racing green livery",
    "alpine":       "Formula 1 car, Alpine blue and pink livery",
    "williams":     "Formula 1 car, Williams dark blue and light blue livery",
    "rb":           "Formula 1 car, Racing Bulls dark blue and silver livery",
    "sauber":       "Formula 1 car, Sauber green and black Stake livery",
    "haas":         "Formula 1 car, Haas white and red livery with black accents",
}


def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{seconds / 60:.1f}m"


def main():
    copy_to_public = "--copy-to-public" in sys.argv

    from meshy_client import get_meshy_client

    print()
    print("+" + "=" * 58 + "+")
    print("|  Meshy.ai Batch Generator -- F1 2025 Cars               |")
    print("+" + "=" * 58 + "+")
    print(f"  Teams:  {len(TEAMS)}")
    print(f"  Output: {MODELS_DIR}")
    if copy_to_public:
        print(f"  Copy:   {PUBLIC_MODELS}")
    print()

    print("  [1/4] Connecting to Meshy.ai...")
    try:
        client = get_meshy_client()
    except ValueError as e:
        print(f"  X {e}")
        sys.exit(1)
    print("  OK Connected to Meshy.ai API")
    print()

    # Pre-scan what needs generating
    print("  [2/4] Scanning teams...")
    to_generate = []
    skipped = []
    for team_id in TEAM_ORDER:
        team_dir = MODELS_DIR / team_id
        input_img = team_dir / "image_1.jpg"
        glb_path = team_dir / "meshy.glb"

        if not input_img.exists():
            print(f"    -  {TEAMS[team_id]['name']:30s}  -- no input image, skipping")
            skipped.append((team_id, "no image"))
            continue
        if glb_path.exists():
            sz = glb_path.stat().st_size / 1e6
            print(f"    OK {TEAMS[team_id]['name']:30s}  -- already generated ({sz:.1f} MB)")
            skipped.append((team_id, f"exists ({sz:.1f} MB)"))
            continue

        img_kb = input_img.stat().st_size / 1024
        print(f"    *  {TEAMS[team_id]['name']:30s}  -- queued (image: {img_kb:.0f} KB)")
        to_generate.append(team_id)

    print()
    if not to_generate:
        print("  Nothing to generate -- all teams already have Meshy GLBs.")
        return

    print(f"  [3/4] Generating {len(to_generate)} models via Meshy.ai...")
    print(f"         Estimated time: ~{len(to_generate) * 3}-{len(to_generate) * 6} minutes")
    print(f"         (depends on Meshy queue)")
    print()

    results = {}
    batch_start = time.time()

    for idx, team_id in enumerate(to_generate, 1):
        team_dir = MODELS_DIR / team_id
        input_img = team_dir / "image_1.jpg"
        glb_path = team_dir / "meshy.glb"
        car_name = TEAMS[team_id]["name"]
        texture_prompt = TEXTURE_PROMPTS.get(team_id, "Formula 1 race car")

        print(f"  +-- [{idx}/{len(to_generate)}] {car_name}")
        print(f"  |  Input:   {input_img.name} ({input_img.stat().st_size / 1024:.0f} KB)")
        print(f"  |  Output:  {team_id}/meshy.glb")
        print(f"  |  Prompt:  {texture_prompt[:60]}...")

        start = time.time()

        try:
            print(f"  |  > Creating Meshy task...")
            sys.stdout.flush()

            task_id = client.create_task(
                image_path=input_img,
                texture_prompt=texture_prompt,
                enable_pbr=True,
                target_polycount=30000,
            )
            print(f"  |  > Task ID: {task_id}")

            def on_progress(status, pct):
                print(f"  |  > {status}: {pct}%")
                sys.stdout.flush()

            result_path = client.wait_and_download(
                task_id=task_id,
                output_path=glb_path,
                timeout=600,  # 10 min per car
                poll_interval=10,
                progress_callback=on_progress,
            )

            elapsed = time.time() - start
            size_mb = result_path.stat().st_size / 1e6
            print(f"  |  OK GLB: {size_mb:.1f} MB")

            # Optionally copy to public/models/
            if copy_to_public:
                public_name = TEAMS[team_id]["public"]
                dest = PUBLIC_MODELS / public_name
                shutil.copy2(result_path, dest)
                print(f"  |  OK Copied to public/models/{public_name}")

            remaining = len(to_generate) - idx
            avg_per = elapsed if idx == 1 else (time.time() - batch_start) / idx
            eta = avg_per * remaining

            print(f"  +-- Done in {_fmt_time(elapsed)}" +
                  (f" | ETA remaining: ~{_fmt_time(eta)}" if remaining > 0 else ""))
            print()

            results[team_id] = ("ok", elapsed, size_mb)

        except Exception as e:
            elapsed = time.time() - start
            print(f"  |  X FAILED: {e}")
            print(f"  +-- Failed after {_fmt_time(elapsed)}")
            print()
            results[team_id] = ("failed", elapsed, str(e))

    # Summary
    total_time = time.time() - batch_start
    total_size = sum(r[2] for r in results.values() if r[0] == "ok")
    ok_count = sum(1 for r in results.values() if r[0] == "ok")
    fail_count = sum(1 for r in results.values() if r[0] == "failed")

    print()
    print("+" + "=" * 58 + "+")
    print("|  [4/4] BATCH COMPLETE                                   |")
    print("+" + "-" * 58 + "+")
    print(f"|  Generated: {ok_count}/{len(to_generate)} models" +
          (f" ({fail_count} failed)" if fail_count > 0 else ""))
    print(f"|  Total size: {total_size:.1f} MB")
    print(f"|  Total time: {_fmt_time(total_time)}")
    print("+" + "-" * 58 + "+")

    for team_id in TEAM_ORDER:
        car = TEAMS[team_id]["name"]
        if team_id in results:
            status, elapsed, extra = results[team_id]
            if status == "ok":
                print(f"|  OK {car:28s} {extra:5.1f} MB  {_fmt_time(elapsed)}")
            else:
                print(f"|  X  {car:28s} FAILED     {_fmt_time(elapsed)}")
        elif any(s[0] == team_id for s in skipped):
            reason = next(s[1] for s in skipped if s[0] == team_id)
            print(f"|  -  {car:28s} {reason}")

    print("+" + "=" * 58 + "+")
    print()


if __name__ == "__main__":
    main()
