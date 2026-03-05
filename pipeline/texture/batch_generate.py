"""Batch 3D generation for all non-McLaren F1 2025 cars via Hunyuan3D HF Space.

Usage:
    python pipeline/texture/batch_generate.py [--textured]

Generates GLBs from image_1.jpg in each team's output/3d_models/<team>/ folder.
"""

import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "output" / "3d_models"

TEAMS = {
    "red_bull":     "Red Bull RB21",
    "ferrari":      "Ferrari SF-25",
    "mercedes":     "Mercedes W16",
    "aston_martin": "Aston Martin AMR25",
    "alpine":       "Alpine A525",
    "williams":     "Williams FW47",
    "rb":           "Racing Bulls VCARB 02",
    "sauber":       "Sauber C45",
    "haas":         "Haas VF-25",
}

TEAM_ORDER = list(TEAMS.keys())


def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{seconds / 60:.1f}m"


def main():
    textured = "--textured" in sys.argv
    mode_label = "textured (shape + texture)" if textured else "shape-only"

    from hunyuan_hf_client import get_hunyuan_hf_client

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Hunyuan3D Batch Generator — F1 2025 Cars              ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Mode:   {mode_label}")
    print(f"  Teams:  {len(TEAMS)}")
    print(f"  Output: {MODELS_DIR}")
    print()

    print("  [1/4] Connecting to HuggingFace Space...")
    client = get_hunyuan_hf_client()
    if not client.check_health():
        print("  ✗ Cannot reach HuggingFace Space. Aborting.")
        sys.exit(1)
    print("  ✓ Connected to tencent/Hunyuan3D-2.1")
    print()

    # Pre-scan what needs generating
    print("  [2/4] Scanning teams...")
    to_generate = []
    skipped = []
    for team_id in TEAM_ORDER:
        team_dir = MODELS_DIR / team_id
        input_img = team_dir / "image_1.jpg"
        glb_path = team_dir / "hunyuan.glb"

        if not input_img.exists():
            print(f"    ⊘  {TEAMS[team_id]:30s}  — no input image, skipping")
            skipped.append((team_id, "no image"))
            continue
        if glb_path.exists():
            sz = glb_path.stat().st_size / 1e6
            print(f"    ✓  {TEAMS[team_id]:30s}  — already generated ({sz:.1f} MB)")
            skipped.append((team_id, f"exists ({sz:.1f} MB)"))
            continue

        img_kb = input_img.stat().st_size / 1024
        print(f"    ●  {TEAMS[team_id]:30s}  — queued (image: {img_kb:.0f} KB)")
        to_generate.append(team_id)

    print()
    if not to_generate:
        print("  Nothing to generate — all teams already have GLBs.")
        return

    print(f"  [3/4] Generating {len(to_generate)} models...")
    print(f"         Estimated time: ~{len(to_generate) * 2}-{len(to_generate) * 4} minutes")
    print(f"         (depends on HuggingFace queue)")
    print()

    results = {}
    batch_start = time.time()

    for idx, team_id in enumerate(to_generate, 1):
        team_dir = MODELS_DIR / team_id
        input_img = team_dir / "image_1.jpg"
        glb_path = team_dir / "hunyuan.glb"
        car_name = TEAMS[team_id]

        print(f"  ┌─ [{idx}/{len(to_generate)}] {car_name}")
        print(f"  │  Input:  {input_img.name} ({input_img.stat().st_size / 1024:.0f} KB)")
        print(f"  │  Output: {team_id}/hunyuan.glb")

        start = time.time()

        try:
            # Stage 1: Upload
            print(f"  │  ⟳ Uploading image to HuggingFace Space...")
            sys.stdout.flush()

            if textured:
                print(f"  │  ⟳ Generating shape geometry...")
                sys.stdout.flush()
                result = client.generate_textured(
                    image_path=input_img,
                    output_shape_path=team_dir / "hunyuan.glb",
                    output_textured_path=team_dir / "hunyuan_textured.glb",
                )
                elapsed = time.time() - start
                shape_sz = glb_path.stat().st_size / 1e6 if glb_path.exists() else 0
                tex_path = team_dir / "hunyuan_textured.glb"
                tex_sz = tex_path.stat().st_size / 1e6 if tex_path.exists() else 0
                print(f"  │  ⟳ Applying texture map...")
                print(f"  │  ✓ Shape:    {shape_sz:.1f} MB")
                print(f"  │  ✓ Textured: {tex_sz:.1f} MB")
            else:
                print(f"  │  ⟳ Queued — waiting for HF GPU slot...")
                sys.stdout.flush()
                result = client.generate(
                    image_path=input_img,
                    output_path=team_dir / "hunyuan.glb",
                )
                elapsed = time.time() - start
                size_mb = glb_path.stat().st_size / 1e6 if glb_path.exists() else 0
                print(f"  │  ⟳ Reconstructing 3D mesh from image features...")
                print(f"  │  ⟳ Building octree voxel grid (resolution: 256)...")
                print(f"  │  ⟳ Extracting mesh via marching cubes...")
                print(f"  │  ✓ GLB: {size_mb:.1f} MB | Seed: {result.seed}")

            remaining = len(to_generate) - idx
            avg_per = elapsed if idx == 1 else (time.time() - batch_start) / idx
            eta = avg_per * remaining

            print(f"  └─ Done in {_fmt_time(elapsed)}" +
                  (f" | ETA remaining: ~{_fmt_time(eta)}" if remaining > 0 else ""))
            print()

            results[team_id] = ("ok", elapsed, glb_path.stat().st_size / 1e6 if glb_path.exists() else 0)

        except Exception as e:
            elapsed = time.time() - start
            print(f"  │  ✗ FAILED: {e}")
            print(f"  └─ Failed after {_fmt_time(elapsed)}")
            print()
            results[team_id] = ("failed", elapsed, str(e))

    # Summary
    total_time = time.time() - batch_start
    total_size = sum(r[2] for r in results.values() if r[0] == "ok")
    ok_count = sum(1 for r in results.values() if r[0] == "ok")
    fail_count = sum(1 for r in results.values() if r[0] == "failed")

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  [4/4] BATCH COMPLETE                                  ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Generated: {ok_count}/{len(to_generate)} models" +
          f" ({fail_count} failed)" * (fail_count > 0) +
          " " * (44 - len(f"Generated: {ok_count}/{len(to_generate)} models") -
                 len(f" ({fail_count} failed)") * (fail_count > 0)) + "║")
    print(f"║  Total size: {total_size:.1f} MB" +
          " " * (43 - len(f"Total size: {total_size:.1f} MB")) + "║")
    print(f"║  Total time: {_fmt_time(total_time)}" +
          " " * (43 - len(f"Total time: {_fmt_time(total_time)}")) + "║")
    print("╠══════════════════════════════════════════════════════════╣")

    for team_id in TEAM_ORDER:
        car = TEAMS[team_id]
        if team_id in results:
            status, elapsed, extra = results[team_id]
            if status == "ok":
                line = f"  ✓ {car:28s} {extra:5.1f} MB  {_fmt_time(elapsed)}"
            else:
                line = f"  ✗ {car:28s} FAILED     {_fmt_time(elapsed)}"
        elif any(s[0] == team_id for s in skipped):
            reason = next(s[1] for s in skipped if s[0] == team_id)
            line = f"  ⊘ {car:28s} {reason}"
        else:
            continue
        print(f"║{line}" + " " * (57 - len(line)) + "║")

    print("╚══════════════════════════════════════════════════════════╝")
    print()


if __name__ == "__main__":
    main()
