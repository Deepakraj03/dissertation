"""
Rover NAVCAM image downloader — Curiosity MSL via JPL Planetary Data Server.

Downloads raw navigation camera JPEGs (511×511 px grayscale) from the
MSL NAVCAM archive at planetarydata.jpl.nasa.gov. No API key required.

Directory structure on the server:
  /img/data/msl/msl_navcam_raw/EXTRAS/FULL/SOL{XXXXX}/
    NLA_XXXXXX.JPG   (left NAVCAM)
    NRA_XXXXXX.JPG   (right NAVCAM)

Usage:
    python download_rover.py               # 2000 images, default diversity sol sample
    python download_rover.py --max 500     # quick test
    python download_rover.py --workers 8   # more parallel threads
"""

import argparse
import json
import random
import re
import time
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "Data" / "Rover" / "raw"
PDS_BASE = "https://planetarydata.jpl.nasa.gov/img/data/msl/msl_navcam_raw/EXTRAS/FULL"

# Curiosity active sols: SOL00001–SOL04709 (7832 directories confirmed)
# Sample every Nth sol for terrain diversity across the mission
SOL_SAMPLE_STRIDE = 15   # one sol in fifteen → covers full mission geography

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"})


def list_sol_dirs() -> list[str]:
    """Return list of all SOL directory names from the PDS index."""
    try:
        r = SESSION.get(PDS_BASE + "/", timeout=30)
        r.raise_for_status()
        dirs = re.findall(r'href="(SOL\d+/)"', r.text)
        return sorted(dirs)
    except Exception as e:
        print(f"  [WARN] Could not list SOL directories: {e}")
        return []


def list_images_in_sol(sol_dir: str) -> list[str]:
    """Return deduplicated list of JPEG filenames in a given SOL directory."""
    url = f"{PDS_BASE}/{sol_dir}"
    try:
        r = SESSION.get(url, timeout=20)
        r.raise_for_status()
        imgs = re.findall(r'href="(N[LR]A_\w+\.JPG)"', r.text)
        return list(dict.fromkeys(imgs))  # preserve order, remove duplicates
    except Exception:
        return []


def download_image(url: str, out_path: Path) -> bool:
    """Download a single JPEG with retry on 503. Returns True on success."""
    if out_path.exists() and out_path.stat().st_size > 50_000:
        return True
    for attempt in range(4):
        try:
            r = SESSION.get(url, timeout=60, stream=True)
            if r.status_code == 503:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            content = r.content
            if not content[:3] == b"\xff\xd8\xff":
                return False
            out_path.write_bytes(content)
            if out_path.stat().st_size < 50_000:
                out_path.unlink()
                return False
            return True
        except Exception as e:
            if attempt == 3:
                print(f"  [FAIL] {url}: {e}")
            time.sleep(2 ** attempt)
    return False


def collect_targets(sol_dirs: list[str], max_images: int,
                    images_per_sol: int = 8) -> list[tuple[str, Path]]:
    """
    Sample sol directories and collect (url, out_path) tuples.
    Caps at images_per_sol per sol for terrain diversity.
    """
    tasks: list[tuple[str, Path]] = []
    curiosity_dir = DATA_DIR / "curiosity"
    curiosity_dir.mkdir(parents=True, exist_ok=True)

    # Stride-sample for diversity; shuffle so partial runs vary
    sampled = sol_dirs[::SOL_SAMPLE_STRIDE]
    random.shuffle(sampled)

    print(f"Sampling from {len(sampled)} of {len(sol_dirs)} sols (stride={SOL_SAMPLE_STRIDE})")

    for sol_dir in sampled:
        if len(tasks) >= max_images:
            break
        imgs = list_images_in_sol(sol_dir)
        random.shuffle(imgs)
        for fname in imgs[:images_per_sol]:
            if len(tasks) >= max_images:
                break
            url      = f"{PDS_BASE}/{sol_dir}{fname}"
            out_path = curiosity_dir / f"{sol_dir.rstrip('/')}_{fname}"
            tasks.append((url, out_path))
        time.sleep(0.2)

    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max",     type=int, default=2000,
                        help="Max images to download")
    parser.add_argument("--workers", type=int, default=3,
                        help="Parallel download threads")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = DATA_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    already_paths = {m["path"] for m in manifest}

    print("Listing sol directories from JPL PDS…")
    sol_dirs = list_sol_dirs()
    if not sol_dirs:
        print("ERROR: Could not reach PDS server. Check network.")
        return
    print(f"Found {len(sol_dirs)} sol directories ({sol_dirs[0]} … {sol_dirs[-1]})")

    print("\nBuilding download queue…")
    tasks = collect_targets(sol_dirs, max_images=args.max)
    tasks = [(u, p) for u, p in tasks if str(p) not in already_paths]
    print(f"New downloads: {len(tasks)}")

    if not tasks:
        print("Nothing new to download.")
        return

    # Sequential download with throttle — PDS server allows ~1 req/s
    print(f"\nDownloading {len(tasks)} images (sequential, 1 req/s throttle)…\n")
    downloaded, failed = 0, 0
    for i, (url, path) in enumerate(tasks, 1):
        ok = download_image(url, path)
        if ok:
            downloaded += 1
            sol = path.name.split("_")[0]
            cam = "left" if "_NLA_" in path.name else "right"
            manifest.append({
                "path":   str(path),
                "sol":    sol,
                "camera": cam,
                "size_kb": path.stat().st_size // 1024,
            })
        else:
            failed += 1
        time.sleep(0.8)   # ~1 req/s — stays within PDS server limits
        if i % 50 == 0 or i == len(tasks):
            print(f"  {i}/{len(tasks)} — {downloaded} ok, {failed} failed")

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nDone. {downloaded} downloaded, {failed} failed.")
    print(f"Images in: {DATA_DIR}")
    print(f"Manifest:  {manifest_path}")


if __name__ == "__main__":
    main()
