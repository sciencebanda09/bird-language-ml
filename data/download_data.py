"""
Bird Language ML - Data Acquisition
Downloads bird audio from xeno-canto API and prepares metadata CSV.

xeno-canto is the world's largest open repository of bird sounds:
  - 700,000+ recordings, 10,000+ species
  - Creative Commons licensed
  - Free API: https://xeno-canto.org/explore/api

Usage:
    python data/download_data.py --species "House Sparrow" "Common Myna" --max_per_species 100
    python data/download_data.py --all_india --max_per_species 50
"""

import os
import time
import json
import argparse
import urllib.request
import urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional

import pandas as pd
from tqdm import tqdm

XENO_CANTO_API = "https://xeno-canto.org/api/2/recordings"
DATA_DIR = Path("data/audio")
META_CSV = Path("data/metadata.csv")


# ──────────────────────────────────────────────────────────────────────────────
# xeno-canto API Client
# ──────────────────────────────────────────────────────────────────────────────

def search_xeno_canto(
    species: str,
    quality: str = "A:B",       # A=best, B=good, C=medium
    country: Optional[str] = None,
    call_type: Optional[str] = None,
    page: int = 1,
) -> Dict:
    """Query xeno-canto API for bird recordings."""
    query_parts = [f'"{species}"']
    if quality:
        query_parts.append(f"q:{quality}")
    if country:
        query_parts.append(f"cnt:{country}")
    if call_type:
        query_parts.append(f"type:{call_type}")

    query = " ".join(query_parts)
    params = urllib.parse.urlencode({"query": query, "page": page})
    url = f"{XENO_CANTO_API}?{params}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "BirdLanguageML/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"  API error for '{species}': {e}")
        return {"recordings": [], "numRecordings": 0}


def infer_call_type(xc_type: str) -> str:
    """Map xeno-canto type tags to our call type labels."""
    xc_type = (xc_type or "").lower()
    mapping = {
        "alarm": "alarm_call",
        "flight": "flight_call",
        "contact": "contact_call",
        "call": "contact_call",
        "song": "mating_song",
        "dawn": "dawn_chorus",
        "begging": "begging_call",
        "distress": "distress_call",
        "territorial": "territorial_call",
        "aggressive": "territorial_call",
        "foraging": "foraging_sound",
        "feeding": "foraging_sound",
    }
    for key, val in mapping.items():
        if key in xc_type:
            return val
    return "contact_call"  # default


def download_file(url: str, dest: Path) -> bool:
    """Download a single audio file."""
    if dest.exists():
        return True
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"  Download failed {url}: {e}")
        return False


def collect_metadata(recordings: List[Dict], species: str) -> List[Dict]:
    """Extract relevant fields from xeno-canto recording objects."""
    rows = []
    for rec in recordings:
        file_url = rec.get("file", "")
        if not file_url or not file_url.startswith("http"):
            continue
        rows.append({
            "xc_id": rec.get("id", ""),
            "species": species,
            "call_type": infer_call_type(rec.get("type", "")),
            "country": rec.get("cnt", ""),
            "quality": rec.get("q", ""),
            "duration": rec.get("length", ""),
            "date": rec.get("date", ""),
            "latitude": rec.get("lat", ""),
            "longitude": rec.get("lng", ""),
            "file_url": file_url,
            "filename": f"{species.replace(' ', '_')}_{rec.get('id', '')}.mp3",
        })
    return rows


def download_species(
    species: str,
    max_recordings: int = 100,
    audio_dir: Path = DATA_DIR,
    workers: int = 4,
) -> List[Dict]:
    """Download all available high-quality recordings for one species."""
    audio_dir.mkdir(parents=True, exist_ok=True)
    all_meta = []
    page = 1

    print(f"\n Fetching: {species}")
    while len(all_meta) < max_recordings:
        data = search_xeno_canto(species, page=page)
        if not data.get("recordings"):
            break
        meta = collect_metadata(data["recordings"], species)
        all_meta.extend(meta)
        total_pages = int(data.get("numPages", 1))
        if page >= total_pages:
            break
        page += 1
        time.sleep(0.5)  # Be respectful to the API

    all_meta = all_meta[:max_recordings]
    print(f"  Found {len(all_meta)} recordings → downloading...")

    # Assign local filepath
    for row in all_meta:
        row["filepath"] = str(audio_dir / row["filename"])

    # Parallel download
    def _download(row):
        return download_file(row["file_url"], Path(row["filepath"]))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_download, r): r for r in all_meta}
        ok = 0
        for fut in tqdm(as_completed(futures), total=len(futures), desc=species[:20]):
            if fut.result():
                ok += 1

    # Keep only successfully downloaded files
    all_meta = [r for r in all_meta if Path(r["filepath"]).exists()]
    print(f"  Downloaded {len(all_meta)}/{max_recordings} recordings")
    return all_meta


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

INDIA_SPECIES = [
    "House Sparrow", "Common Myna", "Rock Pigeon", "Rose-ringed Parakeet",
    "Red-vented Bulbul", "Asian Koel", "Jungle Crow", "Common Kingfisher",
    "Purple Sunbird", "Oriental Magpie-Robin", "Common Tailorbird",
    "Red-wattled Lapwing", "Spotted Dove", "Indian Robin", "Greater Coucal",
    "Black Kite", "Common Hoopoe", "Green Bee-eater", "Indian Roller",
    "Coppersmith Barbet",
]


def main():
    parser = argparse.ArgumentParser(description="Download bird audio from xeno-canto")
    parser.add_argument("--species", nargs="+", help="Species to download")
    parser.add_argument("--all_india", action="store_true", help="Download all India focus species")
    parser.add_argument("--max_per_species", type=int, default=150)
    parser.add_argument("--audio_dir", default="data/audio")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    species_list = []
    if args.all_india:
        species_list = INDIA_SPECIES
    if args.species:
        species_list.extend(args.species)
    if not species_list:
        species_list = INDIA_SPECIES[:5]  # default: top 5 India species

    audio_dir = Path(args.audio_dir)
    all_metadata = []

    for sp in species_list:
        meta = download_species(sp, args.max_per_species, audio_dir, args.workers)
        all_metadata.extend(meta)

    # Save metadata CSV
    META_CSV.parent.mkdir(exist_ok=True)
    df = pd.DataFrame(all_metadata)
    df.to_csv(META_CSV, index=False)

    print(f"\n{'='*50}")
    print(f"Total recordings: {len(df)}")
    print(f"Species: {df['species'].nunique()}")
    print(f"Call types: {df['call_type'].value_counts().to_dict()}")
    print(f"Metadata saved: {META_CSV}")
    print(f"Audio dir: {audio_dir}")
    print("\nNext step:")
    print("  python -c \"from src.train import train; train()\"")


if __name__ == "__main__":
    main()
