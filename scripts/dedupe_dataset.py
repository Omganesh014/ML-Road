"""
TruthBridge Dataset Deduplication Script
Processes all datasets in data/ folder, finds duplicates using perceptual hashing,
and outputs a hash database for client-side duplicate detection.
"""

import imagehash
import os
import json
from PIL import Image
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
HASH_SIZE = 16
HAMMING_THRESHOLD = 5


def compute_hash(img_path):
    try:
        return imagehash.phash(Image.open(img_path), hash_size=HASH_SIZE)
    except Exception as e:
        print(f"Error hashing {img_path}: {e}", file=sys.stderr)
        return None


def hamming_distance(hash1, hash2):
    return hash1 - hash2


def process_image(img_path):
    h = compute_hash(img_path)
    if h is None:
        return None
    file_stat = os.stat(img_path)
    return {
        "hash": str(h),
        "path": img_path,
        "size": file_stat.st_size,
        "dataset": Path(img_path).parts[1] if len(Path(img_path).parts) > 1 else "unknown"
    }


def find_duplicate_groups(hashes, threshold=HAMMING_THRESHOLD):
    groups = []
    processed = set()

    for i, entry in enumerate(hashes):
        if i in processed:
            continue

        group = [entry]
        entry_hash = entry["hash"]

        for j, other in enumerate(hashes[i + 1:], start=i + 1):
            if j in processed:
                continue

            dist = hamming_distance(entry_hash, other["hash"])
            if dist <= threshold:
                group.append(other)
                processed.add(j)

        if len(group) > 1:
            groups.append(group)
            for idx in [i] + [hashes.index(g) for g in group[1:]]:
                processed.add(idx)

    return groups


def main():
    data_dir = Path(__file__).parent.parent / "data"
    output_path = Path(__file__).parent.parent / "public" / "models"
    output_path.mkdir(parents=True, exist_ok=True)

    all_hashes = []
    total_files = 0

    print("Scanning datasets...")
    for dataset_folder in data_dir.iterdir():
        if not dataset_folder.is_dir():
            continue

        print(f"  Processing: {dataset_folder.name}")
        for img_path in dataset_folder.rglob("*"):
            if img_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                total_files += 1

    print(f"Found {total_files} images. Computing hashes...")

    image_paths = []
    for dataset_folder in data_dir.iterdir():
        if not dataset_folder.is_dir():
            continue
        for img_path in dataset_folder.rglob("*"):
            if img_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                image_paths.append(str(img_path))

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_image, p): p for p in image_paths}
        for idx, future in enumerate(as_completed(futures)):
            result = future.result()
            if result:
                all_hashes.append(result)
            if (idx + 1) % 500 == 0:
                print(f"  Hashed {idx + 1}/{len(image_paths)} images...")

    print(f"Computed {len(all_hashes)} valid hashes")

    print("Finding duplicate groups (hamming distance <= 5)...")
    duplicate_groups = find_duplicate_groups(all_hashes, threshold=HAMMING_THRESHOLD)

    total_duplicates = sum(len(g) - 1 for g in duplicate_groups)
    print(f"Found {len(duplicate_groups)} duplicate groups ({total_duplicates} duplicate images)")

    hash_db = {
        "version": "1.0",
        "hash_size": HASH_SIZE,
        "hamming_threshold": HAMMING_THRESHOLD,
        "total_images": len(all_hashes),
        "duplicate_groups_count": len(duplicate_groups),
        "duplicates_count": total_duplicates,
        "hashes": all_hashes
    }

    output_file = output_path / "image_hashes.json"
    with open(output_file, 'w') as f:
        json.dump(hash_db, f)

    print(f"\nHash database saved to: {output_file}")
    print(f"Total unique images: {len(all_hashes) - total_duplicates}")

    dup_report_path = output_path / "duplicates_report.json"
    dup_report = {
        "total_duplicates": total_duplicates,
        "groups": [{"hash": g[0]["hash"], "count": len(g), "paths": [x["path"] for x in g]} for g in duplicate_groups]
    }
    with open(dup_report_path, 'w') as f:
        json.dump(dup_report, f, indent=2)

    print(f"Duplicates report saved to: {dup_report_path}")


if __name__ == "__main__":
    main()