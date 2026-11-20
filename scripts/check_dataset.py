"""Read-only counts and byte-identical overlap check; no image contents are exported."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def audit(root):
    counts = Counter()
    hashes = defaultdict(set)
    for phase in ("train", "val"):
        for label in ("ants", "bees"):
            directory = Path(root) / phase / label
            if not directory.is_dir():
                raise ValueError(f"Missing partition/class directory: {phase}/{label}")
            for path in directory.iterdir():
                if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                counts[f"{phase}/{label}"] += 1
                hashes[hashlib.sha256(path.read_bytes()).hexdigest()].add((phase, label))
    overlap = sum(any(p == "train" for p, _ in parts) and any(p == "val" for p, _ in parts) for parts in hashes.values())
    return {"counts": dict(counts), "byte_identical_cross_partition_hashes": overlap,
            "limitation": "Does not detect visually similar images, shared subjects, source overlap or all metadata risks."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    print(json.dumps(audit(parser.parse_args().data_dir), indent=2))
