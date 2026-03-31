from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def discover_seed_images(seed_run_dir: Path) -> List[Path]:
    images_dir = seed_run_dir / 'images'
    if not images_dir.exists():
        raise FileNotFoundError(f'Seed images directory missing: {images_dir}')
    return sorted([p for p in images_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])


def discover_seed_metadata(seed_run_dir: Path) -> List[Path]:
    metadata_dir = seed_run_dir / 'metadata'
    if not metadata_dir.exists():
        raise FileNotFoundError(f'Seed metadata directory missing: {metadata_dir}')
    return sorted([p for p in metadata_dir.iterdir() if p.suffix.lower() == '.json'])


def load_seed_bundle(seed_run_dir: Path) -> Tuple[List[Path], List[Dict[str, Any]], Dict[str, Any]]:
    images = discover_seed_images(seed_run_dir)
    metadata_files = discover_seed_metadata(seed_run_dir)
    metadata: List[Dict[str, Any]] = []
    for path in metadata_files:
        with open(path, 'r', encoding='utf-8') as f:
            metadata.append(json.load(f))

    manifest_path = seed_run_dir / 'manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(f'Seed manifest missing: {manifest_path}')
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    if len(images) < 4:
        raise ValueError(f'Expected at least 4 seed images, found {len(images)}')

    return images, metadata, manifest


def infer_home_pose(seed_metadata: List[Dict[str, Any]]) -> Dict[str, float]:
    if not seed_metadata:
        return {'x': 0.0, 'y': 0.0, 'z': 0.0}
    first = seed_metadata[0]
    return first.get('vehicle_position_ned_m') or {'x': 0.0, 'y': 0.0, 'z': 0.0}
