from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Dict, Tuple

from .contracts import RunPaths


def prepare_phase2_run(seed_run_dir: Path, output_root: Path) -> Tuple[RunPaths, Dict]:
    if not seed_run_dir.exists():
        raise FileNotFoundError(f'Seed run directory not found: {seed_run_dir}')

    manifest_path = seed_run_dir / 'manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(f'Missing seed manifest: {manifest_path}')

    with open(manifest_path, 'r', encoding='utf-8') as f:
        seed_manifest = json.load(f)

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_dir = output_root / f'phase2_run_{timestamp}'
    adaptive_dir = run_dir / 'adaptive'
    images_dir = adaptive_dir / 'images'
    metadata_dir = adaptive_dir / 'metadata'
    metashape_dir = run_dir / 'metashape'
    sparse_metrics_dir = metashape_dir / 'sparse_metrics'
    candidate_dir = metashape_dir / 'candidate_scores'
    final_dir = run_dir / 'final'

    for d in [run_dir, adaptive_dir, images_dir, metadata_dir, metashape_dir, sparse_metrics_dir, candidate_dir, final_dir]:
        d.mkdir(parents=True, exist_ok=True)

    seed_copy_dir = run_dir / 'seed'
    shutil.copytree(seed_run_dir, seed_copy_dir, dirs_exist_ok=True)

    paths = RunPaths(
        run_dir=run_dir,
        adaptive_dir=adaptive_dir,
        images_dir=images_dir,
        metadata_dir=metadata_dir,
        metashape_dir=metashape_dir,
        sparse_metrics_dir=sparse_metrics_dir,
        candidate_dir=candidate_dir,
        final_dir=final_dir,
    )

    return paths, seed_manifest
