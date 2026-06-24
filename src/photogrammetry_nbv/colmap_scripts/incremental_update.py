#!/usr/bin/env python3
"""Add new images to an existing COLMAP reconstruction and update the sparse model.

When incremental mode is enabled (default), the pipeline uses:
  1. image_registrator  — register new images into the existing model (PnP)
  2. point_triangulator — triangulate all untriangulated 2D correspondences
  3. bundle_adjuster    — refine all camera poses and 3D points

This is more reliable than mapper --input_path, which was observed to silently
skip unregistered images.  If the incremental pipeline fails, the script falls
back to a full mapper rebuild.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import time
from pathlib import Path

from common import (
    collect_sparse_summary,
    find_best_sparse_model,
    load_cfg,
    load_request_json,
    save_summary_json,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)


def main() -> None:
    request = load_request_json()
    workspace = Path(request['workspace']).expanduser()
    new_image_paths = [Path(p).expanduser() for p in request['new_image_paths']]
    output_json = Path(request['output_json']).expanduser()
    colmap_bin = request.get('colmap_bin', 'colmap')
    cfg = load_cfg(request.get('config', {}))

    incremental_cfg = cfg.get('incremental', {})
    use_incremental = incremental_cfg.get('enabled', True)

    images_dir = workspace / 'images'
    database = workspace / 'database.db'
    sparse_dir = workspace / 'sparse'
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy new images into the workspace
    for src in new_image_paths:
        dst = images_dir / src.name
        if not dst.exists():
            shutil.copy2(str(src), str(dst))

    # 2. Feature extraction (COLMAP skips already-extracted images in the DB)
    feat_cfg = cfg.get('feature_extraction', {})
    camera_model = feat_cfg.get('camera_model', 'PINHOLE')
    max_num_features = feat_cfg.get('max_num_features', 8192)

    _run([
        colmap_bin, 'feature_extractor',
        '--database_path', str(database),
        '--image_path', str(images_dir),
        '--ImageReader.camera_model', camera_model,
        '--SiftExtraction.max_num_features', str(max_num_features),
    ])

    # 3. Feature matching (DB caches existing verified matches)
    _run([
        colmap_bin, 'exhaustive_matcher',
        '--database_path', str(database),
    ])

    # 4. Reconstruction — incremental registration or full rebuild
    mapper_cfg = cfg.get('mapper', {})
    best_model = find_best_sparse_model(sparse_dir)
    has_existing_model = (best_model / 'cameras.bin').exists()

    if use_incremental and has_existing_model:
        try:
            _run_incremental_registration(
                colmap_bin, database, images_dir, best_model, incremental_cfg,
            )
        except RuntimeError as exc:
            log.warning('Incremental registration failed, falling back to full rebuild: %s', exc)
            _run_full_mapper(colmap_bin, database, images_dir, sparse_dir, mapper_cfg)
    else:
        if use_incremental and not has_existing_model:
            log.info('No existing model found — running full mapper')
        _run_full_mapper(colmap_bin, database, images_dir, sparse_dir, mapper_cfg)

    # 5. Export metrics
    save_summary_json(output_json, collect_sparse_summary(workspace, cfg))


# ---------------------------------------------------------------------------
# Incremental registration pipeline
# ---------------------------------------------------------------------------

def _run_incremental_registration(
    colmap_bin: str,
    database: Path,
    images_dir: Path,
    model_path: Path,
    incremental_cfg: dict,
) -> None:
    """Register new images, triangulate points, and run bundle adjustment.

    Unlike mapper --input_path (which silently skips images), this pipeline
    uses dedicated COLMAP commands that reliably register new images into an
    existing reconstruction.
    """
    ba_iterations = int(incremental_cfg.get('ba_max_num_iterations', 100))

    # Step 1: Register unregistered images into the existing model
    log.info('Running image_registrator on %s', model_path)
    t0 = time.monotonic()
    _run([
        colmap_bin, 'image_registrator',
        '--database_path', str(database),
        '--input_path', str(model_path),
        '--output_path', str(model_path),
    ])
    log.info('image_registrator completed in %.1f s', time.monotonic() - t0)

    # Step 2: Triangulate all untriangulated 2D correspondences
    log.info('Running point_triangulator')
    t1 = time.monotonic()
    _run([
        colmap_bin, 'point_triangulator',
        '--database_path', str(database),
        '--image_path', str(images_dir),
        '--input_path', str(model_path),
        '--output_path', str(model_path),
    ])
    log.info('point_triangulator completed in %.1f s', time.monotonic() - t1)

    # Step 3: Global bundle adjustment to refine all poses and points
    log.info('Running bundle_adjuster (%d iterations)', ba_iterations)
    ba_output = model_path.parent / '_ba_temp'
    ba_output.mkdir(parents=True, exist_ok=True)
    try:
        t2 = time.monotonic()
        _run([
            colmap_bin, 'bundle_adjuster',
            '--input_path', str(model_path),
            '--output_path', str(ba_output),
            '--BundleAdjustment.max_num_iterations', str(ba_iterations),
        ])
        log.info('bundle_adjuster completed in %.1f s', time.monotonic() - t2)
        # Replace model with refined version
        for fname in ('cameras.bin', 'images.bin', 'points3D.bin'):
            src = ba_output / fname
            if src.exists():
                shutil.copy2(str(src), str(model_path / fname))
    finally:
        if ba_output.exists():
            shutil.rmtree(str(ba_output), ignore_errors=True)


# ---------------------------------------------------------------------------
# Full mapper fallback
# ---------------------------------------------------------------------------

def _run_full_mapper(
    colmap_bin: str,
    database: Path,
    images_dir: Path,
    sparse_dir: Path,
    mapper_cfg: dict,
) -> None:
    log.info('Running full mapper (no --input_path)')
    t0 = time.monotonic()
    cmd = [
        colmap_bin, 'mapper',
        '--database_path', str(database),
        '--image_path', str(images_dir),
        '--output_path', str(sparse_dir),
    ]
    if 'ba_global_max_num_iterations' in mapper_cfg:
        cmd += ['--Mapper.ba_global_max_num_iterations', str(mapper_cfg['ba_global_max_num_iterations'])]
    if 'filter_max_reproj_error' in mapper_cfg:
        cmd += ['--Mapper.filter_max_reproj_error', str(mapper_cfg['filter_max_reproj_error'])]
    _run(cmd)
    log.info('Full mapper completed in %.1f s', time.monotonic() - t0)


def _run(cmd: list) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f'COLMAP command failed: {" ".join(cmd)}\n'
            f'STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}'
        )


if __name__ == '__main__':
    main()
