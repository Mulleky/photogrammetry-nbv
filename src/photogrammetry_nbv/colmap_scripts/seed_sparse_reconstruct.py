#!/usr/bin/env python3
"""Run a standalone sparse reconstruction on just the seed images and export a PLY."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from common import find_best_sparse_model, load_cfg, load_request_json


def main() -> None:
    request = load_request_json()
    seed_workspace = Path(request['seed_workspace']).expanduser()
    seed_images = [Path(p).expanduser() for p in request['seed_images']]
    output_dir = Path(request['output_dir']).expanduser()
    colmap_bin = request.get('colmap_bin', 'colmap')
    cfg = load_cfg(request.get('config', {}))

    images_dir = seed_workspace / 'images'
    database = seed_workspace / 'database.db'
    sparse_dir = seed_workspace / 'sparse'
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    for src in seed_images:
        dst = images_dir / src.name
        if not dst.exists():
            shutil.copy2(str(src), str(dst))

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

    _run([
        colmap_bin, 'exhaustive_matcher',
        '--database_path', str(database),
    ])

    mapper_cfg = cfg.get('mapper', {})
    mapper_cmd = [
        colmap_bin, 'mapper',
        '--database_path', str(database),
        '--image_path', str(images_dir),
        '--output_path', str(sparse_dir),
    ]
    if 'ba_global_max_num_iterations' in mapper_cfg:
        mapper_cmd += ['--Mapper.ba_global_max_num_iterations', str(mapper_cfg['ba_global_max_num_iterations'])]
    if 'filter_max_reproj_error' in mapper_cfg:
        mapper_cmd += ['--Mapper.filter_max_reproj_error', str(mapper_cfg['filter_max_reproj_error'])]
    mapper_cmd += ['--Mapper.ba_use_gpu', '1']
    _run(mapper_cmd)

    # Export to PLY
    sparse_model = find_best_sparse_model(sparse_dir)
    ply_path = output_dir / 'seed_sparse_cloud.ply'
    _run([
        colmap_bin, 'model_converter',
        '--input_path', str(sparse_model),
        '--output_path', str(ply_path),
        '--output_type', 'PLY',
    ])


def _run(cmd: list) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f'COLMAP command failed: {" ".join(cmd)}\n'
            f'STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}'
        )


if __name__ == '__main__':
    main()
