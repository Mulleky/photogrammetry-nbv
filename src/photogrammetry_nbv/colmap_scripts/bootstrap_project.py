#!/usr/bin/env python3
"""Bootstrap a COLMAP sparse reconstruction from seed images."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from common import collect_sparse_summary, load_cfg, load_request_json, save_summary_json


def main() -> None:
    request = load_request_json()
    workspace = Path(request['workspace']).expanduser()
    image_paths = [Path(p).expanduser() for p in request['image_paths']]
    output_json = Path(request['output_json']).expanduser()
    colmap_bin = request.get('colmap_bin', 'colmap')
    cfg = load_cfg(request.get('config', {}))

    images_dir = workspace / 'images'
    database = workspace / 'database.db'
    sparse_dir = workspace / 'sparse'
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    for src in image_paths:
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
    _run(mapper_cmd)

    save_summary_json(output_json, collect_sparse_summary(workspace, cfg))


def _run(cmd: list) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f'COLMAP command failed: {" ".join(cmd)}\n'
            f'STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}'
        )


if __name__ == '__main__':
    main()
