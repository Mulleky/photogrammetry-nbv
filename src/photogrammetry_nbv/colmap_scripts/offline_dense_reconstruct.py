#!/usr/bin/env python3
"""Run COLMAP dense reconstruction (undistortion + patch_match_stereo + fusion)."""
from __future__ import annotations

import subprocess
from pathlib import Path

from common import load_cfg, load_request_json


def main() -> None:
    request = load_request_json()
    workspace = Path(request['workspace']).expanduser()
    output_dir = Path(request['output_dir']).expanduser()
    colmap_bin = request.get('colmap_bin', 'colmap')
    cfg = load_cfg(request.get('config', {}))
    dense_cfg = cfg.get('dense', {})
    output_dir.mkdir(parents=True, exist_ok=True)

    sparse_dir = workspace / 'sparse' / '0'
    dense_dir = workspace / 'dense'
    dense_dir.mkdir(parents=True, exist_ok=True)

    _run([
        colmap_bin, 'image_undistorter',
        '--image_path', str(workspace / 'images'),
        '--input_path', str(sparse_dir),
        '--output_path', str(dense_dir),
        '--output_type', 'COLMAP',
    ])

    pms_cmd = [
        colmap_bin, 'patch_match_stereo',
        '--workspace_path', str(dense_dir),
        '--workspace_format', 'COLMAP',
    ]
    if 'max_image_size' in dense_cfg:
        pms_cmd += ['--PatchMatchStereo.max_image_size', str(dense_cfg['max_image_size'])]
    if dense_cfg.get('geom_consistency', False):
        pms_cmd += ['--PatchMatchStereo.geom_consistency', 'true']
    _run(pms_cmd)

    fused_path = output_dir / 'dense_cloud.ply'
    _run([
        colmap_bin, 'stereo_fusion',
        '--workspace_path', str(dense_dir),
        '--workspace_format', 'COLMAP',
        '--output_path', str(fused_path),
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
