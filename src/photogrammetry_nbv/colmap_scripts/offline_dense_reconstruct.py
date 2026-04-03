#!/usr/bin/env python3
"""Run COLMAP dense reconstruction (undistortion + patch_match_stereo + fusion)."""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from common import find_best_sparse_model, load_cfg, load_request_json

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)


def main() -> None:
    request = load_request_json()
    workspace = Path(request['workspace']).expanduser()
    output_dir = Path(request['output_dir']).expanduser()
    colmap_bin = request.get('colmap_bin', 'colmap')
    cfg = load_cfg(request.get('config', {}))
    dense_cfg = cfg.get('dense', {})
    output_dir.mkdir(parents=True, exist_ok=True)

    sparse_dir = find_best_sparse_model(workspace / 'sparse')
    dense_dir = workspace / 'dense'
    dense_dir.mkdir(parents=True, exist_ok=True)

    # --- Final global BA to polish all poses before dense reconstruction ---
    incremental_cfg = cfg.get('incremental', {})
    if incremental_cfg.get('final_global_ba', True) and (sparse_dir / 'cameras.bin').exists():
        ba_iterations = int(incremental_cfg.get('ba_max_num_iterations', 100)) * 2  # 200 default
        ba_output = workspace / 'sparse' / '_final_ba'
        ba_output.mkdir(parents=True, exist_ok=True)
        log.info('Running final global BA (%d iterations) before dense reconstruction', ba_iterations)
        try:
            _run([
                colmap_bin, 'bundle_adjuster',
                '--input_path', str(sparse_dir),
                '--output_path', str(ba_output),
                '--BundleAdjustment.max_num_iterations', str(ba_iterations),
                '--BundleAdjustmentCeres.use_gpu', '1',
            ])
            # Use BA-refined model for dense reconstruction
            for fname in ('cameras.bin', 'images.bin', 'points3D.bin'):
                src = ba_output / fname
                if src.exists():
                    shutil.copy2(str(src), str(sparse_dir / fname))
            log.info('Final global BA completed successfully')
        except RuntimeError as exc:
            log.warning('Final global BA failed (proceeding with existing model): %s', exc)
        finally:
            if ba_output.exists():
                shutil.rmtree(str(ba_output), ignore_errors=True)

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
