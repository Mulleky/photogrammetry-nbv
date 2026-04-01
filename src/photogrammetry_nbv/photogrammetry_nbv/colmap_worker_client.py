from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


class ColmapWorkerClient:
    """Thin subprocess wrapper around headless COLMAP Python scripts."""

    def __init__(self, colmap_bin: str, script_dir: Path) -> None:
        self.colmap_bin = colmap_bin
        self.script_dir = script_dir

    def bootstrap_project(self, workspace: Path, image_paths: List[Path], output_json: Path, config: Dict) -> None:
        self._run_script('bootstrap_project.py', {
            'workspace': str(workspace),
            'image_paths': [str(p) for p in image_paths],
            'output_json': str(output_json),
            'colmap_bin': self.colmap_bin,
            'config': config,
        })

    def incremental_update(self, workspace: Path, new_image_paths: List[Path], output_json: Path, config: Dict) -> None:
        self._run_script('incremental_update.py', {
            'workspace': str(workspace),
            'new_image_paths': [str(p) for p in new_image_paths],
            'output_json': str(output_json),
            'colmap_bin': self.colmap_bin,
            'config': config,
        })

    def export_sparse_metrics(self, workspace: Path, output_json: Path, config: Dict) -> None:
        self._run_script('export_sparse_metrics.py', {
            'workspace': str(workspace),
            'output_json': str(output_json),
            'colmap_bin': self.colmap_bin,
            'config': config,
        })

    def offline_dense_reconstruct(self, workspace: Path, output_dir: Path, config: Dict) -> None:
        self._run_script('offline_dense_reconstruct.py', {
            'workspace': str(workspace),
            'output_dir': str(output_dir),
            'colmap_bin': self.colmap_bin,
            'config': config,
        })

    def seed_sparse_reconstruct(self, seed_workspace: Path, seed_images: List[Path], output_dir: Path, config: Dict) -> None:
        self._run_script('seed_sparse_reconstruct.py', {
            'seed_workspace': str(seed_workspace),
            'seed_images': [str(p) for p in seed_images],
            'output_dir': str(output_dir),
            'colmap_bin': self.colmap_bin,
            'config': config,
        })

    def export_model_ply(self, model_dir: Path, ply_path: Path) -> None:
        """Export a COLMAP sparse model directory to a PLY point cloud."""
        ply_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.colmap_bin, 'model_converter',
            '--input_path', str(model_dir),
            '--output_path', str(ply_path),
            '--output_type', 'PLY',
        ]
        completed = subprocess.run(cmd, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(
                f'COLMAP model_converter failed\n'
                f'STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}')

    def _run_script(self, script_name: str, args: Dict) -> None:
        request_json = self.script_dir / f'.{script_name}.request.json'
        with open(request_json, 'w', encoding='utf-8') as f:
            json.dump(args, f, indent=2)
        script_path = self.script_dir / script_name
        cmd = [sys.executable, str(script_path), '--args', str(request_json)]
        completed = subprocess.run(cmd, capture_output=True, text=True)
        if completed.returncode != 0:
            msg = (
                f"COLMAP script failed: {script_name}\n"
                f"STDOUT:\n{completed.stdout}\n"
                f"STDERR:\n{completed.stderr}"
            )
            raise RuntimeError(msg)
