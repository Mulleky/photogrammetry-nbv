from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List


class MetashapeWorkerClient:
    """Thin subprocess wrapper around headless Metashape scripts."""

    def __init__(self, metashape_cmd: str, script_dir: Path) -> None:
        self.metashape_cmd = metashape_cmd
        self.script_dir = script_dir

    def bootstrap_project(self, project_path: Path, image_paths: List[Path], output_json: Path, config: Dict) -> None:
        self._run_script('bootstrap_project.py', {
            'project_path': str(project_path),
            'image_paths': [str(p) for p in image_paths],
            'output_json': str(output_json),
            'config': config,
        })

    def incremental_update(self, project_path: Path, new_image_paths: List[Path], output_json: Path, config: Dict) -> None:
        self._run_script('incremental_update.py', {
            'project_path': str(project_path),
            'new_image_paths': [str(p) for p in new_image_paths],
            'output_json': str(output_json),
            'config': config,
        })

    def export_sparse_metrics(self, project_path: Path, output_json: Path, config: Dict) -> None:
        self._run_script('export_sparse_metrics.py', {
            'project_path': str(project_path),
            'output_json': str(output_json),
            'config': config,
        })

    def offline_dense_reconstruct(self, project_path: Path, output_dir: Path, config: Dict) -> None:
        self._run_script('offline_dense_reconstruct.py', {
            'project_path': str(project_path),
            'output_dir': str(output_dir),
            'config': config,
        })

    def _run_script(self, script_name: str, args: Dict) -> None:
        wrapper_json = self.script_dir / f'.{script_name}.request.json'
        with open(wrapper_json, 'w', encoding='utf-8') as f:
            json.dump(args, f, indent=2)
        script_path = self.script_dir / script_name
        cmd = [self.metashape_cmd, '-r', str(script_path), '--args', str(wrapper_json)]
        completed = subprocess.run(cmd, capture_output=True, text=True)
        if completed.returncode != 0:
            msg = (
                f"Metashape script failed: {script_name}\n"
                f"STDOUT:\n{completed.stdout}\n"
                f"STDERR:\n{completed.stderr}"
            )
            raise RuntimeError(msg)
