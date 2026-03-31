from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from .contracts import SparseMetricsSnapshot, WeakRegion


def load_sparse_metrics(metrics_json_path: Path, iteration: int) -> SparseMetricsSnapshot:
    if not metrics_json_path.exists():
        raise FileNotFoundError(f'Sparse metrics JSON missing: {metrics_json_path}')
    with open(metrics_json_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    weak_regions = [
        WeakRegion(
            region_id=str(r.get('region_id', f'region_{idx:03d}')),
            centroid_xyz=list(r.get('centroid_xyz', [0.0, 0.0, 0.0])),
            severity=float(r.get('severity', 0.0)),
            point_count=int(r.get('point_count', 0)),
            components=dict(r.get('components', {})),
        )
        for idx, r in enumerate(payload.get('weak_regions', []))
    ]

    return SparseMetricsSnapshot(
        iteration=iteration,
        total_cameras=int(payload.get('total_cameras', 0)),
        aligned_cameras=int(payload.get('aligned_cameras', 0)),
        sparse_point_count=int(payload.get('sparse_point_count', 0)),
        global_metrics=dict(payload.get('global_metrics', {})),
        weak_regions=weak_regions,
        per_camera_stats=list(payload.get('per_camera_stats', [])),
        raw_path=str(metrics_json_path),
        knn_distance_metrics=dict(payload.get('knn_distance_metrics', {})),
    )
