from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    import Metashape  # type: ignore
except Exception:
    Metashape = None


def load_request_json() -> Dict[str, Any]:
    if '--args' not in sys.argv:
        raise RuntimeError('Expected --args <request_json>')
    idx = sys.argv.index('--args')
    request_path = Path(sys.argv[idx + 1]).expanduser()
    with open(request_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def ensure_metashape():
    if Metashape is None:
        raise RuntimeError('Metashape Python API is not available in this runtime.')


def save_summary_json(output_json: Path, payload: Dict[str, Any]) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def get_or_create_chunk(doc, label: str):
    for chunk in doc.chunks:
        if chunk.label == label:
            return chunk
    chunk = doc.addChunk()
    chunk.label = label
    return chunk


def load_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    return config['metashape'] if 'metashape' in config else config


def collect_sparse_summary(chunk, cfg: Dict[str, Any]) -> Dict[str, Any]:
    cameras = list(chunk.cameras) if hasattr(chunk, 'cameras') else []
    aligned = [c for c in cameras if getattr(c, 'transform', None) is not None]
    sparse_points = []
    if getattr(chunk, 'tie_points', None) and getattr(chunk.tie_points, 'points', None):
        sparse_points = [p for p in chunk.tie_points.points if p.valid]
    weak_regions = build_weak_regions(chunk, cfg.get('weak_region_extraction', {}))
    global_metrics = {
        'mean_reconstruction_uncertainty': float(sum(r['components'].get('mean_reconstruction_uncertainty', 0.0) for r in weak_regions) / max(1, len(weak_regions))) if weak_regions else 0.0,
        'mean_projection_accuracy': float(sum(r['components'].get('mean_projection_accuracy', 0.0) for r in weak_regions) / max(1, len(weak_regions))) if weak_regions else 0.0,
        'mean_reprojection_error': float(sum(r['components'].get('mean_reprojection_error', 0.0) for r in weak_regions) / max(1, len(weak_regions))) if weak_regions else 0.0,
    }
    return {
        'total_cameras': len(cameras),
        'aligned_cameras': len(aligned),
        'sparse_point_count': len(sparse_points),
        'global_metrics': global_metrics,
        'weak_regions': weak_regions,
        'per_camera_stats': [],
    }


def build_weak_regions(chunk, weak_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    enabled = weak_cfg.get('enabled', True)
    if not enabled:
        return []
    if not getattr(chunk, 'tie_points', None) or not getattr(chunk.tie_points, 'points', None):
        return [{
            'region_id': 'fallback_000',
            'centroid_xyz': [0.0, 0.0, 0.0],
            'severity': 0.1,
            'point_count': 0,
            'components': {
                'mean_reconstruction_uncertainty': 0.0,
                'mean_projection_accuracy': 0.0,
                'mean_reprojection_error': 0.0,
                'mean_inverse_track_length': 0.0,
            }
        }]

    valid_points = [p for p in chunk.tie_points.points if p.valid]
    if not valid_points:
        return []

    max_regions = int(weak_cfg.get('max_regions', 12))
    top_fraction = float(weak_cfg.get('top_fraction', 0.15))
    sample_count = max(1, int(len(valid_points) * top_fraction))
    sampled = valid_points[:sample_count]
    groups: List[List[Any]] = [[] for _ in range(max_regions)]
    for idx, pt in enumerate(sampled):
        groups[idx % max_regions].append(pt)

    weights = weak_cfg.get('severity_weights', {})
    regions = []
    for ridx, group in enumerate(groups):
        if len(group) < int(weak_cfg.get('min_points_per_region', 10)):
            continue
        coords = [[float(p.coord.x), float(p.coord.y), float(p.coord.z)] for p in group]
        centroid = [sum(c[0] for c in coords) / len(coords), sum(c[1] for c in coords) / len(coords), sum(c[2] for c in coords) / len(coords)]
        mean_ru = 1.0; mean_pa = 1.0; mean_re = 1.0; mean_itl = 1.0
        severity = (
            weights.get('reconstruction_uncertainty', 0.4) * mean_ru +
            weights.get('projection_accuracy', 0.25) * mean_pa +
            weights.get('reprojection_error', 0.25) * mean_re +
            weights.get('inverse_track_length', 0.10) * mean_itl
        )
        regions.append({
            'region_id': f'region_{ridx:03d}',
            'centroid_xyz': centroid,
            'severity': severity,
            'point_count': len(group),
            'components': {
                'mean_reconstruction_uncertainty': mean_ru,
                'mean_projection_accuracy': mean_pa,
                'mean_reprojection_error': mean_re,
                'mean_inverse_track_length': mean_itl,
            }
        })
    return regions
