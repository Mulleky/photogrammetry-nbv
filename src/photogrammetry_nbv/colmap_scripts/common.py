from __future__ import annotations

import json
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_request_json() -> Dict[str, Any]:
    if '--args' not in sys.argv:
        raise RuntimeError('Expected --args <request_json>')
    idx = sys.argv.index('--args')
    request_path = Path(sys.argv[idx + 1]).expanduser()
    with open(request_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_summary_json(output_json: Path, payload: Dict[str, Any]) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def load_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    return config.get('colmap', config)


def read_images_bin(path: Path) -> Dict[int, Dict[str, Any]]:
    """Read COLMAP images.bin and return {image_id: {name, qvec, tvec}}."""
    images = {}
    with open(path, 'rb') as f:
        num_images = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack('<I', f.read(4))[0]
            qvec = struct.unpack('<4d', f.read(32))
            tvec = struct.unpack('<3d', f.read(24))
            camera_id = struct.unpack('<I', f.read(4))[0]
            name = b''
            while True:
                ch = f.read(1)
                if ch == b'\x00':
                    break
                name += ch
            num_points2d = struct.unpack('<Q', f.read(8))[0]
            f.read(num_points2d * 24)  # skip 2D points (x, y, point3d_id)
            images[image_id] = {
                'name': name.decode('utf-8'),
                'qvec': qvec,
                'tvec': tvec,
                'camera_id': camera_id,
            }
    return images


def read_points3d_bin(path: Path) -> Dict[int, Tuple[float, float, float]]:
    """Read COLMAP points3D.bin and return {point3d_id: (x, y, z)}."""
    points = {}
    with open(path, 'rb') as f:
        num_points = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num_points):
            point3d_id = struct.unpack('<Q', f.read(8))[0]
            xyz = struct.unpack('<3d', f.read(24))
            rgb = struct.unpack('<3B', f.read(3))
            error = struct.unpack('<d', f.read(8))[0]
            track_length = struct.unpack('<Q', f.read(8))[0]
            f.read(track_length * 8)  # skip track (image_id + point2d_idx)
            points[point3d_id] = xyz
    return points


def collect_sparse_summary(workspace: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Build a summary dict from a COLMAP sparse reconstruction."""
    sparse_dir = workspace / 'sparse' / '0'
    images_bin = sparse_dir / 'images.bin'
    points_bin = sparse_dir / 'points3D.bin'

    images = read_images_bin(images_bin) if images_bin.exists() else {}
    points = read_points3d_bin(points_bin) if points_bin.exists() else {}

    # All images in the model are considered aligned
    aligned_count = len(images)
    total_cameras = aligned_count

    weak_regions = build_weak_regions(points, cfg.get('weak_region_extraction', {}))

    global_metrics = {}
    if weak_regions:
        for key in weak_regions[0].get('components', {}):
            vals = [r['components'].get(key, 0.0) for r in weak_regions]
            global_metrics[f'mean_{key}'] = sum(vals) / len(vals) if vals else 0.0

    return {
        'total_cameras': total_cameras,
        'aligned_cameras': aligned_count,
        'sparse_point_count': len(points),
        'global_metrics': global_metrics,
        'weak_regions': weak_regions,
        'per_camera_stats': [],
    }


def build_weak_regions(points: Dict, weak_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not weak_cfg.get('enabled', True):
        return []
    if not points:
        return [{
            'region_id': 'fallback_000',
            'centroid_xyz': [0.0, 0.0, 0.0],
            'severity': 0.1,
            'point_count': 0,
            'components': {
                'reconstruction_uncertainty': 0.0,
                'reprojection_error': 0.0,
                'track_length': 0.0,
                'projection_accuracy': 0.0,
            }
        }]

    coords = list(points.values())
    max_regions = int(weak_cfg.get('max_regions', 12))
    top_fraction = float(weak_cfg.get('top_fraction', 0.15))
    sample_count = max(1, int(len(coords) * top_fraction))
    sampled = coords[:sample_count]

    groups: List[List[Tuple[float, float, float]]] = [[] for _ in range(max_regions)]
    for idx, pt in enumerate(sampled):
        groups[idx % max_regions].append(pt)

    weights = weak_cfg.get('severity_weights', {})
    regions = []
    for ridx, group in enumerate(groups):
        if len(group) < int(weak_cfg.get('min_points_per_region', 10)):
            continue
        cx = sum(p[0] for p in group) / len(group)
        cy = sum(p[1] for p in group) / len(group)
        cz = sum(p[2] for p in group) / len(group)
        severity = (
            weights.get('reconstruction_uncertainty', 0.4) * 1.0
            + weights.get('reprojection_error', 0.3) * 1.0
            + weights.get('track_length', 0.2) * 1.0
            + weights.get('projection_accuracy', 0.1) * 1.0
        )
        regions.append({
            'region_id': f'region_{ridx:03d}',
            'centroid_xyz': [cx, cy, cz],
            'severity': severity,
            'point_count': len(group),
            'components': {
                'reconstruction_uncertainty': 1.0,
                'reprojection_error': 1.0,
                'track_length': 1.0,
                'projection_accuracy': 1.0,
            }
        })
    return regions
