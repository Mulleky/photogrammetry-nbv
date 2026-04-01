from __future__ import annotations

import json
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


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


def find_best_sparse_model(sparse_dir: Path) -> Path:
    """Return the sparse sub-model with the most registered images.

    COLMAP mapper numbers models 0, 1, 2... but the largest component is not
    always model 0.  Iterate all numbered subdirs, read images.bin, return the
    one with the highest camera count.  Falls back to sparse_dir/'0' if no
    valid model is found.
    """
    best_path = sparse_dir / '0'
    best_count = -1
    for sub in sorted(sparse_dir.iterdir()):
        if not sub.is_dir() or not sub.name.isdigit():
            continue
        images_bin = sub / 'images.bin'
        if not images_bin.exists():
            continue
        try:
            with open(images_bin, 'rb') as f:
                n = struct.unpack('<Q', f.read(8))[0]
            if n > best_count:
                best_count = n
                best_path = sub
        except Exception:
            continue
    return best_path


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


def read_points3d_bin_with_tracks(path: Path) -> Dict[int, Dict[str, Any]]:
    """Read COLMAP points3D.bin including track information.

    Returns {point3d_id: {'xyz': (x,y,z), 'error': float, 'track': [(image_id, point2d_idx), ...]}}.
    """
    points: Dict[int, Dict[str, Any]] = {}
    with open(path, 'rb') as f:
        num_points = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num_points):
            point3d_id = struct.unpack('<Q', f.read(8))[0]
            xyz = struct.unpack('<3d', f.read(24))
            _rgb = struct.unpack('<3B', f.read(3))
            error = struct.unpack('<d', f.read(8))[0]
            track_length = struct.unpack('<Q', f.read(8))[0]
            track = []
            for _ in range(track_length):
                image_id, point2d_idx = struct.unpack('<II', f.read(8))
                track.append((image_id, point2d_idx))
            points[point3d_id] = {'xyz': xyz, 'error': error, 'track': track}
    return points


def read_cameras_bin(path: Path) -> Dict[int, Dict[str, Any]]:
    """Read COLMAP cameras.bin and return camera intrinsics.

    Returns {camera_id: {'model': str, 'width': int, 'height': int, 'params': [float, ...]}}.
    Handles PINHOLE (4 params: fx, fy, cx, cy) and SIMPLE_PINHOLE (3 params: f, cx, cy).
    """
    CAMERA_MODELS = {
        0: ('SIMPLE_PINHOLE', 3),
        1: ('PINHOLE', 4),
        2: ('SIMPLE_RADIAL', 4),
        3: ('RADIAL', 5),
    }
    cameras: Dict[int, Dict[str, Any]] = {}
    with open(path, 'rb') as f:
        num_cameras = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num_cameras):
            camera_id = struct.unpack('<I', f.read(4))[0]
            model_id = struct.unpack('<i', f.read(4))[0]
            width = struct.unpack('<Q', f.read(8))[0]
            height = struct.unpack('<Q', f.read(8))[0]
            model_name, num_params = CAMERA_MODELS.get(model_id, ('UNKNOWN', 0))
            params = list(struct.unpack(f'<{num_params}d', f.read(num_params * 8)))
            cameras[camera_id] = {
                'model': model_name,
                'width': int(width),
                'height': int(height),
                'params': params,
            }
    return cameras


def compute_knn_distances(points_xyz: np.ndarray, k: int = 5, percentile: float = 95.0) -> Dict[str, float]:
    """Compute k-nearest-neighbor distance statistics for a point cloud.

    Args:
        points_xyz: (N, 3) array of 3D point positions.
        k: Number of nearest neighbors.
        percentile: Percentile of kNN distances to report.

    Returns:
        Dict with 'mean_knn_distance', 'percentile_knn_distance', 'k'.
    """
    if len(points_xyz) < k + 1:
        return {'mean_knn_distance': float('inf'), 'percentile_knn_distance': float('inf'), 'k': k}
    try:
        from scipy.spatial import KDTree
        tree = KDTree(points_xyz)
        dists, _ = tree.query(points_xyz, k=k + 1)  # +1 because closest is self
        knn_dists = dists[:, -1]  # k-th neighbor distance
    except ImportError:
        # Fallback: brute-force pairwise (fine for < 50k points)
        from numpy.linalg import norm
        n = len(points_xyz)
        knn_dists = np.zeros(n)
        for i in range(n):
            d = norm(points_xyz - points_xyz[i], axis=1)
            d.sort()
            knn_dists[i] = d[k]  # index k = k-th neighbor (0 is self)
    return {
        'mean_knn_distance': float(np.mean(knn_dists)),
        'percentile_knn_distance': float(np.percentile(knn_dists, percentile)),
        'k': k,
    }


def collect_sparse_summary(workspace: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Build a summary dict from a COLMAP sparse reconstruction."""
    sparse_dir = find_best_sparse_model(workspace / 'sparse')
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

    # Compute kNN density metrics
    knn_cfg = cfg.get('knn_distance', {})
    knn_k = int(knn_cfg.get('k', 5))
    knn_pct = float(knn_cfg.get('percentile', 95.0))
    knn_distance_metrics: Dict[str, float] = {}
    if points:
        pts_array = np.array(list(points.values()), dtype=np.float64)
        knn_distance_metrics = compute_knn_distances(pts_array, k=knn_k, percentile=knn_pct)

    return {
        'total_cameras': total_cameras,
        'aligned_cameras': aligned_count,
        'sparse_point_count': len(points),
        'global_metrics': global_metrics,
        'weak_regions': weak_regions,
        'per_camera_stats': [],
        'knn_distance_metrics': knn_distance_metrics,
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
