#!/usr/bin/env python3
"""
compare_scorers.py — Compare evaluation metrics across scorer runs.

Reads eval/report.json, sparse_metrics/, COLMAP binaries, and the GT mesh from
each run directory and produces plots across four categories:

  Dense cloud metrics (from eval/report.json):
    1. F-score vs distance threshold
    2. Completeness vs distance threshold
    3. Accuracy vs distance threshold
    4. Cloud-to-cloud distance (bar chart)

  Sparse model metrics (from sparse_metrics/ and COLMAP binaries):
    5. Sparse point count over iterations
    6. kNN distance convergence over iterations
    7. Track length distribution (histogram)
    8. Sparse model summary (bar chart)

  GT-mesh-related metrics (sparse cloud aligned to NED vs GT surface):
    9.  Sparse-to-GT threshold metrics (3-panel: F-score, completeness, accuracy)
    10. Per-region completeness radar (azimuth sectors around rock centre)
    11. Hausdorff & GT distance summary (grouped bars)
    12. Accuracy vs track length (binned line chart)

Usage:
    python3 compare_scorers.py \
        <run_dir_1> <run_dir_2> <run_dir_3>

    # Optionally override GT mesh, output dir, etc.:
    python3 compare_scorers.py ... --gt-mesh /path/to/mesh.obj --output-dir /some/path
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree


# ── Scorer display names ─────────────────────────────────────────────────────

_SCORER_LABELS = {
    'covisibility': 'Co-visibility',
    'repair_weighted_covisibility': 'Repair-weighted',
    'baseline_aware_repair_weighted_covisibility': 'Baseline-aware',
    'gt_phase_adaptive_hybrid': 'Adaptive Hybrid',
}

# Ordered palettes — each run gets a unique color/marker by index
_COLOR_PALETTE = [
    '#1976D2',  # blue
    '#E64A19',  # deep orange
    '#388E3C',  # green
    '#7B1FA2',  # purple
    '#F9A825',  # amber
    '#00838F',  # teal
    '#C62828',  # red
    '#4527A0',  # deep purple
    '#2E7D32',  # dark green
    '#EF6C00',  # orange
]

_MARKER_PALETTE = ['o', 's', '^', 'D', 'v', 'P', 'X', 'h', '<', '>']

# Populated by assign_run_styles() before plotting
_run_styles: Dict[str, dict] = {}   # run_key -> {color, marker, label}


def _detect_gate_mode(run_dir: Path) -> Optional[str]:
    """Read gate_mode from the first shadow log, if present."""
    shadow_dir = run_dir / 'candidates' / 'hybrid_shadow'
    if not shadow_dir.is_dir():
        return None
    for f in sorted(shadow_dir.glob('shadow_iter_*.json')):
        try:
            record = json.loads(f.read_text())
            return record.get('gate_mode')
        except (json.JSONDecodeError, OSError):
            continue
    return None


def assign_run_styles(run_keys: list, run_dirs: list) -> None:
    """Assign unique color/marker/label to each run by index."""
    _run_styles.clear()
    from collections import Counter
    name_counts = Counter(run_keys)
    for i, (key, run_dir) in enumerate(zip(run_keys, run_dirs)):
        base_label = _SCORER_LABELS.get(key, key)
        if name_counts[key] > 1:
            # Disambiguate hybrid runs by gate mode (oracle/heuristic/learned)
            gate_mode = _detect_gate_mode(run_dir)
            if gate_mode:
                label = f'{base_label} ({gate_mode})'
            else:
                # Fallback to timestamp if no shadow logs
                ts = str(run_dir.name).replace('unified_run_', '')
                label = f'{base_label} ({ts})'
        else:
            label = base_label
        _run_styles[f'{key}_{i}'] = {
            'color': _COLOR_PALETTE[i % len(_COLOR_PALETTE)],
            'marker': _MARKER_PALETTE[i % len(_MARKER_PALETTE)],
            'label': label,
        }


def _label(name: str) -> str:
    if name in _run_styles:
        return _run_styles[name]['label']
    return _SCORER_LABELS.get(name, name)


def _color(name: str) -> str:
    if name in _run_styles:
        return _run_styles[name]['color']
    return '#757575'


def _marker(name: str) -> str:
    if name in _run_styles:
        return _run_styles[name]['marker']
    return 'D'


# ── COLMAP binary readers (standalone, no external deps) ─────────────────────

def _read_points3d_bin_with_tracks(path: Path) -> Dict[int, Dict[str, Any]]:
    """Read COLMAP points3D.bin including track information."""
    points: Dict[int, Dict[str, Any]] = {}
    with open(path, 'rb') as f:
        num_points = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num_points):
            point3d_id = struct.unpack('<Q', f.read(8))[0]
            xyz = struct.unpack('<3d', f.read(24))
            _rgb = struct.unpack('<3B', f.read(3))
            error = struct.unpack('<d', f.read(8))[0]
            track_length = struct.unpack('<Q', f.read(8))[0]
            f.read(track_length * 8)  # skip track entries
            points[point3d_id] = {
                'xyz': xyz, 'error': error, 'track_length': track_length,
            }
    return points


def _find_best_sparse_model(sparse_dir: Path) -> Optional[Path]:
    """Find the sparse model subdirectory with the most registered images."""
    best, best_count = None, -1
    if not sparse_dir.exists():
        return None
    for sub in sorted(sparse_dir.iterdir()):
        if sub.is_dir() and sub.name.isdigit() and (sub / 'images.bin').exists():
            try:
                with open(sub / 'images.bin', 'rb') as f:
                    n = struct.unpack('<Q', f.read(8))[0]
                if n > best_count:
                    best_count, best = n, sub
            except (OSError, struct.error):
                continue
    return best


# ── OBJ mesh reader + uniform surface sampler ────────────────────────────────

def load_obj_mesh(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read a Wavefront OBJ → (vertices Nx3, faces Mx3)."""
    verts: List[List[float]] = []
    faces: List[List[int]] = []
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('v '):
                p = line.split()
                verts.append([float(p[1]), float(p[2]), float(p[3])])
            elif line.startswith('f '):
                idx = [int(p.split('/')[0]) - 1 for p in line.split()[1:]]
                for k in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[k], idx[k + 1]])
    if not verts:
        raise ValueError(f'No vertices in {path}')
    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int64)


def sample_mesh_uniformly(
    verts: np.ndarray, faces: np.ndarray, n: int, rng: np.random.Generator,
) -> np.ndarray:
    """Area-weighted uniform surface sampling via barycentric coordinates."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    total = areas.sum()
    if total == 0:
        raise ValueError('Mesh has zero area')
    fi = rng.choice(len(faces), size=n, p=areas / total)
    r1, r2 = rng.random(n), rng.random(n)
    sr1 = np.sqrt(r1)
    u, v, w = 1 - sr1, sr1 * (1 - r2), sr1 * r2
    return (
        u[:, None] * verts[faces[fi, 0]]
        + v[:, None] * verts[faces[fi, 1]]
        + w[:, None] * verts[faces[fi, 2]]
    ).astype(np.float64)


def _apply_T(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    ones = np.ones((len(pts), 1))
    return (T @ np.hstack([pts, ones]).T).T[:, :3]


# ── Umeyama alignment helpers ───────────────────────────────────────────────

def _qvec_to_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z,  2*x*y - 2*w*z,    2*x*z + 2*w*y],
        [2*x*y + 2*w*z,      1 - 2*x*x - 2*z*z,  2*y*z - 2*w*x],
        [2*x*z - 2*w*y,      2*y*z + 2*w*x,    1 - 2*x*x - 2*y*y],
    ])


def _read_images_bin(path: Path) -> Dict[str, np.ndarray]:
    """Return {image_name: camera_centre_xyz} from COLMAP images.bin."""
    centres: Dict[str, np.ndarray] = {}
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            struct.unpack('<I', f.read(4))  # image_id
            qvec = np.array(struct.unpack('<4d', f.read(32)))
            tvec = np.array(struct.unpack('<3d', f.read(24)))
            struct.unpack('<I', f.read(4))  # camera_id
            name = b''
            while True:
                ch = f.read(1)
                if ch == b'\x00':
                    break
                name += ch
            n2d = struct.unpack('<Q', f.read(8))[0]
            f.read(n2d * 24)
            R = _qvec_to_R(qvec)
            centres[name.decode()] = -R.T @ tvec
    return centres


def _load_ned_positions(meta_dirs: List[Path]) -> Dict[str, np.ndarray]:
    positions: Dict[str, np.ndarray] = {}
    for d in meta_dirs:
        if not d.exists():
            continue
        for jf in sorted(d.glob('*.json')):
            m = json.loads(jf.read_text())
            img = m.get('image_file')
            pos = m.get('vehicle_position_ned_m')
            if img and pos and 'x' in pos:
                positions[img] = np.array([pos['x'], pos['y'], pos['z']])
    return positions


def _umeyama(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """dst ≈ s * R @ src + t  (7-DOF similarity)."""
    n = len(src)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    var_s = (sc ** 2).sum() / n
    cov = (dc.T @ sc) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / var_s)
    return s, R, dst.mean(0) - s * R @ src.mean(0)


def _read_ply_xyz(path: Path) -> np.ndarray:
    """Read XYZ coordinates from a PLY file (ascii or binary_little_endian)."""
    with open(path, 'rb') as f:
        header_lines = []
        while True:
            line = f.readline().decode('ascii', errors='ignore').strip()
            header_lines.append(line)
            if line == 'end_header':
                break
        # Parse header
        vertex_count = 0
        fmt = 'ascii'
        for hl in header_lines:
            if hl.startswith('element vertex'):
                vertex_count = int(hl.split()[-1])
            elif hl.startswith('format'):
                fmt = hl.split()[1]
        if vertex_count == 0:
            return np.empty((0, 3), dtype=np.float64)
        if fmt == 'ascii':
            pts = []
            for _ in range(vertex_count):
                parts = f.readline().decode().split()
                pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
            return np.array(pts, dtype=np.float64)
        else:
            # Count float properties to know the stride
            props = []
            in_vertex = False
            for hl in header_lines:
                if hl.startswith('element vertex'):
                    in_vertex = True
                    continue
                if in_vertex and hl.startswith('property'):
                    props.append(hl)
                elif in_vertex and hl.startswith('element'):
                    break
            stride = len(props) * 4  # assume float32 per property
            data = np.frombuffer(f.read(vertex_count * stride), dtype=np.float32)
            data = data.reshape(vertex_count, len(props))
            return data[:, :3].astype(np.float64)


# ── Sparse-to-GT alignment + metrics ────────────────────────────────────────

def align_sparse_to_ned(
    run_dir: Path,
    rock_center: np.ndarray,
    bbox_half: float = 3.0,
) -> Optional[Dict]:
    """Align the final sparse model to NED and return points with metadata.

    Returns dict with keys: pts_ned (Nx3), track_lengths (N,), reproj_errors (N,),
    alignment_info, or None if alignment fails.
    """
    sparse_dir = run_dir / 'colmap' / 'sparse'
    model_dir = _find_best_sparse_model(sparse_dir)
    if model_dir is None:
        return None

    images_bin = model_dir / 'images.bin'
    points_bin = model_dir / 'points3D.bin'
    if not images_bin.exists() or not points_bin.exists():
        return None

    # Read COLMAP cameras and sparse points
    colmap_centres = _read_images_bin(images_bin)
    meta_dirs = [run_dir / 'seed' / 'metadata', run_dir / 'adaptive' / 'metadata']
    ned_positions = _load_ned_positions(meta_dirs)

    src, dst = [], []
    for name, c in colmap_centres.items():
        if name in ned_positions:
            src.append(c)
            dst.append(ned_positions[name])
    if len(src) < 3:
        return None

    src_a, dst_a = np.array(src), np.array(dst)
    s, R, t = _umeyama(src_a, dst_a)

    # Read sparse points with track info
    points = _read_points3d_bin_with_tracks(points_bin)
    if not points:
        return None

    xyz_colmap = np.array([p['xyz'] for p in points.values()], dtype=np.float64)
    track_lengths = np.array([p['track_length'] for p in points.values()], dtype=np.float64)
    reproj_errors = np.array([p['error'] for p in points.values()], dtype=np.float64)

    # Transform to NED
    pts_ned = s * (R @ xyz_colmap.T).T + t

    # Bbox filter around rock center
    mask = np.all(np.abs(pts_ned - rock_center) <= bbox_half, axis=1)
    pts_ned = pts_ned[mask]
    track_lengths = track_lengths[mask]
    reproj_errors = reproj_errors[mask]

    transformed_cams = s * (R @ src_a.T).T + t
    residuals = np.linalg.norm(transformed_cams - dst_a, axis=1)

    return {
        'pts_ned': pts_ned,
        'track_lengths': track_lengths,
        'reproj_errors': reproj_errors,
        'num_points': len(pts_ned),
        'alignment_rms_m': float(np.sqrt(np.mean(residuals ** 2))),
        'matched_cameras': len(src),
    }


def compute_gt_metrics(
    gt_samples: np.ndarray, cloud: np.ndarray, thresholds: List[float],
) -> Dict:
    """Compute completeness, accuracy, F-score, Hausdorff, and C2C vs GT."""
    if len(cloud) == 0:
        empty: Dict = {}
        for t in thresholds:
            k = f'{t * 1000:.0f}mm'
            empty[f'completeness_{k}'] = 0.0
            empty[f'accuracy_{k}'] = 0.0
            empty[f'fscore_{k}'] = 0.0
        empty.update(hausdorff_gt2r_m=float('inf'), hausdorff_r2gt_m=float('inf'),
                     mean_c2c_m=float('inf'), p95_c2c_m=float('inf'))
        return empty

    gt2r, _ = KDTree(cloud).query(gt_samples, k=1)
    r2gt, _ = KDTree(gt_samples).query(cloud, k=1)

    out: Dict = {}
    for t in thresholds:
        k = f'{t * 1000:.0f}mm'
        c = float(np.mean(gt2r <= t))
        a = float(np.mean(r2gt <= t))
        f = 2 * c * a / (c + a) if c + a > 0 else 0.0
        out[f'completeness_{k}'] = round(c * 100, 3)
        out[f'accuracy_{k}'] = round(a * 100, 3)
        out[f'fscore_{k}'] = round(f * 100, 3)

    out['hausdorff_gt2r_m'] = float(gt2r.max())
    out['hausdorff_r2gt_m'] = float(r2gt.max())
    out['mean_c2c_m'] = float(gt2r.mean())
    out['p95_c2c_m'] = float(np.percentile(gt2r, 95))
    out['_gt2r'] = gt2r  # keep raw distances for per-region analysis
    out['_r2gt'] = r2gt
    return out


def load_cleaned_dense_cloud(run_dir: Path) -> Optional[np.ndarray]:
    """Load the cleaned dense cloud from eval/cleaned_nbv.ply."""
    cleaned = run_dir / 'eval' / 'cleaned_nbv.ply'
    if not cleaned.exists():
        return None
    pts = _read_ply_xyz(cleaned)
    return pts if len(pts) > 0 else None


# ── Data loading ──────────────────────────────────────────────────────────────

def load_run(run_dir: Path) -> Tuple[str, Dict]:
    """Load report.json and return (scorer_name, nbv_cloud_data)."""
    report_path = run_dir / 'eval' / 'report.json'
    if not report_path.exists():
        sys.exit(f'ERROR: {report_path} not found')

    report = json.loads(report_path.read_text())
    scorer_name = report.get('mission_params', {}).get('scorer_name', run_dir.name)

    nbv = report.get('clouds', {}).get('nbv', {})
    if 'metrics' not in nbv:
        sys.exit(f'ERROR: no dense cloud metrics in {report_path}')

    return scorer_name, nbv


def load_iteration_metrics(run_dir: Path) -> List[Dict]:
    """Load per-iteration sparse metrics, sorted by iteration number."""
    metrics_dir = run_dir / 'sparse_metrics'
    if not metrics_dir.exists():
        return []

    results = []
    for jf in sorted(metrics_dir.glob('metrics_iter_*.json')):
        m = re.search(r'metrics_iter_(\d+)\.json', jf.name)
        if not m:
            continue
        iteration = int(m.group(1))
        data = json.loads(jf.read_text())
        knn = data.get('knn_distance_metrics', {})
        results.append({
            'iteration': iteration,
            'sparse_point_count': data.get('sparse_point_count', 0),
            'aligned_cameras': data.get('aligned_cameras', 0),
            'total_images_on_disk': data.get('total_images_on_disk', 0),
            'knn_mean': knn.get('mean_knn_distance', 0.0),
            'knn_p95': knn.get('percentile_knn_distance', 0.0),
        })

    results.sort(key=lambda r: r['iteration'])
    return results


def load_final_sparse_stats(run_dir: Path) -> Optional[Dict]:
    """Load track length and reprojection error from the final COLMAP model."""
    sparse_dir = run_dir / 'colmap' / 'sparse'
    model_dir = _find_best_sparse_model(sparse_dir)
    if model_dir is None:
        return None

    points_path = model_dir / 'points3D.bin'
    if not points_path.exists():
        return None

    points = _read_points3d_bin_with_tracks(points_path)
    if not points:
        return None

    track_lengths = np.array([p['track_length'] for p in points.values()], dtype=np.float64)
    reproj_errors = np.array([p['error'] for p in points.values()], dtype=np.float64)

    return {
        'track_lengths': track_lengths,
        'reproj_errors': reproj_errors,
        'num_points': len(points),
        'mean_track_length': float(np.mean(track_lengths)),
        'median_track_length': float(np.median(track_lengths)),
        'mean_reproj_error': float(np.mean(reproj_errors)),
        'median_reproj_error': float(np.median(reproj_errors)),
    }


def extract_threshold_metrics(
    metrics: Dict, prefix: str
) -> Tuple[List[float], List[float]]:
    """Extract (thresholds_mm, values) for a given metric prefix."""
    pairs = []
    for key, val in metrics.items():
        m = re.match(rf'{prefix}_(\d+)mm', key)
        if m:
            pairs.append((int(m.group(1)), float(val)))
    pairs.sort(key=lambda p: p[0])
    return [p[0] for p in pairs], [p[1] for p in pairs]


# ── Dense cloud plots ────────────────────────────────────────────────────────

def _style_threshold_ax(ax, ylabel: str, thresholds_mm: List[float]) -> None:
    ax.set_xlabel('Distance threshold (mm)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(thresholds_mm)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)


def plot_threshold_metric(
    runs: List[Tuple[str, Dict]],
    metric_prefix: str,
    title: str,
    ylabel: str,
    save_path: Path,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))

    for scorer_name, nbv in runs:
        thresholds_mm, values = extract_threshold_metrics(
            nbv['metrics'], metric_prefix
        )
        ax.plot(
            thresholds_mm, values,
            marker=_marker(scorer_name),
            color=_color(scorer_name),
            label=_label(scorer_name),
            linewidth=2, markersize=8,
        )

    ax.set_title(title, fontsize=14, fontweight='bold')
    _style_threshold_ax(ax, ylabel, thresholds_mm)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f'  Saved: {save_path}')
    return fig


def plot_c2c_distances(
    runs: List[Tuple[str, Dict]],
    save_path: Path,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))

    c2c_keys = ['mean_c2c_m', 'median_c2c_m', 'p95_c2c_m']
    c2c_labels = ['Mean', 'Median', '95th percentile']
    x = np.arange(len(c2c_labels))
    width = 0.25

    for i, (scorer_name, nbv) in enumerate(runs):
        metrics = nbv['metrics']
        vals = [metrics.get(k, 0.0) * 1000 for k in c2c_keys]  # convert to mm
        bars = ax.bar(
            x + i * width, vals, width,
            label=_label(scorer_name),
            color=_color(scorer_name),
            alpha=0.85,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val:.1f}',
                ha='center', va='bottom', fontsize=9,
            )

    ax.set_title('Cloud-to-Cloud Distance', fontsize=14, fontweight='bold')
    ax.set_xlabel('Statistic', fontsize=12)
    ax.set_ylabel('Distance (mm)', fontsize=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels(c2c_labels, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f'  Saved: {save_path}')
    return fig


# ── Sparse model plots ───────────────────────────────────────────────────────

def plot_iteration_series(
    runs_iter: List[Tuple[str, List[Dict]]],
    y_key: str,
    title: str,
    ylabel: str,
    save_path: Path,
) -> plt.Figure:
    """Generic line chart for per-iteration metrics."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for scorer_name, iter_data in runs_iter:
        if not iter_data:
            continue
        iterations = [d['iteration'] for d in iter_data]
        values = [d[y_key] for d in iter_data]
        ax.plot(
            iterations, values,
            marker=_marker(scorer_name),
            color=_color(scorer_name),
            label=_label(scorer_name),
            linewidth=2, markersize=5, markevery=max(1, len(iterations) // 15),
        )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f'  Saved: {save_path}')
    return fig


def plot_track_length_distribution(
    runs_sparse: List[Tuple[str, Dict]],
    save_path: Path,
) -> plt.Figure:
    """Overlaid histograms of track length per scorer."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Find global max track length for consistent bins
    all_max = max(
        int(np.max(stats['track_lengths'])) for _, stats in runs_sparse if stats
    )
    bins = np.arange(2, min(all_max + 2, 50), 1)

    for scorer_name, stats in runs_sparse:
        if not stats:
            continue
        tl = stats['track_lengths']
        ax.hist(
            tl, bins=bins,
            color=_color(scorer_name),
            alpha=0.5,
            label=f'{_label(scorer_name)} (mean={stats["mean_track_length"]:.1f})',
            edgecolor=_color(scorer_name),
            linewidth=0.8,
        )

    ax.set_title('Track Length Distribution (Final Sparse Model)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Track length', fontsize=12)
    ax.set_ylabel('Number of points', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f'  Saved: {save_path}')
    return fig


def plot_sparse_summary_bars(
    runs_sparse: List[Tuple[str, Dict]],
    save_path: Path,
) -> plt.Figure:
    """Grouped bar chart: mean/median track length, mean reprojection error."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    scorer_names = [name for name, _ in runs_sparse]
    labels = [_label(n) for n in scorer_names]
    colors = [_color(n) for n in scorer_names]

    # Panel 1: Mean track length
    vals = [s['mean_track_length'] if s else 0 for _, s in runs_sparse]
    bars = axes[0].bar(labels, vals, color=colors, alpha=0.85)
    for bar, val in zip(bars, vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                     f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    axes[0].set_title('Mean Track Length', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Track length', fontsize=11)
    axes[0].grid(True, axis='y', alpha=0.3)

    # Panel 2: Median track length
    vals = [s['median_track_length'] if s else 0 for _, s in runs_sparse]
    bars = axes[1].bar(labels, vals, color=colors, alpha=0.85)
    for bar, val in zip(bars, vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    axes[1].set_title('Median Track Length', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Track length', fontsize=11)
    axes[1].grid(True, axis='y', alpha=0.3)

    # Panel 3: Mean reprojection error
    vals = [s['mean_reproj_error'] if s else 0 for _, s in runs_sparse]
    bars = axes[2].bar(labels, vals, color=colors, alpha=0.85)
    for bar, val in zip(bars, vals):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    axes[2].set_title('Mean Reprojection Error', fontsize=13, fontweight='bold')
    axes[2].set_ylabel('Error (px)', fontsize=11)
    axes[2].grid(True, axis='y', alpha=0.3)

    fig.suptitle('Sparse Model Quality Summary', fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f'  Saved: {save_path}')
    return fig


# ── GT-mesh-related plots ───────────────────────────────────────────────────

def plot_sparse_gt_threshold(
    runs_gt: List[Tuple[str, Dict]],
    save_path: Path,
) -> plt.Figure:
    """3-panel plot: sparse-to-GT F-score, completeness, accuracy vs threshold."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    prefixes = ['fscore', 'completeness', 'accuracy']
    titles = ['F-score', 'Completeness', 'Accuracy']

    for ax, prefix, title in zip(axes, prefixes, titles):
        for scorer_name, gt_metrics in runs_gt:
            thresholds_mm, values = extract_threshold_metrics(gt_metrics, prefix)
            if not thresholds_mm:
                continue
            ax.plot(
                thresholds_mm, values,
                marker=_marker(scorer_name), color=_color(scorer_name),
                label=_label(scorer_name), linewidth=2, markersize=8,
            )
        ax.set_title(f'Sparse-to-GT {title}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Distance threshold (mm)', fontsize=11)
        ax.set_ylabel(f'{title} (%)', fontsize=11)
        if thresholds_mm:
            ax.set_xticks(thresholds_mm)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)

    fig.suptitle('Sparse Cloud vs Ground Truth', fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f'  Saved: {save_path}')
    return fig


def plot_per_region_completeness(
    runs_gt: List[Tuple[str, Dict, np.ndarray]],
    gt_samples: np.ndarray,
    rock_center: np.ndarray,
    n_sectors: int,
    threshold_m: float,
    save_path: Path,
) -> plt.Figure:
    """Radar plot of completeness by azimuth sector around rock centre.

    runs_gt: [(scorer_name, gt_metrics_with_raw_dists, cloud_pts), ...]
    """
    # Compute azimuth of each GT sample relative to rock centre (in NED XY plane)
    dx = gt_samples[:, 0] - rock_center[0]
    dy = gt_samples[:, 1] - rock_center[1]
    gt_azimuths = np.arctan2(dy, dx)  # -pi to pi

    sector_edges = np.linspace(-np.pi, np.pi, n_sectors + 1)
    sector_labels = [f'{int(np.degrees(sector_edges[i]))}\u00b0' for i in range(n_sectors)]
    angles = np.linspace(0, 2 * np.pi, n_sectors, endpoint=False)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for scorer_name, gt_metrics, cloud_pts in runs_gt:
        if cloud_pts is None or len(cloud_pts) == 0:
            continue
        tree = KDTree(cloud_pts)
        sector_completeness = []
        for i in range(n_sectors):
            lo, hi = sector_edges[i], sector_edges[i + 1]
            mask = (gt_azimuths >= lo) & (gt_azimuths < hi)
            if mask.sum() == 0:
                sector_completeness.append(0.0)
                continue
            dists, _ = tree.query(gt_samples[mask], k=1)
            sector_completeness.append(float(np.mean(dists <= threshold_m)) * 100)

        vals = sector_completeness + [sector_completeness[0]]  # close polygon
        theta = list(angles) + [angles[0]]
        ax.plot(theta, vals, marker=_marker(scorer_name), color=_color(scorer_name),
                label=_label(scorer_name), linewidth=2, markersize=6)
        ax.fill(theta, vals, color=_color(scorer_name), alpha=0.08)

    ax.set_thetagrids(np.degrees(angles), sector_labels, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_ylabel('')
    ax.set_title(
        f'Per-Region Completeness @ {threshold_m*1000:.0f} mm\n(azimuth sectors around rock centre)',
        fontsize=13, fontweight='bold', pad=20,
    )
    ax.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.25, 1.1))
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f'  Saved: {save_path}')
    return fig


def plot_hausdorff_summary(
    runs_gt_sparse: List[Tuple[str, Dict]],
    runs_gt_dense: List[Tuple[str, Dict]],
    save_path: Path,
) -> plt.Figure:
    """Grouped bar chart: Hausdorff (GT→recon), mean C2C, p95 C2C for sparse and dense."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, label, runs in zip(axes, ['Sparse Cloud', 'Dense Cloud'],
                                [runs_gt_sparse, runs_gt_dense]):
        scorer_names = [n for n, _ in runs]
        labels = [_label(n) for n in scorer_names]
        colors = [_color(n) for n in scorer_names]

        metrics_keys = [
            ('hausdorff_gt2r_m', 'Hausdorff\n(GT→recon)'),
            ('mean_c2c_m', 'Mean C2C'),
            ('p95_c2c_m', 'P95 C2C'),
        ]
        x = np.arange(len(metrics_keys))
        width = 0.25

        for i, (scorer_name, gm) in enumerate(runs):
            vals = [gm.get(k, 0.0) * 1000 for k, _ in metrics_keys]  # to mm
            bars = ax.bar(x + i * width, vals, width, label=_label(scorer_name),
                          color=colors[i], alpha=0.85)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=8)

        ax.set_title(f'{label} vs GT', fontsize=13, fontweight='bold')
        ax.set_ylabel('Distance (mm)', fontsize=11)
        ax.set_xticks(x + width)
        ax.set_xticklabels([lbl for _, lbl in metrics_keys], fontsize=9)
        ax.legend(fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle('Hausdorff & Distance Summary', fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f'  Saved: {save_path}')
    return fig


def plot_accuracy_vs_track_length(
    runs_aligned: List[Tuple[str, Dict]],
    gt_samples: np.ndarray,
    thresholds_m: List[float],
    save_path: Path,
) -> plt.Figure:
    """Accuracy (% within threshold) binned by track length.

    runs_aligned: [(scorer_name, aligned_data), ...] where aligned_data has
    pts_ned, track_lengths, reproj_errors.
    """
    # Use the middle threshold
    t = thresholds_m[len(thresholds_m) // 2]
    bins = [(2, 3), (3, 5), (5, 8), (8, 12), (12, 999)]
    bin_labels = ['2', '3-4', '5-7', '8-11', '12+']

    fig, ax = plt.subplots(figsize=(8, 5))

    for scorer_name, adata in runs_aligned:
        pts = adata['pts_ned']
        tl = adata['track_lengths']
        if len(pts) == 0:
            continue

        gt_tree = KDTree(gt_samples)
        dists, _ = gt_tree.query(pts, k=1)

        bin_accs = []
        for lo, hi in bins:
            mask = (tl >= lo) & (tl < hi)
            if mask.sum() == 0:
                bin_accs.append(np.nan)
            else:
                bin_accs.append(float(np.mean(dists[mask] <= t)) * 100)

        ax.plot(bin_labels, bin_accs, marker=_marker(scorer_name),
                color=_color(scorer_name), label=_label(scorer_name),
                linewidth=2, markersize=8)

    ax.set_title(f'Sparse Point Accuracy vs Track Length @ {t*1000:.0f} mm',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Track length (number of observing cameras)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f'  Saved: {save_path}')
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

_DEFAULT_GT_MESH = (
    '~/PX4-Autopilot/Tools/simulation/gz/models/lunar_sample_15016/'
    'meshes/15016-0_SFM_Web-Resolution-Model_Coordinate-Registered.obj'
)

_DEFAULT_T_GT = np.array([
    [0,  20,  0,   0.0],
    [20,  0,  0,   8.0],
    [0,   0, -20, -0.8],
    [0,   0,  0,   1.0],
], dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser(
        description='Compare evaluation metrics across scorer runs'
    )
    ap.add_argument(
        'run_dirs', nargs='+', type=str,
        help='Paths to unified_run_* directories (one per scorer)'
    )
    ap.add_argument(
        '--output-dir', type=str,
        default='/home/dreamslab/Desktop/results/final results',
        help='Base directory for output (a timestamped subfolder is created)'
    )
    ap.add_argument(
        '--gt-mesh', type=str, default=_DEFAULT_GT_MESH,
        help='Path to GT mesh (.obj)'
    )
    ap.add_argument(
        '--rock-center', nargs=3, type=float, default=[0.0, 8.0, -0.8],
        metavar=('X', 'Y', 'Z'), help='Rock centre in NED (m)'
    )
    ap.add_argument(
        '--gt-samples', type=int, default=100_000,
        help='Number of GT surface samples'
    )
    ap.add_argument(
        '--thresholds', nargs='+', type=float,
        default=[0.016, 0.032, 0.048, 0.064, 0.080],
        help='Metric thresholds (m)'
    )
    ap.add_argument(
        '--no-show', action='store_true',
        help='Save figures without displaying windows'
    )
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    if args.no_show:
        import matplotlib as _mpl
        _mpl.use('Agg')

    # Create a new timestamped subfolder for each run
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f'comparison_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    rock_center = np.array(args.rock_center)
    rng = np.random.default_rng(args.seed)

    # ── Load GT mesh ─────────────────────────────────────────────────────
    gt_mesh_path = Path(args.gt_mesh).expanduser()
    if not gt_mesh_path.exists():
        print(f'WARNING: GT mesh not found: {gt_mesh_path} — skipping GT metrics')
        gt_samples = None
    else:
        print(f'Loading GT mesh: {gt_mesh_path.name}')
        gt_verts, gt_faces = load_obj_mesh(gt_mesh_path)
        gt_verts = _apply_T(gt_verts, _DEFAULT_T_GT)
        gt_samples = sample_mesh_uniformly(gt_verts, gt_faces, args.gt_samples, rng)
        print(f'  {len(gt_verts):,} vertices, {args.gt_samples:,} surface samples')

    # ── Load all runs ────────────────────────────────────────────────────
    run_paths: List[Path] = []
    raw_scorer_names: List[str] = []
    raw_nbvs: List[Dict] = []
    for rd in args.run_dirs:
        run_path = Path(rd).expanduser().resolve()
        if not run_path.exists():
            sys.exit(f'ERROR: run directory not found: {run_path}')
        scorer_name, nbv = load_run(run_path)
        raw_scorer_names.append(scorer_name)
        raw_nbvs.append(nbv)
        run_paths.append(run_path)

    # Assign unique colors/markers/labels per run
    assign_run_styles(raw_scorer_names, run_paths)

    # Build keyed run lists using indexed keys
    dense_runs: List[Tuple[str, Dict]] = []
    for i, (scorer_name, run_path, nbv) in enumerate(zip(raw_scorer_names, run_paths, raw_nbvs)):
        key = f'{scorer_name}_{i}'
        dense_runs.append((key, nbv))
        print(f'Loaded: {_label(key)} <- {run_path.name}')

    # Load iteration metrics
    iter_runs: List[Tuple[str, List[Dict]]] = []
    for i, run_path in enumerate(run_paths):
        key = f'{raw_scorer_names[i]}_{i}'
        iter_data = load_iteration_metrics(run_path)
        iter_runs.append((key, iter_data))
        print(f'  Iterations: {len(iter_data)} metrics files')

    # Load final sparse stats
    sparse_runs: List[Tuple[str, Optional[Dict]]] = []
    for i, run_path in enumerate(run_paths):
        key = f'{raw_scorer_names[i]}_{i}'
        stats = load_final_sparse_stats(run_path)
        sparse_runs.append((key, stats))
        if stats:
            print(f'  Sparse: {stats["num_points"]} pts, '
                  f'mean track={stats["mean_track_length"]:.2f}, '
                  f'mean reproj={stats["mean_reproj_error"]:.3f} px')
        else:
            print(f'  Sparse: not available')

    # Load aligned sparse clouds + cleaned dense clouds (for GT metrics)
    aligned_runs: List[Tuple[str, Optional[Dict]]] = []
    dense_cloud_runs: List[Tuple[str, Optional[np.ndarray]]] = []
    for i, run_path in enumerate(run_paths):
        key = f'{raw_scorer_names[i]}_{i}'
        aligned = align_sparse_to_ned(run_path, rock_center)
        aligned_runs.append((key, aligned))
        if aligned:
            print(f'  Aligned sparse: {aligned["num_points"]} pts, '
                  f'RMS={aligned["alignment_rms_m"]*1000:.1f} mm, '
                  f'{aligned["matched_cameras"]} cameras')
        else:
            print(f'  Aligned sparse: not available')

        dense_pts = load_cleaned_dense_cloud(run_path)
        dense_cloud_runs.append((key, dense_pts))
        if dense_pts is not None:
            print(f'  Cleaned dense:  {len(dense_pts):,} pts')
        else:
            print(f'  Cleaned dense:  not available')

    print()

    # ── Dense cloud plots (1-4) ──────────────────────────────────────────
    print('Dense cloud metrics:')
    plot_threshold_metric(
        dense_runs, 'fscore', 'F-score vs Distance Threshold',
        'F-score (%)', output_dir / 'fscore_comparison.png',
    )
    plot_threshold_metric(
        dense_runs, 'completeness', 'Completeness vs Distance Threshold',
        'Completeness (%)', output_dir / 'completeness_comparison.png',
    )
    plot_threshold_metric(
        dense_runs, 'accuracy', 'Accuracy vs Distance Threshold',
        'Accuracy (%)', output_dir / 'accuracy_comparison.png',
    )
    plot_c2c_distances(
        dense_runs, output_dir / 'c2c_distance_comparison.png',
    )

    # ── Sparse model plots (5-8) ─────────────────────────────────────────
    print('\nSparse model metrics:')

    plot_iteration_series(
        iter_runs, 'sparse_point_count',
        'Sparse Point Count over Iterations',
        'Number of 3D points',
        output_dir / 'sparse_point_count.png',
    )

    plot_iteration_series(
        iter_runs, 'knn_p95',
        'kNN Distance Convergence (p95)',
        'p95 kNN distance (COLMAP units)',
        output_dir / 'knn_convergence.png',
    )

    valid_sparse = [(n, s) for n, s in sparse_runs if s is not None]
    if valid_sparse:
        plot_track_length_distribution(
            valid_sparse, output_dir / 'track_length_distribution.png',
        )
        plot_sparse_summary_bars(
            valid_sparse, output_dir / 'sparse_model_summary.png',
        )
    else:
        print('  SKIP: no sparse model data available for track length / summary plots')

    # ── GT-mesh-related plots (9-12) ─────────────────────────────────────
    if gt_samples is not None:
        print('\nGT-mesh-related metrics:')

        # Compute sparse-to-GT metrics
        sparse_gt_runs: List[Tuple[str, Dict]] = []
        for key, adata in aligned_runs:
            if adata is not None:
                gm = compute_gt_metrics(gt_samples, adata['pts_ned'], args.thresholds)
                sparse_gt_runs.append((key, gm))
                print(f'  {_label(key)} sparse-to-GT: '
                      f'F@{args.thresholds[2]*1000:.0f}mm='
                      f'{gm.get(f"fscore_{args.thresholds[2]*1000:.0f}mm", 0):.1f}%, '
                      f'Hausdorff={gm["hausdorff_gt2r_m"]*1000:.1f} mm')

        # Compute dense-to-GT metrics (for Hausdorff comparison)
        dense_gt_runs: List[Tuple[str, Dict]] = []
        for key, dpts in dense_cloud_runs:
            if dpts is not None:
                gm = compute_gt_metrics(gt_samples, dpts, args.thresholds)
                dense_gt_runs.append((key, gm))

        # Plot 9: Sparse-to-GT threshold metrics (3-panel)
        if sparse_gt_runs:
            plot_sparse_gt_threshold(
                sparse_gt_runs, output_dir / 'sparse_to_gt_metrics.png',
            )
        else:
            print('  SKIP: no aligned sparse data for sparse-to-GT plot')

        # Plot 10: Per-region completeness radar
        # Use dense cleaned cloud for more meaningful coverage analysis
        radar_data: List[Tuple[str, Dict, np.ndarray]] = []
        for (sn, gm), (_, dpts) in zip(dense_gt_runs, dense_cloud_runs):
            if dpts is not None:
                radar_data.append((sn, gm, dpts))
        if radar_data:
            # Use the middle threshold for the radar
            radar_thresh = args.thresholds[len(args.thresholds) // 2]
            plot_per_region_completeness(
                radar_data, gt_samples, rock_center,
                n_sectors=12, threshold_m=radar_thresh,
                save_path=output_dir / 'per_region_completeness.png',
            )
        else:
            print('  SKIP: no cleaned dense clouds for per-region completeness')

        # Plot 11: Hausdorff & distance summary (sparse + dense side by side)
        if sparse_gt_runs and dense_gt_runs:
            plot_hausdorff_summary(
                sparse_gt_runs, dense_gt_runs,
                output_dir / 'hausdorff_distance_summary.png',
            )
        else:
            print('  SKIP: insufficient data for Hausdorff summary')

        # Plot 12: Accuracy vs track length
        valid_aligned = [(n, a) for n, a in aligned_runs if a is not None]
        if valid_aligned:
            plot_accuracy_vs_track_length(
                valid_aligned, gt_samples, args.thresholds,
                output_dir / 'accuracy_vs_track_length.png',
            )
        else:
            print('  SKIP: no aligned sparse data for accuracy vs track length')
    else:
        print('\nSKIP: GT mesh not available — GT-related plots omitted')

    print(f'\nAll plots saved to: {output_dir}')

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
