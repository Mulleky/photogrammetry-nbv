#!/usr/bin/env python3
"""
evaluate_run.py — Reconstruction quality evaluation pipeline.

Pipeline per input cloud:
  1. Apply optional 4×4 world/ICP transform
  2. Coarse bbox crop  (rock_center ± crop_half_extent)
  3. Ground-plane removal (RANSAC on lowest 20% of points)
  4. Statistical Outlier Removal (SOR, k=20, std_ratio=2.0)
  5. Distance-gate against GT mesh surface (keep < gate_dist_m)

Metrics (reported at each threshold in --thresholds):
  - Completeness : % of GT surface within threshold of recon cloud
  - Accuracy     : % of recon cloud within threshold of GT surface
  - F-score      : harmonic mean of completeness and accuracy
  - Mean C2C     : mean distance GT→recon cloud (cloud-to-cloud, no mesh needed)

Writes:
  <output_dir>/cleaned_<label>.ply   — cleaned cloud in NED frame
  <output_dir>/report.json           — all metrics

Requires: numpy, scipy
Optional: open3d (for Poisson mesh reconstruction — not implemented here)

Usage example:
  python evaluate_run.py \\
      --gt-mesh ~/PX4-Autopilot/Tools/simulation/gz/models/lunar_sample_15016/meshes/15016-0_SFM_Web-Resolution-Model_Coordinate-Registered.obj \\
      --gt-transform gt_ned.npy \\
      --clouds seed:~/run/aligned_seed.ply nbv:~/run/aligned_nbv.ply \\
      --rock-center 0 8 -0.8 \\
      --output-dir ./eval_output

--gt-transform is a 4×4 numpy matrix (.npy) that maps the mesh from its
model-local frame into NED.  If omitted, the mesh is used as-is.

To build gt_ned.npy for the lunar_sample_15016 placed at Gazebo ENU (8,0,0.8):
  import numpy as np
  # ENU→NED rotation: swap X↔Y, negate Z
  R = np.array([[0,1,0],[1,0,0],[0,0,-1]], dtype=float)
  t = np.array([0, 8, -0.8])          # NED position of model origin
  T = np.eye(4); T[:3,:3] = R; T[:3,3] = t
  np.save('gt_ned.npy', T)
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import KDTree


# ===========================================================================
# OBJ reader + uniform surface sampler
# ===========================================================================

def load_obj_mesh(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return (vertices Nx3, faces Mx3-int) from a .obj file."""
    verts: List[List[float]] = []
    faces: List[List[int]] = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line.startswith('v '):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):
                parts = line.split()[1:]
                # handle v/vt/vn style by splitting on '/'
                indices = [int(p.split('/')[0]) - 1 for p in parts]
                # triangulate fan if >3 vertices
                for k in range(1, len(indices) - 1):
                    faces.append([indices[0], indices[k], indices[k + 1]])
    if not verts:
        raise ValueError(f'No vertices found in {path}')
    if not faces:
        raise ValueError(f'No faces found in {path}')
    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int64)


def sample_mesh_uniformly(
    verts: np.ndarray,
    faces: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Area-weighted uniform surface sampling. Returns (n_samples, 3)."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    total_area = areas.sum()
    if total_area == 0:
        raise ValueError('Mesh has zero total area.')
    probs = areas / total_area

    # Sample faces proportional to area
    face_indices = rng.choice(len(faces), size=n_samples, p=probs)
    # Random barycentric coordinates
    r1 = rng.random(n_samples)
    r2 = rng.random(n_samples)
    sqrt_r1 = np.sqrt(r1)
    u = 1.0 - sqrt_r1
    v = sqrt_r1 * (1.0 - r2)
    w = sqrt_r1 * r2
    f0 = verts[faces[face_indices, 0]]
    f1 = verts[faces[face_indices, 1]]
    f2 = verts[faces[face_indices, 2]]
    samples = u[:, None] * f0 + v[:, None] * f1 + w[:, None] * f2
    return samples.astype(np.float64)


# ===========================================================================
# PLY reader (float/double, binary little-endian or ascii)
# ===========================================================================

def read_ply_xyz(path: Path) -> np.ndarray:
    """Read a PLY file and return vertex positions as (N, 3) float64."""
    with open(path, 'rb') as f:
        raw = f.read()

    end_marker = b'end_header\n'
    end_idx = raw.index(end_marker) + len(end_marker)
    header = raw[:end_idx].decode('utf-8')
    body = raw[end_idx:]
    header_lines = header.strip().split('\n')

    fmt = 'ascii'
    for line in header_lines:
        if line.startswith('format'):
            fmt = line.split()[1]

    vertex_count = 0
    for line in header_lines:
        if line.startswith('element vertex'):
            vertex_count = int(line.split()[-1])

    props: List[Tuple[str, str]] = []
    in_vertex = False
    for line in header_lines:
        if line.startswith('element vertex'):
            in_vertex = True
            continue
        if line.startswith('element') and in_vertex:
            break
        if in_vertex and line.startswith('property'):
            parts = line.split()
            props.append((parts[1], parts[2]))

    def _find(name: str) -> int:
        return next(i for i, (_, n) in enumerate(props) if n == name)

    xi, yi, zi = _find('x'), _find('y'), _find('z')

    if fmt == 'ascii':
        lines = body.decode('utf-8').strip().split('\n')
        pts = np.zeros((vertex_count, 3))
        for i in range(vertex_count):
            p = lines[i].split()
            pts[i] = [float(p[xi]), float(p[yi]), float(p[zi])]
        return pts

    # binary_little_endian
    dtype_map = {
        'float': 'f4', 'double': 'f8',
        'uchar': 'u1', 'int': 'i4', 'uint': 'u4',
        'short': 'i2', 'ushort': 'u2',
    }
    dt = np.dtype([(name, '<' + dtype_map.get(typ, 'f4')) for typ, name in props])
    data = np.frombuffer(body[:vertex_count * dt.itemsize], dtype=dt)
    return np.column_stack([
        data['x'].astype(np.float64),
        data['y'].astype(np.float64),
        data['z'].astype(np.float64),
    ])


# ===========================================================================
# PLY writer (binary little-endian)
# ===========================================================================

def write_ply_xyz(path: Path, pts: np.ndarray) -> None:
    """Write (N, 3) float64 array as a binary PLY."""
    n = len(pts)
    header = (
        'ply\n'
        'format binary_little_endian 1.0\n'
        f'element vertex {n}\n'
        'property float x\n'
        'property float y\n'
        'property float z\n'
        'end_header\n'
    )
    data = pts.astype(np.float32)
    with open(path, 'wb') as f:
        f.write(header.encode('utf-8'))
        f.write(data.tobytes())


# ===========================================================================
# Cloud cleaning steps
# ===========================================================================

def apply_transform(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply 4×4 homogeneous transform to (N, 3) points."""
    n = len(pts)
    ones = np.ones((n, 1), dtype=np.float64)
    pts_h = np.hstack([pts, ones])          # (N, 4)
    return (T @ pts_h.T).T[:, :3]           # (N, 3)


def crop_bbox(
    pts: np.ndarray,
    center: np.ndarray,
    half_extent: float,
) -> np.ndarray:
    """Keep points within an axis-aligned cube."""
    mask = np.all(np.abs(pts - center) <= half_extent, axis=1)
    return pts[mask]


def remove_ground_ransac(
    pts: np.ndarray,
    inlier_thresh_m: float = 0.03,
    n_iter: int = 500,
    low_fraction: float = 0.20,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    RANSAC plane fit on the lowest `low_fraction` of points (by z in NED,
    which means the *largest* z values since NED z grows downward).
    Removes points within inlier_thresh_m of the best-fit plane.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if len(pts) < 10:
        return pts

    # In NED, ground has the *largest* z values (least negative altitude).
    # Pick the lowest 20% by z descending.
    z_threshold = np.percentile(pts[:, 2], (1.0 - low_fraction) * 100)
    low_pts = pts[pts[:, 2] >= z_threshold]
    if len(low_pts) < 3:
        return pts

    best_inlier_count = 0
    best_plane: Optional[np.ndarray] = None

    for _ in range(n_iter):
        idx = rng.choice(len(low_pts), 3, replace=False)
        p0, p1, p2 = low_pts[idx[0]], low_pts[idx[1]], low_pts[idx[2]]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal = normal / norm
        d = -np.dot(normal, p0)

        dist = np.abs(pts @ normal + d)
        count = int((dist <= inlier_thresh_m).sum())
        if count > best_inlier_count:
            best_inlier_count = count
            best_plane = np.append(normal, d)

    if best_plane is None:
        return pts

    dist = np.abs(pts @ best_plane[:3] + best_plane[3])
    return pts[dist > inlier_thresh_m]


def statistical_outlier_removal(
    pts: np.ndarray,
    k: int = 20,
    std_ratio: float = 2.0,
) -> np.ndarray:
    """Remove points whose mean k-NN distance exceeds mean + std_ratio*std."""
    if len(pts) <= k:
        return pts
    tree = KDTree(pts)
    dists, _ = tree.query(pts, k=k + 1)   # first hit is self (dist=0)
    mean_dists = dists[:, 1:].mean(axis=1)
    threshold = mean_dists.mean() + std_ratio * mean_dists.std()
    return pts[mean_dists <= threshold]


def distance_gate(
    pts: np.ndarray,
    gt_samples: np.ndarray,
    gate_dist_m: float,
) -> np.ndarray:
    """Keep recon points within gate_dist_m of any GT surface sample."""
    if len(pts) == 0 or len(gt_samples) == 0:
        return pts
    tree = KDTree(gt_samples)
    dists, _ = tree.query(pts, k=1)
    return pts[dists <= gate_dist_m]


# ===========================================================================
# Metrics
# ===========================================================================

def coverage_at_thresholds(
    gt_samples: np.ndarray,
    recon_pts: np.ndarray,
    thresholds: List[float],
) -> Dict[str, float]:
    """
    Completeness: fraction of GT surface within threshold of recon cloud.
    Accuracy:     fraction of recon cloud within threshold of GT surface.
    F-score:      harmonic mean of completeness and accuracy.
    Mean C2C:     mean GT→recon distance.
    """
    if len(recon_pts) == 0:
        return {
            **{f'completeness_{t*1000:.0f}mm': 0.0 for t in thresholds},
            **{f'accuracy_{t*1000:.0f}mm': 0.0 for t in thresholds},
            **{f'fscore_{t*1000:.0f}mm': 0.0 for t in thresholds},
            'mean_c2c_m': float('inf'),
            'median_c2c_m': float('inf'),
            'p95_c2c_m': float('inf'),
        }

    recon_tree = KDTree(recon_pts)
    gt_tree = KDTree(gt_samples)

    # GT → recon (completeness direction)
    gt_to_recon, _ = recon_tree.query(gt_samples, k=1)
    # recon → GT (accuracy direction)
    recon_to_gt, _ = gt_tree.query(recon_pts, k=1)

    results: Dict[str, float] = {}
    for t in thresholds:
        key = f'{t*1000:.0f}mm'
        comp = float(np.mean(gt_to_recon <= t))
        acc = float(np.mean(recon_to_gt <= t))
        denom = comp + acc
        fscore = 2 * comp * acc / denom if denom > 0 else 0.0
        results[f'completeness_{key}'] = round(comp * 100, 3)
        results[f'accuracy_{key}'] = round(acc * 100, 3)
        results[f'fscore_{key}'] = round(fscore * 100, 3)

    results['mean_c2c_m'] = round(float(gt_to_recon.mean()), 6)
    results['median_c2c_m'] = round(float(np.median(gt_to_recon)), 6)
    results['p95_c2c_m'] = round(float(np.percentile(gt_to_recon, 95)), 6)
    return results


# ===========================================================================
# Main pipeline
# ===========================================================================

def clean_cloud(
    pts: np.ndarray,
    gt_samples: np.ndarray,
    rock_center: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict]:
    """Run all cleaning steps, return (cleaned_pts, step_counts)."""
    counts: Dict[str, int] = {'raw': len(pts)}

    pts = crop_bbox(pts, rock_center, args.crop_half_extent)
    counts['after_bbox'] = len(pts)

    pts = remove_ground_ransac(
        pts,
        inlier_thresh_m=args.ground_thresh_m,
        n_iter=args.ransac_iters,
        low_fraction=args.ground_low_fraction,
        rng=rng,
    )
    counts['after_ground_removal'] = len(pts)

    pts = statistical_outlier_removal(pts, k=args.sor_k, std_ratio=args.sor_std_ratio)
    counts['after_sor'] = len(pts)

    pts = distance_gate(pts, gt_samples, gate_dist_m=args.gate_dist_m)
    counts['after_distance_gate'] = len(pts)

    return pts, counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Evaluate reconstruction quality against a GT mesh.')

    parser.add_argument(
        '--gt-mesh', required=True,
        help='Ground-truth mesh (.obj)')
    parser.add_argument(
        '--gt-transform', default=None,
        help='4×4 transform (numpy .npy) from mesh model frame to NED. '
             'If omitted, mesh is used as-is.')
    parser.add_argument(
        '--clouds', nargs='+', required=True,
        help='label:path pairs, e.g.  seed:aligned_seed.ply  nbv:aligned_nbv.ply')
    parser.add_argument(
        '--rock-center', nargs=3, type=float, default=[0.0, 8.0, -0.8],
        metavar=('X', 'Y', 'Z'),
        help='Rock center in NED (default: 0 8 -0.8)')
    parser.add_argument(
        '--output-dir', required=True,
        help='Directory for cleaned PLYs and report.json')

    # Cloud transform
    parser.add_argument(
        '--cloud-transform', default=None,
        help='Optional 4×4 .npy transform applied to every input cloud '
             '(use if clouds are not yet in NED, e.g. raw COLMAP frame).')

    # Cleaning parameters
    parser.add_argument('--crop-half-extent', type=float, default=1.5,
                        help='Bbox half-extent around rock_center (m, default 1.5)')
    parser.add_argument('--ground-thresh-m', type=float, default=0.03,
                        help='RANSAC inlier distance for ground plane (m, default 0.03)')
    parser.add_argument('--ground-low-fraction', type=float, default=0.20,
                        help='Fraction of lowest points fed to RANSAC (default 0.20)')
    parser.add_argument('--ransac-iters', type=int, default=500,
                        help='RANSAC iterations (default 500)')
    parser.add_argument('--sor-k', type=int, default=20,
                        help='SOR neighbourhood size (default 20)')
    parser.add_argument('--sor-std-ratio', type=float, default=2.0,
                        help='SOR std-deviation multiplier (default 2.0)')
    parser.add_argument('--gate-dist-m', type=float, default=0.05,
                        help='Distance gate vs GT surface (m, default 0.05)')

    # Evaluation parameters
    parser.add_argument(
        '--thresholds', nargs='+', type=float,
        default=[0.005, 0.01, 0.02, 0.05],
        help='Distance thresholds for metrics in metres (default: 5 10 20 50 mm)')
    parser.add_argument(
        '--gt-samples', type=int, default=100_000,
        help='Number of GT surface samples (default 100000)')
    parser.add_argument('--seed', type=int, default=42, help='RNG seed')

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    rock_center = np.array(args.rock_center, dtype=np.float64)

    # ------------------------------------------------------------------
    # 1. Load & sample GT mesh
    # ------------------------------------------------------------------
    print(f'Loading GT mesh: {args.gt_mesh}')
    gt_verts, gt_faces = load_obj_mesh(Path(args.gt_mesh).expanduser())
    print(f'  Vertices: {len(gt_verts):,}  Faces: {len(gt_faces):,}')

    # Apply optional world transform
    if args.gt_transform is not None:
        T_gt = np.load(args.gt_transform)
        if T_gt.shape != (4, 4):
            print(f'ERROR: gt-transform must be a 4×4 matrix, got {T_gt.shape}')
            sys.exit(1)
        gt_verts = apply_transform(gt_verts, T_gt)
        print(f'  Applied GT world transform.')

    print(f'  Sampling {args.gt_samples:,} surface points...')
    gt_samples = sample_mesh_uniformly(gt_verts, gt_faces, args.gt_samples, rng)
    print(f'  GT AABB: x=[{gt_samples[:,0].min():.3f},{gt_samples[:,0].max():.3f}] '
          f'y=[{gt_samples[:,1].min():.3f},{gt_samples[:,1].max():.3f}] '
          f'z=[{gt_samples[:,2].min():.3f},{gt_samples[:,2].max():.3f}]')

    # Load optional cloud transform
    T_cloud: Optional[np.ndarray] = None
    if args.cloud_transform is not None:
        T_cloud = np.load(args.cloud_transform)
        if T_cloud.shape != (4, 4):
            print(f'ERROR: cloud-transform must be 4×4, got {T_cloud.shape}')
            sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Process each input cloud
    # ------------------------------------------------------------------
    report: Dict = {
        'gt_mesh': str(args.gt_mesh),
        'gt_samples': args.gt_samples,
        'rock_center_ned': list(args.rock_center),
        'thresholds_m': args.thresholds,
        'cleaning': {
            'crop_half_extent_m': args.crop_half_extent,
            'ground_thresh_m': args.ground_thresh_m,
            'sor_k': args.sor_k,
            'sor_std_ratio': args.sor_std_ratio,
            'gate_dist_m': args.gate_dist_m,
        },
        'clouds': {},
    }

    for spec in args.clouds:
        if ':' not in spec:
            print(f'ERROR: --clouds entries must be label:path, got "{spec}"')
            sys.exit(1)
        label, cloud_path_str = spec.split(':', 1)
        cloud_path = Path(cloud_path_str).expanduser()

        print(f'\n=== {label}: {cloud_path} ===')
        if not cloud_path.exists():
            print(f'  WARNING: file not found, skipping.')
            report['clouds'][label] = {'error': 'file not found'}
            continue

        pts = read_ply_xyz(cloud_path)
        print(f'  Raw points: {len(pts):,}')

        if T_cloud is not None:
            pts = apply_transform(pts, T_cloud)

        pts, step_counts = clean_cloud(pts, gt_samples, rock_center, args, rng)
        print(f'  After cleaning: {len(pts):,} points')
        print(f'  Steps: {step_counts}')

        if len(pts) == 0:
            print('  WARNING: no points remain after cleaning. Check rock_center and transforms.')
            report['clouds'][label] = {
                'step_counts': step_counts,
                'error': 'no points after cleaning',
            }
            continue

        # Save cleaned cloud
        cleaned_ply = output_dir / f'cleaned_{label}.ply'
        write_ply_xyz(cleaned_ply, pts)
        print(f'  Cleaned cloud -> {cleaned_ply}')

        # Compute metrics
        print(f'  Computing metrics against GT ({args.gt_samples:,} samples)...')
        metrics = coverage_at_thresholds(gt_samples, pts, args.thresholds)

        # Print summary table
        print(f'  {"Threshold":<12} {"Completeness":>14} {"Accuracy":>12} {"F-score":>10}')
        for t in args.thresholds:
            key = f'{t*1000:.0f}mm'
            comp = metrics[f'completeness_{key}']
            acc = metrics[f'accuracy_{key}']
            fs = metrics[f'fscore_{key}']
            print(f'  {key:<12} {comp:>13.1f}% {acc:>11.1f}% {fs:>9.1f}%')
        print(f'  Mean C2C:   {metrics["mean_c2c_m"]*1000:.2f} mm')
        print(f'  Median C2C: {metrics["median_c2c_m"]*1000:.2f} mm')
        print(f'  P95 C2C:    {metrics["p95_c2c_m"]*1000:.2f} mm')

        report['clouds'][label] = {
            'cloud_path': str(cloud_path),
            'cleaned_ply': str(cleaned_ply),
            'step_counts': step_counts,
            'metrics': metrics,
        }

    # ------------------------------------------------------------------
    # 3. Cross-cloud comparison (if two clouds provided)
    # ------------------------------------------------------------------
    cloud_labels = [k for k in report['clouds'] if 'metrics' in report['clouds'][k]]
    if len(cloud_labels) == 2:
        a_label, b_label = cloud_labels[0], cloud_labels[1]
        a_m = report['clouds'][a_label]['metrics']
        b_m = report['clouds'][b_label]['metrics']
        print(f'\n=== Comparison: {b_label} vs {a_label} ===')
        comparison: Dict = {}
        for t in args.thresholds:
            key = f'{t*1000:.0f}mm'
            for metric in ('completeness', 'accuracy', 'fscore'):
                mk = f'{metric}_{key}'
                delta = b_m[mk] - a_m[mk]
                comparison[f'd_{mk}'] = round(delta, 3)
                sign = '+' if delta >= 0 else ''
                print(f'  {metric}@{key}: {a_m[mk]:.1f}% -> {b_m[mk]:.1f}%  ({sign}{delta:.1f}pp)')
        delta_c2c = b_m['mean_c2c_m'] - a_m['mean_c2c_m']
        comparison['d_mean_c2c_m'] = round(delta_c2c, 6)
        sign = '+' if delta_c2c >= 0 else ''
        print(f'  Mean C2C: {a_m["mean_c2c_m"]*1000:.2f} mm -> {b_m["mean_c2c_m"]*1000:.2f} mm  '
              f'({sign}{delta_c2c*1000:.2f} mm)')
        report['comparison'] = {
            'baseline': a_label,
            'improved': b_label,
            'deltas': comparison,
        }

    # ------------------------------------------------------------------
    # 4. Write report
    # ------------------------------------------------------------------
    report_path = output_dir / 'report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\nReport written to {report_path}')


if __name__ == '__main__':
    main()
