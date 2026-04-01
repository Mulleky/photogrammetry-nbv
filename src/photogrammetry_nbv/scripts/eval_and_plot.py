#!/usr/bin/env python3
"""
eval_and_plot.py — Full evaluation + visualization for a photogrammetry-covisibility NBV run.

Auto-discovers all files from the run directory. Runs alignment, cleaning, and
quality metrics inline, then saves two figures:

  Figure 1 — Reconstruction quality metrics (F-score, completeness, accuracy,
              C2C distances, cleaning funnel, sparse evolution, score breakdown)
  Figure 2 — 3D drone trajectory: seed spline, NBV views, candidate hemisphere,
              rock centre (gold star), and aligned reconstruction cloud

Usage:
    python3 eval_and_plot.py \\
        --run-dir ~/photogrammetry_NBV/data/photogrammetry/unified_run_YYYYMMDD_HHMMSS \\
        --gt-mesh ~/PX4-Autopilot/Tools/simulation/gz/models/lunar_sample_15016/\\
meshes/15016-0_SFM_Web-Resolution-Model_Coordinate-Registered.obj

    # Optional: override rock centre, bbox, thresholds, etc.
    python3 eval_and_plot.py --help

    # Skip interactive window (e.g. on a headless server):
    python3 eval_and_plot.py ... --no-show

Output written to <run-dir>/eval/  (report.json + metrics.png + trajectory_3d.png).
"""
from __future__ import annotations

# ── backend must be set before pyplot is imported ──────────────────────────────
import sys as _sys
if '--no-show' in _sys.argv:
    import matplotlib as _mpl
    _mpl.use('Agg')

import argparse
import json
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D        # noqa: F401  (kept for legend handles)
from matplotlib.patches import Patch       # noqa: F401
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401  (registers 3d projection)
from scipy.interpolate import splev, splprep
from scipy.spatial import KDTree


# ═══════════════════════════════════════════════════════════════════════════════
# §1  COLMAP binary reader
# ═══════════════════════════════════════════════════════════════════════════════

def _qvec_to_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z,  2*x*y-2*w*z,    2*x*z+2*w*y],
        [2*x*y+2*w*z,    1-2*x*x-2*z*z,  2*y*z-2*w*x],
        [2*x*z-2*w*y,    2*y*z+2*w*x,    1-2*x*x-2*y*y],
    ])


def read_images_bin(path: Path) -> Dict[str, np.ndarray]:
    """Return {image_name: camera_centre_xyz} from COLMAP images.bin."""
    centres: Dict[str, np.ndarray] = {}
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            struct.unpack('<I', f.read(4))                        # image_id
            qvec = np.array(struct.unpack('<4d', f.read(32)))
            tvec = np.array(struct.unpack('<3d', f.read(24)))
            struct.unpack('<I', f.read(4))                        # camera_id
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


# ═══════════════════════════════════════════════════════════════════════════════
# §2  PLY I/O
# ═══════════════════════════════════════════════════════════════════════════════

def _ply_props(header_lines: List[str]) -> Tuple[List[Tuple[str, str]], str, int]:
    """Parse PLY header → (props, format, vertex_count)."""
    fmt, vertex_count = 'ascii', 0
    props: List[Tuple[str, str]] = []
    in_v = False
    for line in header_lines:
        if line.startswith('format'):
            fmt = line.split()[1]
        elif line.startswith('element vertex'):
            vertex_count = int(line.split()[-1]); in_v = True
        elif line.startswith('element') and in_v:
            in_v = False
        elif in_v and line.startswith('property'):
            p = line.split(); props.append((p[1], p[2]))
    return props, fmt, vertex_count


def read_ply_xyz(path: Path) -> np.ndarray:
    """Read vertex positions from a PLY file → (N, 3) float64."""
    with open(path, 'rb') as f:
        raw = f.read()
    end = raw.index(b'end_header\n') + len(b'end_header\n')
    header_lines = raw[:end].decode().strip().split('\n')
    body = raw[end:]
    props, fmt, n = _ply_props(header_lines)
    xi = next(i for i, (_, nm) in enumerate(props) if nm == 'x')
    yi = next(i for i, (_, nm) in enumerate(props) if nm == 'y')
    zi = next(i for i, (_, nm) in enumerate(props) if nm == 'z')
    if fmt == 'ascii':
        rows = body.decode().strip().split('\n')
        pts = np.zeros((n, 3))
        for i in range(n):
            p = rows[i].split()
            pts[i] = [float(p[xi]), float(p[yi]), float(p[zi])]
        return pts
    _dmap = {'float': 'f4', 'double': 'f8', 'uchar': 'u1', 'int': 'i4',
              'uint': 'u4', 'short': 'i2', 'ushort': 'u2'}
    dt = np.dtype([(nm, '<' + _dmap.get(tp, 'f4')) for tp, nm in props])
    data = np.frombuffer(body[:n * dt.itemsize], dtype=dt)
    return np.column_stack([data['x'].astype(np.float64),
                             data['y'].astype(np.float64),
                             data['z'].astype(np.float64)])


def write_ply_xyz(path: Path, pts: np.ndarray) -> None:
    header = (
        'ply\nformat binary_little_endian 1.0\n'
        f'element vertex {len(pts)}\n'
        'property float x\nproperty float y\nproperty float z\nend_header\n'
    )
    with open(path, 'wb') as f:
        f.write(header.encode())
        f.write(pts.astype(np.float32).tobytes())


# ═══════════════════════════════════════════════════════════════════════════════
# §3  OBJ reader + uniform surface sampler
# ═══════════════════════════════════════════════════════════════════════════════

def load_obj_mesh(path: Path) -> Tuple[np.ndarray, np.ndarray]:
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


def sample_mesh_uniformly(verts: np.ndarray, faces: np.ndarray,
                           n: int, rng: np.random.Generator) -> np.ndarray:
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    total = areas.sum()
    if total == 0:
        raise ValueError('Mesh has zero area')
    fi = rng.choice(len(faces), size=n, p=areas / total)
    r1, r2 = rng.random(n), rng.random(n)
    sr1 = np.sqrt(r1)
    u, v, w = 1 - sr1, sr1 * (1 - r2), sr1 * r2
    return (u[:, None] * verts[faces[fi, 0]]
            + v[:, None] * verts[faces[fi, 1]]
            + w[:, None] * verts[faces[fi, 2]]).astype(np.float64)


def apply_T(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    ones = np.ones((len(pts), 1))
    return (T @ np.hstack([pts, ones]).T).T[:, :3]


# ═══════════════════════════════════════════════════════════════════════════════
# §4  Umeyama + cloud alignment
# ═══════════════════════════════════════════════════════════════════════════════

def umeyama(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
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


def _load_ned_positions(meta_dirs: List[Path]) -> Dict[str, np.ndarray]:
    positions: Dict[str, np.ndarray] = {}
    for d in meta_dirs:
        for jf in sorted(d.glob('*.json')):
            m = json.loads(jf.read_text())
            img = m.get('image_file')
            pos = m.get('vehicle_position_ned_m')
            if img and pos and 'x' in pos:
                positions[img] = np.array([pos['x'], pos['y'], pos['z']])
    return positions


def align_cloud(
    cloud_path: Path,
    images_bin: Path,
    meta_dirs: List[Path],
    bbox_center: np.ndarray,
    bbox_half: float,
) -> Tuple[np.ndarray, Dict]:
    """Align a COLMAP cloud to NED via Umeyama.  Returns (pts_ned, info)."""
    colmap_centres = read_images_bin(images_bin)
    ned_positions = _load_ned_positions([d for d in meta_dirs if d.exists()])

    src, dst = [], []
    for name, c in colmap_centres.items():
        if name in ned_positions:
            src.append(c)
            dst.append(ned_positions[name])

    if len(src) < 3:
        raise RuntimeError(
            f'Only {len(src)} matched cameras in {images_bin} — need ≥3')

    src_a, dst_a = np.array(src), np.array(dst)
    s, R, t = umeyama(src_a, dst_a)
    transformed = s * (R @ src_a.T).T + t
    residuals = np.linalg.norm(transformed - dst_a, axis=1)

    pts = read_ply_xyz(cloud_path)
    pts_ned = s * (R @ pts.T).T + t

    mask = np.all(np.abs(pts_ned - bbox_center) <= bbox_half, axis=1)
    pts_ned = pts_ned[mask]

    return pts_ned, {
        'matched': len(src),
        'scale': round(s, 6),
        'rms_residual_m': round(float(np.sqrt(np.mean(residuals ** 2))), 6),
        'max_residual_m': round(float(residuals.max()), 6),
        'bbox_kept': int(mask.sum()),
        'bbox_total': len(mask),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# §5  Cloud cleaning
# ═══════════════════════════════════════════════════════════════════════════════

def _crop_bbox(pts: np.ndarray, center: np.ndarray, half: float) -> np.ndarray:
    return pts[np.all(np.abs(pts - center) <= half, axis=1)]


def _remove_ground(pts: np.ndarray, thresh: float, n_iter: int,
                    low_frac: float, rng: np.random.Generator) -> np.ndarray:
    if len(pts) < 10:
        return pts
    z_thresh = np.percentile(pts[:, 2], (1 - low_frac) * 100)
    low = pts[pts[:, 2] >= z_thresh]
    if len(low) < 3:
        return pts
    best_count, best_plane = 0, None
    for _ in range(n_iter):
        idx = rng.choice(len(low), 3, replace=False)
        p0, p1, p2 = low[idx[0]], low[idx[1]], low[idx[2]]
        nv = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(nv)
        if nn < 1e-10:
            continue
        nv /= nn
        d = -nv.dot(p0)
        c = int((np.abs(pts @ nv + d) <= thresh).sum())
        if c > best_count:
            best_count = c
            best_plane = np.append(nv, d)
    if best_plane is None:
        return pts
    return pts[np.abs(pts @ best_plane[:3] + best_plane[3]) > thresh]


def _sor(pts: np.ndarray, k: int, std_ratio: float) -> np.ndarray:
    if len(pts) <= k:
        return pts
    dists, _ = KDTree(pts).query(pts, k=k + 1)
    md = dists[:, 1:].mean(1)
    return pts[md <= md.mean() + std_ratio * md.std()]


def _dist_gate(pts: np.ndarray, gt: np.ndarray, gate: float) -> np.ndarray:
    if len(pts) == 0:
        return pts
    d, _ = KDTree(gt).query(pts, k=1)
    return pts[d <= gate]


def clean_cloud(pts: np.ndarray, gt_samples: np.ndarray, rock: np.ndarray,
                crop_half: float, ground_thresh: float, n_ransac: int,
                sor_k: int, sor_std: float, gate: float,
                rng: np.random.Generator) -> Tuple[np.ndarray, Dict[str, int]]:
    steps: Dict[str, int] = {'raw': len(pts)}
    pts = _crop_bbox(pts, rock, crop_half)
    steps['after_bbox'] = len(pts)
    pts = _remove_ground(pts, ground_thresh, n_ransac, 0.20, rng)
    steps['after_ground_removal'] = len(pts)
    pts = _sor(pts, sor_k, sor_std)
    steps['after_sor'] = len(pts)
    pts = _dist_gate(pts, gt_samples, gate)
    steps['after_distance_gate'] = len(pts)
    return pts, steps


# ═══════════════════════════════════════════════════════════════════════════════
# §6  Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(gt_samples: np.ndarray, recon: np.ndarray,
                    thresholds: List[float]) -> Dict:
    if len(recon) == 0:
        empty = {f'{m}_{t*1000:.0f}mm': 0.0
                 for m in ('completeness', 'accuracy', 'fscore')
                 for t in thresholds}
        empty.update(mean_c2c_m=float('inf'), median_c2c_m=float('inf'),
                     p95_c2c_m=float('inf'))
        return empty
    gt2r, _ = KDTree(recon).query(gt_samples, k=1)
    r2gt, _ = KDTree(gt_samples).query(recon, k=1)
    out: Dict = {}
    for t in thresholds:
        key = f'{t * 1000:.0f}mm'
        c = float(np.mean(gt2r <= t))
        a = float(np.mean(r2gt <= t))
        f = 2 * c * a / (c + a) if c + a > 0 else 0.0
        out[f'completeness_{key}'] = round(c * 100, 3)
        out[f'accuracy_{key}'] = round(a * 100, 3)
        out[f'fscore_{key}'] = round(f * 100, 3)
    out['mean_c2c_m'] = float(gt2r.mean())
    out['median_c2c_m'] = float(np.median(gt2r))
    out['p95_c2c_m'] = float(np.percentile(gt2r, 95))
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# §7  Run data loaders
# ═══════════════════════════════════════════════════════════════════════════════

def load_mission_params(run_dir: Path) -> Dict:
    params: Dict = {}
    p2 = run_dir / 'phase2_manifest.json'
    if p2.exists():
        params.update(json.loads(p2.read_text()))
    sm = run_dir / 'seed' / 'manifest.json'
    if sm.exists():
        params['seed_manifest'] = json.loads(sm.read_text())
    return params


def load_seed_trajectory(run_dir: Path) -> List[Dict]:
    """Seed captures sorted by capture_index, with actual NED position."""
    meta_dir = run_dir / 'seed' / 'metadata'
    if not meta_dir.exists():
        return []
    captures = []
    for jf in sorted(meta_dir.glob('*.json')):
        m = json.loads(jf.read_text())
        pos = m.get('vehicle_position_ned_m')
        if pos and 'x' in pos:
            captures.append({
                'index': m.get('capture_index', 0),
                'label': m.get('view_label', jf.stem),
                'ned': np.array([pos['x'], pos['y'], pos['z']]),
            })
    captures.sort(key=lambda c: c['index'])
    return captures


def load_nbv_positions(run_dir: Path) -> List[Dict]:
    """NBV captures sorted by iteration, with actual NED position."""
    meta_dir = run_dir / 'adaptive' / 'metadata'
    if not meta_dir.exists():
        return []
    nbv = []
    for jf in sorted(meta_dir.glob('*.json')):
        m = json.loads(jf.read_text())
        pos = m.get('vehicle_position_ned_m')
        if pos and 'x' in pos:
            nbv.append({
                'iteration': m.get('iteration', 0),
                'candidate_id': m.get('candidate_id', '?'),
                'ned': np.array([pos['x'], pos['y'], pos['z']]),
            })
    nbv.sort(key=lambda c: c['iteration'])
    return nbv


def load_candidates(run_dir: Path) -> Optional[np.ndarray]:
    """(N, 3) NED positions from candidate_pool_iter_00.json."""
    f = run_dir / 'candidates' / 'candidate_pool_iter_00.json'
    if not f.exists():
        return None
    pool = json.loads(f.read_text())
    return np.array([[c['x'], c['y'], c['z']] for c in pool])


def load_sparse_metrics(run_dir: Path) -> List[Dict]:
    """One dict per iteration file in sparse_metrics/."""
    d = run_dir / 'sparse_metrics'
    if not d.exists():
        return []
    results = []
    for jf in sorted(d.glob('metrics_iter_*.json')):
        m = json.loads(jf.read_text())
        knn = m.get('knn_distance_metrics') or {}
        results.append({
            'iter': int(jf.stem.rsplit('_', 1)[-1]),
            'cameras': m.get('total_cameras', 0),
            'points': m.get('sparse_point_count', 0),
            'knn_p95': knn.get('percentile_knn_distance'),
            'knn_mean': knn.get('mean_knn_distance'),
        })
    results.sort(key=lambda r: r['iter'])
    return results


def load_score_evolution(run_dir: Path) -> List[Dict]:
    """Top-ranked candidate score per NBV iteration."""
    d = run_dir / 'candidates'
    if not d.exists():
        return []
    scores = []
    for jf in sorted(d.glob('candidate_scores_iter_*.json')):
        data = json.loads(jf.read_text())
        if data:
            top = data[0]
            scores.append({
                'iter': int(jf.stem.rsplit('_', 1)[-1]),
                'final_score': top.get('final_score', 0.0),
                'terms': top.get('terms', {}),
                'weights': top.get('weights', {}),
            })
    scores.sort(key=lambda s: s['iter'])
    return scores


def _find_seed_images_bin(run_dir: Path) -> Optional[Path]:
    for sub in ('0', '1', '2'):
        p = run_dir / 'seed_colmap' / 'sparse' / sub / 'images.bin'
        if p.exists():
            return p
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# §8  Display coordinate helpers  (NED → ENU for intuitive 3-D plotting)
# ═══════════════════════════════════════════════════════════════════════════════

def _to_enu(pts: np.ndarray) -> np.ndarray:
    """NED → ENU: East = y_ned, North = x_ned, Up = −z_ned."""
    return np.column_stack([pts[:, 1], pts[:, 0], -pts[:, 2]])


def _pt_enu(p: np.ndarray) -> np.ndarray:
    return np.array([p[1], p[0], -p[2]])


# ═══════════════════════════════════════════════════════════════════════════════
# §9  Figure 1 — quality metrics
# ═══════════════════════════════════════════════════════════════════════════════

_C_SEED = '#546E7A'
_C_NBV = '#00ACC1'
_PALETTE = ['#1E88E5', '#43A047', '#FB8C00', '#8E24AA', '#E53935']


def _bar_labels(ax, bars, fmt='{:.1f}'):
    for b in bars:
        h = b.get_height()
        if h > 1.5:
            ax.text(b.get_x() + b.get_width() / 2, h + 0.8,
                    fmt.format(h), ha='center', va='bottom', fontsize=6.5)


def plot_metrics_figure(report: Dict, sparse_metrics: List[Dict],
                         score_evolution: List[Dict],
                         output_dir: Path) -> plt.Figure:
    thresholds = report['thresholds_m']
    t_labels = [f'{t * 1000:.0f}mm' for t in thresholds]
    cloud_labels = [k for k in report['clouds'] if 'metrics' in report['clouds'][k]]
    bar_colors = [_C_SEED, _C_NBV]
    bw = 0.35
    x = np.arange(len(t_labels))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Reconstruction Quality Evaluation', fontsize=14,
                 fontweight='bold', y=1.02)

    ax_f, ax_c, ax_a = axes

    # ── F-score ─────────────────────────────────────────────────────────────
    for i, lbl in enumerate(cloud_labels):
        vals = [report['clouds'][lbl]['metrics'][f'fscore_{tl}'] for tl in t_labels]
        offset = (i - len(cloud_labels) / 2 + 0.5) * bw
        bars = ax_f.bar(x + offset, vals, bw, label=lbl,
                        color=bar_colors[i % len(bar_colors)], zorder=3)
        _bar_labels(ax_f, bars)
    ax_f.set_xticks(x); ax_f.set_xticklabels(t_labels)
    ax_f.set_ylabel('F-score (%)'); ax_f.set_title('F-score at distance thresholds')
    ax_f.set_ylim(0, 108); ax_f.legend(fontsize=9)
    ax_f.grid(axis='y', alpha=0.35, zorder=0)

    # ── Completeness ─────────────────────────────────────────────────────────
    for i, lbl in enumerate(cloud_labels):
        vals = [report['clouds'][lbl]['metrics'][f'completeness_{tl}'] for tl in t_labels]
        ax_c.plot(t_labels, vals, marker='o', label=lbl,
                  color=bar_colors[i % len(bar_colors)], linewidth=2)
    ax_c.set_ylabel('Completeness (%)'); ax_c.set_title('Completeness')
    ax_c.set_ylim(0, 108); ax_c.legend(fontsize=8); ax_c.grid(alpha=0.3)

    # ── Accuracy ─────────────────────────────────────────────────────────────
    for i, lbl in enumerate(cloud_labels):
        vals = [report['clouds'][lbl]['metrics'][f'accuracy_{tl}'] for tl in t_labels]
        ax_a.plot(t_labels, vals, marker='s', label=lbl,
                  color=bar_colors[i % len(bar_colors)], linewidth=2)
    ax_a.set_ylabel('Accuracy (%)'); ax_a.set_title('Accuracy')
    ax_a.set_ylim(0, 108); ax_a.legend(fontsize=8); ax_a.grid(alpha=0.3)

    # ── C2C distances (mid-right) ───────────────────────────────────────────
    # ax_d = fig.add_subplot(gs[1, 2])
    # c2c_keys = ['mean_c2c_m', 'median_c2c_m', 'p95_c2c_m']
    # c2c_names = ['Mean', 'Median', 'P95']
    # xd = np.arange(3)
    # for i, lbl in enumerate(cloud_labels):
    #     vals = [report['clouds'][lbl]['metrics'][k] * 1000 for k in c2c_keys]
    #     offset = (i - len(cloud_labels) / 2 + 0.5) * bw
    #     bars = ax_d.bar(xd + offset, vals, bw, label=lbl,
    #                     color=bar_colors[i % len(bar_colors)])
    #     _bar_labels(ax_d, bars, fmt='{:.0f}')
    # ax_d.set_xticks(xd); ax_d.set_xticklabels(c2c_names)
    # ax_d.set_ylabel('Distance (mm)'); ax_d.set_title('Cloud-to-cloud distance (GT→recon)')
    # ax_d.legend(fontsize=8); ax_d.grid(axis='y', alpha=0.3)

    # ── Cleaning funnel (top-right) ─────────────────────────────────────────
    # ax_cl = fig.add_subplot(gs[0, 2])
    # funnel_lbl = cloud_labels[-1] if cloud_labels else None
    # if funnel_lbl:
    #     step_keys = ['raw', 'after_bbox', 'after_ground_removal', 'after_sor', 'after_distance_gate']
    #     step_names = ['Raw', 'Bbox crop', 'Ground\nremoval', 'SOR', 'Distance\ngate']
    #     sc = report['clouds'][funnel_lbl].get('step_counts', {})
    #     vals = [sc.get(k, 0) for k in step_keys]
    #     colors_cl = plt.cm.Blues(np.linspace(0.4, 0.85, len(step_names)))
    #     bars = ax_cl.barh(range(len(step_names)), vals, color=colors_cl)
    #     ax_cl.set_yticks(range(len(step_names)))
    #     ax_cl.set_yticklabels(step_names, fontsize=8)
    #     ax_cl.set_xlabel('Point count')
    #     ax_cl.set_title(f'Cleaning funnel  ({funnel_lbl})')
    #     max_v = max(vals) if vals else 1
    #     for j, v in enumerate(vals):
    #         if v > 0:
    #             ax_cl.text(v + max_v * 0.015, j, f'{v:,}',
    #                        va='center', fontsize=7.5)
    #     ax_cl.grid(axis='x', alpha=0.3)

    # ── Sparse point count + KNN evolution (bottom, 2 cols) ────────────────
    # ax_sp = fig.add_subplot(gs[2, :2])
    # if sparse_metrics:
    #     iters = [m['iter'] for m in sparse_metrics]
    #     pts_c = [m['points'] for m in sparse_metrics]
    #     ax_sp.plot(iters, pts_c, marker='o', color='#1E88E5', linewidth=2,
    #                label='Sparse points')
    #     ax_sp.fill_between(iters, pts_c, alpha=0.12, color='#1E88E5')
    #     for it, pc in zip(iters, pts_c):
    #         ax_sp.text(it, pc + max(pts_c) * 0.02, f'{pc:,}',
    #                    ha='center', fontsize=7, color='#1565C0')
    #     ax_sp.set_xlabel('Iteration  (0 = seed only)')
    #     ax_sp.set_ylabel('Sparse point count', color='#1565C0')
    #     ax_sp.tick_params(axis='y', labelcolor='#1565C0')
    #     ax_sp.set_title('Sparse reconstruction evolution')
    #     ax_sp.set_xticks(iters)
    #     ax_sp.grid(alpha=0.3)
    #
    #     knn_iters = [m['iter'] for m in sparse_metrics if m['knn_p95'] is not None]
    #     knn_vals = [m['knn_p95'] for m in sparse_metrics if m['knn_p95'] is not None]
    #     if knn_iters:
    #         ax2 = ax_sp.twinx()
    #         ax2.plot(knn_iters, knn_vals, marker='s', color='#E53935', linewidth=2,
    #                  linestyle='--', label='KNN P95')
    #         ax2.set_ylabel('KNN P95 dist (COLMAP units)', color='#E53935')
    #         ax2.tick_params(axis='y', labelcolor='#E53935')
    #         h1, l1 = ax_sp.get_legend_handles_labels()
    #         h2, l2 = ax2.get_legend_handles_labels()
    #         ax_sp.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper left')

    # ── Score breakdown (bottom-right) ─────────────────────────────────────
    # ax_sc = fig.add_subplot(gs[2, 2])
    # if score_evolution:
    #     sc_x = [s['iter'] + 1 for s in score_evolution]
    #     finals = [s['final_score'] for s in score_evolution]
    #     pos_terms = [('covisibility', '#1E88E5'), ('novelty', '#43A047')]
    #     neg_terms = [('movement_cost', '#FB8C00'), ('angular_separation_penalty', '#E53935')]
    #     bottom_pos = np.zeros(len(score_evolution))
    #     bottom_neg = np.zeros(len(score_evolution))
    #     for term, color in pos_terms:
    #         vals_w = np.array([
    #             s['terms'].get(term, 0) * s['weights'].get(term, 1.0)
    #             for s in score_evolution
    #         ])
    #         ax_sc.bar(sc_x, vals_w, bottom=bottom_pos, color=color, alpha=0.75,
    #                   label=f'+{term}', width=0.4, zorder=3)
    #         bottom_pos += vals_w
    #     for term, color in neg_terms:
    #         vals_w = np.array([
    #             -(s['terms'].get(term, 0) * s['weights'].get(term, 0.0))
    #             for s in score_evolution
    #         ])
    #         ax_sc.bar(sc_x, vals_w, bottom=bottom_neg, color=color, alpha=0.75,
    #                   label=f'−{term}', width=0.4, zorder=3)
    #         bottom_neg += vals_w
    #     ax_sc.plot(sc_x, finals, marker='D', color='black', linewidth=1.5,
    #                zorder=5, label='Final score', markersize=6)
    #     ax_sc.axhline(0, color='black', linewidth=0.5)
    #     ax_sc.set_xlabel('NBV iteration')
    #     ax_sc.set_ylabel('Score')
    #     ax_sc.set_title('Selected candidate score breakdown')
    #     ax_sc.set_xticks(sc_x)
    #     ax_sc.legend(fontsize=6.5, loc='upper right', ncol=1)
    #     ax_sc.grid(axis='y', alpha=0.3, zorder=0)

    # ── Mission params footer ───────────────────────────────────────────────
    mp = report.get('mission_params', {})
    if mp:
        sm_info = mp.get('seed_manifest', {})
        parts = [
            f"scorer: {mp.get('scorer_name', '?')}",
            f"stop: {mp.get('stopping_criterion', '?')}",
            f"knn_thresh: {mp.get('knn_distance_threshold', '?')}",
            f"budget: {mp.get('image_budget', '?')} / {mp.get('max_image_budget', '?')}",
            f"seed: {mp.get('seed_count', sm_info.get('ring_image_count', '?'))} imgs",
            f"radius: {sm_info.get('capture_radius_m', '?')} m",
        ]
        fig.text(0.5, -0.04, '   |   '.join(parts), ha='center', va='bottom',
                 fontsize=8, color='#444', style='italic',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#F5F5F5',
                           edgecolor='#ccc', alpha=0.9))

    fig.tight_layout()
    out = output_dir / 'metrics.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'  Saved → {out}')
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# §10  Figure 2 — 3-D trajectory / candidate hemisphere
# ═══════════════════════════════════════════════════════════════════════════════

_C_SEED_3D = '#1565C0'
_C_NBV_3D = '#E65100'
_C_CAND = '#BDBDBD'
_C_CLOUD = '#64B5F6'
_C_ROCK = 'gold'


def plot_trajectory_figure(
    seed_captures: List[Dict],
    nbv_captures: List[Dict],
    candidates: Optional[np.ndarray],
    rock_center: np.ndarray,
    cleaned_cloud: Optional[np.ndarray],
    output_dir: Path,
) -> plt.Figure:
    fig = plt.figure(figsize=(14, 11))
    fig.suptitle('Drone Trajectory & Candidate Viewpoints  (ENU frame)',
                 fontsize=13, fontweight='bold')
    ax = fig.add_subplot(111, projection='3d')

    rc_e = _pt_enu(rock_center)

    # ── Cleaned reconstruction cloud (subsampled) ───────────────────────────
    if cleaned_cloud is not None and len(cleaned_cloud) > 0:
        sub = cleaned_cloud
        if len(sub) > 5000:
            sub = sub[np.random.default_rng(42).choice(len(sub), 5000, replace=False)]
        enu = _to_enu(sub)
        ax.scatter(enu[:, 0], enu[:, 1], enu[:, 2],
                   s=1, c=_C_CLOUD, alpha=0.20, label='Recon cloud (NBV)')

    # ── Candidate hemisphere (iter-00 pool) ─────────────────────────────────
    if candidates is not None and len(candidates) > 0:
        c_enu = _to_enu(candidates)
        ax.scatter(c_enu[:, 0], c_enu[:, 1], c_enu[:, 2],
                   s=14, c=_C_CAND, alpha=0.55, label=f'Candidates ({len(candidates)})',
                   depthshade=False)

    # ── Seed captures + spline trajectory ───────────────────────────────────
    if seed_captures:
        seed_pts = np.array([c['ned'] for c in seed_captures])
        seed_enu = _to_enu(seed_pts)

        # Capture positions
        ax.scatter(seed_enu[:, 0], seed_enu[:, 1], seed_enu[:, 2],
                   s=55, c=_C_SEED_3D, marker='o', zorder=5,
                   label=f'Seed captures ({len(seed_captures)})',
                   edgecolors='white', linewidth=0.6)
        # Label first and last
        for i_lbl, (pt, tag) in enumerate([(seed_enu[0], 'S₀'),
                                            (seed_enu[-1], f'S{len(seed_captures)-1}')]):
            ax.text(pt[0] + 0.12, pt[1], pt[2] + 0.18, tag,
                    fontsize=8, color=_C_SEED_3D, fontweight='bold')

        # Spline through seed points (in ENU for smooth geodesic-like curve)
        if len(seed_pts) >= 4:
            # Remove any consecutive duplicates before splprep
            keep = [0] + [i for i in range(1, len(seed_enu))
                          if np.linalg.norm(seed_enu[i] - seed_enu[i-1]) > 1e-4]
            sp = seed_enu[keep]
            if len(sp) >= 4:
                try:
                    tck, _ = splprep([sp[:, 0], sp[:, 1], sp[:, 2]], s=0.0, k=3)
                    fine = np.array(splev(np.linspace(0, 1, 400), tck)).T
                    ax.plot(fine[:, 0], fine[:, 1], fine[:, 2],
                            color=_C_SEED_3D, linewidth=1.8, alpha=0.65,
                            label='Seed trajectory (spline)')
                except Exception:
                    # Fallback: straight segments
                    ax.plot(seed_enu[:, 0], seed_enu[:, 1], seed_enu[:, 2],
                            color=_C_SEED_3D, linewidth=1.5, alpha=0.65,
                            label='Seed trajectory')

    # ── NBV visited positions ────────────────────────────────────────────────
    if nbv_captures:
        nbv_pts = np.array([c['ned'] for c in nbv_captures])
        nbv_enu = _to_enu(nbv_pts)
        ax.scatter(nbv_enu[:, 0], nbv_enu[:, 1], nbv_enu[:, 2],
                   s=140, c=_C_NBV_3D, marker='^', zorder=6,
                   label=f'NBV captures ({len(nbv_captures)})',
                   edgecolors='white', linewidth=0.7)
        for i, pt in enumerate(nbv_enu):
            ax.text(pt[0] + 0.12, pt[1], pt[2] + 0.18, f'N{i + 1}',
                    fontsize=8.5, color=_C_NBV_3D, fontweight='bold')

    # ── Rock centre (gold star) ──────────────────────────────────────────────
    ax.scatter([rc_e[0]], [rc_e[1]], [rc_e[2]],
               s=450, c=_C_ROCK, marker='*', zorder=10,
               edgecolors='#B8860B', linewidth=1.2, label='Rock centre')

    # ── Axes + view ─────────────────────────────────────────────────────────
    ax.set_xlabel('East (m)', labelpad=9)
    ax.set_ylabel('North (m)', labelpad=9)
    ax.set_zlabel('Altitude (m)', labelpad=9)
    ax.legend(loc='upper left', fontsize=8.5, framealpha=0.88, markerscale=1.2)
    ax.view_init(elev=28, azim=-50)

    out = output_dir / 'trajectory_3d.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'  Saved → {out}')
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# §11  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description='Evaluate + plot a photogrammetry-covisibility NBV run.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument('--run-dir', required=True,
                    help='unified_run_* directory')
    ap.add_argument('--gt-mesh', required=True,
                    help='Ground-truth mesh (.obj)')
    ap.add_argument('--gt-transform', default=None,
                    help='4×4 .npy NED transform for GT mesh. '
                         'If omitted, defaults to lunar_sample_15016 at ENU (8,0,0.8) scale 20.')
    ap.add_argument('--rock-center', nargs=3, type=float, default=[0.0, 8.0, -0.8],
                    metavar=('X', 'Y', 'Z'), help='Rock centre in NED (m)')
    ap.add_argument('--align-bbox-half', type=float, default=3.0,
                    help='Bbox half-extent applied after Umeyama alignment (m)')
    ap.add_argument('--crop-half-extent', type=float, default=1.5,
                    help='Bbox half-extent for cleaning step (m)')
    ap.add_argument('--ground-thresh-m', type=float, default=0.03,
                    help='RANSAC inlier threshold for ground removal (m)')
    ap.add_argument('--ransac-iters', type=int, default=500)
    ap.add_argument('--sor-k', type=int, default=20)
    ap.add_argument('--sor-std-ratio', type=float, default=2.0)
    ap.add_argument('--gate-dist-m', type=float, default=0.05,
                    help='Distance gate vs GT surface (m)')
    ap.add_argument('--thresholds', nargs='+', type=float,
                    default=[0.005, 0.01, 0.02, 0.05],
                    help='Metric thresholds (m)')
    ap.add_argument('--gt-samples', type=int, default=100_000,
                    help='GT surface samples')
    ap.add_argument('--output-dir', default=None,
                    help='Output directory (default: <run-dir>/eval)')
    ap.add_argument('--no-show', action='store_true',
                    help='Save figures without opening a window')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        print(f'ERROR: --run-dir not found: {run_dir}')
        sys.exit(1)

    output_dir = (Path(args.output_dir).expanduser()
                  if args.output_dir else run_dir / 'eval')
    output_dir.mkdir(parents=True, exist_ok=True)

    rock_center = np.array(args.rock_center)
    rng = np.random.default_rng(args.seed)

    # ── GT transform ─────────────────────────────────────────────────────────
    if args.gt_transform:
        T_gt = np.load(args.gt_transform)
    else:
        print('  No --gt-transform given — using default for lunar_sample_15016 '
              'at Gazebo ENU (8, 0, 0.8), mesh scale 20.')
        T_gt = np.array([
            [0,  20,  0,   0.0],
            [20,  0,  0,   8.0],
            [0,   0, -20, -0.8],
            [0,   0,  0,   1.0],
        ], dtype=float)

    # ── GT mesh ───────────────────────────────────────────────────────────────
    print('Loading GT mesh...')
    gt_verts, gt_faces = load_obj_mesh(Path(args.gt_mesh).expanduser())
    gt_verts = apply_T(gt_verts, T_gt)
    print(f'  {len(gt_verts):,} vertices, {len(gt_faces):,} faces')
    print(f'  Sampling {args.gt_samples:,} surface points...')
    gt_samples = sample_mesh_uniformly(gt_verts, gt_faces, args.gt_samples, rng)
    aabb = (gt_samples.min(0), gt_samples.max(0))
    print(f'  GT AABB: x=[{aabb[0][0]:.3f},{aabb[1][0]:.3f}]  '
          f'y=[{aabb[0][1]:.3f},{aabb[1][1]:.3f}]  '
          f'z=[{aabb[0][2]:.3f},{aabb[1][2]:.3f}]')

    # ── Run data ─────────────────────────────────────────────────────────────
    print('Loading run data...')
    mission_params = load_mission_params(run_dir)
    seed_captures = load_seed_trajectory(run_dir)
    nbv_captures = load_nbv_positions(run_dir)
    candidates = load_candidates(run_dir)
    sparse_metrics = load_sparse_metrics(run_dir)
    score_evolution = load_score_evolution(run_dir)

    print(f'  Seed captures : {len(seed_captures)}')
    print(f'  NBV captures  : {len(nbv_captures)}')
    print(f'  Candidates    : {len(candidates) if candidates is not None else 0}')
    print(f'  Sparse iters  : {len(sparse_metrics)}')
    print(f'  Score entries : {len(score_evolution)}')
    if mission_params:
        print(f'  Stop criterion: {mission_params.get("stopping_criterion", "?")}  '
              f'knn_thresh={mission_params.get("knn_distance_threshold", "?")}  '
              f'budget={mission_params.get("image_budget", "?")}'
              f'/{mission_params.get("max_image_budget", "?")}')

    # ── Build cloud pipeline specs ────────────────────────────────────────────
    cloud_specs = [
        (
            'seed',
            run_dir / 'final' / 'seed_sparse_cloud.ply',
            _find_seed_images_bin(run_dir),
            [run_dir / 'seed' / 'metadata'],
        ),
        (
            'nbv',
            run_dir / 'final' / 'dense_cloud.ply',
            run_dir / 'colmap' / 'sparse' / '0' / 'images.bin',
            [run_dir / 'seed' / 'metadata', run_dir / 'adaptive' / 'metadata'],
        ),
    ]

    report: Dict = {
        'run_dir': str(run_dir),
        'gt_mesh': str(args.gt_mesh),
        'gt_samples': args.gt_samples,
        'rock_center_ned': list(args.rock_center),
        'thresholds_m': args.thresholds,
        'mission_params': mission_params,
        'cleaning': {
            'crop_half_extent_m': args.crop_half_extent,
            'ground_thresh_m': args.ground_thresh_m,
            'sor_k': args.sor_k,
            'sor_std_ratio': args.sor_std_ratio,
            'gate_dist_m': args.gate_dist_m,
        },
        'clouds': {},
    }

    cleaned_nbv: Optional[np.ndarray] = None

    for label, cloud_path, images_bin, meta_dirs in cloud_specs:
        print(f'\n=== {label}: {cloud_path.name} ===')

        if not cloud_path.exists():
            print(f'  SKIP: {cloud_path} not found')
            report['clouds'][label] = {'error': 'cloud file not found'}
            continue
        if images_bin is None or not images_bin.exists():
            print(f'  SKIP: images.bin not found for {label}')
            report['clouds'][label] = {'error': 'images.bin not found'}
            continue

        try:
            pts_ned, align_info = align_cloud(
                cloud_path, images_bin, meta_dirs, rock_center, args.align_bbox_half)
            print(f'  Aligned: {align_info["matched"]} cameras  '
                  f'scale={align_info["scale"]:.4f}  '
                  f'RMS={align_info["rms_residual_m"]*1000:.1f} mm  '
                  f'({align_info["bbox_kept"]:,}/{align_info["bbox_total"]:,} pts after bbox)')
        except RuntimeError as e:
            print(f'  ERROR: {e}')
            report['clouds'][label] = {'error': str(e)}
            continue

        pts, steps = clean_cloud(
            pts_ned, gt_samples, rock_center,
            args.crop_half_extent, args.ground_thresh_m, args.ransac_iters,
            args.sor_k, args.sor_std_ratio, args.gate_dist_m, rng)
        print(f'  Cleaning steps: {steps}')

        if len(pts) == 0:
            print('  WARNING: no points after cleaning — check rock_center / transforms')
            report['clouds'][label] = {
                'alignment': align_info, 'step_counts': steps,
                'error': 'no points after cleaning'}
            continue

        cleaned_path = output_dir / f'cleaned_{label}.ply'
        write_ply_xyz(cleaned_path, pts)
        print(f'  Saved cleaned cloud → {cleaned_path}')

        if label == 'nbv':
            cleaned_nbv = pts

        print(f'  Computing metrics against GT ({args.gt_samples:,} samples)...')
        metrics = compute_metrics(gt_samples, pts, args.thresholds)
        t_labels = [f'{t * 1000:.0f}mm' for t in args.thresholds]
        print(f'  {"Threshold":<10} {"Completeness":>13} {"Accuracy":>11} {"F-score":>9}')
        for tl in t_labels:
            print(f'  {tl:<10} {metrics[f"completeness_{tl}"]:>12.1f}%'
                  f' {metrics[f"accuracy_{tl}"]:>10.1f}%'
                  f' {metrics[f"fscore_{tl}"]:>8.1f}%')
        print(f'  Mean C2C: {metrics["mean_c2c_m"]*1000:.1f} mm  |  '
              f'Median: {metrics["median_c2c_m"]*1000:.1f} mm  |  '
              f'P95: {metrics["p95_c2c_m"]*1000:.1f} mm')

        report['clouds'][label] = {
            'cloud_path': str(cloud_path),
            'cleaned_ply': str(cleaned_path),
            'alignment': align_info,
            'step_counts': steps,
            'metrics': metrics,
        }

    # ── Cross-cloud comparison ─────────────────────────────────────────────
    valid = [k for k in report['clouds'] if 'metrics' in report['clouds'][k]]
    if len(valid) == 2:
        a, b = valid[0], valid[1]
        am, bm = report['clouds'][a]['metrics'], report['clouds'][b]['metrics']
        print(f'\n=== Comparison: {b} vs {a} ===')
        cmp: Dict = {}
        t_labels = [f'{t * 1000:.0f}mm' for t in args.thresholds]
        for tl in t_labels:
            for mk in ('completeness', 'accuracy', 'fscore'):
                k = f'{mk}_{tl}'
                delta = bm[k] - am[k]
                cmp[f'd_{k}'] = round(delta, 3)
                sign = '+' if delta >= 0 else ''
                print(f'  {mk}@{tl}: {am[k]:.1f}% → {bm[k]:.1f}%  ({sign}{delta:.1f}pp)')
        dc2c = bm['mean_c2c_m'] - am['mean_c2c_m']
        cmp['d_mean_c2c_m'] = round(dc2c, 6)
        sign = '+' if dc2c >= 0 else ''
        print(f'  Mean C2C: {am["mean_c2c_m"]*1000:.1f} mm → '
              f'{bm["mean_c2c_m"]*1000:.1f} mm  ({sign}{dc2c*1000:.1f} mm)')
        report['comparison'] = {'baseline': a, 'improved': b, 'deltas': cmp}

    # ── JSON report ───────────────────────────────────────────────────────────
    report_path = output_dir / 'report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\nReport → {report_path}')

    # ── Figures ───────────────────────────────────────────────────────────────
    print('\nGenerating figures...')
    plot_metrics_figure(report, sparse_metrics, score_evolution, output_dir)
    plot_trajectory_figure(seed_captures, nbv_captures, candidates,
                            rock_center, cleaned_nbv, output_dir)

    if not args.no_show:
        plt.show()
    else:
        plt.close('all')

    print(f'\nDone. All outputs in: {output_dir}')


if __name__ == '__main__':
    main()
