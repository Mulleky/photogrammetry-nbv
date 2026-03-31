#!/usr/bin/env python3
"""
Umeyama similarity-transform alignment of a COLMAP point cloud to NED frame.

Reads camera positions from COLMAP images.bin, matches them to NED positions
from metadata JSONs, computes a 7-DOF transform (scale + rotation + translation),
and applies it to the input point cloud (PLY). Optionally crops to a bounding box.

Usage:
    python align_cloud.py \
        --cloud dense_cloud.ply \
        --images-bin colmap/sparse/0/images.bin \
        --metadata seed/metadata adaptive/metadata \
        --output aligned_cloud.ply \
        [--bbox-center 0 0 0] \
        [--bbox-half-extent 6.0]
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# COLMAP images.bin reader
# ---------------------------------------------------------------------------

def read_images_bin(path: Path) -> Dict[str, np.ndarray]:
    """Return {image_name: camera_centre_xyz} from images.bin."""
    centres = {}
    with open(path, 'rb') as f:
        num_images = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num_images):
            _image_id = struct.unpack('<I', f.read(4))[0]
            qvec = np.array(struct.unpack('<4d', f.read(32)))
            tvec = np.array(struct.unpack('<3d', f.read(24)))
            _camera_id = struct.unpack('<I', f.read(4))[0]
            name = b''
            while True:
                ch = f.read(1)
                if ch == b'\x00':
                    break
                name += ch
            num_points2d = struct.unpack('<Q', f.read(8))[0]
            f.read(num_points2d * 24)

            # Camera centre = -R^T @ t
            R = _qvec_to_rotmat(qvec)
            centre = -R.T @ tvec
            centres[name.decode('utf-8')] = centre
    return centres


def _qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y,     2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y],
    ])


# ---------------------------------------------------------------------------
# NED metadata loader
# ---------------------------------------------------------------------------

def load_ned_positions(metadata_dirs: List[Path]) -> Dict[str, np.ndarray]:
    """Return {image_filename: ned_xyz} from metadata JSON files."""
    positions = {}
    for d in metadata_dirs:
        for jf in sorted(d.glob('*.json')):
            with open(jf) as f:
                meta = json.load(f)
            img_file = meta.get('image_file')
            pos = meta.get('vehicle_position_ned_m')
            if img_file and pos and 'x' in pos:
                positions[img_file] = np.array([pos['x'], pos['y'], pos['z']])
    return positions


# ---------------------------------------------------------------------------
# Umeyama similarity transform
# ---------------------------------------------------------------------------

def umeyama(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute 7-DOF similarity transform: dst ≈ s * R @ src + t

    Args:
        src: (N, 3) source points (COLMAP frame)
        dst: (N, 3) target points (NED frame)

    Returns:
        s: scale factor
        R: (3, 3) rotation matrix
        t: (3,) translation vector
    """
    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    src_c = src - mu_src
    dst_c = dst - mu_dst

    var_src = np.sum(src_c ** 2) / n

    cov = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / var_src
    t = mu_dst - s * R @ mu_src

    return float(s), R, t


# ---------------------------------------------------------------------------
# PLY I/O (simple ASCII/binary_little_endian support)
# ---------------------------------------------------------------------------

def read_ply(path: Path) -> Tuple[np.ndarray, List[str], bytes]:
    """Read a PLY file. Returns (vertices_Nx3, header_lines, raw_bytes_after_header)."""
    with open(path, 'rb') as f:
        raw = f.read()

    end_idx = raw.index(b'end_header\n') + len(b'end_header\n')
    header = raw[:end_idx].decode('utf-8')
    body = raw[end_idx:]
    header_lines = header.strip().split('\n')

    # Detect format
    fmt = 'ascii'
    for line in header_lines:
        if line.startswith('format'):
            fmt = line.split()[1]

    # Find vertex count
    vertex_count = 0
    for line in header_lines:
        if line.startswith('element vertex'):
            vertex_count = int(line.split()[-1])

    # Detect property names/order to find x, y, z columns
    props = []
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

    x_idx = next(i for i, (_, name) in enumerate(props) if name == 'x')
    y_idx = next(i for i, (_, name) in enumerate(props) if name == 'y')
    z_idx = next(i for i, (_, name) in enumerate(props) if name == 'z')

    if fmt == 'ascii':
        lines = body.decode('utf-8').strip().split('\n')
        verts = np.zeros((vertex_count, 3))
        for i in range(vertex_count):
            parts = lines[i].split()
            verts[i] = [float(parts[x_idx]), float(parts[y_idx]), float(parts[z_idx])]
        return verts, header_lines, body
    else:
        # binary_little_endian
        dtype_map = {'float': 'f4', 'double': 'f8', 'uchar': 'u1', 'int': 'i4', 'uint': 'u4', 'short': 'i2', 'ushort': 'u2'}
        dt = np.dtype([(name, '<' + dtype_map.get(typ, 'f4')) for typ, name in props])
        data = np.frombuffer(body[:vertex_count * dt.itemsize], dtype=dt)
        verts = np.column_stack([data['x'].astype(np.float64), data['y'].astype(np.float64), data['z'].astype(np.float64)])
        return verts, header_lines, body


def write_ply_transformed(path: Path, src_path: Path, verts_new: np.ndarray) -> None:
    """Rewrite a PLY with transformed vertex positions, preserving all other data."""
    with open(src_path, 'rb') as f:
        raw = f.read()

    end_idx = raw.index(b'end_header\n') + len(b'end_header\n')
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

    props = []
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

    x_idx = next(i for i, (_, name) in enumerate(props) if name == 'x')
    y_idx = next(i for i, (_, name) in enumerate(props) if name == 'y')
    z_idx = next(i for i, (_, name) in enumerate(props) if name == 'z')

    if fmt == 'ascii':
        lines = body.decode('utf-8').strip().split('\n')
        out_lines = []
        for i in range(min(vertex_count, len(verts_new))):
            parts = lines[i].split()
            parts[x_idx] = f'{verts_new[i, 0]:.8f}'
            parts[y_idx] = f'{verts_new[i, 1]:.8f}'
            parts[z_idx] = f'{verts_new[i, 2]:.8f}'
            out_lines.append(' '.join(parts))
        # keep any remaining non-vertex lines
        remainder = lines[vertex_count:]
        with open(path, 'w') as f:
            f.write(header)
            f.write('\n'.join(out_lines + remainder))
            if out_lines or remainder:
                f.write('\n')
    else:
        dtype_map = {'float': 'f4', 'double': 'f8', 'uchar': 'u1', 'int': 'i4', 'uint': 'u4', 'short': 'i2', 'ushort': 'u2'}
        dt = np.dtype([(name, '<' + dtype_map.get(typ, 'f4')) for typ, name in props])
        row_size = dt.itemsize
        data = np.frombuffer(body[:vertex_count * row_size], dtype=dt).copy()
        x_dtype = dt.fields['x'][0]
        data['x'] = verts_new[:, 0].astype(x_dtype)
        data['y'] = verts_new[:, 1].astype(x_dtype)
        data['z'] = verts_new[:, 2].astype(x_dtype)
        with open(path, 'wb') as f:
            f.write(header.encode('utf-8'))
            f.write(data.tobytes())
            f.write(body[vertex_count * row_size:])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='Align COLMAP cloud to NED via Umeyama transform')
    parser.add_argument('--cloud', required=True, help='Input PLY point cloud')
    parser.add_argument('--images-bin', required=True, help='Path to COLMAP images.bin')
    parser.add_argument('--metadata', nargs='+', required=True, help='Metadata directories (seed + adaptive)')
    parser.add_argument('--output', required=True, help='Output aligned PLY')
    parser.add_argument('--bbox-center', nargs=3, type=float, default=None, help='Bounding box centre (NED)')
    parser.add_argument('--bbox-half-extent', type=float, default=None, help='Half-extent of crop cube')
    args = parser.parse_args()

    # Load correspondences
    colmap_centres = read_images_bin(Path(args.images_bin))
    ned_positions = load_ned_positions([Path(d) for d in args.metadata])

    # Match by image filename
    matched_colmap = []
    matched_ned = []
    for img_name, colmap_pos in colmap_centres.items():
        if img_name in ned_positions:
            matched_colmap.append(colmap_pos)
            matched_ned.append(ned_positions[img_name])

    if len(matched_colmap) < 3:
        print(f'ERROR: Only {len(matched_colmap)} matched cameras (need >=3). Check metadata.')
        sys.exit(1)

    src = np.array(matched_colmap)
    dst = np.array(matched_ned)

    s, R, t = umeyama(src, dst)

    # Residuals
    transformed = s * (R @ src.T).T + t
    residuals = np.linalg.norm(transformed - dst, axis=1)
    print(f'Matched cameras: {len(matched_colmap)}')
    print(f'Scale: {s:.6f}')
    print(f'RMS residual: {np.sqrt(np.mean(residuals**2)):.6f} m')
    print(f'Max residual: {np.max(residuals):.6f} m')

    # Transform cloud
    verts, header_lines, body = read_ply(Path(args.cloud))
    verts_aligned = s * (R @ verts.T).T + t

    # Optional bbox crop
    if args.bbox_center is not None and args.bbox_half_extent is not None:
        bc = np.array(args.bbox_center)
        he = args.bbox_half_extent
        mask = np.all(np.abs(verts_aligned - bc) <= he, axis=1)
        verts_aligned = verts_aligned[mask]
        print(f'Bbox crop: {mask.sum()}/{len(mask)} points kept')

        # For cropped output, write a simple PLY
        out = Path(args.output)
        with open(out, 'w') as f:
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write(f'element vertex {len(verts_aligned)}\n')
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('end_header\n')
            for pt in verts_aligned:
                f.write(f'{pt[0]:.8f} {pt[1]:.8f} {pt[2]:.8f}\n')
    else:
        write_ply_transformed(Path(args.output), Path(args.cloud), verts_aligned)

    print(f'Aligned cloud written to {args.output}')


if __name__ == '__main__':
    main()
