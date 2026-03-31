from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from ..contracts import CandidateViewpoint, ScoreBreakdown, ScoreContext, SparseMetricsSnapshot
from ..scoring_interface import BaseScorer

# Add colmap_scripts to import path so we can reuse the binary readers
import sys as _sys
try:
    from ament_index_python.packages import get_package_share_directory as _get_share
    _COLMAP_SCRIPTS = Path(_get_share('photogrammetry_nbv')) / 'colmap_scripts'
except Exception:
    # Fallback for source-tree runs without ament
    _COLMAP_SCRIPTS = Path(__file__).resolve().parents[2] / 'colmap_scripts'
if str(_COLMAP_SCRIPTS) not in _sys.path:
    _sys.path.insert(0, str(_COLMAP_SCRIPTS))

from common import read_points3d_bin_with_tracks, read_cameras_bin, read_images_bin  # type: ignore


class CovisibilityScorer(BaseScorer):
    """
    NBV scorer that maximises geometric co-visibility with the existing
    COLMAP reconstruction.

    For each candidate pose the scorer projects all tracked 3D points into
    the candidate's camera frustum and counts how many are visible.  The
    candidate that would observe the most already-tracked points is ranked
    highest, subject to diversity and movement-cost penalties.
    """

    def score_candidates(
        self,
        candidates: Sequence[CandidateViewpoint],
        sparse_metrics: SparseMetricsSnapshot,
        context: ScoreContext,
    ) -> List[ScoreBreakdown]:
        weights = self.config.get('weights', {})
        min_track = int(self.config.get('min_track_length', 2))
        min_angle_deg = float(self.config.get('min_angular_separation_deg', 15.0))

        # --- Load COLMAP data ---
        workspace = Path(context.colmap_workspace) if context.colmap_workspace else None
        if workspace is None or not workspace.exists():
            # Fallback: return zero scores if no workspace available
            return self._zero_scores(candidates)

        sparse_dir = workspace / 'sparse' / '0'
        points3d_path = sparse_dir / 'points3D.bin'
        cameras_path = sparse_dir / 'cameras.bin'
        images_path = sparse_dir / 'images.bin'

        if not all(p.exists() for p in [points3d_path, cameras_path, images_path]):
            return self._zero_scores(candidates)

        points3d = read_points3d_bin_with_tracks(points3d_path)
        cameras = read_cameras_bin(cameras_path)
        images = read_images_bin(images_path)

        # --- Get camera intrinsics (use first camera) ---
        cam = next(iter(cameras.values()))
        width, height = cam['width'], cam['height']
        if cam['model'] == 'PINHOLE':
            fx, fy, cx, cy = cam['params']
        elif cam['model'] == 'SIMPLE_PINHOLE':
            f, cx, cy = cam['params']
            fx = fy = f
        else:
            fx = fy = max(width, height)
            cx, cy = width / 2, height / 2

        # --- Compute NED-to-COLMAP alignment ---
        # We need this to project NED candidate poses into COLMAP world frame.
        # Use image poses from images.bin (COLMAP frame) matched against
        # visited_viewpoints metadata (NED frame) via Umeyama.
        colmap_centres = self._extract_camera_centres(images)
        ned_centres = self._extract_ned_centres(context.visited_viewpoints)

        s, R_align, t_align = self._compute_alignment(colmap_centres, ned_centres, images)
        # Transform: colmap_point = s * R_align @ ned_point + t_align
        # Inverse: ned_point = R_align^T @ (colmap_point - t_align) / s
        # We need NED->COLMAP for candidates, and COLMAP->NED for 3D points
        # Actually: project COLMAP 3D points into candidate cameras.
        # The candidate pose is in NED. We need to transform it to COLMAP frame
        # to build the camera extrinsic, OR transform 3D points to NED and
        # build camera extrinsic in NED.
        # Simpler: transform all 3D points to NED once, then project with NED camera.

        # Transform COLMAP 3D points to NED frame
        tracked_pts_colmap = []
        tracked_ids = []
        for pid, pdata in points3d.items():
            if len(pdata['track']) >= min_track:
                tracked_pts_colmap.append(pdata['xyz'])
                tracked_ids.append(pid)

        if not tracked_pts_colmap:
            return self._zero_scores(candidates)

        pts_colmap = np.array(tracked_pts_colmap, dtype=np.float64)  # (N, 3)

        if s > 0 and R_align is not None:
            # COLMAP -> NED: p_ned = R_align^T @ (p_colmap - t_align) / s
            R_inv = R_align.T
            pts_ned = (R_inv @ (pts_colmap - t_align).T).T / s
        else:
            # No alignment possible (< 3 correspondences). Use raw COLMAP coords.
            pts_ned = pts_colmap

        # --- Filter 3D points to scoring bounding box around rock ---
        scoring_bbox_half = float(self.config.get('scoring_bbox_half_extent', 1.5))
        rock = np.array(context.rock_center_xyz, dtype=np.float64)
        offsets = np.abs(pts_ned - rock)
        in_bbox = np.all(offsets <= scoring_bbox_half, axis=1)
        pts_ned = pts_ned[in_bbox]

        if len(pts_ned) == 0:
            return self._zero_scores(candidates)

        # --- Compute existing image viewing directions (for angular separation) ---
        existing_dirs = self._existing_view_directions(context.visited_viewpoints, context.rock_center_xyz)

        # --- Score each candidate ---
        out: List[ScoreBreakdown] = []
        max_possible = len(pts_ned)

        for cand in candidates:
            # Compute per-candidate gimbal pitch to look at rock center
            delta = rock - np.array([cand.x, cand.y, cand.z], dtype=np.float64)
            horiz_dist = math.sqrt(delta[0]**2 + delta[1]**2)
            cand_gimbal_pitch = -math.atan2(delta[2], horiz_dist) if horiz_dist > 0.01 else -math.pi / 2

            # Build world-to-camera transform for this candidate in NED
            R_cam = self._build_camera_rotation(cand.yaw, cand_gimbal_pitch)
            t_cam = np.array([cand.x, cand.y, cand.z], dtype=np.float64)

            # Transform points to camera frame: p_cam = R_cam @ (p_world - t_cam)
            pts_cam = (R_cam @ (pts_ned - t_cam).T).T  # (N, 3)

            # Filter: in front of camera (z_cam > 0)
            in_front = pts_cam[:, 2] > 0
            pts_front = pts_cam[in_front]

            if len(pts_front) == 0:
                covis_count = 0
            else:
                # Project to pixel coordinates
                u = fx * pts_front[:, 0] / pts_front[:, 2] + cx
                v = fy * pts_front[:, 1] / pts_front[:, 2] + cy

                # Filter: within image bounds
                in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
                covis_count = int(np.sum(in_bounds))

            # Normalize covisibility
            covis_score = covis_count / max(1, max_possible)

            # Angular separation penalty
            angular_penalty = 0.0
            if existing_dirs:
                cand_dir = self._view_direction(cand, context.rock_center_xyz)
                min_angle = min(
                    math.degrees(math.acos(np.clip(np.dot(cand_dir, ed), -1.0, 1.0)))
                    for ed in existing_dirs
                )
                if min_angle < min_angle_deg:
                    angular_penalty = 1.0 - (min_angle / min_angle_deg)

            # Novelty (distance from visited viewpoints)
            novelty = self._novelty(cand, context.visited_viewpoints)

            # Movement cost
            movement_cost = self._movement_cost(cand, context.current_position_xyz, context.current_yaw_rad)

            final = (
                weights.get('covisibility', 1.0) * covis_score
                + weights.get('novelty', 0.3) * novelty
                - weights.get('movement_cost', 0.2) * movement_cost
                - weights.get('angular_separation_penalty', 0.15) * angular_penalty
            )

            out.append(ScoreBreakdown(
                candidate_id=cand.candidate_id,
                final_score=float(final),
                terms={
                    'covisibility': float(covis_score),
                    'covisibility_count': float(covis_count),
                    'novelty': float(novelty),
                    'movement_cost': float(movement_cost),
                    'angular_separation_penalty': float(angular_penalty),
                },
                weights={k: float(v) for k, v in weights.items()},
                scorer_name='covisibility',
            ))

        return sorted(out, key=lambda s: s.final_score, reverse=True)

    # ------------------------------------------------------------------
    # Camera geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_camera_rotation(yaw: float, gimbal_pitch: float) -> np.ndarray:
        """Build world-to-camera rotation from NED yaw + gimbal pitch.

        NED convention: x=North, y=East, z=Down.
        Camera convention: x=Right, y=Down, z=Forward (OpenCV).

        The drone faces yaw direction (rotation about NED-down axis), then
        the gimbal tilts the camera by pitch (rotation about camera-right axis).
        """
        # Rotation about NED z-axis (down) by yaw
        cy, sy = math.cos(yaw), math.sin(yaw)
        R_yaw = np.array([
            [cy, sy, 0],
            [-sy, cy, 0],
            [0, 0, 1],
        ], dtype=np.float64)

        # Rotation about y-axis (right) by gimbal pitch
        cp, sp = math.cos(gimbal_pitch), math.sin(gimbal_pitch)
        R_pitch = np.array([
            [cp, 0, -sp],
            [0,  1,  0],
            [sp, 0,  cp],
        ], dtype=np.float64)

        # NED-to-camera body frame conversion:
        # NED forward (x) -> camera forward (z)
        # NED right (y) -> camera right (x)
        # NED down (z) -> camera down (y)
        R_ned_to_cam = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ], dtype=np.float64)

        # Combined: world -> body (yaw) -> pitched body -> camera frame
        return R_ned_to_cam @ R_pitch @ R_yaw

    @staticmethod
    def _view_direction(cand: CandidateViewpoint, rock_center: List[float]) -> np.ndarray:
        """Unit vector from candidate toward rock center."""
        d = np.array([rock_center[0] - cand.x, rock_center[1] - cand.y, rock_center[2] - cand.z])
        norm = np.linalg.norm(d)
        return d / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])

    @staticmethod
    def _existing_view_directions(visited: Sequence[Dict], rock_center: List[float]) -> List[np.ndarray]:
        """Compute viewing directions for all visited viewpoints."""
        dirs = []
        rc = np.array(rock_center)
        for vp in visited:
            pos = vp.get('position_ned_m', {})
            if 'x' in pos:
                p = np.array([pos['x'], pos['y'], pos['z']])
                d = rc - p
                norm = np.linalg.norm(d)
                if norm > 1e-6:
                    dirs.append(d / norm)
        return dirs

    # ------------------------------------------------------------------
    # NED-to-COLMAP alignment (Umeyama)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_camera_centres(images: Dict[int, Dict]) -> Dict[str, np.ndarray]:
        """Extract camera centres from COLMAP images.bin data. Centre = -R^T @ t."""
        centres = {}
        for _img_id, img_data in images.items():
            R = _qvec_to_rotmat(np.array(img_data['qvec']))
            t = np.array(img_data['tvec'])
            centres[img_data['name']] = -R.T @ t
        return centres

    @staticmethod
    def _extract_ned_centres(visited: Sequence[Dict]) -> Dict[str, np.ndarray]:
        """Build a lookup of image filename -> NED position from visited viewpoints.

        This relies on visited_viewpoints having an 'image_file' key for adaptive
        captures and the seed metadata. For seed images we use 'source'='seed'.
        """
        centres = {}
        for i, vp in enumerate(visited):
            pos = vp.get('position_ned_m', {})
            if 'x' in pos:
                # Use index-based key as fallback; controller stores image_file for adaptive
                key = vp.get('image_file', f'_vp_{i}')
                centres[key] = np.array([pos['x'], pos['y'], pos['z']])
        return centres

    def _compute_alignment(
        self,
        colmap_centres: Dict[str, np.ndarray],
        ned_centres: Dict[str, np.ndarray],
        images: Dict[int, Dict],
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Compute Umeyama alignment from NED to COLMAP frame.

        Matching is done by image filename. If filenames don't match (common
        when seed images have different names), fall back to ordering-based
        matching using all registered COLMAP images against visited viewpoints
        in order.

        Returns (scale, R, t) such that: p_colmap ≈ s * R @ p_ned + t
        """
        # Try filename matching first
        matched_colmap = []
        matched_ned = []
        for name, c_pos in colmap_centres.items():
            if name in ned_centres:
                matched_colmap.append(c_pos)
                matched_ned.append(ned_centres[name])

        # Fallback: order-based matching (COLMAP image order vs visited order)
        if len(matched_colmap) < 3:
            colmap_ordered = []
            for _img_id in sorted(images.keys()):
                name = images[_img_id]['name']
                if name in colmap_centres:
                    colmap_ordered.append(colmap_centres[name])

            ned_ordered = list(ned_centres.values())
            n = min(len(colmap_ordered), len(ned_ordered))
            if n >= 3:
                matched_colmap = colmap_ordered[:n]
                matched_ned = ned_ordered[:n]

        if len(matched_colmap) < 3:
            # Cannot compute alignment
            return 0.0, np.eye(3), np.zeros(3)

        src = np.array(matched_ned)
        dst = np.array(matched_colmap)
        return _umeyama(src, dst)

    # ------------------------------------------------------------------
    # Score term helpers (reused from weighted_sum pattern)
    # ------------------------------------------------------------------

    def _novelty(self, cand: CandidateViewpoint, visited: Sequence[Dict]) -> float:
        if not visited:
            return 1.0
        dists = []
        for vp in visited:
            pos = vp.get('position_ned_m', {})
            dx = cand.x - float(pos.get('x', cand.x))
            dy = cand.y - float(pos.get('y', cand.y))
            dz = cand.z - float(pos.get('z', cand.z))
            dists.append(math.sqrt(dx * dx + dy * dy + dz * dz))
        scale = max(1.0, float(self.config.get('novelty_distance_scale_m', 5.0)))
        return min(1.0, min(dists) / scale)

    def _movement_cost(self, cand: CandidateViewpoint, current_xyz: Sequence[float], current_yaw: float) -> float:
        dx = cand.x - float(current_xyz[0])
        dy = cand.y - float(current_xyz[1])
        dz = cand.z - float(current_xyz[2])
        travel = math.sqrt(dx * dx + dy * dy + dz * dz)
        yaw_cost = abs(_wrap_pi(cand.yaw - current_yaw))
        travel_scale = max(1.0, float(self.config.get('travel_distance_scale_m', 10.0)))
        yaw_scale = max(0.1, float(self.config.get('yaw_change_scale_rad', math.pi)))
        return 0.7 * (travel / travel_scale) + 0.3 * (yaw_cost / yaw_scale)

    def _zero_scores(self, candidates: Sequence[CandidateViewpoint]) -> List[ScoreBreakdown]:
        return [
            ScoreBreakdown(
                candidate_id=c.candidate_id, final_score=0.0,
                terms={}, weights={}, scorer_name='covisibility',
            )
            for c in candidates
        ]


# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------

def _qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y,     2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y],
    ])


def _umeyama(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute 7-DOF similarity transform: dst ≈ s * R @ src + t"""
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


def _wrap_pi(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle
