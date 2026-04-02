from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..contracts import CandidateViewpoint, ScoreBreakdown, ScoreContext, SparseMetricsSnapshot
from ..scoring_interface import BaseScorer
from .covisibility_scorer import (
    CovisibilityScorer,
    _qvec_to_rotmat,
    _umeyama,
    _wrap_pi,
)

# Reuse COLMAP binary readers (already on sys.path from covisibility_scorer import)
from common import read_points3d_bin_with_tracks, read_cameras_bin, read_images_bin, find_best_sparse_model  # type: ignore


class RepairWeightedCovisibilityScorer(BaseScorer):
    """
    NBV scorer that replaces plain co-visibility count with a repair-weighted
    co-visibility term.  Each visible sparse point contributes a weight based
    on how *weak* or under-supported it currently is.

    Weakness components (configurable):
      - track_support : inverse of track length (short tracks -> high weakness)
      - local_density : kNN-based sparsity (isolated points -> high weakness)
      - reprojection  : mean reprojection error (disabled by default)

    The candidate with the largest total "repair mass" scores highest.
    Novelty, movement cost, and angular separation penalty are preserved
    unchanged from CovisibilityScorer.
    """

    # ---- Reuse static helpers from CovisibilityScorer ----
    _build_camera_rotation = staticmethod(CovisibilityScorer._build_camera_rotation)
    _view_direction = staticmethod(CovisibilityScorer._view_direction)
    _existing_view_directions = staticmethod(CovisibilityScorer._existing_view_directions)
    _extract_camera_centres = staticmethod(CovisibilityScorer._extract_camera_centres)
    _extract_ned_centres = staticmethod(CovisibilityScorer._extract_ned_centres)

    def _compute_alignment(self, colmap_centres, ned_centres, images):
        return CovisibilityScorer._compute_alignment(self, colmap_centres, ned_centres, images)

    def _novelty(self, cand, visited):
        return CovisibilityScorer._novelty(self, cand, visited)

    def _movement_cost(self, cand, current_xyz, current_yaw):
        return CovisibilityScorer._movement_cost(self, cand, current_xyz, current_yaw)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def score_candidates(
        self,
        candidates: Sequence[CandidateViewpoint],
        sparse_metrics: SparseMetricsSnapshot,
        context: ScoreContext,
    ) -> List[ScoreBreakdown]:
        cfg = self.config
        weights = cfg.get('weights', {})
        min_track = int(cfg.get('min_track_length', 2))
        min_angle_deg = float(cfg.get('min_angular_separation_deg', 15.0))

        # Repair-weighted config
        rw_cfg = cfg.get('repair_weighted_covisibility', {})
        weakness_cfg = rw_cfg.get('weakness_weights', {})
        alpha_track = float(weakness_cfg.get('track_support', 0.6))
        alpha_density = float(weakness_cfg.get('local_density', 0.4))
        alpha_reproj = float(weakness_cfg.get('reprojection', 0.0))
        target_track_length = int(rw_cfg.get('target_track_length', 6))
        density_k = int(rw_cfg.get('density_k', 5))
        max_scored_points = int(rw_cfg.get('max_scored_points', 0))  # 0 = no cap

        # --- Load COLMAP data ---
        workspace = Path(context.colmap_workspace) if context.colmap_workspace else None
        if workspace is None or not workspace.exists():
            return self._zero_scores(candidates)

        sparse_dir = find_best_sparse_model(workspace / 'sparse')
        points3d_path = sparse_dir / 'points3D.bin'
        cameras_path = sparse_dir / 'cameras.bin'
        images_path = sparse_dir / 'images.bin'

        if not all(p.exists() for p in [points3d_path, cameras_path, images_path]):
            return self._zero_scores(candidates)

        points3d = read_points3d_bin_with_tracks(points3d_path)
        cameras = read_cameras_bin(cameras_path)
        images = read_images_bin(images_path)

        # Camera intrinsics
        cam = next(iter(cameras.values()))
        width, height = cam['width'], cam['height']
        fx, fy, cx, cy = self._get_intrinsics(cam)

        # Alignment
        colmap_centres = self._extract_camera_centres(images)
        ned_centres = self._extract_ned_centres(context.visited_viewpoints)
        s, R_align, t_align = self._compute_alignment(colmap_centres, ned_centres, images)

        # Collect tracked points and metadata
        tracked_pts_colmap = []
        tracked_ids = []
        tracked_track_lengths = []
        tracked_reproj_errors = []
        for pid, pdata in points3d.items():
            tlen = len(pdata['track'])
            if tlen >= min_track:
                tracked_pts_colmap.append(pdata['xyz'])
                tracked_ids.append(pid)
                tracked_track_lengths.append(tlen)
                tracked_reproj_errors.append(pdata.get('error', 0.0))

        if not tracked_pts_colmap:
            return self._zero_scores(candidates)

        pts_colmap = np.array(tracked_pts_colmap, dtype=np.float64)
        track_lengths = np.array(tracked_track_lengths, dtype=np.float64)
        reproj_errors = np.array(tracked_reproj_errors, dtype=np.float64)

        # Transform to NED
        if s > 0 and R_align is not None:
            R_inv = R_align.T
            pts_ned = (R_inv @ (pts_colmap - t_align).T).T / s
        else:
            pts_ned = pts_colmap

        # Filter to scoring bbox
        scoring_bbox_half = float(cfg.get('scoring_bbox_half_extent', 1.5))
        rock = np.array(context.rock_center_xyz, dtype=np.float64)
        offsets = np.abs(pts_ned - rock)
        in_bbox = np.all(offsets <= scoring_bbox_half, axis=1)
        pts_ned = pts_ned[in_bbox]
        track_lengths = track_lengths[in_bbox]
        reproj_errors = reproj_errors[in_bbox]

        if len(pts_ned) == 0:
            return self._zero_scores(candidates)

        # --- Precompute per-point weakness weights ---
        weakness_weights = self._compute_weakness_weights(
            pts_ned, track_lengths, reproj_errors,
            alpha_track, alpha_density, alpha_reproj,
            target_track_length, density_k,
        )

        # Optional: cap to top-K weakest points
        if max_scored_points > 0 and len(pts_ned) > max_scored_points:
            top_k_idx = np.argsort(weakness_weights)[-max_scored_points:]
            pts_ned = pts_ned[top_k_idx]
            weakness_weights = weakness_weights[top_k_idx]

        total_weakness = float(np.sum(weakness_weights))
        eps = 1e-8

        # Existing view directions for angular separation
        existing_dirs = self._existing_view_directions(context.visited_viewpoints, context.rock_center_xyz)

        # --- Score each candidate ---
        out: List[ScoreBreakdown] = []
        for cand in candidates:
            # Gimbal pitch toward rock
            delta = rock - np.array([cand.x, cand.y, cand.z], dtype=np.float64)
            horiz_dist = math.sqrt(delta[0] ** 2 + delta[1] ** 2)
            gimbal_pitch = -math.atan2(delta[2], horiz_dist) if horiz_dist > 0.01 else -math.pi / 2

            R_cam = self._build_camera_rotation(cand.yaw, gimbal_pitch)
            t_cam = np.array([cand.x, cand.y, cand.z], dtype=np.float64)
            pts_cam = (R_cam @ (pts_ned - t_cam).T).T

            # Frustum check
            in_front = pts_cam[:, 2] > 0
            visible_mask = np.zeros(len(pts_ned), dtype=bool)
            if np.any(in_front):
                pts_front = pts_cam[in_front]
                u = fx * pts_front[:, 0] / pts_front[:, 2] + cx
                v = fy * pts_front[:, 1] / pts_front[:, 2] + cy
                in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
                front_indices = np.where(in_front)[0]
                visible_mask[front_indices[in_bounds]] = True

            # Repair mass
            repair_mass = float(np.sum(weakness_weights[visible_mask]))
            normalized_repair_score = repair_mass / max(total_weakness, eps)
            visible_count = int(np.sum(visible_mask))

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

            novelty = self._novelty(cand, context.visited_viewpoints)
            movement_cost = self._movement_cost(cand, context.current_position_xyz, context.current_yaw_rad)

            final = (
                weights.get('repair', weights.get('covisibility', 1.0)) * normalized_repair_score
                + weights.get('novelty', 0.3) * novelty
                - weights.get('movement_cost', 0.2) * movement_cost
                - weights.get('angular_separation_penalty', 0.15) * angular_penalty
            )

            out.append(ScoreBreakdown(
                candidate_id=cand.candidate_id,
                final_score=float(final),
                terms={
                    'repair_mass': repair_mass,
                    'normalized_repair_score': float(normalized_repair_score),
                    'visible_weak_point_count': float(visible_count),
                    'mean_visible_weakness': float(
                        np.mean(weakness_weights[visible_mask]) if visible_count > 0 else 0.0
                    ),
                    'total_weakness_in_box': total_weakness,
                    'novelty': float(novelty),
                    'movement_cost': float(movement_cost),
                    'angular_separation_penalty': float(angular_penalty),
                },
                weights={k: float(v) for k, v in weights.items()},
                scorer_name='repair_weighted_covisibility',
            ))

        return sorted(out, key=lambda s: s.final_score, reverse=True)

    # ------------------------------------------------------------------
    # Weakness computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_weakness_weights(
        pts_ned: np.ndarray,
        track_lengths: np.ndarray,
        reproj_errors: np.ndarray,
        alpha_track: float,
        alpha_density: float,
        alpha_reproj: float,
        target_track_length: int,
        density_k: int,
    ) -> np.ndarray:
        """Compute a combined weakness weight for each point.

        Returns an array of shape (N,) with values in [0, 1].
        """
        n = len(pts_ned)
        weakness = np.zeros(n, dtype=np.float64)

        # 1) Track-support weakness: clipped inverse support
        if alpha_track > 0:
            track_w = np.clip(
                (target_track_length - track_lengths) / max(target_track_length, 1),
                0.0, 1.0,
            )
            weakness += alpha_track * track_w

        # 2) Local-density weakness: kNN distance
        if alpha_density > 0 and n > 1:
            density_w = _knn_density_weakness(pts_ned, min(density_k, n - 1))
            weakness += alpha_density * density_w

        # 3) Optional reprojection weakness
        if alpha_reproj > 0 and np.any(reproj_errors > 0):
            reproj_w = _reproj_weakness(reproj_errors)
            weakness += alpha_reproj * reproj_w

        # Clamp and sanitize
        np.clip(weakness, 0.0, 1.0, out=weakness)
        weakness = np.nan_to_num(weakness, nan=0.0, posinf=0.0, neginf=0.0)
        return weakness

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_intrinsics(cam: Dict) -> Tuple[float, float, float, float]:
        width, height = cam['width'], cam['height']
        if cam['model'] == 'PINHOLE':
            fx, fy, cx, cy = cam['params']
        elif cam['model'] == 'SIMPLE_PINHOLE':
            f, cx, cy = cam['params']
            fx = fy = f
        else:
            fx = fy = float(max(width, height))
            cx, cy = width / 2.0, height / 2.0
        return fx, fy, cx, cy

    def _zero_scores(self, candidates: Sequence[CandidateViewpoint]) -> List[ScoreBreakdown]:
        return [
            ScoreBreakdown(
                candidate_id=c.candidate_id, final_score=0.0,
                terms={}, weights={}, scorer_name='repair_weighted_covisibility',
            )
            for c in candidates
        ]


# ------------------------------------------------------------------
# Module-level helpers for weakness computation
# ------------------------------------------------------------------

def _knn_density_weakness(pts: np.ndarray, k: int) -> np.ndarray:
    """Compute density-based weakness using k-nearest-neighbor distances.

    Points in sparser regions get higher weakness values.
    Uses percentile-based normalization for robustness against outliers.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(pts)
    # k+1 because the query point itself is always the closest neighbor
    dists, _ = tree.query(pts, k=k + 1)
    # Mean distance to k nearest neighbors (skip self at index 0)
    mean_knn_dist = np.mean(dists[:, 1:], axis=1)

    # Percentile-based normalization (robust to outliers)
    p5 = np.percentile(mean_knn_dist, 5)
    p95 = np.percentile(mean_knn_dist, 95)
    if p95 - p5 < 1e-8:
        return np.zeros(len(pts), dtype=np.float64)

    normalized = np.clip((mean_knn_dist - p5) / (p95 - p5), 0.0, 1.0)
    return normalized


def _reproj_weakness(reproj_errors: np.ndarray) -> np.ndarray:
    """Normalize reprojection errors to [0, 1] using percentile scaling."""
    p5 = np.percentile(reproj_errors, 5)
    p95 = np.percentile(reproj_errors, 95)
    if p95 - p5 < 1e-8:
        return np.zeros(len(reproj_errors), dtype=np.float64)
    return np.clip((reproj_errors - p5) / (p95 - p5), 0.0, 1.0)
