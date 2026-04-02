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
from .repair_weighted_covisibility_scorer import RepairWeightedCovisibilityScorer

# Reuse COLMAP binary readers
from common import read_points3d_bin_with_tracks, read_cameras_bin, read_images_bin, find_best_sparse_model  # type: ignore


class BaselineAwareRepairWeightedCovisibilityScorer(BaseScorer):
    """
    Geometry-aware NBV scorer that extends repair-weighted co-visibility by
    rewarding candidates that observe weak points from geometrically *useful*
    new directions.

    For each visible weak point, the scorer computes a geometry gain based on
    the angular novelty of the candidate's viewing direction relative to
    existing observation rays.  A candidate that sees a weak point from a
    nearly redundant angle gets little credit; one that adds a new baseline
    gets full credit.

    Scoring equation:
      geometry_aware_repair_mass(c) = sum_p weakness(p) * geometry_gain(p, c)
      final(c) = w_repair * norm_repair + w_novelty * novelty
                 - w_move * move_cost - w_angular * angular_penalty
    """

    # ---- Reuse static helpers from CovisibilityScorer ----
    _build_camera_rotation = staticmethod(CovisibilityScorer._build_camera_rotation)
    _view_direction = staticmethod(CovisibilityScorer._view_direction)
    _existing_view_directions = staticmethod(CovisibilityScorer._existing_view_directions)
    _extract_camera_centres = staticmethod(CovisibilityScorer._extract_camera_centres)
    _extract_ned_centres = staticmethod(CovisibilityScorer._extract_ned_centres)

    # Reuse instance methods via delegation
    def _compute_alignment(self, colmap_centres, ned_centres, images):
        return CovisibilityScorer._compute_alignment(self, colmap_centres, ned_centres, images)

    def _novelty(self, cand, visited):
        return CovisibilityScorer._novelty(self, cand, visited)

    def _movement_cost(self, cand, current_xyz, current_yaw):
        return CovisibilityScorer._movement_cost(self, cand, current_xyz, current_yaw)

    _get_intrinsics = staticmethod(RepairWeightedCovisibilityScorer._get_intrinsics)
    _compute_weakness_weights = staticmethod(RepairWeightedCovisibilityScorer._compute_weakness_weights)

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

        # Repair-weighted config (shared weakness parameters)
        rw_cfg = cfg.get('repair_weighted_covisibility', cfg.get('baseline_aware_repair_weighted_covisibility', {}))
        weakness_cfg = rw_cfg.get('weakness_weights', {})
        alpha_track = float(weakness_cfg.get('track_support', 0.6))
        alpha_density = float(weakness_cfg.get('local_density', 0.4))
        alpha_reproj = float(weakness_cfg.get('reprojection', 0.0))
        target_track_length = int(rw_cfg.get('target_track_length', 6))
        density_k = int(rw_cfg.get('density_k', 5))
        max_scored_points = int(rw_cfg.get('max_scored_points', 0))

        # Geometry-aware config
        geo_cfg = cfg.get('baseline_aware_repair_weighted_covisibility', {})
        theta_low_deg = float(geo_cfg.get('theta_low_deg', 5.0))
        theta_target_deg = float(geo_cfg.get('theta_target_deg', 25.0))
        theta_high_deg = float(geo_cfg.get('theta_high_deg', 45.0))

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

        # Collect tracked points with metadata
        tracked_pts_colmap = []
        tracked_ids = []
        tracked_track_lengths = []
        tracked_reproj_errors = []
        tracked_image_ids: List[List[int]] = []  # for observation geometry
        for pid, pdata in points3d.items():
            tlen = len(pdata['track'])
            if tlen >= min_track:
                tracked_pts_colmap.append(pdata['xyz'])
                tracked_ids.append(pid)
                tracked_track_lengths.append(tlen)
                tracked_reproj_errors.append(pdata.get('error', 0.0))
                tracked_image_ids.append([obs[0] for obs in pdata['track']])

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
        bbox_indices = np.where(in_bbox)[0]

        pts_ned = pts_ned[in_bbox]
        track_lengths = track_lengths[in_bbox]
        reproj_errors = reproj_errors[in_bbox]
        tracked_image_ids = [tracked_image_ids[i] for i in bbox_indices]

        if len(pts_ned) == 0:
            return self._zero_scores(candidates)

        # --- Precompute weakness weights (Layer 1) ---
        weakness_weights = self._compute_weakness_weights(
            pts_ned, track_lengths, reproj_errors,
            alpha_track, alpha_density, alpha_reproj,
            target_track_length, density_k,
        )

        # --- Precompute existing observation geometry (Layer 2) ---
        # For each point, compute existing viewing rays from all observing cameras
        # Camera centres in COLMAP frame, then transform to NED
        all_colmap_cam_centres = {}
        for img_id, img_data in images.items():
            R_img = _qvec_to_rotmat(np.array(img_data['qvec']))
            t_img = np.array(img_data['tvec'])
            all_colmap_cam_centres[img_id] = -R_img.T @ t_img

        existing_rays_per_point: List[Optional[np.ndarray]] = []
        for i, img_ids in enumerate(tracked_image_ids):
            cam_centres_colmap = []
            for iid in img_ids:
                if iid in all_colmap_cam_centres:
                    cam_centres_colmap.append(all_colmap_cam_centres[iid])

            if not cam_centres_colmap:
                existing_rays_per_point.append(None)
                continue

            cam_centres_col = np.array(cam_centres_colmap, dtype=np.float64)
            # Transform camera centres to NED
            if s > 0 and R_align is not None:
                cam_centres_ned = (R_inv @ (cam_centres_col - t_align).T).T / s
            else:
                cam_centres_ned = cam_centres_col

            # Compute unit rays from each camera centre toward the point
            rays = pts_ned[i] - cam_centres_ned  # shape (n_obs, 3)
            norms = np.linalg.norm(rays, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            rays = rays / norms
            existing_rays_per_point.append(rays)

        # Optional: cap to top-K weakest points
        if max_scored_points > 0 and len(pts_ned) > max_scored_points:
            top_k_idx = np.argsort(weakness_weights)[-max_scored_points:]
            pts_ned = pts_ned[top_k_idx]
            weakness_weights = weakness_weights[top_k_idx]
            existing_rays_per_point = [existing_rays_per_point[i] for i in top_k_idx]

        total_weakness = float(np.sum(weakness_weights))
        eps = 1e-8

        # Existing view directions for angular separation penalty
        existing_dirs = self._existing_view_directions(context.visited_viewpoints, context.rock_center_xyz)

        # --- Score each candidate (Layers 3 & 4) ---
        out: List[ScoreBreakdown] = []
        for cand in candidates:
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

            # Compute geometry-aware repair mass
            geometry_aware_repair_mass = 0.0
            geometry_gains = []
            visible_indices = np.where(visible_mask)[0]
            redundant_count = 0

            for idx in visible_indices:
                # Candidate ray toward this point
                cand_ray = pts_ned[idx] - t_cam
                cand_ray_norm = np.linalg.norm(cand_ray)
                if cand_ray_norm < 1e-8:
                    geometry_gains.append(0.0)
                    continue
                cand_ray = cand_ray / cand_ray_norm

                existing_rays = existing_rays_per_point[idx]
                if existing_rays is None or len(existing_rays) == 0:
                    # No existing observations -> maximum gain
                    gain = 1.0
                else:
                    gain = _compute_geometry_gain(
                        cand_ray, existing_rays,
                        theta_low_deg, theta_target_deg, theta_high_deg,
                    )

                if gain < 0.1:
                    redundant_count += 1

                geometry_gains.append(gain)
                geometry_aware_repair_mass += weakness_weights[idx] * gain

            normalized_repair_score = geometry_aware_repair_mass / max(total_weakness, eps)
            visible_count = int(np.sum(visible_mask))
            mean_gain = float(np.mean(geometry_gains)) if geometry_gains else 0.0

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
                weights.get('geometry_aware_repair', weights.get('repair', weights.get('covisibility', 1.0)))
                    * normalized_repair_score
                + weights.get('novelty', 0.3) * novelty
                - weights.get('movement_cost', 0.2) * movement_cost
                - weights.get('angular_separation_penalty', 0.15) * angular_penalty
            )

            out.append(ScoreBreakdown(
                candidate_id=cand.candidate_id,
                final_score=float(final),
                terms={
                    'geometry_aware_repair_mass': float(geometry_aware_repair_mass),
                    'normalized_geometry_aware_repair_score': float(normalized_repair_score),
                    'mean_geometry_gain': mean_gain,
                    'visible_weak_points': float(visible_count),
                    'redundant_visible_weak_points': float(redundant_count),
                    'total_weakness_in_box': total_weakness,
                    'novelty': float(novelty),
                    'movement_cost': float(movement_cost),
                    'angular_separation_penalty': float(angular_penalty),
                },
                weights={k: float(v) for k, v in weights.items()},
                scorer_name='baseline_aware_repair_weighted_covisibility',
            ))

        return sorted(out, key=lambda s: s.final_score, reverse=True)

    def _zero_scores(self, candidates: Sequence[CandidateViewpoint]) -> List[ScoreBreakdown]:
        return [
            ScoreBreakdown(
                candidate_id=c.candidate_id, final_score=0.0,
                terms={}, weights={},
                scorer_name='baseline_aware_repair_weighted_covisibility',
            )
            for c in candidates
        ]


# ------------------------------------------------------------------
# Geometry gain computation
# ------------------------------------------------------------------

def _compute_geometry_gain(
    cand_ray: np.ndarray,
    existing_rays: np.ndarray,
    theta_low_deg: float,
    theta_target_deg: float,
    theta_high_deg: float,
) -> float:
    """Compute geometry gain for a candidate ray relative to existing observation rays.

    Uses minimum angle to any existing ray (conservative: measures redundancy).
    Band-pass gain function:
      - angle < theta_low  -> gain ~0  (too similar)
      - theta_low..theta_target -> linear ramp 0..1
      - theta_target..theta_high -> gain = 1
      - angle > theta_high -> gain = 1 (saturated)
    """
    # Compute angles between candidate ray and all existing rays
    dots = existing_rays @ cand_ray
    dots = np.clip(dots, -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(dots))

    # Use minimum angle to any existing ray
    min_angle = float(np.min(angles_deg))

    if min_angle <= theta_low_deg:
        return 0.0
    elif min_angle <= theta_target_deg:
        return (min_angle - theta_low_deg) / max(theta_target_deg - theta_low_deg, 1e-8)
    else:
        # Saturate at 1.0 for angles above theta_target
        return 1.0
