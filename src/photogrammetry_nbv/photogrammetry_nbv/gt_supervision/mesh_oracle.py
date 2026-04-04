from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import KDTree

from .coverage_state import CoverageState


# ── OBJ mesh reader + uniform surface sampler ──────────────────────────────
# Extracted from compare_scorers.py to avoid importing from the scripts/ dir.

def load_obj_mesh(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read a Wavefront OBJ -> (vertices Nx3, faces Mx3)."""
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


# ── Camera frustum helpers ─────────────────────────────────────────────────
# Reuses the same NED camera model as CovisibilityScorer.

def _build_camera_rotation(yaw: float, gimbal_pitch: float) -> np.ndarray:
    """Build world-to-camera rotation from NED yaw + gimbal pitch."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    R_yaw = np.array([
        [cy, sy, 0],
        [-sy, cy, 0],
        [0, 0, 1],
    ], dtype=np.float64)

    cp, sp = math.cos(gimbal_pitch), math.sin(gimbal_pitch)
    R_pitch = np.array([
        [cp, 0, -sp],
        [0,  1,  0],
        [sp, 0,  cp],
    ], dtype=np.float64)

    R_ned_to_cam = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
    ], dtype=np.float64)

    return R_ned_to_cam @ R_pitch @ R_yaw


def _frustum_visible_mask(
    pts_ned: np.ndarray,
    cam_pos: np.ndarray,
    R_cam: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    width: int, height: int,
) -> np.ndarray:
    """Return boolean mask of which pts_ned are visible from the camera."""
    pts_cam = (R_cam @ (pts_ned - cam_pos).T).T
    in_front = pts_cam[:, 2] > 0
    mask = np.zeros(len(pts_ned), dtype=bool)
    if not np.any(in_front):
        return mask
    pts_front = pts_cam[in_front]
    u = fx * pts_front[:, 0] / pts_front[:, 2] + cx
    v = fy * pts_front[:, 1] / pts_front[:, 2] + cy
    in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    front_indices = np.where(in_front)[0]
    mask[front_indices[in_bounds]] = True
    return mask


# ── MeshOracle ─────────────────────────────────────────────────────────────

class MeshOracle:
    """
    GT mesh oracle for computing marginal coverage rewards.

    Loads a GT mesh, samples its surface, and evaluates how many previously-
    uncovered samples a candidate viewpoint would observe.
    """

    # Default camera intrinsics (typical simulation camera)
    DEFAULT_WIDTH = 640
    DEFAULT_HEIGHT = 480
    DEFAULT_FX = 462.0
    DEFAULT_FY = 462.0

    def __init__(
        self,
        mesh_path: str,
        n_samples: int = 30000,
        seed: int = 42,
    ):
        verts, faces = load_obj_mesh(Path(mesh_path))
        rng = np.random.default_rng(seed)
        self.gt_samples = sample_mesh_uniformly(verts, faces, n_samples, rng)
        self._kd_tree = KDTree(self.gt_samples)

    @property
    def n_samples(self) -> int:
        return len(self.gt_samples)

    def compute_candidate_reward(
        self,
        cam_pos: np.ndarray,
        yaw: float,
        rock_center: np.ndarray,
        coverage_state: CoverageState,
        fx: float, fy: float, cx: float, cy: float,
        width: int, height: int,
    ) -> float:
        """
        Compute marginal GT coverage gain for a candidate viewpoint.

        Returns the fraction of previously-uncovered GT samples that
        become visible from this candidate's frustum.
        """
        # Compute gimbal pitch aimed at rock center
        delta = rock_center - cam_pos
        horiz_dist = math.sqrt(delta[0]**2 + delta[1]**2)
        gimbal_pitch = -math.atan2(delta[2], horiz_dist) if horiz_dist > 0.01 else -math.pi / 2

        R_cam = _build_camera_rotation(yaw, gimbal_pitch)

        # Only evaluate uncovered samples
        uncovered = coverage_state.uncovered_mask
        if not np.any(uncovered):
            return 0.0

        uncovered_pts = self.gt_samples[uncovered]
        vis_mask = _frustum_visible_mask(
            uncovered_pts, cam_pos, R_cam, fx, fy, cx, cy, width, height,
        )
        newly_visible = int(np.sum(vis_mask))
        return newly_visible / max(1, coverage_state.uncovered_count)

    def compute_candidate_reward_from_context(
        self,
        candidate_id: str,
        score: Any,
        context: Any,
        coverage_state: CoverageState,
        config: Dict,
    ) -> float:
        """
        Convenience method called from the hybrid scorer.

        Extracts candidate position from the ScoreBreakdown + ScoreContext,
        then delegates to compute_candidate_reward.
        """
        # Reconstruct candidate position from score terms
        # The hybrid scorer passes ScoreContext which has visited_viewpoints
        # but the candidate position is not directly in ScoreBreakdown.
        # We need to find it from the candidate pool via candidate_id.
        # For simplicity, use a fallback approach: extract from the score
        # context's colmap workspace.

        # The candidate's pose isn't stored in ScoreBreakdown.
        # We need to recover it. The simplest approach: store candidate
        # positions in the hybrid scorer and look up by ID.
        # For now, return 0.0 and let the hybrid scorer override this.
        return 0.0

    def compute_reward_for_candidates(
        self,
        candidates: List[Tuple[str, np.ndarray, float]],
        rock_center: np.ndarray,
        coverage_state: CoverageState,
        fx: float, fy: float, cx: float, cy: float,
        width: int, height: int,
    ) -> Dict[str, float]:
        """
        Compute marginal coverage reward for multiple candidates.

        candidates: list of (candidate_id, cam_pos_xyz, yaw)
        Returns: {candidate_id: reward}
        """
        rewards = {}
        for cand_id, cam_pos, yaw in candidates:
            rewards[cand_id] = self.compute_candidate_reward(
                cam_pos, yaw, rock_center, coverage_state,
                fx, fy, cx, cy, width, height,
            )
        return rewards

    def update_coverage_for_viewpoint(
        self,
        cam_pos: np.ndarray,
        yaw: float,
        rock_center: np.ndarray,
        coverage_state: CoverageState,
        fx: float, fy: float, cx: float, cy: float,
        width: int, height: int,
    ) -> int:
        """
        Mark GT samples visible from this viewpoint as covered.
        Returns count of newly covered samples.
        """
        delta = rock_center - cam_pos
        horiz_dist = math.sqrt(delta[0]**2 + delta[1]**2)
        gimbal_pitch = -math.atan2(delta[2], horiz_dist) if horiz_dist > 0.01 else -math.pi / 2
        R_cam = _build_camera_rotation(yaw, gimbal_pitch)

        vis_mask = _frustum_visible_mask(
            self.gt_samples, cam_pos, R_cam, fx, fy, cx, cy, width, height,
        )
        newly_covered = vis_mask & coverage_state.uncovered_mask
        count = int(np.sum(newly_covered))
        coverage_state.mark_covered(np.where(newly_covered)[0])
        return count
