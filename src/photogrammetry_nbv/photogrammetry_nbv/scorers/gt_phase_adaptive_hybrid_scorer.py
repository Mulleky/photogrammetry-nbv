from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..contracts import CandidateViewpoint, ScoreBreakdown, ScoreContext, SparseMetricsSnapshot
from ..scoring_interface import BaseScorer
from .covisibility_scorer import CovisibilityScorer
from .baseline_aware_repair_weighted_covisibility_scorer import (
    BaselineAwareRepairWeightedCovisibilityScorer,
)


class GTPhaseAdaptiveHybridScorer(BaseScorer):
    """
    Wrapper scorer that runs both CovisibilityScorer and
    BaselineAwareRepairWeightedCovisibilityScorer on every candidate set,
    then uses a gate to select ONE scorer's full ranking as the output.

    Hard switch: no score blending.  The selected scorer's ranking is
    returned unchanged.

    Gate modes:
      - shadow:         Always use coverage scorer, log both rankings.
      - heuristic:      Budget-fraction / kNN threshold rule.
      - oracle:         GT mesh supervision picks the better top-1 candidate.
      - learned_switch: JSON decision-tree policy selects the scorer.
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self._coverage_scorer = CovisibilityScorer(config)
        self._geometry_scorer = BaselineAwareRepairWeightedCovisibilityScorer(config)

        hybrid_cfg = config.get('hybrid', {})
        self._gate_mode: str = hybrid_cfg.get('gate_mode', 'shadow')
        self._hybrid_cfg = hybrid_cfg

        # Heuristic params
        heur = hybrid_cfg.get('heuristic', {})
        self._heur_budget_threshold: float = float(heur.get('budget_fraction_threshold', 0.4))
        self._heur_knn_threshold: float = float(heur.get('knn_p95_threshold', 0.14))

        # Shadow logging
        shadow_cfg = hybrid_cfg.get('shadow', {})
        self._shadow_enabled: bool = bool(shadow_cfg.get('enabled', True))
        self._shadow_top_k: int = int(shadow_cfg.get('log_top_k', 5))

        # Lazy-init fields for oracle and learned modes
        self._mesh_oracle: Any = None
        self._coverage_state: Any = None
        self._tree_policy: Any = None

        # Load learned policy if configured
        if self._gate_mode == 'learned_switch':
            policy_path = hybrid_cfg.get('learned_switch', {}).get('policy_json_path', '')
            if policy_path:
                from ..adaptive.load_tree_policy import TreePolicy
                self._tree_policy = TreePolicy(policy_path)

        # Init oracle if configured
        if self._gate_mode == 'oracle':
            self._init_oracle()

    def _init_oracle(self) -> None:
        oracle_cfg = self._hybrid_cfg.get('oracle', {})
        mesh_path = oracle_cfg.get('gt_mesh_path', '')
        if not mesh_path:
            return
        from ..gt_supervision.mesh_oracle import MeshOracle
        from ..gt_supervision.coverage_state import CoverageState
        n_samples = int(oracle_cfg.get('n_gt_samples', 30000))
        self._mesh_oracle = MeshOracle(
            mesh_path=mesh_path,
            n_samples=n_samples,
        )
        self._coverage_state = CoverageState(
            n_samples=self._mesh_oracle.n_samples,
            coverage_threshold_m=float(oracle_cfg.get('coverage_threshold_m', 0.02)),
        )

    # ------------------------------------------------------------------
    # Main scoring entry point
    # ------------------------------------------------------------------

    def score_candidates(
        self,
        candidates: Sequence[CandidateViewpoint],
        sparse_metrics: SparseMetricsSnapshot,
        context: ScoreContext,
    ) -> List[ScoreBreakdown]:
        # Build candidate lookup for oracle gate
        self._cand_lookup: Dict[str, CandidateViewpoint] = {
            c.candidate_id: c for c in candidates
        }

        # Lazily update oracle coverage state from previously visited viewpoints
        if self._mesh_oracle is not None and self._coverage_state is not None:
            self._update_oracle_coverage(context)

        # 1. Run both scorers
        scores_cov = self._coverage_scorer.score_candidates(candidates, sparse_metrics, context)
        scores_geo = self._geometry_scorer.score_candidates(candidates, sparse_metrics, context)

        # 2. Extract gate features
        features = self._extract_gate_features(sparse_metrics, context, scores_cov, scores_geo)

        # 3. Gate decision
        choice = self._gate(features, scores_cov, scores_geo, context)

        # 4. Shadow logging
        oracle_reward_cov: Optional[float] = None
        oracle_reward_geo: Optional[float] = None
        if self._gate_mode == 'oracle' and self._mesh_oracle is not None:
            oracle_reward_cov = features.get('_oracle_reward_cov')
            oracle_reward_geo = features.get('_oracle_reward_geo')

        if self._shadow_enabled:
            self._write_shadow_log(
                context, features, choice,
                scores_cov, scores_geo,
                oracle_reward_cov, oracle_reward_geo,
            )

        # 5. Return chosen ranking unchanged
        if choice == 'coverage':
            return scores_cov
        else:
            return scores_geo

    def _update_oracle_coverage(self, context: ScoreContext) -> None:
        """Lazily mark GT samples as covered for all visited viewpoints."""
        n_visited = len(context.visited_viewpoints)
        if not hasattr(self, '_oracle_coverage_synced_to'):
            self._oracle_coverage_synced_to = 0

        if self._oracle_coverage_synced_to >= n_visited:
            return

        rock = np.array(context.rock_center_xyz, dtype=np.float64)
        for vp in context.visited_viewpoints[self._oracle_coverage_synced_to:]:
            pos = vp.get('position_ned_m', {})
            if 'x' not in pos:
                continue
            cam_pos = np.array([pos['x'], pos['y'], pos['z']], dtype=np.float64)
            yaw = float(vp.get('yaw_rad', 0.0))
            self._mesh_oracle.update_coverage_for_viewpoint(
                cam_pos, yaw, rock, self._coverage_state,
                self._mesh_oracle.DEFAULT_FX, self._mesh_oracle.DEFAULT_FY,
                self._mesh_oracle.DEFAULT_WIDTH / 2, self._mesh_oracle.DEFAULT_HEIGHT / 2,
                self._mesh_oracle.DEFAULT_WIDTH, self._mesh_oracle.DEFAULT_HEIGHT,
            )
        self._oracle_coverage_synced_to = n_visited

    # ------------------------------------------------------------------
    # Gate feature extraction
    # ------------------------------------------------------------------

    def _extract_gate_features(
        self,
        sm: SparseMetricsSnapshot,
        ctx: ScoreContext,
        scores_a: List[ScoreBreakdown],
        scores_b: List[ScoreBreakdown],
    ) -> Dict[str, float]:
        budget_frac = ctx.images_used / max(1, ctx.image_budget)

        top_a = scores_a[0].final_score if scores_a else 0.0
        top_b = scores_b[0].final_score if scores_b else 0.0
        margin_a = (scores_a[0].final_score - scores_a[1].final_score) if len(scores_a) >= 2 else 0.0
        margin_b = (scores_b[0].final_score - scores_b[1].final_score) if len(scores_b) >= 2 else 0.0
        top_agree = float(
            scores_a[0].candidate_id == scores_b[0].candidate_id
        ) if scores_a and scores_b else 0.0

        return {
            'iteration': float(sm.iteration),
            'budget_fraction': budget_frac,
            'sparse_point_count': float(sm.sparse_point_count),
            'mean_track_length': sm.global_metrics.get('mean_track_length', 0.0),
            'mean_reprojection_error': sm.global_metrics.get('mean_reprojection_error', 0.0),
            'mean_knn_distance': sm.knn_distance_metrics.get('mean_knn_distance', 0.0),
            'percentile_knn_distance': sm.knn_distance_metrics.get('percentile_knn_distance', 0.0),
            'n_visited': float(len(ctx.visited_viewpoints)),
            'top_score_cov': top_a,
            'top_score_geo': top_b,
            'top_margin_cov': margin_a,
            'top_margin_geo': margin_b,
            'top_pick_agrees': top_agree,
        }

    # ------------------------------------------------------------------
    # Gate logic
    # ------------------------------------------------------------------

    def _gate(
        self,
        features: Dict[str, float],
        scores_cov: List[ScoreBreakdown],
        scores_geo: List[ScoreBreakdown],
        context: ScoreContext,
    ) -> str:
        """Return 'coverage' or 'geometry'."""
        if self._gate_mode == 'shadow':
            return 'coverage'

        if self._gate_mode == 'heuristic':
            return self._heuristic_gate(features)

        if self._gate_mode == 'oracle':
            return self._oracle_gate(features, scores_cov, scores_geo, context)

        if self._gate_mode == 'learned_switch':
            return self._learned_gate(features)

        # Fallback
        return 'coverage'

    def _heuristic_gate(self, features: Dict[str, float]) -> str:
        budget_frac = features['budget_fraction']
        knn_p95 = features['percentile_knn_distance']

        if budget_frac < self._heur_budget_threshold or knn_p95 > self._heur_knn_threshold:
            return 'geometry'
        return 'coverage'

    def _oracle_gate(
        self,
        features: Dict[str, float],
        scores_cov: List[ScoreBreakdown],
        scores_geo: List[ScoreBreakdown],
        context: ScoreContext,
    ) -> str:
        if self._mesh_oracle is None or not scores_cov or not scores_geo:
            return 'coverage'

        rock = np.array(context.rock_center_xyz, dtype=np.float64)
        fx = self._mesh_oracle.DEFAULT_FX
        fy = self._mesh_oracle.DEFAULT_FY
        cx = self._mesh_oracle.DEFAULT_WIDTH / 2
        cy = self._mesh_oracle.DEFAULT_HEIGHT / 2
        w = self._mesh_oracle.DEFAULT_WIDTH
        h = self._mesh_oracle.DEFAULT_HEIGHT

        def _reward_for(cand_id: str) -> float:
            cand = self._cand_lookup.get(cand_id)
            if cand is None:
                return 0.0
            cam_pos = np.array([cand.x, cand.y, cand.z], dtype=np.float64)
            return self._mesh_oracle.compute_candidate_reward(
                cam_pos, cand.yaw, rock, self._coverage_state,
                fx, fy, cx, cy, w, h,
            )

        reward_cov = _reward_for(scores_cov[0].candidate_id)
        reward_geo = _reward_for(scores_geo[0].candidate_id)

        features['_oracle_reward_cov'] = reward_cov
        features['_oracle_reward_geo'] = reward_geo

        return 'geometry' if reward_geo > reward_cov else 'coverage'

    def _learned_gate(self, features: Dict[str, float]) -> str:
        if self._tree_policy is None:
            return 'coverage'
        return self._tree_policy.predict(features)

    # ------------------------------------------------------------------
    # Shadow logging
    # ------------------------------------------------------------------

    def _write_shadow_log(
        self,
        context: ScoreContext,
        features: Dict[str, float],
        choice: str,
        scores_cov: List[ScoreBreakdown],
        scores_geo: List[ScoreBreakdown],
        oracle_reward_cov: Optional[float],
        oracle_reward_geo: Optional[float],
    ) -> None:
        workspace = Path(context.colmap_workspace) if context.colmap_workspace else None
        if workspace is None:
            return
        run_dir = workspace.parent
        shadow_dir = run_dir / 'candidates' / 'hybrid_shadow'
        shadow_dir.mkdir(parents=True, exist_ok=True)

        # Strip internal keys from features for logging
        log_features = {k: v for k, v in features.items() if not k.startswith('_')}

        k = self._shadow_top_k
        record = {
            'iteration': context.current_iteration,
            'gate_mode': self._gate_mode,
            'gate_choice': choice,
            'features': log_features,
            'scorer_cov_top': [s.to_dict() for s in scores_cov[:k]],
            'scorer_geo_top': [s.to_dict() for s in scores_geo[:k]],
            'oracle_reward_cov': oracle_reward_cov,
            'oracle_reward_geo': oracle_reward_geo,
            'timestamp_ns': int(time.time_ns()),
        }

        path = shadow_dir / f'shadow_iter_{context.current_iteration:03d}.json'
        with open(path, 'w') as f:
            json.dump(record, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def _zero_scores(self, candidates: Sequence[CandidateViewpoint]) -> List[ScoreBreakdown]:
        return [
            ScoreBreakdown(
                candidate_id=c.candidate_id, final_score=0.0,
                terms={}, weights={},
                scorer_name='gt_phase_adaptive_hybrid',
            )
            for c in candidates
        ]
