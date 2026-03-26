from __future__ import annotations

import math
from typing import Dict, List, Sequence

from ..contracts import CandidateViewpoint, ScoreBreakdown, ScoreContext, SparseMetricsSnapshot, WeakRegion
from ..scoring_interface import BaseScorer


class WeightedSumScorer(BaseScorer):
    """
    Default sparse-cloud-driven scorer.

    It does not maximize coverage. Instead, it estimates candidate utility from:
      - weak-region support from sparse-point diagnostics,
      - view novelty relative to visited viewpoints,
      - movement cost from the current pose,
      - revisit redundancy.
    """

    def score_candidates(
        self,
        candidates: Sequence[CandidateViewpoint],
        sparse_metrics: SparseMetricsSnapshot,
        context: ScoreContext,
    ) -> List[ScoreBreakdown]:
        weights = self.config.get('weights', {})
        out: List[ScoreBreakdown] = []
        for cand in candidates:
            weak_support = self._weak_region_support(cand, sparse_metrics.weak_regions)
            novelty = self._novelty(cand, context.visited_viewpoints)
            movement_cost = self._movement_cost(cand, context.current_position_xyz, context.current_yaw_rad)
            revisit_penalty = self._revisit_penalty(cand, context.visited_viewpoints)

            final = (
                weights.get('weak_region_support', 1.0) * weak_support
                + weights.get('novelty', 0.5) * novelty
                - weights.get('movement_cost', 0.3) * movement_cost
                - weights.get('revisit_penalty', 0.25) * revisit_penalty
            )

            out.append(
                ScoreBreakdown(
                    candidate_id=cand.candidate_id,
                    final_score=float(final),
                    terms={
                        'weak_region_support': float(weak_support),
                        'novelty': float(novelty),
                        'movement_cost': float(movement_cost),
                        'revisit_penalty': float(revisit_penalty),
                    },
                    weights={
                        'weak_region_support': float(weights.get('weak_region_support', 1.0)),
                        'novelty': float(weights.get('novelty', 0.5)),
                        'movement_cost': float(weights.get('movement_cost', 0.3)),
                        'revisit_penalty': float(weights.get('revisit_penalty', 0.25)),
                    },
                    scorer_name='weighted_sum',
                )
            )
        return sorted(out, key=lambda s: s.final_score, reverse=True)

    def _weak_region_support(self, cand: CandidateViewpoint, weak_regions: Sequence[WeakRegion]) -> float:
        if not weak_regions:
            return 0.0
        total = 0.0
        for region in weak_regions:
            rx, ry, rz = region.centroid_xyz
            dcx = rx - cand.x
            dcy = ry - cand.y
            dcz = rz - cand.z
            dist = math.sqrt(dcx * dcx + dcy * dcy + dcz * dcz)
            az_gain = max(0.0, math.cos(self._bearing_error(cand, rx, ry)))
            proximity = 1.0 / max(1.0, dist)
            total += region.severity * (0.7 * az_gain + 0.3 * proximity)
        return total / float(len(weak_regions))

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
        return min(1.0, min(dists) / max(1.0, self.config.get('novelty_distance_scale_m', 5.0)))

    def _movement_cost(self, cand: CandidateViewpoint, current_xyz: Sequence[float], current_yaw_rad: float) -> float:
        dx = cand.x - float(current_xyz[0])
        dy = cand.y - float(current_xyz[1])
        dz = cand.z - float(current_xyz[2])
        travel = math.sqrt(dx * dx + dy * dy + dz * dz)
        yaw_cost = abs(_wrap_pi(cand.yaw - current_yaw_rad))
        travel_scale = max(1.0, float(self.config.get('travel_distance_scale_m', 10.0)))
        yaw_scale = max(0.1, float(self.config.get('yaw_change_scale_rad', math.pi)))
        return 0.7 * (travel / travel_scale) + 0.3 * (yaw_cost / yaw_scale)

    def _revisit_penalty(self, cand: CandidateViewpoint, visited: Sequence[Dict]) -> float:
        if not visited:
            return 0.0
        penalties = []
        for vp in visited:
            yaw = float(vp.get('yaw_rad', cand.yaw))
            pos = vp.get('position_ned_m', {})
            dx = cand.x - float(pos.get('x', cand.x))
            dy = cand.y - float(pos.get('y', cand.y))
            dz = cand.z - float(pos.get('z', cand.z))
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            yaw_gap = abs(_wrap_pi(cand.yaw - yaw))
            penalties.append(max(0.0, 1.0 - dist / 5.0) * max(0.0, 1.0 - yaw_gap / (math.pi / 2.0)))
        return max(penalties) if penalties else 0.0

    def _bearing_error(self, cand: CandidateViewpoint, target_x: float, target_y: float) -> float:
        desired = math.atan2(target_y - cand.y, target_x - cand.x)
        return abs(_wrap_pi(desired - cand.yaw))


def _wrap_pi(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle
