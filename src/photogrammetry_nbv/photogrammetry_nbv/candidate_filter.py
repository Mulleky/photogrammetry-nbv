from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, List

from .contracts import CandidateViewpoint


def filter_candidates(
    candidates: Sequence[CandidateViewpoint],
    current_position_xyz: List[float],
    min_altitude_ned: float,
    max_altitude_ned: float,
    min_spacing_m: float,
    max_travel_m: Optional[float] = None,
) -> List[CandidateViewpoint]:
    survivors: List[CandidateViewpoint] = []
    for cand in candidates:
        flags: Dict[str, bool] = {}
        flags['altitude_ok'] = max_altitude_ned <= cand.z <= min_altitude_ned
        travel = _distance(current_position_xyz, [cand.x, cand.y, cand.z])
        flags['travel_ok'] = True if max_travel_m is None else travel <= max_travel_m

        if all(flags.values()):
            cand.feasibility_flags.update(flags)
            survivors.append(cand)
        else:
            cand.feasibility_flags.update(flags)
    return farthest_point_downselect(survivors, max_count=None, min_spacing_m=min_spacing_m)


def farthest_point_downselect(
    candidates: Sequence[CandidateViewpoint],
    max_count: Optional[int],
    min_spacing_m: float,
) -> List[CandidateViewpoint]:
    if not candidates:
        return []
    ordered = list(candidates)
    selected = [ordered[0]]
    for cand in ordered[1:]:
        if all(_distance([cand.x, cand.y, cand.z], [s.x, s.y, s.z]) >= min_spacing_m for s in selected):
            selected.append(cand)
        if max_count is not None and len(selected) >= max_count:
            break
    return selected


def crop_to_target_count_diverse(
    candidates: Sequence[CandidateViewpoint],
    target_count: int,
) -> List[CandidateViewpoint]:
    if len(candidates) <= target_count:
        return list(candidates)
    selected = [candidates[0]]
    remaining = list(candidates[1:])
    while remaining and len(selected) < target_count:
        best = None
        best_dist = -1.0
        for cand in remaining:
            d = min(_distance([cand.x, cand.y, cand.z], [s.x, s.y, s.z]) for s in selected)
            if d > best_dist:
                best = cand
                best_dist = d
        selected.append(best)
        remaining.remove(best)
    return selected


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))
