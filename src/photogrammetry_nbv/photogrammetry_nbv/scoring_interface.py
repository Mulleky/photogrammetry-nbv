from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Sequence

from .contracts import CandidateViewpoint, ScoreBreakdown, ScoreContext, SparseMetricsSnapshot


class BaseScorer(ABC):
    """
    Stable interface for candidate scoring.

    New scoring functions should subclass BaseScorer and return one ScoreBreakdown
    per candidate. The controller only depends on this contract, so you can replace
    the internal score function without touching the flight or COLMAP code.
    """

    def __init__(self, config: Dict):
        self.config = config

    @abstractmethod
    def score_candidates(
        self,
        candidates: Sequence[CandidateViewpoint],
        sparse_metrics: SparseMetricsSnapshot,
        context: ScoreContext,
    ) -> List[ScoreBreakdown]:
        raise NotImplementedError
