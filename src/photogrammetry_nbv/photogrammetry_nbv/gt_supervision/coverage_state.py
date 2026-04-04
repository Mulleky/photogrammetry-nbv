from __future__ import annotations

from typing import Dict

import numpy as np


class CoverageState:
    """Tracks cumulative coverage of GT surface samples across NBV iterations."""

    def __init__(self, n_samples: int, coverage_threshold_m: float = 0.02):
        self.covered = np.zeros(n_samples, dtype=bool)
        self.threshold = coverage_threshold_m

    def mark_covered(self, indices: np.ndarray) -> None:
        self.covered[indices] = True

    @property
    def n_samples(self) -> int:
        return len(self.covered)

    @property
    def coverage_fraction(self) -> float:
        return float(np.mean(self.covered))

    @property
    def uncovered_mask(self) -> np.ndarray:
        return ~self.covered

    @property
    def uncovered_count(self) -> int:
        return int(np.sum(~self.covered))

    def to_dict(self) -> Dict:
        return {
            'total_samples': self.n_samples,
            'covered_count': int(np.sum(self.covered)),
            'coverage_fraction': self.coverage_fraction,
        }
