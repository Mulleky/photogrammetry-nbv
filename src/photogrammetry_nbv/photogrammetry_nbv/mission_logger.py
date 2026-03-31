from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence

from .contracts import CandidateViewpoint, ScoreBreakdown, SparseMetricsSnapshot


class MissionLogger:
    def __init__(self, candidate_dir: Path, metrics_dir: Path) -> None:
        self.candidate_dir = candidate_dir
        self.metrics_dir = metrics_dir

    def log_candidates(self, iteration: int, payload: Sequence[CandidateViewpoint]) -> Path:
        path = self.candidate_dir / f'candidate_pool_iter_{iteration:02d}.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump([c.to_dict() for c in payload], f, indent=2)
        return path

    def log_scores(self, iteration: int, payload: Sequence[ScoreBreakdown]) -> Path:
        path = self.candidate_dir / f'candidate_scores_iter_{iteration:02d}.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump([s.to_dict() for s in payload], f, indent=2)
        return path

    def log_selected(self, iteration: int, selected_payload: Dict[str, Any]) -> Path:
        path = self.candidate_dir / f'selected_candidate_iter_{iteration:02d}.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(selected_payload, f, indent=2)
        return path

    def log_sparse_metrics(self, iteration: int, snapshot: SparseMetricsSnapshot) -> Path:
        path = self.metrics_dir / f'sparse_metrics_iter_{iteration:02d}.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(snapshot.to_dict(), f, indent=2)
        return path
