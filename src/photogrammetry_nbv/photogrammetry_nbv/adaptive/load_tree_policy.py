from __future__ import annotations

import json
from typing import Any, Dict, List


class TreePolicy:
    """
    Runtime decision-tree policy loaded from a JSON file.

    No sklearn dependency -- just walks the tree structure produced by
    train_gt_phase_switch.py.

    JSON format:
    {
        "type": "decision_tree",
        "feature_names": ["budget_fraction", "mean_track_length", ...],
        "tree": {
            "feature": "budget_fraction",
            "threshold": 0.4,
            "left": {"leaf": "coverage"},
            "right": {
                "feature": "percentile_knn_distance",
                "threshold": 0.015,
                "left": {"leaf": "geometry"},
                "right": {"leaf": "coverage"}
            }
        }
    }
    """

    def __init__(self, json_path: str):
        with open(json_path) as f:
            data = json.load(f)
        self.tree: Dict[str, Any] = data['tree']
        self.feature_names: List[str] = data.get('feature_names', [])

    def predict(self, features: Dict[str, float]) -> str:
        """Walk the tree and return 'coverage' or 'geometry'."""
        node = self.tree
        while 'leaf' not in node:
            val = features.get(node['feature'], 0.0)
            node = node['left'] if val <= node['threshold'] else node['right']
        return node['leaf']
