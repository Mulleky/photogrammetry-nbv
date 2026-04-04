#!/usr/bin/env python3
"""
train_gt_phase_switch.py -- Offline training script for the GT phase-adaptive
hybrid scorer's learned gate.

Reads shadow logs (with oracle rewards) from completed mission runs, trains a
shallow decision tree, and exports it to JSON for runtime use.

Usage:
    python3 -m photogrammetry_nbv.adaptive.train_gt_phase_switch \
        /path/to/run1/candidates/hybrid_shadow \
        /path/to/run2/candidates/hybrid_shadow \
        --output models/gt_phase_switch_tree.json \
        --max-depth 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


# Features used for training (must match what the hybrid scorer logs)
GATE_FEATURES = [
    'budget_fraction',
    'sparse_point_count',
    'mean_track_length',
    'mean_reprojection_error',
    'mean_knn_distance',
    'percentile_knn_distance',
    'n_visited',
    'top_score_cov',
    'top_score_geo',
    'top_margin_cov',
    'top_margin_geo',
    'top_pick_agrees',
]


def load_shadow_logs(dirs: List[Path]) -> List[Dict]:
    """Load all shadow log JSONs from the given directories."""
    records = []
    for d in dirs:
        if not d.is_dir():
            print(f'Warning: {d} is not a directory, skipping', file=sys.stderr)
            continue
        for f in sorted(d.glob('shadow_iter_*.json')):
            with open(f) as fh:
                record = json.load(fh)
            # Only use records that have oracle rewards
            if record.get('oracle_reward_cov') is not None and record.get('oracle_reward_geo') is not None:
                records.append(record)
    return records


def build_dataset(records: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build X (features) and y (labels) arrays from shadow log records.

    Label: 0 = coverage scorer had higher oracle reward
           1 = geometry scorer had higher oracle reward
    """
    X_rows = []
    y_rows = []
    for rec in records:
        feats = rec.get('features', {})
        row = [float(feats.get(f, 0.0)) for f in GATE_FEATURES]
        X_rows.append(row)

        r_cov = float(rec['oracle_reward_cov'])
        r_geo = float(rec['oracle_reward_geo'])
        y_rows.append(1 if r_geo > r_cov else 0)

    return np.array(X_rows), np.array(y_rows), GATE_FEATURES


def train_tree(X: np.ndarray, y: np.ndarray, max_depth: int = 4) -> Any:
    """Train a sklearn DecisionTreeClassifier."""
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X, y)
    return clf


def sklearn_tree_to_json(clf: Any, feature_names: List[str]) -> Dict:
    """Convert a fitted sklearn DecisionTreeClassifier to a nested JSON dict."""
    tree = clf.tree_
    label_map = {0: 'coverage', 1: 'geometry'}

    def _recurse(node_id: int) -> Dict:
        if tree.children_left[node_id] == tree.children_right[node_id]:
            # Leaf node
            class_idx = int(np.argmax(tree.value[node_id][0]))
            return {'leaf': label_map[class_idx]}
        return {
            'feature': feature_names[tree.feature[node_id]],
            'threshold': float(tree.threshold[node_id]),
            'left': _recurse(tree.children_left[node_id]),
            'right': _recurse(tree.children_right[node_id]),
        }

    return _recurse(0)


def export_policy(clf: Any, feature_names: List[str], output_path: Path) -> None:
    """Export the trained tree as a JSON policy file."""
    tree_dict = sklearn_tree_to_json(clf, feature_names)
    policy = {
        'type': 'decision_tree',
        'feature_names': feature_names,
        'tree': tree_dict,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(policy, f, indent=2)
    print(f'Policy exported to {output_path}')


def print_diagnostics(clf: Any, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> None:
    """Print training diagnostics."""
    from sklearn.metrics import classification_report, confusion_matrix

    y_pred = clf.predict(X)
    print('\n=== Training Diagnostics ===')
    print(f'Samples: {len(y)}')
    print(f'Coverage labels (0): {int(np.sum(y == 0))}')
    print(f'Geometry labels (1): {int(np.sum(y == 1))}')
    print(f'Training accuracy: {float(np.mean(y_pred == y)):.3f}')

    print('\nConfusion matrix (rows=true, cols=pred):')
    print(confusion_matrix(y, y_pred))

    print('\nClassification report:')
    present = sorted(set(y) | set(y_pred))
    names = ['coverage', 'geometry']
    print(classification_report(y, y_pred, labels=present, target_names=[names[i] for i in present]))

    importances = clf.feature_importances_
    print('Feature importance:')
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        if imp > 0:
            print(f'  {name}: {imp:.3f}')


def main():
    parser = argparse.ArgumentParser(
        description='Train GT phase-adaptive switch policy from shadow logs',
    )
    parser.add_argument(
        'shadow_dirs', nargs='+', type=Path,
        help='Directories containing hybrid_shadow/ log JSONs',
    )
    parser.add_argument(
        '--output', '-o', type=Path,
        default=Path('models/gt_phase_switch_tree.json'),
        help='Output path for the JSON policy file',
    )
    parser.add_argument(
        '--max-depth', type=int, default=4,
        help='Maximum depth of the decision tree',
    )
    args = parser.parse_args()

    records = load_shadow_logs(args.shadow_dirs)
    if not records:
        print('No shadow logs with oracle rewards found. Run missions with '
              'gate_mode=oracle first.', file=sys.stderr)
        sys.exit(1)

    print(f'Loaded {len(records)} shadow log records')
    X, y, feature_names = build_dataset(records)

    if len(np.unique(y)) < 2:
        print('Warning: only one class present in labels. The tree will be trivial.',
              file=sys.stderr)

    clf = train_tree(X, y, max_depth=args.max_depth)
    print_diagnostics(clf, X, y, feature_names)
    export_policy(clf, feature_names, args.output)


if __name__ == '__main__':
    main()
