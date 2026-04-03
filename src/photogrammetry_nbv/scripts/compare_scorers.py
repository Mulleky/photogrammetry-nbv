#!/usr/bin/env python3
"""
compare_scorers.py — Compare dense-cloud evaluation metrics across scorer runs.

Reads eval/report.json from each run directory and produces four separate plots:
  1. F-score vs distance threshold
  2. Completeness vs distance threshold
  3. Accuracy vs distance threshold
  4. Cloud-to-cloud distance (bar chart: mean, median, p95)

Usage:
    python3 compare_scorers.py \
        <run_dir_1> <run_dir_2> <run_dir_3>

    # Optionally override output directory:
    python3 compare_scorers.py ... --output-dir /some/path
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ── Scorer display names ─────────────────────────────────────────────────────

_SCORER_LABELS = {
    'covisibility': 'Co-visibility',
    'repair_weighted_covisibility': 'Repair-weighted',
    'baseline_aware_repair_weighted_covisibility': 'Baseline-aware',
}

_SCORER_COLORS = {
    'covisibility': '#1976D2',
    'repair_weighted_covisibility': '#E64A19',
    'baseline_aware_repair_weighted_covisibility': '#388E3C',
}

_SCORER_MARKERS = {
    'covisibility': 'o',
    'repair_weighted_covisibility': 's',
    'baseline_aware_repair_weighted_covisibility': '^',
}


def _label(name: str) -> str:
    return _SCORER_LABELS.get(name, name)


def _color(name: str) -> str:
    return _SCORER_COLORS.get(name, '#757575')


def _marker(name: str) -> str:
    return _SCORER_MARKERS.get(name, 'D')


# ── Data loading ──────────────────────────────────────────────────────────────

def load_run(run_dir: Path) -> Tuple[str, Dict]:
    """Load report.json and return (scorer_name, nbv_cloud_data)."""
    report_path = run_dir / 'eval' / 'report.json'
    if not report_path.exists():
        sys.exit(f'ERROR: {report_path} not found')

    report = json.loads(report_path.read_text())
    scorer_name = report.get('mission_params', {}).get('scorer_name', run_dir.name)

    nbv = report.get('clouds', {}).get('nbv', {})
    if 'metrics' not in nbv:
        sys.exit(f'ERROR: no dense cloud metrics in {report_path}')

    return scorer_name, nbv


def extract_threshold_metrics(
    metrics: Dict, prefix: str
) -> Tuple[List[float], List[float]]:
    """Extract (thresholds_mm, values) for a given metric prefix."""
    pairs = []
    for key, val in metrics.items():
        m = re.match(rf'{prefix}_(\d+)mm', key)
        if m:
            pairs.append((int(m.group(1)), float(val)))
    pairs.sort(key=lambda p: p[0])
    return [p[0] for p in pairs], [p[1] for p in pairs]


# ── Plotting ──────────────────────────────────────────────────────────────────

def _style_threshold_ax(ax, ylabel: str, thresholds_mm: List[float]) -> None:
    ax.set_xlabel('Distance threshold (mm)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(thresholds_mm)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)


def plot_threshold_metric(
    runs: List[Tuple[str, Dict]],
    metric_prefix: str,
    title: str,
    ylabel: str,
    save_path: Path,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))

    for scorer_name, nbv in runs:
        thresholds_mm, values = extract_threshold_metrics(
            nbv['metrics'], metric_prefix
        )
        ax.plot(
            thresholds_mm, values,
            marker=_marker(scorer_name),
            color=_color(scorer_name),
            label=_label(scorer_name),
            linewidth=2, markersize=8,
        )

    ax.set_title(title, fontsize=14, fontweight='bold')
    _style_threshold_ax(ax, ylabel, thresholds_mm)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f'  Saved: {save_path}')
    return fig


def plot_c2c_distances(
    runs: List[Tuple[str, Dict]],
    save_path: Path,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))

    scorer_names = [r[0] for r in runs]
    labels = [_label(n) for n in scorer_names]
    colors = [_color(n) for n in scorer_names]

    c2c_keys = ['mean_c2c_m', 'median_c2c_m', 'p95_c2c_m']
    c2c_labels = ['Mean', 'Median', '95th percentile']
    x = np.arange(len(c2c_labels))
    width = 0.25

    for i, (scorer_name, nbv) in enumerate(runs):
        metrics = nbv['metrics']
        vals = [metrics.get(k, 0.0) * 1000 for k in c2c_keys]  # convert to mm
        bars = ax.bar(
            x + i * width, vals, width,
            label=_label(scorer_name),
            color=_color(scorer_name),
            alpha=0.85,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val:.1f}',
                ha='center', va='bottom', fontsize=9,
            )

    ax.set_title('Cloud-to-Cloud Distance', fontsize=14, fontweight='bold')
    ax.set_xlabel('Statistic', fontsize=12)
    ax.set_ylabel('Distance (mm)', fontsize=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels(c2c_labels, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f'  Saved: {save_path}')
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description='Compare dense-cloud metrics across scorer runs'
    )
    ap.add_argument(
        'run_dirs', nargs=3, type=str,
        help='Paths to three unified_run_* directories (one per scorer)'
    )
    ap.add_argument(
        '--output-dir', type=str,
        default='/home/dreamslab/Desktop/results/final results',
        help='Directory to save plots'
    )
    ap.add_argument(
        '--no-show', action='store_true',
        help='Save figures without displaying windows'
    )
    args = ap.parse_args()

    if args.no_show:
        import matplotlib as _mpl
        _mpl.use('Agg')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all runs
    runs: List[Tuple[str, Dict]] = []
    for rd in args.run_dirs:
        run_path = Path(rd).expanduser().resolve()
        if not run_path.exists():
            sys.exit(f'ERROR: run directory not found: {run_path}')
        scorer_name, nbv = load_run(run_path)
        runs.append((scorer_name, nbv))
        print(f'Loaded: {_label(scorer_name)} <- {run_path.name}')

    print()

    # Plot 1: F-score
    plot_threshold_metric(
        runs, 'fscore', 'F-score vs Distance Threshold',
        'F-score (%)', output_dir / 'fscore_comparison.png',
    )

    # Plot 2: Completeness
    plot_threshold_metric(
        runs, 'completeness', 'Completeness vs Distance Threshold',
        'Completeness (%)', output_dir / 'completeness_comparison.png',
    )

    # Plot 3: Accuracy
    plot_threshold_metric(
        runs, 'accuracy', 'Accuracy vs Distance Threshold',
        'Accuracy (%)', output_dir / 'accuracy_comparison.png',
    )

    # Plot 4: Cloud-to-cloud distance
    plot_c2c_distances(
        runs, output_dir / 'c2c_distance_comparison.png',
    )

    print(f'\nAll plots saved to: {output_dir}')

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
