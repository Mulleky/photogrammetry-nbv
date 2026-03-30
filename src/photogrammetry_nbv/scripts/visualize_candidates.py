#!/usr/bin/env python3
"""
Standalone 3D visualizer for the NBV candidate pipeline.

Renders four layers:
  1. Raw Fibonacci hemisphere candidates
  2. Feasibility-filtered survivors
  3. Diversity-cropped final pool
  4. Top-scored candidates

Plus a rock-centre marker. Legend entries toggle visibility on click.

Usage:
    python visualize_candidates.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Allow importing from the package without installing
_PACKAGE_DIR = Path(__file__).resolve().parent.parent / 'photogrammetry_nbv'
sys.path.insert(0, str(_PACKAGE_DIR.parent))

from photogrammetry_nbv.candidate_generator import generate_fibonacci_hemisphere
from photogrammetry_nbv.candidate_filter import filter_candidates, crop_to_target_count_diverse
from photogrammetry_nbv.contracts import ScoreContext
from photogrammetry_nbv.scorers import SCORER_REGISTRY

import yaml


def main() -> None:
    # ---- Config (edit as needed or load from YAML) ----
    rock_center = [0.0, 0.0, 0.0]
    radius = 5.5
    raw_count = 200
    target_count = 95
    min_elev = 10.0
    max_elev = 45.0
    min_alt_ned = -20.0
    max_alt_ned = -0.1
    min_spacing = 1.0
    max_travel = 30.0
    current_pos = [5.0, 0.0, -6.0]
    top_n = 10

    # ---- Pipeline ----
    raw = generate_fibonacci_hemisphere(
        center_xyz=rock_center, radius=radius, candidate_count=raw_count,
        min_elevation_deg=min_elev, max_elevation_deg=max_elev,
    )
    feasible = filter_candidates(raw, current_pos, min_alt_ned, max_alt_ned, min_spacing, max_travel)
    final_pool = crop_to_target_count_diverse(feasible, target_count)

    # ---- Scoring (optional — needs scoring.yaml) ----
    scoring_yaml = Path(__file__).resolve().parent.parent / 'config' / 'scoring.yaml'
    scored_ids = []
    if scoring_yaml.exists():
        with open(scoring_yaml) as f:
            scoring_cfg = yaml.safe_load(f)
        scorer_cls = SCORER_REGISTRY[scoring_cfg['scorer']['name']]
        scorer = scorer_cls(scoring_cfg['scorer'])
        # Minimal dummy snapshot
        from photogrammetry_nbv.contracts import SparseMetricsSnapshot, WeakRegion
        dummy_snapshot = SparseMetricsSnapshot(
            iteration=0, total_cameras=12, aligned_cameras=12, sparse_point_count=500,
            global_metrics={}, weak_regions=[
                WeakRegion('r0', rock_center, 1.0, 100, {}),
            ],
        )
        context = ScoreContext(
            current_position_xyz=current_pos, current_yaw_rad=0.0,
            rock_center_xyz=rock_center, visited_viewpoints=[],
            image_budget=20, images_used=12, current_iteration=0,
        )
        scores = scorer.score_candidates(final_pool, dummy_snapshot, context)
        scored_ids = [s.candidate_id for s in scores[:top_n]]

    # ---- Plot ----
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    def _xyz(cands):
        return [c.x for c in cands], [c.y for c in cands], [c.z for c in cands]

    ax.scatter(*_xyz(raw), c='lightgrey', s=8, alpha=0.3, label=f'Raw ({len(raw)})')
    ax.scatter(*_xyz(feasible), c='dodgerblue', s=15, alpha=0.5, label=f'Feasible ({len(feasible)})')
    ax.scatter(*_xyz(final_pool), c='orange', s=25, alpha=0.7, label=f'Final pool ({len(final_pool)})')

    if scored_ids:
        top_cands = [c for c in final_pool if c.candidate_id in scored_ids]
        ax.scatter(*_xyz(top_cands), c='red', s=60, marker='*', label=f'Top {len(top_cands)} scored')

    ax.scatter(*rock_center, c='green', s=200, marker='D', label='Rock centre')
    ax.scatter(*current_pos, c='magenta', s=100, marker='^', label='Drone position')

    ax.set_xlabel('X (North)')
    ax.set_ylabel('Y (East)')
    ax.set_zlabel('Z (Down)')
    ax.set_title('NBV Candidate Pipeline Visualisation')

    leg = ax.legend(loc='upper left')
    # Toggle visibility on legend click
    lined = {}
    for legline, origline in zip(leg.legendHandles, ax.collections):
        legline.set_picker(5)
        lined[legline] = origline

    def on_pick(event):
        legline = event.artist
        origline = lined.get(legline)
        if origline is None:
            return
        vis = not origline.get_visible()
        origline.set_visible(vis)
        legline.set_alpha(1.0 if vis else 0.2)
        fig.canvas.draw()

    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
