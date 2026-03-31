from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .candidate_filter import crop_to_target_count_diverse, filter_candidates
from .candidate_generator import generate_fibonacci_hemisphere
from .contracts import ScoreContext
from .metrics_extractor import load_sparse_metrics
from .mission_logger import MissionLogger
from .scorers import SCORER_REGISTRY
from .seed_loader import infer_home_pose, load_seed_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Offline phase-2 scoring validation.')
    parser.add_argument('--seed-run-dir', required=True)
    parser.add_argument('--metrics-json', required=True)
    parser.add_argument('--phase2-config', required=True)
    parser.add_argument('--scoring-config', required=True)
    parser.add_argument('--output-dir', required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_run_dir = Path(args.seed_run_dir).expanduser()
    metrics_json = Path(args.metrics_json).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.phase2_config, 'r', encoding='utf-8') as f:
        phase2_cfg = yaml.safe_load(f)
    with open(args.scoring_config, 'r', encoding='utf-8') as f:
        scoring_cfg = yaml.safe_load(f)

    _, seed_metadata, _ = load_seed_bundle(seed_run_dir)
    rock_center = [
        float(phase2_cfg['/**']['ros__parameters']['rock_center_x']),
        float(phase2_cfg['/**']['ros__parameters']['rock_center_y']),
        float(phase2_cfg['/**']['ros__parameters']['rock_center_z']),
    ]
    current = infer_home_pose(seed_metadata)
    current_xyz = [float(current['x']), float(current['y']), float(current['z'])]

    snapshot = load_sparse_metrics(metrics_json, iteration=0)

    params = phase2_cfg['/**']['ros__parameters']
    raw = generate_fibonacci_hemisphere(
        center_xyz=rock_center,
        radius=float(params['candidate_radius_m']),
        candidate_count=int(params['raw_candidate_count']),
        min_elevation_deg=float(params['min_elevation_deg']),
        max_elevation_deg=float(params['max_elevation_deg']),
    )
    feasible = filter_candidates(
        raw,
        current_position_xyz=current_xyz,
        min_altitude_ned=float(params['min_altitude_ned']),
        max_altitude_ned=float(params['max_altitude_ned']),
        min_spacing_m=float(params['candidate_min_spacing_m']),
        max_travel_m=float(params['max_candidate_travel_m']),
    )
    final_pool = crop_to_target_count_diverse(feasible, int(params['target_candidate_count']))

    visited = []
    for item in seed_metadata:
        visited.append({
            'position_ned_m': item.get('vehicle_position_ned_m', {}),
            'yaw_rad': item.get('target_yaw_rad', 0.0),
        })

    context = ScoreContext(
        current_position_xyz=current_xyz,
        current_yaw_rad=float(seed_metadata[-1].get('target_yaw_rad', 0.0)) if seed_metadata else 0.0,
        rock_center_xyz=rock_center,
        visited_viewpoints=visited,
        image_budget=int(params['image_budget']),
        images_used=len(seed_metadata),
        current_iteration=0,
    )

    scorer_name = str(scoring_cfg['scorer']['name'])
    scorer_cls = SCORER_REGISTRY[scorer_name]
    scorer = scorer_cls(scoring_cfg['scorer'])
    scores = scorer.score_candidates(final_pool, snapshot, context)

    logger = MissionLogger(output_dir, output_dir)
    logger.log_candidates(0, final_pool)
    logger.log_scores(0, scores)
    top = scores[0]
    lookup = {c.candidate_id: c for c in final_pool}
    logger.log_selected(0, {
        'selected_candidate': lookup[top.candidate_id].to_dict(),
        'score': top.to_dict(),
    })
    print(json.dumps({'status': 'ok', 'candidate_count': len(final_pool), 'top_candidate_id': top.candidate_id}, indent=2))


if __name__ == '__main__':
    main()
