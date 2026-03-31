# photogrammetry_nbv

ROS 2 package implementing a two-phase autonomous photogrammetry mission:
- **Phase 1**: Fibonacci hemisphere seed orbit (20 images)
- **Phase 2**: Co-visibility-based NBV loop with COLMAP sparse reconstruction

See the [workspace README](../../README.md) for full setup and launch instructions.

---

## Package components

### unified_controller_node.py
Main mission state machine. Runs Phase 1 and Phase 2 in a single ROS 2 node within one sim session.

**State machine:**
```
WAIT_FOR_TOPICS → WARMUP_STREAM → TAKEOFF →
[MOVE_TO_SEED → SETTLE_SEED → CAPTURE_SEED]×20 →
BOOTSTRAP_PROJECT →
[SCORE_NEXT_VIEW → FLY_TO_VIEW → SETTLE → CAPTURE → UPDATE_PROJECT]* →
RETURN_HOME → LAND → OFFLINE_DENSE_RECON → SEED_SPARSE_RECON → FINISHED
```

### candidate_generator.py
Generates Fibonacci hemisphere candidate viewpoints around the rock centre.
Used for both Phase 1 seed poses and Phase 2 NBV candidates.

### candidate_filter.py
Filters candidates by altitude bounds, minimum spacing, and maximum travel distance from current position.

### scorers/covisibility_scorer.py
Scores each candidate by:
- **Co-visibility**: fraction of reconstructed rock points visible from the candidate
- **Novelty**: distance from previously visited viewpoints
- **Movement cost**: travel distance + yaw change penalty
- **Angular separation penalty**: penalises candidates too close (angle) to existing views

Scores are weighted sums configurable in `config/scoring.yaml`.

### colmap_worker_client.py
Subprocess wrapper for calling the COLMAP Python scripts.
Handles bootstrap, incremental update, dense reconstruction, and sparse metrics export.

### metrics_extractor.py
Reads COLMAP `metrics_iter_*.json` output (point count, KNN distances, reprojection errors).
Used by `_should_stop()` to evaluate the `knn_distance` stopping criterion.

---

## Configuration files

### config/unified_mission.yaml
All parameters for the unified mission. Key groups:

**Phase 1 — seed hemisphere**
```yaml
seed_radius_m: 5.0          # orbit radius
ring_image_count: 20        # number of seed images
```

**Phase 2 — NBV**
```yaml
rock_center_x: 0.0          # NED position of reconstruction target
rock_center_y: 8.0
rock_center_z: -0.8
stopping_criterion: knn_distance  # image_budget | knn_distance | both
image_budget: 15            # soft stop for knn_distance criterion
max_image_budget: 35        # hard stop regardless of criterion
knn_distance_threshold: 0.05
candidate_radius_m: 4.0
min_elevation_deg: 10.0
max_elevation_deg: 50.0
```

### config/scoring.yaml
Scorer selection and weights:
```yaml
scorer:
  name: covisibility
  weights:
    covisibility: 1.0
    novelty: 0.8
    movement_cost: 0.2
    angular_separation_penalty: 0.4
  scoring_bbox_half_extent: 1.5   # only rock points within this box drive scoring
```

### config/colmap.yaml
COLMAP pipeline settings (SIFT features, exhaustive matching, mapper, dense MVS).

---

## COLMAP scripts

Located in `colmap_scripts/`. Called as subprocesses by `colmap_worker_client.py`.

| Script | Called when |
|---|---|
| `bootstrap_project.py` | After seed capture completes (BOOTSTRAP_PROJECT state) |
| `incremental_update.py` | After each NBV capture (UPDATE_PROJECT state) |
| `export_sparse_metrics.py` | After each update to compute KNN distances |
| `seed_sparse_reconstruct.py` | After landing (seed-only sparse reconstruction) |
| `offline_dense_reconstruct.py` | After landing (full dense cloud) |

---

## Post-run scripts

Located in `scripts/`.

### align_cloud.py
Aligns a COLMAP PLY cloud to NED frame using Umeyama similarity transform computed from matching COLMAP camera centres to metadata NED positions.

```bash
python3 scripts/align_cloud.py \
    --cloud final/dense_cloud.ply \
    --images-bin colmap/sparse/0/images.bin \
    --metadata seed/metadata adaptive/metadata \
    --output aligned_nbv.ply \
    --bbox-center 0 8 -0.8 \
    --bbox-half-extent 3.0
```

### evaluate_run.py
Full reconstruction quality evaluation pipeline. Requires only `numpy` and `scipy`.

**Pipeline per cloud:**
1. Apply optional world transform
2. Bbox crop (rock_center ± extent)
3. RANSAC ground plane removal
4. Statistical Outlier Removal (SOR)
5. Distance gate against GT mesh surface

**Metrics:** Completeness, Accuracy, F-score at configurable distance thresholds; mean/median/P95 cloud-to-cloud distance.

```bash
python3 scripts/evaluate_run.py \
    --gt-mesh <path/to/gt_mesh.obj> \
    --gt-transform /tmp/gt_ned.npy \
    --clouds seed:/tmp/aligned_seed.ply nbv:/tmp/aligned_nbv.ply \
    --rock-center 0 8 -0.8 \
    --output-dir eval/ \
    --thresholds 0.005 0.01 0.02 0.05
```

See the [workspace README](../../README.md) for how to build `gt_ned.npy`.

---

## Adding a new scorer

1. Create `photogrammetry_nbv/scorers/my_scorer.py` implementing `BaseScorer`
2. Add it to `photogrammetry_nbv/scorers/__init__.py` in `SCORER_REGISTRY`
3. Set `scorer.name: my_scorer` in `config/scoring.yaml`

No changes needed to the controller.
