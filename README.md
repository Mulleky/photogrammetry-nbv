# Photogrammetry Covisibility NBV

A ROS 2 package for autonomous Next-Best-View (NBV) photogrammetry using covisibility-driven scoring. A simulated drone captures images of a target object, incrementally reconstructs its 3D model via COLMAP, and selects subsequent viewpoints by scoring candidates against the evolving sparse point cloud. The system runs entirely in simulation using PX4 SITL and Gazebo.

## Software Stack

| Component | Version |
|-----------|---------|
| Ubuntu | 24.04 LTS |
| ROS 2 | Jazzy |
| Gazebo | Harmonic |
| PX4 Autopilot | v1.15+ (SITL) |
| COLMAP | 3.8+ |
| Python | 3.12 |
| MicroXRCE-DDS Agent | latest |

### Python Dependencies

NumPy, SciPy, scikit-learn, PyYAML, matplotlib, plyfile. All ROS dependencies (rclpy, px4_msgs, sensor_msgs, ros_gz_bridge, ros_gz_image) are resolved through the ROS 2 workspace.

## Cloning and Building

```bash
# Clone
git clone <repo-url> ~/photogrammetry-covisibility
cd ~/photogrammetry-covisibility

# Source ROS 2 and px4_msgs workspace
source /opt/ros/jazzy/setup.bash
source ~/px4_msgs_ws/install/setup.bash

# Build
colcon build --packages-select photogrammetry_nbv --symlink-install
source install/setup.bash
```

After modifying any source code, rebuild with:

```bash
cd ~/photogrammetry-covisibility
colcon build --packages-select photogrammetry_nbv --symlink-install
source install/setup.bash
```

Because --symlink-install is used, changes to Python files under photogrammetry_nbv/ take effect without rebuilding. A rebuild is only required when setup.py, package.xml, launch files, or config files change.

## PX4 Simulation Assets

The Gazebo world and drone model files are not part of the PX4-Autopilot source tree by default — this repo carries the custom ones under `models/`, and they need to be copied into someone else's PX4-Autopilot checkout before the simulation will find them:

| File/folder in this repo | Destination in `PX4-Autopilot/` |
|---|---|
| `models/sample_15016.sdf` | `Tools/simulation/gz/worlds/sample_15016.sdf` (world SDF) |
| `models/lunar_sample_15016/` | `Tools/simulation/gz/models/lunar_sample_15016/` (custom world model: meshes + `model.sdf`/`model.config`) |
| `models/x500_gimbal/` | `Tools/simulation/gz/models/x500_gimbal/` (drone model; references the stock `x500` model already shipped with PX4, so it has no meshes of its own) |
| `models/4022_gz_px4_gsplat` | `ROMFS/px4fmu_common/init.d-posix/airframes/4022_gz_px4_gsplat` (drone airframe startup script) |

After copying the airframe script, register it in `ROMFS/px4fmu_common/init.d-posix/airframes/CMakeLists.txt` (add the filename to the list) so it's picked up by the SITL build, then rebuild PX4 (`make px4_sitl`).

## Running the Simulation

All components launch from a single command in one terminal:

```bash
source /opt/ros/jazzy/setup.bash
source ~/px4_msgs_ws/install/setup.bash
source ~/photogrammetry-covisibility/install/setup.bash

ros2 launch photogrammetry_nbv unified_mission.launch.py
```

This starts PX4 SITL, the MicroXRCE-DDS agent, the Gazebo-ROS bridge (image and parameter bridges), and the unified controller node. Launch arguments can override defaults:

```bash
ros2 launch photogrammetry_nbv unified_mission.launch.py \
    px4_gz_world:=sample_15016 \
    start_px4:=true \
    start_xrce_agent:=true
```

If PX4 or the XRCE agent are already running externally, set start_px4:=false or start_xrce_agent:=false.

## Mission Pipeline

The unified controller runs a two-phase state machine:

1. Phase 1 (Seed) -- the drone flies a ring orbit at a fixed radius and captures a set of seed images to bootstrap the initial COLMAP sparse reconstruction.
2. Phase 2 (NBV Loop) -- for each iteration up to the image budget:
   - Generate candidate viewpoints on a Fibonacci hemisphere around the target.
   - Filter and downsample candidates for diversity.
   - Score all candidates against the current sparse model.
   - Fly to the top-scoring candidate, capture an image, and incrementally update the COLMAP reconstruction.
3. Finalization -- the drone lands, a dense reconstruction is run offline, and a separate seed-only sparse reconstruction is saved for comparison.

## Scoring System

All scorers compute a weighted linear combination:

  score = w_primary * primary_term + w_novelty * novelty
          - w_movement * movement_cost - w_angular * angular_penalty

Each term is normalized to approximately [0, 1]. The weights and the definition of the primary term differ per scorer:

- Covisibility -- the primary term is the fraction of sparse 3D points visible from the candidate's projected camera frustum. Rewards viewpoints that see the most existing structure.
- Repair-Weighted Covisibility -- the primary term is a weighted repair mass. Each 3D point carries a weakness score (a weighted combination of inverse track length, local kNN sparsity, and reprojection error), and the candidate's score is the sum of weakness over its visible points, normalized by total scene weakness. Prioritizes views that observe geometrically weak regions.
- Baseline-Aware Repair-Weighted Covisibility -- extends the repair-weighted scorer with a per-point geometry gain factor. The gain is a band-pass function of the minimum triangulation angle between the candidate and all existing observations of each point, rewarding new baselines that improve triangulation geometry.
- GT Phase-Adaptive Hybrid -- a meta-scorer that wraps the covisibility and baseline-aware scorers and gates between them each iteration. Gate modes include a budget-fraction heuristic, a ground-truth oracle (for supervised data collection), and a learned decision-tree policy trained offline on shadow logs.

Scorer selection and all weights are configured in config/scoring.yaml without code changes.

## Alignment and Evaluation

COLMAP reconstructs in an arbitrary coordinate frame. To compare against ground truth and to project candidates into the sparse model, the codebase uses the Umeyama similarity transform -- a closed-form least-squares solution for scale, rotation, and translation that aligns COLMAP camera centers to their known NED positions from flight metadata.

The evaluation pipeline cleans the aligned dense cloud through a sequence of steps: bounding-box crop around the target, RANSAC ground-plane removal (fit on the lowest 20% of points in NED-z), statistical outlier removal (SOR), and a distance gate against the GT mesh surface. Metrics are computed between the cleaned cloud and uniformly sampled GT surface points: completeness, accuracy, F-score at multiple distance thresholds, and mean/median/P95 cloud-to-cloud distance. Hausdorff distance (GT→recon and recon→GT) is computed separately by `compare_scorers.py`, which also handles multi-run comparisons.

`--gt-transform` is a 4×4 `.npy` matrix mapping the GT mesh's model-local frame into NED; if omitted, the mesh is used as-is. For `lunar_sample_15016` placed at Gazebo ENU (8, 0, 0.8), build it with an ENU→NED rotation (swap X/Y, negate Z) plus translation:

```python
import numpy as np
R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=float)
t = np.array([0, 8, -0.8])          # NED position of model origin
T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
np.save('gt_ned.npy', T)
```

`eval_and_plot.py` already defaults to an equivalent built-in matrix (with mesh scale 20 baked in) for `lunar_sample_15016` when `--gt-transform` is omitted, so you only need to build your own `.npy` for a different GT model or placement.

### Running Evaluation

The quickest path is the combined script, which auto-discovers seed/NBV clouds in a run directory, aligns them, computes metrics, and plots:

```bash
python3 src/photogrammetry_nbv/scripts/eval_and_plot.py \
    --run-dir ~/photogrammetry_NBV/data/photogrammetry/unified_run_20260620_085130 \
    --gt-mesh ~/PX4-Autopilot/Tools/simulation/gz/models/lunar_sample_15016/meshes/15016-0_SFM_Web-Resolution-Model_Coordinate-Registered.obj \
    --no-show
```

This writes `report.json`, `trajectory_3d.png`, `metrics.png`, and a two-panel `diagnostics.png` (cloud-to-cloud distance summary and sparse reconstruction evolution over iterations) into the run directory.

To run alignment and evaluation as separate steps instead:

```bash
# 1. Align the reconstructed cloud to the NED frame
python3 src/photogrammetry_nbv/scripts/align_cloud.py \
    --cloud final/dense_cloud.ply \
    --images-bin colmap/sparse/0/images.bin \
    --metadata seed/metadata adaptive/metadata \
    --output aligned_nbv.ply \
    --bbox-center 0 8 -0.8 \
    --bbox-half-extent 3.0

# 2. Compute completeness/accuracy/F-score/C2C metrics against GT
python3 src/photogrammetry_nbv/scripts/evaluate_run.py \
    --gt-mesh <path/to/gt_mesh.obj> \
    --gt-transform /tmp/gt_ned.npy \
    --clouds seed:/tmp/aligned_seed.ply nbv:/tmp/aligned_nbv.ply \
    --rock-center 0 8 -0.8 \
    --output-dir eval/ \
    --thresholds 0.005 0.01 0.02 0.05
```

To compute Hausdorff distance and compare metrics across multiple scorer runs:

```bash
python3 src/photogrammetry_nbv/scripts/compare_scorers.py \
    ~/photogrammetry_NBV/data/photogrammetry/unified_run_20260620_085130 \
    ~/photogrammetry_NBV/data/photogrammetry/unified_run_20260621_103000 \
    --gt-mesh ~/PX4-Autopilot/Tools/simulation/gz/models/lunar_sample_15016/meshes/15016-0_SFM_Web-Resolution-Model_Coordinate-Registered.obj \
    --output-dir ~/photogrammetry_NBV/data/comparisons \
    --no-show
```

This accepts one or more `unified_run_*` directories (one per scorer/run to compare) and writes `hausdorff_distance_summary.png` (GT→recon and recon→GT Hausdorff, plus mean/P95 C2C, for sparse and dense clouds) alongside other comparison plots in a timestamped subfolder of `--output-dir`.

See [src/photogrammetry_nbv/README.md](src/photogrammetry_nbv/README.md) for the full per-script argument reference.

## Project Structure

```
src/photogrammetry_nbv/
├── config/
│   ├── unified_mission.yaml          Flight, candidate generation, and path params
│   ├── scoring.yaml                  Scorer selection, weights, and scorer-specific params
│   ├── colmap.yaml                   COLMAP pipeline settings (SIFT, matching, BA, dense)
│   └── metashape.yaml                Alternative Metashape backend config
├── launch/
│   └── unified_mission.launch.py     Single-command launch (PX4, XRCE, bridge, controller)
├── photogrammetry_nbv/
│   ├── unified_controller_node.py    Main state machine (Phase 1 + Phase 2)
│   ├── phase2_controller_node.py     Standalone Phase 2 controller
│   ├── candidate_generator.py        Fibonacci hemisphere viewpoint generation
│   ├── candidate_filter.py           Altitude, spacing, and diversity filtering
│   ├── scoring_interface.py          BaseScorer abstract interface
│   ├── contracts.py                  Data contracts (CandidateViewpoint, ScoreBreakdown, etc.)
│   ├── colmap_worker_client.py       COLMAP subprocess orchestration
│   ├── metrics_extractor.py          Sparse model metric extraction
│   ├── seed_loader.py                Phase 1 seed image loader
│   ├── mission_logger.py             Per-iteration JSON logging
│   ├── run_context.py                Run directory management
│   ├── scorers/
│   │   ├── covisibility_scorer.py
│   │   ├── repair_weighted_covisibility_scorer.py
│   │   ├── baseline_aware_repair_weighted_covisibility_scorer.py
│   │   ├── gt_phase_adaptive_hybrid_scorer.py
│   │   └── weighted_sum_scorer.py
│   ├── gt_supervision/
│   │   ├── mesh_oracle.py            GT mesh frustum coverage oracle
│   │   └── coverage_state.py         Tracks covered GT surface samples
│   └── adaptive/
│       ├── train_gt_phase_switch.py  Offline decision-tree training on shadow logs
│       └── load_tree_policy.py       Runtime JSON tree inference (no sklearn dependency)
├── colmap_scripts/
│   ├── bootstrap_project.py          Initial COLMAP project setup
│   ├── incremental_update.py         Per-iteration sparse model update
│   ├── offline_dense_reconstruct.py  Post-mission dense reconstruction
│   ├── seed_sparse_reconstruct.py    Seed-only sparse reconstruction
│   ├── export_sparse_metrics.py      Extract track lengths, reprojection errors, weak regions
│   └── common.py                     Shared COLMAP utilities
└── scripts/
    ├── evaluate_run.py               GT-aligned dense cloud evaluation
    ├── compare_scorers.py            Multi-run comparison plots and metrics
    ├── align_cloud.py                Standalone Umeyama cloud alignment
    ├── eval_and_plot.py              Combined eval + visualization
    └── visualize_candidates.py       Candidate pool visualization
```

## Modularity

The scoring system is decoupled from flight control and reconstruction through the BaseScorer abstract interface. New scorers are added by subclassing BaseScorer, implementing score_candidates(), and registering the class in the scorer registry. The controller selects the active scorer by name from scoring.yaml at startup.

COLMAP interaction is isolated behind colmap_worker_client.py, which shells out to the scripts in colmap_scripts/. Swapping to a different SfM backend (e.g., Metashape) requires only replacing the worker client and its scripts.

Candidate generation, filtering, and scoring are independent pipeline stages connected through the CandidateViewpoint and ScoreBreakdown data contracts in contracts.py.

## Configuration

All runtime parameters are set through YAML config files with no code changes required:

- unified_mission.yaml -- seed orbit geometry, image budget, stopping criteria, candidate generation parameters (count, radius, elevation bands, spacing), flight parameters (altitude, tolerances, gimbal angles), and ROS topic names.
- scoring.yaml -- active scorer name, scorer weights, weakness component weights, geometry gain thresholds, hybrid gate mode and gate parameters, shadow logging settings.
- colmap.yaml -- SIFT feature count, matching strategy, bundle adjustment iterations, dense reconstruction settings, weak-region extraction parameters, kNN density metrics, and incremental update strategy.
