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

The evaluation pipeline cleans the aligned dense cloud through a sequence of steps: bounding-box crop around the target, RANSAC ground-plane removal (fit on the lowest 20% of points in NED-z), statistical outlier removal (SOR), and a distance gate against the GT mesh surface. Metrics are computed between the cleaned cloud and uniformly sampled GT surface points: completeness, accuracy, F-score at multiple distance thresholds, mean and P95 cloud-to-cloud distance, and Hausdorff distance.

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
