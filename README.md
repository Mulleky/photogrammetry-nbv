# photogrammetry-covisibility

ROS 2 workspace for autonomous photogrammetric reconstruction using co-visibility-based Next-Best-View (NBV) planning with a PX4 drone in Gazebo.

The system runs two phases in a single mission:
- **Phase 1 (Seed)**: Fibonacci hemisphere orbit captures 20 images for COLMAP bootstrap
- **Phase 2 (NBV)**: Iteratively scores and visits candidate viewpoints using co-visibility of reconstructed 3D points

---

## Repository layout

```
src/
├── photogrammetry_init/     # Legacy 4-view init node (not used in main mission)
└── photogrammetry_nbv/
    ├── config/
    │   ├── unified_mission.yaml   # All mission parameters
    │   ├── colmap.yaml            # COLMAP pipeline settings
    │   └── scoring.yaml           # NBV scorer weights
    ├── launch/
    │   └── unified_mission.launch.py
    ├── photogrammetry_nbv/
    │   ├── unified_controller_node.py   # Main mission state machine
    │   ├── candidate_generator.py       # Fibonacci hemisphere sampling
    │   ├── candidate_filter.py          # Altitude/spacing/travel filters
    │   ├── scorers/
    │   │   └── covisibility_scorer.py   # Co-visibility NBV scoring
    │   ├── colmap_worker_client.py      # Subprocess COLMAP wrapper
    │   └── metrics_extractor.py        # COLMAP output parser
    └── scripts/
        ├── align_cloud.py              # Umeyama alignment: COLMAP→NED
        └── evaluate_run.py             # Reconstruction quality metrics
```

---

## Prerequisites

- ROS 2 Humble
- PX4-Autopilot (SITL)
- Micro XRCE-DDS Agent (`MicroXRCEAgent`)
- `ros_gz_bridge`, `ros_gz_image`
- COLMAP (on `PATH` as `colmap`)
- Python: `numpy`, `scipy`, `cv2` (opencv-python), `yaml`

Build the workspace once:

```bash
cd ~/photogrammetry-covisibility
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

---

## Running a mission

The unified mission requires **3 terminals**.

---

### Terminal 1 — PX4 SITL + Gazebo

```bash
cd ~/PX4-Autopilot
PX4_GZ_NO_FOLLOW=1 PX4_GZ_WORLD=sample_15016 make px4_sitl gz_px4_gsplat
```

Wait until you see `INFO [commander] Ready for takeoff!` in the PX4 output before proceeding.

---

### Terminal 2 — Micro XRCE-DDS Agent

Bridges PX4 uORB topics to ROS 2 DDS.

```bash
MicroXRCEAgent udp4 -p 8888
```

Leave this running for the entire session.

---

### Terminal 3 — ROS 2 bridge + mission node

```bash
source /opt/ros/humble/setup.bash
source ~/photogrammetry-covisibility/install/setup.bash

ros2 launch photogrammetry_nbv unified_mission.launch.py
```

This single launch file starts:
- `ros_gz_bridge` — bridges Gazebo topics (gimbal, vehicle status, etc.)
- `ros_gz_image image_bridge` — bridges `/rgbd/image`
- `unified_controller_node` — runs the full Phase 1 + Phase 2 mission

The node will arm, take off, collect seed images, run COLMAP, then iterate NBV views automatically.  
Output is written to `~/photogrammetry_NBV/data/photogrammetry/unified_run_<timestamp>/`.

---

### Optional: run with custom parameters

Override any parameter from `unified_mission.yaml` on the command line:

```bash
ros2 launch photogrammetry_nbv unified_mission.launch.py \
    px4_autopilot_path:=~/PX4-Autopilot \
    px4_gz_world:=sample_15016 \
    px4_make_target:=gz_px4_gsplat
```

To change mission parameters without editing the YAML, pass them as ROS 2 params directly to the node (see ROS 2 parameter overriding docs).

---

## Mission output structure

```
unified_run_<timestamp>/
├── seed/
│   ├── images/          # 20 seed images (ring_00 … ring_19)
│   ├── metadata/        # Per-image JSON (NED pose, gimbal, camera info)
│   └── manifest.json    # Seed mission summary
├── adaptive/
│   ├── images/          # NBV images (iter_00, iter_01 …)
│   └── metadata/        # Per-image JSON
├── colmap/
│   ├── sparse/0/        # Final sparse reconstruction (images.bin, points3D.bin)
│   └── dense/           # Dense depth maps and stereo output
├── seed_colmap/
│   └── sparse/          # Seed-only reconstruction
├── sparse_metrics/
│   └── metrics_iter_*.json   # KNN distance, point count per iteration
├── candidates/
│   ├── candidate_pool_iter_*.json
│   └── candidate_scores_iter_*.json
├── final/
│   ├── dense_cloud.ply       # Full dense cloud (seed + NBV)
│   └── seed_sparse_cloud.ply # Seed-only sparse cloud
└── phase2_manifest.json      # Stopping criterion, budget, scorer used
```

---

## Key parameters (unified_mission.yaml)

| Parameter | Default | Description |
|---|---|---|
| `seed_radius_m` | 5.0 | Radius of Phase 1 hemisphere orbit |
| `ring_image_count` | 20 | Number of seed images |
| `rock_center_x/y/z` | 0, 8, -0.8 | Rock centre in NED (m) |
| `image_budget` | 15 | Phase 2 soft stop (used with `knn_distance` criterion) |
| `max_image_budget` | 35 | Hard stop regardless of criterion |
| `stopping_criterion` | `knn_distance` | `image_budget` \| `knn_distance` \| `both` |
| `knn_distance_threshold` | 0.05 | KNN percentile stop threshold |
| `candidate_radius_m` | 4.0 | NBV candidate orbit radius |
| `min_elevation_deg` | 10 | Lower elevation bound for candidates |
| `max_elevation_deg` | 50 | Upper elevation bound for candidates |
| `takeoff_altitude` | 3.0 | Initial takeoff altitude (m) |

---

## Stopping criterion

Phase 2 stops when **either** condition is met (depending on `stopping_criterion`):

- `image_budget`: stops after `image_budget` NBV images
- `knn_distance`: stops when the 95th-percentile KNN distance in the sparse cloud drops below `knn_distance_threshold`
- `both`: stops when *either* condition is met
- `max_image_budget` is a hard cap regardless of criterion

---

## Post-mission evaluation

After a run completes, evaluate reconstruction quality against the GT mesh using a two-step process.

### Step 1 — Align cloud to NED frame

The COLMAP reconstruction lives in an arbitrary metric frame. `align_cloud.py` computes a 7-DOF Umeyama transform from COLMAP camera centres to NED positions recorded in the metadata JSONs.

**Dense cloud (seed + NBV):**
```bash
RUN=~/photogrammetry_NBV/data/photogrammetry/unified_run_<timestamp>

python3 ~/photogrammetry-covisibility/src/photogrammetry_nbv/scripts/align_cloud.py \
    --cloud $RUN/final/dense_cloud.ply \
    --images-bin $RUN/colmap/sparse/0/images.bin \
    --metadata $RUN/seed/metadata $RUN/adaptive/metadata \
    --output /tmp/aligned_nbv.ply \
    --bbox-center 0 8 -0.8 \
    --bbox-half-extent 3.0
```

**Seed-only sparse cloud:**
```bash
python3 ~/photogrammetry-covisibility/src/photogrammetry_nbv/scripts/align_cloud.py \
    --cloud $RUN/final/seed_sparse_cloud.ply \
    --images-bin $RUN/seed_colmap/sparse/0/images.bin \
    --metadata $RUN/seed/metadata \
    --output /tmp/aligned_seed.ply \
    --bbox-center 0 8 -0.8 \
    --bbox-half-extent 3.0
```

> If the seed alignment fails with "< 3 matched cameras", try `seed_colmap/sparse/1/images.bin`.

### Step 2 — Build the GT mesh transform (one-time)

The GT OBJ mesh is in model-local coordinates. This maps it to NED given the rock's Gazebo ENU pose (8, 0, 0.8) and mesh scale 20:

```bash
python3 -c "
import numpy as np
# ENU→NED: x_ned=y_enu, y_ned=x_enu, z_ned=-z_enu, with scale=20 and translation
T = np.array([
    [0,  20, 0,     0   ],
    [20,  0, 0,     8.0 ],
    [0,   0, -20,  -0.8 ],
    [0,   0,  0,    1.0 ],
], dtype=float)
np.save('/tmp/gt_ned.npy', T)
print('Saved gt_ned.npy')
"
```

### Step 3 — Run evaluate_run.py

```bash
python3 ~/photogrammetry-covisibility/src/photogrammetry_nbv/scripts/evaluate_run.py \
    --gt-mesh ~/PX4-Autopilot/Tools/simulation/gz/models/lunar_sample_15016/meshes/15016-0_SFM_Web-Resolution-Model_Coordinate-Registered.obj \
    --gt-transform /tmp/gt_ned.npy \
    --clouds seed:/tmp/aligned_seed.ply nbv:/tmp/aligned_nbv.ply \
    --rock-center 0 8 -0.8 \
    --output-dir $RUN/eval \
    --thresholds 0.005 0.01 0.02 0.05
```

Output in `$RUN/eval/`:
- `cleaned_seed.ply`, `cleaned_nbv.ply` — cropped, denoised clouds (open in CloudCompare)
- `report.json` — all metrics

**Metrics reported:**

| Metric | Definition |
|---|---|
| Completeness | % of GT surface within threshold of recon cloud |
| Accuracy | % of recon cloud within threshold of GT surface |
| F-score | Harmonic mean of completeness and accuracy |
| Mean/Median/P95 C2C | Cloud-to-cloud distance GT→recon |

---

## Scoring modularity

The NBV scorer is swappable without touching the controller:

1. Add a class under `photogrammetry_nbv/scorers/` implementing `BaseScorer`
2. Register it in `photogrammetry_nbv/scorers/__init__.py`
3. Set `scorer.name` in `config/scoring.yaml`
