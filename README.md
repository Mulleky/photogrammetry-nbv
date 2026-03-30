# Photogrammetry NBV

Active shape estimation of an unknown object using a drone, ROS 2, PX4, and COLMAP. The system runs in two phases: a deterministic ring-orbit capture to seed the reconstruction, followed by an adaptive Next-Best-View (NBV) loop that selects each subsequent viewpoint based on the quality of the growing sparse point cloud.

## Table of Contents

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Repository Structure](#repository-structure)
- [Installation and Build](#installation-and-build)
- [Phase 1 — Ring Orbit Seed Capture](#phase-1--ring-orbit-seed-capture)
- [Phase 2 — Adaptive NBV Reconstruction](#phase-2--adaptive-nbv-reconstruction)
- [Post-Run Evaluation](#post-run-evaluation)
- [Visualisation Tools](#visualisation-tools)
- [Configuration Reference](#configuration-reference)

---

## Overview

Reconstructing an unknown 3D object from drone imagery requires deciding not just *how* to process images, but *where to fly* to capture them. A fixed orbit captures every viewpoint equally regardless of where the reconstruction is weakest. This system instead uses the reconstruction itself as feedback: after each new image is registered into COLMAP, the sparse point cloud is analysed for weak regions, and the next viewpoint is chosen to address them.

The pipeline has two sequential phases:

**Phase 1** flies a deterministic ring orbit around the object to produce a set of overlapping seed images. These are sufficient for COLMAP to bootstrap a sparse reconstruction and establish a metric coordinate frame.

**Phase 2** enters a closed loop: generate candidate viewpoints on a Fibonacci hemisphere, score them against the current reconstruction quality, fly to the best one, capture an image, update COLMAP, and repeat until the image budget is exhausted. After the budget is spent the system lands, runs MVS dense reconstruction, and then separately reconstructs the seed-only sparse cloud for comparison.

---

## System Requirements

- Ubuntu 24.04
- ROS 2 Jazzy
- PX4 Autopilot (SITL) with Gazebo
- Micro XRCE-DDS Agent
- COLMAP (built from source)
- Python 3.10+
- px4_msgs ROS 2 workspace

Python dependencies (beyond ROS 2):

```bash
pip install open3d numpy
```

---

## Repository Structure

```
photogrammetry-nbv/
├── src/
│   ├── photogrammetry_init/          # Phase 1 package
│   │   ├── config/
│   │   │   ├── mission.yaml          # Ring orbit parameters
│   │   │   └── bridge.yaml           # Gazebo-ROS topic bridges
│   │   ├── launch/
│   │   │   └── photogrammetry_init.launch.py
│   │   └── photogrammetry_init/
│   │       └── four_view_init_node.py
│   │
│   └── photogrammetry_nbv/           # Phase 2 package
│       ├── config/
│       │   ├── phase2_mission.yaml   # NBV and evaluation parameters
│       │   ├── colmap.yaml           # COLMAP pipeline settings
│       │   └── scoring.yaml          # NBV scorer weights
│       ├── colmap_scripts/           # Headless COLMAP subprocess scripts
│       │   ├── bootstrap_project.py
│       │   ├── incremental_update.py
│       │   ├── offline_dense_reconstruct.py
│       │   ├── seed_sparse_reconstruct.py
│       │   └── export_sparse_metrics.py
│       ├── launch/
│       │   └── phase2_online_mission.launch.py
│       ├── photogrammetry_nbv/       # Core Python modules
│       │   ├── phase2_controller_node.py
│       │   ├── candidate_generator.py
│       │   ├── candidate_filter.py
│       │   ├── colmap_worker_client.py
│       │   ├── contracts.py
│       │   ├── run_context.py
│       │   ├── seed_loader.py
│       │   ├── mission_logger.py
│       │   ├── metrics_extractor.py
│       │   └── scorers/
│       │       └── weighted_sum_scorer.py
│       └── scripts/                  # Standalone utility scripts
│           ├── visualize_candidates.py
│           └── align_cloud.py
│
└── data/                             # Runtime output (gitignored)
    └── photogrammetry/
        ├── run_<timestamp>/          # Phase 1 output
        └── phase2_run_<timestamp>/   # Phase 2 output
```

---

## Installation and Build

### 1. Clone and enter the workspace

```bash
git clone <repo-url> ~/photogrammetry-nbv
cd ~/photogrammetry-nbv
```

### 2. Source dependencies

```bash
source /opt/ros/jazzy/setup.bash
source ~/workspaces/px4_msgs_ws/install/setup.bash
```

### 3. Build

```bash
colcon build --symlink-install
source install/setup.bash
```

`--symlink-install` means Python source file edits take effect without rebuilding. Config file changes (`.yaml`) still require a rebuild since they are copied at build time.

### 4. Rebuild a single package after changes

```bash
colcon build --symlink-install --packages-select photogrammetry_init
# or
colcon build --symlink-install --packages-select photogrammetry_nbv
source install/setup.bash
```

---

## Phase 1 — Ring Orbit Seed Capture

### Purpose

Phase 1 flies a single horizontal orbit around the object and captures `ring_image_count` evenly-spaced images. These seed images are the input to COLMAP in Phase 2. The orbit design addresses a fundamental requirement of structure-from-motion: adjacent images must share sufficient overlap for feature matching. At 90° spacing a typical drone camera has near-zero horizontal overlap between consecutive images and COLMAP cannot initialise. At 30° spacing or less, overlap is well within the range COLMAP requires.

### How the orbit is computed

The drone's spawn position is recorded when the first local position message arrives. The angle from the object centre to the spawn position defines the start of the orbit. Viewpoints are then distributed counterclockwise at equal angular intervals of `360° / ring_image_count`. Each viewpoint is placed at `capture_radius` metres from the object and the yaw is set to face the object. In PX4 NED coordinates, counterclockwise means decreasing `atan2(y, x)` angle.

### Launch

PX4, the Gazebo simulation, the XRCE-DDS bridge, and the init node all start from a single launch file:

```bash
cd ~/photogrammetry-nbv
source /opt/ros/jazzy/setup.bash
source ~/workspaces/px4_msgs_ws/install/setup.bash
source install/setup.bash

ros2 launch photogrammetry_init photogrammetry_init.launch.py
```

Optional arguments:

```bash
ros2 launch photogrammetry_init photogrammetry_init.launch.py \
  px4_autopilot_path:=/home/dreamslab/PX4-Autopilot \
  px4_gz_world:=sample_15016 \
  px4_make_target:=gz_px4_gsplat \
  start_px4:=true \
  start_xrce_agent:=true \
  start_bridge:=true
```

### Output

Each run writes to `~/photogrammetry_NBV/data/photogrammetry/run_<timestamp>/`:

```
run_<timestamp>/
├── images/          # ring_00.jpg … ring_11.jpg
├── metadata/        # one JSON per image with NED pose, yaw, camera intrinsics
├── calibration/     # camera_info_latest.json
└── manifest.json    # mission type, geometry, ring_image_count
```

### Key configuration (`photogrammetry_init/config/mission.yaml`)

| Parameter | Default | Description |
|---|---|---|
| `ring_image_count` | `12` | Number of images in the seed orbit |
| `capture_radius` | `5.0` | Orbit radius from object centre (m) |
| `takeoff_altitude` | `2.5` | Capture altitude above home (m) |
| `object_x` / `object_y` | `0.0` / `8.0` | Object position in NED (m) |
| `gimbal_pitch_rad` | `-0.35` | Downward gimbal tilt during capture |
| `land_after_capture` | `false` | Land when orbit completes |

---

## Phase 2 — Adaptive NBV Reconstruction

### Purpose

Phase 2 consumes the seed images from Phase 1 and iteratively grows the reconstruction by selecting the next viewpoint that is predicted to reduce reconstruction weakness most efficiently.

### Prerequisites

```
kill $(pgrep -f four_view_init_node)
```

Phase 1 must have completed and its output directory must be symlinked as `latest_seed`:

```bash
ln -sfn ~/photogrammetry_NBV/data/photogrammetry/run_<timestamp> \
        ~/photogrammetry_NBV/data/photogrammetry/latest_seed
```

### Launch

If PX4, Gazebo, and the XRCE agent are already running from Phase 1, pass `start_px4:=false start_xrce_agent:=false`:

```bash
cd ~/photogrammetry-nbv
source /opt/ros/jazzy/setup.bash
source ~/workspaces/px4_msgs_ws/install/setup.bash
source install/setup.bash

ros2 launch photogrammetry_nbv phase2_online_mission.launch.py \
  start_px4:=false \
  start_xrce_agent:=false
```

For a fully fresh start (Phase 2 standalone, no prior Phase 1 session):

```bash
ros2 launch photogrammetry_nbv phase2_online_mission.launch.py
```

### Mission state machine

```
WAIT_FOR_TOPICS → WARMUP_STREAM → BOOTSTRAP_PROJECT
  → SCORE_NEXT_VIEW → FLY_TO_VIEW → SETTLE → CAPTURE
  → UPDATE_PROJECT → (loop back to SCORE_NEXT_VIEW)
  → RETURN_HOME → LAND
  → OFFLINE_DENSE_RECON
  → SEED_SPARSE_RECON
  → FINISHED
```

### Candidate generation — Fibonacci hemisphere

Candidate viewpoints are distributed over a hemisphere above the object using a **Fibonacci spiral**. The golden angle (`π(3 − √5)`) is used as the azimuthal step between successive points, which produces a sequence that avoids clustering and covers the sphere quasi-uniformly without any grid structure. Only the upper hemisphere is sampled, and elevation is bounded between `min_elevation_deg` and `max_elevation_deg` to keep the drone in a practical flight envelope.

This produces `raw_candidate_count` (default 200) candidate positions, each with a yaw pointing toward the object centre.

### Candidate filtering

The raw pool is reduced in three stages:

1. **Altitude bounds** — candidates outside the NED altitude window `[min_altitude_ned, max_altitude_ned]` are removed.
2. **Travel distance** — candidates farther than `max_candidate_travel_m` from the drone's current position are removed.
3. **Minimum spacing** (`farthest_point_downselect`) — a greedy pass removes any candidate closer than `candidate_min_spacing_m` to an already-selected one, enforcing spatial spread.
4. **Diversity crop** (`crop_to_target_count_diverse`) — if more than `target_candidate_count` candidates remain, a farthest-point selection trims the pool while maximising spread.

### Scoring — weighted sum

Each candidate in the final pool is scored by the `weighted_sum` scorer, which combines four terms:

- **Weak region support** — how well the candidate's viewpoint covers the spatially-clustered regions of the sparse cloud where reprojection error and inverse track length are highest. These weak regions are extracted from COLMAP's `points3D.txt` after every update.
- **Novelty** — reward for visiting positions far from previously-visited viewpoints, encouraging exploration.
- **Movement cost** — penalty proportional to travel distance from the current position, favouring nearby candidates.
- **Revisit penalty** — penalty for returning to a position already visited in this session.

Weights are fully configurable in `config/scoring.yaml` without any code changes.

### COLMAP reconstruction pipeline

COLMAP runs as a subprocess via `ColmapWorkerClient`, which writes a JSON request file and calls the appropriate headless script. The pipeline is:

- **Bootstrap** (`bootstrap_project.py`) — run on the seed images at the start of Phase 2. Runs feature extraction (SIFT), exhaustive matching, and the mapper to produce the initial `sparse/0` model.
- **Incremental update** (`incremental_update.py`) — called after each adaptive capture. Adds the new image to the database, re-runs matching, and continues the mapper from the existing reconstruction.
- **Sparse metrics export** — after each update, `points3D.txt` is parsed to extract reprojection errors and track lengths, which feed directly into the weak region extraction and scorer.
- **Dense reconstruction** (`offline_dense_reconstruct.py`) — after landing: image undistortion, `patch_match_stereo` for per-image depth maps, and `stereo_fusion` to fuse them into a dense PLY.
- **Seed-only sparse** (`seed_sparse_reconstruct.py`) — after dense reconstruction: re-runs the full COLMAP pipeline on the 12 seed images in an isolated workspace (`colmap_seed_only/`), then exports to `seed_sparse_cloud.ply`. This provides a baseline for comparison against the NBV-augmented reconstruction.

### Output

Each Phase 2 run writes to `~/photogrammetry_NBV/data/photogrammetry/phase2_run_<timestamp>/`:

```
phase2_run_<timestamp>/
├── seed/                        # Copy of the Phase 1 run used as input
├── adaptive/
│   ├── images/                  # Adaptively-captured images
│   └── metadata/                # Per-image NED pose and candidate info
├── colmap/
│   ├── images/                  # All images (seed + adaptive) for COLMAP
│   ├── database.db
│   ├── sparse/0/                # Final sparse model (all images)
│   ├── sparse_txt/              # Text export of final sparse model
│   ├── sparse_cloud.ply         # Exported via model_converter
│   ├── dense/                   # MVS workspace
│   ├── sparse_metrics/          # metrics_iter_XX.json per iteration
│   └── candidate_scores/        # Candidate pool, scores, and selected per iter
├── colmap_seed_only/            # Isolated COLMAP workspace for seed-only recon
├── final/
│   ├── dense_cloud.ply          # MVS dense point cloud
│   └── seed_sparse_cloud.ply   # Sparse cloud from seed images only
└── phase2_manifest.json
```

### Key configuration (`photogrammetry_nbv/config/phase2_mission.yaml`)

| Parameter | Default | Description |
|---|---|---|
| `seed_run_dir` | `~/…/latest_seed` | Path to Phase 1 output |
| `image_budget` | `20` | Total images including seeds |
| `rock_center_x/y/z` | `0 / 8 / 0` | Object position in NED (m) |
| `candidate_radius_m` | `4.5` | Hemisphere radius (m) |
| `raw_candidate_count` | `200` | Fibonacci hemisphere sample count |
| `target_candidate_count` | `95` | Pool size after diversity crop |
| `min_elevation_deg` | `25.0` | Lower elevation bound for candidates |
| `max_elevation_deg` | `80.0` | Upper elevation bound for candidates |
| `eval_bbox_half_extent` | `6.0` | Half-extent (m) of evaluation crop box |

---

## Post-Run Evaluation

Two clouds are produced for comparison: `seed_sparse_cloud.ply` (from Phase 1 images only) and `dense_cloud.ply` (from all images). To compare them meaningfully against a ground-truth mesh they must first be aligned to the same coordinate frame.

### Why alignment is needed

COLMAP without external pose priors reconstructs in an arbitrary frame — unknown scale, rotation, and translation relative to the Gazebo world. The seed and dense clouds are therefore in different frames from the ground-truth `.obj` mesh.

### Umeyama similarity transform

Because every camera's NED position is recorded in the metadata JSONs, and COLMAP independently estimates those same camera positions in its own frame, there is a set of known point correspondences between the two frames. The **Umeyama algorithm** solves for the 7-DOF similarity transform (scale, rotation, translation) that best maps COLMAP's camera centres onto the known NED positions. This is fully automatic — no manual registration is required — and directly recovers the metric scale that COLMAP cannot determine on its own.

The transform is applied to the point cloud, placing it in metric NED coordinates aligned with the ground-truth mesh.

### Running alignment

```bash
python3 ~/photogrammetry-nbv/src/photogrammetry_nbv/scripts/align_cloud.py \
  --cloud       ~/photogrammetry_NBV/data/photogrammetry/phase2_run_<timestamp>/final/dense_cloud.ply \
  --images-bin  ~/photogrammetry_NBV/data/photogrammetry/phase2_run_<timestamp>/colmap/sparse/0/images.bin \
  --metadata    ~/photogrammetry_NBV/data/photogrammetry/phase2_run_<timestamp>/seed/metadata \
                ~/photogrammetry_NBV/data/photogrammetry/phase2_run_<timestamp>/adaptive/metadata \
  --output      aligned_dense.ply \
  --bbox-center 0 8 0 \
  --bbox-half-extent 6.0
```

Run the same command for `seed_sparse_cloud.ply` (change `--cloud` and `--output`).

The script prints RMS and maximum residuals on the camera correspondences. A maximum residual under ~5 cm indicates a reliable alignment. High residuals typically mean the COLMAP reconstruction fragmented into multiple components, which reduces the number of usable correspondence pairs.

### Bounding box crop

The `--bbox-center` and `--bbox-half-extent` arguments crop the aligned cloud to a cube around the rock, removing background points and COLMAP noise that would otherwise corrupt coverage and distance metrics. The half-extent is also stored in the phase2 manifest (`eval_bbox_half_extent`) so it stays consistent between runs.

### Viewing results in CloudCompare

```bash
sudo apt install cloudcompare

# Export the final sparse model to PLY if not already done
/home/dreamslab/colmap/build/src/colmap/exe/colmap model_converter \
  --input_path  ~/photogrammetry_NBV/data/photogrammetry/phase2_run_<timestamp>/colmap/sparse/0 \
  --output_path ~/photogrammetry_NBV/data/photogrammetry/phase2_run_<timestamp>/colmap/sparse_cloud.ply \
  --output_type PLY

# Open both aligned clouds together
cloudcompare aligned_seed_sparse.ply aligned_dense.ply
```

In CloudCompare, use `Tools → Distances → Cloud/Cloud Dist` for cloud-to-cloud distance, and `Tools → Distances → Cloud/Mesh Dist` to compare against the ground-truth mesh once both are in the same frame.

---

## Visualisation Tools

### Candidate pipeline visualiser

Shows all four stages of the candidate generation and filtering pipeline in a single interactive 3D plot. Can optionally overlay the candidates actually selected during a real Phase 2 run.

```bash
# Generation and filtering only
python3 ~/photogrammetry-nbv/src/photogrammetry_nbv/scripts/visualize_candidates.py

# With actual selected candidates from a run
python3 ~/photogrammetry-nbv/src/photogrammetry_nbv/scripts/visualize_candidates.py \
  --run-dir ~/photogrammetry_NBV/data/photogrammetry/phase2_run_<timestamp>
```

Layers shown:

| Colour | Layer |
|---|---|
| Light grey | Raw Fibonacci hemisphere (200 candidates) |
| Blue | Feasible after altitude + travel + spacing filter |
| Green | Final diversity-cropped pool (95 candidates) |
| Orange stars | Actually selected NBV candidates (if `--run-dir` given) |
| Gold star | Rock centre |

Click any legend entry to toggle that layer.

---

## Configuration Reference

### `photogrammetry_init/config/mission.yaml` — Phase 1

Controls the ring orbit geometry, flight parameters, timing, and topic names.

### `photogrammetry_nbv/config/phase2_mission.yaml` — Phase 2

Controls the image budget, candidate hemisphere geometry, altitude limits, COLMAP binary path, and evaluation bbox. The `eval_bbox_half_extent` parameter is written into the phase2 manifest for use by the evaluation scripts.

### `photogrammetry_nbv/config/colmap.yaml` — COLMAP pipeline

Controls feature extraction (camera model, single-camera mode), matcher type (exhaustive by default), mapper thresholds, weak region extraction parameters (severity weights, top fraction, max regions), and dense reconstruction quality.

### `photogrammetry_nbv/config/scoring.yaml` — NBV scorer

Controls which scorer is used and its weights. The default `weighted_sum` scorer exposes four independently-weighted terms. To add a new scorer, implement the `BaseScorer` interface, register it in `scorers/__init__.py`, and update the `name` field in this file.

