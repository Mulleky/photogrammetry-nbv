# photogrammetry_init

Minimal ROS 2 Python package for a **photogrammetry-only** bootstrap mission:

1. Start PX4 SITL + Gazebo.
2. Start the Micro XRCE-DDS Agent.
3. Bridge Gazebo camera topics into ROS 2.
4. Stream PX4 offboard setpoints.
5. Switch to offboard, arm, and take off.
6. Visit 4 deterministic viewpoints around a known object.
7. Save 4 RGB images plus metadata for later photogrammetry.

## Workspace layout

```text
photogrammetry_NBV/
├── src/
│   └── photogrammetry_init/
│       ├── config/
│       │   ├── bridge.yaml
│       │   └── mission.yaml
│       ├── launch/
│       │   └── photogrammetry_init.launch.py
│       ├── photogrammetry_init/
│       │   ├── __init__.py
│       │   └── four_view_init_node.py
│       ├── scripts/
│       │   └── four_view_init_node.py
│       ├── package.xml
│       ├── setup.py
│       └── setup.cfg
└── data/
    └── photogrammetry/
        └── run_<timestamp>/
```

## Build

```bash
cd ~/photogrammetry_NBV
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

## Launch

```bash
ros2 launch photogrammetry_init photogrammetry_init.launch.py \
  px4_autopilot_path:=~/PX4-Autopilot \
  px4_gz_world:=sample_15016 \
  px4_make_target:=gz_px4_gsplat
```

## Important adjustments before first run

- If your PX4 ROS 2 topics do **not** use the `_v1` suffix, edit `config/mission.yaml`:
  - `/fmu/out/vehicle_local_position_v1` -> `/fmu/out/vehicle_local_position`
  - `/fmu/out/vehicle_status_v1` -> `/fmu/out/vehicle_status`
- If the Gazebo model is not named `px4_gsplat_0`, update the gimbal bridge topic names in `config/bridge.yaml`.
- If your system requires QGroundControl to allow arming, connect QGC before running the node.
- The node assumes PX4 **NED** coordinates.

## Output

Each run creates:
- `images/` with 4 saved RGB frames
- `metadata/` with 1 JSON file per image
- `calibration/camera_info_latest.json`
- `manifest.json`
