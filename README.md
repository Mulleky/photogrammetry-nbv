# Photogrammetry NBV

Photogrammetry-based Next-Best-View planning for active shape estimation.

## Overview
This project explores NBV planning using photogrammetric reconstruction quality metrics, with applications in simulation and real-world data capture.

## Main components
- Data acquisition
- Image selection / NBV logic
- Photogrammetric reconstruction
- Reconstruction quality assessment
- Simulation and experimental pipelines

## Planned stack
- ROS 2
- Gazebo / PX4
- Python
- Metashape Professional

## Repository structure
- `scripts/` automation scripts
- `launch/` ROS 2 or simulation launch files
- `configs/` parameter/config files
- `docs/` notes, figures, manuscript support material

## Status
Work in progress.

## Commands
```
cd ~/photogrammetry-nbv
source /opt/ros/jazzy/setup.bash
source ~/workspaces/px4_msgs_ws/install/setup.bash
colcon build --symlink-install
source install/setup.bash

ros2 launch photogrammetry_init photogrammetry_init.launch.py
```
Phase 2

```
source /home/dreamslab/photogrammetry-nbv/install/setup.bash
ros2 launch photogrammetry_nbv phase2_online_mission.launch.py \
    start_px4:=false \
    start_xrce_agent:=false \
    start_bridge:=false
```


