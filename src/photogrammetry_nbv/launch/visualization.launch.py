"""
visualization.launch.py
=======================
Add this launch file to your workspace and call it from a 4th terminal
AFTER the main unified_mission.launch.py is already running.

It starts:
  1. colmap_rviz_publisher  – reads points3D.bin + relays camera image
  2. rviz2                  – pre-configured split view

Usage
-----
    # Terminal 4 — visualization (source your workspace first)
    source /opt/ros/humble/setup.bash
    source ~/photogrammetry-covisibility/install/setup.bash

    ros2 launch photogrammetry_nbv visualization.launch.py \\
        run_dir:=~/photogrammetry_NBV/data/photogrammetry/unified_run_TIMESTAMP

Where unified_run_TIMESTAMP is the folder created by the mission node.
The sparse directory is inferred automatically as <run_dir>/colmap/sparse/0.

Optional overrides
------------------
    poll_interval_s:=3.0      # how often to check for a new sparse model
    image_topic:=/rgbd/image  # camera source topic
    frame_id:=map             # TF frame for the cloud
    rviz_config:=/full/path/to/nbv_visualization.rviz
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _make_nodes(context, *args, **kwargs):
    base_dir  = LaunchConfiguration("base_dir").perform(context)
    poll_s    = LaunchConfiguration("poll_interval_s").perform(context)
    img_topic = LaunchConfiguration("image_topic").perform(context)
    frame_id  = LaunchConfiguration("frame_id").perform(context)
    rviz_cfg  = LaunchConfiguration("rviz_config").perform(context)

    publisher_node = Node(
        package="photogrammetry_nbv",
        executable="colmap_rviz_publisher",
        name="colmap_rviz_publisher",
        output="screen",
        parameters=[
            {
                "base_dir":         os.path.expanduser(base_dir),
                "image_topic":      img_topic,
                "poll_interval_s":  float(poll_s),
                "frame_id":         frame_id,
            }
        ],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_cfg],
    )

    return [publisher_node, rviz_node]


def generate_launch_description():
    default_rviz = PathJoinSubstitution([
        FindPackageShare("photogrammetry_nbv"),
        "rviz", "nbv_visualization.rviz",
    ])

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "base_dir",
                default_value=os.path.expanduser(
                    "~/photogrammetry_NBV/data/photogrammetry"
                ),
                description=(
                    "Parent directory containing unified_run_* folders. "
                    "The publisher auto-discovers the latest one."
                ),
            ),
            DeclareLaunchArgument(
                "poll_interval_s",
                default_value="2.0",
                description="Seconds between checks for a new sparse model",
            ),
            DeclareLaunchArgument(
                "image_topic",
                default_value="/rgbd/image",
                description="ROS 2 topic carrying the drone's RGB image",
            ),
            DeclareLaunchArgument(
                "frame_id",
                default_value="map",
                description="TF frame id attached to the published PointCloud2",
            ),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=default_rviz,
                description="Full path to the .rviz config file",
            ),
            OpaqueFunction(function=_make_nodes),
        ]
    )
