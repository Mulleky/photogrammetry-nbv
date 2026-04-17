#!/usr/bin/env python3
"""
colmap_rviz_publisher.py
========================
Publishes live COLMAP sparse point cloud and the raw RGB camera image
to RViz2 so you can watch reconstruction progress in real time.

Topics published
----------------
/colmap/sparse_cloud   sensor_msgs/PointCloud2   – COLMAP sparse points (RGB coloured)
/colmap/image_raw      sensor_msgs/Image         – passthrough of the drone's RGB image

How it works
------------
1. Subscribes to the Gazebo-bridged camera topic (default /rgbd/image).
2. Watches the COLMAP sparse output directory for changes to points3D.bin.
   After each NBV iteration the existing colmap_worker_client writes a new
   sparse model there; this node detects the update via mtime polling and
   re-publishes the cloud without touching your main controller.
3. Publishes a PointCloud2 in the 'map' frame so RViz2 doesn't need a TF.

Usage (add to your launch file OR run standalone)
-------------------------------------------------
    ros2 run photogrammetry_nbv colmap_rviz_publisher \
        --ros-args \
        -p sparse_dir:=/path/to/unified_run_XYZ/colmap/sparse/0 \
        -p image_topic:=/rgbd/image \
        -p poll_interval_s:=2.0

Parameters
----------
sparse_dir       : str   – path to the COLMAP sparse/0 directory that
                           contains points3D.bin  (REQUIRED)
image_topic      : str   – source camera topic   (default: /rgbd/image)
poll_interval_s  : float – how often to check for a new sparse model (s)
                           (default: 2.0)
frame_id         : str   – TF frame for the cloud (default: map)
"""

import struct
import time
import os
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header


# ---------------------------------------------------------------------------
# COLMAP binary reader helpers
# ---------------------------------------------------------------------------

def _read_next_bytes(f, num_bytes, fmt):
    data = f.read(num_bytes)
    return struct.unpack("<" + fmt, data)


def read_points3D_binary(path: str):
    """
    Parse a COLMAP points3D.bin file.

    Returns
    -------
    xyz  : (N, 3) float32 – 3-D positions
    rgb  : (N, 3) uint8   – colours (0-255)
    """
    pts_xyz = []
    pts_rgb = []

    with open(path, "rb") as f:
        num_points = _read_next_bytes(f, 8, "Q")[0]
        for _ in range(num_points):
            # point3D_id (Q), xyz (3d), rgb (3B), error (d), track_length (Q)
            vals = _read_next_bytes(f, 8 + 24 + 3 + 8 + 8, "Q3d3BdQ")
            pid, x, y, z, r, g, b, error, track_len = vals
            pts_xyz.append([x, y, z])
            pts_rgb.append([r, g, b])
            # skip track elements: each is (image_id uint32, point2D_idx uint32)
            f.read(8 * track_len)

    if not pts_xyz:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    return (
        np.array(pts_xyz, dtype=np.float32),
        np.array(pts_rgb, dtype=np.uint8),
    )


def xyz_rgb_to_pointcloud2(xyz: np.ndarray, rgb: np.ndarray,
                            frame_id: str, stamp) -> PointCloud2:
    """
    Convert Nx3 float32 xyz + Nx3 uint8 rgb arrays into a PointCloud2 message
    with fields x, y, z, rgb (packed as float).
    """
    N = xyz.shape[0]

    # Pack rgb into a single float (RVIZ convention: BGRA uint32 cast to float)
    r = rgb[:, 0].astype(np.uint32)
    g = rgb[:, 1].astype(np.uint32)
    b = rgb[:, 2].astype(np.uint32)
    rgb_packed = (r << 16) | (g << 8) | b
    rgb_float = rgb_packed.view(np.float32)

    # Interleave: x y z rgb  (4 bytes each → 16 bytes per point)
    data = np.zeros((N, 4), dtype=np.float32)
    data[:, 0] = xyz[:, 0]
    data[:, 1] = xyz[:, 1]
    data[:, 2] = xyz[:, 2]
    data[:, 3] = rgb_float

    msg = PointCloud2()
    msg.header = Header(frame_id=frame_id)
    msg.header.stamp = stamp
    msg.height = 1
    msg.width = N
    msg.is_dense = True
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = 16 * N

    msg.fields = [
        PointField(name="x",   offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name="y",   offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name="z",   offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg.data = data.tobytes()
    return msg


# ---------------------------------------------------------------------------
# ROS 2 Node
# ---------------------------------------------------------------------------

class ColmapRvizPublisher(Node):

    def __init__(self):
        super().__init__("colmap_rviz_publisher")

        # ── parameters ─────────────────────────────────────────────────────
        self.declare_parameter("sparse_dir", "")
        self.declare_parameter("image_topic", "/rgbd/image")
        self.declare_parameter("poll_interval_s", 2.0)
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("base_dir", "")

        self._sparse_dir = self.get_parameter("sparse_dir").value
        self._image_topic = self.get_parameter("image_topic").value
        self._poll_s = self.get_parameter("poll_interval_s").value
        self._frame_id = self.get_parameter("frame_id").value
        self._base_dir = self.get_parameter("base_dir").value

        # auto-discover latest unified_run_* when only base_dir is given
        self._auto_discover = bool(self._base_dir) and not self._sparse_dir

        if not self._sparse_dir and not self._base_dir:
            self.get_logger().error(
                "Neither 'sparse_dir' nor 'base_dir' is set. Pass one via "
                "--ros-args -p base_dir:=~/photogrammetry_NBV/data/photogrammetry"
            )

        self._points3D_bin = (
            Path(self._sparse_dir) / "points3D.bin"
            if self._sparse_dir else None
        )
        self._last_mtime: float = 0.0
        self._last_point_count: int = 0

        # ── QoS ────────────────────────────────────────────────────────────
        # Keep last + reliable so RViz2 always gets the most recent cloud
        # even if it connects after the first publish.
        latching_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # ── publishers ─────────────────────────────────────────────────────
        self._cloud_pub = self.create_publisher(
            PointCloud2, "/colmap/sparse_cloud", latching_qos
        )
        self._image_pub = self.create_publisher(
            Image, "/colmap/image_raw", 10
        )

        # ── subscriber ─────────────────────────────────────────────────────
        self._image_sub = self.create_subscription(
            Image,
            self._image_topic,
            self._image_cb,
            10,
        )

        # ── polling timer ──────────────────────────────────────────────────
        self._timer = self.create_timer(self._poll_s, self._poll_cb)

        self.get_logger().info(
            f"colmap_rviz_publisher started\n"
            f"  sparse_dir   : {self._sparse_dir or '(auto-discover)'}\n"
            f"  base_dir     : {self._base_dir or '(not set)'}\n"
            f"  image_topic  : {self._image_topic}\n"
            f"  poll_interval: {self._poll_s} s\n"
            f"  frame_id     : {self._frame_id}\n"
            + ("Scanning for latest unified_run_* …" if self._auto_discover
               else "Watching for points3D.bin updates …")
        )

    # ── callbacks ──────────────────────────────────────────────────────────

    def _image_cb(self, msg: Image):
        """Re-publish the raw RGB image so RViz2 can display it."""
        self._image_pub.publish(msg)

    def _try_discover(self):
        """Find the most recently created unified_run_* dir under base_dir."""
        base = Path(self._base_dir).expanduser()
        if not base.exists():
            return
        candidates = sorted(
            [p for p in base.iterdir()
             if p.is_dir() and p.name.startswith("unified_run_")],
            reverse=True,  # lexicographic desc = newest timestamp first
        )
        if not candidates:
            return
        latest = candidates[0]
        sparse = latest / "colmap" / "sparse" / "0"
        self._sparse_dir = str(sparse)
        self._points3D_bin = sparse / "points3D.bin"
        self.get_logger().info(
            f"Auto-discovered run dir: {latest.name} — "
            f"watching {self._points3D_bin}"
        )

    def _poll_cb(self):
        """Check if points3D.bin has been updated; if so, re-publish cloud."""
        if self._points3D_bin is None and self._auto_discover:
            self._try_discover()

        if self._points3D_bin is None or not self._points3D_bin.exists():
            return

        try:
            mtime = self._points3D_bin.stat().st_mtime
        except OSError:
            return

        if mtime <= self._last_mtime:
            return  # file unchanged

        self._last_mtime = mtime
        self.get_logger().info("points3D.bin changed – reading new cloud …")

        try:
            xyz, rgb = read_points3D_binary(str(self._points3D_bin))
        except Exception as exc:
            self.get_logger().warn(f"Failed to read points3D.bin: {exc}")
            return

        N = xyz.shape[0]
        if N == 0:
            self.get_logger().warn("points3D.bin has 0 points – skipping.")
            return

        stamp = self.get_clock().now().to_msg()
        cloud_msg = xyz_rgb_to_pointcloud2(xyz, rgb, self._frame_id, stamp)
        self._cloud_pub.publish(cloud_msg)

        delta = N - self._last_point_count
        self._last_point_count = N
        self.get_logger().info(
            f"Published sparse cloud: {N} points "
            f"({'+'if delta>=0 else ''}{delta} since last update)"
        )


# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = ColmapRvizPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
