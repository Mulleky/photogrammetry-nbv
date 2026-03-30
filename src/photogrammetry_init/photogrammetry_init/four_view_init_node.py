#!/usr/bin/env python3
"""
Photogrammetry-only initialization mission for PX4 + ROS 2.

Mission profile:
1. Stream OffboardControlMode + TrajectorySetpoint.
2. Switch to offboard mode and arm.
3. Take off vertically to a fixed altitude.
4. Fly a configurable CCW ring orbit around the object, capturing one image per stop.
5. Hover or land.

Notes:
- Coordinates are interpreted in PX4 NED: +x North, +y East, +z Down.
- The ring starts at the drone's spawn bearing to the object and steps CCW.
- Topic names are parameters because PX4 DDS topic suffixes differ across setups.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float64


class MissionPhase(Enum):
    WAIT_FOR_TOPICS = auto()
    STREAM_SETPOINTS = auto()
    TAKEOFF = auto()
    MOVE_TO_VIEWPOINT = auto()
    SETTLE = auto()
    CAPTURE = auto()
    FINISHED = auto()


@dataclass
class Viewpoint:
    label: str
    x: float
    y: float
    z: float
    yaw: float


class FourViewInitNode(Node):
    def __init__(self) -> None:
        super().__init__('four_view_init_node')
        self._declare_parameters()
        self._load_parameters()

        self.qos_px4 = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.qos_default = QoSProfile(depth=10)

        self.offboard_pub = self.create_publisher(
            OffboardControlMode,
            self.offboard_control_mode_topic,
            self.qos_px4,
        )
        self.trajectory_pub = self.create_publisher(
            TrajectorySetpoint,
            self.trajectory_setpoint_topic,
            self.qos_px4,
        )
        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand,
            self.vehicle_command_topic,
            self.qos_px4,
        )
        self.gimbal_pitch_pub = self.create_publisher(Float64, self.gimbal_pitch_topic, self.qos_default)
        self.gimbal_yaw_pub = self.create_publisher(Float64, self.gimbal_yaw_topic, self.qos_default)

        self.local_pos_sub = self.create_subscription(
            VehicleLocalPosition,
            self.local_position_topic,
            self._local_position_cb,
            self.qos_px4,
        )
        self.vehicle_status_sub = self.create_subscription(
            VehicleStatus,
            self.vehicle_status_topic,
            self._vehicle_status_cb,
            self.qos_px4,
        )
        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self._image_cb,
            qos_profile_sensor_data,
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self._camera_info_cb,
            qos_profile_sensor_data,
        )

        self.phase = MissionPhase.WAIT_FOR_TOPICS
        self.offboard_setpoint_counter = 0
        self.current_target = [0.0, 0.0, -self.takeoff_altitude]
        self.current_target_yaw = 0.0
        self.home_xy: Optional[Tuple[float, float]] = None
        self.local_position: Optional[VehicleLocalPosition] = None
        self.vehicle_status: Optional[VehicleStatus] = None
        self.latest_image: Optional[Image] = None
        self.latest_camera_info: Optional[CameraInfo] = None
        self.viewpoints: List[Viewpoint] = []
        self.view_index = 0
        self.settle_started_at: Optional[float] = None
        self.capture_count = 0
        self.mode_sent = False
        self.arm_sent = False
        self.land_sent = False

        self._prepare_output_dirs()
        self._write_run_manifest()

        timer_period = 1.0 / self.offboard_rate_hz
        self.timer = self.create_timer(timer_period, self._run_mission)
        self.get_logger().info(f'four_view_init_node started ({self.ring_image_count}-image ring orbit). Data will be written to: {self.run_dir}')
        self.get_logger().info(
            f'Subscribed topics -> local_position: {self.local_position_topic}, '
            f'vehicle_status: {self.vehicle_status_topic}, image: {self.image_topic}, '
            f'camera_info: {self.camera_info_topic}'
        )

    def _declare_parameters(self) -> None:
        params = [
            ('offboard_rate_hz', 10.0),
            ('setpoint_warmup_cycles', 15),
            ('takeoff_altitude', 6.0),
            ('capture_radius', 5.0),
            ('ring_image_count', 12),
            ('position_tolerance_xy', 0.35),
            ('position_tolerance_z', 0.35),
            ('settle_time_sec', 2.0),
            ('object_x', 0.0),
            ('object_y', 0.0),
            ('land_after_capture', False),
            ('gimbal_pitch_rad', -0.35),
            ('gimbal_yaw_rad', 0.0),
            ('image_format', 'jpg'),
            ('output_root', '~/photogrammetry_NBV/data/photogrammetry'),
            ('offboard_control_mode_topic', '/fmu/in/offboard_control_mode'),
            ('trajectory_setpoint_topic', '/fmu/in/trajectory_setpoint'),
            ('vehicle_command_topic', '/fmu/in/vehicle_command'),
            ('local_position_topic', '/fmu/out/vehicle_local_position_v1'),
            ('vehicle_status_topic', '/fmu/out/vehicle_status_v1'),
            ('image_topic', '/rgbd/image'),
            ('camera_info_topic', '/rgbd/camera_info'),
            ('gimbal_pitch_topic', '/gimbal/pitch'),
            ('gimbal_yaw_topic', '/gimbal/yaw'),
        ]
        for name, default in params:
            self.declare_parameter(name, default)

    def _load_parameters(self) -> None:
        gp = self.get_parameter
        self.offboard_rate_hz = float(gp('offboard_rate_hz').value)
        self.setpoint_warmup_cycles = int(gp('setpoint_warmup_cycles').value)
        self.takeoff_altitude = float(gp('takeoff_altitude').value)
        self.capture_radius = float(gp('capture_radius').value)
        self.ring_image_count = int(gp('ring_image_count').value)
        self.position_tolerance_xy = float(gp('position_tolerance_xy').value)
        self.position_tolerance_z = float(gp('position_tolerance_z').value)
        self.settle_time_sec = float(gp('settle_time_sec').value)
        self.object_x = float(gp('object_x').value)
        self.object_y = float(gp('object_y').value)
        self.land_after_capture = bool(gp('land_after_capture').value)
        self.gimbal_pitch_rad = float(gp('gimbal_pitch_rad').value)
        self.gimbal_yaw_rad = float(gp('gimbal_yaw_rad').value)
        self.image_format = str(gp('image_format').value).lower()
        self.output_root = os.path.expanduser(str(gp('output_root').value))

        self.offboard_control_mode_topic = str(gp('offboard_control_mode_topic').value)
        self.trajectory_setpoint_topic = str(gp('trajectory_setpoint_topic').value)
        self.vehicle_command_topic = str(gp('vehicle_command_topic').value)
        self.local_position_topic = str(gp('local_position_topic').value)
        self.vehicle_status_topic = str(gp('vehicle_status_topic').value)
        self.image_topic = str(gp('image_topic').value)
        self.camera_info_topic = str(gp('camera_info_topic').value)
        self.gimbal_pitch_topic = str(gp('gimbal_pitch_topic').value)
        self.gimbal_yaw_topic = str(gp('gimbal_yaw_topic').value)

    def _prepare_output_dirs(self) -> None:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.run_dir = Path(self.output_root).expanduser() / f'run_{timestamp}'
        self.images_dir = self.run_dir / 'images'
        self.metadata_dir = self.run_dir / 'metadata'
        self.calibration_dir = self.run_dir / 'calibration'
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.calibration_dir.mkdir(parents=True, exist_ok=True)

    def _write_run_manifest(self) -> None:
        manifest = {
            'mission_type': 'photogrammetry_ring_init',
            'frame': 'PX4_NED',
            'object_x': self.object_x,
            'object_y': self.object_y,
            'takeoff_altitude_m': self.takeoff_altitude,
            'capture_radius_m': self.capture_radius,
            'ring_image_count': self.ring_image_count,
            'land_after_capture': self.land_after_capture,
        }
        with open(self.run_dir / 'manifest.json', 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)

    def _local_position_cb(self, msg: VehicleLocalPosition) -> None:
        self.local_position = msg
        if self.home_xy is None and math.isfinite(msg.x) and math.isfinite(msg.y):
            self.home_xy = (float(msg.x), float(msg.y))
            self.current_target[0] = float(msg.x)
            self.current_target[1] = float(msg.y)
            self.current_target[2] = -self.takeoff_altitude
            self.current_target_yaw = self._yaw_to_object(float(msg.x), float(msg.y))
            self.viewpoints = self._build_viewpoints()
            self.get_logger().info(f'Home XY locked at ({msg.x:.2f}, {msg.y:.2f}).')

    def _vehicle_status_cb(self, msg: VehicleStatus) -> None:
        self.vehicle_status = msg

    def _image_cb(self, msg: Image) -> None:
        first_image = self.latest_image is None
        self.latest_image = msg
        if first_image:
            self.get_logger().info('First RGB image received.')

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        first_camera_info = self.latest_camera_info is None
        self.latest_camera_info = msg
        if first_camera_info:
            self.get_logger().info('First CameraInfo message received.')
        calib_path = self.calibration_dir / 'camera_info_latest.json'
        payload = {
            'width': msg.width,
            'height': msg.height,
            'distortion_model': msg.distortion_model,
            'd': list(msg.d),
            'k': list(msg.k),
            'r': list(msg.r),
            'p': list(msg.p),
        }
        with open(calib_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)

    def _build_viewpoints(self) -> List[Viewpoint]:
        assert self.home_xy is not None
        home_x, home_y = self.home_xy
        z = -self.takeoff_altitude
        # Start angle: bearing from object to drone spawn (so first stop is near spawn)
        start_angle = math.atan2(home_y - self.object_y, home_x - self.object_x)
        step = 2.0 * math.pi / self.ring_image_count
        viewpoints: List[Viewpoint] = []
        for i in range(self.ring_image_count):
            angle = start_angle - i * step  # CCW in NED (decreasing atan2)
            x = self.object_x + self.capture_radius * math.cos(angle)
            y = self.object_y + self.capture_radius * math.sin(angle)
            yaw = self._yaw_to_object(x, y)
            viewpoints.append(Viewpoint(label=f'ring_{i:02d}', x=x, y=y, z=z, yaw=yaw))
        return viewpoints

    def _yaw_to_object(self, x: float, y: float) -> float:
        dx = self.object_x - x
        dy = self.object_y - y
        return float(math.atan2(dy, dx))

    def _run_mission(self) -> None:
        self._publish_offboard_control_mode()
        self._publish_setpoint(self.current_target, self.current_target_yaw)
        self._publish_gimbal_commands()

        if self.phase == MissionPhase.WAIT_FOR_TOPICS:
            if self.local_position is None:
                return

            if not self.viewpoints:
                return

            if self.latest_image is None:
                self.get_logger().warn(
                    'No RGB image yet; proceeding to takeoff and will wait at the first capture point.',
                    throttle_duration_sec=5.0,
                )

            self.phase = MissionPhase.STREAM_SETPOINTS
            self.get_logger().info('PX4 position ready. Streaming warmup setpoints before entering offboard.')
            return

        if self.phase == MissionPhase.STREAM_SETPOINTS:
            self.offboard_setpoint_counter += 1
            if self.offboard_setpoint_counter >= self.setpoint_warmup_cycles:
                if not self.mode_sent:
                    self._publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                    self.mode_sent = True
                    self.get_logger().info('Sent offboard mode command.')
                if not self.arm_sent:
                    self._publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                    self.arm_sent = True
                    self.get_logger().info('Sent arm command.')
                self.phase = MissionPhase.TAKEOFF
                self.get_logger().info(f'Transitioning to TAKEOFF. Ring of {self.ring_image_count} viewpoints queued.')
            return

        if self.phase == MissionPhase.TAKEOFF:
            if self._at_target(self.current_target):
                self.phase = MissionPhase.MOVE_TO_VIEWPOINT
                self.get_logger().info('Takeoff target reached. Starting ring capture route.')
            return

        if self.phase == MissionPhase.MOVE_TO_VIEWPOINT:
            if self.view_index >= len(self.viewpoints):
                self.phase = MissionPhase.FINISHED
                self.get_logger().info('All viewpoints completed.')
                return
            vp = self.viewpoints[self.view_index]
            self.current_target = [vp.x, vp.y, vp.z]
            self.current_target_yaw = vp.yaw
            if self._at_target(self.current_target):
                self.phase = MissionPhase.SETTLE
                self.settle_started_at = time.time()
                self.get_logger().info(f'Reached {vp.label} viewpoint. Settling before capture.')
            return

        if self.phase == MissionPhase.SETTLE:
            if self.settle_started_at is None:
                self.settle_started_at = time.time()
                return
            if time.time() - self.settle_started_at >= self.settle_time_sec:
                self.phase = MissionPhase.CAPTURE
            return

        if self.phase == MissionPhase.CAPTURE:
            vp = self.viewpoints[self.view_index]
            if self.latest_image is None:
                self.get_logger().warn(
                    f'No RGB image available yet at {vp.label} viewpoint; waiting before capture.',
                    throttle_duration_sec=5.0,
                )
                return
            success = self._save_capture(vp)
            if success:
                self.capture_count += 1
                self.view_index += 1
                self.phase = MissionPhase.MOVE_TO_VIEWPOINT
            else:
                self.get_logger().warn('Capture failed; retrying on next cycle.', throttle_duration_sec=2.0)
            return

        if self.phase == MissionPhase.FINISHED:
            if self.land_after_capture and not self.land_sent:
                self._publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                self.land_sent = True
                self.get_logger().info('Sent landing command. Mission complete.')
            return

    def _publish_offboard_control_mode(self) -> None:
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = self._timestamp_us()
        self.offboard_pub.publish(msg)

    def _publish_setpoint(self, position: List[float], yaw: float) -> None:
        msg = TrajectorySetpoint()
        msg.position = [float(position[0]), float(position[1]), float(position[2])]
        msg.yaw = float(yaw)
        msg.timestamp = self._timestamp_us()
        self.trajectory_pub.publish(msg)

    def _publish_vehicle_command(self, command: int, param1: float = 0.0, param2: float = 0.0) -> None:
        msg = VehicleCommand()
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.command = int(command)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = self._timestamp_us()
        self.vehicle_command_pub.publish(msg)

    def _publish_gimbal_commands(self) -> None:
        pitch_msg = Float64()
        pitch_msg.data = float(self.gimbal_pitch_rad)
        yaw_msg = Float64()
        yaw_msg.data = float(self.gimbal_yaw_rad)
        self.gimbal_pitch_pub.publish(pitch_msg)
        self.gimbal_yaw_pub.publish(yaw_msg)

    def _at_target(self, target: List[float]) -> bool:
        if self.local_position is None:
            return False
        dx = float(self.local_position.x) - float(target[0])
        dy = float(self.local_position.y) - float(target[1])
        dz = float(self.local_position.z) - float(target[2])
        return abs(dx) <= self.position_tolerance_xy and abs(dy) <= self.position_tolerance_xy and abs(dz) <= self.position_tolerance_z

    def _save_capture(self, vp: Viewpoint) -> bool:
        if self.latest_image is None:
            self.get_logger().warn('Capture skipped: latest_image is still None.')
            return False
        try:
            image_np = self._image_msg_to_cv(self.latest_image)
        except Exception as exc:  # pragma: no cover
            self.get_logger().error(
                f'Image conversion failed: {exc}. '
                f'encoding={self.latest_image.encoding}, '
                f'width={self.latest_image.width}, height={self.latest_image.height}'
            )
            return False

        stamp = self._image_timestamp_string(self.latest_image)
        ext = 'png' if self.image_format == 'png' else 'jpg'
        filename = f'{self.capture_count:02d}_{vp.label}_{stamp}.{ext}'
        image_path = self.images_dir / filename
        write_ok = cv2.imwrite(str(image_path), image_np)
        if not write_ok:
            self.get_logger().error(f'Failed to write image to {image_path}')
            return False

        meta = {
            'capture_index': self.capture_count,
            'view_label': vp.label,
            'image_file': filename,
            'timestamp_ros_sec': int(self.latest_image.header.stamp.sec),
            'timestamp_ros_nanosec': int(self.latest_image.header.stamp.nanosec),
            'target_position_ned_m': {'x': vp.x, 'y': vp.y, 'z': vp.z},
            'target_yaw_rad': vp.yaw,
            'object_position_ned_m': {'x': self.object_x, 'y': self.object_y},
            'vehicle_position_ned_m': self._vehicle_position_dict(),
            'vehicle_heading_rad': float(getattr(self.local_position, 'heading', float('nan'))) if self.local_position is not None else None,
            'gimbal_pitch_rad': self.gimbal_pitch_rad,
            'gimbal_yaw_rad': self.gimbal_yaw_rad,
        }
        if self.latest_camera_info is not None:
            meta['camera_info'] = {
                'width': self.latest_camera_info.width,
                'height': self.latest_camera_info.height,
                'distortion_model': self.latest_camera_info.distortion_model,
                'd': list(self.latest_camera_info.d),
                'k': list(self.latest_camera_info.k),
                'r': list(self.latest_camera_info.r),
                'p': list(self.latest_camera_info.p),
            }

        meta_path = self.metadata_dir / f'{self.capture_count:02d}_{vp.label}_{stamp}.json'
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)

        self.get_logger().info(f'Saved {vp.label} capture -> {image_path.name}')
        return True

    def _vehicle_position_dict(self) -> Optional[Dict[str, float]]:
        if self.local_position is None:
            return None
        return {
            'x': float(self.local_position.x),
            'y': float(self.local_position.y),
            'z': float(self.local_position.z),
        }

    def _image_msg_to_cv(self, msg: Image) -> np.ndarray:
        enc = msg.encoding.lower()
        if enc not in {'rgb8', 'bgr8', 'rgba8', 'bgra8', 'mono8'}:
            raise ValueError(f'Unsupported encoding: {msg.encoding}')

        arr = np.frombuffer(msg.data, dtype=np.uint8)
        if enc == 'mono8':
            img = arr.reshape((msg.height, msg.width))
        elif enc in {'rgb8', 'bgr8'}:
            img = arr.reshape((msg.height, msg.width, 3))
            if enc == 'rgb8':
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img = arr.reshape((msg.height, msg.width, 4))
            if enc == 'rgba8':
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def _image_timestamp_string(self, msg: Image) -> str:
        return f'{int(msg.header.stamp.sec):010d}_{int(msg.header.stamp.nanosec):09d}'

    def _timestamp_us(self) -> int:
        return int(self.get_clock().now().nanoseconds / 1000)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FourViewInitNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
