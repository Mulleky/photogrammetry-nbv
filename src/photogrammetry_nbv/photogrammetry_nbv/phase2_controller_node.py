#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import rclpy
import yaml
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float64

from .candidate_filter import crop_to_target_count_diverse, filter_candidates
from .candidate_generator import generate_fibonacci_hemisphere
from .contracts import CandidateViewpoint, ScoreContext
from .metashape_worker_client import MetashapeWorkerClient
from .metrics_extractor import load_sparse_metrics
from .mission_logger import MissionLogger
from .run_context import prepare_phase2_run
from .scorers import SCORER_REGISTRY
from .seed_loader import infer_home_pose, load_seed_bundle


class Phase2State(Enum):
    WAIT_FOR_TOPICS = auto()
    WARMUP_STREAM = auto()
    BOOTSTRAP_PROJECT = auto()
    SCORE_NEXT_VIEW = auto()
    FLY_TO_VIEW = auto()
    SETTLE = auto()
    CAPTURE = auto()
    UPDATE_PROJECT = auto()
    RETURN_HOME = auto()
    LAND = auto()
    OFFLINE_DENSE_RECON = auto()
    FINISHED = auto()


class Phase2ControllerNode(Node):
    def __init__(self) -> None:
        super().__init__('phase2_controller_node')
        self._declare_parameters()
        self._load_parameters()

        self.qos_px4 = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.qos_default = QoSProfile(depth=10)

        self.offboard_pub = self.create_publisher(OffboardControlMode, self.offboard_control_mode_topic, self.qos_px4)
        self.trajectory_pub = self.create_publisher(TrajectorySetpoint, self.trajectory_setpoint_topic, self.qos_px4)
        self.vehicle_command_pub = self.create_publisher(VehicleCommand, self.vehicle_command_topic, self.qos_px4)
        self.gimbal_pitch_pub = self.create_publisher(Float64, self.gimbal_pitch_topic, self.qos_default)
        self.gimbal_yaw_pub = self.create_publisher(Float64, self.gimbal_yaw_topic, self.qos_default)

        self.local_pos_sub = self.create_subscription(VehicleLocalPosition, self.local_position_topic, self._local_position_cb, self.qos_px4)
        self.vehicle_status_sub = self.create_subscription(VehicleStatus, self.vehicle_status_topic, self._vehicle_status_cb, self.qos_px4)
        self.image_sub = self.create_subscription(Image, self.image_topic, self._image_cb, self.qos_default)
        self.camera_info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self._camera_info_cb, self.qos_default)

        self.phase = Phase2State.WAIT_FOR_TOPICS
        self.local_position: Optional[VehicleLocalPosition] = None
        self.vehicle_status: Optional[VehicleStatus] = None
        self.latest_image: Optional[Image] = None
        self.latest_camera_info: Optional[CameraInfo] = None

        self.home_pose = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.current_target = [0.0, 0.0, -self.takeoff_altitude]
        self.current_target_yaw = 0.0
        self.selected_candidate: Optional[CandidateViewpoint] = None
        self.settle_started_at: Optional[float] = None
        self.current_iteration = 0
        self.offboard_warmup_counter = 0
        self.mode_sent = False
        self.arm_sent = False
        self.land_sent = False
        self.project_bootstrapped = False
        self.images_used = 0
        self.visited_viewpoints: List[Dict] = []
        self.last_capture_image_name: Optional[str] = None

        seed_run_dir = Path(self.seed_run_dir).expanduser()
        output_root = Path(self.output_root).expanduser()
        self.paths, self.seed_manifest = prepare_phase2_run(seed_run_dir, output_root)
        self.seed_images, self.seed_metadata, _ = load_seed_bundle(seed_run_dir)
        self.home_pose = infer_home_pose(self.seed_metadata)
        self.current_target = [self.home_pose['x'], self.home_pose['y'], -self.takeoff_altitude]
        self.current_target_yaw = 0.0
        self.images_used = len(self.seed_images)
        for item in self.seed_metadata:
            self.visited_viewpoints.append({
                'position_ned_m': item.get('vehicle_position_ned_m', {}),
                'yaw_rad': item.get('target_yaw_rad', 0.0),
                'source': 'seed',
            })

        self.project_path = self.paths.metashape_dir / 'project.psx'
        self.worker = MetashapeWorkerClient(self.metashape_cmd, Path(self.metashape_script_dir).expanduser())
        self.logger = MissionLogger(self.paths.candidate_dir, self.paths.sparse_metrics_dir)
        self.scorer = self._build_scorer()

        self._write_phase2_manifest()
        self.timer = self.create_timer(1.0 / self.offboard_rate_hz, self._run)
        self.get_logger().info(f'phase2_controller_node initialized. Run dir: {self.paths.run_dir}')

    def _declare_parameters(self) -> None:
        params = [
            ('offboard_rate_hz', 10.0), ('setpoint_warmup_cycles', 15), ('takeoff_altitude', 6.0),
            ('position_tolerance_xy', 0.35), ('position_tolerance_z', 0.35), ('settle_time_sec', 2.0),
            ('image_budget', 20), ('raw_candidate_count', 200), ('target_candidate_count', 95),
            ('candidate_radius_m', 5.5), ('candidate_min_spacing_m', 1.0), ('min_elevation_deg', 10.0),
            ('max_elevation_deg', 80.0), ('rock_center_x', 0.0), ('rock_center_y', 0.0), ('rock_center_z', 0.0),
            ('min_altitude_ned', -20.0), ('max_altitude_ned', -0.1), ('max_candidate_travel_m', 30.0),
            ('land_after_budget', True), ('gimbal_pitch_rad', -0.35), ('gimbal_yaw_rad', 0.0), ('image_format', 'jpg'),
            ('seed_run_dir', '~/photogrammetry_NBV/data/photogrammetry/latest_seed'), ('output_root', '~/photogrammetry_NBV/data/photogrammetry'),
            ('metashape_cmd', 'metashape.sh'), ('metashape_script_dir', '~/photogrammetry_NBV/src/photogrammetry_nbv/metashape_scripts'),
            ('metashape_config_path', '~/photogrammetry_NBV/src/photogrammetry_nbv/config/metashape.yaml'), ('scoring_config_path', '~/photogrammetry_NBV/src/photogrammetry_nbv/config/scoring.yaml'),
            ('offboard_control_mode_topic', '/fmu/in/offboard_control_mode'), ('trajectory_setpoint_topic', '/fmu/in/trajectory_setpoint'),
            ('vehicle_command_topic', '/fmu/in/vehicle_command'), ('local_position_topic', '/fmu/out/vehicle_local_position_v1'),
            ('vehicle_status_topic', '/fmu/out/vehicle_status_v1'), ('image_topic', '/rgbd/image'), ('camera_info_topic', '/rgbd/camera_info'),
            ('gimbal_pitch_topic', '/gimbal/pitch'), ('gimbal_yaw_topic', '/gimbal/yaw'),
        ]
        for name, default in params:
            self.declare_parameter(name, default)

    def _load_parameters(self) -> None:
        gp = self.get_parameter
        self.offboard_rate_hz = float(gp('offboard_rate_hz').value)
        self.setpoint_warmup_cycles = int(gp('setpoint_warmup_cycles').value)
        self.takeoff_altitude = float(gp('takeoff_altitude').value)
        self.position_tolerance_xy = float(gp('position_tolerance_xy').value)
        self.position_tolerance_z = float(gp('position_tolerance_z').value)
        self.settle_time_sec = float(gp('settle_time_sec').value)
        self.image_budget = int(gp('image_budget').value)
        self.raw_candidate_count = int(gp('raw_candidate_count').value)
        self.target_candidate_count = int(gp('target_candidate_count').value)
        self.candidate_radius_m = float(gp('candidate_radius_m').value)
        self.candidate_min_spacing_m = float(gp('candidate_min_spacing_m').value)
        self.min_elevation_deg = float(gp('min_elevation_deg').value)
        self.max_elevation_deg = float(gp('max_elevation_deg').value)
        self.rock_center_x = float(gp('rock_center_x').value)
        self.rock_center_y = float(gp('rock_center_y').value)
        self.rock_center_z = float(gp('rock_center_z').value)
        self.min_altitude_ned = float(gp('min_altitude_ned').value)
        self.max_altitude_ned = float(gp('max_altitude_ned').value)
        self.max_candidate_travel_m = float(gp('max_candidate_travel_m').value)
        self.land_after_budget = bool(gp('land_after_budget').value)
        self.gimbal_pitch_rad = float(gp('gimbal_pitch_rad').value)
        self.gimbal_yaw_rad = float(gp('gimbal_yaw_rad').value)
        self.image_format = str(gp('image_format').value).lower()
        self.seed_run_dir = str(gp('seed_run_dir').value)
        self.output_root = str(gp('output_root').value)
        self.metashape_cmd = str(gp('metashape_cmd').value)
        self.metashape_script_dir = str(gp('metashape_script_dir').value)
        self.metashape_config_path = os.path.expanduser(str(gp('metashape_config_path').value))
        self.scoring_config_path = os.path.expanduser(str(gp('scoring_config_path').value))

        self.offboard_control_mode_topic = str(gp('offboard_control_mode_topic').value)
        self.trajectory_setpoint_topic = str(gp('trajectory_setpoint_topic').value)
        self.vehicle_command_topic = str(gp('vehicle_command_topic').value)
        self.local_position_topic = str(gp('local_position_topic').value)
        self.vehicle_status_topic = str(gp('vehicle_status_topic').value)
        self.image_topic = str(gp('image_topic').value)
        self.camera_info_topic = str(gp('camera_info_topic').value)
        self.gimbal_pitch_topic = str(gp('gimbal_pitch_topic').value)
        self.gimbal_yaw_topic = str(gp('gimbal_yaw_topic').value)

        with open(self.metashape_config_path, 'r', encoding='utf-8') as f:
            self.metashape_cfg = yaml.safe_load(f)
        with open(self.scoring_config_path, 'r', encoding='utf-8') as f:
            self.scoring_cfg = yaml.safe_load(f)

    def _build_scorer(self):
        scorer_name = str(self.scoring_cfg['scorer']['name'])
        scorer_cls = SCORER_REGISTRY[scorer_name]
        return scorer_cls(self.scoring_cfg['scorer'])

    def _write_phase2_manifest(self) -> None:
        manifest = {
            'mission_type': 'photogrammetry_phase2_sparse_nbv',
            'seed_run_dir': self.seed_run_dir,
            'image_budget': self.image_budget,
            'raw_candidate_count': self.raw_candidate_count,
            'target_candidate_count': self.target_candidate_count,
            'candidate_radius_m': self.candidate_radius_m,
            'rock_center_ned_m': {'x': self.rock_center_x, 'y': self.rock_center_y, 'z': self.rock_center_z},
            'scorer_name': self.scoring_cfg['scorer']['name'],
        }
        with open(self.paths.run_dir / 'phase2_manifest.json', 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)

    def _local_position_cb(self, msg: VehicleLocalPosition) -> None:
        self.local_position = msg

    def _vehicle_status_cb(self, msg: VehicleStatus) -> None:
        self.vehicle_status = msg

    def _image_cb(self, msg: Image) -> None:
        self.latest_image = msg

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        self.latest_camera_info = msg

    def _run(self) -> None:
        self._publish_offboard_control_mode()
        self._publish_setpoint(self.current_target, self.current_target_yaw)
        self._publish_gimbal_commands()

        if self.phase == Phase2State.WAIT_FOR_TOPICS:
            if self.local_position is None or self.latest_image is None:
                return
            self.phase = Phase2State.WARMUP_STREAM
            return

        if self.phase == Phase2State.WARMUP_STREAM:
            self.offboard_warmup_counter += 1
            if self.offboard_warmup_counter >= self.setpoint_warmup_cycles:
                if not self.mode_sent:
                    self._publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                    self.mode_sent = True
                if not self.arm_sent:
                    self._publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                    self.arm_sent = True
                self.phase = Phase2State.BOOTSTRAP_PROJECT
            return

        if self.phase == Phase2State.BOOTSTRAP_PROJECT:
            if not self._at_target([self.home_pose['x'], self.home_pose['y'], -self.takeoff_altitude]):
                return
            if not self.project_bootstrapped:
                bootstrap_json = self.paths.sparse_metrics_dir / 'metrics_iter_00.json'
                self.worker.bootstrap_project(self.project_path, self.seed_images, bootstrap_json, self.metashape_cfg)
                self.project_bootstrapped = True
            self.phase = Phase2State.SCORE_NEXT_VIEW
            return

        if self.phase == Phase2State.SCORE_NEXT_VIEW:
            if self.images_used >= self.image_budget:
                self.phase = Phase2State.RETURN_HOME
                return
            metrics_json = self.paths.sparse_metrics_dir / f'metrics_iter_{self.current_iteration:02d}.json'
            self.worker.export_sparse_metrics(self.project_path, metrics_json, self.metashape_cfg)
            snapshot = load_sparse_metrics(metrics_json, iteration=self.current_iteration)
            self.logger.log_sparse_metrics(self.current_iteration, snapshot)

            current_xyz = self._current_position_xyz()
            raw = generate_fibonacci_hemisphere(
                center_xyz=[self.rock_center_x, self.rock_center_y, self.rock_center_z],
                radius=self.candidate_radius_m,
                candidate_count=self.raw_candidate_count,
                min_elevation_deg=self.min_elevation_deg,
                max_elevation_deg=self.max_elevation_deg,
            )
            feasible = filter_candidates(raw, current_xyz, self.min_altitude_ned, self.max_altitude_ned, self.candidate_min_spacing_m, self.max_candidate_travel_m)
            final_pool = crop_to_target_count_diverse(feasible, self.target_candidate_count)
            self.logger.log_candidates(self.current_iteration, final_pool)

            context = ScoreContext(
                current_position_xyz=current_xyz,
                current_yaw_rad=self._current_heading(),
                rock_center_xyz=[self.rock_center_x, self.rock_center_y, self.rock_center_z],
                visited_viewpoints=self.visited_viewpoints,
                image_budget=self.image_budget,
                images_used=self.images_used,
                current_iteration=self.current_iteration,
            )
            scores = self.scorer.score_candidates(final_pool, snapshot, context)
            self.logger.log_scores(self.current_iteration, scores)
            best = scores[0]
            lookup = {c.candidate_id: c for c in final_pool}
            self.selected_candidate = lookup[best.candidate_id]
            self.logger.log_selected(self.current_iteration, {'candidate': self.selected_candidate.to_dict(), 'score': best.to_dict()})

            self.current_target = [self.selected_candidate.x, self.selected_candidate.y, self.selected_candidate.z]
            self.current_target_yaw = self.selected_candidate.yaw
            self.phase = Phase2State.FLY_TO_VIEW
            return

        if self.phase == Phase2State.FLY_TO_VIEW:
            if self._at_target(self.current_target):
                self.settle_started_at = time.time()
                self.phase = Phase2State.SETTLE
            return

        if self.phase == Phase2State.SETTLE:
            if self.settle_started_at is None:
                self.settle_started_at = time.time()
                return
            if time.time() - self.settle_started_at >= self.settle_time_sec:
                self.phase = Phase2State.CAPTURE
            return

        if self.phase == Phase2State.CAPTURE:
            path = self._save_capture()
            if path is not None:
                self.phase = Phase2State.UPDATE_PROJECT
            return

        if self.phase == Phase2State.UPDATE_PROJECT:
            image_path = self.paths.images_dir / self.last_capture_image_name
            metrics_json = self.paths.sparse_metrics_dir / f'metrics_iter_{self.current_iteration + 1:02d}.json'
            self.worker.incremental_update(self.project_path, [image_path], metrics_json, self.metashape_cfg)
            self.visited_viewpoints.append({
                'position_ned_m': {'x': self.current_target[0], 'y': self.current_target[1], 'z': self.current_target[2]},
                'yaw_rad': self.current_target_yaw,
                'source': 'adaptive',
                'candidate_id': self.selected_candidate.candidate_id if self.selected_candidate else None,
            })
            self.images_used += 1
            self.current_iteration += 1
            self.phase = Phase2State.SCORE_NEXT_VIEW
            return

        if self.phase == Phase2State.RETURN_HOME:
            self.current_target = [self.home_pose['x'], self.home_pose['y'], -self.takeoff_altitude]
            self.current_target_yaw = 0.0
            if self._at_target(self.current_target):
                self.phase = Phase2State.LAND if self.land_after_budget else Phase2State.OFFLINE_DENSE_RECON
            return

        if self.phase == Phase2State.LAND:
            if not self.land_sent:
                self._publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                self.land_sent = True
            self.phase = Phase2State.OFFLINE_DENSE_RECON
            return

        if self.phase == Phase2State.OFFLINE_DENSE_RECON:
            self.worker.offline_dense_reconstruct(self.project_path, self.paths.final_dir, self.metashape_cfg)
            self.phase = Phase2State.FINISHED
            return

    def _save_capture(self) -> Optional[Path]:
        if self.latest_image is None or self.selected_candidate is None:
            return None
        image_np = self._image_msg_to_cv(self.latest_image)
        stamp = f"{int(self.latest_image.header.stamp.sec):010d}_{int(self.latest_image.header.stamp.nanosec):09d}"
        ext = 'png' if self.image_format == 'png' else 'jpg'
        filename = f"{self.current_iteration:02d}_{self.selected_candidate.candidate_id}_{stamp}.{ext}"
        image_path = self.paths.images_dir / filename
        if not cv2.imwrite(str(image_path), image_np):
            return None

        metadata = {
            'iteration': self.current_iteration,
            'candidate_id': self.selected_candidate.candidate_id,
            'image_file': filename,
            'selected_candidate': self.selected_candidate.to_dict(),
            'vehicle_position_ned_m': {'x': float(self.local_position.x), 'y': float(self.local_position.y), 'z': float(self.local_position.z)} if self.local_position else None,
            'vehicle_heading_rad': self._current_heading(),
            'gimbal_pitch_rad': self.gimbal_pitch_rad,
            'gimbal_yaw_rad': self.gimbal_yaw_rad,
        }
        if self.latest_camera_info is not None:
            metadata['camera_info'] = {
                'width': self.latest_camera_info.width,
                'height': self.latest_camera_info.height,
                'distortion_model': self.latest_camera_info.distortion_model,
                'd': list(self.latest_camera_info.d),
                'k': list(self.latest_camera_info.k),
                'r': list(self.latest_camera_info.r),
                'p': list(self.latest_camera_info.p),
            }
        with open(self.paths.metadata_dir / f"{self.current_iteration:02d}_{self.selected_candidate.candidate_id}_{stamp}.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        self.last_capture_image_name = filename
        return image_path

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
        pitch_msg = Float64(); pitch_msg.data = float(self.gimbal_pitch_rad)
        yaw_msg = Float64(); yaw_msg.data = float(self.gimbal_yaw_rad)
        self.gimbal_pitch_pub.publish(pitch_msg)
        self.gimbal_yaw_pub.publish(yaw_msg)

    def _at_target(self, target: List[float]) -> bool:
        if self.local_position is None:
            return False
        dx = float(self.local_position.x) - float(target[0])
        dy = float(self.local_position.y) - float(target[1])
        dz = float(self.local_position.z) - float(target[2])
        return abs(dx) <= self.position_tolerance_xy and abs(dy) <= self.position_tolerance_xy and abs(dz) <= self.position_tolerance_z

    def _current_position_xyz(self) -> List[float]:
        if self.local_position is None:
            return [self.home_pose['x'], self.home_pose['y'], -self.takeoff_altitude]
        return [float(self.local_position.x), float(self.local_position.y), float(self.local_position.z)]

    def _current_heading(self) -> float:
        return float(getattr(self.local_position, 'heading', 0.0)) if self.local_position is not None else 0.0

    def _image_msg_to_cv(self, msg: Image) -> np.ndarray:
        enc = msg.encoding.lower()
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        if enc == 'mono8':
            return arr.reshape((msg.height, msg.width))
        if enc in {'rgb8', 'bgr8'}:
            img = arr.reshape((msg.height, msg.width, 3))
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if enc == 'rgb8' else img
        img = arr.reshape((msg.height, msg.width, 4))
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR) if enc == 'rgba8' else cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def _timestamp_us(self) -> int:
        return int(self.get_clock().now().nanoseconds / 1000)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Phase2ControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
