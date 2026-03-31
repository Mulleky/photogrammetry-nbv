#!/usr/bin/env python3
"""
Unified photogrammetry mission controller.

Runs Phase 1 (seed ring orbit) and Phase 2 (co-visibility NBV loop) in a
single node within one sim session.  No restarts, no manual symlinks.

State machine:
  WAIT_FOR_TOPICS -> WARMUP_STREAM -> TAKEOFF ->
  [MOVE_TO_SEED -> SETTLE_SEED -> CAPTURE_SEED]* -> BOOTSTRAP_PROJECT ->
  [SCORE_NEXT_VIEW -> FLY_TO_VIEW -> SETTLE -> CAPTURE -> UPDATE_PROJECT]* ->
  RETURN_HOME -> LAND -> OFFLINE_DENSE_RECON -> SEED_SPARSE_RECON -> FINISHED
"""
from __future__ import annotations

import json
import math
import os
import time
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
import yaml
from px4_msgs.msg import (
    OffboardControlMode, TrajectorySetpoint, VehicleCommand,
    VehicleLocalPosition, VehicleStatus,
)
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float64

from .candidate_filter import crop_to_target_count_diverse, filter_candidates
from .candidate_generator import generate_fibonacci_hemisphere
from .colmap_worker_client import ColmapWorkerClient
from .contracts import CandidateViewpoint, ScoreContext
from .metrics_extractor import load_sparse_metrics
from .mission_logger import MissionLogger
from .run_context import prepare_phase2_run
from .scorers import SCORER_REGISTRY


class State(Enum):
    # Startup
    WAIT_FOR_TOPICS = auto()
    WARMUP_STREAM = auto()
    TAKEOFF = auto()
    # Phase 1 — seed ring orbit
    MOVE_TO_SEED = auto()
    SETTLE_SEED = auto()
    CAPTURE_SEED = auto()
    # Transition
    BOOTSTRAP_PROJECT = auto()
    # Phase 2 — NBV loop
    SCORE_NEXT_VIEW = auto()
    FLY_TO_VIEW = auto()
    SETTLE = auto()
    CAPTURE = auto()
    UPDATE_PROJECT = auto()
    # Wrap-up
    RETURN_HOME = auto()
    LAND = auto()
    OFFLINE_DENSE_RECON = auto()
    SEED_SPARSE_RECON = auto()
    FINISHED = auto()


class UnifiedControllerNode(Node):
    def __init__(self) -> None:
        super().__init__('unified_controller_node')
        self._declare_parameters()
        self._load_parameters()

        # ---- QoS profiles ----
        self.qos_px4 = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST, depth=1,
        )
        self.qos_default = QoSProfile(depth=10)

        # ---- Publishers ----
        self.offboard_pub = self.create_publisher(OffboardControlMode, self.offboard_control_mode_topic, self.qos_px4)
        self.trajectory_pub = self.create_publisher(TrajectorySetpoint, self.trajectory_setpoint_topic, self.qos_px4)
        self.vehicle_command_pub = self.create_publisher(VehicleCommand, self.vehicle_command_topic, self.qos_px4)
        self.gimbal_pitch_pub = self.create_publisher(Float64, self.gimbal_pitch_topic, self.qos_default)
        self.gimbal_yaw_pub = self.create_publisher(Float64, self.gimbal_yaw_topic, self.qos_default)

        # ---- Subscribers ----
        self.create_subscription(VehicleLocalPosition, self.local_position_topic, self._local_position_cb, self.qos_px4)
        self.create_subscription(VehicleStatus, self.vehicle_status_topic, self._vehicle_status_cb, self.qos_px4)
        self.create_subscription(Image, self.image_topic, self._image_cb, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, self.camera_info_topic, self._camera_info_cb, qos_profile_sensor_data)

        # ---- State ----
        self.state = State.WAIT_FOR_TOPICS
        self.local_position: Optional[VehicleLocalPosition] = None
        self.vehicle_status: Optional[VehicleStatus] = None
        self.latest_image: Optional[Image] = None
        self.latest_camera_info: Optional[CameraInfo] = None

        self.home_xy: Optional[Tuple[float, float]] = None
        self.current_target = [0.0, 0.0, -self.takeoff_altitude]
        self.current_target_yaw = 0.0
        self.current_gimbal_pitch = self.gimbal_pitch_rad
        self.settle_started_at: Optional[float] = None
        self.offboard_warmup_counter = 0
        self.mode_sent = False
        self.arm_sent = False
        self.land_sent = False

        # Phase 1 state
        self.seed_viewpoints: List[dict] = []
        self.seed_view_index = 0
        self.seed_capture_count = 0
        self.seed_images_captured: List[Path] = []
        self.seed_metadata_list: List[dict] = []

        # Phase 2 state
        self.selected_candidate: Optional[CandidateViewpoint] = None
        self.current_iteration = 0
        self.project_bootstrapped = False
        self.images_used = 0
        self.visited_viewpoints: List[Dict] = []
        self.last_capture_image_name: Optional[str] = None

        # ---- Output directories ----
        self._prepare_output_dirs()

        # ---- Timer ----
        self.timer = self.create_timer(1.0 / self.offboard_rate_hz, self._run)
        self.get_logger().info(f'Unified controller started. Output: {self.run_dir}')

    # ==================================================================
    # Parameters
    # ==================================================================

    def _declare_parameters(self) -> None:
        params = [
            # Shared
            ('offboard_rate_hz', 10.0), ('setpoint_warmup_cycles', 15),
            ('takeoff_altitude', 6.0),
            ('position_tolerance_xy', 0.35), ('position_tolerance_z', 0.35),
            ('settle_time_sec', 2.0),
            ('gimbal_pitch_rad', -0.35), ('gimbal_yaw_rad', 0.0),
            ('image_format', 'jpg'),
            ('output_root', '~/photogrammetry_NBV/data/photogrammetry'),
            # Phase 1 — seed hemisphere
            ('seed_radius_m', 2.0), ('ring_image_count', 20),
            # Phase 2 — NBV
            ('rock_center_x', 0.0), ('rock_center_y', 0.0), ('rock_center_z', 0.0),
            ('image_budget', 20), ('max_image_budget', 30),
            ('stopping_criterion', 'knn_distance'),
            ('knn_distance_threshold', 0.05),
            ('raw_candidate_count', 200), ('target_candidate_count', 95),
            ('candidate_radius_m', 5.5), ('candidate_min_spacing_m', 1.0),
            ('min_elevation_deg', 10.0), ('max_elevation_deg', 45.0),
            ('min_altitude_ned', -20.0), ('max_altitude_ned', -0.1),
            ('max_candidate_travel_m', 30.0),
            ('land_after_budget', True),
            ('eval_bbox_half_extent', 6.0),
            # Paths
            ('colmap_bin', 'colmap'),
            ('colmap_script_dir', '~/photogrammetry-covisibility/src/photogrammetry_nbv/colmap_scripts'),
            ('colmap_config_path', '~/photogrammetry-covisibility/src/photogrammetry_nbv/config/colmap.yaml'),
            ('scoring_config_path', '~/photogrammetry-covisibility/src/photogrammetry_nbv/config/scoring.yaml'),
            # Topics
            ('offboard_control_mode_topic', '/fmu/in/offboard_control_mode'),
            ('trajectory_setpoint_topic', '/fmu/in/trajectory_setpoint'),
            ('vehicle_command_topic', '/fmu/in/vehicle_command'),
            ('local_position_topic', '/fmu/out/vehicle_local_position_v1'),
            ('vehicle_status_topic', '/fmu/out/vehicle_status_v3'),
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
        self.position_tolerance_xy = float(gp('position_tolerance_xy').value)
        self.position_tolerance_z = float(gp('position_tolerance_z').value)
        self.settle_time_sec = float(gp('settle_time_sec').value)
        self.gimbal_pitch_rad = float(gp('gimbal_pitch_rad').value)
        self.gimbal_yaw_rad = float(gp('gimbal_yaw_rad').value)
        self.image_format = str(gp('image_format').value).lower()
        self.output_root = os.path.expanduser(str(gp('output_root').value))

        # Phase 1
        self.seed_radius_m = float(gp('seed_radius_m').value)
        self.ring_image_count = int(gp('ring_image_count').value)

        # Phase 2
        self.rock_center_x = float(gp('rock_center_x').value)
        self.rock_center_y = float(gp('rock_center_y').value)
        self.rock_center_z = float(gp('rock_center_z').value)
        self.image_budget = int(gp('image_budget').value)
        self.max_image_budget = int(gp('max_image_budget').value)
        self.stopping_criterion = str(gp('stopping_criterion').value)
        self.knn_distance_threshold = float(gp('knn_distance_threshold').value)
        self.raw_candidate_count = int(gp('raw_candidate_count').value)
        self.target_candidate_count = int(gp('target_candidate_count').value)
        self.candidate_radius_m = float(gp('candidate_radius_m').value)
        self.candidate_min_spacing_m = float(gp('candidate_min_spacing_m').value)
        self.min_elevation_deg = float(gp('min_elevation_deg').value)
        self.max_elevation_deg = float(gp('max_elevation_deg').value)
        self.min_altitude_ned = float(gp('min_altitude_ned').value)
        self.max_altitude_ned = float(gp('max_altitude_ned').value)
        self.max_candidate_travel_m = float(gp('max_candidate_travel_m').value)
        self.land_after_budget = bool(gp('land_after_budget').value)
        self.eval_bbox_half_extent = float(gp('eval_bbox_half_extent').value)

        self.colmap_bin = str(gp('colmap_bin').value)
        self.colmap_script_dir = str(gp('colmap_script_dir').value)
        self.colmap_config_path = os.path.expanduser(str(gp('colmap_config_path').value))
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

        with open(self.colmap_config_path, 'r') as f:
            self.colmap_cfg = yaml.safe_load(f)
        with open(self.scoring_config_path, 'r') as f:
            self.scoring_cfg = yaml.safe_load(f)

    # ==================================================================
    # Output directory setup
    # ==================================================================

    def _prepare_output_dirs(self) -> None:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.run_dir = Path(self.output_root) / f'unified_run_{timestamp}'

        # Phase 1 (seed) output
        self.seed_dir = self.run_dir / 'seed'
        self.seed_images_dir = self.seed_dir / 'images'
        self.seed_metadata_dir = self.seed_dir / 'metadata'
        self.seed_calibration_dir = self.seed_dir / 'calibration'

        # Phase 2 (adaptive) output
        self.adaptive_dir = self.run_dir / 'adaptive'
        self.adaptive_images_dir = self.adaptive_dir / 'images'
        self.adaptive_metadata_dir = self.adaptive_dir / 'metadata'
        self.colmap_dir = self.run_dir / 'colmap'
        self.seed_colmap_dir = self.run_dir / 'seed_colmap'
        self.sparse_metrics_dir = self.run_dir / 'sparse_metrics'
        self.candidate_dir = self.run_dir / 'candidates'
        self.final_dir = self.run_dir / 'final'

        for d in [
            self.seed_images_dir, self.seed_metadata_dir, self.seed_calibration_dir,
            self.adaptive_images_dir, self.adaptive_metadata_dir,
            self.colmap_dir, self.seed_colmap_dir, self.sparse_metrics_dir,
            self.candidate_dir, self.final_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # Callbacks
    # ==================================================================

    def _local_position_cb(self, msg: VehicleLocalPosition) -> None:
        self.local_position = msg
        if self.home_xy is None and math.isfinite(msg.x) and math.isfinite(msg.y):
            self.home_xy = (float(msg.x), float(msg.y))
            self.current_target = [float(msg.x), float(msg.y), -self.takeoff_altitude]
            self.current_target_yaw = math.atan2(
                self.rock_center_y - float(msg.y), self.rock_center_x - float(msg.x))
            self.seed_viewpoints = self._build_seed_viewpoints()
            self.get_logger().info(f'Home XY locked at ({msg.x:.2f}, {msg.y:.2f})')

    def _vehicle_status_cb(self, msg: VehicleStatus) -> None:
        self.vehicle_status = msg

    def _image_cb(self, msg: Image) -> None:
        first = self.latest_image is None
        self.latest_image = msg
        if first:
            self.get_logger().info('First RGB image received.')

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        first = self.latest_camera_info is None
        self.latest_camera_info = msg
        if first:
            self.get_logger().info('First CameraInfo received.')
            payload = {
                'width': msg.width, 'height': msg.height,
                'distortion_model': msg.distortion_model,
                'd': list(msg.d), 'k': list(msg.k), 'r': list(msg.r), 'p': list(msg.p),
            }
            with open(self.seed_calibration_dir / 'camera_info_latest.json', 'w') as f:
                json.dump(payload, f, indent=2)

    # ==================================================================
    # Phase 1 helpers — seed hemisphere
    # ==================================================================

    def _build_seed_viewpoints(self) -> List[dict]:
        raw = generate_fibonacci_hemisphere(
            center_xyz=[self.rock_center_x, self.rock_center_y, self.rock_center_z],
            radius=self.seed_radius_m,
            candidate_count=self.ring_image_count,
            min_elevation_deg=self.min_elevation_deg,
            max_elevation_deg=self.max_elevation_deg,
        )
        vps = []
        for i, cand in enumerate(raw):
            dx = self.rock_center_x - cand.x
            dy = self.rock_center_y - cand.y
            dz = self.rock_center_z - cand.z
            yaw = math.atan2(dy, dx)
            horiz_dist = math.sqrt(dx * dx + dy * dy)
            gimbal_pitch = -math.atan2(dz, horiz_dist) if horiz_dist > 0.01 else -math.pi / 2
            vps.append({
                'label': f'ring_{i:02d}',
                'x': cand.x, 'y': cand.y, 'z': cand.z,
                'yaw': yaw, 'gimbal_pitch': gimbal_pitch,
            })
        return vps

    def _save_seed_capture(self, vp: dict) -> bool:
        if self.latest_image is None:
            return False
        image_np = self._image_msg_to_cv(self.latest_image)
        stamp = self._stamp_str(self.latest_image)
        ext = 'png' if self.image_format == 'png' else 'jpg'
        filename = f"{self.seed_capture_count:02d}_{vp['label']}_{stamp}.{ext}"
        image_path = self.seed_images_dir / filename

        if not cv2.imwrite(str(image_path), image_np):
            return False

        meta = {
            'capture_index': self.seed_capture_count,
            'view_label': vp['label'],
            'image_file': filename,
            'timestamp_ros_sec': int(self.latest_image.header.stamp.sec),
            'timestamp_ros_nanosec': int(self.latest_image.header.stamp.nanosec),
            'target_position_ned_m': {'x': vp['x'], 'y': vp['y'], 'z': vp['z']},
            'target_yaw_rad': vp['yaw'],
            'object_position_ned_m': {'x': self.rock_center_x, 'y': self.rock_center_y},
            'vehicle_position_ned_m': self._vehicle_pos_dict(),
            'vehicle_heading_rad': self._current_heading(),
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

        meta_path = self.seed_metadata_dir / f"{self.seed_capture_count:02d}_{vp['label']}_{stamp}.json"
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        self.seed_images_captured.append(image_path)
        self.seed_metadata_list.append(meta)
        self.get_logger().info(f"[Phase1] Captured {vp['label']} -> {filename}")
        return True

    # ==================================================================
    # Phase 1 -> Phase 2 transition
    # ==================================================================

    def _init_phase2(self) -> None:
        """Prepare Phase 2 after seed orbit completes."""
        # Write seed manifest (matches format expected by Phase 2 code)
        manifest = {
            'mission_type': 'photogrammetry_ring_init',
            'frame': 'PX4_NED',
            'object_x': self.rock_center_x,
            'object_y': self.rock_center_y,
            'takeoff_altitude_m': self.takeoff_altitude,
            'capture_radius_m': self.seed_radius_m,
            'ring_image_count': self.ring_image_count,
        }
        with open(self.seed_dir / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)

        # Populate visited viewpoints from seed captures so the covisibility scorer
        # has NED<->COLMAP correspondences (via image filename) at iteration 0.
        # Use target_position_ned_m (always defined) rather than vehicle_position_ned_m
        # (which can be None if local_position dropped momentarily during capture).
        # images_used intentionally stays at 0 — it counts only NBV captures so that
        # image_budget and max_image_budget apply to Phase 2 only.
        for meta in self.seed_metadata_list:
            self.visited_viewpoints.append({
                'position_ned_m': meta['target_position_ned_m'],
                'yaw_rad': meta.get('target_yaw_rad', 0.0),
                'image_file': meta.get('image_file', ''),
                'source': 'seed',
            })

        # Set up COLMAP worker and scorer
        self.worker = ColmapWorkerClient(self.colmap_bin, Path(self.colmap_script_dir).expanduser())
        self.logger = MissionLogger(self.candidate_dir, self.sparse_metrics_dir)

        scorer_name = str(self.scoring_cfg['scorer']['name'])
        scorer_cls = SCORER_REGISTRY[scorer_name]
        self.scorer = scorer_cls(self.scoring_cfg['scorer'])

        # Write phase 2 manifest
        p2_manifest = {
            'mission_type': 'unified_photogrammetry_covisibility',
            'seed_count': len(self.seed_images_captured),
            'image_budget': self.image_budget,
            'max_image_budget': self.max_image_budget,
            'stopping_criterion': self.stopping_criterion,
            'knn_distance_threshold': self.knn_distance_threshold,
            'scorer_name': scorer_name,
            'rock_center_ned_m': {
                'x': self.rock_center_x, 'y': self.rock_center_y, 'z': self.rock_center_z,
            },
        }
        with open(self.run_dir / 'phase2_manifest.json', 'w') as f:
            json.dump(p2_manifest, f, indent=2)

        self.get_logger().info(
            f'[Phase2] Initialized. Seed captures: {len(self.seed_images_captured)}, '
            f'visited entries: {len(self.visited_viewpoints)}. '
            f'Scorer: {scorer_name}. Stopping: {self.stopping_criterion}')

    # ==================================================================
    # Phase 2 helpers
    # ==================================================================

    def _should_stop(self, snapshot) -> bool:
        if self.images_used >= self.max_image_budget:
            return True
        criterion = self.stopping_criterion
        if criterion == 'image_budget':
            return self.images_used >= self.image_budget
        if criterion == 'knn_distance':
            knn = snapshot.knn_distance_metrics
            if not knn:
                return False
            return knn.get('percentile_knn_distance', float('inf')) <= self.knn_distance_threshold
        if criterion == 'both':
            budget_done = self.images_used >= self.image_budget
            knn = snapshot.knn_distance_metrics
            knn_done = bool(knn) and knn.get('percentile_knn_distance', float('inf')) <= self.knn_distance_threshold
            return budget_done or knn_done
        return self.images_used >= self.image_budget

    def _save_adaptive_capture(self) -> Optional[Path]:
        if self.latest_image is None or self.selected_candidate is None:
            return None
        image_np = self._image_msg_to_cv(self.latest_image)
        stamp = self._stamp_str(self.latest_image)
        ext = 'png' if self.image_format == 'png' else 'jpg'
        filename = f"{self.current_iteration:02d}_{self.selected_candidate.candidate_id}_{stamp}.{ext}"
        image_path = self.adaptive_images_dir / filename

        if not cv2.imwrite(str(image_path), image_np):
            return None

        meta = {
            'iteration': self.current_iteration,
            'candidate_id': self.selected_candidate.candidate_id,
            'image_file': filename,
            'selected_candidate': self.selected_candidate.to_dict(),
            'vehicle_position_ned_m': self._vehicle_pos_dict(),
            'vehicle_heading_rad': self._current_heading(),
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
        with open(self.adaptive_metadata_dir / f"{self.current_iteration:02d}_{self.selected_candidate.candidate_id}_{stamp}.json", 'w') as f:
            json.dump(meta, f, indent=2)

        self.last_capture_image_name = filename
        self.get_logger().info(f'[Phase2] Captured iter {self.current_iteration} -> {filename}')
        return image_path

    # ==================================================================
    # Main state machine
    # ==================================================================

    def _run(self) -> None:
        self._publish_offboard_control_mode()
        self._publish_setpoint(self.current_target, self.current_target_yaw)
        self._publish_gimbal_commands()

        # --- Startup ---
        if self.state == State.WAIT_FOR_TOPICS:
            if self.local_position is None:
                if self.offboard_warmup_counter % 50 == 0:
                    self.get_logger().info('[Wait] Waiting for local position...')
                self.offboard_warmup_counter += 1
                return
            if not self.seed_viewpoints:
                self.get_logger().warn('[Wait] local_position received but seed_viewpoints is empty — check rock_center and elevation params')
                return
            self.offboard_warmup_counter = 0
            self.state = State.WARMUP_STREAM
            self.get_logger().info('Position ready. Warming up offboard stream...')
            return

        if self.state == State.WARMUP_STREAM:
            self.offboard_warmup_counter += 1
            if self.offboard_warmup_counter >= self.setpoint_warmup_cycles:
                if not self.mode_sent:
                    self._publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                    self.mode_sent = True
                if not self.arm_sent:
                    self._publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                    self.arm_sent = True
                self.state = State.TAKEOFF
                self.get_logger().info('Armed. Taking off...')
            return

        if self.state == State.TAKEOFF:
            if self._at_target(self.current_target):
                self.state = State.MOVE_TO_SEED
                self.get_logger().info(f'Takeoff complete. Starting Phase 1 ring orbit ({self.ring_image_count} views).')
            return

        # --- Phase 1: Seed ring orbit ---
        if self.state == State.MOVE_TO_SEED:
            if self.seed_view_index >= len(self.seed_viewpoints):
                # All seed viewpoints captured — transition to Phase 2
                self.get_logger().info(f'[Phase1] Complete. {self.seed_capture_count} images captured. Transitioning to Phase 2...')
                self._init_phase2()
                self.state = State.BOOTSTRAP_PROJECT
                return
            vp = self.seed_viewpoints[self.seed_view_index]
            self.current_target = [vp['x'], vp['y'], vp['z']]
            self.current_target_yaw = vp['yaw']
            self.current_gimbal_pitch = vp['gimbal_pitch']
            if self._at_target(self.current_target):
                self.settle_started_at = time.time()
                self.state = State.SETTLE_SEED
            return

        if self.state == State.SETTLE_SEED:
            if self.settle_started_at and time.time() - self.settle_started_at >= self.settle_time_sec:
                self.state = State.CAPTURE_SEED
            return

        if self.state == State.CAPTURE_SEED:
            if self.latest_image is None:
                return
            vp = self.seed_viewpoints[self.seed_view_index]
            if self._save_seed_capture(vp):
                self.seed_capture_count += 1
                self.seed_view_index += 1
                self.state = State.MOVE_TO_SEED
            return

        # --- Transition: Bootstrap COLMAP with seed images ---
        if self.state == State.BOOTSTRAP_PROJECT:
            if not self.project_bootstrapped:
                metrics_json = self.sparse_metrics_dir / 'metrics_iter_00.json'
                self.worker.bootstrap_project(
                    self.colmap_dir, self.seed_images_captured, metrics_json, self.colmap_cfg)
                self.project_bootstrapped = True
                snapshot = load_sparse_metrics(metrics_json, iteration=0)
                pt_count = snapshot.sparse_point_count
                if pt_count < 50:
                    self.get_logger().error(
                        f'[Phase2] WARNING: COLMAP produced only {pt_count} sparse points from seed images '
                        f'(threshold: 50). Rock may lack texture or seed poses may be degenerate. '
                        f'NBV scoring may be unreliable.')
                else:
                    self.get_logger().info(f'[Phase2] COLMAP bootstrapped with seed images. Sparse points: {pt_count}')
            self.state = State.SCORE_NEXT_VIEW
            return

        # --- Phase 2: NBV loop ---
        if self.state == State.SCORE_NEXT_VIEW:
            # Export metrics first so stopping criterion can use kNN
            metrics_json = self.sparse_metrics_dir / f'metrics_iter_{self.current_iteration:02d}.json'
            self.worker.export_sparse_metrics(self.colmap_dir, metrics_json, self.colmap_cfg)
            snapshot = load_sparse_metrics(metrics_json, iteration=self.current_iteration)
            self.logger.log_sparse_metrics(self.current_iteration, snapshot)

            if self._should_stop(snapshot):
                self.get_logger().info(
                    f'[Phase2] Stopping: {self.stopping_criterion} '
                    f'(images={self.images_used}, knn={snapshot.knn_distance_metrics})')
                self.state = State.RETURN_HOME
                return

            current_xyz = self._current_position_xyz()
            raw = generate_fibonacci_hemisphere(
                center_xyz=[self.rock_center_x, self.rock_center_y, self.rock_center_z],
                radius=self.candidate_radius_m,
                candidate_count=self.raw_candidate_count,
                min_elevation_deg=self.min_elevation_deg,
                max_elevation_deg=self.max_elevation_deg,
            )
            feasible = filter_candidates(raw, current_xyz, self.min_altitude_ned, self.max_altitude_ned,
                                         self.candidate_min_spacing_m, self.max_candidate_travel_m)
            # Hard-exclude candidates too close to previously visited NBV viewpoints.
            # Seed viewpoints (source='seed') are excluded from this check — they orbit
            # at a different radius and would otherwise block all NBV candidates.
            nbv_xyzs = [
                (vp['position_ned_m']['x'], vp['position_ned_m']['y'], vp['position_ned_m']['z'])
                for vp in self.visited_viewpoints
                if 'position_ned_m' in vp and vp.get('source') == 'adaptive'
            ]
            if nbv_xyzs:
                min_revisit_dist = self.candidate_min_spacing_m
                feasible = [
                    c for c in feasible
                    if all(
                        math.sqrt((c.x - vx)**2 + (c.y - vy)**2 + (c.z - vz)**2) >= min_revisit_dist
                        for vx, vy, vz in nbv_xyzs
                    )
                ]
            final_pool = crop_to_target_count_diverse(feasible, self.target_candidate_count)
            self.logger.log_candidates(self.current_iteration, final_pool)

            if not final_pool:
                self.get_logger().error(
                    '[Phase2] No feasible candidates after filtering. Returning home.')
                self.state = State.RETURN_HOME
                return

            context = ScoreContext(
                current_position_xyz=current_xyz,
                current_yaw_rad=self._current_heading(),
                rock_center_xyz=[self.rock_center_x, self.rock_center_y, self.rock_center_z],
                visited_viewpoints=self.visited_viewpoints,
                image_budget=self.image_budget,
                images_used=self.images_used,
                current_iteration=self.current_iteration,
                colmap_workspace=str(self.colmap_dir),
            )
            scores = self.scorer.score_candidates(final_pool, snapshot, context)
            self.logger.log_scores(self.current_iteration, scores)

            if not scores:
                self.get_logger().error(
                    '[Phase2] Scorer returned empty scores. Returning home.')
                self.state = State.RETURN_HOME
                return

            best = scores[0]
            lookup = {c.candidate_id: c for c in final_pool}
            self.selected_candidate = lookup[best.candidate_id]
            self.logger.log_selected(self.current_iteration, {
                'candidate': self.selected_candidate.to_dict(), 'score': best.to_dict()})

            self.current_target = [self.selected_candidate.x, self.selected_candidate.y, self.selected_candidate.z]
            self.current_target_yaw = self.selected_candidate.yaw

            # Compute gimbal pitch to look at rock center from selected candidate
            dx = self.rock_center_x - self.selected_candidate.x
            dy = self.rock_center_y - self.selected_candidate.y
            dz = self.rock_center_z - self.selected_candidate.z
            horiz_dist = math.sqrt(dx * dx + dy * dy)
            self.current_gimbal_pitch = -math.atan2(dz, horiz_dist) if horiz_dist > 0.01 else -math.pi / 2

            self.get_logger().info(
                f'[Phase2] Iter {self.current_iteration}: '
                f'selected {best.candidate_id} (score={best.final_score:.3f})')
            self.state = State.FLY_TO_VIEW
            return

        if self.state == State.FLY_TO_VIEW:
            if self._at_target(self.current_target):
                self.settle_started_at = time.time()
                self.state = State.SETTLE
            return

        if self.state == State.SETTLE:
            if self.settle_started_at and time.time() - self.settle_started_at >= self.settle_time_sec:
                self.state = State.CAPTURE
            return

        if self.state == State.CAPTURE:
            path = self._save_adaptive_capture()
            if path is not None:
                self.state = State.UPDATE_PROJECT
            return

        if self.state == State.UPDATE_PROJECT:
            image_path = self.adaptive_images_dir / self.last_capture_image_name
            metrics_json = self.sparse_metrics_dir / f'metrics_iter_{self.current_iteration + 1:02d}.json'
            self.worker.incremental_update(self.colmap_dir, [image_path], metrics_json, self.colmap_cfg)
            self.visited_viewpoints.append({
                'position_ned_m': {
                    'x': self.current_target[0],
                    'y': self.current_target[1],
                    'z': self.current_target[2],
                },
                'yaw_rad': self.current_target_yaw,
                'image_file': self.last_capture_image_name,
                'source': 'adaptive',
                'candidate_id': self.selected_candidate.candidate_id if self.selected_candidate else None,
            })
            self.images_used += 1
            self.current_iteration += 1
            self.state = State.SCORE_NEXT_VIEW
            return

        # --- Wrap-up ---
        if self.state == State.RETURN_HOME:
            self.current_gimbal_pitch = self.gimbal_pitch_rad  # reset to config default
            assert self.home_xy is not None
            self.current_target = [self.home_xy[0], self.home_xy[1], -self.takeoff_altitude]
            self.current_target_yaw = 0.0
            if self._at_target(self.current_target):
                self.state = State.LAND if self.land_after_budget else State.OFFLINE_DENSE_RECON
            return

        if self.state == State.LAND:
            if not self.land_sent:
                self._publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                self.land_sent = True
            self.state = State.OFFLINE_DENSE_RECON
            return

        if self.state == State.OFFLINE_DENSE_RECON:
            self.get_logger().info('[Wrap-up] Starting dense reconstruction...')
            try:
                self.worker.offline_dense_reconstruct(self.colmap_dir, self.final_dir, self.colmap_cfg)
                self.get_logger().info('[Wrap-up] Dense reconstruction done.')
            except RuntimeError as e:
                self.get_logger().error(f'[Wrap-up] Dense reconstruction failed:\n{e}')
            self.state = State.SEED_SPARSE_RECON
            return

        if self.state == State.SEED_SPARSE_RECON:
            self.get_logger().info('[Wrap-up] Exporting seed-only sparse cloud...')
            try:
                self.worker.seed_sparse_reconstruct(
                    self.seed_colmap_dir, self.seed_images_captured, self.final_dir, self.colmap_cfg)
                self.get_logger().info('[Wrap-up] Seed sparse cloud done. Mission complete.')
            except RuntimeError as e:
                self.get_logger().error(f'[Wrap-up] Seed sparse reconstruction failed:\n{e}')
            self.state = State.FINISHED
            return

    # ==================================================================
    # Shared helpers
    # ==================================================================

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
        p = Float64(); p.data = float(self.current_gimbal_pitch)
        y = Float64(); y.data = float(self.gimbal_yaw_rad)
        self.gimbal_pitch_pub.publish(p)
        self.gimbal_yaw_pub.publish(y)

    def _at_target(self, target: List[float]) -> bool:
        if self.local_position is None:
            return False
        dx = float(self.local_position.x) - float(target[0])
        dy = float(self.local_position.y) - float(target[1])
        dz = float(self.local_position.z) - float(target[2])
        return (abs(dx) <= self.position_tolerance_xy and
                abs(dy) <= self.position_tolerance_xy and
                abs(dz) <= self.position_tolerance_z)

    def _current_position_xyz(self) -> List[float]:
        if self.local_position is None:
            hx, hy = self.home_xy or (0.0, 0.0)
            return [hx, hy, -self.takeoff_altitude]
        return [float(self.local_position.x), float(self.local_position.y), float(self.local_position.z)]

    def _current_heading(self) -> float:
        return float(getattr(self.local_position, 'heading', 0.0)) if self.local_position else 0.0

    def _vehicle_pos_dict(self) -> Optional[Dict[str, float]]:
        if self.local_position is None:
            return None
        return {'x': float(self.local_position.x), 'y': float(self.local_position.y), 'z': float(self.local_position.z)}

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

    def _stamp_str(self, msg: Image) -> str:
        return f'{int(msg.header.stamp.sec):010d}_{int(msg.header.stamp.nanosec):09d}'

    def _timestamp_us(self) -> int:
        return int(self.get_clock().now().nanoseconds / 1000)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = UnifiedControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
