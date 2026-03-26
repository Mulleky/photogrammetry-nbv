from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class CandidateViewpoint:
    candidate_id: str
    x: float
    y: float
    z: float
    yaw: float
    radius: float
    azimuth_rad: float
    elevation_rad: float
    feasibility_flags: Dict[str, bool] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WeakRegion:
    region_id: str
    centroid_xyz: List[float]
    severity: float
    point_count: int
    components: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SparseMetricsSnapshot:
    iteration: int
    total_cameras: int
    aligned_cameras: int
    sparse_point_count: int
    global_metrics: Dict[str, float]
    weak_regions: List[WeakRegion] = field(default_factory=list)
    per_camera_stats: List[Dict[str, Any]] = field(default_factory=list)
    raw_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload['weak_regions'] = [wr.to_dict() for wr in self.weak_regions]
        return payload


@dataclass
class ScoreBreakdown:
    candidate_id: str
    final_score: float
    terms: Dict[str, float]
    weights: Dict[str, float]
    scorer_name: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScoreContext:
    current_position_xyz: List[float]
    current_yaw_rad: float
    rock_center_xyz: List[float]
    visited_viewpoints: List[Dict[str, Any]]
    image_budget: int
    images_used: int
    current_iteration: int


@dataclass
class RunPaths:
    run_dir: Path
    adaptive_dir: Path
    images_dir: Path
    metadata_dir: Path
    metashape_dir: Path
    sparse_metrics_dir: Path
    candidate_dir: Path
    final_dir: Path

    def to_dict(self) -> Dict[str, Any]:
        return {k: str(v) for k, v in asdict(self).items()}
