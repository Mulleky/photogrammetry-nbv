from __future__ import annotations

import math
from typing import List

from .contracts import CandidateViewpoint


def generate_fibonacci_hemisphere(
    center_xyz: List[float],
    radius: float,
    candidate_count: int,
    min_elevation_deg: float,
    max_elevation_deg: float,
    z_mode: str = 'above_object',
) -> List[CandidateViewpoint]:
    """
    Generate candidate viewpoints on an upper hemisphere around the object.

    center_xyz is interpreted in PX4 NED. Because z grows downward in NED, "above"
    means smaller z values than the object.
    """
    if candidate_count <= 0:
        return []

    cx, cy, cz = center_xyz
    min_el = math.radians(min_elevation_deg)
    max_el = math.radians(max_elevation_deg)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))

    candidates: List[CandidateViewpoint] = []
    accepted = 0
    i = 0
    while accepted < candidate_count and i < candidate_count * 8:
        t = (i + 0.5) / float(candidate_count * 2)
        z_unit = 1.0 - 2.0 * t
        xy = math.sqrt(max(0.0, 1.0 - z_unit * z_unit))
        azimuth = i * golden_angle
        x_u = math.cos(azimuth) * xy
        y_u = math.sin(azimuth) * xy
        elevation = math.asin(max(-1.0, min(1.0, z_unit)))

        if elevation < min_el or elevation > max_el:
            i += 1
            continue

        x = cx + radius * x_u
        y = cy + radius * y_u
        z = cz - radius * abs(z_unit) if z_mode == 'above_object' else cz + radius * z_unit

        dx = cx - x
        dy = cy - y
        yaw = math.atan2(dy, dx)

        candidates.append(
            CandidateViewpoint(
                candidate_id=f'cand_{accepted:03d}',
                x=float(x),
                y=float(y),
                z=float(z),
                yaw=float(yaw),
                radius=float(radius),
                azimuth_rad=float(math.atan2(y_u, x_u)),
                elevation_rad=float(elevation),
            )
        )
        accepted += 1
        i += 1

    return candidates
