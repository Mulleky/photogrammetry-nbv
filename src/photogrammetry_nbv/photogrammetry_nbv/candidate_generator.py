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

    Uses a large Fibonacci sphere to collect all points in the elevation band,
    then uniformly subsamples to candidate_count so the result spans the full
    [min_elevation_deg, max_elevation_deg] range rather than clustering at the top.
    """
    if candidate_count <= 0:
        return []

    cx, cy, cz = center_xyz
    min_el = math.radians(min_elevation_deg)
    max_el = math.radians(max_elevation_deg)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))

    # Oversample the Fibonacci sphere so the elevation band contains well more
    # than candidate_count points. Fraction of upper hemisphere in band:
    #   f = (sin(max_el) - sin(min_el)) / 2   (normalised over [-90°, 90°])
    band_fraction = max(0.01, (math.sin(max_el) - math.sin(min_el)) / 2.0)
    total_to_sample = max(candidate_count * 20, int(candidate_count * 4 / band_fraction))

    all_in_band: List[CandidateViewpoint] = []
    for i in range(total_to_sample):
        t = (i + 0.5) / float(total_to_sample)
        z_unit = 1.0 - 2.0 * t
        xy = math.sqrt(max(0.0, 1.0 - z_unit * z_unit))
        azimuth = i * golden_angle
        x_u = math.cos(azimuth) * xy
        y_u = math.sin(azimuth) * xy
        elevation = math.asin(max(-1.0, min(1.0, z_unit)))

        if elevation < min_el or elevation > max_el:
            continue

        x = cx + radius * x_u
        y = cy + radius * y_u
        z = cz - radius * abs(z_unit) if z_mode == 'above_object' else cz + radius * z_unit

        dx = cx - x
        dy = cy - y
        yaw = math.atan2(dy, dx)

        all_in_band.append(
            CandidateViewpoint(
                candidate_id=f'cand_{len(all_in_band):03d}',
                x=float(x),
                y=float(y),
                z=float(z),
                yaw=float(yaw),
                radius=float(radius),
                azimuth_rad=float(math.atan2(y_u, x_u)),
                elevation_rad=float(elevation),
            )
        )

    if len(all_in_band) == 0:
        return []

    # Uniformly subsample to candidate_count so the result spans the full band.
    if len(all_in_band) <= candidate_count:
        result = all_in_band
    else:
        stride = len(all_in_band) / candidate_count
        result = [all_in_band[int(k * stride)] for k in range(candidate_count)]

    # Re-index candidate IDs after subsampling.
    for idx, vp in enumerate(result):
        result[idx] = CandidateViewpoint(
            candidate_id=f'cand_{idx:03d}',
            x=vp.x, y=vp.y, z=vp.z, yaw=vp.yaw,
            radius=vp.radius,
            azimuth_rad=vp.azimuth_rad,
            elevation_rad=vp.elevation_rad,
        )

    return result
