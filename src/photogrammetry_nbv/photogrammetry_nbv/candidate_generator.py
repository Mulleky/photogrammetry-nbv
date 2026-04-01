from __future__ import annotations

import math
from collections import Counter
from dataclasses import replace
from typing import List, Union

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
        z = cz - radius * z_unit if z_mode == 'above_object' else cz + radius * z_unit

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


def generate_stratified_orbit_candidates(
    center_xyz: List[float],
    radius: float,
    elevation_rings_deg: List[float],
    azimuth_count_per_ring: Union[int, List[int]],
    z_mode: str = 'above_object',
    candidate_prefix: str = 'cand',
    azimuth_sector_count: int = 8,
) -> List[CandidateViewpoint]:
    """
    Generate viewpoints on explicit elevation rings, uniformly spaced in azimuth.

    Unlike the Fibonacci generator, this places deterministic rings so lateral
    (low-elevation) coverage is explicit rather than an artefact of band sampling.
    Each candidate has planner metadata stored in .extra.
    """
    if not elevation_rings_deg:
        return []

    # Normalise azimuth_count_per_ring to a list
    if isinstance(azimuth_count_per_ring, int):
        counts = [azimuth_count_per_ring] * len(elevation_rings_deg)
    else:
        counts = list(azimuth_count_per_ring)
        if len(counts) != len(elevation_rings_deg):
            raise ValueError(
                'azimuth_count_per_ring length must match elevation_rings_deg length')

    cx, cy, cz = center_xyz
    all_candidates: List[CandidateViewpoint] = []

    for ring_idx, (el_deg, n) in enumerate(zip(elevation_rings_deg, counts)):
        if n <= 0:
            continue
        el_rad = math.radians(el_deg)
        planner_band = 'lateral' if el_deg <= 25.0 else 'upper'
        sector_width = 360.0 / azimuth_sector_count

        for k in range(n):
            az_deg = 360.0 * k / n
            az_rad = math.radians(az_deg)

            x = cx + radius * math.cos(el_rad) * math.cos(az_rad)
            y = cy + radius * math.cos(el_rad) * math.sin(az_rad)
            if z_mode == 'above_object':
                z = cz - radius * math.sin(el_rad)
            else:
                z = cz + radius * math.sin(el_rad)

            yaw = math.atan2(cy - y, cx - x)
            sector = int(az_deg / sector_width) % azimuth_sector_count

            all_candidates.append(
                CandidateViewpoint(
                    candidate_id=f'{candidate_prefix}_{len(all_candidates):03d}',
                    x=float(x),
                    y=float(y),
                    z=float(z),
                    yaw=float(yaw),
                    radius=float(radius),
                    azimuth_rad=float(az_rad),
                    elevation_rad=float(el_rad),
                    extra={
                        'planner_mode': 'stratified',
                        'planner_band': planner_band,
                        'ring_index': ring_idx,
                        'ring_elevation_deg': el_deg,
                        'azimuth_sector': sector,
                    },
                )
            )

    # Re-index with final prefix
    return [
        replace(c, candidate_id=f'{candidate_prefix}_{idx:03d}')
        for idx, c in enumerate(all_candidates)
    ]


def generate_hybrid_candidates(
    center_xyz: List[float],
    radius: float,
    lateral_elevations_deg: List[float],
    upper_elevations_deg: List[float],
    lateral_count: int,
    upper_count: int,
    z_mode: str = 'above_object',
) -> List[CandidateViewpoint]:
    """
    Build a two-band candidate pool: a dense lateral belt plus a sparse upper support band.

    lateral_count viewpoints are distributed across lateral_elevations_deg with more
    samples on lower rings (inverse-rank weighting). upper_count viewpoints are split
    evenly across upper_elevations_deg.
    """
    # --- Lateral band ---
    if lateral_elevations_deg and lateral_count > 0:
        n_lat = len(lateral_elevations_deg)
        # Inverse-rank weights: ring 0 gets most viewpoints
        raw_weights = [1.0 / (i + 1) for i in range(n_lat)]
        total_w = sum(raw_weights)
        lat_counts = [max(1, round(lateral_count * w / total_w)) for w in raw_weights]
        # Correct rounding drift against target
        diff = sum(lat_counts) - lateral_count
        if diff != 0:
            lat_counts[-1] = max(1, lat_counts[-1] - diff)
        lateral = generate_stratified_orbit_candidates(
            center_xyz, radius, lateral_elevations_deg, lat_counts,
            z_mode=z_mode, candidate_prefix='lat',
        )
        for c in lateral:
            c.extra['planner_band'] = 'lateral'
    else:
        lateral = []

    # --- Upper band ---
    if upper_elevations_deg and upper_count > 0:
        n_upp = len(upper_elevations_deg)
        per_ring = max(1, upper_count // n_upp)
        upp_counts = [per_ring] * n_upp
        # Distribute remainder to first rings
        remainder = upper_count - sum(upp_counts)
        for i in range(abs(remainder)):
            if remainder > 0:
                upp_counts[i % n_upp] += 1
            else:
                upp_counts[-(i % n_upp) - 1] = max(1, upp_counts[-(i % n_upp) - 1] - 1)
        upper = generate_stratified_orbit_candidates(
            center_xyz, radius, upper_elevations_deg, upp_counts,
            z_mode=z_mode, candidate_prefix='upp',
        )
        for c in upper:
            c.extra['planner_band'] = 'upper'
    else:
        upper = []

    combined = lateral + upper
    # Re-index all IDs globally
    return [replace(c, candidate_id=f'cand_{idx:03d}') for idx, c in enumerate(combined)]


def balance_candidates_by_coverage(
    candidates: List[CandidateViewpoint],
    visited_xyzs: List[tuple],
    rock_center_xyz: List[float],
    azimuth_sector_count: int,
    target_count: int,
) -> List[CandidateViewpoint]:
    """
    Prefer candidates from azimuth sectors that have fewest prior adaptive visits.

    Within each sector, candidates are ordered to preserve elevation diversity
    (sorted by elevation so alternating low/high are kept as we truncate).
    The scorer is not invoked here — this is a planning-only reordering.
    """
    if not candidates or target_count <= 0:
        return []

    cx, cy, _ = rock_center_xyz
    sector_width = 360.0 / azimuth_sector_count

    # Count how many visited viewpoints are in each sector
    visited_sector_count: Counter = Counter()
    for vx, vy, _vz in visited_xyzs:
        az = math.degrees(math.atan2(vy - cy, vx - cx)) % 360.0
        s = int(az / sector_width) % azimuth_sector_count
        visited_sector_count[s] += 1

    # Tag each candidate with its azimuth sector (may already be set)
    tagged: List[CandidateViewpoint] = []
    for c in candidates:
        az = math.degrees(math.atan2(c.y - cy, c.x - cx)) % 360.0
        sector = int(az / sector_width) % azimuth_sector_count
        updated_extra = dict(c.extra)
        updated_extra['azimuth_sector'] = sector
        tagged.append(replace(c, extra=updated_extra))

    # Sort: primary key = visited count in sector (ascending = prefer under-covered),
    # secondary key = elevation (descending = prefer higher/safer candidates within each sector)
    tagged.sort(key=lambda c: (
        visited_sector_count.get(c.extra['azimuth_sector'], 0),
        -c.elevation_rad,
    ))

    return tagged[:target_count]
