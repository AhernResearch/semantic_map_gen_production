"""Scene-package floor bands, trajectory membership, and depth filtering."""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from adapters.proposals.floor_ids import _as_float, normalize_floor_id
from adapters.scripts.common_scene import read_json

# A current-floor query normally uses the live start pose.  Some MP3D
# packages have overlapping vertical floor bands, however, and the simulator
# start pose can then fall in the neighbouring band while being immediately
# adjacent to a recorded navigable pose on the real floor.  Keep the snap
# deliberately local so it corrects that representation issue without
# silently changing genuinely multi-storey starts.
DEFAULT_QUERY_START_NAV_SNAP_DISTANCE_M = 2.0

def _trajectory_nav_positions(scene_package_text: str) -> tuple[tuple[float, float, float], ...]:
    """Load valid navigable trajectory positions for local floor disambiguation.

    The package trajectory is produced during map collection, rather than
    from benchmark annotations, so it is valid runtime map evidence.  Cache
    it per immutable scene package: resolving every query must not repeatedly
    parse the complete dense trajectory.
    """

    trajectory_path = Path(scene_package_text) / "observations" / "trajectory.json"
    if not trajectory_path.exists():
        return ()
    try:
        loaded = read_json(trajectory_path)
    except Exception:
        return ()
    if not isinstance(loaded, list):
        return ()
    positions: list[tuple[float, float, float]] = []
    for row in loaded:
        point = row.get("nav_position") if isinstance(row, dict) else None
        if not isinstance(point, (list, tuple)) or len(point) < 3:
            continue
        x, y, z = (_as_float(point[axis]) for axis in range(3))
        if x is not None and y is not None and z is not None:
            positions.append((x, y, z))
    return tuple(positions)


def floor_id_for_start_position(
    start_position: Any,
    floor_context: dict[str, Any] | None,
    *,
    nav_snap_distance_m: float = DEFAULT_QUERY_START_NAV_SNAP_DISTANCE_M,
) -> str | None:
    """Resolve a current-floor start using nearby navigable map evidence.

    Height-band resolution remains the compatibility fallback.  A nearby
    recorded navigable point is more reliable only when the scene package is
    available and lies within a small 3-D radius; this avoids using a distant
    trajectory segment to override an otherwise valid start floor.
    """

    if not isinstance(start_position, (list, tuple)) or len(start_position) < 3:
        return None
    x, y, z = (_as_float(start_position[axis]) for axis in range(3))
    if x is None or y is None or z is None:
        return None
    height_floor = floor_id_for_y(y, floor_context)
    if not isinstance(floor_context, dict):
        return height_floor
    scene_package = floor_context.get("scene_package")
    if not isinstance(scene_package, str) or not scene_package:
        return height_floor
    positions = _trajectory_nav_positions(scene_package)
    if not positions:
        return height_floor
    nearest = min(
        positions,
        key=lambda point: (point[0] - x) ** 2 + (point[1] - y) ** 2 + (point[2] - z) ** 2,
    )
    distance = math.dist((x, y, z), nearest)
    if distance > max(0.0, float(nav_snap_distance_m)):
        return height_floor
    return floor_id_for_y(nearest[1], floor_context) or height_floor

def floor_context_from_scene_package(scene_package: Path | None) -> dict[str, Any]:
    if scene_package is None:
        return {"source": "missing_scene_package", "bands": []}
    levels_path = scene_package / "maps" / "floor_levels.json"
    if levels_path.exists():
        try:
            levels = read_json(levels_path)
        except Exception:
            levels = []
        bands = []
        if isinstance(levels, list):
            for index, level in enumerate(levels):
                if not isinstance(level, dict):
                    continue
                center_y = _as_float(level.get("center_y"))
                if center_y is None:
                    center_y = _as_float(level.get("floor_y"))
                min_y = _as_float(level.get("min_y"))
                max_y = _as_float(level.get("max_y"))
                if center_y is None and min_y is not None and max_y is not None:
                    center_y = 0.5 * (min_y + max_y)
                if center_y is None:
                    continue
                if min_y is None:
                    min_y = center_y - 0.85
                if max_y is None:
                    max_y = center_y + 1.85
                floor_id = normalize_floor_id(
                    level.get("floor_id", level.get("level_id", level.get("level_index", index)))
                )
                if floor_id is None:
                    continue
                bands.append(
                    {
                        "floor_id": floor_id,
                        "min_y": min_y,
                        "max_y": max_y,
                        "center_y": center_y,
                        "sample_count": int(level.get("sample_count") or level.get("frame_count") or 0),
                        "xz_area_m2": float(level.get("xz_area_m2") or level.get("area_m2") or 0.0),
                        "xz_cell_count": int(level.get("xz_cell_count") or 0),
                        "floor_quality": str(level.get("floor_quality") or "unknown"),
                    }
                )
        if bands:
            return {
                "source": "maps_floor_levels",
                "bands": bands,
                "sample_count": sum(int(band.get("sample_count") or 0) for band in bands),
                "scene_package": str(scene_package),
                "floor_levels": str(levels_path),
            }
    return {
        # Floor bands are a map-preprocessing product.  Never infer them from
        # semantic GT: that would make normal query-time map fusion depend on
        # an oracle annotation file.  Callers may still run an explicit audit
        # against GT elsewhere, but candidates remain floor-unknown here.
        "source": "missing_or_invalid_maps_floor_levels",
        "bands": [],
        "sample_count": 0,
        "scene_package": str(scene_package),
        "floor_levels": str(levels_path),
    }


def floor_id_for_y(y: float | None, floor_context: dict[str, Any] | None) -> str | None:
    if y is None or not isinstance(floor_context, dict):
        return None
    bands = floor_context.get("bands") or []
    if not bands:
        return None
    containing = []
    for band in bands:
        try:
            if float(band["min_y"]) <= y <= float(band["max_y"]):
                containing.append(band)
        except Exception:
            continue
    if containing:
        best = min(containing, key=lambda band: abs(y - float(band.get("center_y") or 0.0)))
        return normalize_floor_id(best.get("floor_id"))
    best = min(bands, key=lambda band: abs(y - float(band.get("center_y") or 0.0)))
    return normalize_floor_id(best.get("floor_id")) if best else None


def frame_floor_ids_from_scene_package(
    scene_package: Path,
    floor_context: dict[str, Any] | None = None,
) -> tuple[dict[int, str], dict[str, Any]]:
    """Assign camera frames to source-package floors without semantic GT.

    A current dense package records the cluster selected by the exporter in
    ``trajectory.floor_y``.  Individual nav positions along a selected
    trajectory may climb a ramp or a stair; assigning those frames again by
    nearest raw Y centre splits one producer-declared floor across adjacent
    map tasks.  Trust the declaration only when *every* row maps exactly to a
    current ``floor_levels.json`` centre.  If a later floor-level repair has
    made the declaration stale, reject it as a whole and fall back to raw nav
    Y rather than silently mixing two memberships.
    """

    context = floor_context or floor_context_from_scene_package(scene_package)
    if not context.get("bands"):
        return {}, {
            "source": "missing_or_invalid_floor_levels",
            "frame_count": 0,
            "assigned_frame_count": 0,
        }

    trajectory_path = scene_package / "observations" / "trajectory.json"
    rows: list[dict[str, Any]] = []
    if trajectory_path.exists():
        try:
            loaded = read_json(trajectory_path)
            rows = [row for row in loaded if isinstance(row, dict)] if isinstance(loaded, list) else []
        except Exception:
            rows = []

    current_centers: dict[str, float] = {}
    for band in context.get("bands") or []:
        floor_id = normalize_floor_id(band.get("floor_id"))
        if floor_id is not None:
            current_centers[floor_id] = float(band.get("center_y") or 0.0)

    declared_assignments: dict[int, str] = {}
    declared_mismatch_count = 0
    for row in rows:
        try:
            frame_index = int(row.get("frame_index"))
            declared_y = _as_float(row.get("floor_y"))
        except (TypeError, ValueError):
            declared_y = None
        if declared_y is None:
            declared_mismatch_count += 1
            continue
        matches = [
            floor_id
            for floor_id, center_y in current_centers.items()
            if abs(float(declared_y) - center_y) <= 1e-4
        ]
        if len(matches) != 1:
            declared_mismatch_count += 1
            continue
        declared_assignments[frame_index] = matches[0]

    declared_floor_ids = set(declared_assignments.values())
    current_floor_ids = set(current_centers)
    declared_covers_current_levels = declared_floor_ids == current_floor_ids
    if rows and len(declared_assignments) == len(rows) and declared_covers_current_levels:
        return declared_assignments, {
            "source": "trajectory_declared_floor_y_current",
            "trajectory_path": str(trajectory_path),
            "frame_count": len(rows),
            "assigned_frame_count": len(declared_assignments),
            "unassigned_frame_count": 0,
            "declared_floor_label_current_count": len(declared_assignments),
            "declared_floor_label_mismatch_count": 0,
            "declared_floor_label_covers_current_levels": True,
            "raw_nav_position_assignment_count": 0,
            "legacy_cluster_center_fallback_count": 0,
            "pose_height_fallback_enabled": False,
        }

    assignments: dict[int, str] = {}
    raw_nav_position_count = 0
    legacy_cluster_center_count = 0
    for row in rows:
        try:
            frame_index = int(row.get("frame_index"))
        except (TypeError, ValueError):
            continue
        nav_position = row.get("nav_position")
        floor_y = None
        if isinstance(nav_position, (list, tuple)) and len(nav_position) >= 2:
            floor_y = _as_float(nav_position[1])
            if floor_y is not None:
                raw_nav_position_count += 1
        if floor_y is None:
            floor_y = _as_float(row.get("floor_y"))
            if floor_y is not None:
                legacy_cluster_center_count += 1
        floor_id = normalize_floor_id(floor_id_for_y(floor_y, context))
        if floor_id is not None:
            assignments[frame_index] = floor_id

    return assignments, {
        "source": (
            "trajectory_nav_position_y_after_stale_declared_floor_rejection"
            if rows and (declared_mismatch_count or not declared_covers_current_levels)
            else "trajectory_nav_position_y"
            if raw_nav_position_count
            else "legacy_trajectory_floor_cluster_center"
            if legacy_cluster_center_count
            else "missing_trajectory_floor_membership"
        ),
        "trajectory_path": str(trajectory_path),
        "frame_count": len(rows),
        "assigned_frame_count": len(assignments),
        "unassigned_frame_count": max(0, len(rows) - len(assignments)),
        "declared_floor_label_current_count": len(declared_assignments),
        "declared_floor_label_mismatch_count": declared_mismatch_count,
        "declared_floor_label_covers_current_levels": declared_covers_current_levels,
        "raw_nav_position_assignment_count": raw_nav_position_count,
        "legacy_cluster_center_fallback_count": legacy_cluster_center_count,
        "pose_height_fallback_enabled": False,
    }


def filter_depth_to_floor_band(
    depth_m: np.ndarray,
    pose_cv: np.ndarray,
    intrinsics: np.ndarray,
    *,
    lower_y: float,
    upper_y: float,
) -> tuple[np.ndarray, dict[str, int]]:
    """Back-project one depth image and keep points inside one floor slab."""

    depth = np.asarray(depth_m, dtype=np.float32)
    h, w = depth.shape[:2]
    v, u = np.indices((h, w), dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    x_cam = (u - float(intrinsics[0, 2])) * depth / float(intrinsics[0, 0])
    y_cam = (v - float(intrinsics[1, 2])) * depth / float(intrinsics[1, 1])
    z_cam = depth
    world_y = (
        float(pose_cv[1, 0]) * x_cam
        + float(pose_cv[1, 1]) * y_cam
        + float(pose_cv[1, 2]) * z_cam
        + float(pose_cv[1, 3])
    )
    keep = valid & (world_y >= float(lower_y)) & (world_y <= float(upper_y))
    filtered = np.where(keep, depth, 0.0).astype(np.float32)
    return filtered, {
        "valid_input_px": int(valid.sum()),
        "kept_px": int(keep.sum()),
        "removed_cross_level_px": int((valid & ~keep).sum()),
    }
