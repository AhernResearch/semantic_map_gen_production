"""Authoritative, GT-free policy for buildable map floors and depth slabs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from adapters.scripts.common_scene import read_json

from .floors import (
    DEFAULT_MIN_FLOOR_AREA_M2,
    DEFAULT_MIN_FLOOR_BASE_POSES,
    DEFAULT_MIN_FLOOR_FRAMES,
    floor_context_from_scene_package,
    frame_floor_ids_from_scene_package,
    normalize_floor_id,
)


# Matches the exporter-side ``min_floor_height_m`` contract.  A closer
# navigable plateau may be a valid local split-level, but it cannot be used as
# a global ceiling for an independently reconstructed map: doing so leaves
# less than one metre of vertical evidence above the selected ground level.
MIN_INDEPENDENT_FLOOR_CLEARANCE_M = 1.0


CANONICAL_FLOOR_COLLECTION_SCHEMA = "mapfusion_canonical_floor_collection_v3"
CANONICAL_FLOOR_POLICY_VERSION = "nav_geometry_observation_support_v3"
# All map producers consume the same membership and vertical slab.  The
# version is deliberately separate from the policy identifier because it is
# written into the materialized VLMaps floor-view metadata.
CANONICAL_FLOOR_VIEW_VERSION = "canonical_buildable_floor_depth_slab_v3"


def _effective_thresholds(
    *,
    min_frames: int,
    min_base_poses: int,
    min_area_m2: float,
) -> dict[str, int | float]:
    """Apply the non-negotiable lower bounds for a map floor.

    Callers may request a stricter audit, but a cache build must never relax
    the support gates and turn a stair landing into a canonical map.
    """

    return {
        "min_frames": max(DEFAULT_MIN_FLOOR_FRAMES, int(min_frames)),
        "min_independent_base_poses": max(DEFAULT_MIN_FLOOR_BASE_POSES, int(min_base_poses)),
        "min_geometry_cluster_area_m2": max(DEFAULT_MIN_FLOOR_AREA_M2, float(min_area_m2)),
    }


def canonical_floor_collection(
    scene_package: Path,
    floor_context: dict[str, Any] | None = None,
    *,
    min_frames: int = DEFAULT_MIN_FLOOR_FRAMES,
    min_base_poses: int = DEFAULT_MIN_FLOOR_BASE_POSES,
    min_area_m2: float = DEFAULT_MIN_FLOOR_AREA_M2,
) -> dict[str, Any]:
    """Select physical floors from nav geometry plus rendered-view support.

    ``floor_levels.json`` deliberately retains every navigation height plateau
    for collection diagnostics.  It is not itself a map-floor list: a stair
    landing can have a valid height and area but too little independent camera
    support.  This function is the only selector used by map builders and by
    the vertical depth slab that protects their objects from cross-floor data.
    """

    context = floor_context or floor_context_from_scene_package(scene_package)
    thresholds = _effective_thresholds(
        min_frames=min_frames,
        min_base_poses=min_base_poses,
        min_area_m2=min_area_m2,
    )
    assignments, assignment_meta = frame_floor_ids_from_scene_package(scene_package, context)
    trajectory_path = scene_package / "observations" / "trajectory.json"
    rows: list[dict[str, Any]] = []
    if trajectory_path.exists():
        loaded = read_json(trajectory_path)
        if isinstance(loaded, list):
            rows = [row for row in loaded if isinstance(row, dict)]
    rows_by_index: dict[int, dict[str, Any]] = {}
    for row in rows:
        try:
            rows_by_index[int(row.get("frame_index"))] = row
        except (TypeError, ValueError):
            continue

    floors: list[dict[str, Any]] = []
    for band in context.get("bands") or []:
        floor_id = normalize_floor_id(band.get("floor_id"))
        if floor_id is None:
            continue
        frame_indices = sorted(index for index, assigned in assignments.items() if assigned == floor_id)
        base_keys: set[tuple[Any, ...]] = set()
        for index in frame_indices:
            row = rows_by_index.get(index, {})
            source_index = row.get("source_base_index")
            if source_index is not None:
                base_keys.add(("source_base_index", str(source_index)))
                continue
            nav = row.get("nav_position")
            if isinstance(nav, (list, tuple)) and len(nav) >= 3:
                try:
                    base_keys.add(("nav_position", round(float(nav[0]), 3), round(float(nav[2]), 3)))
                except (TypeError, ValueError):
                    pass
        frame_count = len(frame_indices)
        base_pose_count = len(base_keys)
        area_m2 = float(band.get("xz_area_m2") or 0.0)
        reasons: list[str] = []
        if frame_count < thresholds["min_frames"]:
            reasons.append("too_few_frames")
        if base_pose_count < thresholds["min_independent_base_poses"]:
            reasons.append("too_few_independent_base_poses")
        if area_m2 < thresholds["min_geometry_cluster_area_m2"]:
            reasons.append("floor_cluster_area_too_small")
        floors.append(
            {
                "floor_id": floor_id,
                "floor_y": float(band.get("center_y") or 0.0),
                "geometry_cluster_sample_count": int(band.get("sample_count") or 0),
                "geometry_cluster_area_m2": area_m2,
                "frame_count": frame_count,
                "independent_base_pose_count": base_pose_count,
                "frame_indices": frame_indices,
                "buildable": not reasons,
                "skip_reasons": reasons,
            }
        )
    floors.sort(key=lambda row: float(row["floor_y"]))
    return {
        "schema": CANONICAL_FLOOR_COLLECTION_SCHEMA,
        "policy_version": CANONICAL_FLOOR_POLICY_VERSION,
        "scene_package": str(scene_package),
        "geometry_source": "floor_levels:navigable_ground_height_clusters",
        "observation_membership_source": assignment_meta.get("source"),
        "requested_thresholds": {
            "min_frames": max(1, int(min_frames)),
            "min_independent_base_poses": max(1, int(min_base_poses)),
            "min_geometry_cluster_area_m2": max(0.0, float(min_area_m2)),
        },
        "thresholds": thresholds,
        "frame_floor_assignment": assignment_meta,
        "floors": floors,
        "buildable_floor_ids": [row["floor_id"] for row in floors if row["buildable"]],
        "skipped_floor_ids": [row["floor_id"] for row in floors if not row["buildable"]],
    }


def canonical_query_floor_context(scene_package: Path) -> dict[str, Any]:
    """Return the canonical map floors used by every query-time producer.

    ``floor_levels.json`` retains small navigation plateaus for collection
    diagnostics.  They are deliberately absent from the cache task list, so
    admitting them again while resolving a query start can select a floor for
    which HMSG, ConceptGraphs, and VLMaps correctly produced no map.  Filter
    the runtime context through the same buildable-floor contract as all
    three cache builders while preserving the original floor identifiers.
    """

    raw_context = floor_context_from_scene_package(scene_package)
    collection = canonical_floor_collection(
        scene_package,
        floor_context=raw_context,
    )
    buildable_ids = set(collection["buildable_floor_ids"])
    bands = [
        dict(band)
        for band in raw_context.get("bands") or []
        if normalize_floor_id(band.get("floor_id")) in buildable_ids
    ]
    return {
        **raw_context,
        "source": "canonical_buildable_floor_context",
        "bands": bands,
        "raw_floor_ids": [
            normalize_floor_id(band.get("floor_id"))
            for band in raw_context.get("bands") or []
        ],
        "buildable_floor_ids": sorted(buildable_ids),
        "skipped_floor_ids": list(collection["skipped_floor_ids"]),
        "canonical_floor_policy_version": CANONICAL_FLOOR_POLICY_VERSION,
    }


def canonical_floor_vertical_band(
    floor_id: str | int,
    floor_context: dict[str, Any],
    *,
    buildable_floor_ids: Iterable[str | int] | None = None,
    lower_margin_m: float = 0.35,
    gap_to_next_floor_m: float = 0.35,
    default_height_m: float = 3.0,
    min_valid_floor_area_m2: float = DEFAULT_MIN_FLOOR_AREA_M2,
) -> dict[str, Any]:
    """Return a depth slab bounded by an independent higher map floor.

    Every buildable navigation plateau is retained for coverage.  It is not,
    however, automatically a valid *global vertical boundary*: a neighbouring
    plateau that leaves under :data:`MIN_INDEPENDENT_FLOOR_CLEARANCE_M` after
    the inter-floor margin is a split-level/nearby wing, not a ceiling.  Skip
    consecutive such plateaus while finding the next upper boundary.  HMSG
    may preserve internal strata inside the resulting external source band.

    The area-only fallback remains solely for legacy non-map callers.
    """

    requested = normalize_floor_id(floor_id)
    bands = list(floor_context.get("bands") or [])
    selected = next((band for band in bands if normalize_floor_id(band.get("floor_id")) == requested), None)
    if selected is None:
        raise ValueError(f"unknown canonical floor {floor_id}")
    if buildable_floor_ids is None:
        valid_ids = {
            normalize_floor_id(band.get("floor_id"))
            for band in bands
            if float(band.get("xz_area_m2") or 0.0) >= max(DEFAULT_MIN_FLOOR_AREA_M2, float(min_valid_floor_area_m2))
        }
        boundary_source = "legacy_area_only_fallback"
    else:
        valid_ids = {normalize_floor_id(value) for value in buildable_floor_ids}
        valid_ids.discard(None)
        boundary_source = "canonical_floor_collection"
    if requested not in valid_ids:
        raise ValueError(f"floor {floor_id} is not buildable under the canonical floor policy")
    floor_y = float(selected["center_y"])
    valid = sorted(
        (band for band in bands if normalize_floor_id(band.get("floor_id")) in valid_ids),
        key=lambda band: float(band["center_y"]),
    )
    selected_index = next(
        index
        for index, band in enumerate(valid)
        if normalize_floor_id(band.get("floor_id")) == requested
    )
    ignored_close_floor_ids: list[str] = []
    next_floor_y: float | None = None
    prior_y = floor_y
    for candidate in valid[selected_index + 1 :]:
        candidate_y = float(candidate["center_y"])
        clearance = candidate_y - prior_y - float(gap_to_next_floor_m)
        if clearance < MIN_INDEPENDENT_FLOOR_CLEARANCE_M:
            candidate_id = normalize_floor_id(candidate.get("floor_id"))
            if candidate_id is not None:
                ignored_close_floor_ids.append(candidate_id)
            prior_y = candidate_y
            continue
        next_floor_y = candidate_y
        break
    lower_y = floor_y - float(lower_margin_m)
    upper_y = floor_y + float(default_height_m)
    if next_floor_y is not None:
        upper_y = min(upper_y, next_floor_y - float(gap_to_next_floor_m))
    if upper_y <= lower_y + 0.5:
        raise ValueError(f"invalid floor slab for floor {floor_id}: [{lower_y:.3f}, {upper_y:.3f}]")
    return {
        "policy_version": CANONICAL_FLOOR_POLICY_VERSION,
        "floor_id": requested,
        "floor_y": floor_y,
        "lower_y": lower_y,
        "upper_y": upper_y,
        "next_valid_floor_y": next_floor_y,
        "ignored_close_floor_ids": ignored_close_floor_ids,
        "minimum_independent_floor_clearance_m": MIN_INDEPENDENT_FLOOR_CLEARANCE_M,
        "boundary_floor_ids": sorted(valid_ids),
        "boundary_source": boundary_source,
        "lower_margin_m": float(lower_margin_m),
        "gap_to_next_floor_m": float(gap_to_next_floor_m),
        "default_height_m": float(default_height_m),
    }
