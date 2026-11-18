"""Stable floor API backed by focused identity and scene-context modules."""

from __future__ import annotations

from typing import Any

from .floor_context import (
    DEFAULT_QUERY_START_NAV_SNAP_DISTANCE_M,
    filter_depth_to_floor_band,
    floor_context_from_scene_package,
    floor_id_for_start_position,
    floor_id_for_y,
    frame_floor_ids_from_scene_package,
)
from .floor_ids import (
    DEFAULT_MIN_FLOOR_AREA_M2,
    DEFAULT_MIN_FLOOR_BASE_POSES,
    DEFAULT_MIN_FLOOR_FRAMES,
    _candidate_y,
    candidate_floor_id,
    normalize_floor_id,
    same_floor,
    set_candidate_floor_id,
)

def query_level_id(spec: dict[str, Any] | None) -> str | None:
    """Return a floor explicitly requested by the runtime query.

    Benchmark ``constraints.level_id`` and ``context.level_id`` describe the
    annotated answer and are evaluator metadata, not information available to
    a deployed system.  Reading them here both leaks GT and can misroute a
    query when a geometry-derived canonical floor numbering differs from the
    dataset annotation.  Runtime planners place an explicit user floor request
    at the top level (``level_id`` or one of the named runtime aliases).
    """
    if not isinstance(spec, dict):
        return None
    for key in ("runtime_level_id", "requested_level_id", "query_level_id", "level_id"):
        value = spec.get(key)
        normalized = normalize_floor_id(value)
        if normalized is not None:
            return normalized
    return None


def query_scope(spec: dict[str, Any] | None) -> str | None:
    """Return the explicit runtime scope of a query.

    A query may be evaluated across every floor, or it may describe the
    robot's current floor.  This is a runtime contract, not a benchmark GT
    annotation.  In particular, this helper never reads ``constraints`` or
    ``context``.
    """

    if not isinstance(spec, dict):
        return None
    value = str(spec.get("query_scope") or spec.get("runtime_scope") or "").strip().lower()
    aliases = {
        "current-floor": "current_floor",
        "current_floor": "current_floor",
        "start-floor": "current_floor",
        "start_floor": "current_floor",
        "all-floor": "all_floors",
        "all-floors": "all_floors",
        "all_floor": "all_floors",
        "all_floors": "all_floors",
        "multi-floor": "all_floors",
        "multi_floor": "all_floors",
    }
    return aliases.get(value)


def resolve_query_level_id(
    spec: dict[str, Any] | None,
    floor_context: dict[str, Any] | None,
) -> str | None:
    """Resolve an online floor request without consuming evaluator metadata.

    Explicit runtime floor ids take precedence.  For a ``current_floor``
    query, the robot/start-pose height is mapped to the geometry-derived floor
    bands.  Ordinary open-ended queries remain multi-floor.
    """

    explicit = query_level_id(spec)
    if explicit is not None:
        return explicit
    if query_scope(spec) != "current_floor" or not isinstance(spec, dict):
        return None
    start = spec.get("start_position")
    return floor_id_for_start_position(start, floor_context)

def assign_floor_ids(
    rows: list[dict[str, Any]],
    floor_context: dict[str, Any] | None,
    *,
    source: str = "height_band",
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    for row in rows:
        if not overwrite and candidate_floor_id(row) is not None:
            continue
        floor_id = floor_id_for_y(_candidate_y(row), floor_context)
        set_candidate_floor_id(row, floor_id, source=source)
    return rows


def filter_candidates_to_floor(rows: list[dict[str, Any]], floor_id: str | None) -> list[dict[str, Any]]:
    requested = normalize_floor_id(floor_id)
    if requested is None:
        return rows
    return [row for row in rows if candidate_floor_id(row) == requested]


def __getattr__(name: str):
    """Lazily preserve the historical map-policy import boundary.

    ``floor_policy`` consumes the context helpers above, so eager re-export
    would be circular when that focused module is imported directly.
    """

    if name in {"canonical_floor_collection", "canonical_floor_vertical_band"}:
        from . import floor_policy

        return getattr(floor_policy, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
