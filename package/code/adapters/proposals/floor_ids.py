"""Canonical floor identities attached to query candidates."""

from __future__ import annotations

import math
import re
from typing import Any

DEFAULT_MIN_FLOOR_FRAMES = 32
DEFAULT_MIN_FLOOR_BASE_POSES = 4
DEFAULT_MIN_FLOOR_AREA_M2 = 10.0

def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def normalize_floor_id(value: Any) -> str | None:
    """Return one canonical key for equivalent floor-id spellings.

    The dense exporter writes folder names such as ``floor_00`` while query
    programs normally use integer ``level_id`` values.  Normalizing at the
    shared floor boundary avoids accidental cross-floor fallthrough caused by
    comparing ``"00"`` and ``"0"`` as unrelated strings.
    """

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    # ``floor_01`` is the on-disk convention of the dense exporter, while
    # query programs and HMSG provenance often carry the same id as ``1``.
    # Only unwrap an exact floor/level token; arbitrary semantic names must
    # remain untouched rather than being guessed into a numeric level.
    token = re.fullmatch(r"(?:floor|level)[_-]?(-?\d+(?:\.0+)?)", text, flags=re.IGNORECASE)
    if token:
        text = token.group(1)
    try:
        numeric = float(text)
    except (TypeError, ValueError):
        return text
    if math.isfinite(numeric) and numeric.is_integer():
        return str(int(numeric))
    return text

def candidate_floor_id(item: dict[str, Any] | None) -> str | None:
    if not isinstance(item, dict):
        return None
    for value in (item.get("floor_id"), item.get("level_id")):
        normalized = normalize_floor_id(value)
        if normalized is not None:
            return normalized
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    for key in ("floor_id", "level_id", "hmsg_floor_id", "scene_level_id"):
        value = metadata.get(key)
        normalized = normalize_floor_id(value)
        if normalized is not None:
            return normalized
    return None


def set_candidate_floor_id(item: dict[str, Any], floor_id: str | None, *, source: str) -> None:
    value = normalize_floor_id(floor_id)
    if value is None:
        return
    item["floor_id"] = value
    metadata = item.setdefault("metadata", {})
    if isinstance(metadata, dict):
        metadata["floor_id"] = value
        metadata["floor_assignment_source"] = source


def same_floor(left: dict[str, Any] | None, right: dict[str, Any] | None, *, unknown_matches: bool = False) -> bool:
    left_floor = candidate_floor_id(left)
    right_floor = candidate_floor_id(right)
    if left_floor is None or right_floor is None:
        return bool(unknown_matches)
    return left_floor == right_floor


def _candidate_y(item: dict[str, Any]) -> float | None:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    for container, key in ((metadata, "bbox_world_min"), (metadata, "bbox_world_max")):
        value = container.get(key) if isinstance(container, dict) else None
        if isinstance(value, list | tuple) and len(value) >= 2:
            y = _as_float(value[1])
            if y is not None:
                return y
    point = item.get("world_position") or metadata.get("source_world_position_xyz")
    if isinstance(point, list | tuple) and len(point) >= 2:
        return _as_float(point[1])
    return None
