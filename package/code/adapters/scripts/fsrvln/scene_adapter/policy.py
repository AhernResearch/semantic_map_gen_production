"""Canonical-floor selection and immutable adapter-view fingerprints."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from adapters.proposals.floors import (
    canonical_floor_collection,
    canonical_floor_vertical_band,
    filter_depth_to_floor_band,
    floor_context_from_scene_package,
    normalize_floor_id,
)
from adapters.proposals.floor_policy import (
    CANONICAL_FLOOR_COLLECTION_SCHEMA,
    CANONICAL_FLOOR_POLICY_VERSION,
)


C_HM3DSEM = np.diag([1.0, -1.0, -1.0, 1.0])
SOURCE_VIEW_SELECTION_VERSION = "canonical_buildable_floor_slab_v6_exporter_membership_independent_ceiling"

def resolve_canonical_floor(
    scene_root: Path,
    floor_id: int,
    *,
    camera_height_m: float,
    camera_y_tolerance_m: float,
    label_tolerance_m: float,
    lower_margin_m: float,
    gap_to_next_floor_m: float,
    default_height_m: float,
    min_area_m2: float,
) -> dict[str, Any]:
    """Return the runtime-only frame-selection rule for one package floor.

    The scene package owns the canonical floor convention.  FSRVLN's native
    floor detector is still allowed to split a *single* floor internally, but
    it must never see frames from another physical level.
    """
    path = scene_root / "maps" / "floor_levels.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing canonical floor metadata: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    floors = payload if isinstance(payload, list) else payload.get("floors") or payload.get("levels") or []
    selected = next(
        (
            row
            for row in floors
            if int(row.get("level_index", row.get("floor_index", row.get("level_id", -1)))) == int(floor_id)
        ),
        None,
    )
    if selected is None:
        available = sorted(
            int(row.get("level_index", row.get("floor_index", row.get("level_id", -1))))
            for row in floors
        )
        raise ValueError(f"Unknown canonical floor {floor_id} for {scene_root.name}; available={available}")
    floor_y = float(selected["floor_y"])
    source_floor_record = {
        key: value
        for key, value in selected.items()
        if key not in {"candidate_indices", "sample_indices", "points"}
    }
    collection = canonical_floor_collection(
        scene_root,
        floor_context_from_scene_package(scene_root),
        min_area_m2=min_area_m2,
    )
    buildable_floor_ids = collection["buildable_floor_ids"]
    if normalize_floor_id(floor_id) not in buildable_floor_ids:
        floor_record = next(
            (row for row in collection["floors"] if normalize_floor_id(row.get("floor_id")) == normalize_floor_id(floor_id)),
            None,
        )
        reasons = [] if floor_record is None else floor_record["skip_reasons"]
        raise ValueError(f"canonical floor {floor_id} is not buildable: {reasons}")
    band = canonical_floor_vertical_band(
        floor_id,
        floor_context_from_scene_package(scene_root),
        buildable_floor_ids=buildable_floor_ids,
        lower_margin_m=lower_margin_m,
        gap_to_next_floor_m=gap_to_next_floor_m,
        default_height_m=default_height_m,
    )
    return {
        "floor_id": int(floor_id),
        "floor_y": floor_y,
        "expected_camera_y": floor_y + float(camera_height_m),
        "camera_height_m": float(camera_height_m),
        "camera_y_tolerance_m": float(camera_y_tolerance_m),
        "label_tolerance_m": float(label_tolerance_m),
        "vertical_band": {
            "policy_version": band["policy_version"],
            "lower_y": band["lower_y"],
            "upper_y": band["upper_y"],
            "lower_margin_m": float(lower_margin_m),
            "gap_to_next_floor_m": float(gap_to_next_floor_m),
            "default_height_m": float(default_height_m),
            "next_valid_floor_y": band["next_valid_floor_y"],
            "ignored_close_floor_ids": band["ignored_close_floor_ids"],
            "minimum_independent_floor_clearance_m": band["minimum_independent_floor_clearance_m"],
            "boundary_floor_ids": band["boundary_floor_ids"],
            "boundary_source": band["boundary_source"],
        },
        "canonical_floor_collection": collection,
        "source_floor_record": source_floor_record,
    }


def filter_depth_to_canonical_floor(
    depth_m: np.ndarray,
    pose_cv: np.ndarray,
    intrinsics: np.ndarray,
    *,
    lower_y: float,
    upper_y: float,
) -> tuple[np.ndarray, dict[str, int]]:
    """Remove RGB-D points that geometrically belong to another level."""
    return filter_depth_to_floor_band(
        depth_m,
        pose_cv,
        intrinsics,
        lower_y=lower_y,
        upper_y=upper_y,
    )


_GRAPH_FINALIZATION_SETTINGS = frozenset(
    {"merge_objects_graph", "feature_dbscan_max_points"}
)


def view_materialization_settings(settings: dict[str, object]) -> dict[str, object]:
    """Return only settings that can alter generated RGB-D/pose view files.

    Room-object consolidation and representative feature aggregation happen
    after feature extraction.  Treating either as a view setting used to
    delete valid dense feature checkpoints whenever the final graph policy
    changed would force an unnecessary GPU re-run.
    """

    return {key: value for key, value in settings.items() if key not in _GRAPH_FINALIZATION_SETTINGS}


def reusable_view_matches(
    out: Path,
    cfg_path: Path,
    *,
    source_scene_package: Path,
    effective_scene_id: str,
    canonical_floor_id: int | None,
    settings: dict[str, object],
) -> bool:
    """Return true only for a complete view produced with the same settings.

    FSRVLN itself has feature/merge checkpoints, but recreating the adapter
    view used to rewrite every filtered depth image before those checkpoints
    could be reused.  The manifest is the boundary: matching inputs are safe
    to resume; callers must pass --force-view when changing preprocessing.
    """
    manifest_path = out / "adapter_manifest.json"
    if not (manifest_path.exists() and cfg_path.exists()):
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("source_scene_package") != str(source_scene_package):
        return False
    # A complete file count alone is not enough.  HMSG's own checkpoint
    # metadata does not encode the selected source-frame identities, so an
    # old view without this fingerprint must be rebuilt once before it can be
    # safely reused.
    if not manifest.get("view_input_fingerprint"):
        return False
    if manifest.get("effective_scene_id") != effective_scene_id:
        return False
    existing_floor = (manifest.get("canonical_floor") or {}).get("floor_id")
    if existing_floor != canonical_floor_id:
        return False
    if canonical_floor_id is not None:
        existing_canonical = manifest.get("canonical_floor") or {}
        existing_collection = existing_canonical.get("canonical_floor_collection") or {}
        existing_band = existing_canonical.get("vertical_band") or {}
        # A prior adapter omitted the slab policy version even though it
        # filtered the same pixels.  Treat that artifact as stale: downstream
        # room refinement relies on the field to prove that view, BEV, and
        # graph used one canonical-floor contract.
        if (
            existing_collection.get("schema") != CANONICAL_FLOOR_COLLECTION_SCHEMA
            or existing_collection.get("policy_version") != CANONICAL_FLOOR_POLICY_VERSION
            or existing_band.get("policy_version") != CANONICAL_FLOOR_POLICY_VERSION
            or existing_band.get("boundary_source") != "canonical_floor_collection"
        ):
            return False
    existing_settings = manifest.get("fsrvln_settings") or {}
    materialization_settings = view_materialization_settings(settings)
    if any(existing_settings.get(key) != value for key, value in materialization_settings.items()):
        return False
    selected_count = int(manifest.get("selected_frame_count") or 0)
    if selected_count <= 0:
        return False
    return all(
        len(list((out / subdir).glob(pattern))) == selected_count
        for subdir, pattern in (("rgb", "*.png"), ("depth", "*.png"), ("pose", "*.txt"))
    )


def clear_view_payload(out: Path) -> None:
    """Remove only regenerated adapter payloads, never HMSG result/checkpoint data."""
    for subdir in ("rgb", "depth", "pose"):
        path = out / subdir
        if not path.exists():
            continue
        for child in path.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()


def invalidate_hmsg_checkpoints(save_root: Path, effective_scene_id: str) -> list[str]:
    """Drop only HMSG cache products tied to a changed adapter view.

    HMSG's native checkpoint metadata validates frame count but not the exact
    selected source frames.  Keeping these three files after a view rewrite
    can therefore reconstruct a graph from stale features.  Timestamped graph
    outputs and diagnostics are deliberately preserved for auditability.
    """

    tmp = save_root / "hm3dsem" / effective_scene_id / "tmp"
    removed: list[str] = []
    for name in (
        "feature_checkpoint.pkl",
        "merge_checkpoint.pkl",
        "mask_feature_checkpoint.pkl",
        "room_views.npz",
        "full_pcd_cache.ply",
        "full_pcd_cache.json",
    ):
        path = tmp / name
        if path.exists():
            path.unlink()
            removed.append(str(path))
    return removed


def view_input_fingerprint(
    *,
    source_scene_package: Path,
    effective_scene_id: str,
    requested_settings: dict[str, object],
    canonical_floor: dict[str, Any] | None,
    selected_source_frame_indices: list[int],
    source_data_signature: dict[str, object],
) -> str:
    """Stable identity of the RGB-D view that HMSG is permitted to reuse."""

    payload = {
        "version": SOURCE_VIEW_SELECTION_VERSION,
        "source_scene_package": str(source_scene_package),
        "effective_scene_id": effective_scene_id,
        "requested_settings": view_materialization_settings(requested_settings),
        "canonical_floor": canonical_floor,
        "selected_source_frame_indices": selected_source_frame_indices,
        "source_data_signature": source_data_signature,
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_data_signature(
    scene_root: Path,
    rgb_paths: list[Path],
    depth_paths: list[Path],
    selected_indices: list[int],
) -> dict[str, object]:
    """Fingerprint the small metadata plus exactly the RGB-D files in use."""

    def stat_row(path: Path) -> dict[str, int | str]:
        stat = path.stat()
        return {"path": str(path.relative_to(scene_root)), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}

    metadata_paths = [
        scene_root / "package_meta.json",
        scene_root / "observations" / "poses.txt",
        scene_root / "observations" / "intrinsics.json",
        scene_root / "observations" / "trajectory.json",
        scene_root / "maps" / "floor_levels.json",
    ]
    return {
        "metadata": [stat_row(path) for path in metadata_paths if path.exists()],
        "selected_frames": [
            {
                "index": int(index),
                "rgb": stat_row(rgb_paths[index]),
                "depth": stat_row(depth_paths[index]),
            }
            for index in selected_indices
        ],
    }
