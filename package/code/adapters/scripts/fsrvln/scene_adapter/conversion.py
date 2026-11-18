"""Scene-package to HM3DSem-compatible FSRVLN view conversion."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from adapters.proposals.floors import (
    floor_context_from_scene_package,
    frame_floor_ids_from_scene_package,
    normalize_floor_id,
)
from adapters.scripts.common_scene import (
    DEFAULT_RESULTS,
    ensure_empty_dir,
    link_or_copy,
    load_poses_txt,
    load_scene_package,
    read_json,
    save_depth_png_mm,
    sorted_files,
    write_json,
)

from .policy import (
    C_HM3DSEM,
    SOURCE_VIEW_SELECTION_VERSION,
    clear_view_payload,
    filter_depth_to_canonical_floor,
    invalidate_hmsg_checkpoints,
    resolve_canonical_floor,
    reusable_view_matches,
    source_data_signature,
    view_input_fingerprint,
)
from .reconstruction_config import write_reconstruction_config


def restore_finalization_config(
    view_dir: Path,
    config_path: Path,
    *,
    graph_save_root: Path,
) -> bool:
    """Restore a missing config from its immutable view manifest only.

    Historical cache transfers occasionally retained the dense RGB-D view and
    feature checkpoint but omitted the small YAML config.  Re-rendering this
    config changes neither the view payload nor the checkpoint, so it is safe
    to do immediately before CPU graph finalization.
    """

    manifest_path = view_dir / "adapter_manifest.json"
    try:
        manifest = read_json(manifest_path)
        settings = manifest.get("fsrvln_settings")
        scene_id = str(manifest.get("effective_scene_id") or "")
    except (OSError, ValueError):
        return False
    if not scene_id or not isinstance(settings, dict):
        return False
    write_reconstruction_config(
        config_path,
        scene_id=scene_id,
        data_view_root=view_dir.parent.parent,
        graph_save_root=graph_save_root,
        settings=settings,
    )
    return True


def upgrade_graph_finalization_policy(
    out: Path,
    cfg_path: Path,
    requested_settings: dict[str, object],
) -> bool:
    """Upgrade graph-only settings without touching the RGB-D view/checkpoints."""

    manifest_path = out / "adapter_manifest.json"
    try:
        manifest = read_json(manifest_path)
        settings = manifest.get("fsrvln_settings")
    except (OSError, ValueError):
        return False
    if not isinstance(settings, dict):
        return False
    changed = False
    if (
        requested_settings.get("merge_objects_graph") is True
        and settings.get("merge_objects_graph") is not True
    ):
        settings["merge_objects_graph"] = True
        changed = True
    requested_feature_cap = requested_settings.get("feature_dbscan_max_points")
    if (
        isinstance(requested_feature_cap, int)
        and requested_feature_cap > 0
        and settings.get("feature_dbscan_max_points") != requested_feature_cap
    ):
        settings["feature_dbscan_max_points"] = requested_feature_cap
        changed = True
    if not changed:
        return False
    manifest["fsrvln_settings"] = settings
    write_json(manifest_path, manifest)
    try:
        config_text = cfg_path.read_text(encoding="utf-8")
    except OSError:
        return True
    updated, _ = re.subn(
        r"(?m)^(\s*merge_objects_graph:\s*)false\s*$",
        r"\1true",
        config_text,
        count=1,
    )
    if isinstance(requested_feature_cap, int) and requested_feature_cap > 0:
        cap_line = f"  feature_dbscan_max_points: {requested_feature_cap}"
        updated, cap_replacements = re.subn(
            r"(?m)^\s*feature_dbscan_max_points:\s*.*$",
            cap_line,
            updated,
            count=1,
        )
        if not cap_replacements:
            updated, _ = re.subn(
                r"(?m)^(\s*feature_dbscan_eps:\s*.*)$",
                r"\1\n" + cap_line,
                updated,
                count=1,
            )
    cfg_path.write_text(updated, encoding="utf-8")
    return True

def convert(
    scene_package: Path,
    output_root: Path,
    config_root: Path,
    mode: str,
    force: bool,
    *,
    skip_frames: int = 1,
    min_pcd_points: int = 50,
    voxel_size: float = 0.05,
    grid_resolution: float = 0.05,
    room_wall_threshold_ratio: float = 0.25,
    min_valid_floor_height_m: float = 1.0,
    premerge_min_pcd_points: int = 12,
    merge_type: str = "sequential",
    max_hierarchical_merge_rounds: int | None = None,
    resume_feature_checkpoint: bool = True,
    resume_merge_checkpoint: bool = True,
    max_black_ratio: float | None = None,
    min_valid_depth_ratio: float | None = None,
    save_root: Path | None = None,
    resize_width: int | None = None,
    resize_height: int | None = None,
    obj_labels: str = "MP3D_EQA_STRUCTURAL_LABELS",
    device: str = "cuda",
    source_frame_stride: int = 1,
    target_source_frames_per_m2: float | None = None,
    min_source_frames: int | None = None,
    max_source_frames: int | None = None,
    source_yaw_offsets: tuple[float, ...] | None = None,
    source_yaw_tolerance_deg: float = 1e-3,
    merge_objects_graph: bool = True,
    sam_type: str = "vit_h",
    sam_checkpoint: str = "checkpoints/sam_vit_h_4b8939.pth",
    sam_points_per_side: int = 24,
    sam_points_per_batch: int = 144,
    sam_pred_iou_thresh: float = 0.88,
    sam_crop_n_layers: int = 0,
    sam_stability_score_thresh: float = 0.92,
    sam_min_mask_region_area: int = 150,
    denoise_full_pcd: bool = False,
    radius_outlier_full_pcd: bool = False,
    full_pcd_dbscan_eps: float = 0.01,
    full_pcd_dbscan_min_points: int = 100,
    full_pcd_radius_nb_points: int = 1000,
    full_pcd_radius: float = 1.0,
    save_intermediate_results: bool = True,
    save_full_pcd_during_create: bool = True,
    save_masked_pcds: bool = True,
    save_full_pcd: bool = True,
    save_full_pcd_feats: bool = True,
    canonical_floor_id: int | None = None,
    canonical_floor_camera_height_m: float = 1.5,
    canonical_floor_camera_y_tolerance_m: float = 0.85,
    canonical_floor_label_tolerance_m: float = 0.05,
    canonical_floor_lower_margin_m: float = 0.35,
    canonical_floor_gap_to_next_m: float = 0.35,
    canonical_floor_default_height_m: float = 3.0,
    canonical_floor_min_area_m2: float = 10.0,
) -> tuple[Path, Path]:
    output_root = output_root.resolve()
    config_root = config_root.resolve()
    pkg = load_scene_package(scene_package)
    root = pkg["root"]
    source_scene_id = str(pkg["meta"]["scene_id"])
    canonical_floor = (
        resolve_canonical_floor(
            root,
            int(canonical_floor_id),
            camera_height_m=canonical_floor_camera_height_m,
            camera_y_tolerance_m=canonical_floor_camera_y_tolerance_m,
            label_tolerance_m=canonical_floor_label_tolerance_m,
            lower_margin_m=canonical_floor_lower_margin_m,
            gap_to_next_floor_m=canonical_floor_gap_to_next_m,
            default_height_m=canonical_floor_default_height_m,
            min_area_m2=canonical_floor_min_area_m2,
        )
        if canonical_floor_id is not None
        else None
    )
    scene_id = (
        f"{source_scene_id}__floor_{int(canonical_floor_id):02d}"
        if canonical_floor_id is not None
        else source_scene_id
    )
    out = output_root / "fsrvln" / scene_id
    cfg_path = config_root / f"semantic_scene_reconstruction_habitat_{scene_id}.yaml"
    save_path = (DEFAULT_RESULTS / "fsrvln") if save_root is None else save_root.resolve()
    requested_settings: dict[str, object] = {
        "skip_frames": skip_frames,
        "min_pcd_points": min_pcd_points,
        "premerge_min_pcd_points": premerge_min_pcd_points,
        "feature_dbscan_max_points": 4096,
        "merge_type": merge_type,
        "max_hierarchical_merge_rounds": max_hierarchical_merge_rounds,
        "resume_feature_checkpoint": bool(resume_feature_checkpoint),
        "resume_merge_checkpoint": bool(resume_merge_checkpoint),
        "voxel_size": voxel_size,
        "grid_resolution": grid_resolution,
        "room_wall_threshold_ratio": room_wall_threshold_ratio,
        "min_valid_floor_height_m": min_valid_floor_height_m,
        "max_black_ratio": max_black_ratio,
        "min_valid_depth_ratio": min_valid_depth_ratio,
        "save_root": str(save_path),
        "resize_width": resize_width,
        "resize_height": resize_height,
        "obj_labels": obj_labels,
        "device": device,
        "source_frame_stride": source_frame_stride,
        "source_view_selection_version": SOURCE_VIEW_SELECTION_VERSION,
        "target_source_frames_per_m2": target_source_frames_per_m2,
        "min_source_frames": min_source_frames,
        "max_source_frames": max_source_frames,
        "source_yaw_offsets": list(source_yaw_offsets or ()),
        "source_yaw_tolerance_deg": source_yaw_tolerance_deg,
        "merge_objects_graph": bool(merge_objects_graph),
        "sam_type": sam_type,
        "sam_checkpoint": sam_checkpoint,
        "sam_points_per_side": sam_points_per_side,
        "sam_points_per_batch": sam_points_per_batch,
        "sam_pred_iou_thresh": sam_pred_iou_thresh,
        "sam_crop_n_layers": sam_crop_n_layers,
        "sam_stability_score_thresh": sam_stability_score_thresh,
        "sam_min_mask_region_area": sam_min_mask_region_area,
        "denoise_full_pcd": bool(denoise_full_pcd),
        "radius_outlier_full_pcd": bool(radius_outlier_full_pcd),
        "full_pcd_dbscan_eps": full_pcd_dbscan_eps,
        "full_pcd_dbscan_min_points": full_pcd_dbscan_min_points,
        "full_pcd_radius_nb_points": full_pcd_radius_nb_points,
        "full_pcd_radius": full_pcd_radius,
        "save_intermediate_results": bool(save_intermediate_results),
        "save_full_pcd_during_create": bool(save_full_pcd_during_create),
        "save_masked_pcds": bool(save_masked_pcds),
        "save_full_pcd": bool(save_full_pcd),
        "save_full_pcd_feats": bool(save_full_pcd_feats),
        # HM3DSemDataset now consumes this file when it is present instead of
        # assuming its historical 90-degree camera model.
        "camera_intrinsics_mode": "scene_package_intrinsics_json_v1",
    }
    if not force and reusable_view_matches(
        out,
        cfg_path,
        source_scene_package=root,
        effective_scene_id=scene_id,
        canonical_floor_id=int(canonical_floor_id) if canonical_floor_id is not None else None,
        settings=requested_settings,
    ):
        if upgrade_graph_finalization_policy(out, cfg_path, requested_settings):
            print("[scene-package-to-fsrvln] upgraded graph-only room consolidation policy; reusing feature checkpoints")
        print(f"[scene-package-to-fsrvln] reusing matching view: {out}")
        return out, cfg_path
    invalidated_checkpoints = invalidate_hmsg_checkpoints(save_path, scene_id)
    ensure_empty_dir(out, force=force)
    # A configuration change (for example source stride) must not leave old
    # numbered RGB-D frames behind for the HMSG dataset loader to discover.
    clear_view_payload(out)
    for rel in ["rgb", "depth", "pose"]:
        (out / rel).mkdir(parents=True, exist_ok=True)

    rgb_paths = sorted_files(root / "observations" / "rgb", "*.png")
    depth_paths = sorted_files(root / "observations" / "depth", "*.npy")
    poses_cv = load_poses_txt(root / "observations" / "poses.txt")
    if not (len(rgb_paths) == len(depth_paths) == len(poses_cv)):
        raise RuntimeError("RGB, depth, and pose counts do not match.")

    trajectory_path = root / "observations" / "trajectory.json"
    trajectory = {}
    if trajectory_path.exists():
        import json

        trajectory = {int(item["frame_index"]): item for item in json.loads(trajectory_path.read_text(encoding="utf-8"))}
    intrinsics_path = root / "observations" / "intrinsics.json"
    intrinsics_payload: dict[str, Any] | None = None
    intrinsics = None
    if intrinsics_path.exists():
        payload = json.loads(intrinsics_path.read_text(encoding="utf-8"))
        matrix = np.asarray(payload.get("K"), dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)) or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
            raise ValueError(f"Invalid camera matrix in {intrinsics_path}")
        intrinsics_payload = payload
        intrinsics = matrix
    elif canonical_floor is not None:
        raise FileNotFoundError(f"Canonical floor depth filtering requires {intrinsics_path}")
    candidate_indices: list[int] = []
    rejected: list[dict[str, float | int]] = []
    skipped_by_yaw = 0
    skipped_by_canonical_floor = 0
    skipped_by_canonical_floor_label = 0
    skipped_by_missing_floor_membership = 0
    canonical_floor_assignments: dict[int, str] = {}
    canonical_floor_assignment_meta: dict[str, Any] = {}
    requested_canonical_floor_id: str | None = None
    if canonical_floor is not None:
        requested_canonical_floor_id = normalize_floor_id(canonical_floor["floor_id"])
        canonical_floor_assignments, canonical_floor_assignment_meta = frame_floor_ids_from_scene_package(
            root,
            floor_context_from_scene_package(root),
        )
    for frame_idx in range(len(rgb_paths)):
        if canonical_floor is not None:
            assigned_floor_id = normalize_floor_id(canonical_floor_assignments.get(frame_idx))
            if assigned_floor_id is None:
                skipped_by_missing_floor_membership += 1
                continue
            if assigned_floor_id != requested_canonical_floor_id:
                skipped_by_canonical_floor_label += 1
                continue
        if source_yaw_offsets:
            yaw = trajectory.get(frame_idx, {}).get("yaw_offset_deg")
            if yaw is None or not any(abs(float(yaw) - target) <= source_yaw_tolerance_deg for target in source_yaw_offsets):
                skipped_by_yaw += 1
                continue
        quality = trajectory.get(frame_idx, {}).get("quality", {})
        black = quality.get("black_ratio")
        valid = quality.get("valid_depth_ratio")
        reject = False
        if max_black_ratio is not None and black is not None and float(black) > max_black_ratio:
            reject = True
        if min_valid_depth_ratio is not None and valid is not None and float(valid) < min_valid_depth_ratio:
            reject = True
        if reject:
            rejected.append(
                {
                    "frame_index": frame_idx,
                    "black_ratio": float(black) if black is not None else -1.0,
                    "valid_depth_ratio": float(valid) if valid is not None else -1.0,
                }
            )
            continue
        candidate_indices.append(frame_idx)

    source_frame_stride = max(1, int(source_frame_stride))
    selected_indices = candidate_indices[::source_frame_stride]
    source_selection: dict[str, object] = {
        "version": SOURCE_VIEW_SELECTION_VERSION,
        "source_frame_stride": source_frame_stride,
        "target_source_frames_per_m2": target_source_frames_per_m2,
        "min_source_frames": min_source_frames,
        "max_source_frames": max_source_frames,
        "candidate_frame_count": len(candidate_indices),
        "post_stride_frame_count": len(selected_indices),
        "target_frame_count": None,
        "uniform_subsampled": False,
        "sampling_pool": "post_stride",
        "canonical_floor_assignment": canonical_floor_assignment_meta,
    }
    if canonical_floor is not None and target_source_frames_per_m2 is not None:
        area_m2 = float(canonical_floor["source_floor_record"].get("xz_area_m2") or 0.0)
        if area_m2 > 0.0:
            target = int(np.ceil(area_m2 * float(target_source_frames_per_m2)))
            if min_source_frames is not None:
                target = max(target, int(min_source_frames))
            if max_source_frames is not None:
                target = min(target, int(max_source_frames))
            target = max(1, target)
            source_selection["floor_area_m2"] = area_m2
            source_selection["target_frame_count"] = target
            if len(selected_indices) > target:
                take = np.linspace(0, len(selected_indices) - 1, num=target, dtype=np.int64)
                selected_indices = [selected_indices[int(index)] for index in np.unique(take)]
                source_selection["uniform_subsampled"] = True
            elif len(selected_indices) < target and len(candidate_indices) >= target:
                # Preserve the requested stride when it already supplies the
                # target density, but relax it for sparse floors rather than
                # silently running below the configured minimum.
                take = np.linspace(0, len(candidate_indices) - 1, num=target, dtype=np.int64)
                selected_indices = [candidate_indices[int(index)] for index in np.unique(take)]
                source_selection["uniform_subsampled"] = True
                source_selection["sampling_pool"] = "all_candidates_relaxed_for_target"
            elif len(selected_indices) < target:
                selected_indices = list(candidate_indices)
                source_selection["sampling_pool"] = "all_available_below_target"
    source_selection["selected_frame_count"] = len(selected_indices)
    requested_frame_count = source_selection.get("target_frame_count")
    if requested_frame_count is None:
        requested_frame_count = min_source_frames
    if requested_frame_count is not None:
        requested_frame_count = max(1, int(requested_frame_count))
        source_selection["requested_frame_count"] = requested_frame_count
        source_selection["required_selected_frame_count"] = min(
            requested_frame_count,
            len(candidate_indices),
        )
    source_selection["available_candidate_frame_count"] = len(candidate_indices)
    if canonical_floor is not None and not selected_indices:
        raise RuntimeError(
            f"Canonical floor {canonical_floor['floor_id']} selected no frames for {source_scene_id}; "
            f"assignment_source={canonical_floor_assignment_meta.get('source')}, "
            f"assigned_frames={canonical_floor_assignment_meta.get('assigned_frame_count', 0)}"
        )

    view_intrinsics: dict[str, Any] | None = None
    if intrinsics is not None:
        source_width, source_height = Image.open(rgb_paths[0]).size
        target_width = int(resize_width) if resize_width is not None else source_width
        target_height = int(resize_height) if resize_height is not None else source_height
        source_calibration_width = int(intrinsics_payload.get("width", source_width)) if intrinsics_payload else source_width
        source_calibration_height = int(intrinsics_payload.get("height", source_height)) if intrinsics_payload else source_height
        if source_calibration_width <= 0 or source_calibration_height <= 0:
            raise ValueError(f"Invalid calibration image size in {intrinsics_path}")
        view_matrix = intrinsics.copy()
        view_matrix[0, :] *= float(target_width) / float(source_calibration_width)
        view_matrix[1, :] *= float(target_height) / float(source_calibration_height)
        view_matrix[2, :] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        view_intrinsics = {
            "schema": "mapfusion_fsrvln_intrinsics_v1",
            "width": target_width,
            "height": target_height,
            "K": view_matrix.tolist(),
            "source_intrinsics": str(intrinsics_path),
            "source_width": source_calibration_width,
            "source_height": source_calibration_height,
            "camera_frame": (intrinsics_payload or {}).get("camera_frame"),
        }
        write_json(out / "intrinsics.json", view_intrinsics)

    input_fingerprint = view_input_fingerprint(
        source_scene_package=root,
        effective_scene_id=scene_id,
        requested_settings=requested_settings,
        canonical_floor=canonical_floor,
        selected_source_frame_indices=selected_indices,
        source_data_signature=source_data_signature(root, rgb_paths, depth_paths, selected_indices),
    )

    depth_filter_rows: list[dict[str, int]] = []
    for out_i, src_i in enumerate(selected_indices):
        rgb = rgb_paths[src_i]
        depth = depth_paths[src_i]
        pose_cv = poses_cv[src_i]
        stem = f"{out_i:06d}"
        filtered_depth = None
        if canonical_floor is not None:
            assert intrinsics is not None
            band = canonical_floor["vertical_band"]
            filtered_depth, filter_stats = filter_depth_to_canonical_floor(
                np.load(depth),
                pose_cv,
                intrinsics,
                lower_y=float(band["lower_y"]),
                upper_y=float(band["upper_y"]),
            )
            depth_filter_rows.append(filter_stats)
        if resize_width is not None and resize_height is not None:
            rgb_img = Image.open(rgb).convert("RGB")
            rgb_img = rgb_img.resize((resize_width, resize_height), Image.Resampling.LANCZOS)
            rgb_img.save(out / "rgb" / f"{stem}.png")
            depth_source = np.load(depth) if filtered_depth is None else filtered_depth
            depth_mm = np.clip(depth_source.astype(np.float64) * 1000.0, 0, 65535).astype(np.uint16)
            depth_img = Image.fromarray(depth_mm)
            depth_img = depth_img.resize((resize_width, resize_height), Image.Resampling.NEAREST)
            depth_img.save(out / "depth" / f"{stem}.png")
        else:
            link_or_copy(rgb, out / "rgb" / f"{stem}.png", mode=mode)
            if filtered_depth is None:
                save_depth_png_mm(np.load(depth), out / "depth" / f"{stem}.png")
            else:
                save_depth_png_mm(filtered_depth, out / "depth" / f"{stem}.png")
        # HM3DSemDataset returns pose_file @ C. Write pose_file = canonical @ C.
        pose_for_loader = pose_cv @ C_HM3DSEM
        np.savetxt(out / "pose" / f"{stem}.txt", pose_for_loader.reshape(1, 16), fmt="%.9f")

    write_reconstruction_config(
        cfg_path,
        scene_id=scene_id,
        data_view_root=output_root,
        graph_save_root=save_path,
        settings=requested_settings,
    )
    write_json(
        out / "adapter_manifest.json",
        {
            "adapter": "scene_package_to_fsrvln",
            "source_scene_package": str(root),
            "source_scene_id": source_scene_id,
            "effective_scene_id": scene_id,
            "output": str(out),
            "config": str(cfg_path),
            "link_mode": mode,
            "fsrvln_settings": requested_settings,
            "source_frame_count": len(rgb_paths),
            "selected_frame_count": len(selected_indices),
            "candidate_frame_count": len(candidate_indices),
            "source_selection": source_selection,
            "view_input_fingerprint": input_fingerprint,
            "invalidated_hmsg_checkpoints": invalidated_checkpoints,
            "rejected_frame_count": len(rejected),
            "skipped_by_yaw_count": skipped_by_yaw,
            "skipped_by_canonical_floor_count": skipped_by_canonical_floor,
            "skipped_by_canonical_floor_label_count": skipped_by_canonical_floor_label,
            "skipped_by_missing_floor_membership_count": skipped_by_missing_floor_membership,
            "canonical_floor": canonical_floor,
            "canonical_floor_depth_filter": (
                {
                    "frames": len(depth_filter_rows),
                    "valid_input_px": int(sum(row["valid_input_px"] for row in depth_filter_rows)),
                    "kept_px": int(sum(row["kept_px"] for row in depth_filter_rows)),
                    "removed_cross_level_px": int(sum(row["removed_cross_level_px"] for row in depth_filter_rows)),
                }
                if depth_filter_rows
                else None
            ),
            "hmsg_intrinsics": view_intrinsics,
            "selected_source_frame_indices": selected_indices,
            "rejected_frames": rejected[:200],
            "notes": "Compatible with FSR-VLN HM3DSemDataset; depth png is uint16 millimeters and intrinsics.json carries the calibrated camera matrix when available.",
        },
    )
    print(out)
    print(cfg_path)
    return out, cfg_path
