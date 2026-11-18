"""Render the FSRVLN reconstruction config from a preserved view manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write_reconstruction_config(
    path: Path,
    *,
    scene_id: str,
    data_view_root: Path,
    graph_save_root: Path,
    settings: dict[str, Any],
) -> None:
    """Write the graph-stage config without touching RGB-D views or checkpoints."""

    data_view_root = data_view_root.resolve()
    graph_save_root = graph_save_root.resolve()

    def value(name: str, default: Any) -> Any:
        return settings.get(name, default)

    def flag(name: str, default: bool) -> str:
        return str(bool(value(name, default))).lower()

    merge_rounds = value("max_hierarchical_merge_rounds", None)
    lines = [
        "main:",
        f"  device: {value('device', 'cuda')}",
        "  use_gpt: false",
        "  dataset: hm3dsem",
        f"  scene_id: {scene_id}",
        f"  dataset_path: {data_view_root / 'fsrvln'}",
        "  depth_cut: 10.0",
        f"  save_path: {graph_save_root}",
        "models:",
        "  clip:",
        "    type: ViT-L/14",
        "    checkpoint: checkpoints/open_clip_pytorch_model.bin",
        "  sam:",
        f"    checkpoint: {value('sam_checkpoint', 'checkpoints/sam_vit_h_4b8939.pth')}",
        f"    type: {value('sam_type', 'vit_h')}",
        f"    points_per_side: {value('sam_points_per_side', 24)}",
        f"    pred_iou_thresh: {value('sam_pred_iou_thresh', 0.88)}",
        f"    points_per_batch: {value('sam_points_per_batch', 144)}",
        f"    crop_n_layers: {value('sam_crop_n_layers', 0)}",
        f"    stability_score_thresh: {value('sam_stability_score_thresh', 0.92)}",
        f"    min_mask_region_area: {value('sam_min_mask_region_area', 150)}",
        "pipeline:",
        f"  voxel_size: {value('voxel_size', 0.05)}",
        f"  skip_frames: {value('skip_frames', 1)}",
        "  init_overlap_thresh: 0.75",
        "  overlap_thresh_factor: 0.025",
        "  iou_thresh: 0.05",
        "  clip_masked_weight: 0.4418",
        "  clip_bbox_margin: 50",
        "  feature_dbscan_eps: 0.01",
        f"  feature_dbscan_max_points: {value('feature_dbscan_max_points', 4096)}",
        "  max_mask_distance: 10000",
        f"  denoise_full_pcd: {flag('denoise_full_pcd', False)}",
        f"  radius_outlier_full_pcd: {flag('radius_outlier_full_pcd', False)}",
        f"  full_pcd_dbscan_eps: {value('full_pcd_dbscan_eps', 0.01)}",
        f"  full_pcd_dbscan_min_points: {value('full_pcd_dbscan_min_points', 100)}",
        f"  full_pcd_radius_nb_points: {value('full_pcd_radius_nb_points', 1000)}",
        f"  full_pcd_radius: {value('full_pcd_radius', 1.0)}",
        f"  min_pcd_points: {value('min_pcd_points', 50)}",
        f"  premerge_min_pcd_points: {value('premerge_min_pcd_points', 12)}",
        "  max_hierarchical_merge_rounds: null"
        if merge_rounds is None
        else f"  max_hierarchical_merge_rounds: {merge_rounds}",
        f"  resume_feature_checkpoint: {flag('resume_feature_checkpoint', True)}",
        f"  resume_merge_checkpoint: {flag('resume_merge_checkpoint', True)}",
        "  depth_weighting: false",
        f"  grid_resolution: {value('grid_resolution', 0.05)}",
        f"  room_wall_threshold_ratio: {value('room_wall_threshold_ratio', 0.18)}",
        f"  min_valid_floor_height_m: {value('min_valid_floor_height_m', 1.0)}",
        f"  merge_type: {value('merge_type', 'hierarchical')}",
        f"  save_intermediate_results: {flag('save_intermediate_results', True)}",
        f"  save_full_pcd_during_create: {flag('save_full_pcd_during_create', False)}",
        f"  save_masked_pcds: {flag('save_masked_pcds', False)}",
        f"  save_full_pcd: {flag('save_full_pcd', False)}",
        f"  save_full_pcd_feats: {flag('save_full_pcd_feats', False)}",
        f"  obj_labels: {value('obj_labels', 'MP3D_EQA_STRUCTURAL_LABELS')}",
        f"  merge_objects_graph: {flag('merge_objects_graph', True)}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
