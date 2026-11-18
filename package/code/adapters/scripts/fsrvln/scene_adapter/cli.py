"""Command-line interface for the FSRVLN scene-view adapter."""

from __future__ import annotations

import argparse
from pathlib import Path

from adapters.scripts.common_scene import DEFAULT_CONFIGS, DEFAULT_DATA_VIEWS

from .conversion import convert

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an FSR-VLN HM3DSem-compatible data view.")
    parser.add_argument("--scene-package", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_DATA_VIEWS)
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIGS)
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--skip-frames", type=int, default=1)
    parser.add_argument("--min-pcd-points", type=int, default=50)
    parser.add_argument("--premerge-min-pcd-points", type=int, default=12)
    parser.add_argument("--merge-type", choices=["sequential", "hierarchical"], default="sequential")
    parser.add_argument("--max-hierarchical-merge-rounds", type=int, default=None)
    parser.add_argument("--resume-feature-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-merge-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--grid-resolution", type=float, default=0.05)
    parser.add_argument("--room-wall-threshold-ratio", type=float, default=0.25)
    parser.add_argument("--min-valid-floor-height-m", type=float, default=1.0)
    parser.add_argument("--max-black-ratio", type=float, default=None)
    parser.add_argument("--min-valid-depth-ratio", type=float, default=None)
    parser.add_argument("--save-root", type=Path, default=None)
    parser.add_argument("--resize-width", type=int, default=None)
    parser.add_argument("--resize-height", type=int, default=None)
    parser.add_argument("--obj-labels", default="MP3D_EQA_STRUCTURAL_LABELS")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--source-frame-stride", type=int, default=1)
    parser.add_argument(
        "--target-source-frames-per-m2",
        type=float,
        default=None,
        help="For canonical floors, uniformly cap post-stride source views according to observed floor footprint.",
    )
    parser.add_argument("--min-source-frames", type=int, default=None)
    parser.add_argument("--max-source-frames", type=int, default=None)
    parser.add_argument("--source-yaw-offsets", nargs="*", type=float, default=None)
    parser.add_argument("--source-yaw-tolerance-deg", type=float, default=1e-3)
    parser.add_argument("--merge-objects-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sam-type", default="vit_h")
    parser.add_argument("--sam-checkpoint", default="checkpoints/sam_vit_h_4b8939.pth")
    parser.add_argument("--sam-points-per-side", type=int, default=24)
    parser.add_argument("--sam-points-per-batch", type=int, default=144)
    parser.add_argument("--sam-pred-iou-thresh", type=float, default=0.88)
    parser.add_argument("--sam-crop-n-layers", type=int, default=0)
    parser.add_argument("--sam-stability-score-thresh", type=float, default=0.92)
    parser.add_argument("--sam-min-mask-region-area", type=int, default=150)
    parser.add_argument("--denoise-full-pcd", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--radius-outlier-full-pcd", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--full-pcd-dbscan-eps", type=float, default=0.01)
    parser.add_argument("--full-pcd-dbscan-min-points", type=int, default=100)
    parser.add_argument("--full-pcd-radius-nb-points", type=int, default=1000)
    parser.add_argument("--full-pcd-radius", type=float, default=1.0)
    parser.add_argument("--save-intermediate-results", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-full-pcd-during-create", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-masked-pcds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-full-pcd", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-full-pcd-feats", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--canonical-floor-id",
        type=int,
        default=None,
        help="Restrict this FSRVLN run to one canonical scene-package floor.",
    )
    parser.add_argument("--canonical-floor-camera-height-m", type=float, default=1.5)
    parser.add_argument("--canonical-floor-camera-y-tolerance-m", type=float, default=0.85)
    parser.add_argument("--canonical-floor-label-tolerance-m", type=float, default=0.05)
    parser.add_argument("--canonical-floor-lower-margin-m", type=float, default=0.35)
    parser.add_argument("--canonical-floor-gap-to-next-m", type=float, default=0.35)
    parser.add_argument("--canonical-floor-default-height-m", type=float, default=3.0)
    parser.add_argument("--canonical-floor-min-area-m2", type=float, default=10.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert(
        args.scene_package,
        args.output_root,
        args.config_root,
        args.mode,
        args.force,
        skip_frames=args.skip_frames,
        min_pcd_points=args.min_pcd_points,
        premerge_min_pcd_points=args.premerge_min_pcd_points,
        merge_type=args.merge_type,
        max_hierarchical_merge_rounds=args.max_hierarchical_merge_rounds,
        resume_feature_checkpoint=args.resume_feature_checkpoint,
        resume_merge_checkpoint=args.resume_merge_checkpoint,
        voxel_size=args.voxel_size,
        grid_resolution=args.grid_resolution,
        room_wall_threshold_ratio=args.room_wall_threshold_ratio,
        min_valid_floor_height_m=args.min_valid_floor_height_m,
        max_black_ratio=args.max_black_ratio,
        min_valid_depth_ratio=args.min_valid_depth_ratio,
        save_root=args.save_root,
        resize_width=args.resize_width,
        resize_height=args.resize_height,
        obj_labels=args.obj_labels,
        device=args.device,
        source_frame_stride=args.source_frame_stride,
        target_source_frames_per_m2=args.target_source_frames_per_m2,
        min_source_frames=args.min_source_frames,
        max_source_frames=args.max_source_frames,
        source_yaw_offsets=tuple(args.source_yaw_offsets or ()),
        source_yaw_tolerance_deg=args.source_yaw_tolerance_deg,
        merge_objects_graph=args.merge_objects_graph,
        sam_type=args.sam_type,
        sam_checkpoint=args.sam_checkpoint,
        sam_points_per_side=args.sam_points_per_side,
        sam_points_per_batch=args.sam_points_per_batch,
        sam_pred_iou_thresh=args.sam_pred_iou_thresh,
        sam_crop_n_layers=args.sam_crop_n_layers,
        sam_stability_score_thresh=args.sam_stability_score_thresh,
        sam_min_mask_region_area=args.sam_min_mask_region_area,
        denoise_full_pcd=args.denoise_full_pcd,
        radius_outlier_full_pcd=args.radius_outlier_full_pcd,
        full_pcd_dbscan_eps=args.full_pcd_dbscan_eps,
        full_pcd_dbscan_min_points=args.full_pcd_dbscan_min_points,
        full_pcd_radius_nb_points=args.full_pcd_radius_nb_points,
        full_pcd_radius=args.full_pcd_radius,
        save_intermediate_results=args.save_intermediate_results,
        save_full_pcd_during_create=args.save_full_pcd_during_create,
        save_masked_pcds=args.save_masked_pcds,
        save_full_pcd=args.save_full_pcd,
        save_full_pcd_feats=args.save_full_pcd_feats,
        canonical_floor_id=args.canonical_floor_id,
        canonical_floor_camera_height_m=args.canonical_floor_camera_height_m,
        canonical_floor_camera_y_tolerance_m=args.canonical_floor_camera_y_tolerance_m,
        canonical_floor_label_tolerance_m=args.canonical_floor_label_tolerance_m,
        canonical_floor_lower_margin_m=args.canonical_floor_lower_margin_m,
        canonical_floor_gap_to_next_m=args.canonical_floor_gap_to_next_m,
        canonical_floor_default_height_m=args.canonical_floor_default_height_m,
        canonical_floor_min_area_m2=args.canonical_floor_min_area_m2,
    )
