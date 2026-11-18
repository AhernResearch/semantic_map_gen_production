#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    _RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
    _RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # Pillow < 9
    _RESAMPLE_BILINEAR = Image.BILINEAR
    _RESAMPLE_LANCZOS = Image.LANCZOS


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common_scene import (  # noqa: E402
    ensure_empty_dir,
    make_contact_sheet,
    natural_key,
    read_json,
    sorted_files,
    write_json,
    write_poses_txt,
)


CV_TO_HABITAT = np.diag([1.0, -1.0, -1.0, 1.0])


def load_poses(path: Path) -> np.ndarray:
    rows = np.loadtxt(path, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[None, :]
    return rows.reshape((-1, 4, 4))


def load_intrinsics(path: Path) -> dict[str, Any]:
    data = read_json(path)
    data["width"] = int(data["width"])
    data["height"] = int(data["height"])
    data["K"] = np.asarray(data["K"], dtype=np.float64)
    return data


def scale_intrinsics(intr: dict[str, Any], scale: float) -> dict[str, Any]:
    out = dict(intr)
    out["width"] = int(round(int(intr["width"]) * scale))
    out["height"] = int(round(int(intr["height"]) * scale))
    k = np.asarray(intr["K"], dtype=np.float64).copy()
    k[0, :] *= scale
    k[1, :] *= scale
    k[2, :] = [0.0, 0.0, 1.0]
    out["K"] = k
    out["flattened"] = k.reshape(-1).tolist()
    out["camera_frame"] = "opencv_vlmaps_x_right_y_down_z_forward"
    fx = float(k[0, 0])
    out["hfov_deg"] = float(math.degrees(2.0 * math.atan(out["width"] / (2.0 * fx))))
    return out


def resize_depth(depth: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(depth.astype(np.float32), mode="F")
    return np.asarray(image.resize(size, resample=_RESAMPLE_BILINEAR), dtype=np.float32)


def copy_or_resize_rgb(src: Path, dst: Path, size: tuple[int, int]) -> None:
    with Image.open(src) as image:
        rgb = image.convert("RGB")
        if rgb.size != size:
            rgb = rgb.resize(size, resample=_RESAMPLE_LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        rgb.save(dst, format="PNG")


def metric_depth_paths(metric_depth_dir: Path) -> list[Path]:
    paths = sorted(metric_depth_dir.glob("*.npy"), key=natural_key)
    if not paths:
        raise FileNotFoundError(f"No metric depth .npy files found in {metric_depth_dir}")
    return paths


def load_completed_points(completion_root: Path, frame_indices: list[int], *, max_points_per_frame: int) -> tuple[np.ndarray, np.ndarray]:
    points_chunks: list[np.ndarray] = []
    color_chunks: list[np.ndarray] = []
    rng = np.random.default_rng(20260616)
    for frame_idx in frame_indices:
        path = completion_root / "frames" / f"{frame_idx:06d}" / "completed_dense_world_points_yup.npz"
        if not path.exists():
            continue
        data = np.load(path)
        points = np.asarray(data["points_world"], dtype=np.float32)
        colors = np.asarray(data["colors_rgb"], dtype=np.uint8)
        valid = np.all(np.isfinite(points), axis=1)
        points = points[valid]
        colors = colors[valid]
        if max_points_per_frame > 0 and points.shape[0] > max_points_per_frame:
            keep = rng.choice(points.shape[0], size=max_points_per_frame, replace=False)
            points = points[keep]
            colors = colors[keep]
        points_chunks.append(points)
        color_chunks.append(colors)
    if not points_chunks:
        raise FileNotFoundError(f"No completed dense world point files found under {completion_root}/frames")
    return np.concatenate(points_chunks, axis=0), np.concatenate(color_chunks, axis=0)


def load_slam_map_points(slam_package: Path) -> np.ndarray:
    path = slam_package / "lidar_map.npz"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find SLAM global map points: {path}")
    data = np.load(path)
    key = "points" if "points" in data else "xyz" if "xyz" in data else data.files[0]
    points = np.asarray(data[key], dtype=np.float32)[:, :3]
    return points[np.all(np.isfinite(points), axis=1)]


def robust_bounds(points: np.ndarray, poses: np.ndarray, margin: float) -> tuple[np.ndarray, np.ndarray]:
    q_min = np.percentile(points[:, :3], 0.5, axis=0)
    q_max = np.percentile(points[:, :3], 99.5, axis=0)
    camera_xyz = poses[:, :3, 3]
    q_min = np.minimum(q_min, camera_xyz.min(axis=0))
    q_max = np.maximum(q_max, camera_xyz.max(axis=0))
    q_min[[0, 2]] -= margin
    q_max[[0, 2]] += margin
    return q_min.astype(np.float64), q_max.astype(np.float64)


def splat_max_height(
    topdown: np.ndarray,
    top_y: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    y: np.ndarray,
    colors: np.ndarray,
) -> None:
    width = topdown.shape[1]
    linear = rows * width + cols
    order = np.argsort(y)
    flat_rgb = topdown.reshape((-1, 3))
    flat_y = top_y.reshape(-1)
    for idx in order:
        lin = int(linear[idx])
        yy = float(y[idx])
        if yy >= flat_y[lin]:
            flat_y[lin] = yy
            flat_rgb[lin] = colors[idx]


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    if radius <= 0:
        return mask
    out = mask.copy()
    h, w = mask.shape
    source = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            sy0 = max(0, -dy)
            sy1 = min(h, h - dy)
            dy0 = max(0, dy)
            dy1 = min(h, h + dy)
            sx0 = max(0, -dx)
            sx1 = min(w, w - dx)
            dx0 = max(0, dx)
            dx1 = min(w, w + dx)
            out[dy0:dy1, dx0:dx1] |= source[sy0:sy1, sx0:sx1]
    return out


def build_bev_maps(
    root: Path,
    scene_id: str,
    points: np.ndarray,
    colors: np.ndarray | None,
    poses: np.ndarray,
    *,
    bev_source: str,
    cell_size: float,
    margin: float,
    min_height_above_floor: float,
    max_height_above_floor: float,
    path_radius_m: float,
) -> dict[str, Any]:
    maps = root / "maps"
    maps.mkdir(parents=True, exist_ok=True)
    finite = np.all(np.isfinite(points), axis=1)
    points = points[finite]
    if colors is not None:
        colors = colors[finite]
    mn, mx = robust_bounds(points, poses, margin)
    height = int(math.ceil((mx[2] - mn[2]) / cell_size)) + 1
    width = int(math.ceil((mx[0] - mn[0]) / cell_size)) + 1
    shape = (height, width)

    floor_y = float(np.percentile(points[:, 1], 2))
    rows = np.floor((points[:, 2] - mn[2]) / cell_size).astype(np.int64)
    cols = np.floor((points[:, 0] - mn[0]) / cell_size).astype(np.int64)
    inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    height_band = (
        (points[:, 1] >= floor_y + min_height_above_floor)
        & (points[:, 1] <= floor_y + max_height_above_floor)
    )
    keep = inside & height_band
    rows_k = rows[keep]
    cols_k = cols[keep]
    y_k = points[keep, 1]

    topdown = np.full((height, width, 3), 244, dtype=np.uint8)
    top_y = np.full(shape, -np.inf, dtype=np.float32)
    if rows_k.size:
        if colors is not None:
            splat_max_height(topdown, top_y, rows_k, cols_k, y_k, colors[keep])
        else:
            density = np.zeros(shape, dtype=np.uint16)
            np.add.at(density, (rows_k, cols_k), 1)
            observed_density = density > 0
            tone = (255 - np.clip(np.log1p(density) / np.log(12), 0.0, 1.0) * 215).astype(np.uint8)
            topdown[observed_density] = np.stack(
                [tone[observed_density], tone[observed_density], tone[observed_density]],
                axis=1,
            )
            top_y[observed_density] = floor_y + max_height_above_floor
    observed = np.isfinite(top_y)
    occupancy = dilate_mask(observed, radius=1).astype(np.uint8) * 255

    free = np.zeros(shape, dtype=np.uint8)
    camera_xyz = poses[:, :3, 3]
    cam_rows = np.floor((camera_xyz[:, 2] - mn[2]) / cell_size).astype(np.int64)
    cam_cols = np.floor((camera_xyz[:, 0] - mn[0]) / cell_size).astype(np.int64)
    radius_cells = max(1, int(round(path_radius_m / cell_size)))
    for row, col in zip(cam_rows, cam_cols):
        if row < 0 or row >= height or col < 0 or col >= width:
            continue
        r0 = max(0, row - radius_cells)
        r1 = min(height, row + radius_cells + 1)
        c0 = max(0, col - radius_cells)
        c1 = min(width, col + radius_cells + 1)
        free[r0:r1, c0:c1] = 255

    Image.fromarray(topdown).save(maps / "top_down_rgb.png")
    Image.fromarray(occupancy).save(maps / "occupancy.png")
    Image.fromarray(free).save(maps / "free_space.png")
    np.save(maps / "observed_rgb_mask.npy", observed)
    np.save(maps / "free_space.npy", free.astype(bool))
    coord = {
        "scene_id": scene_id,
        "world_frame": "real/SLAM world, y-up by convention",
        "camera_frame": "opencv_vlmaps_x_right_y_down_z_forward",
        "camera_pose_file": "observations/poses.txt",
        "cell_size_m": cell_size,
        "meters_per_pixel": cell_size,
        "floor_y": floor_y,
        "floor_source": f"p02({bev_source}.world_y)",
        "bev_shape": [height, width],
        "image_size": {"width": width, "height": height},
        "world_bounds": {
            "x_min": float(mn[0]),
            "x_max": float(mx[0]),
            "y_min": float(mn[1]),
            "y_max": float(mx[1]),
            "z_min": float(mn[2]),
            "z_max": float(mx[2]),
        },
        "world_to_bev": "u=(x-x_min)/cell_size_m; v=(z-z_min)/cell_size_m",
        "bev_to_world": "x=x_min+u*cell_size_m; z=z_min+v*cell_size_m; y defaults to floor_y",
        "image_coordinates": "u is image column, v is image row, origin is top-left",
        "bev_source": bev_source,
        "observed_fraction": float(np.mean(observed)),
    }
    write_json(maps / "coordinate_meta.json", coord)
    return coord


def write_readme(root: Path, scene_id: str, args: argparse.Namespace) -> None:
    lines = [
        f"# RealData Dense Scene Package: {scene_id}",
        "",
        "This scene package adapts the 2026-06-15 real robot recording into the MapFusion scene-package contract.",
        "",
        "## Inputs",
        "",
        f"- RGB/stage0: `{args.stage_dir}`",
        f"- SLAM aligned poses: `{args.slam_package}`",
        f"- Metric dense depth: `{args.metric_depth_dir}`",
        f"- Dense completion cloud: `{args.completion_root}`",
        "",
        "## Contract",
        "",
        "- `observations/rgb/*.png`: rectified RGB frames.",
        "- `observations/depth/*.npy`: dense metric depth maps in meters.",
        "- `observations/poses.txt`: camera-to-world OpenCV-frame poses in the SLAM y-up world.",
        "- `maps/top_down_rgb.png`: SLAM global PCD BEV by default; this is the structural map for inspection.",
        "- `annotations/semantic_object_gt.json`: empty placeholder for real data; object labels must be inferred by the map methods.",
        "",
        "The BEV is a debugging view, not a hand-labeled ground truth map. The actual map methods consume the RGB/depth/pose observations.",
        "Dense monocular depth is intentionally not used as the default structural BEV source because small depth errors create view-ray artifacts.",
        "",
    ]
    (root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def build_package(args: argparse.Namespace) -> Path:
    scene_id = args.scene_id
    output = args.output_root / scene_id
    ensure_empty_dir(output, force=args.force)
    for rel in [
        "observations/rgb",
        "observations/depth",
        "observations/semantic",
        "maps",
        "annotations",
        "visualizations",
        "realdata",
    ]:
        (output / rel).mkdir(parents=True, exist_ok=True)

    intr_src = load_intrinsics(args.stage_dir / "intrinsics.json")
    scale = float(args.image_scale)
    if scale <= 0:
        raise ValueError("--image-scale must be positive")
    intr = scale_intrinsics(intr_src, scale)
    width = int(intr["width"])
    height = int(intr["height"])
    rgb_paths = sorted_files(args.stage_dir / "rgb", "*.png")
    poses = load_poses(args.slam_package / "poses.txt")
    base_pose_path = args.slam_package / "poses_world_base.txt"
    base_poses = load_poses(base_pose_path) if base_pose_path.exists() else poses
    depth_paths = metric_depth_paths(args.metric_depth_dir)
    frame_count = min(len(rgb_paths), len(poses), len(base_poses), len(depth_paths))
    if args.max_frames > 0:
        frame_count = min(frame_count, args.max_frames)
    if frame_count <= 0:
        raise RuntimeError("No aligned RGB/depth/pose frames available.")

    frame_indices = list(range(frame_count))
    timestamps = []
    trajectory = []
    valid_ratios = []
    for out_idx, src_idx in enumerate(frame_indices):
        stem = f"{out_idx:06d}"
        copy_or_resize_rgb(rgb_paths[src_idx], output / "observations/rgb" / f"{stem}.png", (width, height))
        depth = np.load(depth_paths[src_idx]).astype(np.float32)
        if depth.shape != (height, width):
            depth = resize_depth(depth, (width, height))
        valid = np.isfinite(depth) & (depth > args.depth_min) & (depth < args.depth_max)
        valid_ratios.append(float(np.mean(valid)))
        np.save(output / "observations/depth" / f"{stem}.npy", depth.astype(np.float32))
        np.save(output / "observations/semantic" / f"{stem}.npy", np.zeros((height, width), dtype=np.int32))
        timestamps.append({"frame_index": out_idx, "source_frame_index": src_idx})
        trajectory.append(
            {
                "frame_index": out_idx,
                "source_frame_index": src_idx,
                "source_base_index": src_idx,
                "camera_position": poses[src_idx, :3, 3].tolist(),
                "nav_position": base_poses[src_idx, :3, 3].tolist(),
                "quality": {
                    "valid_depth_ratio": valid_ratios[-1],
                    "median_valid_depth_m": float(np.median(depth[valid])) if np.any(valid) else 0.0,
                },
            }
        )

    poses_out = poses[frame_indices]
    write_poses_txt(output / "observations/poses.txt", poses_out)
    write_poses_txt(output / "observations/poses_vlmaps.txt", poses_out)
    write_poses_txt(output / "observations/poses_habitat_camera.txt", poses_out @ CV_TO_HABITAT)
    intr_json = dict(intr)
    intr_json["K"] = np.asarray(intr_json["K"], dtype=np.float64).tolist()
    write_json(output / "observations/intrinsics.json", intr_json)
    write_json(output / "observations/timestamps.json", timestamps)
    write_json(output / "observations/trajectory.json", trajectory)

    if args.bev_source == "slam_map":
        points = load_slam_map_points(args.slam_package)
        colors = None
    elif args.bev_source == "completed_depth":
        points, colors = load_completed_points(
            args.completion_root,
            frame_indices,
            max_points_per_frame=args.max_bev_points_per_frame,
        )
    else:
        raise ValueError(f"Unknown --bev-source: {args.bev_source}")
    coord = build_bev_maps(
        output,
        scene_id,
        points,
        colors,
        poses_out,
        bev_source=args.bev_source,
        cell_size=args.cell_size,
        margin=args.map_margin,
        min_height_above_floor=args.bev_min_height_above_floor,
        max_height_above_floor=args.bev_max_height_above_floor,
        path_radius_m=args.path_radius_m,
    )
    write_json(
        output / "annotations/semantic_object_gt.json",
        {
            "scene_id": scene_id,
            "objects": [],
            "note": "Real data has no manually supplied object GT here. Map methods should infer object labels from RGB/depth.",
        },
    )
    summary = read_json(args.completion_root / "completion_summary.json") if (args.completion_root / "completion_summary.json").exists() else {}
    write_json(
        output / "package_meta.json",
        {
            "scene_id": scene_id,
            "dataset": "REALDATA_0615",
            "source_scene": str(args.stage_dir.resolve()),
            "selected_scene": scene_id,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "image_scale": scale,
            "depth_source": "depth_anything3_metric_depth_lidarfit",
            "bev_source": args.bev_source,
            "pose_source": str((args.slam_package / "poses.txt").resolve()),
            "completion_root": str(args.completion_root.resolve()),
            "metric_depth_dir": str(args.metric_depth_dir.resolve()),
            "depth_valid_ratio_mean": float(np.mean(valid_ratios)) if valid_ratios else 0.0,
            "depth_completion_summary": {
                "frame_count": summary.get("frame_count"),
                "point_stride": summary.get("point_stride"),
                "completed_point_count_total": summary.get("completed_point_count_total"),
                "metric_depth_valid_ratio_median": summary.get("metric_depth_valid_ratio_median"),
            },
            "coordinate_meta": coord,
        },
    )
    write_json(
        output / "realdata/adapter_manifest.json",
        {
            "stage_dir": str(args.stage_dir.resolve()),
            "slam_package": str(args.slam_package.resolve()),
            "metric_depth_dir": str(args.metric_depth_dir.resolve()),
            "completion_root": str(args.completion_root.resolve()),
            "scene_id": scene_id,
            "image_scale": scale,
            "frame_count": frame_count,
            "mean_valid_depth_ratio": float(np.mean(valid_ratios)) if valid_ratios else 0.0,
            "bev_source": args.bev_source,
            "bev_point_count_used": int(points.shape[0]),
        },
    )
    make_contact_sheet(sorted_files(output / "observations/rgb", "*.png")[:20], output / "visualizations/rgb_contact_sheet.png", cols=5)
    shutil.copy2(output / "maps/top_down_rgb.png", output / "visualizations/topdown_rgb.png")
    write_readme(output, scene_id, args)
    print(
        json.dumps(
            {
                "scene_package": str(output),
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "mean_valid_depth_ratio": float(np.mean(valid_ratios)) if valid_ratios else 0.0,
                "bev_observed_fraction": coord["observed_fraction"],
            },
            indent=2,
        )
    )
    return output


def parse_args() -> argparse.Namespace:
    data_root = Path.cwd()
    parser = argparse.ArgumentParser(
        description="Build a MapFusion scene package from real 0615 RGB, SLAM y-up poses, DA3+LiDAR metric depth, and a SLAM PCD BEV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage-dir",
        type=Path,
        default=data_root / "derived/realdata_0615_stage0/real_0615_cam0_lidar1_sync_stage0",
    )
    parser.add_argument(
        "--slam-package",
        type=Path,
        default=data_root / "derived/realdata_0615_slam_aligned/real_0615_cam0_lidar1_sync_slam_yup_axisfix",
    )
    parser.add_argument(
        "--metric-depth-dir",
        type=Path,
        default=data_root / "derived/realdata_0615_da3_depth/cam0_front_lidar1_sync_metric_lidarfit/metric_depth_npy_fullres",
    )
    parser.add_argument(
        "--completion-root",
        type=Path,
        default=data_root
        / "derived/realdata_0615_depth_completion_slam_world/cam0_front_lidar1_sync_axisfix_da3_lidarfit_slam_yup_stride8",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=data_root / "derived/realdata_scene_packages_completed_slam",
    )
    parser.add_argument("--scene-id", default="real_0615_cam0_lidar1_sync_dense_slam_yup_axisfix")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--image-scale", type=float, default=0.5, help="Downscale RGB/depth/intrinsics for practical map-method runs.")
    parser.add_argument("--depth-min", type=float, default=0.05)
    parser.add_argument("--depth-max", type=float, default=12.0)
    parser.add_argument("--bev-source", choices=["slam_map", "completed_depth"], default="slam_map")
    parser.add_argument("--cell-size", type=float, default=0.04)
    parser.add_argument("--map-margin", type=float, default=0.7)
    parser.add_argument("--bev-min-height-above-floor", type=float, default=-0.1)
    parser.add_argument("--bev-max-height-above-floor", type=float, default=4.5)
    parser.add_argument("--path-radius-m", type=float, default=0.35)
    parser.add_argument("--max-bev-points-per-frame", type=int, default=75000)
    return parser.parse_args()


def main() -> None:
    build_package(parse_args())


if __name__ == "__main__":
    main()
