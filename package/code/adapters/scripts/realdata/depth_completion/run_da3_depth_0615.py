from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
DA3_ROOT = REPO_ROOT / "Depth-Anything-3"
DA3_SRC = DA3_ROOT / "src"
DEFAULT_DATA_ROOT = Path.cwd()
if str(DA3_SRC) not in sys.path:
    sys.path.insert(0, str(DA3_SRC))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_indices(text: str | None, total: int) -> list[int]:
    if not text or text.lower() == "all":
        return list(range(total))
    out: list[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        out.append(int(chunk))
    for index in out:
        if index < 0 or index >= total:
            raise IndexError(f"index {index} out of range for {total} images")
    return out


def depth_to_color(depth: np.ndarray, *, percentile: float = 2.0) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    if int(valid.sum()) > 10:
        lo = float(np.percentile(depth[valid], percentile))
        hi = float(np.percentile(depth[valid], 100.0 - percentile))
    else:
        lo = float(np.nanmin(depth)) if depth.size else 0.0
        hi = float(np.nanmax(depth)) if depth.size else 1.0
    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi) or abs(hi - lo) < 1e-8:
        hi = lo + 1e-6
    norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    # near = warm, far = cool
    color = cv2.applyColorMap((255 * (1.0 - norm)).astype(np.uint8), cv2.COLORMAP_TURBO)
    color[~valid] = (0, 0, 0)
    return color


def scalar_flag(value: object) -> int:
    if isinstance(value, (bool, int, float, np.integer, np.floating)):
        return int(value)
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if isinstance(value, dict):
        for key in ("is_metric", "value", "flag"):
            if key in value:
                return scalar_flag(value[key])
        return 0
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def make_panel(rgb_bgr: np.ndarray, depth_color: np.ndarray, *, label: str) -> np.ndarray:
    target_h = min(rgb_bgr.shape[0], depth_color.shape[0])
    rgb = cv2.resize(rgb_bgr, (int(round(rgb_bgr.shape[1] * target_h / rgb_bgr.shape[0])), target_h))
    dep = cv2.resize(depth_color, (int(round(depth_color.shape[1] * target_h / depth_color.shape[0])), target_h))
    panel = np.hstack([rgb, dep])
    cv2.putText(panel, label, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.putText(panel, label, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def model_license(model_dir: Path) -> str:
    """License declared in the model card front matter (HF README.md), or '' when there is none."""
    card = model_dir / "README.md"
    if not card.is_file():
        return ""
    lines = card.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines or lines[0].strip() != "---":
        return ""
    for line in lines[1:40]:
        if line.strip() == "---":
            break
        if line.lower().startswith("license:"):
            return line.split(":", 1)[1].strip().strip("'\"")
    return ""


def is_metric_series(model_dir: Path) -> bool:
    """DA3METRIC-* checkpoints (monocular metric head). Their raw output is NOT in metres: the model card says
    metric_depth = focal_px * net_output / 300. DA3NESTED-* is already in metres and DA3-GIANT/LARGE are relative.
    The checkpoint's own config.json (model_name) decides; the directory name is only a fallback, so renaming
    the folder cannot silently disable the conversion."""
    name = ""
    config = model_dir / "config.json"
    if config.is_file():
        try:
            name = str(json.loads(config.read_text(encoding="utf-8")).get("model_name") or "")
        except (OSError, ValueError):
            name = ""
    name = (name or model_dir.name).lower()
    return "metric" in name and "nested" not in name


def metric_focal_px(K: np.ndarray, source_hw: tuple[int, int], processed_hw: tuple[int, int]) -> float:
    """Mean focal length in pixels of the *processed* (network input) image, from the source-resolution K."""
    (src_h, src_w), (proc_h, proc_w) = source_hw, processed_hw
    return 0.5 * (float(K[0, 0]) * proc_w / src_w + float(K[1, 1]) * proc_h / src_h)


def load_intrinsics(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    K = np.asarray(payload.get("K"), dtype=np.float32)
    return K if K.shape == (3, 3) else None


def run(args: argparse.Namespace) -> Path:
    from depth_anything_3.api import DepthAnything3

    image_dir = args.image_dir.resolve()
    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    )
    if not image_paths:
        raise RuntimeError(f"no images found in {image_dir}")
    selected_indices = parse_indices(args.indices, len(image_paths))
    if args.start > 0 or args.end >= 0 or args.stride > 1 or args.max_frames > 0:
        end = args.end if args.end >= 0 else len(selected_indices)
        selected_indices = selected_indices[args.start : end : max(1, args.stride)]
        if args.max_frames > 0:
            selected_indices = selected_indices[: args.max_frames]
    selected_paths = [image_paths[index] for index in selected_indices]

    out = args.output_dir.resolve()
    if out.exists() and args.force:
        import shutil

        shutil.rmtree(out)
    depth_dir = out / "depth_npy"
    depth_fullres_dir = out / "depth_npy_fullres"
    color_dir = out / "depth_vis"
    color_fullres_dir = out / "depth_vis_fullres"
    panel_dir = out / "panels"
    proc_dir = out / "processed_rgb"
    for directory in [depth_dir, depth_fullres_dir, color_dir, color_fullres_dir, panel_dir, proc_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    intrinsics = None
    if args.intrinsics_json:
        K = load_intrinsics(args.intrinsics_json.resolve())
        if K is not None:
            intrinsics = np.stack([K for _ in selected_paths], axis=0)

    device = args.device
    license_id = model_license(args.model_dir.resolve())
    if not license_id:
        print("[da3-depth] NOTE: no license found in the model card; check the checkpoint's license before commercial use.", flush=True)
    elif "nc" in license_id.lower().split("-"):
        print(f"[da3-depth] WARNING: checkpoint license is {license_id} (NON-COMMERCIAL). "
              "Use DA3METRIC-LARGE (apache-2.0) for commercial work.", flush=True)
    metric_mode = {"on": True, "off": False}.get(args.metric_focal_scale, is_metric_series(args.model_dir.resolve()))
    if metric_mode and intrinsics is None and args.metric_focal_px is None:
        raise RuntimeError(
            "This checkpoint needs focal length to produce metres (metric_depth = focal_px * net_output / 300). "
            "Pass --intrinsics-json with a valid K, or --metric-focal-px, or --metric-focal-scale off."
        )
    print(f"[da3-depth] loading model from {args.model_dir} on {device}; metric focal scaling: {metric_mode}", flush=True)
    model = DepthAnything3.from_pretrained(str(args.model_dir.resolve()))
    model = model.to(device=device)
    model.eval()

    rows = []
    batch_size = max(1, args.batch_size)
    print(
        f"[da3-depth] running inference on {len(selected_paths)} image(s), batch_size={batch_size}",
        flush=True,
    )
    for batch_start in range(0, len(selected_paths), batch_size):
        batch_paths = selected_paths[batch_start : batch_start + batch_size]
        batch_indices = selected_indices[batch_start : batch_start + batch_size]
        pending_paths: list[Path] = []
        pending_indices: list[int] = []
        for source_index, image_path in zip(batch_indices, batch_paths):
            stem = f"{source_index:06d}_{image_path.stem}"
            if (
                args.resume
                and (depth_dir / f"{stem}.npy").exists()
                and (depth_fullres_dir / f"{stem}.npy").exists()
                and (panel_dir / f"{stem}_panel.png").exists()
            ):
                rows.append(
                    {
                        "source_index": int(source_index),
                        "source_image": str(image_path),
                        "skipped_existing": True,
                        "depth_npy": str((depth_dir / f"{stem}.npy").relative_to(out)),
                        "depth_npy_fullres": str((depth_fullres_dir / f"{stem}.npy").relative_to(out)),
                        "depth_vis": str((color_dir / f"{stem}.png").relative_to(out)),
                        "depth_vis_fullres": str((color_fullres_dir / f"{stem}.png").relative_to(out)),
                        "processed_rgb": str((proc_dir / f"{stem}.png").relative_to(out)),
                        "panel": str((panel_dir / f"{stem}_panel.png").relative_to(out)),
                    }
                )
            else:
                pending_paths.append(image_path)
                pending_indices.append(source_index)
        if not pending_paths:
            print(
                f"[da3-depth] batch {batch_start // batch_size + 1}: all existing, skipped",
                flush=True,
            )
            continue

        batch_intrinsics = None
        if intrinsics is not None:
            # Intrinsics are identical for this rectified image directory, but keep this
            # indexed for future mixed-camera runs.
            index_to_pos = {idx: selected_indices.index(idx) for idx in pending_indices}
            batch_intrinsics = np.stack([intrinsics[index_to_pos[idx]] for idx in pending_indices], axis=0)

        print(
            f"[da3-depth] batch {batch_start // batch_size + 1}: {len(pending_paths)} image(s)",
            flush=True,
        )
        prediction = model.inference(
            [str(path) for path in pending_paths],
            intrinsics=batch_intrinsics,
            process_res=args.process_res,
            process_res_method=args.process_res_method,
            export_dir=None,
            export_format="mini_npz",
            use_ray_pose=args.use_ray_pose,
            ref_view_strategy=args.ref_view_strategy,
        )

        for local_idx, (source_index, image_path) in enumerate(zip(pending_indices, pending_paths)):
            depth = prediction.depth[local_idx].astype(np.float32)
            original_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if original_bgr is None:
                raise RuntimeError(f"failed to read source image: {image_path}")
            original_h, original_w = original_bgr.shape[:2]
            focal_px = None
            if metric_mode:
                if args.metric_focal_px is not None:
                    focal_px = float(args.metric_focal_px)
                else:
                    K_frame = intrinsics[selected_indices.index(source_index)]
                    focal_px = metric_focal_px(K_frame, (original_h, original_w), depth.shape[:2])
                depth = (depth * (focal_px / 300.0)).astype(np.float32)
            processed_rgb = prediction.processed_images[local_idx]
            processed_bgr = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)
            depth_color = depth_to_color(depth)

            stem = f"{source_index:06d}_{image_path.stem}"
            depth_path = depth_dir / f"{stem}.npy"
            depth_fullres_path = depth_fullres_dir / f"{stem}.npy"
            color_path = color_dir / f"{stem}.png"
            color_fullres_path = color_fullres_dir / f"{stem}.png"
            processed_path = proc_dir / f"{stem}.png"
            panel_path = panel_dir / f"{stem}_panel.png"
            depth_fullres = cv2.resize(depth, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
            np.save(depth_path, depth)
            np.save(depth_fullres_path, depth_fullres.astype(np.float32))
            cv2.imwrite(str(color_path), depth_color)
            cv2.imwrite(str(color_fullres_path), depth_to_color(depth_fullres))
            cv2.imwrite(str(processed_path), processed_bgr)
            cv2.imwrite(str(panel_path), make_panel(processed_bgr, depth_color, label=f"{image_path.name} depth"))
            rows.append(
                {
                    "source_index": int(source_index),
                    "source_image": str(image_path),
                    "source_shape": [int(original_h), int(original_w)],
                    "processed_shape": [int(v) for v in depth.shape],
                    "is_metric": scalar_flag(prediction.is_metric),
                    "metric_focal_scaled": bool(metric_mode),
                    "focal_px_processed": focal_px,
                    "depth_npy": str(depth_path.relative_to(out)),
                    "depth_npy_fullres": str(depth_fullres_path.relative_to(out)),
                    "depth_vis": str(color_path.relative_to(out)),
                    "depth_vis_fullres": str(color_fullres_path.relative_to(out)),
                    "processed_rgb": str(processed_path.relative_to(out)),
                    "panel": str(panel_path.relative_to(out)),
                    "depth_min": float(np.nanmin(depth)),
                    "depth_p50": float(np.nanpercentile(depth, 50)),
                    "depth_max": float(np.nanmax(depth)),
                }
            )

    rows.sort(key=lambda item: item["source_index"])
    num_skipped = sum(1 for row in rows if row.get("skipped_existing"))
    num_written = len(rows) - num_skipped

    if args.make_montage:
        panels = [cv2.imread(str(out / row["panel"])) for row in rows]
        panels = [image for image in panels if image is not None]
        if panels:
            width = min(image.shape[1] for image in panels)
            resized = [cv2.resize(image, (width, int(round(image.shape[0] * width / image.shape[1])))) for image in panels]
            montage = np.vstack(resized)
            cv2.imwrite(str(out / "depth_panels_montage.png"), montage)

    payload = {
        "model_dir": str(args.model_dir.resolve()),
        "image_dir": str(image_dir),
        "process_res": args.process_res,
        "process_res_method": args.process_res_method,
        "batch_size": batch_size,
        "num_available": len(image_paths),
        "num_selected": len(selected_paths),
        "num_written": num_written,
        "num_skipped": num_skipped,
        "use_ray_pose": bool(args.use_ray_pose),
        "ref_view_strategy": args.ref_view_strategy,
        "intrinsics_json": str(args.intrinsics_json.resolve()) if args.intrinsics_json else None,
        "is_metric": scalar_flag(rows[0].get("is_metric", 0)) if rows else 0,
        "results": rows,
        "model_license": license_id or None,
        "metric_focal_scaling": bool(metric_mode),
        "note": (
            "Depth multiplied by focal_px/300 (DA3METRIC model-card formula). This puts values near metres but is NOT "
            "reliable on its own (median abs-rel 10-19% on simulator ground truth); always fit to LiDAR "
            "(calibrate_da3_depth_with_lidar.py), which brings it to ~3%."
            if metric_mode
            else "Relative depth. Use LiDAR projection to calibrate metric scale if meters are required."
        ),
    }
    write_json(out / "depth_results.json", payload)
    lines = [
        "# DA3 Depth Smoke",
        "",
        f"- model: `{args.model_dir.resolve()}`",
        f"- images: `{image_dir}`",
        f"- output: `{out}`",
        f"- requested images: `{len(selected_paths)}`",
        f"- completed records: `{len(rows)}`",
        f"- batch_size: `{batch_size}`",
        f"- is_metric: `{int(rows[0].get('is_metric', 0)) if rows else 0}`",
        "",
        "Depth arrays are stored in `depth_npy/`. Visualizations are stored in `depth_vis/` and `panels/`.",
        "",
    ]
    if (out / "depth_panels_montage.png").exists():
        lines += ["![depth montage](depth_panels_montage.png)", ""]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(out), "frames": len(rows), "results": str(out / "depth_results.json")}, indent=2))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Depth Anything 3 on 0615 extracted RGB frames and save depth maps.")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_DATA_ROOT
        / "derived"
        / "realdata_0615_stage0"
        / "real_0615_cam0_lidar1_sync_stage0"
        / "rgb",
    )
    parser.add_argument(
        "--intrinsics-json",
        type=Path,
        default=DEFAULT_DATA_ROOT
        / "derived"
        / "realdata_0615_stage0"
        / "real_0615_cam0_lidar1_sync_stage0"
        / "intrinsics.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATA_ROOT / "derived" / "realdata_0615_da3_depth" / "cam0_front_lidar1_sync_da3",
    )
    parser.add_argument("--model-dir", type=Path, default=Path("Depth-Anything-3/DA3METRIC-LARGE"),
                        help="Default is DA3METRIC-LARGE (Apache-2.0). DA3-GIANT-1.1 is CC BY-NC (non-commercial).")
    parser.add_argument("--metric-focal-scale", choices=("auto", "on", "off"), default="auto",
                        help="Apply metric_depth = focal_px * net_output / 300. auto: on for DA3METRIC-* checkpoints.")
    parser.add_argument("--metric-focal-px", type=float, default=None,
                        help="Override the focal length (pixels, at the processed resolution) used for --metric-focal-scale.")
    parser.add_argument("--indices", default="0,20,40")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--process-res-method", default="upper_bound_resize")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-ray-pose", action="store_true")
    parser.add_argument("--ref-view-strategy", default="middle")
    parser.add_argument("--make-montage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
