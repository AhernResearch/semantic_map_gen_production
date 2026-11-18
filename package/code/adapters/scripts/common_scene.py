from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS = REPO_ROOT / "artifacts" / "current"
DEFAULT_SCENE_PACKAGES = DEFAULT_ARTIFACTS / "intermediates" / "scene_packages"
DEFAULT_DATA_VIEWS = DEFAULT_ARTIFACTS / "intermediates" / "data_views"
DEFAULT_RESULTS = DEFAULT_ARTIFACTS / "intermediates" / "method_outputs"
DEFAULT_CONFIGS = REPO_ROOT / "adapters" / "configs"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def write_json(path: Path, data: Any) -> None:
    """Atomically publish JSON so an interrupted run cannot leave a stub."""

    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        data,
        indent=2,
        ensure_ascii=False,
        default=_json_default,
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(serialized)
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def natural_key(path: Path) -> tuple[Any, ...]:
    out: list[Any] = []
    for part in path.stem.replace("_", "-").split("-"):
        out.append(int(part) if part.isdigit() else part)
    return tuple(out)


def sorted_files(path: Path, pattern: str) -> list[Path]:
    return sorted(path.glob(pattern), key=natural_key)


def ensure_empty_dir(path: Path, force: bool = False) -> None:
    if path.exists() and force:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, mode: str = "symlink") -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(src.resolve(), dst)
    else:
        raise ValueError(f"Unknown link mode: {mode}")


def load_scene_package(scene_package: Path) -> dict[str, Any]:
    scene_package = scene_package.resolve()
    meta = read_json(scene_package / "package_meta.json")
    coord = read_json(scene_package / "maps" / "coordinate_meta.json")
    intr = read_json(scene_package / "observations" / "intrinsics.json")
    return {
        "root": scene_package,
        "meta": meta,
        "coord": coord,
        "intrinsics": intr,
    }


def world_to_bev(point: Iterable[float], coord: dict[str, Any]) -> list[float]:
    x, _, z = [float(v) for v in point]
    return [
        (x - float(coord["world_bounds"]["x_min"])) / float(coord["cell_size_m"]),
        (z - float(coord["world_bounds"]["z_min"])) / float(coord["cell_size_m"]),
    ]


def bev_to_world(pixel: Iterable[float], coord: dict[str, Any], y: float | None = None) -> list[float]:
    u, v = [float(x) for x in pixel]
    y_val = float(coord.get("floor_y", 0.0) if y is None else y)
    return [
        float(coord["world_bounds"]["x_min"]) + u * float(coord["cell_size_m"]),
        y_val,
        float(coord["world_bounds"]["z_min"]) + v * float(coord["cell_size_m"]),
    ]


def camera_intrinsics_from_hfov(width: int, height: int, hfov_deg: float) -> np.ndarray:
    hfov = math.radians(float(hfov_deg))
    fx = width / (2.0 * math.tan(hfov / 2.0))
    fy = fx
    return np.array([[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def save_depth_png_mm(depth_m: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    depth_mm = np.clip(depth_m.astype(np.float64) * 1000.0, 0, 65535).astype(np.uint16)
    Image.fromarray(depth_mm).save(path)


def load_poses_txt(path: Path) -> np.ndarray:
    poses = np.loadtxt(path, dtype=np.float64)
    if poses.ndim == 1:
        poses = poses[None, :]
    return poses.reshape((-1, 4, 4))


def write_poses_txt(path: Path, poses: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, poses.reshape((poses.shape[0], 16)), fmt="%.9f")


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: tuple[int, int, int]) -> None:
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox(xy, text, font=font)
    pad = 3
    bg = (255, 255, 255)
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=bg)
    draw.text(xy, text, fill=fill, font=font)


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    fill: tuple[int, int, int],
    width: int = 3,
) -> None:
    draw.line((start[0], start[1], end[0], end[1]), fill=fill, width=width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    norm = math.hypot(dx, dy)
    if norm <= 1e-6:
        return
    ux, uy = dx / norm, dy / norm
    px, py = -uy, ux
    head = 10.0
    wing = 5.5
    points = [
        (end[0], end[1]),
        (end[0] - ux * head + px * wing, end[1] - uy * head + py * wing),
        (end[0] - ux * head - px * wing, end[1] - uy * head - py * wing),
    ]
    draw.polygon(points, fill=fill)


def draw_bev_coordinate_axes(
    draw: ImageDraw.ImageDraw,
    coord: dict[str, Any],
    image_size: tuple[int, int],
    *,
    scale: float = 1.0,
    font: ImageFont.ImageFont | None = None,
) -> None:
    """Overlay the map-frame axes used by world_to_bev on a BEV image."""
    bounds = coord.get("world_bounds")
    if not isinstance(bounds, dict):
        return
    cell = float(coord.get("meters_per_pixel") or coord.get("cell_size_m") or 0.0)
    if cell <= 0.0:
        return
    try:
        x_min = float(bounds["x_min"])
        z_min = float(bounds["z_min"])
    except (KeyError, TypeError, ValueError):
        return

    def to_xy(world_x: float, world_z: float) -> tuple[float, float]:
        return ((world_x - x_min) / cell * float(scale), (world_z - z_min) / cell * float(scale))

    width, height = image_size
    origin = to_xy(0.0, 0.0)
    margin = 18.0
    ox = min(max(origin[0], margin), max(margin, width - margin))
    oy = min(max(origin[1], margin), max(margin, height - margin))
    anchor = (ox, oy)

    axis_len_m = 1.0
    x_end_raw = to_xy(axis_len_m, 0.0)
    y_end_raw = to_xy(0.0, axis_len_m)
    x_vec = (x_end_raw[0] - origin[0], x_end_raw[1] - origin[1])
    y_vec = (y_end_raw[0] - origin[0], y_end_raw[1] - origin[1])
    max_len = max(36.0, min(90.0, min(width, height) * 0.12))

    def clipped_end(vec: tuple[float, float]) -> tuple[float, float]:
        norm = math.hypot(vec[0], vec[1])
        if norm <= 1e-6:
            return anchor
        length = min(max_len, norm)
        return (anchor[0] + vec[0] / norm * length, anchor[1] + vec[1] / norm * length)

    panel = (10, 52, 194, 134)
    draw.rounded_rectangle(panel, radius=7, fill=(255, 255, 255), outline=(203, 213, 225))
    draw.text((20, 61), "map frame", fill=(15, 23, 42), font=font)
    draw.text((20, 79), f"cell={cell:.3f}m/px", fill=(71, 85, 105), font=font)
    draw.text((20, 97), "+x right, +y/world z down", fill=(71, 85, 105), font=font)
    draw.text((20, 115), "bar = 1m", fill=(71, 85, 105), font=font)

    draw.ellipse((anchor[0] - 4, anchor[1] - 4, anchor[0] + 4, anchor[1] + 4), fill=(15, 23, 42), outline=(255, 255, 255), width=2)
    x_end = clipped_end(x_vec)
    y_end = clipped_end(y_vec)
    _draw_arrow(draw, anchor, x_end, fill=(220, 38, 38), width=4)
    _draw_arrow(draw, anchor, y_end, fill=(37, 99, 235), width=4)
    draw.text((x_end[0] + 5, x_end[1] - 8), "+x", fill=(185, 28, 28), font=font, stroke_width=2, stroke_fill=(255, 255, 255))
    draw.text((y_end[0] + 5, y_end[1] - 8), "+y", fill=(29, 78, 216), font=font, stroke_width=2, stroke_fill=(255, 255, 255))


def make_contact_sheet(image_paths: list[Path], out_path: Path, cols: int = 5, thumb_width: int = 240) -> None:
    if not image_paths:
        return
    thumbs = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        ratio = thumb_width / img.width
        thumb = img.resize((thumb_width, int(round(img.height * ratio))))
        thumbs.append((path.stem, thumb))
    tile_h = max(t.height for _, t in thumbs) + 28
    rows = int(math.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * thumb_width, rows * tile_h), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    for i, (label, thumb) in enumerate(thumbs):
        x = (i % cols) * thumb_width
        y = (i // cols) * tile_h
        sheet.paste(thumb, (x, y + 24))
        draw.text((x + 8, y + 5), label, fill=(20, 24, 30))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
