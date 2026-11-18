# Depth-Anything-3 (vendored source only)

Upstream: https://github.com/ByteDance-Seed/Depth-Anything-3 — **code is Apache-2.0** (see `LICENSE`).
Only `src/`, `LICENSE` and `pyproject.toml` are included, unmodified (mtimes are all the upstream snapshot time; no
project-specific strings found). Example images and docs were left out.

**Model weights are NOT included and have their own licenses** (check each model card):
- `DA3METRIC-LARGE`, `DA3MONO-LARGE`, `DA3-BASE`, `DA3-SMALL`: Apache-2.0 -> usable commercially. This project now defaults to `DA3METRIC-LARGE`.
- `DA3-GIANT*`, `DA3-LARGE*`, `DA3NESTED-GIANT-LARGE*`: CC BY-NC 4.0 -> non-commercial only.

Download the default weights: `bash adapters/scripts/realdata/download_da3_metric_large_hf_mirror.sh`
(expected location: `Depth-Anything-3/DA3METRIC-LARGE/`).
