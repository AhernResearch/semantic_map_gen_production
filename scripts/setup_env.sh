#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
python_bin="${SEMANTIC_GEN_PYTHON:-python3.10}"
pypi_index="${SEMANTIC_GEN_PYPI_INDEX:-https://pypi.org/simple}"
torch_index="${SEMANTIC_GEN_TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
requested="${1:-all}"
case "$requested" in
    all|maps|da3) ;;
    *) echo "Usage: scripts/setup_env.sh [all|maps|da3]" >&2; exit 2 ;;
esac

"$python_bin" -c 'import platform, sys; assert sys.platform == "linux" and platform.machine() == "x86_64" and sys.version_info[:2] == (3, 10), "Linux x86_64 / Python 3.10 required"'
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_CONFIG_FILE=/dev/null
unset PIP_INDEX_URL PIP_EXTRA_INDEX_URL PIP_FIND_LINKS
export PYTHONNOUSERSITE=1

install_environment() {
    local kind="$1"
    local env_dir="$root/package/.venv-$kind"
    local python="$env_dir/bin/python"
    echo "Installing $kind into $env_dir"
    if [[ ! -x "$python" ]]; then
        "$python_bin" -m venv "$env_dir"
    fi
    "$python" -m pip install --index-url "$pypi_index" 'pip==26.3' 'setuptools==79.0.1' 'wheel==0.48.0'
    "$python" -m pip install --index-url "$pypi_index" --extra-index-url "$torch_index" \
        'torch==2.5.1+cu124' 'torchvision==0.20.1+cu124'
    if [[ "$kind" == da3 ]]; then
        "$python" -m pip install --index-url "$pypi_index" -r package/kit/requirements-depth.lock
        local site
        site="$("$python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
        printf '%s\n' "$root/package/code/Depth-Anything-3/src" > "$site/semantic_map_sources.pth"
        "$python" -m pip check
        "$python" -c 'import torch, cv2; from depth_anything_3.api import DepthAnything3; print(torch.__version__, cv2.__version__)'
        return
    fi

    "$python" -m pip install --index-url "$pypi_index" \
        -r package/kit/requirements-mapping-base.lock -r package/kit/requirements-mapping.lock
    # GradSLAM's metadata pins Open3D 0.10, which the frozen CG stack did not use.
    # Its imports run against the explicit versions in the mapping requirements.
    "$python" -m pip install --index-url "$pypi_index" --no-deps 'gradslam==0.1.0'

    local site vendor
    site="$("$python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
    vendor="$root/package/code/third_party"
    printf '%s\n' "$vendor"/{sam2,segment-anything,MobileSAM,CLIP} > "$site/semantic_map_sources.pth"

    export PATH="$env_dir/bin:$PATH"
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"
    if [[ ! -x "$CUDA_HOME/bin/nvcc" ]] || ! "$CUDA_HOME/bin/nvcc" --version | grep -q 'release 12\.4'; then
        echo "CUDA Toolkit 12.4 (nvcc) is required to build the CG extensions. Set CUDA_HOME." >&2
        return 1
    fi
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
    export MAX_JOBS="${MAX_JOBS:-2}"
    export FORCE_CUDA=1
    "$python" -m pip install --index-url "$pypi_index" --no-build-isolation --no-deps \
        'chamferdist==1.0.3' \
        'pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47'
    "$python" -c 'import cv2, torch, open3d, gradslam, clip, chamferdist, pytorch3d; import sam2, segment_anything, mobile_sam; print(torch.__version__, cv2.__version__)'
}

if [[ "$requested" != maps ]]; then install_environment da3; fi
if [[ "$requested" != da3 ]]; then install_environment maps; fi
