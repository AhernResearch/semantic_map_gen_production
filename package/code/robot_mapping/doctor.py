"""Offline checks for both inference runtimes, canonical sources and model caches."""
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

from model_assets import inspect_assets


def safetensors_complete(path, minimum=1):
    """Validate the index/byte extent without loading multi-GB tensors into RAM."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size < minimum:
        return False
    try:
        with path.open('rb') as stream:
            length = int.from_bytes(stream.read(8), 'little')
            if length <= 0 or length > 16 * 1024 * 1024:
                return False
            header = json.loads(stream.read(length))
        extent = max(v['data_offsets'][1] for k, v in header.items() if k != '__metadata__')
        return 8 + length + extent == path.stat().st_size
    except (OSError, ValueError, KeyError, TypeError):
        return False


def cached_file(repo, filename, cache_dir=None):
    from huggingface_hub import try_to_load_from_cache
    result = try_to_load_from_cache(repo, filename, cache_dir=str(cache_dir) if cache_dir else None)
    return Path(result) if isinstance(result, str) and Path(result).is_file() else None


def main():
    import run_config as rc
    checks = []
    for name in ['torch', 'numpy', 'cv2', 'open3d', 'open_clip', 'rosbags', 'faiss', 'chamferdist', 'pytorch3d', 'skfmm', 'pyvista', 'hydra', 'ultralytics', 'pytorch_lightning', 'sam2', 'segment_anything', 'mobile_sam', 'clip']:
        try:
            module = importlib.import_module(name)
            origin = getattr(module, '__file__', None)
            canonical = name not in ['sam2', 'segment_anything', 'mobile_sam', 'clip'] or (origin and Path(origin).resolve().is_relative_to(rc.THIRD_PARTY))
            checks.append({'item': name, 'passed': bool(canonical), 'origin': origin})
        except Exception as error:
            checks.append({'item': name, 'passed': False, 'error': str(error)})
    try:
        import torch
        checks.append({'item': 'CUDA inference', 'passed': torch.cuda.is_available(), 'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    except Exception as error:
        checks.append({'item': 'CUDA inference', 'passed': False, 'error': str(error)})
    checks.extend(inspect_assets(rc.MODEL_ROOT, rc.CACHE_ROOT, depth_model=rc.DA3_MODEL))
    for name in ['HoloAgent/agentic_robot/fsr_vln/ovo/entities/ovomapping_offline.py', 'concept-graphs/conceptgraph/slam/streamlined_mapping.py', 'vlmaps/vlmaps/map/vlmap_builder_cam.py']:
        checks.append({'item': name, 'passed': (rc.THIRD_PARTY / name).is_file()})

    # Use the same source tree the DA3 adapter inserts into sys.path. Import and
    # CUDA availability are checked in the depth venv rather than this process.
    depth_python = rc.PACKAGE_ROOT / '.venv-da3/bin/python'
    probe = "import json,torch,numpy,cv2; from depth_anything_3.api import DepthAnything3; import depth_anything_3.api as api; print('DA3_PROBE='+json.dumps({'api':api.__file__,'cuda':torch.cuda.is_available(),'torch':torch.__version__,'numpy':numpy.__version__}))"
    depth_env = os.environ.copy()
    depth_env['PYTHONPATH'] = str(rc.CODE_ROOT / 'Depth-Anything-3/src') + os.pathsep + depth_env.get('PYTHONPATH', '')
    try:
        result = subprocess.run([str(depth_python), '-c', probe], env=depth_env, capture_output=True, text=True, timeout=60)
        line = next((x for x in result.stdout.splitlines() if x.startswith('DA3_PROBE=')), None)
        detail = json.loads(line.removeprefix('DA3_PROBE=')) if line else {'error': (result.stdout + result.stderr)[-3000:]}
        passed = result.returncode == 0 and detail.get('cuda') and Path(detail['api']).resolve().is_relative_to(rc.CODE_ROOT / 'Depth-Anything-3')
        checks.append({'item': 'DA3 runtime/API/CUDA', 'path': str(depth_python), 'passed': bool(passed), 'detail': detail})
    except Exception as error:
        checks.append({'item': 'DA3 runtime/API/CUDA', 'path': str(depth_python), 'passed': False, 'error': str(error)})
    model_config = rc.DA3_MODEL / 'config.json'
    weights = rc.DA3_MODEL / 'model.safetensors'
    try:
        config_valid = isinstance(json.loads(model_config.read_text()), dict)
    except (OSError, ValueError):
        config_valid = False
    checks.append({'item': 'DA3 model/config complete', 'path': str(rc.DA3_MODEL), 'passed': config_valid and safetensors_complete(weights, 1_000_000_000)})
    cache_specs = [
        ('CLIP-B32 OpenAI retrieval cache', 'timm/vit_base_patch32_clip_224.openai', rc.CLIP_CACHE, 500_000_000, False),
        ('CLIP-H14 LAION OVO/CG cache', 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K', None, 3_000_000_000, True),
    ]
    for item, repo, cache_dir, minimum, need_config in cache_specs:
        try:
            path = cached_file(repo, 'open_clip_model.safetensors', cache_dir)
            config = cached_file(repo, 'open_clip_config.json', cache_dir) if need_config else None
            config_ok = not need_config or (config and isinstance(json.loads(config.read_text()), dict))
            checks.append({'item': item, 'repo': repo, 'path': str(path) if path else None, 'passed': bool(path and safetensors_complete(path, minimum) and config_ok), 'offline_only': True})
        except Exception as error:
            checks.append({'item': item, 'passed': False, 'error': str(error), 'offline_only': True})
    checks.append({'item': 'input bag', 'passed': rc.bag_path().is_dir(), 'path': str(rc.bag_path())})
    checks.append({'item': 'supplied calibration', 'passed': rc.calibration_path().is_file(), 'path': str(rc.calibration_path())})
    # Metadata is read offline; this also prevents missing/wrong topics reaching
    # expensive model inference. The prepare stage repeats the type check.
    try:
        from rosbags.highlevel import AnyReader
        if rc.CONFIG.get('dataset',{}).get('prepared_observations'):
            from portable_observations import BundleReader as AnyReader
        from rosbags.typesys import Stores, get_typestore
        expected = {'image': 'sensor_msgs/msg/Image', 'lidar': 'sensor_msgs/msg/PointCloud2', 'odom': 'nav_msgs/msg/Odometry'}
        with AnyReader([rc.bag_path()], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:
            for role, topic in rc.TOPICS.items():
                connections = [c for c in reader.connections if c.topic == topic]
                checks.append({'item': 'configured ' + role + ' topic', 'topic': topic, 'passed': bool(connections) and all(c.msgtype == expected[role] and c.msgcount > 0 for c in connections), 'message_count': sum(c.msgcount for c in connections)})
    except Exception as error:
        checks.append({'item': 'bag topic metadata', 'passed': False, 'error': str(error)})
    result = {'canonical_package': str(rc.PACKAGE_ROOT), 'python': sys.executable, 'checks': checks, 'all_passed': all(c['passed'] for c in checks), 'scope': 'Offline environment, cached model completeness and input topic/type checks; does not measure map or navigation accuracy.'}
    (rc.WORKSPACE / 'doctor.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result['all_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
