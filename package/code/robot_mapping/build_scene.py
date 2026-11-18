"""Create scene and HMSG RGB-D carriers through the original clean adapters."""
import subprocess
import sys
import run_config as rc

stage = rc.WORKSPACE / 'aligned_scene_input'
script = rc.CODE_ROOT / 'adapters/scripts/realdata/ingestion/build_scene_package_from_completed_depth.py'
subprocess.run([sys.executable, str(script), '--stage-dir', str(stage), '--slam-package', str(stage / 'slam'),
                '--metric-depth-dir', str(stage / 'metric_depth'), '--completion-root', str(stage),
                '--output-root', str(rc.WORKSPACE / 'scene_packages'), '--scene-id', rc.SCENE_ID,
                '--image-scale', '1', '--depth-max', '20', '--bev-source', 'slam_map'], check=True)
script = rc.CODE_ROOT / 'adapters/scripts/fsrvln/scene_package_to_fsrvln.py'
subprocess.run([sys.executable, str(script), '--scene-package', str(rc.WORKSPACE / 'scene_packages' / rc.SCENE_ID),
                '--output-root', str(rc.WORKSPACE / 'mapper_inputs'),
                '--config-root', str(rc.WORKSPACE / 'native_configs/adapter'), '--save-root', str(rc.WORKSPACE / 'native_maps/HMSG'),
                '--mode', 'copy', '--skip-frames', '1', '--source-frame-stride', '1',
                '--resize-width', '640', '--resize-height', '360', '--no-merge-objects-graph'], check=True)
