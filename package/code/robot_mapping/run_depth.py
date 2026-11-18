"""Run the clean package's DA3 adapter with explicit dataset/output paths."""
import subprocess
import sys
from pathlib import Path
import run_config as rc

script = rc.CODE_ROOT / 'adapters/scripts/realdata/depth_completion/run_da3_depth_0615.py'
model = rc.DA3_MODEL
command = [sys.executable, str(script), '--image-dir', str(rc.WORKSPACE / 'supplied_dataset/rgb_rectified'),
           '--intrinsics-json', str(rc.WORKSPACE / 'supplied_dataset/intrinsics.json'),
           '--output-dir', str(rc.WORKSPACE / 'da3'), '--model-dir', str(model),
           '--indices', 'all', '--batch-size', '1', '--process-res', str(rc.CONFIG.get('depth', {}).get('process_resolution', 518)),
           '--device', 'cuda', '--no-make-montage']
subprocess.run(command, check=True)
