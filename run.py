#!/usr/bin/env python3
"""Production command line for offline native three-map generation."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / 'package'
PYTHON = PACKAGE / '.venv-maps/bin/python'


def environment():
    env = os.environ.copy()
    for key in ('PYTHONPATH', 'WHICHGROUND_WORKSPACE', 'WHICHGROUND_RUN_CONFIG'):
        env.pop(key, None)
    model_root = Path(env.get('SEMANTIC_GEN_MODEL_ROOT', PACKAGE / 'data/models/semantic_mapping')).expanduser().resolve()
    cache_root = Path(env.get('SEMANTIC_GEN_CACHE_ROOT', ROOT / 'cache')).expanduser().resolve()
    env.update(HF_HUB_CACHE=str(cache_root / 'huggingface'), HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled', DISABLE_WANDB='true',
               SEMANTIC_GEN_MODEL_ROOT=str(model_root), SEMANTIC_GEN_CACHE_ROOT=str(cache_root),
               SEMANTIC_GEN_CLIP_ROOT=str(model_root / 'clip'),
               PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', TOKENIZERS_PARALLELISM='false',
               MPLBACKEND='Agg')
    return env


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='action', required=True)
    build = sub.add_parser('build', help='Run native maps, optional label locking and delivery collection')
    build.add_argument('--bag', type=Path)
    build.add_argument('--prepared-input', type=Path)
    build.add_argument('--calibration', type=Path)
    build.add_argument('--label-yaml', type=Path)
    build.add_argument('--output-root', type=Path, default=ROOT / 'runs')
    build.add_argument('--max-samples', type=int, default=0)
    build.add_argument('--prepare-only', action='store_true')
    build.add_argument('--resume', type=Path)
    stage = sub.add_parser('stage', help='Inspect or resume a native stage using its frozen run config')
    stage.add_argument('--config', required=True, type=Path)
    stage.add_argument('--stage', required=True)
    stage.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    if not PYTHON.is_file():
        ap.error('Run scripts/setup_env.sh to build the Python environments')
    if args.action == 'build':
        if args.resume and args.prepared_input:
            ap.error('--resume already uses the run’s original prepared input')
        if args.bag and args.prepared_input:
            ap.error('choose --bag or --prepared-input')
        if not args.resume and not (args.bag or args.prepared_input):
            ap.error('build requires --bag, --prepared-input or --resume')
        command = [str(PYTHON), str(ROOT / 'runner/build_maps.py'),
                   '--package-root', str(PACKAGE), '--output-root', str(args.output_root.resolve()),
                   '--max-samples', str(args.max_samples)]
        for key in ['bag', 'prepared_input', 'calibration', 'label_yaml', 'resume']:
            value = getattr(args, key)
            if value:
                command += ['--' + key.replace('_', '-'), str(value.resolve())]
        if args.prepare_only:
            command.append('--prepare-only')
    else:
        command = [str(PYTHON), str(PACKAGE / 'code/robot_mapping/pipeline.py'),
                   '--config', str(args.config.resolve()), '--stage', args.stage]
        if args.resume:
            command.append('--resume')
    print('Local semantic map generation:', ' '.join(command), flush=True)
    return subprocess.call(command, env=environment(), cwd=ROOT)


if __name__ == '__main__':
    sys.exit(main())
