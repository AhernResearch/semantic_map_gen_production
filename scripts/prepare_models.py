#!/usr/bin/env python3
"""Print pinned asset paths, initialize offline refs and verify supplied weights."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'package/code/robot_mapping'))
sys.path.insert(0, str(ROOT))

from model_assets import asset_paths, inspect_assets, manifest, sha256
from run import environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['list', 'init-cache', 'verify'])
    args = parser.parse_args()
    env = environment()
    models, cache = map(Path, (env['SEMANTIC_GEN_MODEL_ROOT'], env['SEMANTIC_GEN_CACHE_ROOT']))
    if args.action == 'list':
        rows = []
        for entry in manifest():
            path, ref = asset_paths(entry, models, cache)
            rows.append(dict(entry, destination=str(path), cache_ref=str(ref) if ref else None))
        print(json.dumps(rows, indent=2))
        return 0
    if args.action == 'init-cache':
        for entry in manifest():
            path, ref = asset_paths(entry, models, cache)
            if ref is None:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists() and not path.is_symlink():
                if 'inline_text' in entry:
                    with path.open('x') as stream:
                        stream.write(entry['inline_text'])
                elif 'model_path' in entry:
                    source = models / entry['model_path']
                    if source.is_file():
                        path.symlink_to(os.path.relpath(source, path.parent))
            if path.is_file() and path.stat().st_size == entry['bytes'] and sha256(path) == entry['sha256']:
                ref.parent.mkdir(parents=True, exist_ok=True)
                if not ref.exists():
                    with ref.open('x') as stream:
                        stream.write(entry['revision'])
    checks = inspect_assets(models, cache)
    print(json.dumps(checks, indent=2))
    return int(not all(row['passed'] for row in checks))


if __name__ == '__main__':
    raise SystemExit(main())
