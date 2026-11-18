"""Verify pinned model inputs without importing inference libraries."""
import hashlib
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2]
MANIFEST = PACKAGE / 'docs/mapping_provenance/weights.json'


def manifest():
    return json.loads(MANIFEST.read_text())


def sha256(path):
    with path.open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def asset_paths(entry, model_root, cache_root, depth_model=None):
    if entry['root'] == 'models':
        path = Path(entry['path'])
        if depth_model is not None and path.parts[0] == 'DA3METRIC-LARGE':
            return depth_model.joinpath(*path.parts[1:]), None
        return model_root / path, None
    path = cache_root / entry['root'] / entry['path']
    return path, path.parents[2] / 'refs/main'


def inspect_assets(model_root, cache_root, *, depth_model=None, stages=None, digest=sha256):
    checks = []
    for entry in manifest():
        if stages is not None and not stages.intersection(entry['stages']):
            continue
        path, ref = asset_paths(entry, model_root, cache_root, depth_model)
        error = None
        try:
            if not path.is_file():
                error = 'missing file'
            elif path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
                error = 'size or SHA-256 mismatch'
            elif ref is not None and (
                not ref.is_file() or ref.read_text().strip() != entry['revision']
            ):
                error = f'missing or incorrect cache ref {ref}'
        except OSError as exc:
            error = str(exc)
        checks.append(dict(item='model identity', path=str(path), stages=entry['stages'],
                           passed=error is None, error=error, source=entry['source']))
    return checks
