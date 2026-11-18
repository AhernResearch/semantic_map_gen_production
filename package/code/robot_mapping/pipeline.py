"""Configurable real-robot pipeline with explicit, fingerprint-checked resumption."""
from pathlib import Path
import argparse
import datetime
import fcntl
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from write_run_report import write_report
from model_assets import asset_paths, inspect_assets, manifest


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


class FileDigests:
    """Cache content hashes only while inode/size/mtime/ctime remain unchanged."""
    def __init__(self, path, verify_content=False):
        self.path = Path(path)
        try:
            self.cache = json.loads(self.path.read_text()) if not verify_content else {}
        except (OSError, ValueError):
            self.cache = {}

    def digest(self, path, force_content=False):
        path = Path(path).resolve()
        stat = path.stat()
        identity = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
        old = self.cache.get(str(path))
        if not force_content and old and old.get('identity') == identity:
            return old['sha256']
        hasher = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                hasher.update(chunk)
        # Detect concurrent writes rather than signing a mixed stream.
        after = path.stat()
        if [after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns] != identity:
            raise RuntimeError(f'File changed while fingerprinting: {path}')
        result = hasher.hexdigest()
        self.cache[str(path)] = {'identity': identity, 'sha256': result}
        return result

    def save(self):
        atomic_json(self.path, self.cache)


def file_manifest(paths, digests, source_only=False, allow_missing=False):
    files = {}
    for path in paths:
        path = Path(path)
        if not path.exists():
            if allow_missing:
                files[str(path)] = {'missing': True}
                continue
            raise RuntimeError(f'Missing required artifact/input: {path}')
        selected = sorted(path.rglob('*')) if path.is_dir() else [path]
        selected = [p for p in selected if p.is_file() and not any(part in {'.git', '__pycache__'} for part in p.parts) and p.suffix != '.pyc']
        if source_only:
            selected = [p for p in selected if p.suffix.lower() in {'.py', '.js', '.html', '.css', '.csv', '.npy', '.npz', '.yaml', '.yml', '.json', '.toml', '.txt', '.gz', '.md', '.sh', '.c', '.h', '.hpp', '.cpp', '.cu', '.cmake', '.cfg', '.ini'} or p.name in {'LICENSE', 'NOTICE', 'CMakeLists.txt'}]
        if not selected:
            if allow_missing:
                files[str(path)] = {'empty': True}
                continue
            raise RuntimeError(f'Empty required artifact/input: {path}')
        for file in selected:
            if file.stat().st_size == 0:
                # Empty source/config marker files are legitimate. Artifact
                # nonemptiness is checked separately on the stage roots.
                files[str(file)] = {'bytes': 0, 'sha256': digests.digest(file, force_content=source_only)}
            else:
                files[str(file)] = {'bytes': file.stat().st_size, 'sha256': digests.digest(file, force_content=source_only)}
    return files


def stage_artifacts(name, cfg):
    root = cfg.WORKSPACE
    scene = cfg.SCENE_ID
    cg = root / 'native_maps/ConceptGraphs/data' / scene
    maps = {
        'doctor': [root / 'doctor.json'],
        'prepare': [root / 'supplied_dataset'],
        'features': [root / 'semantic_keyframes/clip_features.npz', root / 'semantic_keyframes/retrieval.json', root / 'semantic_keyframes/retrieval_preview.jpg'],
        'depth': [root / 'da3/depth_results.json', root / 'da3/depth_npy_fullres'],
        'align': [root / 'aligned_scene_input', root / 'depth_alignment_report.json'],
        'scene': [cfg.stage_input(), root / 'scene_packages' / scene],
        'hmsg': [root / 'native_maps/HMSG' / scene / 'ovo_map.ckpt'],
        'cg_prepare': [cg / 'results', cg / 'traj.txt', cg / 'SOURCE.json'],
        'cg_detect': [cg / 'exp_new_robot/detections', cg / 'classes.json'],
        'cg_map': [cg / 'pcd_saves/full_pcd_yolo_world_new_robot_post.pkl.gz'],
        'vlmaps': [root / 'native_maps/VLMaps' / scene / 'vlmap_cam/vlmaps_cam.h5df'],
        'validate': [root / 'native_maps/validation.json', root / 'native_maps/preview'],
        'preview': [root / 'viewer/index.html', root / 'viewer/data.js', root / 'viewer/cloud_viewer.js', root / 'viewer/vendor', root / 'viewer/full_clouds_manifest.json'],
        'package': [root / 'share' / (scene + '_native_three_maps.tar.gz'), root / 'share' / (scene + '_native_three_maps.tar.gz.sha256'), root / 'share/native_package_result.json'],
    }
    if name == 'hmsg':
        graphs = sorted((root / 'native_maps/HMSG' / scene).glob('graph_*'))
        if not graphs:
            raise RuntimeError('HMSG hierarchy has no graph_* output')
        maps[name].append(graphs[-1])
    if name == 'preview' and (root / 'viewer/full_clouds_manifest.json').exists():
        # Sign all native cloud chunks exported by this stage, without claiming
        # optional later all-frame LiDAR exports were produced by this stage.
        full = json.loads((root / 'viewer/full_clouds_manifest.json').read_text())
        maps[name].extend(root / 'viewer/full' / key for key in full)
    return maps[name]


def artifact_manifest(name, cfg, digests):
    paths = stage_artifacts(name, cfg)
    for path in paths:
        if path.is_file() and path.stat().st_size == 0:
            raise RuntimeError(f'Empty {name} artifact: {path}')
    manifest = file_manifest(paths, digests)
    if name in {'doctor', 'validate'}:
        data = json.loads(paths[0].read_text())
        if not data.get('all_passed'):
            raise RuntimeError(f'{name} result does not report all_passed')
    return manifest


DEPENDENCIES = {
    'doctor': [], 'prepare': [], 'features': ['prepare'], 'depth': ['prepare'],
    'align': ['prepare', 'depth'], 'scene': ['align'], 'hmsg': ['scene'],
    'cg_prepare': ['scene'], 'cg_detect': ['cg_prepare'], 'cg_map': ['cg_prepare', 'cg_detect'],
    'vlmaps': ['scene'], 'validate': ['hmsg', 'cg_map', 'vlmaps', 'scene'],
    'preview': ['validate'], 'package': ['validate', 'scene', 'prepare'],
}


def resumable(previous, signature, current_artifacts):
    return bool(previous and previous.get('exit_code') == 0 and previous.get('signature') == signature and previous.get('artifacts') == current_artifacts and current_artifacts)


def signature_inputs(cfg, digests):
    # Only consumed source data: the selected bag and calibration. The other
    # supplied SLAM PCD directory is not an input to this pipeline.
    inputs = file_manifest([cfg.bag_path(), cfg.calibration_path()], digests, allow_missing=True)
    sources = file_manifest([cfg.CODE_ROOT / 'robot_mapping', cfg.CODE_ROOT / 'adapters', cfg.CODE_ROOT / 'Depth-Anything-3', cfg.THIRD_PARTY], digests, source_only=True, allow_missing=True)
    runtime = {d.metadata['Name']: d.version for d in importlib.metadata.distributions() if d.metadata.get('Name')}
    environment_paths = [cfg.PACKAGE_ROOT / '.venv-maps/pyvenv.cfg', cfg.PACKAGE_ROOT / '.venv-da3/pyvenv.cfg', cfg.PACKAGE_ROOT / 'kit/requirements-mapping.lock', cfg.PACKAGE_ROOT / 'kit/requirements-mapping-base.lock', cfg.PACKAGE_ROOT / 'kit/requirements-depth.lock']
    # Include depth-venv-specific distribution metadata as well as this
    # process's inherited map-runtime distribution versions.
    environment_paths += sorted((cfg.PACKAGE_ROOT / '.venv-da3/lib').glob('*/site-packages/*.dist-info/METADATA'))
    environment = file_manifest(environment_paths, digests, allow_missing=True)
    model_paths = []
    for entry in manifest():
        path, ref = asset_paths(entry, cfg.MODEL_ROOT, cfg.CACHE_ROOT, cfg.DA3_MODEL)
        model_paths.append(path)
        if ref is not None:
            model_paths.append(ref)
    models = file_manifest(model_paths, digests, allow_missing=True)
    return {'config_sha256': digests.digest(cfg.CONFIG_PATH), 'inputs_sha256': digest_json(inputs), 'sources_sha256': digest_json(sources), 'models_sha256': digest_json(models), 'environment_sha256': digest_json({'files': environment, 'versions': runtime}), 'scene_id': cfg.SCENE_ID, 'workspace': str(cfg.WORKSPACE)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--workspace', type=Path)
    parser.add_argument('--stage', required=True, choices=['doctor', 'prepare', 'features', 'depth', 'align', 'scene', 'hmsg', 'cg_prepare', 'cg_detect', 'cg_map', 'vlmaps', 'validate', 'preview', 'package', 'all', 'native'])
    parser.add_argument('--reuse-ovo', type=Path, help='Rebuild only HMSG hierarchy from an explicitly supplied existing OVO checkpoint directory')
    parser.add_argument('--resume', action='store_true', help='Skip successful stages only when config/input/source/runtime and artifact hashes match')
    parser.add_argument('--verify-content', action='store_true', help='Rehash all file contents instead of reusing unchanged filesystem-identity hash cache')
    args = parser.parse_args()
    os.environ['WHICHGROUND_RUN_CONFIG'] = str(args.config.expanduser().resolve())
    if args.workspace:
        os.environ['WHICHGROUND_WORKSPACE'] = str(args.workspace.expanduser().resolve())
    import run_config as cfg
    workspace_lock = (cfg.WORKSPACE / 'pipeline.lock').open('a')
    try:
        fcntl.flock(workspace_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Another pipeline is writing this workspace; use a different --workspace or wait for it to finish.')
    if args.reuse_ovo and args.reuse_ovo.expanduser().resolve() != (cfg.WORKSPACE / 'native_maps/HMSG' / cfg.SCENE_ID).resolve():
        raise RuntimeError('--reuse-ovo must point to this workspace/scene HMSG directory; downstream exports require the same scene. Copy an explicitly verified matching checkpoint first if migrating it.')
    root = cfg.PACKAGE_ROOT
    scripts = Path(__file__).resolve().parent
    maps_python = root / '.venv-maps/bin/python'
    depth_python = root / '.venv-da3/bin/python'
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', MPLBACKEND='Agg', WANDB_MODE='disabled', DISABLE_WANDB='true', PYTHONDONTWRITEBYTECODE='1')
    env['PYTHONPATH'] = os.pathsep.join(map(str, [scripts, root / 'code', cfg.THIRD_PARTY / 'sam2', cfg.THIRD_PARTY / 'segment-anything', cfg.THIRD_PARTY / 'MobileSAM', cfg.THIRD_PARTY / 'CLIP']))
    env.update(HF_HUB_CACHE=str(cfg.CACHE_ROOT / 'huggingface'), HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', SEMANTIC_GEN_CLIP_ROOT=str(cfg.MODEL_ROOT / 'clip'))
    env['PATH'] = str(maps_python.parent) + os.pathsep + env.get('PATH', '')
    jobs = {
        'doctor': (maps_python, 'doctor.py', [], cfg.WORKSPACE),
        'prepare': (maps_python, 'prepare_supplied_dataset.py', [], cfg.WORKSPACE),
        'features': (maps_python, 'build_supplied_semantics.py', [], cfg.WORKSPACE),
        'depth': (depth_python, 'run_depth.py', [], cfg.WORKSPACE),
        'align': (maps_python, 'analyze_supplied_results.py', [], cfg.WORKSPACE),
        'scene': (maps_python, 'build_scene.py', [], cfg.WORKSPACE),
        'hmsg': (maps_python, 'run_native_hmsg.py', ['--build-hmsg'] + (['--existing-ovo', str(args.reuse_ovo.expanduser().resolve())] if args.reuse_ovo else []), cfg.THIRD_PARTY / 'HoloAgent/agentic_robot/fsr_vln'),
        'cg_prepare': (maps_python, 'run_native_conceptgraphs.py', ['--stage', 'prepare'], cfg.WORKSPACE),
        'cg_detect': (maps_python, 'run_native_conceptgraphs.py', ['--stage', 'detect'], cfg.THIRD_PARTY / 'concept-graphs'),
        'cg_map': (maps_python, 'run_native_conceptgraphs.py', ['--stage', 'map'], cfg.THIRD_PARTY / 'concept-graphs'),
        'vlmaps': (maps_python, 'run_native_vlmaps.py', [], cfg.THIRD_PARTY / 'vlmaps'),
        'validate': (maps_python, 'export_native_results.py', [], cfg.WORKSPACE),
        'preview': (maps_python, 'build_viewer.py', [], cfg.WORKSPACE),
        'package': (maps_python, 'package_native_results.py', [], cfg.WORKSPACE),
    }
    sequence = list(jobs) if args.stage == 'all' else (['doctor', 'hmsg', 'cg_prepare', 'cg_detect', 'cg_map', 'vlmaps', 'validate', 'preview', 'package'] if args.stage == 'native' else [args.stage])
    logs = cfg.WORKSPACE / 'logs'
    logs.mkdir(exist_ok=True)
    record_path = cfg.WORKSPACE / 'pipeline_run.json'
    record = {'config': str(cfg.CONFIG_PATH), 'clean_source_root': str(root), 'workspace': str(cfg.WORKSPACE), 'scene_id': cfg.SCENE_ID, 'stages': []}
    if record_path.exists():
        previous = json.loads(record_path.read_text())
        if previous.get('workspace', str(cfg.WORKSPACE)) != str(cfg.WORKSPACE) or previous.get('scene_id') != cfg.SCENE_ID:
            raise RuntimeError('Workspace belongs to another scene. Choose a new --workspace and scene_id.')
        record['stages'] = previous.get('stages', [])
    print('Fingerprinting inputs, sources, runtimes and model files (first run reads full contents).', flush=True)
    digests = FileDigests(cfg.WORKSPACE / 'pipeline_hash_cache.json', args.verify_content)
    fingerprints = signature_inputs(cfg, digests)
    base_identity = digest_json({key: fingerprints[key] for key in ['config_sha256', 'inputs_sha256', 'scene_id', 'workspace']})
    record['fingerprints'] = fingerprints
    record['resume_requested'] = args.resume
    # Old unsigned files cannot be adopted as a successful preprocessing run.
    # Changed dataset/config must never mingle stale sampled images or depths.
    if 'prepare' in sequence and (cfg.WORKSPACE / 'supplied_dataset').exists():
        old_prepare = next((r for r in reversed(record['stages']) if r['stage'] == 'prepare' and r.get('base_identity')), None)
        if not old_prepare or old_prepare['base_identity'] != base_identity:
            raise RuntimeError('Existing preprocessing has no matching signed input/config identity. Use a new --workspace; unsigned historical output is not auto-adopted.')
    digests.save()
    write_report(cfg.WORKSPACE, record)
    for name in sequence:
        python, filename, extra, cwd = jobs[name]
        if not python.is_file():
            raise RuntimeError(f'Missing runtime {python}; run scripts/setup_env.sh')
        command = [str(python), str(scripts / filename), *extra]
        dependency_manifest = {}
        for dependency in DEPENDENCIES[name]:
            dependency_manifest[dependency] = artifact_manifest(dependency, cfg, digests)
        if name == 'hmsg' and args.reuse_ovo:
            dependency_manifest['explicit_ovo'] = file_manifest([args.reuse_ovo.expanduser().resolve() / 'ovo_map.ckpt'], digests)
        signature = digest_json({'fingerprints': fingerprints, 'stage': name, 'command': command, 'dependencies': dependency_manifest})
        previous = next((r for r in reversed(record['stages']) if r['stage'] == name and r.get('signature') == signature and r.get('exit_code') == 0), None)
        artifacts = None
        if args.resume and name != 'doctor' and previous:
            try:
                artifacts = artifact_manifest(name, cfg, digests)
            except (OSError, ValueError, RuntimeError):
                artifacts = None
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        row = {'stage': name, 'command': command, 'cwd': str(cwd), 'started_at': stamp, 'signature': signature, 'base_identity': base_identity, 'dependencies_sha256': digest_json(dependency_manifest)}
        if args.resume and name != 'doctor' and resumable(previous, signature, artifacts):
            row.update(exit_code=0, skipped=True, artifacts=artifacts, reused_from=previous['started_at'], log=previous.get('log'))
            record['stages'].append(row)
            atomic_json(record_path, record)
            digests.save()
            write_report(cfg.WORKSPACE, record)
            print(f'[{name}] RESUME: signature and artifact hashes match', flush=True)
            continue
        log = logs / (name + '.' + stamp + '.log')
        latest_log = logs / (name + '.log')
        if latest_log.exists() and not latest_log.is_symlink():
            latest_log.rename(logs / (name + '.legacy.' + stamp + '.log'))
        latest_log.unlink(missing_ok=True)
        latest_log.symlink_to(log.name)
        row['log'] = str(log.relative_to(cfg.WORKSPACE))
        record['stages'].append(row)
        atomic_json(record_path, record)
        write_report(cfg.WORKSPACE, record)
        print(f'[{name}] {filename} -> {log}', flush=True)
        with log.open('w') as stream:
            invalid = [
                check for check in inspect_assets(cfg.MODEL_ROOT, cfg.CACHE_ROOT,
                    depth_model=cfg.DA3_MODEL, stages={name}, digest=digests.digest)
                if not check['passed']
            ]
            if invalid:
                stream.write(json.dumps(invalid, indent=2) + '\n')
                row['exit_code'] = 1
            else:
                result = subprocess.run(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT)
                row['exit_code'] = result.returncode
        if row['exit_code'] == 0:
            try:
                row['artifacts'] = artifact_manifest(name, cfg, digests)
            except (OSError, ValueError, RuntimeError) as error:
                row['exit_code'] = 1
                row['artifact_error'] = str(error)
        row['finished_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        atomic_json(record_path, record)
        digests.save()
        write_report(cfg.WORKSPACE, record)
        if row['exit_code']:
            print(log.read_text()[-6000:], file=sys.stderr)
            if row.get('artifact_error'):
                print(row['artifact_error'], file=sys.stderr)
            raise SystemExit(row['exit_code'])
    print('COMPLETE', args.stage, cfg.WORKSPACE, flush=True)


if __name__ == '__main__':
    main()
