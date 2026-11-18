#!/usr/bin/env python3
"""Collect existing native mapping outputs; never run models or edit sources.

Completion means native artifact/format/pipeline contract checks, not independent
semantic accuracy, complete object surfaces or robot arrival. HMSG is packed as
an independent graph archive in a new output directory; CG/HDF files remain
available at their actual workspace paths.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import tarfile
import time
from zoneinfo import ZoneInfo

STAGES = ['doctor', 'prepare', 'features', 'depth', 'align', 'scene', 'hmsg', 'cg_prepare', 'cg_detect', 'cg_map', 'vlmaps', 'validate', 'preview', 'package']


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def json_file(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError('JSON非有限数值：' + value)))


def parse_time(value):
    if not isinstance(value, str):
        return None
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        try:
            result = datetime.strptime(value, '%Y%m%dT%H%M%S.%fZ').replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result


def row_duration(row):
    started, finished = parse_time(row.get('started_at')), parse_time(row.get('finished_at'))
    if row.get('skipped') or not started or not finished:
        return None
    duration = (finished - started).total_seconds()
    return duration if duration >= 0 else None


def stage_timings(document, latest):
    rows = document.get('stages', [])
    details, values = [], {}
    for name, row in latest.items():
        duration = row_duration(row)
        values[name] = duration
        source = None
        if row.get('skipped'):
            source = next((old for old in reversed(rows) if old.get('stage') == name and old.get('exit_code') == 0 and not old.get('skipped') and old.get('started_at') == row.get('reused_from')), None)
        elif row.get('exit_code') == 0:
            source = row
        details.append({'stage': name, 'started_at': row.get('started_at'), 'finished_at': row.get('finished_at'),
                        'elapsed_s': duration, 'exit_code': row.get('exit_code'), 'skipped': bool(row.get('skipped')),
                        'timing_status': 'skipped_processing_not_run' if row.get('skipped') else ('recorded_processing_duration' if duration is not None else 'missing_or_invalid_timestamps'),
                        'successful_artifact_source': {'started_at': source.get('started_at'), 'finished_at': source.get('finished_at'), 'elapsed_s': row_duration(source)} if source else None})
    return {'stage_timing_s': values, 'stage_timing_details': details,
            'total_pipeline_stage_s': sum(value for value in values.values() if value is not None),
            'stage_rows_in_record': len(rows), 'latest_stage_rows': len(latest), 'historical_or_retry_rows_excluded_from_total': max(0, len(rows) - len(latest)),
            'basis': '每阶段最后一次观测row的实际处理时间；跳过行无处理耗时，成功artifact原始生成时间另列，不重复累计历史重试/复用来源。',
            'whole_current_invocation_wall_time_known': False,
            'limits': 'pipeline_run没有独立invocation ID；跨resume/单stage运行时last-row来源可不同，不把此sum当完整本次end-to-end。整体计时/2s资源采样由producer记录。'}


def input_volume(workspace):
    path = workspace / 'run_preparation.json'
    if not path.is_file():
        return {'metadata_path': None, 'input_mcap_bytes': None, 'input_archive_bytes': None, 'note': '原run没有run_preparation.json，未猜测输入包大小'}
    meta = json_file(path)
    value = meta.get('bag')
    bag = Path(value).expanduser().resolve() if isinstance(value, str) else None
    files = [bag] if bag and bag.is_file() else (sorted(bag.glob('*.mcap')) if bag and bag.is_dir() else [])
    result = {'metadata_path': str(path), 'bag_path': str(bag) if bag else None,
              'input_mcap_files': [{'path': str(file), 'bytes': file.stat().st_size} for file in files],
              'input_mcap_bytes': sum(file.stat().st_size for file in files) if files else None, 'input_archive_bytes': None}
    source_dataset = meta.get('source_dataset')
    extraction = Path(source_dataset) / '.whichground_extraction.json' if isinstance(source_dataset, str) else None
    if extraction and extraction.is_file():
        source = json_file(extraction).get('source_archive')
        if isinstance(source, str) and Path(source).is_file():
            result.update(input_archive_path=str(Path(source).resolve()), input_archive_bytes=Path(source).stat().st_size)
    return result


def under(path, root):
    return path == root or root in path.parents


def output_path(path, workspace):
    path = Path(path).expanduser().resolve()
    if path == workspace or not under(path, workspace) or under(path, workspace / 'native_maps'):
        raise ValueError('输出必须是运行目录内、原生地图目录之外的新目录')
    if path.exists():
        raise ValueError('输出目录已存在；不得覆盖原文件/旧收集版本')
    return path


class Digests:
    def __init__(self):
        self.values = {}

    def record(self, path):
        path = Path(path).resolve()
        key = str(path)
        if key not in self.values:
            self.values[key] = sha256(path)
        return self.values[key]


def choose(candidates, declared, workspace, kind, missing, warnings):
    candidates = sorted({Path(path).resolve() for path in candidates})
    if declared:
        given = Path(declared).expanduser().resolve()
        if not under(given, workspace):
            missing.append(f'{kind}验证报告引用workspace外的旧产物：{given}')
            return None
        if given in candidates:
            return given
        missing.append(f'{kind}验证报告指定产物不存在/格式不匹配：{given}')
        return None
    if len(candidates) == 1:
        warnings.append(f'{kind}由目录发现；原验证报告未给出明确文件路径')
        return candidates[0]
    if not candidates:
        missing.append(f'缺少{kind}原生产物')
    else:
        missing.append(f'{kind}有多个未绑定候选，拒绝猜测最新一个：' + ', '.join(map(str, candidates)))
    return None


def read_pipeline(workspace, digests):
    path = workspace / 'pipeline_run.json'
    if not path.is_file():
        return {'path': None, 'present': False, 'failed': False, 'complete_all_stages': False, 'missing_stages': STAGES, 'stages': []}, {}, None
    digests.record(path)
    document = json_file(path)
    latest = {}
    for row in document.get('stages', []):
        if isinstance(row, dict) and row.get('stage'):
            latest[row['stage']] = row
    failed = [name for name, row in latest.items() if row.get('exit_code') not in [0]]
    missing = [name for name in STAGES if name not in latest]
    expected = {}
    for row in latest.values():
        for name, info in (row.get('artifacts') or {}).items():
            if isinstance(info, dict) and info.get('sha256'):
                expected[str(Path(name).resolve())] = info['sha256']
    result = {'path': str(path), 'sha256': digests.values[str(path)], 'present': True, 'failed': bool(failed),
              'failed_stages': failed, 'missing_stages': missing, 'complete_all_stages': not failed and not missing,
              'stages': [{'stage': name, 'exit_code': row.get('exit_code'), 'skipped': bool(row.get('skipped')), 'log': row.get('log'),
                          'artifact_error': row.get('artifact_error')} for name, row in latest.items()]}
    result['timing'] = stage_timings(document, latest)
    return result, expected, document


def ply_header(path):
    lines = []
    with Path(path).open('rb') as stream:
        size = 0
        while size <= 65536:
            line = stream.readline()
            if not line:
                raise ValueError('PLY header截断')
            size += len(line)
            text = line.decode('ascii').strip()
            lines.append(text)
            if text == 'end_header':
                break
        else:
            raise ValueError('PLY header过大')
    if lines[0] != 'ply' or not any(line.startswith('format ') for line in lines):
        raise ValueError('不是PLY数据')
    vertices = [line for line in lines if line.startswith('element vertex ')]
    count = int(vertices[0].split()[-1]) if len(vertices) == 1 else 0
    if count <= 0 or any(not any(line.startswith('property ') and line.endswith(' ' + axis) for line in lines) for axis in ['x', 'y', 'z']):
        raise ValueError('PLY缺少非空XYZ vertex')
    return count


def inspect_hmsg(path, digests):
    details, all_files = {}, []
    for folder in ['floors', 'rooms', 'views', 'objects']:
        files = sorted((path / folder).glob('*.json'))
        if not files:
            raise ValueError('HMSG缺少' + folder + ' JSON')
        details[folder] = len(files)
        for file in files:
            metadata = json_file(file)
            digests.record(file)
            all_files.append(file)
            if folder != 'views':
                ply = file.with_suffix('.ply')
                if not ply.is_file():
                    raise ValueError('HMSG缺少对应PLY：' + str(ply))
                count = ply_header(ply)
                digests.record(ply)
                all_files.append(ply)
                if folder == 'objects':
                    details['object_points_header_count'] = details.get('object_points_header_count', 0) + count
            if folder == 'objects':
                if str(metadata.get('object_id')) != file.stem:
                    raise ValueError('HMSG对象ID/文件名不一致')
                embedding = metadata.get('embedding')
                if not isinstance(embedding, list) or len(embedding) != 1024 or not all(type(v) in [int, float] and math.isfinite(v) for v in embedding) or not any(embedding):
                    raise ValueError('HMSG真实1024特征结构缺失/非有限/全零')
                room = str(metadata.get('room_id'))
                if not (path / 'rooms' / (room + '.json')).is_file():
                    raise ValueError('HMSG对象room引用缺失')
                for view in metadata.get('view_ids', []):
                    if not (path / 'views' / (str(view) + '.json')).is_file():
                        raise ValueError('HMSG对象view引用缺失')
    tree = hashlib.sha256()
    for file in sorted(all_files):
        tree.update((file.relative_to(path).as_posix() + '\0' + digests.record(file) + '\n').encode())
    return {**details, 'file_count': len(all_files), 'sha256_tree': tree.hexdigest(), 'point_finite_check': '由原native validation逐项检查；collector仅核对PLY头/源SHA和JSON结构'}, all_files


def inspect_cg(path):
    with gzip.open(path, 'rb') as stream:
        header = stream.read(16)
    if not header or header[0] != 0x80:
        raise ValueError('CG gzip内容不是原生pickle协议')
    return {'gzip_pickle_header_valid': True, 'deserialized_by_collector': False,
            'numeric_contract_source': '原native validation；collector不对未知pickle执行反序列化'}


def inspect_vlmaps(path):
    with path.open('rb') as stream:
        header = stream.read(8)
    if header != b'\x89HDF\r\n\x1a\n':
        raise ValueError('VLMaps不是HDF5文件')
    return {'hdf5_header_valid': True, 'format': '原生HDF5 .h5df：grid_feat/grid_pos/grid_rgb/cs/pcd_min等',
            'numeric_contract_source': '原native validation；collector不加载模型或改H5'}


def collect(workspace, output=None, *, require_complete=False, check_only=False):
    started_collect = time.monotonic()
    workspace = Path(workspace).expanduser().resolve()
    if not workspace.is_dir():
        raise ValueError('输入workspace不存在：' + str(workspace))
    destination = None if check_only else output_path(output, workspace)
    digests, missing, warnings = Digests(), [], []
    pipeline, expected, pipeline_document = read_pipeline(workspace, digests)
    if pipeline['failed']:
        missing.append('流水线阶段失败/未结束：' + ', '.join(pipeline['failed_stages']))
    validation_path = workspace / 'native_maps/validation.json'
    validation = None
    if validation_path.is_file():
        digests.record(validation_path)
        try:
            validation = json_file(validation_path)
        except Exception as error:
            missing.append('native validation读取失败：' + str(error))
    else:
        missing.append('缺少native_maps/validation.json')
    validation_passed = bool(validation is not None and validation.get('all_passed') is True)
    if validation is not None:
        if not validation_passed:
            missing.append('native validation.all_passed并非true')
        if any(row.get('passed') is not True for row in validation.get('checks', [])):
            validation_passed = False
            missing.append('native validation存在未通过的逐项检查')
    v = validation or {}
    native = workspace / 'native_maps'
    graph = choose([p for p in (native / 'HMSG').rglob('graph_*') if p.is_dir() and (p / 'objects').is_dir()],
                   v.get('HMSG', {}).get('graph'), workspace, 'HMSG graph', missing, warnings)
    cg = choose(list((native / 'ConceptGraphs').rglob('*_post.pkl.gz')), v.get('ConceptGraphs', {}).get('native'), workspace, 'ConceptGraphs *_post.pkl.gz', missing, warnings)
    vlmaps = choose(list((native / 'VLMaps').rglob('*.h5df')), v.get('VLMaps', {}).get('native'), workspace, 'VLMaps .h5df', missing, warnings)
    artifacts = {'hmsg_graph': None, 'conceptgraphs': None, 'vlmaps': None, 'pipeline_archives': [], 'reports': [], 'logs': []}
    graph_files = []
    for key, path, checker, format_name in [('hmsg_graph', graph, None, 'HMSG层次图目录：floors/rooms/views/objects JSON＋对应PLY'),
                                          ('conceptgraphs', cg, inspect_cg, 'ConceptGraphs原生gzip pickle (.pkl.gz)：实例几何、框、图像/文字特征'),
                                          ('vlmaps', vlmaps, inspect_vlmaps, 'VLMaps原生HDF5 (.h5df)：体素位置/语义特征/颜色/权重/栅格标定')]:
        if path is None:
            continue
        try:
            if key == 'hmsg_graph':
                detail, graph_files = inspect_hmsg(path, digests)
                expected_count = v.get('HMSG', {}).get('objects')
                if expected_count is not None and expected_count != detail['objects']:
                    raise ValueError('实际HMSG对象数与native validation不一致')
                artifacts[key] = {'path': str(path), 'format': format_name, 'structural_check_passed': True, **detail, 'archive_path': None}
            else:
                artifacts[key] = {'path': str(path), 'format': format_name, 'sha256': digests.record(path), 'bytes': path.stat().st_size,
                                  'structural_check_passed': True, **checker(path)}
        except Exception as error:
            missing.append(key + '结构/格式检查失败：' + str(error))
            artifacts[key] = {'path': str(path), 'format': format_name, 'structural_check_passed': False, 'error': str(error)}
    report_paths = [p for p in workspace.glob('*.json') if p.is_file()]
    report_paths += [p for p in [validation_path, workspace / 'share/native_package_result.json'] if p.is_file()]
    for path in sorted(set(report_paths)):
        artifacts['reports'].append({'path': str(path), 'sha256': digests.record(path), 'bytes': path.stat().st_size})
    log_paths = list((workspace / 'logs').glob('*.log')) + list(workspace.glob('*.log'))
    for path in sorted(set(log_paths)):
        artifacts['logs'].append({'path': str(path), 'sha256': digests.record(path), 'bytes': path.stat().st_size})
    for path in sorted((workspace / 'share').glob('*native_three_maps.tar.gz')):
        artifacts['pipeline_archives'].append({'path': str(path), 'sha256': digests.record(path), 'bytes': path.stat().st_size, 'kind': '既有流水线三图总包'})
    package_path = workspace / 'share/native_package_result.json'
    if package_path.is_file():
        package = json_file(package_path)
        declared_archive = Path(package.get('archive', '')).resolve()
        actual = next((row for row in artifacts['pipeline_archives'] if row['path'] == str(declared_archive)), None)
        if not actual or package.get('sha256') != actual['sha256'] or package.get('bytes') != actual['bytes']:
            missing.append('share/native_package_result的总包路径/字节/SHA与实际产物不一致')
    if require_complete:
        if not pipeline['present']:
            missing.append('完成收集要求pipeline_run.json')
        elif not pipeline['complete_all_stages']:
            missing.append('完成收集缺少成功all阶段：' + ', '.join(pipeline.get('missing_stages', [])))
        if not artifacts['pipeline_archives']:
            missing.append('完成收集缺少share原生三图总tar.gz')
        if not package_path.is_file():
            missing.append('完成收集缺少share/native_package_result.json')
    absent_stage_artifacts = [path for path in expected if under(Path(path), workspace) and not Path(path).is_file()]
    if absent_stage_artifacts and require_complete:
        missing.append('最近成功流水线阶段的部分声明产物已经缺失')
    mismatches = [{'path': path, 'expected_sha256': expected[path], 'actual_sha256': digest} for path, digest in digests.values.items() if path in expected and digest != expected[path]]
    if mismatches:
        missing.append('当前产物SHA与最近成功流水线阶段记录不一致')
    all_maps = all(artifacts[key] and artifacts[key].get('structural_check_passed') is True for key in ['hmsg_graph', 'conceptgraphs', 'vlmaps'])
    success = bool(all_maps and validation_passed and not missing and not pipeline['failed'])
    status = 'complete' if success else ('failed' if pipeline['failed'] or not validation_passed or mismatches else 'incomplete')
    result = {'schema_version': 'native_mapping_collection_v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'generated_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'generated_at_timezone': 'Asia/Shanghai',
              'workspace': str(workspace), 'output_dir': str(destination) if destination else None, 'check_only': check_only,
              'require_complete': require_complete, 'success': success, 'status': status, 'missing': missing, 'warnings': warnings,
              'artifacts': artifacts, 'pipeline': pipeline, 'pipeline_sha_mismatches': mismatches,
              'pipeline_declared_artifacts_now_missing': absent_stage_artifacts,
              'native_validation': {'path': str(validation_path) if validation_path.is_file() else None, 'all_passed': validation_passed,
                                    'reported_counts': {key: v.get(key) for key in ['HMSG', 'ConceptGraphs', 'VLMaps']},
                                    'scope': '原报告＋结构/源SHA合同检查；不是独立语义准确率/真机证明'},
              'dataset': v.get('dataset'), 'clean_source_root': (pipeline_document or {}).get('clean_source_root'),
              'stage_timing_s': pipeline.get('timing', {}).get('stage_timing_s', {}),
              'stage_timing_details': pipeline.get('timing', {}).get('stage_timing_details', []),
              'total_pipeline_stage_s': pipeline.get('timing', {}).get('total_pipeline_stage_s'),
              'native_frame_count': v.get('sample_frames'), 'native_frame_count_source': 'native validation.sample_frames（准备样本帧，非每图实际处理帧）',
              'map_observation_frames': {key: v.get(key, {}).get('mapping_observations', v.get(key, {}).get('mapped_observations')) for key in ['HMSG', 'ConceptGraphs', 'VLMaps']},
              'voxel_count': (v.get('VLMaps', {}).get('feature_shape') or [None])[0], 'input_volume': input_volume(workspace),
              'performance_note_zh': '阶段耗时/输入字节/帧数/体素数用于解释规模；仅凭这些不判定4090、磁盘或CPU是绝对瓶颈，需producer实际资源采样与整体计时。',
              'source_hashes': digests.values, 'source_files_modified': False,
              'claims': {'models_trained': False, 'independent_semantic_gt': False, 'complete_physical_surfaces': False,
                         'automatic_full_completion': False, 'robot_navigation_success': False, 'robot_or_service_operation_performed': False},
              'notes_zh': ['仅收集这次workspace已生成的原生HMSG/ConceptGraphs/VLMaps，不补造缺失图或特征。',
                           'HMSG独立tar.gz保留完整JSON/PLY图；原文件不改。CG与VLMaps列实际绝对路径和SHA。',
                           '模型没有在本步骤训练；结构合同通过不等于语义识别全部正确，物体表面可能缺失，自动完整补全尚未实现。',
                           '流水线失败、校验失败或必要产物缺失会写success=false并返回非零退出码。']}
    if destination:
        destination.mkdir(parents=True)
        if graph and graph_files and artifacts['hmsg_graph'].get('structural_check_passed'):
            archive = destination / ('HMSG_' + graph.name + '.tar.gz')
            graph_manifest = {'source_graph': str(graph), 'graph_sha256_tree': artifacts['hmsg_graph']['sha256_tree'],
                              'files': {file.relative_to(graph).as_posix(): digests.record(file) for file in sorted(graph_files)},
                              'semantic_accuracy_independently_verified': False, 'complete_surfaces': False, 'models_trained': False}
            with tarfile.open(archive, 'w:gz', compresslevel=1) as opened:
                for file in sorted(graph_files):
                    opened.add(file, arcname=graph.name + '/' + file.relative_to(graph).as_posix(), recursive=False)
                content = (json.dumps(graph_manifest, ensure_ascii=False, indent=2) + '\n').encode()
                info = tarfile.TarInfo('HMSG_GRAPH_MANIFEST.json');info.size=len(content);info.mode=0o644
                opened.addfile(info, io.BytesIO(content))
            artifacts['hmsg_graph'].update(archive_path=str(archive), archive_sha256=sha256(archive), archive_bytes=archive.stat().st_size)
            altered = [str(file) for file in graph_files if sha256(file) != digests.values[str(file)]]
            if altered:
                result['missing'].append('HMSG打包期间源图文件发生变化：' + ', '.join(altered))
                result['success'], result['status'] = False, 'failed'
        result['collection_elapsed_s'] = time.monotonic() - started_collect
        result['generated_at'] = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
        (destination / 'MAPPING_RESULT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    else:
        result['collection_elapsed_s'] = time.monotonic() - started_collect
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--output')
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if not args.check_only and not args.output:
        parser.error('普通收集需--output新目录；--check-only不写文件')
    failure_destination = None
    try:
        if not args.check_only:
            failure_destination = output_path(args.output, Path(args.workspace).expanduser().resolve())
        result = collect(args.workspace, args.output, require_complete=args.require_complete, check_only=args.check_only)
    except Exception as error:
        result = {'schema_version': 'native_mapping_collection_v1', 'workspace': str(Path(args.workspace).expanduser().resolve()),
                  'generated_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'generated_at_timezone': 'Asia/Shanghai',
                  'success': False, 'status': 'failed', 'missing': [str(error)], 'source_files_modified': False}
        if failure_destination:
            failure_destination.mkdir(parents=True, exist_ok=True)
            (failure_destination / 'MAPPING_RESULT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result['success'] else 2)


if __name__ == '__main__':
    main()
