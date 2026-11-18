#!/usr/bin/env python3
"""Optional YAML finite-vocabulary preparation and native-map label locking.

This classifies genuine native image/voxel features using real CLIP text
features, preserving native geometry/features in independent copies. Class
candidates do not resolve physical annotation IDs or authorize navigation.
"""
from __future__ import annotations
import argparse
import copy
import gc
import gzip
import hashlib
import json
import math
from pathlib import Path
import pickle
import shutil
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
PACKAGE = Path(__file__).resolve().parent.parent / 'package'
EXPECTED_MODEL_SHA = {
    'H':'0084e75319a50ad85ef45377bad5bc38f2f58824459eb690048d51c9f8863be5',
    'B':'40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af',
}
GLOSSARY = {
    '红色消防拴': 'fire hydrant', '红色消防栓': 'fire hydrant', '消防栓': 'fire hydrant',
    '手提式灭火器': 'fire extinguisher', '灭火器': 'fire extinguisher',
    '消防柜': 'fire equipment cabinet', '黄色安全帽': 'yellow hard hat', '黄色消防帽':'yellow hard hat', '安全帽': 'hard hat',
    '垃圾桶': 'trash can', '黑色工具箱': 'black toolbox', '黑色手提箱':'black toolbox', '工具箱': 'toolbox',
    '橙色路锥': 'orange traffic cone', '锥筒': 'traffic cone', '路锥': 'traffic cone',
    '配电箱': 'electrical distribution panel', '动力柜': 'electrical power cabinet',
    '变压器台区': 'transformer', '充电桩': 'charging station', '压力表': 'pressure gauge',
    '水阀': 'water valve', '假人': 'mannequin', '彩色积木拱架': 'colorful block arch',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def ordinary_yaml(path):
    class Loader(yaml.SafeLoader):
        pass
    def mapping(loader, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in result:
                raise ValueError('YAML重复键：' + str(key))
            result[key] = loader.construct_object(value_node, deep=deep)
        return result
    Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    return yaml.load(Path(path).read_text(encoding='utf-8-sig'), Loader=Loader)


def vocabulary(yaml_path, translations=None):
    yaml_path = Path(yaml_path).expanduser().resolve()
    source = ordinary_yaml(yaml_path)
    if not isinstance(source, dict) or type(source.get('version')) is not int or source['version'] != 3:
        raise ValueError('标签YAML必须是version:3 annotations文档')
    rows = source.get('annotations')
    if not isinstance(rows, list) or not rows or len(rows) > 10000:
        raise ValueError('YAML annotations必须是有界非空列表')
    overrides = json.loads(Path(translations).read_text()) if translations else {}
    if not isinstance(overrides, dict):
        raise ValueError('明确英译JSON必须是名字→英文类别字典')
    groups, seen, annotations = {}, set(), []
    for row in rows:
        if not isinstance(row, dict) or type(row.get('id')) not in [int, str] or str(row['id']) in seen:
            raise ValueError('YAML ID缺失、类型不合法或重复；不按数组顺序生成ID')
        seen.add(str(row['id']))
        name = row.get('name')
        if not isinstance(name, str) or not name.strip():
            raise ValueError('YAML名字必须明确非空')
        for field, size in [('robot_pos', 3), ('robot_rotation', 4), ('camera_pos', 3), ('camera_rotation', 4)]:
            if field in row and row[field] is not None:
                value = row[field]
                if not isinstance(value, list) or len(value) != size or not all(type(v) in [int, float] and math.isfinite(v) for v in value):
                    raise ValueError('YAML原始位姿字段不是有限向量：' + field)
        key = name.strip()
        english = overrides.get(name, overrides.get(key, GLOSSARY.get(key)))
        method = 'explicit_translation_json' if name in overrides or key in overrides else 'explicit_project_canonical_glossary'
        if english is None and key.isascii():
            english, method = key, 'explicit_English_label_identity'
        if not isinstance(english, str) or not english.strip() or not english.isascii():
            raise ValueError('类别缺少明确英文映射：' + name + '；提供--translations JSON，禁止猜译')
        english = ' '.join(english.lower().split())
        group = groups.setdefault(english, {'category_index': len(groups), 'english': english, 'yaml_names': [], 'yaml_annotation_ids': [], 'translation_methods': []})
        if name not in group['yaml_names']:
            group['yaml_names'].append(name)
        group['yaml_annotation_ids'].append(row['id'])
        group['translation_methods'].append(method)
        annotations.append({'id': row['id'], 'name': name, 'category_index': group['category_index'],
                            **{field: copy.deepcopy(row[field]) for field in ['robot_pos', 'robot_rotation', 'camera_pos', 'camera_rotation'] if field in row},
                            'pose_role': 'Original read-only annotation metadata; never classifier object coordinates or camera calibration'})
    return {'schema_version': 'yaml_finite_vocabulary_v1', 'yaml_path': str(yaml_path), 'yaml_sha256': sha(yaml_path),
            'map_name': source.get('map'), 'yaml_version': 3, 'annotations': annotations, 'categories': list(groups.values()),
            'detector_classes': list(groups), 'annotation_count': len(rows), 'category_count': len(groups),
            'same_name_multiple_instances': 'Shared class only; no native object is assigned a physical annotation ID by category classification',
            'navigation_authorized': False, 'models_trained': False, 'pose_or_camera_placeholders_used_as_geometry': False}


def new_output(path, workspace):
    workspace = Path(workspace).expanduser().resolve()
    output = Path(path).expanduser().resolve()
    if output == workspace:
        raise ValueError('不能覆盖workspace根；指定新独立目录')
    if output.exists():
        raise ValueError('输出版本已存在；禁止覆写原生/既有标签图')
    if workspace / 'native_maps' == output or workspace / 'native_maps' in output.parents:
        raise ValueError('标签锁定输出不得放入原生native_maps目录')
    output.mkdir(parents=True)
    return output


def prepare(yaml_path, workspace, config_path, output=None, translations=None):
    vocab = vocabulary(yaml_path, translations)
    workspace = Path(workspace).expanduser().resolve()
    preparation_path = workspace / 'LABEL_PREPARATION.json'
    if preparation_path.exists():
        raise ValueError('已存在LABEL_PREPARATION，使用新workspace避免覆盖冻结原标签')
    config_path = Path(config_path).expanduser().resolve()
    config = ordinary_yaml(config_path)
    if not isinstance(config, dict):
        raise ValueError('原建图config必须是字典')
    out = new_output(output or workspace / 'label_alignment/preparation', workspace)
    snapshot = out / 'labels.source.yaml'
    shutil.copy2(yaml_path, snapshot)
    original_sha = sha(config_path)
    changed = copy.deepcopy(config)
    changed.setdefault('conceptgraphs', {})['classes'] = vocab['detector_classes']
    override = out / 'mapping_config.label_vocabulary.yaml'
    override.write_text(yaml.safe_dump(changed, allow_unicode=True, sort_keys=False))
    vocab.update(yaml_snapshot=str(snapshot), config_path=str(override), original_config_path=str(config_path), original_config_sha256=original_sha,
                 config_override_scope='Only conceptgraphs.classes; HMSG/VLMaps class locking occurs after complete native mapping')
    if sha(snapshot) != vocab['yaml_sha256'] or sha(yaml_path) != vocab['yaml_sha256'] or sha(config_path) != original_sha:
        raise RuntimeError('原YAML/config或冻结字节发生变化')
    write(out / 'LABEL_VOCABULARY.json', vocab)
    result = {'success': True, 'phase': 'prepare', 'detector_classes': vocab['detector_classes'], 'yaml_snapshot': str(snapshot),
            'yaml_sha256': vocab['yaml_sha256'], 'config_path': str(override), 'vocabulary_path': str(out / 'LABEL_VOCABULARY.json'),
            'source_ids_names_poses_and_bytes_preserved': True, 'models_trained': False}
    write(preparation_path, result)
    return result


def normalized(values, dimension):
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != dimension:
        raise ValueError(f'真实feature维度必须N×{dimension}')
    valid = np.isfinite(array).all(1)
    norms = np.linalg.norm(np.where(np.isfinite(array), array, 0), axis=1)
    valid &= norms > 1e-8
    result = np.zeros_like(array)
    result[valid] = array[valid] / norms[valid, None]
    return result, valid


def classify(image_features, text_features, min_score, min_margin):
    dimension = text_features.shape[1]
    images, valid = normalized(image_features, dimension)
    text, text_valid = normalized(text_features, dimension)
    if not text_valid.all():
        raise ValueError('文本cache含非有限/全零feature，禁止伪造或兜底')
    similarity = images @ text.T
    best = similarity.argmax(1)
    score = similarity[np.arange(len(images)), best]
    margin = score - np.partition(similarity, -2, axis=1)[:, -2] if len(text) > 1 else np.full(len(images), np.nan)
    accepted = valid & (score >= min_score)
    if len(text) > 1:
        accepted &= margin >= min_margin
    labels = np.where(accepted, best, -1).astype(np.int32)
    return labels, score.astype(np.float32), margin.astype(np.float32), valid


def imported_text(vocab, path, provenance_path=None):
    path = Path(path).resolve()
    provenance_path = Path(provenance_path) if provenance_path else path.parent / 'PROVENANCE.json'
    provenance = json.loads(provenance_path.read_text())
    if provenance.get('feature_file_sha256') != sha(path):
        raise ValueError('真实text cache SHA与provenance不一致')
    english = [cat['english'] for cat in vocab['categories']]
    models = provenance['models']
    if models['H']['text_input'] != models['B']['text_input']:
        raise ValueError('H/B cache英文类别不同')
    ids = list(dict.fromkeys(str(row['id']) for row in provenance['source_views']))
    labels = models['H']['text_input']
    if len(ids) != len(labels):
        raise ValueError('已有真实cache缺少明确text label/ID排序来源')
    lookup = dict(zip(labels, ids))
    with np.load(path, allow_pickle=False) as data:
        arrays = []
        for model, dimension in [('H', 1024), ('B', 512)]:
            rows = []
            for label in english:
                if label not in lookup:
                    raise ValueError('cache没有独立编码该英文类别：' + label + '；需--encode-text')
                vector = data[f'text_{model}_{lookup[label]}']
                if vector.shape != (dimension,):
                    raise ValueError('实际缓存text feature维度错误')
                rows.append(vector)
            arrays.append(np.asarray(rows, dtype=np.float32))
    return arrays[0], arrays[1], {'operation': 'Actual separately encoded text features imported by exact English description; annotation IDs are not matched to native instances',
        'source_feature_path': str(path), 'source_feature_sha256': sha(path), 'source_provenance_sha256': sha(provenance_path), 'models': models,
        'device': 'CPU_cache_import_no_model_inference', 'models_trained': False}


def encode_text(vocab, model_root, device):
    import torch
    import open_clip
    import clip
    root = Path(model_root).resolve()
    hp, bp = root / 'clip/ViT-H-14.safetensors', root / 'clip/ViT-B-32.pt'
    if not hp.is_file() or not bp.is_file():
        raise ValueError('本地H14/B32权重缺失；不下载、不制造text feature')
    labels = [cat['english'] for cat in vocab['categories']]
    target = 'cuda' if device == 'auto' and torch.cuda.is_available() else ('cpu' if device == 'auto' else device)
    outputs = []
    for kind, path in [('H', hp), ('B', bp)]:
        if kind == 'H':
            model, _, _ = open_clip.create_model_and_transforms('ViT-H-14', pretrained=str(path), device=target)
            tokenize = open_clip.get_tokenizer('ViT-H-14')
        else:
            model, _ = clip.load(str(path), device=target, jit=False)
            tokenize = clip.tokenize
        model.eval()
        vectors = []
        with torch.inference_mode():
            for start in range(0, len(labels), 32):
                vectors.append(model.encode_text(tokenize(labels[start:start+32]).to(target)).float().cpu().numpy())
        outputs.append(np.concatenate(vectors))
        del model
        gc.collect()
        if target.startswith('cuda'):
            torch.cuda.empty_cache()
    return outputs[0], outputs[1], {'operation': 'Actual local ViT-H-14 and OpenAI ViT-B/32 plain text encoding; sequential model loads, inference_mode, no training',
        'device': target, 'models': {'H': {'encoder': 'ViT-H-14/laion2b_s32b_b79k', 'path': str(hp), 'sha256': sha(hp), 'dimension': 1024}, 'B': {'encoder':'OpenAI ViT-B/32', 'path': str(bp), 'sha256': sha(bp), 'dimension': 512}}, 'models_trained': False}


def read_text(vocab, cache=None, import_provenance=None, encode=False, model_root=None, device='auto'):
    if cache:
        cache = Path(cache).resolve()
        metadata = cache.parent / 'TEXT_FEATURES.json'
        if metadata.is_file():
            meta = json.loads(metadata.read_text())
            if meta['feature_sha256'] != sha(cache) or meta['english'] != [cat['english'] for cat in vocab['categories']]:
                raise ValueError('锁定text cache SHA/明确类别不匹配')
            with np.load(cache, allow_pickle=False) as data:
                h, b = data['text_H'].copy(), data['text_B'].copy()
            provenance = meta['provenance']
        else:
            h, b, provenance = imported_text(vocab, cache, import_provenance)
    elif encode:
        h, b, provenance = encode_text(vocab, model_root or PACKAGE / 'data/models/semantic_mapping', device)
    else:
        raise ValueError('无真实text cache：显式指定--text-cache或由操作者选择--encode-text；不自动开GPU/制造feature')
    for array, dim in [(h, 1024), (b, 512)]:
        norm, valid = normalized(array, dim)
        if len(array) != vocab['category_count'] or not valid.all():
            raise ValueError('文本cache真实类别数/维度/有限非零合同不通过')
    hm=provenance.get('models',{}).get('H',{});bm=provenance.get('models',{}).get('B',{})
    if 'ViT-H-14' not in hm.get('encoder','') or not any(value in bm.get('encoder','') for value in ['ViT-B/32','ViT-B-32']):
        raise ValueError('文本cache没有明确H14/B32 encoder身份，维度相同不代表feature space兼容')
    for key,model in [('H',hm),('B',bm)]:
        digest=model.get('checkpoint_sha256',model.get('sha256'))
        if digest!=EXPECTED_MODEL_SHA[key]:
            raise ValueError('text cache权重SHA不匹配本部署已声明的原生CLIP空间：'+key)
    return h, b, provenance


def feature_space_contract(native_validation):
    if native_validation.get('HMSG',{}).get('instance_feature_dimension')!=1024 or native_validation.get('ConceptGraphs',{}).get('instance_feature_dimension')!=1024:
        raise ValueError('原native validation未声明HMSG/CG 1024维feature合同')
    files={'HMSG':PACKAGE/'code/robot_mapping/run_native_hmsg.py',
           'ConceptGraphs':PACKAGE/'code/third_party/concept-graphs/conceptgraph/scripts/generate_gsa_results.py',
           'VLMaps':PACKAGE/'code/third_party/vlmaps/vlmaps/lseg/modules/models/lseg_vit.py'}
    markers={'HMSG':'ViT-H-14','ConceptGraphs':'ViT-H-14','VLMaps':'ViT-B/32'}
    for key,path in files.items():
        if not path.is_file() or markers[key] not in path.read_text():
            raise ValueError('当前原生producer无法核实期望CLIP空间：'+key)
    return {'HMSG':'ViT-H-14/1024','ConceptGraphs':'ViT-H-14/1024','VLMaps':'OpenAI ViT-B/32/512 LSeg feature space',
            'producer_files':{key:{'path':str(path),'sha256':sha(path)} for key,path in files.items()},
            'basis':'原native validation维度＋已部署producer明示encoder合同；不把同维度或数组顺序当作实例ID/独立语义验证'}


def candidate_record(index, score, margin, valid, vocab, threshold, margin_threshold, original):
    category = vocab['categories'][int(index)] if index >= 0 else None
    return {'schema_version': 'finite_yaml_category_candidate_v1', 'category_index': int(index),
            'yaml_label_names': category['yaml_names'] if category else [], 'english': category['english'] if category else None,
            'cosine_score': float(score) if valid else None, 'top1_top2_margin': float(margin) if valid and math.isfinite(float(margin)) else None,
            'min_score': threshold, 'min_margin': margin_threshold, 'score_kind': 'Actual normalized image/voxel-to-text cosine, not calibrated detection probability',
            'status': 'category_candidate_unreviewed' if index >= 0 else 'unknown_or_insufficient_feature_support',
            'original_native_label': original, 'source_yaml_sha256': vocab['yaml_sha256'],
            'physical_annotation_id_binding': None, 'physical_instance_verified': False, 'human_review_status': 'pending',
            'geometry_or_image_features_changed': False, 'robot_pos_camera_placeholders_used': False, 'navigation_authorized': False}


def source_maps(workspace):
    workspace = Path(workspace).resolve()
    path = workspace / 'native_maps/validation.json'
    v = json.loads(path.read_text())
    if v.get('all_passed') is not True or any(row.get('passed') is not True for row in v.get('checks', [])):
        raise ValueError('三原生native validation未通过，禁止标记标签锁定成功')
    sources = {'hmsg': Path(v['HMSG']['graph']).resolve(), 'conceptgraphs': Path(v['ConceptGraphs']['native']).resolve(), 'vlmaps': Path(v['VLMaps']['native']).resolve()}
    if any(workspace not in item.parents for item in sources.values()):
        raise ValueError('native validation指向本workspace外的旧map，拒绝混场景')
    return sources, v, path


def equal(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) == type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return type(a) == type(b) and a == b


def postprocess(yaml_path, workspace, output, *, translations=None, text_cache=None, import_provenance=None, encode=False,
                model_root=None, device='auto', min_score_H=.25, min_score_B=.20, min_margin=.02, batch_rows=8192):
    import h5py
    if not -1 <= min_score_H <= 1 or not -1 <= min_score_B <= 1 or not 0 <= min_margin <= 2 or not 1 <= batch_rows <= 65536:
        raise ValueError('score/margin/batch参数不合法')
    vocab = vocabulary(yaml_path, translations)
    sources, native_validation, val_path = source_maps(workspace)
    space_contract=feature_space_contract(native_validation)
    htext, btext, text_provenance = read_text(vocab, text_cache, import_provenance, encode, model_root, device)
    with gzip.open(sources['conceptgraphs'], 'rb') as stream:
        source_cg = pickle.load(stream)  # Only trusted local, validated native pipeline artifacts.
    cg_features = np.stack([np.asarray(obj['clip_ft']) for obj in source_cg['objects']])
    cg_index, cg_score, cg_margin, cg_valid = classify(cg_features, htext, min_score_H, min_margin)
    if not cg_valid.all():
        raise ValueError('CG原生真实image feature维度/有限非零合同不通过')
    hfiles = sorted((sources['hmsg'] / 'objects').glob('*.json'))
    hobjects = [json.loads(path.read_text()) for path in hfiles]
    h_index, h_score, h_margin, h_valid = classify(np.stack([obj['embedding'] for obj in hobjects]), htext, min_score_H, min_margin)
    if not h_valid.all():
        raise ValueError('HMSG原生真实image feature维度/有限非零合同不通过')
    with h5py.File(sources['vlmaps'], 'r') as stream:
        if stream['grid_feat'].ndim != 2 or stream['grid_feat'].shape[1] != 512:
            raise ValueError('VLMaps原生真实grid_feat不是N×512')
    hashes = {str(Path(yaml_path).resolve()): vocab['yaml_sha256'], str(val_path): sha(val_path), str(sources['conceptgraphs']): sha(sources['conceptgraphs']), str(sources['vlmaps']): sha(sources['vlmaps'])}
    for path in sources['hmsg'].rglob('*'):
        if path.is_file():
            hashes[str(path)] = sha(path)
    out = new_output(output, workspace)
    snapshot = out / 'labels.source.yaml';shutil.copy2(yaml_path, snapshot)
    np.savez_compressed(out / 'text_features.npz', text_H=htext, text_B=btext, english=np.asarray(vocab['detector_classes']))
    write(out / 'TEXT_FEATURES.json', {'english': vocab['detector_classes'], 'yaml_sha256': vocab['yaml_sha256'], 'feature_sha256': sha(out/'text_features.npz'), 'provenance': text_provenance})
    write(out / 'LABEL_VOCABULARY.json', vocab)
    hdest = out / 'HMSG' / sources['hmsg'].name
    hdest.parent.mkdir();shutil.copytree(sources['hmsg'], hdest)
    counts = {name: np.zeros(vocab['category_count'], dtype=np.int64) for name in ['HMSG', 'ConceptGraphs', 'VLMaps']}
    unknown = {'HMSG': int(np.sum(h_index < 0)), 'ConceptGraphs': int(np.sum(cg_index < 0)), 'VLMaps': 0}
    parity = []
    h_names = {}
    for i, (path, original) in enumerate(zip(hfiles, hobjects)):
        record = candidate_record(h_index[i], h_score[i], h_margin[i], h_valid[i], vocab, min_score_H, min_margin, original['name'])
        changed = copy.deepcopy(original)
        changed['name'] = record['yaml_label_names'][0] if h_index[i] >= 0 else 'unknown'
        changed['label_alignment'] = record
        h_names[str(original['object_id'])] = changed['name']
        if h_index[i] >= 0:
            counts['HMSG'][h_index[i]] += 1
        write(hdest / 'objects' / path.name, changed)
        print(f'label objects {i+1}/{len(hobjects)} HMSG',flush=True)
        parity.append({'map': 'HMSG', 'object_id': str(original['object_id']), 'all_fields_except_name_and_label_alignment_preserved': equal(original, {k:v for k,v in changed.items() if k != 'label_alignment'} | {'name': original['name']})})
    for path in (hdest / 'views').glob('*.json'):
        view = json.loads(path.read_text());view['original_native_text_discription'] = view.get('text_discription')
        view['text_discription'] = [h_names.get(str(ident), 'unknown') for ident in view.get('object_ids', [])]
        view['label_alignment_scope'] = 'Class candidates only; object IDs/geometry/features are the original native IDs, not YAML annotation bindings'
        write(path, view)
    changed_cg = copy.deepcopy(source_cg)
    for i, (original, changed) in enumerate(zip(source_cg['objects'], changed_cg['objects'])):
        record = candidate_record(cg_index[i], cg_score[i], cg_margin[i], cg_valid[i], vocab, min_score_H, min_margin, original.get('class_name'))
        n = len(original.get('class_name', []))
        changed['class_name'] = [record['yaml_label_names'][0] if cg_index[i] >= 0 else 'unknown'] * n
        changed['class_id'] = [int(cg_index[i]) if cg_index[i] >= 0 else vocab['category_count']] * len(original.get('class_id', []))
        changed['label_alignment'] = record
        if cg_index[i] >= 0:
            counts['ConceptGraphs'][cg_index[i]] += 1
        excluded = {'class_name', 'class_id', 'label_alignment'}
        parity.append({'map': 'ConceptGraphs', 'object_index': i, 'all_geometry_image_text_features_colors_boxes_masks_and_other_fields_preserved': equal({k:v for k,v in original.items() if k not in excluded},{k:v for k,v in changed.items() if k not in excluded})})
        print(f'label objects {i+1}/{len(source_cg["objects"])} ConceptGraphs',flush=True)
    changed_cg['original_native_class_names'] = source_cg.get('class_names')
    changed_cg['class_names'] = [cat['yaml_names'][0] for cat in vocab['categories']] + ['unknown']
    changed_cg['class_names_english'] = vocab['detector_classes'] + ['unknown']
    cgdest = out / 'ConceptGraphs' / (sources['conceptgraphs'].name.replace('_post.pkl.gz','_label_locked_post.pkl.gz'))
    cgdest.parent.mkdir()
    with gzip.open(cgdest, 'wb') as stream:
        pickle.dump(changed_cg, stream, protocol=4)
    vdest = out / 'VLMaps/vlmaps_label_locked.h5df';vdest.parent.mkdir();shutil.copy2(sources['vlmaps'], vdest)
    with h5py.File(sources['vlmaps'], 'r') as original, h5py.File(vdest, 'r+') as target:
        reserved = ['locked_label_index', 'locked_label_score', 'locked_label_margin', 'label_names', 'label_names_en', 'label_yaml_sha256', 'label_yaml_source_bytes', 'locked_text_features_B']
        if any(name in target for name in reserved):
            raise ValueError('输入H5已有locked dataset；不覆盖已有标签版本')
        n = target['grid_feat'].shape[0]
        labels = target.create_dataset('locked_label_index',(n,),dtype='i4',compression='gzip')
        scores = target.create_dataset('locked_label_score',(n,),dtype='f4',compression='gzip')
        margins = target.create_dataset('locked_label_margin',(n,),dtype='f4',compression='gzip')
        for start in range(0,n,batch_rows):
            end = min(n,start+batch_rows)
            idx, score, margin, valid = classify(original['grid_feat'][start:end],btext,min_score_B,min_margin)
            labels[start:end],scores[start:end],margins[start:end] = idx,score, np.where(np.isfinite(margin),margin,0)
            counts['VLMaps'] += np.bincount(idx[idx>=0],minlength=vocab['category_count'])
            unknown['VLMaps'] += int(np.sum(idx<0))
            print(f'label voxels {end}/{n}',flush=True)
        strings=h5py.string_dtype('utf-8')
        target.create_dataset('label_names',data=np.asarray([cat['yaml_names'][0] for cat in vocab['categories']],dtype=object),dtype=strings)
        target.create_dataset('label_names_en',data=np.asarray(vocab['detector_classes'],dtype=object),dtype=strings)
        target.create_dataset('label_yaml_sha256',data=vocab['yaml_sha256'],dtype=strings)
        target.create_dataset('label_yaml_source_bytes',data=np.frombuffer(Path(yaml_path).read_bytes(),dtype=np.uint8))
        target.create_dataset('locked_text_features_B',data=btext)
        target.attrs['label_alignment_provenance'] = json.dumps({'categories':vocab['categories'],'text_features':text_provenance,'min_score':min_score_B,'min_margin':min_margin,'unknown_index':-1,'physical_annotation_bindings':False,'human_review':'pending','models_trained':False},ensure_ascii=False)
        for key in original:
            if original[key].shape:
                same = all(np.array_equal(original[key][s:min(s+batch_rows,len(original[key]))],target[key][s:min(s+batch_rows,len(original[key]))]) for s in range(0,len(original[key]),batch_rows))
            else:
                same = np.array_equal(original[key][()],target[key][()])
            parity.append({'map':'VLMaps','original_dataset':key,'original_values_and_shape_preserved':same})
    if any(not list(row.values())[-1] for row in parity):
        raise RuntimeError('原生geometry/feature保全合同失败')
    if sha(snapshot)!=vocab['yaml_sha256'] or any(sha(path)!=digest for path,digest in hashes.items()):
        raise RuntimeError('原YAML或原生输入文件被改动')
    target_status=[]
    for row in vocab['annotations']:
        cid=row['category_index'];support={name:int(value[cid]) for name,value in counts.items()}
        target_status.append({'annotation_id':row['id'],'source_name':row['name'],'class_candidate_counts':support,
                              'status':'有类别候选，未绑定此物理实例ID' if any(support.values()) else '未检出有支持类别候选',
                              'instance_binding_verified':False,'robot_pos_used_as_object_center':False,'navigation_authorized':False})
    paths={'hmsg':str(hdest),'conceptgraphs':str(cgdest),'vlmaps':str(vdest),'text_features':str(out/'text_features.npz'),'yaml_snapshot':str(snapshot),'report':str(out/'LABEL_ALIGNMENT.json')}
    result={'success':True,'phase':'postprocess','generated_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'artifact_paths':paths,
            'yaml_sha256':vocab['yaml_sha256'],'source_hashes':hashes,'source_yaml_ids_names_poses_and_bytes_preserved':True,
            'categories':vocab['categories'],'category_counts':{name:value.tolist() for name,value in counts.items()},'unknown_counts':unknown,
            'yaml_target_status':target_status,'parity_checks':parity,'text_provenance':text_provenance,
            'artifacts':{key:{'path':paths[key],'format':fmt} for key,fmt in [('hmsg','HMSG JSON+PLY graph directory'),('conceptgraphs','ConceptGraphs native .pkl.gz'),('vlmaps','VLMaps native .h5df with additive locked datasets')]},
            'missing_target_categories':[vocab['categories'][i] for i in range(vocab['category_count']) if all(value[i]==0 for value in counts.values())],
            'feature_space_contract':space_contract,
            'claims':{'models_trained':False,'physical_annotation_bindings_verified':False,'independent_semantic_precision':False,'complete_surfaces':False,'original_native_modified':False,'robot_pos_as_object_center':False,'navigation_authorized':False},
            'original_native_maps_modified':False,'models_trained':False,'physical_annotation_bindings_created':0,
            'independent_semantic_accuracy_verified':False,'complete_surfaces':False,'navigation_authorized':False,
            'score_scope':'有限词表真实feature分类候选；未知阈值未独立标定，不是实例ID定位或百分百物体正确性。'}
    write(paths['report'],result)
    write(out/'LABEL_ALIGNMENT_RESULT.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--yaml',required=True);p.add_argument('--workspace',required=True);p.add_argument('--config',required=True);p.add_argument('--output');p.add_argument('--translations')
    p=sub.add_parser('postprocess');p.add_argument('--yaml',required=True);p.add_argument('--workspace',required=True);p.add_argument('--output',required=True);p.add_argument('--translations');p.add_argument('--text-cache');p.add_argument('--import-provenance');p.add_argument('--encode-text',action='store_true');p.add_argument('--model-root');p.add_argument('--device',default='auto');p.add_argument('--min-score-h',type=float,default=.25);p.add_argument('--min-score-b',type=float,default=.20);p.add_argument('--min-margin',type=float,default=.02);p.add_argument('--batch-rows',type=int,default=8192)
    args=parser.parse_args()
    existed_before=Path(args.output).expanduser().exists() if getattr(args,'output',None) else False
    try:
        if args.command=='prepare':result=prepare(args.yaml,args.workspace,args.config,args.output,args.translations)
        else:result=postprocess(args.yaml,args.workspace,args.output,translations=args.translations,text_cache=args.text_cache,import_provenance=args.import_provenance,encode=args.encode_text,model_root=args.model_root,device=args.device,min_score_H=args.min_score_h,min_score_B=args.min_score_b,min_margin=args.min_margin,batch_rows=args.batch_rows)
    except Exception as error:
        result={'success':False,'phase':args.command,'error':str(error),'artifact_paths':{},'original_native_maps_modified':False,'models_trained':False,'navigation_authorized':False}
        if getattr(args,'output',None):
            out=Path(args.output).expanduser().resolve()
            if not existed_before and out!=Path(args.workspace).resolve():
                result.update(artifacts={},missing_target_categories=[],claims={'models_trained':False,'physical_annotation_bindings_verified':False,'original_native_modified':False,'navigation_authorized':False})
                try:
                    if not out.exists():new_output(out,args.workspace)
                    write(out/'LABEL_ALIGNMENT.json',result)
                except Exception:
                    pass
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
    raise SystemExit(0 if result['success'] else 2)


if __name__=='__main__':main()
