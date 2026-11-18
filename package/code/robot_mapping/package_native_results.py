"""Package verified native maps and explicit clean-source provenance."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import hashlib
import tarfile
import re
import run_config as rc

ROOT=rc.WORKSPACE
validation=json.loads((ROOT/'native_maps/validation.json').read_text())
assert validation['all_passed'], 'Native map validation has not passed'
variant_file=ROOT/'VARIANT_PROVENANCE.json'
variant=json.loads(variant_file.read_text()) if variant_file.is_file() else None
package_scene=variant.get('variant_id') if variant else rc.SCENE_ID
if not isinstance(package_scene,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',package_scene):
    raise ValueError('Package scene or variant identifier must be a simple name')
NAME=package_scene+'_native_three_maps'
OUT=ROOT/'share'/NAME;OUT.mkdir(parents=True,exist_ok=True)

def copy(src,dst):
    src=Path(src);dst=Path(dst)
    if src.is_dir():
        for file in sorted(src.rglob('*')):
            if file.is_file() and '__pycache__' not in file.parts and file.suffix!='.pyc':copy(file,dst/file.relative_to(src))
    else:
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src.resolve(),dst)

graph=Path(validation['HMSG']['graph'])
copy(graph,OUT/'HMSG'/graph.name)
copy(graph.parent/'graph/nav_graph',OUT/'HMSG/navigation_graph_prototype')
copy(Path(validation['ConceptGraphs']['native']).parent,OUT/'ConceptGraphs/native')
copy(validation['VLMaps']['native'],OUT/'VLMaps/vlmaps_cam.h5df')
for folder in ['rgb','depth','pose']:copy(rc.stage_input()/folder,OUT/'observations'/folder)
copy(rc.stage_input()/'intrinsics.json',OUT/'observations/intrinsics.json')
copy(ROOT/'scene_packages'/rc.SCENE_ID/'observations/poses_vlmaps.txt',OUT/'observations/poses_world_yup_opencv.txt')
copy(ROOT/'supplied_dataset/camera_samples.json',OUT/'observations/camera_samples.json')
copy(ROOT/'native_maps/preview',OUT/'preview')
copy(ROOT/'native_maps/validation.json',OUT/'VALIDATION.json')
copy(ROOT/'native_configs',OUT/'provenance/native_configs')
copy(ROOT/'logs',OUT/'provenance/logs')
copy(rc.CONFIG_PATH,OUT/'provenance/run_config.yaml')
copy(rc.PACKAGE_ROOT/'docs/mapping_provenance/source_inventory.json',OUT/'provenance/source_inventory.json')
copy(rc.PACKAGE_ROOT/'docs/mapping_provenance/verified_run_20261015',OUT/'provenance/upstream_patches')
copy(rc.PACKAGE_ROOT/'code/robot_mapping',OUT/'provenance/portable_pipeline')
if variant:
    copy(variant_file,OUT/'VARIANT_PROVENANCE.json')
    if (ROOT/'provenance/cg_instance_guard').is_dir():
        copy(ROOT/'provenance/cg_instance_guard',OUT/'provenance/cg_instance_guard')
    original_run=Path(variant['original_full_pipeline_run'])
    copy(original_run,OUT/'provenance/original_full_pipeline_run.json')
for path in [ROOT/'supplied_dataset/summary.json',ROOT/'depth_alignment_report.json',ROOT/'pipeline_run.json',ROOT/'doctor.json']:
    if path.exists():copy(path,OUT/'provenance'/path.name)
for path in (OUT/'HMSG'/graph.name/'views').glob('*.json'):
    obj=json.loads(path.read_text());obj['img_path']=os.path.relpath(OUT/'observations/rgb'/Path(obj['img_path']).name,path.parent);path.write_text(json.dumps(obj,ensure_ascii=False,indent=2))
objects=json.loads((OUT/'preview/conceptgraphs_objects.json').read_text())
for obj in objects:
    obj['representative_image']='../observations/rgb/'+f"{obj['sample_frame_ids'][0]:06d}.png"
(OUT/'ConceptGraphs/objects.json').write_text(json.dumps(objects,ensure_ascii=False,indent=2))
copy(ROOT/'native_maps/validation.json',OUT/'MANIFEST.json')
(OUT/'README.md').write_text(f'''# {rc.SCENE_ID}: 原生三地图

来自whichground_clean的可配置建图管线。HMSG {validation['HMSG']['objects']}个预测对象、{validation['HMSG']['rooms']}个房间区域、{validation['HMSG']['views']}个视图；ConceptGraphs {validation['ConceptGraphs']['objects']}个过滤后实例；VLMaps {validation['VLMaps']['feature_shape']}体素特征。

输入为{validation['sample_frames']}张抽样图像。相机每{rc.CAMERA_STRIDE}张取1张；HMSG实例融合再每{rc.HMSG_STRIDE}张取1张，CG每{rc.CG_STRIDE}张，VLMaps每{rc.VLMAPS_STRIDE}张。CLIP在抽样后计算，未参与选帧。

三份地图分别在HMSG目录、ConceptGraphs/native中的post.pkl.gz、VLMaps/vlmaps_cam.h5df。HMSG是一组关联文件。世界坐标Y向上；observations/pose保留HMSG相机轴约定，poses_world_yup_opencv.txt使用OpenCV相机轴约定。

preview含降采样PCD与三图PNG。HMSG/CG预览颜色区分预测实例，VLMaps为融合RGB。PCD查看器使用--y-up。HMSG视图图像路径相对该JSON目录；CG原生pickle保留源路径，便携图像读取使用ConceptGraphs/objects.json。

验证通过的是非空、有限数值、引用及坐标结构；语义准确率、独立标定和真机导航未验证。自动房间区域和预测对象数不是建筑或物体真值。附带导航图为算法原型产物。

provenance记录配置、官方源版本/补丁、管线代码及日志。模型权重与原始MCAP不在包中。解压后运行sha256sum -c SHA256SUMS。
''',encoding='utf-8')
if variant:
    with (OUT/'README.md').open('a',encoding='utf-8') as f:
        f.write('\n## 明确命名的关联修复变体\n\n'+
                f"本包为 `{package_scene}`。HMSG和VLMaps沿用同一录制的原结果；CG来自保存检测的CPU关联重放，未重推网络，也没有在此工作区重新执行完整建图。原运行日志为导入的baseline日志，实际修复来源见 `VARIANT_PROVENANCE.json` 和 `provenance/cg_instance_guard/`。\n\n"+
                '原CG对象编号、人工修订和旧评估成绩不能转移到本变体。同帧排他约束通过不是物理实例准确率；单帧弱片段过滤与潜在过分裂风险保留在结果合同。网站活动三图仍为原基线，独立变体未激活机器人执行。\n')

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256SUMS')
assert not any(p.is_symlink() for p in OUT.rglob('*'))
(OUT/'SHA256SUMS').write_text(''.join(digest(p)+'  '+str(p.relative_to(OUT))+'\n' for p in files))
archive=ROOT/'share'/(NAME+'.tar.gz');temporary=archive.with_name(archive.name+'.part')
with tarfile.open(temporary,'w:gz',compresslevel=5) as tar:tar.add(OUT,arcname=NAME)
sha=digest(temporary);temporary.replace(archive)
archive.with_suffix(archive.suffix+'.sha256').write_text(sha+'  '+archive.name+'\n')
result={'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha,'source_package':str(rc.PACKAGE_ROOT)}
(ROOT/'share/native_package_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))
