#!/usr/bin/env python3
"""Prepare validated dataset/configuration; reuse installed models/environments."""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path,PurePosixPath
import shutil
import stat
import tempfile
import zipfile
import numpy as np
import yaml
from new_run_common import PACKAGE,CODE,sha,value_hash,atomic_json,scene_name,protected_output

def zip_files(archive):
    result=[]
    for info in archive.infolist():
        name=PurePosixPath(info.filename)
        if name.is_absolute() or '..' in name.parts or '\\' in info.filename or ':' in info.filename or '\x00' in info.filename:raise ValueError('Unsafe archive member path')
        if stat.S_ISLNK(info.external_attr>>16):raise ValueError('Dataset archive symlinks are not accepted')
        if not info.is_dir():result.append(info)
    if len({i.filename for i in result})!=len(result):raise ValueError('Duplicate archive member names')
    return result

def extract_verified(source,target):
    target=protected_output(target);checksum=sha(source);record=target/'.whichground_extraction.json'
    with zipfile.ZipFile(source) as z:
        files=zip_files(z)
        if not any(i.filename.endswith('.mcap') for i in files) or not any(PurePosixPath(i.filename).name=='metadata.yaml' for i in files):raise ValueError('Archive is not a dataset with MCAP+metadata; code handoffs are not robot datasets')
        if target.exists():
            if not record.is_file():raise ValueError('Existing extraction is not fingerprinted. Provide its directory explicitly or use a fresh --extract-to; no blind ZIP resume')
            prior=json.loads(record.read_text())
            if prior.get('archive_sha256')!=checksum:raise ValueError('Existing extraction belongs to a different archive')
            expected={i.filename for i in files}
            if set(prior.get('files',{}))!=expected:raise ValueError('Extraction member inventory changed')
            if any(not (target/n).is_file() or sha(target/n)!=r['sha256'] for n,r in prior['files'].items()):raise ValueError('Previously extracted dataset content changed')
            return target,{'archive_sha256':checksum,'reused_verified_extraction':True}
        target.parent.mkdir(parents=True,exist_ok=True)
        if shutil.disk_usage(target.parent).free<sum(i.file_size for i in files)+512*1024*1024:raise ValueError('Insufficient space for dataset extraction')
        stage=Path(tempfile.mkdtemp(prefix=target.name+'.extracting.',dir=target.parent));entries={}
        try:
            import hashlib
            for info in files:
                out=stage/info.filename;out.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();size=0
                with z.open(info) as src,out.open('wb') as dst:
                    for block in iter(lambda:src.read(8*1024*1024),b''):dst.write(block);h.update(block);size+=len(block)
                if size!=info.file_size:raise ValueError('Extracted byte count mismatch')
                entries[info.filename]={'bytes':size,'sha256':h.hexdigest(),'zip_crc32':info.CRC}
            atomic_json(stage/'.whichground_extraction.json',{'schema_version':1,'archive_sha256':checksum,'source_archive':str(source),'files':entries,'CRC_checked_during_stream_read':True})
            stage.replace(target)
        except BaseException:shutil.rmtree(stage,ignore_errors=True);raise
    return target,{'archive_sha256':checksum,'reused_verified_extraction':False}

def calibration_data(path):
    text=Path(path).read_text(encoding='utf-8-sig');decoder=json.JSONDecoder();objects=[]
    while text.strip():
        text=text.lstrip();value,end=decoder.raw_decode(text);objects.append(value);text=text[end:]
    camera=next((x for x in objects if x.get('source_frame')=='camera'),None);lidar=next((x for x in objects if x.get('source_frame')=='rslidar_front'),None)
    if camera is None or lidar is None:raise ValueError('Calibration must explicitly include camera and rslidar_front to base transforms')
    for o in [camera,lidar]:
        T=np.asarray(o['transform'],float)
        if T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1]) or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-4) or not np.isclose(np.linalg.det(T[:3,:3]),1,atol=1e-4):raise ValueError('Calibration transform is not a finite rigid transform')
    K=np.asarray(camera['camera_matrix'],float);dimensions=camera['image_size'];d=np.asarray(camera['distortion_coefficients'],float)
    if K.shape!=(3,3) or not np.isfinite(K).all() or min(K[0,0],K[1,1])<=0 or len(dimensions)!=2 or any(not isinstance(v,int) or v<=0 for v in dimensions) or d.size not in [4,5,8,12,14] or not np.isfinite(d).all():raise ValueError('Invalid camera intrinsic/dimensions/distortion')
    return camera,lidar

def inspect_bag(bag,preferred,calibration):
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores,get_typestore
    types={'image':'sensor_msgs/msg/Image','lidar':'sensor_msgs/msg/PointCloud2','odom':'nav_msgs/msg/Odometry'}
    with AnyReader([bag],default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:
        topics={};roles={}
        for role,type_ in types.items():
            candidates=[c for c in reader.connections if c.msgtype==type_ and c.msgcount>0];names=set(c.topic for c in candidates)
            chosen=preferred.get(role) if preferred.get(role) in names else next(iter(names)) if len(names)==1 else None
            if chosen is None:raise ValueError(f'Explicit {role} topic required; candidates={sorted(names)}')
            selected=[c for c in candidates if c.topic==chosen];topics[role]=chosen;roles[role]=selected
        if len(set(topics.values()))!=3:raise ValueError('Image/lidar/odom topics must be distinct')
        first={}
        for c,receipt,raw in reader.messages(connections=[c for v in roles.values() for c in v]):
            role=next(k for k,v in topics.items() if v==c.topic)
            if role in first:continue
            m=reader.deserialize(raw,c.msgtype);first[role]={'topic':c.topic,'type':c.msgtype,'message_count':sum(x.msgcount for x in roles[role]),'frame_id':m.header.frame_id}
            if role=='image':
                if m.encoding.lower()!='nv12' or [m.width,m.height]!=calibration['image_size']:raise ValueError('Camera encoding/dimensions do not match supplied NV12 calibration')
                first[role].update(width=m.width,height=m.height,encoding=m.encoding)
            elif role=='odom':first[role]['child_frame_id']=m.child_frame_id
            else:
                fs={f.name:f for f in m.fields}
                if any(n not in fs or fs[n].datatype!=7 or fs[n].count!=1 for n in ['x','y','z']):raise ValueError('LiDAR XYZ must be FLOAT32 scalar fields')
            if len(first)==3:break
    if len(first)!=3:raise ValueError('Required input role has no messages')
    return topics,first

def prepare(a):
    source=Path(a.dataset).expanduser().resolve();workspace=protected_output(a.workspace);scene_name(a.scene_id)
    if source.is_relative_to(PACKAGE.parent/'real'):raise ValueError('New processing cannot depend on historical incoming/real')
    if not source.exists():raise FileNotFoundError(source)
    extraction={}
    if source.is_file():
        if source.suffix.lower()!='.zip':raise ValueError('Supply extracted dataset directory or ZIP')
        root,extraction=extract_verified(source,a.extract_to or source.with_suffix(''))
    else:root=source
    bags=[root] if (root/'metadata.yaml').is_file() else [p.parent for p in root.rglob('metadata.yaml')]
    if a.bag:
        bag=(root/Path(a.bag).expanduser()).resolve()
    elif len(bags)==1:bag=bags[0]
    else:raise ValueError('Multiple/no bags: provide --bag explicitly')
    cal=(root/Path(a.calibration).expanduser()).resolve() if a.calibration else root/'extrinst.txt'
    if not cal.is_file():raise ValueError('Provide the actual recording calibration with --calibration; old calibration is never substituted')
    camera,lidar=calibration_data(cal)
    template=yaml.safe_load(Path(a.template).expanduser().read_text()) if a.template else yaml.safe_load((CODE/'robot_mapping/configs/new_recording_defaults.yaml').read_text())
    preferred={'image':'/aima/hal/camera/front/image_raw','lidar':'/aima/hal/lidar/front/point_cloud','odom':'/map_odometry'};preferred.update(template.get('dataset',{}).get('topics',{}))
    for role in preferred:
        value=getattr(a,role+'_topic',None)
        if value:preferred[role]=value
    topics,details=inspect_bag(bag,preferred,camera)
    cfg=copy.deepcopy(template);cfg['scene_id']=a.scene_id;cfg['workspace']=str(workspace);cfg['dataset']={'directory':str(root),'bag':str(bag),'calibration':str(cal),'image_max_width':a.image_max_width if a.image_max_width is not None else template.get('dataset',{}).get('image_max_width',960),'topics':topics}
    for key,value in [('camera_stride',a.camera_stride),('lidar_stride',a.lidar_stride),('max_samples',a.max_samples)]:
        if value is not None:cfg.setdefault('sampling',{})[key]=value
    if a.seed is not None:cfg.setdefault('depth',{})['fit_seed']=a.seed
    if any(isinstance(cfg.get('sampling',{}).get(k),bool) or not isinstance(cfg.get('sampling',{}).get(k),int) or cfg['sampling'][k]<1 for k in ['camera_stride','lidar_stride']):raise ValueError('Sampling strides must be positive integers')
    maximum=cfg.get('sampling',{}).get('max_samples',0)
    if isinstance(maximum,bool) or not isinstance(maximum,int) or maximum<0:raise ValueError('max_samples must be a nonnegative integer')
    if a.include_class:
        cfg.setdefault('conceptgraphs',{})['classes']=list(dict.fromkeys(cfg.get('conceptgraphs',{}).get('classes',[])+a.include_class))
    if not cfg.get('conceptgraphs',{}).get('classes') or any(not isinstance(c,str) or not c.strip() for c in cfg['conceptgraphs']['classes']):raise ValueError('Detector vocabulary must be explicit nonempty strings')
    for section in ['hmsg','conceptgraphs','vlmaps']:
        stride=cfg.get(section,{}).get('stride',1)
        if isinstance(stride,bool) or not isinstance(stride,int) or stride<1:raise ValueError('Mapper strides must be positive integers')
    numeric=[cfg['dataset']['image_max_width'],cfg.get('hmsg',{}).get('camera_height',.65),cfg.get('vlmaps',{}).get('cell_size',.1)]
    if not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and v>0 for v in numeric):raise ValueError('Finite positive image width/camera-height/cell size required')
    if (workspace/'pipeline_run.json').is_file() and json.loads((workspace/'pipeline_run.json').read_text()).get('scene_id')!=a.scene_id:raise ValueError('Existing workspace belongs to a different mapped scene')
    workspace.mkdir(parents=True,exist_ok=True);out=workspace/'run_config.yaml';serialized=yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False)
    if out.exists() and yaml.safe_load(out.read_text())!=cfg:raise ValueError('Existing workspace has different configuration; use a new workspace instead of mixing recordings')
    if not out.exists():out.write_text(serialized)
    record={'schema_version':1,'scene_id':a.scene_id,'workspace':str(workspace),'mapping_config':str(out),'configuration_sha256':sha(out),'source_dataset':str(root),'bag':str(bag),'calibration':str(cal),'calibration_sha256':sha(cal),'input_topics':details,'extraction':extraction,'models_and_env_reused':True,'sampling_definition':'fixed original message stride, not novelty-selected keyframes','robot_execution_ready':False}
    atomic_json(workspace/'run_preparation.json',record);return record

def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',required=True);p.add_argument('--scene-id',required=True);p.add_argument('--workspace',required=True);p.add_argument('--bag');p.add_argument('--calibration');p.add_argument('--extract-to');p.add_argument('--template')
    for k in ['image','lidar','odom']:p.add_argument('--'+k+'-topic')
    for k in ['camera-stride','lidar-stride','max-samples','image-max-width','seed']:p.add_argument('--'+k,type=int)
    p.add_argument('--include-class',action='append',default=[],help='Append an explicit detector category, e.g. pressure gauge; does not guarantee detection accuracy')
    return p
if __name__=='__main__':print(json.dumps(prepare(parser().parse_args()),ensure_ascii=False,indent=2))
