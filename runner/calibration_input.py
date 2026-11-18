"""Distinguish recording metadata, target labels and actual rig calibration."""
from __future__ import annotations
import json
from pathlib import Path
import yaml

METADATA_MESSAGE='标定栏选错：metadata.yaml 是 rosbag 录制索引，不是相机/雷达标定。请通过 --calibration 提供对应录制的内外参。'
LABEL_MESSAGE='标定栏选错：这是标注 YAML，请传给 --label-yaml。坐标标定需要相机内参以及相机、雷达到base的外参。'

def calibration_records(path):
    path=Path(path).expanduser().resolve()
    if not path.is_file():raise ValueError('标定文件不存在：'+str(path))
    if path.stat().st_size>4*1024**2:raise ValueError('所选文件过大，不像相机/雷达标定文件')
    text=path.read_text(encoding='utf-8-sig');remaining=text;records=[];fmt='json_records'
    try:
        decoder=json.JSONDecoder()
        while remaining.strip():
            value,end=decoder.raw_decode(remaining.lstrip());records.append(value);remaining=remaining.lstrip()[end:]
    except json.JSONDecodeError:
        try:value=yaml.safe_load(text)
        except yaml.YAMLError as error:raise ValueError('标定文件不是有效JSON/YAML：请选真正的相机/雷达内外参文件') from error
        records=[value];fmt='yaml'
    if len(records)==1:
        value=records[0]
        if isinstance(value,dict) and 'rosbag2_bagfile_information' in value:raise ValueError(METADATA_MESSAGE)
        if isinstance(value,dict) and 'annotations' in value:raise ValueError(LABEL_MESSAGE)
        if isinstance(value,list):records=value;fmt='yaml_list' if fmt=='yaml' else 'json_list'
    if not records or any(not isinstance(o,dict) for o in records):raise ValueError('标定格式不正确：应含camera和rslidar_front两组内外参，不能用录制索引或点位标签代替')
    camera=next((o for o in records if o.get('source_frame')=='camera'),None)
    lidar=next((o for o in records if o.get('source_frame')=='rslidar_front'),None)
    if camera is None or lidar is None:raise ValueError('标定缺少camera或rslidar_front外参；请选择当前录制对应的内外参文件')
    required={'camera_matrix','image_size','distortion_coefficients','transform','target_frame'}
    missing=required-set(camera)
    if missing or not {'transform','target_frame'}.issubset(lidar):raise ValueError('标定缺字段：需要camera_matrix/image_size/distortion_coefficients和两传感器到base的transform')
    return records,fmt

def validate_calibration_file(path):
    calibration_records(path)
    return Path(path).expanduser().resolve()

def normalized_snapshot(source,destination):
    source=Path(source);destination=Path(destination);records,fmt=calibration_records(source);destination.parent.mkdir(parents=True,exist_ok=True)
    if fmt=='json_records':destination.write_bytes(source.read_bytes())
    else:destination.write_text('\n'.join(json.dumps(o,ensure_ascii=False,indent=2) for o in records)+'\n')
    return fmt
