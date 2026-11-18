"""Configure the unchanged native mappers from verified lossless observations."""
from pathlib import Path
import json,copy,shutil
import yaml

def prepare_input(args,workspace,code):
 from portable_observations import unpack,BundleReader,validate
 from prepare_new_run import calibration_data,inspect_bag
 source=Path(args.prepared_input).expanduser().resolve();root=unpack(source,workspace/'prepared_input');manifest=validate(root);cal=root/manifest['calibration_file']
 if args.calibration:
  from calibration_input import validate_calibration_file,normalized_snapshot
  cal=validate_calibration_file(args.calibration);normalized=workspace/'calibration/custom.txt';normalized.parent.mkdir(exist_ok=True);normalized_snapshot(cal,normalized);cal=normalized
 camera,lidar=calibration_data(cal);reader=BundleReader([root]);topics={k:manifest['topics'][k] for k in ['image','lidar','odom']};details={}
 with reader:
  for c,receipt,raw in reader.messages():
   role=c.role
   if role=='imu' or role in details:continue
   m=reader.deserialize(raw,c.msgtype);details[role]={'topic':c.topic,'type':c.msgtype,'message_count':c.msgcount,'frame_id':m.header.frame_id}
   if role=='image':
    if m.encoding.lower()!='nv12' or [m.width,m.height]!=camera['image_size']:raise ValueError('预处理相机编码/尺寸与标定不一致')
   elif role=='odom':details[role]['child_frame_id']=m.child_frame_id
   if len(details)==3:break
 if details['image']['frame_id']!=camera['source_frame'] or details['lidar']['frame_id']!=lidar['source_frame'] or details['odom']['child_frame_id']!=camera['target_frame']:raise ValueError('预处理原始frame与实际外参不一致')
 cfg=yaml.safe_load((code/'configs/new_recording_defaults.yaml').read_text());cfg['scene_id']=workspace.name;cfg['workspace']=str(workspace);cfg['dataset']={'directory':str(root),'bag':'.','calibration':str(cal),'prepared_observations':True,'topics':topics,'image_max_width':cfg['dataset'].get('image_max_width',960)}
 if args.max_samples:cfg['sampling']['max_samples']=args.max_samples
 if args.include_class:cfg['conceptgraphs']['classes']=list(dict.fromkeys(cfg['conceptgraphs']['classes']+args.include_class))
 config=workspace/'run_config.yaml';config.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False));return {'mapping_config':str(config),'input_topics':details,'calibration':str(cal),'manifest':manifest,'source':str(source),'input_type':'lossless_prepared_observations'}
