import run_config as rc
from pathlib import Path
import sys
import json
import argparse
import cv2
import numpy as np
from omegaconf import OmegaConf

ROOT=rc.WORKSPACE;SOURCE=rc.THIRD_PARTY/'concept-graphs'
sys.path.insert(0,str(SOURCE))
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','detect','map'],required=True);p.add_argument('--max-frames',type=int,default=-1);p.add_argument('--association-mode',choices=['guarded','author_baseline']);args=p.parse_args()
name=rc.SCENE_ID;data_root=ROOT/'native_maps/ConceptGraphs/data';scene=data_root/name;scene.mkdir(parents=True,exist_ok=True)
if args.stage=='map' and any((scene/'pcd_saves'/filename).exists() for filename in ['full_pcd_yolo_world_new_robot.pkl.gz','full_pcd_yolo_world_new_robot_post.pkl.gz']):
    raise RuntimeError('Existing native map is immutable. Use a new workspace or the independent replay experiment tool.')
source=ROOT/'mapper_inputs/fsrvln'/name
classes=rc.CONFIG.get('conceptgraphs',{}).get('classes',['electrical cabinet','fire extinguisher','traffic cone','cardboard box','door','window','railing','stairs','chair','table','person','robot','trash can'])
classes_file=ROOT/'native_configs/real_object_classes.txt';classes_file.write_text('\n'.join(classes)+'\n')
intr=json.loads((source/'intrinsics.json').read_text());K=intr['K']
dataset_cfg=ROOT/'native_configs/cg_real_as_replica.yaml'
OmegaConf.save(OmegaConf.create({'dataset_name':'replica','camera_params':{'image_height':intr['height'],'image_width':intr['width'],'fx':K[0][0],'fy':K[1][1],'cx':K[0][2],'cy':K[1][2],'png_depth_scale':1000.}}),dataset_cfg)
if args.stage=='prepare':
    out=scene/'results';out.mkdir(exist_ok=True);poses=[]
    rgbs=sorted((source/'rgb').glob('*'))
    for idx,rgb in enumerate(rgbs):
        name_idx=f'{idx:06d}'
        cv2.imwrite(str(out/f'frame{name_idx}.jpg'),cv2.imread(str(rgb)),[cv2.IMWRITE_JPEG_QUALITY,100])
        depth=source/'depth'/f'{name_idx}.png';cv2.imwrite(str(out/f'depth{name_idx}.png'),cv2.imread(str(depth),cv2.IMREAD_UNCHANGED))
        pose=np.loadtxt(source/'pose'/f'{name_idx}.txt').reshape(4,4)@np.diag([1,-1,-1,1]);poses.append(pose)
    np.savetxt(scene/'traj.txt',np.asarray(poses).reshape(-1,16),fmt='%.9f')
    (scene/'SOURCE.json').write_text(json.dumps({'dataset':rc.SCENE_ID,'carrier_format':'Replica RGB-D directory convention only','frame_count':len(poses),'source':str(source),'relative_pose':False,'coordinate_convention':'Y-up world, OpenCV camera'},indent=2))
    print('CG_INPUT_READY',len(poses));sys.exit(0)

if args.stage=='detect':
    from conceptgraph.scripts.streamlined_detections import main
    cfg=OmegaConf.load(SOURCE/'conceptgraph/hydra_configs/base_detections.yaml')
    cfg.update(dataset_root=str(data_root),dataset_config=str(dataset_cfg),scene_id=name,start=0,end=args.max_frames,stride=rc.CG_STRIDE,desired_height=intr['height'],desired_width=intr['width'],classes_file=str(classes_file),device='cuda:0',exp_suffix='new_robot',relative_pose=False,save_video=False,yolo_model_path=str(rc.MODEL_ROOT/'conceptgraphs/yolov8l-world.pt'),sam_model_path=str(rc.MODEL_ROOT/'conceptgraphs/mobile_sam.pt'))
    main.__wrapped__(cfg)
    (scene/'classes.json').write_text(json.dumps(classes));print('NATIVE_CG_DETECTIONS_COMPLETE',flush=True)
else:
    from conceptgraph.slam.streamlined_mapping import main
    cfg=OmegaConf.load(SOURCE/'conceptgraph/hydra_configs/base_mapping.yaml')
    cfg.update(dataset_root=str(data_root),dataset_config=str(dataset_cfg),scene_id=name,start=0,end=args.max_frames,stride=rc.CG_STRIDE,image_height=intr['height'],image_width=intr['width'],device='cuda:0',relative_pose=False,gsa_variant='yolo_world',detection_folder_name='exp_new_robot/detections',color_file_name='classes',save_suffix='new_robot',spatial_sim_type='overlap',match_method='sim_sum',sim_threshold=.8,obj_min_detections=2,vis_render=False,debug_render=False,save_pcd=True)
    association=rc.CONFIG.get('conceptgraphs',{})
    mode=args.association_mode or association.get('association_mode','guarded')
    cfg.update(association_mode=mode,instance_sameframe_min_iou=float(association.get('sameframe_min_iou',.5)),instance_min_spatial_overlap=float(association.get('min_spatial_overlap',.01)),instance_min_unique_frames=int(association.get('min_unique_frames',2)))
    main.__wrapped__(cfg);print('NATIVE_CONCEPTGRAPHS_COMPLETE',flush=True)
