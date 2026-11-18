import run_config as rc
from pathlib import Path
import argparse
import json
import sys
import yaml
import torch
import shutil
import gc

ROOT=rc.WORKSPACE
SOURCE=rc.THIRD_PARTY/'HoloAgent/agentic_robot/fsr_vln'
sys.path.insert(0,str(SOURCE))
from ovo.entities.ovomapping_offline import OVOSemMap
from ovo.integration.hmsg_bridge import build_hmsg_from_ovo_output

p=argparse.ArgumentParser();p.add_argument('--max-frames',type=int,default=-1);p.add_argument('--build-hmsg',action='store_true');p.add_argument('--existing-ovo',type=Path);args=p.parse_args()
view=rc.stage_input()
if args.max_frames>0 and not args.existing_ovo:
    limited=ROOT/'sanity_inputs'/f'frames_{args.max_frames}'
    for folder in ['rgb','depth','pose']:
        (limited/folder).mkdir(parents=True,exist_ok=True)
        for src in sorted((view/folder).glob('*'))[:args.max_frames]:
            dst=limited/folder/src.name
            if not dst.exists():dst.symlink_to(src)
    shutil.copy2(view/'intrinsics.json',limited/'intrinsics.json')
    view=limited
intr=json.loads((view/'intrinsics.json').read_text());K=intr['K']
cfg=yaml.safe_load((SOURCE/'configs/ovo_hm3d.yaml').read_text())
cfg.update(yaml.safe_load((SOURCE/'configs/slam/vanilla/hm3d.yaml').read_text()))
cfg.update(dataset_name='hm3d',device='cuda:0',frame_limit=args.max_frames,use_wandb=False)
cfg['data']={'input_path':str(view),'scene_name':rc.SCENE_ID}
cfg['cam']={'H':intr['height'],'W':intr['width'],'fx':K[0][0],'fy':K[1][1],'cx':K[0][2],'cy':K[1][2],'depth_scale':1000.,'depth_th':20.}
# VanillaMapper consumes downscale_res, not downscale_ratio. Its
# max_frame_points attribute is not used in map(); do not advertise a cap.
# Preserve today's validated full pixel geometry by default. Stride 2 is an
# explicit performance experiment; track_th is an absolute point threshold.
geometry_stride=rc.CONFIG.get('hmsg',{}).get('downscale_res',1)
if type(geometry_stride) is not int or geometry_stride<1:
    raise ValueError('hmsg.downscale_res must be a positive integer')
cfg['mapping'].pop('downscale_ratio',None)
cfg['mapping'].pop('max_frame_points',None)
cfg['mapping'].update(map_every=rc.HMSG_STRIDE,k_pooling=7,downscale_res=geometry_stride)
cfg['tracking']['track_every']=1
cfg['semantic'].update(segment_every=rc.HMSG_STRIDE,match_distance_th=.05,use_gt_masks=False,use_gt_clips=False)
cfg['semantic']['sam'].update(sam_ckpt_path=str(rc.MODEL_ROOT/'sam2'),sam_version='2.1',sam_encoder='hiera_t',mask_res=384,precomputed=False,multi_crop=False)
cfg['semantic']['clip'].update(model_card='ViT-H-14',embed_type='fixed_weights',use_half=True)
cfg['vis'].update(stream=False,show_stream=False)
name=rc.SCENE_ID if args.max_frames<0 else f'{rc.SCENE_ID}__sanity_{args.max_frames}'
out=args.existing_ovo.resolve() if args.existing_ovo else ROOT/'native_maps/HMSG'/name
mapper=None
if not args.existing_ovo:
    mapper=OVOSemMap(cfg,str(out),rc.SCENE_ID);mapper.run()
if args.build_hmsg:
    labels_path=SOURCE/'memory/hmsg/labels/text_feats_SCANNET200_LABELS.npy'
    if labels_path.exists():
        with labels_path.open('rb') as stream:
            is_lfs_pointer=stream.read(64).startswith(b'version https://git-lfs.github.com/spec/v1')
        if is_lfs_pointer:
            labels_path.rename(labels_path.with_suffix('.npy.lfs-pointer'))
    # Release inference modules before loading the graph's matching CLIP encoder.
    # OVOSemMap.run() already deletes its inference modules after saving.
    mapper=None
    gc.collect()
    torch.cuda.empty_cache()
    graph_cfg=yaml.safe_load((SOURCE/'configs/hmsg_bridge_hm3d.yaml').read_text())
    graph_cfg['models']['clip']={'type':'ViT-H-14','checkpoint':'laion2b_s32b_b79k'}
    # Approximate height from the reconstructed floor (~-0.58 m) and camera
    # trajectory (~0.09 m); this is a mapping parameter, not a calibration.
    graph_cfg['pipeline'].update(full_pcd_dbscan_eps=.2,full_pcd_dbscan_min_points=20,save_intermediate_results=False,camera_height=float(rc.CONFIG.get('hmsg',{}).get('camera_height',.65)))
    path=ROOT/'native_configs/hmsg_new_dataset.yaml';path.write_text(yaml.safe_dump(graph_cfg))
    result=build_hmsg_from_ovo_output(out,view,'HM3D',rc.SCENE_ID,str(path),min_object_points=30)
    print('NATIVE_HMSG_COMPLETE',result,flush=True)
else:print('NATIVE_OVO_SANITY_COMPLETE',out,flush=True)
