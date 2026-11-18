import run_config as rc
from pathlib import Path
import sys
import json
import numpy as np
from omegaconf import OmegaConf

ROOT=rc.WORKSPACE
SOURCE=rc.THIRD_PARTY/'vlmaps';sys.path.insert(0,str(SOURCE))
from vlmaps.map.vlmap_builder_cam import VLMapBuilderCam

checkpoint=rc.MODEL_ROOT/'lseg/demo_e200.ckpt'
if not checkpoint.exists() or checkpoint.stat().st_size<3_000_000_000:
    raise RuntimeError('Official LSeg checkpoint download has not completed')
scene=ROOT/'scene_packages'/rc.SCENE_ID
np.random.seed(0)
out=ROOT/'native_maps/VLMaps'/rc.SCENE_ID;out.mkdir(parents=True,exist_ok=True)
intr=json.loads((scene/'observations/intrinsics.json').read_text())
cfg=OmegaConf.create({'cell_size':float(rc.CONFIG.get('vlmaps',{}).get('cell_size',.1)),'grid_size':1000,'depth_sample_rate':10,
    'skip_frame':rc.VLMAPS_STRIDE,'lseg_checkpoint':str(checkpoint),'cam_calib_mat':np.asarray(intr['K']).reshape(-1).tolist(),
    'pose_info':{'rot_type':'mat','pose_type':'camera_base'}})
OmegaConf.save(cfg,out/'build_config.yaml')
rgbs=sorted((scene/'observations/rgb').glob('*.png'));depths=sorted((scene/'observations/depth').glob('*.npy'))
assert len(rgbs)==len(depths)==rc.sample_count() and len(rgbs)>0
builder=VLMapBuilderCam(out,cfg,scene/'observations/poses_vlmaps.txt',rgbs,depths,np.eye(4),np.eye(4))
builder.create_camera_map()
print('NATIVE_VLMAPS_COMPLETE',out/'vlmap_cam/vlmaps_cam.h5df',flush=True)
