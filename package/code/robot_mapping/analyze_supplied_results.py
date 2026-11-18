import run_config as rc
"""Fit DA3 to provided-calibration LiDAR samples and prepare current-code scene inputs."""
from pathlib import Path
import json
import cv2
import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=rc.WORKSPACE;DATA=ROOT/'supplied_dataset'
STAGE=ROOT/'aligned_scene_input'
for name in ['rgb','metric_depth','slam','previews']:(STAGE/name).mkdir(parents=True,exist_ok=True)
results=json.loads((ROOT/'da3/depth_results.json').read_text())['results'];rng=np.random.default_rng(int(rc.CONFIG.get('depth',{}).get('fit_seed',20261015)));reports=[]
for row in results:
    idx=row['source_index'];name=f'{idx:06d}'
    pred=np.load(ROOT/'da3'/row['depth_npy_fullres']).astype(float);ref=np.load(DATA/'lidar_depth'/f'{name}.npy')
    keep=np.isfinite(pred)&(pred>.1)&(ref>.3)&(ref<20)
    x=pred[keep];y=ref[keep]
    if len(x)<500:raise RuntimeError(f'Insufficient projection support for frame {idx}')
    ids=rng.permutation(len(x));split=int(.8*len(x));train=ids[:min(3000,split)];test=ids[split:]
    fit=least_squares(lambda ab:ab[0]*x[train]+ab[1]-y[train],[float(np.median(y[train]/x[train])),0],loss='soft_l1',f_scale=.2,bounds=([.01,-10],[50,10]))
    a,b=fit.x;depth=a*pred+b;depth[(depth<.2)|(depth>25)|~np.isfinite(depth)]=0
    np.save(STAGE/'metric_depth'/f'{name}.npy',depth.astype(np.float32))
    rgb=cv2.imread(str(DATA/'rgb_rectified'/f'{name}.png'));cv2.imwrite(str(STAGE/'rgb'/f'{name}.png'),rgb)
    raw=np.abs(x[test]-y[test])/y[test];error=np.abs(a*x[test]+b-y[test])/y[test]
    reports.append({'frame':idx,'scale':float(a),'offset':float(b),'reference_pixels':len(x),'train_pixels':len(train),'heldout_pixels':len(test),'raw_median_abs_rel':float(np.median(raw)),'fitted_median_abs_rel':float(np.median(error)),'fitted_p90_abs_rel':float(np.percentile(error,90))})
intr=json.loads((DATA/'intrinsics.json').read_text());(STAGE/'intrinsics.json').write_text(json.dumps(intr,indent=2))
poses=np.load(DATA/'camera_poses.npz')['poses'];S=np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]],float)
np.savetxt(STAGE/'slam/poses.txt',(S[None]@poses).reshape(-1,16),fmt='%.9f')
np.savetxt(STAGE/'slam/poses_world_base.txt',(S[None]@poses@np.linalg.inv(np.asarray(json.loads((DATA/'summary.json').read_text())['base_from_camera']))).reshape(-1,16),fmt='%.9f')
points=np.load(DATA/'map_recorded_odometry.npz')['xyz'];np.savez_compressed(STAGE/'slam/lidar_map.npz',points=points@S[:3,:3].T)
summary={'frames':len(reports),'raw_abs_rel_median':float(np.median([x['raw_median_abs_rel'] for x in reports])),'fitted_abs_rel_median':float(np.median([x['fitted_median_abs_rel'] for x in reports])),'fitted_abs_rel_p90_across_frames':float(np.percentile([x['fitted_median_abs_rel'] for x in reports],90)),'frames_above_20pct':sum(x['fitted_median_abs_rel']>.2 for x in reports),'frames':len(reports),'frame_reports':reports,'interpretation':'Error against held-out projected LiDAR samples, conditional on supplied calibration and estimated clock rebasing. Not a measurement of calibration error alone or full-image depth accuracy.'}
(ROOT/'depth_alignment_report.json').write_text(json.dumps(summary,indent=2))
fig,ax=plt.subplots(figsize=(11,4),constrained_layout=True);ax.plot([r['raw_median_abs_rel'] for r in reports],label='DA3 focal-scaled');ax.plot([r['fitted_median_abs_rel'] for r in reports],label='LiDAR fitted, held-out');ax.set_xlabel('Sample frame index');ax.set_ylabel('Median absolute relative error');ax.set_title('Supplied dataset: held-out LiDAR projection consistency');ax.legend();ax.grid(alpha=.2);fig.savefig(ROOT/'depth_alignment_preview.png',dpi=150);plt.close(fig)
fig=plt.figure(figsize=(11,7));ax=fig.add_subplot(111,projection='3d');sample=points[::max(1,len(points)//150000)];ax.scatter(sample[:,0],sample[:,1],sample[:,2],c=sample[:,2],s=.3,cmap='viridis');ax.plot(poses[:,0,3],poses[:,1,3],poses[:,2,3],c='red');ax.set_xlabel('World x / m');ax.set_ylabel('World y / m');ax.set_zlabel('World z / m');ax.set_title('New bag: supplied extrinsics + recorded odometry');ax.set_box_aspect(np.maximum(np.ptp(sample,axis=0),1));ax.view_init(elev=45,azim=-65);fig.savefig(ROOT/'new_map_preview.png',dpi=150,bbox_inches='tight');plt.close(fig)
print({k:v for k,v in summary.items() if k!='frame_reports'},flush=True)
