import run_config as rc
"""Decode the supplied NV12 bag and apply its explicit base-camera/LiDAR transforms."""
from pathlib import Path
import json
import time
import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores,get_typestore
if rc.CONFIG.get('dataset',{}).get('prepared_observations'):
    from portable_observations import BundleReader as AnyReader
from scipy.spatial.transform import Rotation,Slerp
import open3d as o3d
from lidar_point_time import decode_cloud

ROOT=rc.WORKSPACE
PAIRING_STRIDE=int(rc.CONFIG.get('sampling',{}).get('lidar_pairing_stride',1))
if PAIRING_STRIDE<1:raise ValueError('sampling.lidar_pairing_stride must be positive')
DATA=rc.DATASET
OUT=ROOT/'supplied_dataset';OUT.mkdir(parents=True,exist_ok=True)
for n in ['rgb_raw','rgb_rectified','lidar_samples','lidar_pairing_samples','lidar_depth','projection_panels']:(OUT/n).mkdir(exist_ok=True)
text=rc.calibration_path().read_text();decoder=json.JSONDecoder();transforms=[]
while text.strip():
    obj,end=decoder.raw_decode(text.lstrip());transforms.append(obj);text=text.lstrip()[end:]
cam=next(x for x in transforms if x['source_frame']=='camera')
lid=next(x for x in transforms if x['source_frame']=='rslidar_front')
Tbc=np.asarray(cam['transform'],float);Tbl=np.asarray(lid['transform'],float)
for T in [Tbc,Tbl]:
    assert np.allclose(T[3],[0,0,0,1]) and np.isclose(np.linalg.det(T[:3,:3]),1,atol=1e-5)
    assert np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-5)
sourceK=np.asarray(cam['camera_matrix'],float);D=np.asarray(cam['distortion_coefficients'],float)
source_width,source_height=cam['image_size']
width=min(source_width,int(rc.CONFIG['dataset'].get('image_max_width',960)))
height=round(source_height*width/source_width)
sx=width/source_width;sy=height/source_height
newK,roi=cv2.getOptimalNewCameraMatrix(sourceK,D,(source_width,source_height),0,(source_width,source_height))
K=newK.copy();K[0]*=sx;K[1]*=sy
scaled_source=sourceK.copy();scaled_source[0]*=sx;scaled_source[1]*=sy
mx,my=cv2.initUndistortRectifyMap(scaled_source,D,np.eye(3),K,(width,height),cv2.CV_32FC1)
counts={};timelines={};cameras=[];clouds=[];odoms=[];started=time.time()
expected_types={rc.TOPICS['image']:'sensor_msgs/msg/Image',rc.TOPICS['lidar']:'sensor_msgs/msg/PointCloud2',rc.TOPICS['odom']:'nav_msgs/msg/Odometry'}
with AnyReader([rc.bag_path()],default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:
    # Choose connections by configured topic AND message type. Other cameras,
    # clouds and odometries must never enter this scene's timelines or arrays.
    all_counts={}
    for c in reader.connections:all_counts[c.topic]=all_counts.get(c.topic,0)+c.msgcount
    selected=[c for c in reader.connections if c.topic in expected_types]
    for topic,msgtype in expected_types.items():
        matching=[c for c in selected if c.topic==topic]
        if not matching:raise RuntimeError(f'Missing configured topic {topic}; available topics: {sorted(all_counts)}')
        if any(c.msgtype!=msgtype for c in matching):raise RuntimeError(f'Configured topic {topic} must use {msgtype}')
    for c,receipt,raw in reader.messages(connections=selected):
        m=reader.deserialize(raw,c.msgtype);topic=c.topic;i=counts.get(topic,0);counts[topic]=i+1
        stamp=m.header.stamp.sec*10**9+m.header.stamp.nanosec
        timelines.setdefault(topic,[]).append([receipt,stamp])
        if topic==rc.TOPICS['odom']:
            p=m.pose.pose;odoms.append([p.position.x,p.position.y,p.position.z,p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w])
        elif topic==rc.TOPICS['image'] and i%rc.CAMERA_STRIDE==0 and (rc.MAX_SAMPLES<=0 or len(cameras)<rc.MAX_SAMPLES):
            assert (m.width,m.height,m.encoding.lower())==(source_width,source_height,'nv12')
            buffer=np.asarray(m.data,dtype=np.uint8).reshape(m.height*3//2,m.step)[:,:m.width]
            rgb=cv2.cvtColor(buffer,cv2.COLOR_YUV2BGR_NV12)
            rgb=cv2.resize(rgb,(width,height),interpolation=cv2.INTER_AREA)
            name=f'{len(cameras):06d}'
            cv2.imwrite(str(OUT/'rgb_raw'/f'{name}.jpg'),rgb,[cv2.IMWRITE_JPEG_QUALITY,92])
            rect=cv2.remap(rgb,mx,my,cv2.INTER_LINEAR)
            cv2.imwrite(str(OUT/'rgb_rectified'/f'{name}.png'),rect)
            cameras.append({'index':len(cameras),'source_index':i,'receipt_ns':receipt,'stamp_ns':stamp,'file':name})
        elif topic==rc.TOPICS['lidar'] and (i%rc.LIDAR_STRIDE==0 or i%PAIRING_STRIDE==0):
            payload=decode_cloud(m);xyz=payload['xyz']
            name=f'{i:06d}.npz'
            if i%rc.LIDAR_STRIDE==0:np.savez_compressed(OUT/'lidar_samples'/name,**payload)
            if i%PAIRING_STRIDE==0:np.savez_compressed(OUT/'lidar_pairing_samples'/name,**payload)
            clouds.append({'index':i,'receipt_ns':receipt,'stamp_ns':stamp,'file':name,'points':len(xyz),'map_sample':i%rc.LIDAR_STRIDE==0,
                           'auxiliary_fields':[n for n in ['intensity','ring','timestamp'] if 'raw_'+n in payload]})
        if sum(counts.values())%10000==0:print('scanned',sum(counts.values()),'elapsed_s',round(time.time()-started,1),flush=True)

if not cameras or not clouds or len(odoms)<2:raise RuntimeError('Configured topics require nonempty camera/LiDAR samples and at least two odometry messages')
clock={}
for topic,rows in timelines.items():
    rows=np.asarray(rows,dtype=np.int64);delta=rows[:,0]-rows[:,1]
    correction=int(np.median(delta))
    clock[topic]={'header_to_receipt_offset_ns':correction,'residual_ms_p01_p50_p99':np.percentile((delta-correction)/1e6,[1,50,99]).tolist(),'header_is_unix_epoch':bool(rows[0,1]>10**17),'duplicate_headers':int((np.diff(rows[:,1])==0).sum())}
    np.savez_compressed(OUT/('timeline_'+topic.strip('/').replace('/','_')+'.npz'),receipt_ns=rows[:,0],header_ns=rows[:,1])
odom_times=np.asarray(timelines[rc.TOPICS['odom']],dtype=np.int64)[:,1]
poses=np.asarray(odoms);unique,ids=np.unique(odom_times,return_index=True);poses=poses[ids]
if len(unique)<2:raise RuntimeError('At least two distinct odometry timestamps are required for pose interpolation')
odom_times=unique+clock[rc.TOPICS['odom']]['header_to_receipt_offset_ns']
rot=Slerp((odom_times-odom_times[0])/1e9,Rotation.from_quat(poses[:,3:7]))
def pose_at(ts):
    ts=int(np.clip(ts,odom_times[0],odom_times[-1]));T=np.eye(4)
    T[:3,:3]=rot((ts-odom_times[0])/1e9).as_matrix()
    T[:3,3]=[np.interp(ts,odom_times,poses[:,k]) for k in range(3)]
    return T
lid_times=np.array([r['stamp_ns']+clock[rc.TOPICS['lidar']]['header_to_receipt_offset_ns'] for r in clouds],dtype=np.int64)
world_chunks=[]
for row,ts in zip(clouds,lid_times):
    if not row['map_sample']:continue
    T=pose_at(ts)@Tbl;xyz=np.load(OUT/'lidar_samples'/row['file'])['xyz'];world=xyz@T[:3,:3].T+T[:3,3]
    p=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(world));p=p.voxel_down_sample(.08)
    world_chunks.append(np.asarray(p.points))
cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(world_chunks)))
cloud=cloud.voxel_down_sample(.08);o3d.io.write_point_cloud(str(OUT/'map_recorded_odometry.pcd'),cloud,compressed=True)
np.savez_compressed(OUT/'map_recorded_odometry.npz',xyz=np.asarray(cloud.points).astype(np.float32))
camera_poses=[];sample_report=[];color_points=[];color_values=[]
pairing_ids=np.array([i for i,r in enumerate(clouds) if r['index']%PAIRING_STRIDE==0],dtype=int)
pairing_times=lid_times[pairing_ids]
for row in cameras:
    ts=row['stamp_ns']+clock[rc.TOPICS['image']]['header_to_receipt_offset_ns']
    row['common_time_ns']=int(ts);Twb=pose_at(ts);Twc=Twb@Tbc;camera_poses.append(Twc)
    li=int(pairing_ids[np.argmin(np.abs(pairing_times-ts))]);lrow=clouds[li];Twl=pose_at(lid_times[li])@Tbl
    xyz=np.load(OUT/'lidar_pairing_samples'/lrow['file'])['xyz'];world=xyz@Twl[:3,:3].T+Twl[:3,3]
    pc=(world-Twc[:3,3])@Twc[:3,:3];good=(pc[:,2]>.3)&(pc[:,2]<25)
    pc=pc[good];world=world[good];uv=pc@K.T;uv=np.rint(uv[:,:2]/uv[:,2:3]).astype(int)
    keep=(uv[:,0]>=0)&(uv[:,0]<width)&(uv[:,1]>=0)&(uv[:,1]<height);uv=uv[keep];z=pc[keep,2];world=world[keep]
    depth=np.full((height,width),np.inf,dtype=np.float32);np.minimum.at(depth,(uv[:,1],uv[:,0]),z);depth[~np.isfinite(depth)]=0
    name=row['file'];np.save(OUT/'lidar_depth'/f'{name}.npy',depth)
    rgb=cv2.imread(str(OUT/'rgb_rectified'/f'{name}.png'))
    if row['index']%20==0:
        overlay=rgb.copy();colors=cv2.applyColorMap(np.clip(z/15*255,0,255).astype(np.uint8),cv2.COLORMAP_TURBO).reshape(-1,3)
        for p,c in zip(uv[::20],colors[::20]):cv2.circle(overlay,tuple(p),2,tuple(int(v) for v in c),-1)
        cv2.imwrite(str(OUT/'projection_panels'/f'{name}.jpg'),np.hstack([rgb,overlay]))
    color_points.append(world[::8]);color_values.append(rgb[uv[::8,1],uv[::8,0],::-1]/255.)
    sample_report.append({'frame':row['index'],'lidar_index':lrow['index'],'association_gap_ms':float(abs(lid_times[li]-ts)/1e6),'valid_depth_pixels':int((depth>0).sum())})
colored=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(color_points)));colored.colors=o3d.utility.Vector3dVector(np.concatenate(color_values));colored=colored.voxel_down_sample(.06)
o3d.io.write_point_cloud(str(OUT/'rgb_lidar_colored.pcd'),colored,compressed=True)
np.savez_compressed(OUT/'camera_poses.npz',poses=np.asarray(camera_poses),common_time_ns=np.array([r['common_time_ns'] for r in cameras],np.int64))
np.savez_compressed(OUT/'odometry.npz',common_time_ns=odom_times,pose_xyz_quaternion=poses)
(OUT/'camera_samples.json').write_text(json.dumps(cameras,indent=2))
(OUT/'projection_report.json').write_text(json.dumps(sample_report,indent=2))
(OUT/'intrinsics.json').write_text(json.dumps({'width':width,'height':height,'K':K.tolist(),'source_K':sourceK.tolist(),'D':D.tolist(),'rectified':True,'source':str(rc.calibration_path())},indent=2))
summary={'source_zip':str(DATA.with_suffix('.zip')),'source_dataset':str(DATA),'counts':all_counts,'selected_counts':counts,'selected_topics':rc.TOPICS,'rgb_samples':len(cameras),'camera_stride':rc.CAMERA_STRIDE,'lidar_stride':rc.LIDAR_STRIDE,'max_samples':rc.MAX_SAMPLES,'lidar_samples':sum(r['map_sample'] for r in clouds),'lidar_pairing_stride':PAIRING_STRIDE,'lidar_pairing_samples':sum(r['index']%PAIRING_STRIDE==0 for r in clouds),'lidar_cached_samples':len(clouds),'pairing_policy':'Nearest calibrated common-time scan selected from independent pairing timeline; geometric map stride does not coarsen synchronization','map_points':len(cloud.points),'colored_points':len(colored.points),'clock':clock,'base_from_camera':Tbc.tolist(),'base_from_lidar':Tbl.tolist(),'elapsed_s':time.time()-started,'calibration_policy':'Use the supplied matrix matching the recorded image dimensions and explicit base transforms; reject mismatched inputs.','time_policy':'Rebase each selected topic header clock to recorder time using its median receive-header offset; preserve raw clocks; compensate ego-motion using interpolated odometry. Network delays remain uncalibrated.'}
if rc.CONFIG.get('dataset',{}).get('prepared_observations'):
    summary.update(source_zip=None,source_input_kind='lossless_prepared_observations',source_manifest=str(DATA/'OBSERVATIONS_MANIFEST.json'))
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
