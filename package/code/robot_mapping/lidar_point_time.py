"""Preserve LiDAR fields and conditionally compensate intra-scan ego motion.

Point clocks must agree with the scan-header domain. Clock rebasing/extrinsics
remain the supplied recording assumptions, not independently surveyed truth.
"""
import numpy as np
from scipy.spatial.transform import Rotation,Slerp

FORMATS={1:'i1',2:'u1',3:'<i2',4:'<u2',5:'<i4',6:'<u4',7:'<f4',8:'<f8'}

def decode_cloud(message,min_range=.3,max_range=30.):
    if message.is_bigendian:raise ValueError('Little-endian PointCloud2 required')
    fields={f.name:f for f in message.fields}
    names=[n for n in ['x','y','z','intensity','ring','timestamp'] if n in fields]
    if not all(n in fields for n in ['x','y','z']):raise ValueError('Missing XYZ')
    if any(fields[n].count!=1 or fields[n].datatype not in FORMATS for n in names):raise ValueError('Unsupported scalar field')
    dtype=np.dtype({'names':names,'formats':[FORMATS[fields[n].datatype] for n in names],
                    'offsets':[fields[n].offset for n in names],'itemsize':message.point_step})
    cloud=np.ndarray((message.height,message.width),dtype=dtype,buffer=message.data,strides=(message.row_step,message.point_step))
    xyz=np.column_stack([cloud[n].ravel() for n in ['x','y','z']])
    distance=np.linalg.norm(xyz,axis=1);valid=np.isfinite(xyz).all(1)&(distance>min_range)&(distance<max_range)
    payload={'xyz':xyz[valid],'xyz_raw':xyz[valid].copy(),'source_point_index':np.flatnonzero(valid),
             'raw_point_count':np.asarray(len(xyz))}
    for name in ['intensity','ring','timestamp']:
        if name in fields:payload['raw_'+name]=cloud[name].ravel()[valid].copy()
    return payload

def deskew_to_header(xyz,point_seconds,header_ns,offset_ns,odom_ns,poses,base_from_lidar):
    xyz=np.asarray(xyz,dtype=float);seconds=np.asarray(point_seconds,dtype=float)
    header_seconds=header_ns/1e9
    if seconds.shape!=(len(xyz),) or not np.isfinite(seconds).all():raise ValueError('Finite per-point timestamps required')
    relative=seconds-header_seconds
    if len(relative) and (relative.min()<-.25 or relative.max()>.02):raise ValueError('Point time does not match scan-header clock domain')
    common_header=int(header_ns)+int(offset_ns);relative_ns=np.rint(relative*1e9).astype(np.int64)
    times=common_header+relative_ns;inside=(times>=odom_ns[0])&(times<=odom_ns[-1])
    if not inside.all():raise ValueError('Point-time pose support does not cover this scan; no clamping/extrapolation accepted')
    zero=int(odom_ns[0]);axis=(odom_ns-zero)/1e9;query=(times-zero)/1e9
    slerp=Slerp(axis,Rotation.from_quat(poses[:,3:7]));rotations=slerp(query)
    lidar=np.asarray(base_from_lidar,float);base=xyz@lidar[:3,:3].T+lidar[:3,3]
    translation=np.column_stack([np.interp(query,axis,poses[:,k]) for k in range(3)])
    world=rotations.apply(base)+translation
    ht=(common_header-zero)/1e9
    if not axis[0]<=ht<=axis[-1]:raise ValueError('Header pose outside odometry')
    R=slerp(ht).as_matrix()@lidar[:3,:3]
    t=np.array([np.interp(ht,axis,poses[:,k]) for k in range(3)])+slerp(ht).apply(lidar[:3,3])
    corrected=(world-t)@R
    return corrected.astype(np.float32),{'method':'point-time to scan-header rigid frame','point_count':len(xyz),
          'point_minus_header_seconds_min_max':[float(relative.min()),float(relative.max())],
          'correction_sensor_distance_p50_p90_m':np.percentile(np.linalg.norm(corrected-xyz,axis=1),[50,90]).tolist()}
