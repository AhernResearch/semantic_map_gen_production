"""Lossless ROS2 observation arrays: no resize, thinning, pose changes or inference."""
from pathlib import Path,PurePosixPath
import json,hashlib,tarfile,os,time,heapq,shutil,zipfile
from types import SimpleNamespace as NS
import numpy as np
FORMATS={1:'i1',2:'u1',3:'<i2',4:'<u2',5:'<i4',6:'<u4',7:'<f4',8:'<f8'}
TYPES={'image':'sensor_msgs/msg/Image','lidar':'sensor_msgs/msg/PointCloud2','odom':'nav_msgs/msg/Odometry','imu':'sensor_msgs/msg/Imu'}
DEFAULT_TOPICS={'image':'/aima/hal/camera/front/image_raw','lidar':'/aima/hal/lidar/front/point_cloud','odom':'/map_odometry','imu':'/aima/hal/lidar/front/rsimu_data'}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def write(path,data):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.part');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(p)
def header(ns,frame):return NS(stamp=NS(sec=int(ns)//10**9,nanosec=int(ns)%10**9),frame_id=frame)
def safe(root,rel):
 p=PurePosixPath(rel)
 if p.is_absolute() or '..' in p.parts or '\\' in rel:raise ValueError('Invalid observation path')
 q=Path(root)/rel
 if q.is_symlink() or not q.resolve().is_relative_to(Path(root).resolve()):raise ValueError('Observation path outside selected input')
 return q

def export_bag(bag,output,calibration,labels=None,progress=lambda p:None):
 from rosbags.highlevel import AnyReader
 from rosbags.typesys import Stores,get_typestore
 import yaml
 bag=Path(bag).resolve();out=Path(output);out.mkdir(parents=True,exist_ok=False);index={r:[] for r in TYPES};values={r:[] for r in ['odom','imu']};source_meta=yaml.safe_load((bag/'metadata.yaml').read_text());source_files=[]
 info=source_meta.get('rosbag2_bagfile_information',{})
 for name in info.get('relative_file_paths',[]):
  p=safe(bag,name);s=p.stat();source_files.append(dict(name=name,size=s.st_size,mtime_ns=s.st_mtime_ns))
 if not source_files:raise ValueError('Recording metadata has no files')
 topics=dict(DEFAULT_TOPICS);started=time.monotonic();seq=0;message_info={};counts={};frame_info={}
 with AnyReader([bag],default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:
  for role in TYPES:
   conns=[c for c in reader.connections if c.topic==topics[role] and c.msgtype==TYPES[role]]
   if role!='imu' and not conns:raise ValueError('Missing recorded '+role+' topic')
   counts[role]=sum(c.msgcount for c in conns)
  selected=[c for c in reader.connections if c.topic in topics.values()]
  for c,receipt,raw in reader.messages(connections=selected):
   role=next(r for r,t in topics.items() if t==c.topic)
   if c.msgtype!=TYPES[role]:raise ValueError('Recorded topic/type mismatch')
   m=reader.deserialize(raw,c.msgtype);n=len(index[role]);stamp=int(m.header.stamp.sec)*10**9+int(m.header.stamp.nanosec);row=dict(receipt_ns=int(receipt),header_ns=stamp,frame_id=m.header.frame_id,source_index=n,sequence=seq);seq+=1
   if role=='image':
    if m.encoding.lower()!='nv12':raise ValueError('Current native contract requires recorded NV12 camera')
    expected=int(m.height)*3//2*int(m.step)
    if len(m.data)!=expected:raise ValueError('NV12 byte size/stride mismatch')
    row.update(width=int(m.width),height=int(m.height),step=int(m.step),encoding=m.encoding);rel=f'payload/camera/{n:06d}.npz';safe(out,rel).parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/rel,data=np.asarray(m.data,dtype='u1'));row['file']=rel
   elif role=='lidar':
    if m.is_bigendian:raise ValueError('Current native contract requires little-endian LiDAR')
    fields=[dict(name=f.name,offset=int(f.offset),datatype=int(f.datatype),count=int(f.count)) for f in m.fields];arrays={};N=int(m.width)*int(m.height)
    for k,f in enumerate(m.fields):
     if f.datatype not in FORMATS or f.count<1:raise ValueError('Unknown PointCloud2 field type')
     dtype=np.dtype(FORMATS[f.datatype]);a=np.ndarray((m.height,m.width,f.count),dtype=dtype,buffer=m.data,offset=f.offset,strides=(m.row_step,m.point_step,dtype.itemsize));arrays['field_'+str(k)]=a.reshape(N,f.count).copy()
    if not {'x','y','z'}.issubset({f['name'] for f in fields}):raise ValueError('LiDAR XYZ fields absent')
    row.update(width=int(m.width),height=int(m.height),point_step=int(m.point_step),row_step=int(m.row_step),fields=fields,is_dense=bool(m.is_dense),is_bigendian=False);rel=f'payload/lidar/{n:06d}.npz';safe(out,rel).parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/rel,**arrays);row['file']=rel
   elif role=='odom':
    p=m.pose.pose;row['child_frame_id']=m.child_frame_id;values[role].append([p.position.x,p.position.y,p.position.z,p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w,*m.pose.covariance,*m.twist.covariance,m.twist.twist.linear.x,m.twist.twist.linear.y,m.twist.twist.linear.z,m.twist.twist.angular.x,m.twist.twist.angular.y,m.twist.twist.angular.z])
   else:
    q=m.orientation;w=m.angular_velocity;a=m.linear_acceleration;values[role].append([q.x,q.y,q.z,q.w,w.x,w.y,w.z,a.x,a.y,a.z,*m.orientation_covariance,*m.angular_velocity_covariance,*m.linear_acceleration_covariance])
   index[role].append(row)
   if n%100==0:progress(dict(stage='extracting',role=role,completed=n+1,total=counts[role]))
 for role in ['odom','imu']:
  rel=f'payload/{role}.npz';safe(out,rel).parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/rel,values=np.asarray(values[role],dtype='<f8'));message_info[role]=rel
 for role in TYPES:
  if len(index[role])!=counts[role]:raise ValueError('Recording message counts changed during extraction')
  write(out/f'index/{role}.json',index[role])
 for entry in source_files:
  p=bag/entry['name'];s=p.stat()
  if s.st_size!=entry['size'] or s.st_mtime_ns!=entry['mtime_ns']:raise ValueError('Recording is still changing; stop recording before preprocessing')
  entry['sha256']=sha(p)
 (out/'calibration').mkdir(exist_ok=True);shutil.copy2(calibration,out/'calibration/recording_calibration.txt');shutil.copy2(bag/'metadata.yaml',out/'recording_metadata.yaml')
 if labels:shutil.copy2(labels,out/'annotations.source.yaml')
 files={p.relative_to(out).as_posix():dict(size=p.stat().st_size,sha256=sha(p)) for p in out.rglob('*') if p.is_file()}
 manifest=dict(schema_version='vln_lossless_observations_v1',source_files=source_files,source_bytes=sum(f['size'] for f in source_files),topics=topics,message_types=TYPES,message_counts=counts,index_files={r:'index/'+r+'.json' for r in TYPES},value_files=message_info,calibration_file='calibration/recording_calibration.txt',label_file='annotations.source.yaml' if labels else None,files=files,processor='ROS2 CDR decoding + exact field arrays + lossless NPZ; CPU; no ROS runtime or model needed',all_camera_messages_preserved=True,all_lidar_messages_preserved=True,all_imu_messages_preserved=True,images_resized=False,points_thinned=False,timestamps_or_poses_changed=False,elapsed_s=time.monotonic()-started,source_sequence_order_preserved=True,robot_motion_executed=False)
 write(out/'OBSERVATIONS_MANIFEST.json',manifest);archive=out/'bundle.vlnobs.tar.gz'
 with tarfile.open(archive,'w:gz',compresslevel=1,dereference=True) as z:
  for name in list(files)+['OBSERVATIONS_MANIFEST.json']:z.add(out/name,arcname=name,recursive=False)
 progress(dict(stage='ready',bytes=archive.stat().st_size));return dict(bundle=str(archive),sha256=sha(archive),size=archive.stat().st_size,source_bytes=manifest['source_bytes'],message_counts=counts,manifest=str(out/'OBSERVATIONS_MANIFEST.json'))

def validate(root,hash_files=True):
 root=Path(root);m=json.loads((root/'OBSERVATIONS_MANIFEST.json').read_text())
 if m.get('schema_version')!='vln_lossless_observations_v1':raise ValueError('Not a lossless observation input bundle')
 for role in ['image','lidar','odom']:
  if m.get('message_counts',{}).get(role,0)<1:raise ValueError('Input missing '+role)
 for name,f in m['files'].items():
  p=safe(root,name)
  if not p.is_file() or p.stat().st_size!=f['size'] or hash_files and sha(p)!=f['sha256']:raise ValueError('Observation member is missing or changed: '+name)
 safe(root,m['calibration_file']);return m

def unpack(source,dest):
 source=Path(source).resolve();dest=Path(dest)
 if source.is_dir():validate(source);return source
 dest.mkdir(parents=True,exist_ok=False)
 if zipfile.is_zipfile(source):
  with zipfile.ZipFile(source) as z:
   seen=set();total=0
   for entry in z.infolist():
    if entry.is_dir():continue
    if entry.filename in seen or (entry.external_attr>>16)&0o170000==0o120000:raise ValueError('Input requires unique regular files')
    seen.add(entry.filename);total+=entry.file_size
    if total>150*1024**3:raise ValueError('Observation input too large')
    p=safe(dest,entry.filename);p.parent.mkdir(parents=True,exist_ok=True)
    with z.open(entry) as a,p.open('wb') as b:shutil.copyfileobj(a,b)
 else:
  with tarfile.open(source,'r:*') as z:
   names=set();total=0
   for member in z:
    if member.isdir():continue
    if not member.isfile() or member.name in names:raise ValueError('Input requires unique regular files')
    names.add(member.name);total+=member.size
    if total>150*1024**3:raise ValueError('Observation input too large')
    p=safe(dest,member.name);p.parent.mkdir(parents=True,exist_ok=True)
    with z.extractfile(member) as a,p.open('wb') as b:shutil.copyfileobj(a,b)
 validate(dest);return dest

class BundleReader:
 def __init__(self,paths,default_typestore=None):
  root=Path(paths[0]);self.root=root;self.manifest=validate(root,False);m=self.manifest;self.connections=[NS(topic=m['topics'][r],msgtype=m['message_types'][r],msgcount=m['message_counts'][r],role=r) for r in TYPES if m['message_counts'][r]];self.rows={r:json.loads(safe(root,m['index_files'][r]).read_text()) for r in TYPES};self.values={r:np.load(safe(root,m['value_files'][r]),allow_pickle=False)['values'] for r in ['odom','imu']}
 def __enter__(self):return self
 def __exit__(self,*args):pass
 def messages(self,connections=None):
  connections=self.connections if connections is None else connections;rows=[]
  for c in connections:
   if len(self.rows[c.role])!=c.msgcount:raise ValueError('Observation index/message count mismatch')
   rows+= [(r['receipt_ns'],r['sequence'],c,r) for r in self.rows[c.role]]
  for stamp,seq,c,row in sorted(rows,key=lambda v:(v[0],v[1])):yield c,stamp,(c.role,row)
 def deserialize(self,raw,msgtype):
  role,r=raw;h=header(r['header_ns'],r['frame_id'])
  if role=='image':
   with np.load(safe(self.root,r['file']),allow_pickle=False) as z:data=z['data']
   return NS(header=h,width=r['width'],height=r['height'],step=r['step'],encoding=r['encoding'],data=data)
  if role=='lidar':
   names=[f['name'] for f in r['fields']];formats=[(np.dtype(FORMATS[f['datatype']]),(f['count'],)) if f['count']>1 else np.dtype(FORMATS[f['datatype']]) for f in r['fields']];dtype=np.dtype(dict(names=names,formats=formats,offsets=[f['offset'] for f in r['fields']],itemsize=r['point_step']));buf=bytearray(r['height']*r['row_step']);a=np.ndarray((r['height'],r['width']),dtype=dtype,buffer=buf,strides=(r['row_step'],r['point_step']))
   with np.load(safe(self.root,r['file']),allow_pickle=False) as z:
    for i,f in enumerate(r['fields']):a[f['name']]=z['field_'+str(i)].reshape((r['height'],r['width'],f['count']) if f['count']>1 else (r['height'],r['width']))
   return NS(header=h,width=r['width'],height=r['height'],point_step=r['point_step'],row_step=r['row_step'],fields=[NS(**f) for f in r['fields']],data=buf,is_bigendian=False,is_dense=r['is_dense'])
  v=self.values[role][r['source_index']]
  if role=='odom':
   p=NS(position=NS(x=v[0],y=v[1],z=v[2]),orientation=NS(x=v[3],y=v[4],z=v[5],w=v[6]));tw=NS(linear=NS(x=v[79],y=v[80],z=v[81]),angular=NS(x=v[82],y=v[83],z=v[84]));return NS(header=h,child_frame_id=r['child_frame_id'],pose=NS(pose=p,covariance=v[7:43]),twist=NS(twist=tw,covariance=v[43:79]))
  return NS(header=h,orientation=NS(x=v[0],y=v[1],z=v[2],w=v[3]),angular_velocity=NS(x=v[4],y=v[5],z=v[6]),linear_acceleration=NS(x=v[7],y=v[8],z=v[9]),orientation_covariance=v[10:19],angular_velocity_covariance=v[19:28],linear_acceleration_covariance=v[28:37])
