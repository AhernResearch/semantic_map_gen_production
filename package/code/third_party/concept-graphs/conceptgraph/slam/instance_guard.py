"""Local WhichGround instance-association guards; upstream baseline opt-out.

Co-observed disjoint 2D detections cannot form one foreground map instance.
This is a consistency constraint, not independent semantic/physical GT.
"""
from pathlib import Path
import math

def enabled(cfg):
    mode=cfg.get('association_mode','author_baseline')
    if mode not in ('author_baseline','guarded'):raise ValueError('association_mode must be author_baseline or guarded')
    return mode=='guarded'

def box_iou(a,b):
    a,b=[float(v) for v in a],[float(v) for v in b]
    if len(a)!=4 or len(b)!=4 or not all(math.isfinite(v) for v in a+b):return 0.
    area_a=max(0.,a[2]-a[0])*max(0.,a[3]-a[1]);area_b=max(0.,b[2]-b[0])*max(0.,b[3]-b[1])
    inter=max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
    union=area_a+area_b-inter
    return inter/union if union>0 else 0.

def observations(obj):
    paths=obj.get('color_path',[]);indices=obj.get('image_idx',[]);boxes=obj.get('xyxy',[])
    if len(boxes)!=len(paths):raise ValueError('Instance guard requires aligned color_path/xyxy observations')
    return [(str(p) if p is not None else 'image_idx:'+str(indices[i]),b) for i,(p,b) in enumerate(zip(paths,boxes))]

def same_frame_conflicts(cfg,a,b):
    if not enabled(cfg) or a.get('is_background',False) or b.get('is_background',False):return []
    threshold=float(cfg.get('instance_sameframe_min_iou',.5))
    if not 0<threshold<=1:raise ValueError('instance_sameframe_min_iou must be in (0,1]')
    right={}
    for frame,box in observations(b):right.setdefault(frame,[]).append(box)
    return [{'frame':frame,'iou':box_iou(box,other),'box_a':[float(v) for v in box],'box_b':[float(v) for v in other]} for frame,box in observations(a) for other in right.get(frame,[]) if box_iou(box,other)<threshold]

def compatible(cfg,a,b):return not same_frame_conflicts(cfg,a,b)

def unique_frame_count(obj):return len({p for p,_ in observations(obj)})
