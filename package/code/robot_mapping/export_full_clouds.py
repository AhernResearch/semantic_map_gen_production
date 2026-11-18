"""Export every original/native point to web chunks, with no voxel or point sampling."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import os
import pickle
import shutil
import numpy as np
import open3d as o3d

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()

def export(root,key,parts,source_paths,yup=False,note='',download_source=None):
    folder=root/'viewer/full'/key;folder.mkdir(parents=True,exist_ok=True)
    chunks=[];count=0;invalid=0;lo=np.full(3,np.inf);hi=-lo
    for xyz,colors in parts:
        xyz=np.asarray(xyz);colors=np.asarray(colors) if colors is not None else None
        good=np.isfinite(xyz).all(1);invalid+=int((~good).sum());xyz=xyz[good].copy()
        if colors is not None:colors=colors[good]
        if yup:xyz=xyz@np.array([[1,0,0],[0,0,-1],[0,1,0]],float).T
        for start in range(0,len(xyz),500000):
            points=xyz[start:start+500000]
            if not len(points):continue
            rgb=colors[start:start+500000] if colors is not None else np.tile([.45,.6,.72],(len(points),1))
            rgb=np.clip(np.nan_to_num(rgb),0,1)
            data=np.column_stack([points,rgb]).astype('<f4')
            file=folder/f'{len(chunks):04d}.bin';temporary=file.with_suffix('.part');data.tofile(temporary);temporary.replace(file)
            chunks.append({'file':str(file.relative_to(root/'viewer')),'points':len(points),'bytes':file.stat().st_size,'sha256':digest(file)})
            count+=len(points);lo=np.minimum(lo,points.min(0));hi=np.maximum(hi,points.max(0))
    assert count>0
    download=None
    if download_source:
        src=Path(download_source);dst=folder/('original'+src.suffix)
        if not dst.exists():
            try:os.link(src,dst)
            except OSError:shutil.copy2(src,dst)
        download='/viewer/'+str(dst.relative_to(root/'viewer'))
    result={'points':count,'chunks':chunks,'layout':'xyzrgb_f32','bounds_min':lo.tolist(),'bounds_max':hi.tolist(),
            'invalid_coordinates_removed':invalid,'display_point_sampling':False,'extra_voxel_downsampling':False,
            'note':note,'source_files':[{'path':str(p),'sha256':digest(Path(p))} for p in source_paths],
            'download':download,'complete':True}
    (folder/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    return result

def cloud_parts(paths):
    for path in paths:
        cloud=o3d.io.read_point_cloud(str(path));yield np.asarray(cloud.points),np.asarray(cloud.colors) if cloud.has_colors() else None

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',type=Path,required=True);p.add_argument('--provided-pcd',type=Path);args=p.parse_args()
    root=args.workspace.resolve();data_path=root/'viewer/data.js';data=json.loads(data_path.read_text().removeprefix('window.DATA=').strip().removesuffix(';'))
    outputs={}
    for key,name,note in [('colored','rgb_lidar_colored.pcd','原PCD文件全部点；文件本身已有投影采样及6厘米体素融合。'),('lidar','map_recorded_odometry.pcd','原PCD文件全部点；文件本身已有每4帧雷达抽样和8厘米体素融合。')]:
        src=root/'supplied_dataset'/name;outputs[key]=export(root,key,cloud_parts([src]),[src],note=note,download_source=src)
    validation=json.loads((root/'native_maps/validation.json').read_text())
    graph=Path(validation['HMSG']['graph']);paths=sorted((graph/'objects').glob('*.ply'))
    outputs['hmsg_today']=export(root,'hmsg_today',cloud_parts(paths),paths,yup=True,note='原生HMSG对象PLY的全部点，无额外5厘米预览处理；数量为对象点数合计，保存的RGB颜色。')
    src=Path(validation['ConceptGraphs']['native'])
    with gzip.open(src) as f:objects=pickle.load(f)['objects']
    parts=((o['pcd_np'],o.get('pcd_color_np')) for o in objects)
    outputs['cg_today']=export(root,'cg_today',parts,[src],yup=True,note='原生过滤后CG实例的全部点；建图器本身使用2.5厘米体素及实例过滤。')
    import h5py
    src=Path(validation['VLMaps']['native'])
    with h5py.File(src) as f:
        xyz=f['grid_pos'][:]*float(f['cs'][()])+f['pcd_min'][:];rgb=f['grid_rgb'][:].astype(float)/255
    outputs['vlmaps_today']=export(root,'vlmaps_today',[(xyz,rgb)],[src],yup=True,note='全部VLMaps特征体素，保留文件中每个0.1米体素；没有12万点网页上限。')
    scene=Path(validation['HMSG']['graph']).parent.name
    src=Path(validation['HMSG']['graph']).parent/(scene+'_rgb.ply')
    if src.exists():
        outputs['ovo_full']=export(root,'ovo_full',cloud_parts([src]),[src],yup=True,note='原生OVO RGB点云文件全部点；来源为图像/深度建图，非雷达原始扫描。',download_source=src)
        data['clouds'].append({'key':'ovo_full','title':'今天：OVO原生RGB全点云','points':outputs['ovo_full']['points'],'preview_file_points':outputs['ovo_full']['points'],'display_points':0,'file':None})
    if args.provided_pcd:
        src=args.provided_pcd.resolve();outputs['provided_pcd']=export(root,'provided_pcd',cloud_parts([src]),[src],note='随数据包提供的1012 SLAM原PCD全部点；独立地图坐标，未与今日语义地图配准。',download_source=src)
        data['clouds'].append({'key':'provided_pcd','title':'随包：1012 SLAM原PCD全点云','points':outputs['provided_pcd']['points'],'preview_file_points':outputs['provided_pcd']['points'],'display_points':0,'file':None})
    for cloud in data['clouds']:
        if cloud['key'] in outputs:cloud['full']=outputs[cloud['key']]
    data['clouds']=list({c['key']:c for c in data['clouds']}.values())
    temporary=data_path.with_suffix('.tmp');temporary.write_text('window.DATA='+json.dumps(data,ensure_ascii=False,separators=(',',':'))+';\n');temporary.replace(data_path)
    (root/'viewer/full_clouds_manifest.json').write_text(json.dumps(outputs,ensure_ascii=False,indent=2))
    print(json.dumps({k:{'points':v['points'],'chunks':len(v['chunks']),'invalid':v['invalid_coordinates_removed']} for k,v in outputs.items()},indent=2))

if __name__=='__main__':main()
