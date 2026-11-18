import run_config as rc
"""Validate actual native map outputs and export explicit preview derivatives."""
from pathlib import Path
from collections import Counter
import json
import gzip
import pickle
import re
import numpy as np
import open3d as o3d
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = rc.WORKSPACE
OUT = ROOT / 'native_maps/preview'
OUT.mkdir(parents=True, exist_ok=True)
checks = []

def check(name, condition, detail=None):
    checks.append({'name': name, 'passed': bool(condition), 'detail': detail})

def as_array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)

def valid_xyz(x):
    return x.ndim == 2 and x.shape[1] == 3 and len(x) > 0 and np.isfinite(x).all()

def point_cloud(xyz, colors):
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud

def save_preview(key, cloud):
    cloud = cloud.voxel_down_sample(.05)
    check(key + ' preview geometry', valid_xyz(np.asarray(cloud.points)))
    o3d.io.write_point_cloud(str(OUT / (key + '.pcd')), cloud, compressed=True)
    return cloud

poses = np.loadtxt(ROOT / 'scene_packages' / rc.SCENE_ID / 'observations/poses_vlmaps.txt').reshape(-1, 4, 4)
cg_poses = np.loadtxt(ROOT / 'native_maps/ConceptGraphs/data' / rc.SCENE_ID / 'traj.txt').reshape(-1, 4, 4)
hmsg_poses = np.array([np.loadtxt(p).reshape(4, 4) @ np.diag([1, -1, -1, 1])
                      for p in sorted((rc.stage_input() / 'pose').glob('*.txt'))])
frame_count=rc.sample_count()
check('Shared finite camera poses', poses.shape == (frame_count, 4, 4) and np.isfinite(poses).all())
check('ConceptGraphs shared world coordinates', np.allclose(poses, cg_poses, atol=1e-7))
check('HMSG shared world coordinates', np.allclose(poses, hmsg_poses, atol=1e-7))
check('Proper world camera rotations', np.allclose(np.linalg.det(poses[:, :3, :3]), 1, atol=1e-6))

hroot = ROOT / 'native_maps/HMSG' / rc.SCENE_ID
graph = sorted(hroot.glob('graph_*'))[-1]
records = {folder: [json.loads(p.read_text()) for p in sorted((graph / folder).glob('*.json'))]
           for folder in ['floors', 'rooms', 'views', 'objects']}
ids = {folder: {str(x[folder[:-1] + '_id']) for x in rows}
       for folder, rows in records.items()}
for folder, rows in records.items():
    check('HMSG ' + folder + ' present/unique', len(rows) > 0 and len(rows) == len(ids[folder]))
hcloud = o3d.geometry.PointCloud()
for i, obj in enumerate(records['objects']):
    emb = np.asarray(obj['embedding'])
    check('HMSG embedding ' + obj['object_id'], emb.shape == (1024,) and np.isfinite(emb).all() and np.linalg.norm(emb) > 0)
    check('HMSG object links ' + obj['object_id'], str(obj['room_id']) in ids['rooms'] and set(obj['view_ids']) <= ids['views'] and (obj['best_view_id'] is None or obj['best_view_id'] in ids['views']))
    cloud = o3d.io.read_point_cloud(str(graph / 'objects' / (obj['object_id'] + '.ply')))
    check('HMSG geometry ' + obj['object_id'], valid_xyz(np.asarray(cloud.points)))
    cloud.paint_uniform_color(plt.get_cmap('tab20')(i % 20)[:3])
    hcloud += cloud
for room in records['rooms']:
    check('HMSG room links ' + room['room_id'], str(room['floor_id']) in ids['floors'] and set(room['objects']) <= ids['objects'] and set(room['views']) <= ids['views'])
    emb = np.asarray(room['embeddings'])
    check('HMSG room features ' + room['room_id'], emb.size > 0 and np.isfinite(emb).all())
for view in records['views']:
    check('HMSG view links ' + view['view_id'], str(view['room_id']) in ids['rooms'] and set(view['object_ids']) <= ids['objects'])
    image = Path(view['img_path'])
    check('HMSG view image ' + view['view_id'], image.is_file() and int(image.stem) == int(view['img_id']))
for floor in records['floors']:
    check('HMSG floor links ' + str(floor['floor_id']), set(floor['rooms']) <= ids['rooms'])
hpoints = len(hcloud.points)
hpreview = save_preview('hmsg_today', hcloud)
hstat = {'graph': str(graph), **{k: len(v) for k, v in records.items()},
         'native_object_points': hpoints, 'preview_points': len(hpreview.points),
         'label_counts': dict(Counter(o['name'] for o in records['objects'])),
         'mapping_observations':len(range(0,frame_count,rc.HMSG_STRIDE)), 'instance_feature_dimension': 1024}

cpath = ROOT / 'native_maps/ConceptGraphs/data' / rc.SCENE_ID / 'pcd_saves/full_pcd_yolo_world_new_robot_post.pkl.gz'
with gzip.open(cpath) as stream:
    cg = pickle.load(stream)
check('ConceptGraphs complete input range', cg['cfg']['stride'] == rc.CG_STRIDE and cg['cfg']['end'] == -1)
check('ConceptGraphs objects present', len(cg['objects']) > 0)
ccloud = o3d.geometry.PointCloud()
cobjects = []
for i, obj in enumerate(cg['objects']):
    xyz = as_array(obj['pcd_np'])
    feat = as_array(obj['clip_ft'])
    label = Counter(obj['class_name']).most_common(1)[0][0]
    check('ConceptGraphs geometry ' + str(i), valid_xyz(xyz))
    check('ConceptGraphs feature ' + str(i), feat.shape == (1024,) and np.isfinite(feat).all() and np.linalg.norm(feat) > 0)
    check('ConceptGraphs observations ' + str(i), obj['num_detections'] >= 2 and all(Path(p).is_file() for p in obj['color_path']))
    ccloud += point_cloud(xyz, np.tile(plt.get_cmap('tab20')(i % 20)[:3], (len(xyz), 1)))
    sample_ids = sorted({int(re.search(r'frame(\d+)', Path(p).name).group(1)) for p in obj['color_path']})
    cobjects.append({'object_id': i, 'label': label, 'num_detections': int(obj['num_detections']),
                     'points': len(xyz), 'center_yup': xyz.mean(0).tolist(),
                     'bbox_yup': as_array(obj['bbox_np']).tolist(),
                     'sample_frame_ids': sample_ids, 'source_camera_frame_ids': [v * rc.CAMERA_STRIDE for v in sample_ids]})
cpoints = len(ccloud.points)
cpreview = save_preview('cg_today', ccloud)
(OUT / 'conceptgraphs_objects.json').write_text(json.dumps(cobjects, ensure_ascii=False, indent=2))
cstat = {'native': str(cpath), 'objects': len(cobjects), 'mapping_observations':len(range(0,frame_count,rc.CG_STRIDE)),
         'native_object_points': cpoints, 'preview_points': len(cpreview.points),
         'instance_feature_dimension': 1024,
         'label_counts': dict(Counter(o['label'] for o in cobjects))}

vpath = ROOT / 'native_maps/VLMaps' / rc.SCENE_ID / 'vlmap_cam/vlmaps_cam.h5df'
with h5py.File(vpath) as hdf:
    feat = hdf['grid_feat'][:]
    pos = hdf['grid_pos'][:]
    rgb = hdf['grid_rgb'][:]
    weight = hdf['weight'][:]
    lo = hdf['pcd_min'][:]
    cell = float(hdf['cs'][()])
    mapped = sorted(hdf['mapped_iter_list'][:].tolist())
    occ = hdf['occupied_ids'][:]
    check('VLMaps all selected observations', mapped == list(range(0,frame_count,rc.VLMAPS_STRIDE)))
    check('VLMaps features present/finite', feat.ndim == 2 and feat.shape[0] > 0 and feat.shape[1] == 512 and np.isfinite(feat).all())
    check('VLMaps nonzero features', np.all(np.linalg.norm(feat, axis=1) > 1e-8))
    check('VLMaps positive finite weights', np.isfinite(weight).all() and np.all(weight > 0))
    check('VLMaps carrier matching sizes', len(feat) == len(pos) == len(rgb) == len(weight))
    check('VLMaps voxel bounds', np.all(pos >= 0) and np.all(pos < np.asarray(occ.shape)))
    check('VLMaps unique voxel indices', len(np.unique(pos, axis=0)) == len(pos))
    check('VLMaps occupied id correspondence', np.array_equal(occ[tuple(pos.T)], np.arange(len(pos))))
xyz = pos * cell + lo
vpreview = save_preview('vlmaps_today', point_cloud(xyz, rgb.astype(float) / 255))
vstat = {'native': str(vpath), 'feature_shape': list(feat.shape), 'mapped_observations': len(mapped),
         'voxel_size_m': cell, 'pcd_min': lo.tolist(), 'voxel_grid_shape': list(occ.shape),
         'preview_points': len(vpreview.points)}

fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
for ax, cloud, title in zip(axes, [hpreview, cpreview, vpreview],
        [f"HMSG: {hstat['objects']} predicted objects / {hstat['rooms']} regions",
         f"ConceptGraphs: {cstat['objects']} predicted instances",
         f"VLMaps: {len(feat):,} voxels x 512 features"]):
    p = np.asarray(cloud.points); c = np.asarray(cloud.colors)
    idx = np.linspace(0, len(p) - 1, min(len(p), 120000), dtype=int)
    ax.scatter(p[idx, 0], p[idx, 2], c=np.clip(c[idx], 0, 1), s=.3, rasterized=True)
    ax.set_aspect('equal'); ax.set_title(title, fontsize=10)
    ax.set_xlabel('Shared world X / m'); ax.set_ylabel('Shared world Z / m')
    ax.grid(alpha=.15)
fig.suptitle(rc.SCENE_ID+' | native map outputs | predicted semantics, not accuracy validation')
fig.savefig(OUT / 'three_native_maps.png', dpi=160); plt.close(fig)

result = {'dataset':rc.SCENE_ID,'sample_frames':frame_count,
          'coordinate_convention': 'Shared Y-up world; horizontal XZ; OpenCV camera poses',
          'HMSG': hstat, 'ConceptGraphs': cstat, 'VLMaps': vstat,
          'checks': checks, 'all_passed': all(c['passed'] for c in checks),
          'warnings': ['Counts are predicted fragments/instances, not verified ground-truth objects.',
                       'HMSG default ScanNet vocabulary can mislabel industrial scenes and oversegment background.',
                       'HMSG floor/room regions are inferred from incomplete depth; not surveyed architectural boundaries.',
                       'Map coordinates depend on recorded odometry, supplied extrinsics, clock rebasing and LiDAR-fitted DA3 depth.',
                       'Native upstream source was patched; exact author legacy pipeline and paper metrics have not been reproduced.',
                       'Voxel/geometry/feature checks establish usable data structure, not navigation clearance or semantic accuracy.']}
(ROOT / 'native_maps/validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps({k: v for k, v in result.items() if k != 'checks'}, ensure_ascii=False, indent=2))
print('CHECKS', len(checks), 'FAILED', [c['name'] for c in checks if not c['passed']])
if not result['all_passed']:
    raise SystemExit(1)
