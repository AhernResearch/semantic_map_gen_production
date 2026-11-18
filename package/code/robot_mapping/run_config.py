"""Paths/settings shared by the portable real-robot mapping stages."""
from pathlib import Path
import os
import re
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PACKAGE_ROOT / 'code'
THIRD_PARTY = CODE_ROOT / 'third_party'
config_path = os.environ.get('WHICHGROUND_RUN_CONFIG')
if not config_path:
    raise RuntimeError('Use kit/run_robot_mapping.sh --config <yaml> --stage <stage>')
CONFIG_PATH = Path(config_path).expanduser().resolve()
CONFIG = yaml.safe_load(CONFIG_PATH.read_text())

def path_value(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (CONFIG_PATH.parent / path).resolve()

WORKSPACE = path_value(os.environ.get('WHICHGROUND_WORKSPACE', CONFIG['workspace']))
SCENE_ID = CONFIG['scene_id']
if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', SCENE_ID):
    raise ValueError('scene_id must be a simple directory name')
if WORKSPACE.is_relative_to(PACKAGE_ROOT.parent / 'real'):
    raise ValueError('The historical 700GB workspace is not a new mapping output directory')
DATASET = path_value(CONFIG['dataset']['directory'])
MODEL_ROOT = Path(os.environ.get('SEMANTIC_GEN_MODEL_ROOT', PACKAGE_ROOT / 'data/models/semantic_mapping')).expanduser().resolve()
TOPICS = {
    'image': '/aima/hal/camera/front/image_raw',
    'lidar': '/aima/hal/lidar/front/point_cloud',
    'odom': '/map_odometry',
}
TOPICS.update(CONFIG['dataset'].get('topics', {}))
if set(TOPICS) != {'image', 'lidar', 'odom'} or any(not isinstance(v, str) or not v.startswith('/') for v in TOPICS.values()):
    raise ValueError('dataset.topics must contain image/lidar/odom ROS topic names starting with /')
if len(set(TOPICS.values())) != 3:
    raise ValueError('image, lidar and odom must use distinct topics')
DA3_MODEL = path_value(CONFIG.get('depth', {}).get('model', MODEL_ROOT / 'DA3METRIC-LARGE'))
CACHE_ROOT = Path(os.environ.get('SEMANTIC_GEN_CACHE_ROOT', PACKAGE_ROOT.parent / 'cache')).expanduser().resolve()
CLIP_CACHE = CACHE_ROOT / 'open_clip'
CAMERA_STRIDE = int(CONFIG.get('sampling', {}).get('camera_stride', 15))
LIDAR_STRIDE = int(CONFIG.get('sampling', {}).get('lidar_stride', 4))
MAX_SAMPLES = int(CONFIG.get('sampling', {}).get('max_samples', 0))
HMSG_STRIDE = int(CONFIG.get('hmsg', {}).get('stride', 5))
CG_STRIDE = int(CONFIG.get('conceptgraphs', {}).get('stride', 4))
VLMAPS_STRIDE = int(CONFIG.get('vlmaps', {}).get('stride', 4))
if min(CAMERA_STRIDE, LIDAR_STRIDE, HMSG_STRIDE, CG_STRIDE, VLMAPS_STRIDE) < 1:
    raise ValueError('All sampling strides must be positive')

def stage_input():
    return WORKSPACE / 'mapper_inputs/fsrvln' / SCENE_ID

def sample_count():
    return len(list((stage_input() / 'rgb').glob('*.png')))

def bag_path():
    value = CONFIG['dataset'].get('bag')
    if value:
        return DATASET / value
    bags = [p.parent for p in DATASET.rglob('metadata.yaml')]
    if len(bags) != 1:
        raise ValueError('Set dataset.bag explicitly when dataset contains zero/multiple bags')
    return bags[0]

def calibration_path():
    return DATASET / CONFIG['dataset'].get('calibration', 'extrinst.txt')

WORKSPACE.mkdir(parents=True, exist_ok=True)
(WORKSPACE / 'native_configs').mkdir(exist_ok=True)
