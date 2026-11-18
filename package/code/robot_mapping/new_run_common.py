"""Shared configuration and fingerprint helpers for new recordings."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import tempfile
import yaml

PACKAGE=Path(__file__).resolve().parents[2]
CODE=PACKAGE/'code'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def value_hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=path.parent,prefix=path.name+'.',suffix='.part',delete=False) as f:
        f.write(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temporary=Path(f.name)
    temporary.replace(path)
def scene_name(name):
    if not isinstance(name,str) or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*',name):raise ValueError('scene_id must be a simple ASCII identifier')
    return name
def protected_output(path):
    p=Path(path).expanduser().resolve()
    if p.is_relative_to(CODE) or p.is_relative_to(PACKAGE.parent/'real'):raise ValueError('New outputs cannot be inside code/ or the historical 700GB real workspace')
    return p
def load_config(path):
    path=Path(path).expanduser().resolve();cfg=yaml.safe_load(path.read_text())
    scene_name(cfg['scene_id']);workspace=protected_output(resolve(path,cfg['workspace']))
    dataset=resolve(path,cfg['dataset']['directory'])
    if dataset.is_relative_to(PACKAGE.parent/'real'):raise ValueError('New recordings must not depend on historical incoming/real')
    bag_value=cfg['dataset'].get('bag')
    if bag_value:bag=(dataset/Path(bag_value).expanduser()).resolve()
    else:
        found=[p.parent for p in dataset.rglob('metadata.yaml')]
        if len(found)!=1:raise ValueError('Set dataset.bag explicitly for multiple/no bags')
        bag=found[0]
    calibration=(dataset/Path(cfg['dataset'].get('calibration','extrinst.txt')).expanduser()).resolve()
    return cfg,path,workspace,dataset,bag,calibration
def resolve(config,value):
    p=Path(value).expanduser();return p.resolve() if p.is_absolute() else (Path(config).parent/p).resolve()

class StepLedger:
    """Resume only exact input/source/parameter/output content identities."""
    def __init__(self,path):
        self.path=Path(path);self.data=json.loads(self.path.read_text()) if self.path.exists() else {'schema_version':1,'steps':[]}
    def matching(self,name,signature,outputs):
        current={str(Path(p).resolve()):sha(p) for p in outputs if Path(p).is_file()}
        if len(current)!=len(outputs):return False
        return any(s.get('name')==name and s.get('signature')==signature and s.get('exit_code')==0 and s.get('outputs')==current for s in reversed(self.data['steps']))
    def record(self,name,signature,outputs,*,exit_code=0,reused=False,details=None):
        row={'name':name,'signature':signature,'exit_code':exit_code,'reused':reused,'outputs':{str(Path(p).resolve()):sha(p) for p in outputs if Path(p).is_file()},'details':details or {}}
        self.data['steps'].append(row);atomic_json(self.path,self.data);return row
