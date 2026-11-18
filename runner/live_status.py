"""Actual stage/loop counters and resource activity for the local mapping GUI."""
from __future__ import annotations
import datetime as dt,json,os,re,threading,time
from pathlib import Path
from runtime_profile import gpu_sample,stamp

STAGES=[('doctor','环境与模型检查'),('prepare','解码录制数据／雷达投影'),('features','图像语义特征'),('depth','深度推理'),('align','深度与坐标对齐检查'),('scene','准备三图输入'),('hmsg','HMSG建图'),('cg_prepare','ConceptGraphs输入'),('cg_detect','ConceptGraphs物体检测'),('cg_map','ConceptGraphs融合'),('vlmaps','VLMaps建图'),('validate','三图文件检查'),('preview','生成可视化'),('package','打包原生三图')]
EXTRA={'input_validation':'检查录制与标定','configuration_ready':'配置已准备','label_preparation':'读取标签YAML／设置有限类别','prepared_not_mapped':'仅检查完成，尚未建图','native_three_map_pipeline':'校验源码、数据与权重／启动流水线','label_alignment':'三图有限标签对齐','collecting_outputs':'收集三图／打包结果','complete':'完成','failed':'失败','cancelled':'已中止'}
def load(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return default
def parse_time(value):
    if not value:return None
    try:return dt.datetime.fromisoformat(value.replace('Z','+00:00')).timestamp()
    except ValueError:
        try:return dt.datetime.strptime(value,'%Y%m%dT%H%M%S.%fZ').replace(tzinfo=dt.timezone.utc).timestamp()
        except ValueError:return None
def write(path,value):
    path=Path(path);temp=path.with_name(path.name+'.part');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));temp.replace(path)
def tail(path,limit=65536):
    try:
        with Path(path).open('rb') as stream:stream.seek(0,2);size=stream.tell();stream.seek(max(0,size-limit));raw=stream.read().decode('utf-8',errors='replace')
        return re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',raw)
    except OSError:return ''
def tree_counters(root_pid):
    infos={};ticks=os.sysconf('SC_CLK_TCK');pages=os.sysconf('SC_PAGE_SIZE')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            raw=(proc/'stat').read_text();parts=raw[raw.rfind(')')+2:].split();pid=int(proc.name)
            infos[pid]=(int(parts[1]),parts,proc)
        except (OSError,ValueError,IndexError):continue
    selected={root_pid};changed=True
    while changed:
        more={pid for pid,(ppid,_,_) in infos.items() if ppid in selected};changed=not more.issubset(selected);selected.update(more)
    values={}
    for pid in selected:
        if pid not in infos:continue
        _,parts,proc=infos[pid];io={}
        try:io=dict(line.split(':',1) for line in (proc/'io').read_text().splitlines())
        except (OSError,ValueError):pass
        values[(pid,int(parts[19]))]={'cpu_s':(int(parts[11])+int(parts[12]))/ticks,'rss':int(parts[21])*pages,'read':int(io.get('read_bytes','0')),'write':int(io.get('write_bytes','0'))}
    return values
def stage_rows(workspace,status,at):
    data=load(workspace/'pipeline_run.json',{}) or {};cut=parse_time(status.get('native_invocation_started_at'))
    latest={}
    for row in data.get('stages',[]):
        began=parse_time(row.get('started_at'))
        if cut is None:continue
        if cut is not None and (began is None or began<cut):continue
        latest[row.get('stage')]=row
    result=[];active=None
    for key,label in STAGES:
        row=latest.get(key);state='waiting'
        if row:
            state='reused' if row.get('skipped') and row.get('exit_code')==0 else 'done' if row.get('exit_code')==0 else 'failed' if row.get('exit_code') not in (None,0) else 'running'
            if state in ['running','failed']:active=row
        start=parse_time((row or {}).get('started_at'));finish=parse_time((row or {}).get('finished_at'))
        result.append({'key':key,'label':label,'state':state,'started_at':(row or {}).get('started_at'),'finished_at':(row or {}).get('finished_at'),'elapsed_s':max(0,(finish or at)-start) if start else None})
    return result,active
def loop_progress(text,stage,workspace,status):
    # Counters describe exactly this source loop, never inferred remaining time.
    matches=re.findall(r'\|[^\r\n]*?\|\s*(\d+)\s*/\s*(\d+)\s*\[',text)
    if matches:
        done,total=map(int,matches[-1])
        if 0<total and 0<=done<=total:return {'completed':done,'total':total,'unit':'项','percent':100*done/total,'scope':'当前日志子循环；阶段仍可能有后处理'}
    labels=re.findall(r'(label objects|label voxels)\s+(\d+)\s*/\s*(\d+)',text,re.I)
    if labels:
        kind,done,total=labels[-1];done,total=int(done),int(total)
        if 0<total and 0<=done<=total:return {'completed':done,'total':total,'unit':'体素' if 'voxels' in kind.lower() else '对象','percent':100*done/total,'scope':'当前有限类别分类子任务，非整个流程进度'}
    if stage=='features':
        counts=re.findall(r'encoded\s+(\d+)',text);rows=load(workspace/'supplied_dataset/camera_samples.json',[]) or []
        if counts and rows:
            done=min(int(counts[-1]),len(rows));return {'completed':done,'total':len(rows),'unit':'帧','percent':100*done/len(rows),'scope':'CLIP图像编码帧计数'}
    if stage=='prepare':
        counts=re.findall(r'scanned\s+(\d+)',text);expected=sum(v.get('message_count',0) for v in status.get('topics',{}).values())
        if counts and expected:
            done=min(int(counts[-1]),expected);return {'completed':done,'total':expected,'unit':'消息','percent':100*done/expected,'scope':'录制消息扫描；后续投影与保存仍需时间'}
    expected=status.get('zip_expanded_bytes')
    if stage=='input_validation' and expected:
        candidates=list(workspace.glob('extracted_dataset.extracting.*'))
        if candidates:
            done=sum(p.stat().st_size for p in candidates[0].rglob('*') if p.is_file());return {'completed':min(done,expected),'total':expected,'unit':'字节','percent':100*min(done,expected)/expected,'scope':'ZIP已写出解压字节；之后仍有校验'}
    return None
def activity_state(log_age,resources,terminal=None):
    if terminal in ['failed','cancelled']:return {'state':terminal,'assessment':'流程已停止；查看下方错误与阶段日志'}
    if terminal in ['complete','prepared_not_mapped']:return {'state':'complete','assessment':'该次流程已结束'}
    cpu=resources.get('cpu_cores') or 0;io=(resources.get('read_MiB_s') or 0)+(resources.get('write_MiB_s') or 0);gpu=resources.get('gpu_percent')
    if cpu>=.1 or io>=.2 or (gpu is not None and gpu>=5):return {'state':'active','assessment':'仍观察到计算或读写活动；GPU为整机数据，不独证本任务活跃'}
    if log_age>=120:return {'state':'idle_unconfirmed','assessment':'超过120秒未见日志或明显资源活动，可能等待输入／资源，尚未确认卡死'}
    return {'state':'waiting','assessment':'尚无足够信息判断卡点，持续监看阶段日志'}
class LiveTracker:
    def __init__(self,workspace,root_pid,interval=2,started_perf=None):
        self.workspace=Path(workspace);self.root_pid=root_pid;self.interval=interval;self.started=started_perf if started_perf is not None else time.perf_counter();self.started_wall=time.time()-(time.perf_counter()-self.started);self.stop_event=threading.Event();self.samples=[];self.previous={};self.previous_at=None;self.last_stage=None;self.stage_began=time.perf_counter()
        self.thread=threading.Thread(target=self.run,daemon=True)
    def start(self):self.thread.start();return self
    def stop(self):self.stop_event.set();self.thread.join(timeout=6);return self.samples
    def sample(self):
        at=time.time();perf=time.perf_counter();status=load(self.workspace/'RUN_STATUS.json',{}) or {};rows,active=stage_rows(self.workspace,status,at);outer=status.get('stage','input_validation')
        stage=active.get('stage') if active and outer=='native_three_map_pipeline' else outer
        label=dict(STAGES).get(stage,EXTRA.get(stage,stage));log=self.workspace/active['log'] if active and active.get('log') and outer=='native_three_map_pipeline' else Path(status['stage_log']) if status.get('stage_log') else None
        text=tail(log) if log else '';log_mtime=log.stat().st_mtime if log and log.is_file() else self.started_wall;age=max(0,at-log_mtime)
        if stage!=self.last_stage:self.stage_began=perf;self.last_stage=stage
        values=tree_counters(self.root_pid);elapsed=perf-self.previous_at if self.previous_at else None;cpu=read=written=0
        if elapsed:
            for key,value in values.items():
                old=self.previous.get(key)
                if old:cpu+=max(0,value['cpu_s']-old['cpu_s']);read+=max(0,value['read']-old['read']);written+=max(0,value['write']-old['write'])
        gpu=gpu_sample();device=gpu.get('devices',[{}])[0] if gpu.get('available') else {}
        resources={'gpu_percent':device.get('gpu_util_percent'),'memory_used_MiB':device.get('memory_used_MiB'),'memory_total_MiB':device.get('memory_total_MiB'),'gpu_name':device.get('name'),'cpu_cores':cpu/elapsed if elapsed else None,'rss_GiB':sum(v['rss'] for v in values.values())/1024**3,'read_MiB_s':read/elapsed/1024**2 if elapsed else None,'write_MiB_s':written/elapsed/1024**2 if elapsed else None,'scope':'CPU/IO为本任务进程树；GPU为整机，非本任务独占'}
        self.previous=values;self.previous_at=perf
        done=sum(r['state'] in ['done','reused'] for r in rows);label_on=bool(status.get('label_yaml') or status.get('label_yaml_requested'));total=16+(2 if label_on else 0)
        completed=(1 if status.get('config') else 0)+done+(1 if status.get('labels_prepared') else 0)+(1 if status.get('labels_aligned') else 0)+(1 if outer=='complete' else 0)
        if outer=='complete':completed=total
        if outer=='prepared_not_mapped':
            completed=1+(1 if status.get('labels_prepared') else 0);total=completed
        activity=activity_state(age,resources,outer);activity.update(last_update=stamp(),seconds_since_log=round(age,1))
        error=status.get('error')
        if outer=='failed' or (active and active.get('exit_code') not in (None,0)):
            errors=[line for line in text.replace('\r','\n').splitlines() if re.search('Error|Exception|out of memory|Traceback',line)]
            if errors:error=errors[-1][-600:]
        if error:error=re.sub(r'(?i)((?:token|password|api[_-]?key|authorization)\s*[:=]\s*)\S+',r'\1[redacted]',str(error))
        detail=error or ('正在读取、校验源码/数据/权重，阶段尚未开始' if outer=='native_three_map_pipeline' and not active else '真实循环计数见进度；未提供计数时显示活动进度')
        row={'schema_version':'live_mapping_status_v2','updated_at':stamp(),'workspace':str(self.workspace),'stage':stage,'stage_name':label,'stage_detail':detail,'stage_elapsed_s':round(perf-self.stage_began,1),'total_elapsed_s':round(perf-self.started,1),'progress':{'completed':min(completed,total),'total':total,'unit':'步骤','percent':100*min(completed,total)/total,'scope':'已结束步骤的比例，不是剩余时间估计'},'current_progress':loop_progress(text,stage,self.workspace,status),'resources':resources,'activity':activity,'pipeline_stage_index':next((i+1 for i,(k,_) in enumerate(STAGES) if k==stage),0),'pipeline_stage_count':14,'stage_status':rows,'error':error,'log_path':str(log) if log else None,'label_mode':label_on,'terminal':outer in ['complete','prepared_not_mapped','failed','cancelled']}
        write(self.workspace/'LIVE_STATUS.json',row)
        self.samples.append({'at':row['updated_at'],'stage':stage,'job_CPU_equivalent_cores':resources['cpu_cores'],'job_RSS_sum_GiB':resources['rss_GiB'],'job_disk_read_MiB_s':resources['read_MiB_s'],'job_disk_write_MiB_s':resources['write_MiB_s'],'gpu':gpu})
        return row
    def run(self):
        with (self.workspace/'RESOURCE_SAMPLES.jsonl').open('w') as stream:
            while not self.stop_event.is_set():
                try:self.sample();stream.write(json.dumps(self.samples[-1],ensure_ascii=False)+'\n');stream.flush()
                except Exception as error:write(self.workspace/'LIVE_STATUS_ERROR.json',{'at':stamp(),'telemetry_error_type':type(error).__name__,'mapping_stopped':False})
                self.stop_event.wait(self.interval)
