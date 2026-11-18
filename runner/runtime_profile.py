"""Sample the mapping process group and host GPU; describe evidence, not guesswork."""
from __future__ import annotations
import csv,datetime as dt,json,os,statistics,subprocess,threading,time
from pathlib import Path

def stamp():return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()
def hardware():
    info={'cpu_logical_threads':os.cpu_count(),'GPU_measurement_scope':'整机GPU采样，可能包含其他程序；CPU/I/O按采样器统计本任务进程树/进程组'}
    try:
        info['cpu_model']=next(s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name'))
        memory=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
        info['ram_total_GiB']=round(int(memory['MemTotal'].split()[0])*1024/1024**3,2)
    except (OSError,StopIteration):pass
    return info
def gpu_sample():
    try:
        p=subprocess.run(['nvidia-smi','--query-gpu=name,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3)
        if p.returncode:return {'available':False,'reason':'nvidia-smi不可用或权限不足'}
        rows=[]
        for r in csv.reader(p.stdout.splitlines()):
            r=[x.strip() for x in r]
            if len(r)==7:rows.append({'name':r[0],'index':int(r[1]),'gpu_util_percent':float(r[2]),'memory_util_percent':float(r[3]),'memory_used_MiB':float(r[4]),'memory_total_MiB':float(r[5]),'power_W':float(r[6])})
        return {'available':bool(rows),'devices':rows}
    except (OSError,ValueError,subprocess.TimeoutExpired):return {'available':False,'reason':'GPU采样失败，未编造数值'}
def group_counters(group):
    values={};pages=os.sysconf('SC_PAGE_SIZE');ticks=os.sysconf('SC_CLK_TCK')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            raw=(proc/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
            if int(fields[2])!=group:continue
            # stat fields after comm start at state(3):pgrp(5),utime(14),stime(15),starttime(22),rss(24).
            key=(int(proc.name),int(fields[19]));io={}
            try:io=dict(line.split(':',1) for line in (proc/'io').read_text().splitlines())
            except PermissionError:pass
            values[key]={'cpu_s':(int(fields[11])+int(fields[12]))/ticks,'rss_bytes':int(fields[21])*pages,'read_bytes':int(io.get('read_bytes','0')),'write_bytes':int(io.get('write_bytes','0'))}
        except (OSError,ValueError,IndexError):continue
    return values
def current_stage(workspace):
    try:
        d=json.loads((workspace/'pipeline_run.json').read_text());row=d['stages'][-1]
        return row['stage'] if 'finished_at' not in row else 'between_stages_or_restoring_services'
    except (OSError,KeyError,IndexError,ValueError):return 'startup_fingerprinting_and_loading'
class Monitor:
    def __init__(self,workspace,pid,interval=2):
        self.workspace=Path(workspace);self.pid=pid;self.interval=interval;self.stop_event=threading.Event();self.samples=[];self.previous={};self.previous_time=None
        self.thread=threading.Thread(target=self.run,daemon=True)
    def start(self):self.started=time.perf_counter();self.thread.start();return self
    def run(self):
        path=self.workspace/'RESOURCE_SAMPLES.jsonl'
        with path.open('w') as stream:
            while not self.stop_event.is_set():
                t=time.perf_counter();values=group_counters(self.pid);elapsed=t-self.previous_time if self.previous_time is not None else None
                cpu=read=written=0
                if elapsed and elapsed>0:
                    for key,v in values.items():
                        if key in self.previous:
                            old=self.previous[key];cpu+=max(0,v['cpu_s']-old['cpu_s']);read+=max(0,v['read_bytes']-old['read_bytes']);written+=max(0,v['write_bytes']-old['write_bytes'])
                row={'at':stamp(),'stage':current_stage(self.workspace),'processes':len(values),'job_CPU_equivalent_cores':cpu/elapsed if elapsed else None,'job_RSS_sum_GiB':sum(v['rss_bytes'] for v in values.values())/1024**3,'job_disk_read_MiB_s':read/elapsed/1024**2 if elapsed else None,'job_disk_write_MiB_s':written/elapsed/1024**2 if elapsed else None,'gpu':gpu_sample()}
                self.samples.append(row);stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush();self.previous=values;self.previous_time=t
                self.stop_event.wait(self.interval)
    def stop(self):self.stop_event.set();self.thread.join(timeout=5);return self.samples
def summarize(workspace,samples,total_timing,input_bytes):
    workspace=Path(workspace);groups={}
    for row in samples:groups.setdefault(row['stage'],[]).append(row)
    def mean(rows,key):
        v=[r[key] for r in rows if isinstance(r.get(key),(int,float))];return round(statistics.mean(v),3) if v else None
    table=[]
    for stage,rows in groups.items():
        devices=[r['gpu']['devices'][0] for r in rows if r.get('gpu',{}).get('available') and r['gpu'].get('devices')]
        util=statistics.median(d['gpu_util_percent'] for d in devices) if devices else None
        pressure=max(d['memory_used_MiB']/d['memory_total_MiB'] for d in devices) if devices else None
        cpu=mean(rows,'job_CPU_equivalent_cores');read=mean(rows,'job_disk_read_MiB_s');write=mean(rows,'job_disk_write_MiB_s')
        evidence=[]
        if stage.startswith('startup'):evidence.append('此区间包含输入/源码/权重指纹和启动，不能全算模型推理')
        if util is not None and util>=70:evidence.append('整机GPU利用率较高，可能含GPU计算耗时')
        if pressure is not None and pressure>=.9:evidence.append('整机显存接近上限；需结合其他进程和OOM日志判断')
        if util is not None and util<15 and cpu is not None and cpu>=.5:evidence.append('GPU利用率低而本任务CPU活跃，CPU处理可能重要')
        if (read or 0)+(write or 0)>=20:evidence.append('本任务有明显磁盘读写，压缩/解码/存盘可能重要')
        if not evidence:evidence.append('当前采样不足以定位单一瓶颈')
        table.append({'stage':stage,'samples':len(rows),'GPU_util_median_percent':util,'GPU_memory_peak_fraction':pressure,'job_CPU_mean_cores':cpu,'job_disk_read_mean_MiB_s':read,'job_disk_write_mean_MiB_s':write,'interpretation':'；'.join(evidence),'causal_bottleneck_proven':False})
    report={'generated_at':stamp(),'hardware':hardware(),'source_input_bytes':input_bytes,'timing':total_timing,'resources_by_stage':table,'samples_file':str(workspace/'RESOURCE_SAMPLES.jsonl'),'scope':'采样支持的判断；GPU为整机，CPU/I/O为本任务进程树/进程组，不能用一次运行建立包大小与耗时的因果/线性关系','factors':['包字节数/压缩格式影响读取和解压','图像数量/分辨率和关键帧数量影响模型工作量','LiDAR点数/体素数与跨帧匹配规模影响点云CPU/GPU和内存','模型权重加载/初次缓存/源码与输入SHA影响启动','4090计算利用率/显存压力、CPU并行度、磁盘速度和其他程序影响耗时','三种建图串行执行，不是全部时间都由GPU算力决定']}
    (workspace/'TIMING_REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    lines=['# 生成时间与耗时分析','',f"生成记录：{report['generated_at']}",'',f"输入大小：{input_bytes/1024**3:.3f} GiB；总耗时：{total_timing.get('total_wall_s',0):.1f}秒。",'', '| 阶段 | GPU中位利用率% | 本任务平均CPU等效核 | 平均读MiB/s | 平均写MiB/s | 采样判断 |','|---|---:|---:|---:|---:|---|']
    for r in table:lines.append('|'+ '|'.join(str(r.get(k,'未测')) for k in ['stage','GPU_util_median_percent','job_CPU_mean_cores','job_disk_read_mean_MiB_s','job_disk_write_mean_MiB_s','interpretation'])+'|')
    lines+=['','不能只按包大小或4090型号预测时间。图像/点数、采样、模型、CPU、IO、缓存都影响；上表是观测线索，未证明单一因果瓶颈。其他GPU程序可能贡献整机数值。']
    (workspace/'生成时间与耗时分析.md').write_text('\n'.join(lines))
    return report
