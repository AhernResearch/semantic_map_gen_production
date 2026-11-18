#!/usr/bin/env python3
"""Select one recorded ROS2 bag, run the installed native three-map pipeline."""
from __future__ import annotations
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import time
import uuid
import zipfile

TOOL_ROOT=Path(__file__).resolve().parent
DEFAULT_PACKAGE=TOOL_ROOT.parent/'package'
MCAP_MAGIC=b'\x89MCAP0\r\n'

def now():return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.writing')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));temp.replace(path)
def check_mcap(file):
    file=Path(file)
    if file.stat().st_size<16:raise ValueError(f'录制文件过小：{file}')
    with file.open('rb') as stream:
        start=stream.read(8);stream.seek(-8,2);end=stream.read(8)
    if start!=MCAP_MAGIC or end!=MCAP_MAGIC:raise ValueError(f'MCAP尚未完整写入/传输，或格式不正确：{file}')
def locate_input(path):
    path=Path(path).expanduser().resolve()
    if not path.exists():raise FileNotFoundError(f'未找到录制数据：{path}')
    if path.is_file() and path.suffix.lower()=='.zip':return path,None
    if path.is_file():
        if path.name!='metadata.yaml' and path.suffix.lower()!='.mcap':raise ValueError('请选择ROS2录制目录、metadata.yaml、.mcap或包含这些数据的ZIP')
        if path.suffix.lower()=='.mcap':check_mcap(path)
        path=path.parent
    bags=[path] if (path/'metadata.yaml').is_file() else [p.parent for p in path.rglob('metadata.yaml')]
    if len(bags)!=1:raise ValueError('所选目录包含零个或多个rosbag；请选择具体那一个录制目录')
    bag=bags[0].resolve()
    import yaml
    metadata=yaml.safe_load((bag/'metadata.yaml').read_text())
    info=(metadata or {}).get('rosbag2_bagfile_information',{})
    if info.get('storage_identifier')!='mcap':raise ValueError('当前一键流程支持本机器狗的ROS2 MCAP录制；其他存储格式需先适配')
    relative=info.get('relative_file_paths',[])
    if not relative:raise ValueError('metadata没有录制文件列表，可能还在传输')
    for name in relative:
        file=(bag/name).resolve()
        if not file.is_relative_to(bag) or file.suffix!='.mcap':raise ValueError('metadata里的文件路径或类型不合法')
        if not file.is_file():raise ValueError(f'录制分片未传齐：{file}')
        check_mcap(file)
    return path,bag
def choose_calibration(source,bag,args,workspace=None):
    from calibration_input import validate_calibration_file
    if args.calibration:return validate_calibration_file(args.calibration),'explicit_recording_calibration'
    if bag:
        found=[p for name in ['extrinst.txt','extrinst.json','mapping_recording_calibration.txt','calibration.json'] if (p:=bag/name).is_file()]
        if len(found)==1:return validate_calibration_file(found[0]),'paired_calibration_in_bag_directory'
        if len(found)>1:raise ValueError('录制目录有多个标定文件，请在窗口/--calibration明确选择')
    if source.is_file() and source.suffix.lower()=='.zip' and workspace:
        from prepare_new_run import zip_files
        with zipfile.ZipFile(source) as archive:
            members=zip_files(archive)
            matches=[m for m in members if Path(m.filename).name in ['extrinst.txt','extrinst.json','mapping_recording_calibration.txt','calibration.json']]
            if len(matches)==1:
                member=matches[0]
                if member.file_size>4*1024**2:raise ValueError('ZIP内标定文件过大')
                file=workspace/'calibration/from_selected_zip.txt';file.parent.mkdir(exist_ok=True);file.write_bytes(archive.read(member))
                return validate_calibration_file(file),'paired_calibration_inside_selected_zip'
            if len(matches)>1:raise ValueError('ZIP内有多个标定，请明确选择当前录制的标定文件')
    raise ValueError('录制包里未找到标定：请通过 --calibration 提供对应录制的标定文件。')
def parser():
    p=argparse.ArgumentParser(description='选rosbag后运行提纯版原生三图，完成输出文件位置。')
    p.add_argument('--prepared-input',help='Thor保真原始观测输入包/目录，不需要原MCAP回传')
    p.add_argument('--bag',help='录制目录、metadata.yaml、.mcap或ZIP')
    p.add_argument('--calibration')
    p.add_argument('--label-yaml',type=Path,help='可选version3标签YAML；生成独立有限类别三图，原生文件保留')
    p.add_argument('--output-root',default=str(TOOL_ROOT.parent/'runs'))
    p.add_argument('--package-root',default=str(DEFAULT_PACKAGE));p.add_argument('--resume',type=Path)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--max-samples',type=int,default=0)
    p.add_argument('--include-class',action='append',default=[],help='追加明确英文检测类别；不保证检测成功')
    return p
def run_command(command,environment,workspace=None,samples=None,output_file=None):
    output_stream=Path(output_file).open('w') if output_file else None
    process=subprocess.Popen(command,env=environment,start_new_session=True,stdout=output_stream,stderr=subprocess.STDOUT if output_stream else None)
    monitor=None
    if workspace:
        from runtime_profile import Monitor
        monitor=Monitor(workspace,process.pid).start()
    try:return process.wait()
    except (KeyboardInterrupt,SystemExit):
        # Terminate the pipeline and its model subprocesses.
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait()
        raise
    finally:
        if monitor:
            result=monitor.stop()
            if samples is not None:samples.extend(result)
        if output_stream:output_stream.close()
def main():
    args=parser().parse_args();started=now();begin=time.perf_counter();timing={'generation_started_at':started};resource_samples=[];input_bytes=0
    print('启动时间：'+started,flush=True)
    package=Path(args.package_root).expanduser().resolve();code=package/'code/robot_mapping';python=Path(sys.executable)
    if not python.is_file() or not (code/'pipeline.py').is_file():raise ValueError(f'未找到已部署提纯版环境/脚本：{package}')
    sys.path.insert(0,str(code));from prepare_new_run import prepare,calibration_data
    workspace=None;status={};lock=None;workspace_lock=None;tracker=None
    previous_term=signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,lambda signum,frame:(_ for _ in ()).throw(SystemExit(128+signum)))
    try:
        if args.resume:
            candidate=args.resume.expanduser().resolve();owner=candidate/'AUTOBUILD_RUN.json'
            if not owner.is_file():raise ValueError('只恢复由本工具创建的工作目录；旧实验目录不会覆盖')
            status=json.loads(owner.read_text())
            if status.get('package') != str(package):raise ValueError('恢复目录必须由当前项目创建')
            config=Path(status['config']);calibration=Path(status['calibration']);source=Path(status['source'])
            if args.bag and locate_input(args.bag)[0]!=source:raise ValueError('恢复目录与所选录制不是同一输入')
            if sha(config)!=status['config_sha256'] or sha(calibration)!=status['calibration_sha256']:raise ValueError('配置/标定已变，需新工作目录')
            if status.get('label_yaml') and sha(status['label_yaml'])!=status['label_yaml_sha256']:
                raise ValueError('冻结的标签YAML已变，需新工作目录')
            if args.label_yaml and sha(args.label_yaml.expanduser().resolve())!=status.get('label_yaml_sha256'):raise ValueError('恢复任务的标签YAML不同，需新工作目录')
            workspace_lock=(candidate/'autobuild.lock').open('a')
            try:fcntl.flock(workspace_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise ValueError('已有任务正在运行此目录')
            workspace=candidate
            status.update(invocation_started_at=started,stage='input_validation');write(workspace/'RUN_STATUS.json',status)
            from live_status import LiveTracker
            tracker=LiveTracker(workspace,os.getpid(),started_perf=begin).start()
        else:
            if args.prepared_input:
                if args.bag:raise ValueError('录包和预处理输入请选择一种')
                root=Path(args.output_root).expanduser().resolve()
                prepared_source=Path(args.prepared_input).expanduser().resolve()
                if root.is_relative_to(package) or root.is_relative_to(prepared_source):raise ValueError('预处理输出必须使用独立数据目录')
                if args.max_samples<0:raise ValueError('max-samples必须非负')
                scene='vln_prepared_'+dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6];workspace=root/scene;workspace.mkdir(parents=True,exist_ok=False);print('AUTOBUILD_WORKSPACE='+str(workspace),flush=True)
                from prepared_inputs import prepare_input
                result=prepare_input(args,workspace,code);config=Path(result['mapping_config']);calibration=Path(result['calibration']);source=Path(args.prepared_input).expanduser().resolve();topics=result['input_topics'];status={'schema_version':'rosbag_autobuild_run_v2','created_at':started,'invocation_started_at':started,'stage':'configuration_ready','workspace':str(workspace),'package':str(package),'source':str(source),'scene_id':scene,'config':str(config),'config_sha256':sha(config),'calibration':str(calibration),'calibration_sha256':sha(calibration),'calibration_policy':'verified_prepared_bundle_calibration','input_type':'lossless_prepared_observations','topics':topics,'label_yaml_requested':bool(args.label_yaml),'robot_execution_enabled':False};write(workspace/'RUN_STATUS.json',status)
                from live_status import LiveTracker
                tracker=LiveTracker(workspace,os.getpid(),started_perf=begin).start()
            else:
                if not args.bag:raise ValueError('请通过 --bag 提供录制路径')
                if args.max_samples<0:raise ValueError('max-samples必须不小于0；0表示全部采样关键帧')
                source,bag=locate_input(args.bag)
                root=Path(args.output_root).expanduser().resolve()
                if root.is_relative_to(package) or root.is_relative_to(source) or (bag and root.is_relative_to(bag)):
                    raise ValueError('输出请选择源码和录制之外的独立数据目录')
                stem=re.sub('[^A-Za-z0-9_.-]+','_',source.stem).strip('._-') or 'bag'
                scene='vln_'+stem[:48]+'_'+dt.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
                workspace=root/scene;workspace.mkdir(parents=True,exist_ok=False)
                print('AUTOBUILD_WORKSPACE='+str(workspace),flush=True)
                status={'created_at':started,'invocation_started_at':started,'stage':'input_validation','workspace':str(workspace),'source':str(source),'label_yaml_requested':bool(args.label_yaml)}
                if source.is_file() and source.suffix.lower()=='.zip':
                    with zipfile.ZipFile(source) as archive:status['zip_expanded_bytes']=sum(i.file_size for i in archive.infolist())
                write(workspace/'RUN_STATUS.json',status)
                from live_status import LiveTracker
                tracker=LiveTracker(workspace,os.getpid(),started_perf=begin).start()
                calibration,policy=choose_calibration(source,bag,args,workspace)
                original_calibration=calibration;calibration_hash=sha(calibration)
                snapshot=workspace/'calibration/recording_calibration.txt';snapshot.parent.mkdir(exist_ok=True)
                original_snapshot=workspace/'calibration'/('original_'+calibration.name);shutil.copy2(calibration,original_snapshot)
                from calibration_input import normalized_snapshot
                calibration_format=normalized_snapshot(calibration,snapshot)
                if sha(original_snapshot)!=calibration_hash or sha(original_calibration)!=calibration_hash:raise ValueError('标定在复制时变化，停止以免混用参数')
                calibration=snapshot
                camera,lidar=calibration_data(calibration)
                if camera.get('target_frame')!=lidar.get('target_frame'):raise ValueError('相机与雷达外参必须指向同一base坐标系')
                status.update(schema_version='rosbag_autobuild_run_v2',scene_id=scene,package=str(package),calibration=str(calibration),original_calibration=str(original_calibration),calibration_policy=policy,calibration_original_sha256=calibration_hash,calibration_format=calibration_format,calibration_sha256=sha(calibration),robot_execution_enabled=False,complete_semantic_surfaces_verified=False)
                write(workspace/'RUN_STATUS.json',status)
                preparation_args=argparse.Namespace(dataset=str(source),scene_id=scene,workspace=str(workspace),bag=str(bag) if bag else None,calibration=str(calibration),extract_to=str(workspace/'extracted_dataset') if source.is_file() else None,template=None,image_topic=None,lidar_topic=None,odom_topic=None,camera_stride=None,lidar_stride=None,max_samples=args.max_samples,image_max_width=None,seed=None,include_class=args.include_class)
                result=prepare(preparation_args);config=Path(result['mapping_config']);topics=result['input_topics']
                for role,expected in [('image',camera['source_frame']),('lidar',lidar['source_frame'])]:
                    if topics[role]['frame_id']!=expected:raise ValueError(f'{role}消息frame与标定source_frame不一致；请提供对应录制标定')
                if topics['odom']['child_frame_id']!=camera['target_frame']:raise ValueError('Odometry child_frame与外参base不一致，不能直接把位姿当base使用')
                status.update(config=str(config),config_sha256=sha(config),topics=topics,stage='configuration_ready')
                write(workspace/'RUN_STATUS.json',status)
            if args.label_yaml:
                original=args.label_yaml.expanduser().resolve();snapshot=workspace/'labels/source_annotations.yaml';snapshot.parent.mkdir(exist_ok=True)
                before=sha(original);shutil.copy2(original,snapshot)
                if sha(snapshot)!=before or sha(original)!=before:raise ValueError('标签YAML在复制中变化，停止以免混用版本')
                status.update(stage='label_preparation',label_yaml=str(snapshot),label_yaml_sha256=before,original_label_yaml=str(original));write(workspace/'RUN_STATUS.json',status)
                prep_log=workspace/'labels/prepare.log';status['stage_log']=str(prep_log);write(workspace/'RUN_STATUS.json',status)
                command=[str(python),str(TOOL_ROOT/'label_alignment.py'),'prepare','--yaml',str(snapshot),'--workspace',str(workspace),'--config',str(config)]
                result_code=run_command(command,dict(os.environ,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1'),output_file=prep_log)
                if result_code:raise ValueError('标签YAML预备失败：'+prep_log.read_text(errors='replace')[-2000:])
                label_prep=json.loads((workspace/'LABEL_PREPARATION.json').read_text());config=Path(label_prep['config_path'])
                status.update(config=str(config),config_sha256=sha(config),labels_prepared=True,stage='configuration_ready')
                write(workspace/'RUN_STATUS.json',status)
            write(workspace/'AUTOBUILD_RUN.json',status)
        source=Path(status['source']);actual_bag=Path(json.loads((workspace/'run_preparation.json').read_text())['bag'])
        input_bytes=source.stat().st_size if source.is_file() else sum(p.stat().st_size for p in actual_bag.glob('*.mcap'))
        timing.update(input_configuration_wall_s=time.perf_counter()-begin,input_bytes=input_bytes,input_message_counts={k:v['message_count'] for k,v in status.get('topics',{}).items()})
        delivery=workspace/'delivery';delivery.mkdir(exist_ok=True)
        if args.prepare_only:
            status.update(stage='prepared_not_mapped',updated_at=now());write(workspace/'RUN_STATUS.json',status)
            timing.update(generation_finished_at=now(),total_wall_s=time.perf_counter()-begin)
            write(delivery/'MAPPING_RESULT.json',{'success':False,'status':'prepared_not_mapped','workspace':str(workspace),'config':str(config),'timing':timing,'note':'输入与配置检查完成，未运行模型、未生成三图','robot_execution_enabled':False})
            print('输入检查通过，尚未生成地图。\n配置：'+str(config)+'\nRESULT_JSON:'+str(delivery/'MAPPING_RESULT.json'),flush=True);return 0
        flags=os.O_CREAT|os.O_RDWR|getattr(os,'O_NOFOLLOW',0)
        lock=os.fdopen(os.open('/tmp/vln_native_maps_gpu_'+str(os.getuid())+'.lock',flags,0o600),'r+')
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('已有建图任务占用此用户的GPU，请等待当前任务完成再开始')
        lock.seek(0);lock.truncate();lock.write(json.dumps({'pid':os.getpid(),'workspace':str(workspace),'started_at':now()}));lock.flush()
        environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',WANDB_MODE='disabled',PYTHONUNBUFFERED='1')
        command=[str(python),str(code/'pipeline.py'),'--config',str(config),'--stage','all']
        if args.resume:command.append('--resume')
        status.update(stage='native_three_map_pipeline',command=command,started_at=now(),native_invocation_started_at=now());status.pop('stage_log',None);write(workspace/'RUN_STATUS.json',status)
        print('开始本地原生三图；复用已核对的模型与运行环境。',flush=True)
        native_started=time.perf_counter();exit_code=run_command(command,environment);timing['native_pipeline_wall_s']=time.perf_counter()-native_started
        label_report=None
        if exit_code==0 and status.get('label_yaml'):
            label_dir=workspace/'label_locked'
            if label_dir.exists() and any(label_dir.iterdir()):label_dir=workspace/('label_locked_'+dt.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
            label_log=workspace/'logs'/('label_alignment_'+dt.datetime.now().strftime('%Y%m%d_%H%M%S')+'.log')
            status.update(stage='label_alignment',stage_log=str(label_log));write(workspace/'RUN_STATUS.json',status)
            label_started=time.perf_counter();label_code=run_command([str(python),str(TOOL_ROOT/'label_alignment.py'),'postprocess','--yaml',status['label_yaml'],'--workspace',str(workspace),'--output',str(label_dir),'--encode-text','--model-root',environment['SEMANTIC_GEN_MODEL_ROOT']],environment,output_file=label_log)
            timing['label_alignment_wall_s']=time.perf_counter()-label_started
            if label_code:raise ValueError('三图有限标签对齐失败：'+label_log.read_text(errors='replace')[-2000:])
            label_report=json.loads((label_dir/'LABEL_ALIGNMENT.json').read_text())
            if not label_report.get('success'):raise ValueError('标签对齐报告未通过；详见'+str(label_dir/'LABEL_ALIGNMENT.json'))
            status['labels_aligned']=True
        status.update(stage='collecting_outputs',stage_log=str(delivery/'collector.log'),pipeline_exit_code=exit_code,updated_at=now());write(workspace/'RUN_STATUS.json',status)
        # Collector versions must start in an absent directory; delivery already holds logs/status.
        collection=delivery/('native_collection_'+dt.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
        collector=[str(python),str(TOOL_ROOT/'collect_outputs.py'),'--workspace',str(workspace),'--output',str(collection)]
        if exit_code==0:collector.append('--require-complete')
        print('正在收集三图与打包，详细校验记录保存在delivery/collector.log。',flush=True)
        collect_started=time.perf_counter();collect_code=run_command(collector,environment,output_file=delivery/'collector.log');timing['collect_and_archive_wall_s']=time.perf_counter()-collect_started
        path=delivery/'MAPPING_RESULT.json';collector_result=collection/'MAPPING_RESULT.json'
        collected=json.loads(collector_result.read_text()) if collector_result.exists() else {'success':False,'missing':['收集器没有生成清单']}
        collected['native_collection_directory']=str(collection)
        if exit_code or collect_code:collected.update(success=False,status='failed',pipeline_exit_code=exit_code,collector_exit_code=collect_code)
        collected['run_context']={'config':str(config),'calibration':str(calibration),'calibration_policy':status.get('calibration_policy'),'calibration_sha256':sha(calibration),'reference_scope_registration_verified':False,'robot_execution_enabled':False}
        collected['label_mode']=bool(status.get('label_yaml'))
        if label_report:
            import tarfile
            original_artifacts=collected.get('artifacts',{});collected['native_artifacts']=original_artifacts;locked=label_report['artifacts']
            hmsg_root=Path(locked['hmsg']['path']);hmsg_tar=delivery/(label_dir.name+'_HMSG.tar.gz')
            with tarfile.open(hmsg_tar,'w:gz') as archive:archive.add(hmsg_root,arcname='HMSG_label_locked')
            selected=dict(original_artifacts);selected.update(hmsg_graph={'path':str(hmsg_root),'archive_path':str(hmsg_tar),'format':'独立YAML有限类别HMSG，未修改原生几何'},conceptgraphs=locked['conceptgraphs'],vlmaps=locked['vlmaps'])
            collected.update(artifacts=selected,label_locked_artifacts=locked,label_alignment_report=str(label_dir/'LABEL_ALIGNMENT.json'),label_source_yaml=status['label_yaml'],missing_target_categories=label_report.get('missing_target_categories',[]))
        timing.update(generation_finished_at=now(),total_wall_s=time.perf_counter()-begin)
        from runtime_profile import summarize
        resource_samples=list(tracker.samples) if tracker else resource_samples
        performance=summarize(workspace,resource_samples,timing,input_bytes);collected['generation_timing']=timing;collected['performance_report']=str(workspace/'TIMING_REPORT.json');collected['performance_readme']=str(workspace/'生成时间与耗时分析.md');collected['hardware']=performance['hardware']
        write(path,collected);status.update(stage='complete' if collected.get('success') else 'failed',result_json=str(path),finished_at=now());write(workspace/'RUN_STATUS.json',status)
        print('RESULT_JSON:'+str(path),flush=True)
        if not collected.get('success'):print('建图失败或不完整，请查看清单与阶段日志；不会把失败当成功。',flush=True);return exit_code or collect_code or 1
        for key,label in [('hmsg_graph','HMSG'),('conceptgraphs','ConceptGraphs'),('vlmaps','VLMaps')]:
            artifact=collected.get('artifacts',{}).get(key,{})
            print(label+'：'+str(artifact.get('archive_path') or artifact.get('path') or '详见清单'),flush=True)
        print('开始：'+timing['generation_started_at']+'\n结束：'+timing['generation_finished_at']+'\n总耗时：'+str(round(timing['total_wall_s'],1))+' 秒\n耗时分析：'+str(workspace/'生成时间与耗时分析.md'),flush=True)
        print('原生三图文件已生成。文件/日志位置见上面清单；结构完整性检查不是独立精度证明。',flush=True)
        return 0
    except BaseException as error:
        if workspace:
            status.update(stage='cancelled' if isinstance(error,(KeyboardInterrupt,SystemExit)) else 'failed',error=str(error),updated_at=now());write(workspace/'RUN_STATUS.json',status)
            timing.update(generation_finished_at=now(),total_wall_s=time.perf_counter()-begin)
            result=workspace/'delivery/MAPPING_RESULT.json';write(result,{'success':False,'status':status['stage'],'workspace':str(workspace),'error':str(error),'logs':str(workspace/'logs'),'timing':timing,'robot_execution_enabled':False})
            print('RESULT_JSON:'+str(result),flush=True)
        print('无法完成：'+str(error),file=sys.stderr,flush=True);return 1
    finally:
        if tracker:
            tracker.stop()
            try:tracker.sample()
            except Exception:pass
        if workspace_lock:workspace_lock.close()
        if lock:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        signal.signal(signal.SIGTERM,previous_term)
if __name__=='__main__':
    try:sys.exit(main())
    except (ValueError,OSError) as error:print(str(error),file=sys.stderr);sys.exit(1)
