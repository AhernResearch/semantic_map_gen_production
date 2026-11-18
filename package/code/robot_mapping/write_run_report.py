"""Write a reviewable report from recorded stages and existing result manifests."""
from pathlib import Path
import argparse
import json
import datetime


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def write_report(workspace, record):
    workspace = Path(workspace)
    latest = {}
    for row in record.get('stages', []):
        latest[row['stage']] = row
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    lines = [f"# {record.get('scene_id', workspace.name)} 运行记录", '',
             f"自动更新于UTC {now}。源码：`{record.get('clean_source_root', '')}`。", '',
             f"配置：`{record.get('config', '')}`。输出：`{workspace}`。", '',
             '本报告从运行记录和现有清单生成；阶段状态列出各阶段最后一次记录，不把文件存在视为精度证明。', '',
             '| 阶段 | 最后状态 | 日志 |', '|---|---|---|']
    for name, row in latest.items():
        code = row.get('exit_code')
        status = '签名和输出校验匹配，复用' if row.get('skipped') else ('完成' if code == 0 else ('运行中' if code is None else f'失败（{code}）'))
        lines.append(f"| {name} | {status} | `{row.get('log', 'logs/' + name + '.log')}` |")
    summary = read_json(workspace / 'supplied_dataset/summary.json')
    if summary:
        lines += ['', '## 输入处理', '', f"数据目录：`{summary.get('source_dataset', '')}`。抽样RGB {summary.get('rgb_samples')}张、LiDAR {summary.get('lidar_samples')}条。", '',
                  f"配置选择话题：`{json.dumps(summary.get('selected_topics', {}), ensure_ascii=False)}`。", '',
                  '固定stride抽样发生在CLIP计算之前，不是根据语义／运动自动选择关键帧。']
    validation = read_json(workspace / 'native_maps/validation.json')
    if validation:
        lines += ['', '## 已存在的原生三图检查记录', '', f"结构检查：`all_passed={validation.get('all_passed')}`，{len(validation.get('checks', []))}项。是否对应本次配置须同时核对pipeline_run.json中的阶段签名。", '',
                  '| 地图 | 原生格式 | 记录内容 |', '|---|---|---|']
        for key, fmt, detail in [
            ('HMSG', 'graph_*/ JSON＋PLY', f"{validation.get('HMSG', {}).get('objects')}预测对象"),
            ('ConceptGraphs', '*_post.pkl.gz', f"{validation.get('ConceptGraphs', {}).get('objects')}预测实例"),
            ('VLMaps', 'vlmaps_cam.h5df（HDF5）', f"{validation.get('VLMaps', {}).get('feature_shape')}特征")]:
            lines.append(f'| {key} | {fmt} | {detail} |')
    alignment = read_json(workspace / 'depth_alignment_report.json')
    if alignment:
        raw = alignment.get('raw_abs_rel_median')
        fitted = alignment.get('fitted_abs_rel_median')
        if raw is not None and fitted is not None:
            lines += ['', f'投影LiDAR留出点条件误差：原DA3 {raw:.3%}，拟合后 {fitted:.3%}；不是独立标定误差或全图深度精度。']
    package = read_json(workspace / 'share/native_package_result.json')
    if package:
        lines += ['', '## 已存在的结果包', '', f"`{package.get('archive')}`，{package.get('bytes')}字节。", '', f"SHA256：`{package.get('sha256')}`。"]
    lines += ['', '## 解释与限制', '',
              '三图共同采用Y-up世界坐标；输入标定、录制里程计和时间关联决定其几何。PCD预览是几何／颜色衍生物，不包含全部语义特征。', '',
              '非空、有限数值、引用和坐标结构检查不代表语义准确率、独立地图精度、可通行性或真机导航成功。自动对象／房间数量不是人工真值。', '',
              '详细操作：whichground_clean/docs/07_数据集到三图操作手册.md。700G incoming/real只用于查找和打包历史地图。', '']
    target = workspace / 'RUN_REPORT.md'
    temporary = target.with_suffix('.md.part')
    temporary.write_text('\n'.join(lines), encoding='utf-8')
    temporary.replace(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    args = parser.parse_args()
    record = read_json(args.workspace / 'pipeline_run.json')
    if record is None:
        raise SystemExit('Missing pipeline_run.json')
    print(write_report(args.workspace, record))


if __name__ == '__main__':
    main()
