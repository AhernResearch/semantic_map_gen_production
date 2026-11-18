"""Generate a simple dataset-specific preview page from validated outputs."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
from html import escape
import run_config as rc

v=json.loads((rc.WORKSPACE/'native_maps/validation.json').read_text())
assert v['all_passed']
out=rc.WORKSPACE/'viewer';out.mkdir(exist_ok=True)
rows=[]
for label,key,file,description in [
    ('HMSG','HMSG','hmsg_today.pcd',f"{v['HMSG']['objects']}个预测对象，{v['HMSG']['rooms']}个房间区域，{v['HMSG']['views']}个视图"),
    ('ConceptGraphs','ConceptGraphs','cg_today.pcd',f"{v['ConceptGraphs']['objects']}个过滤后实例"),
    ('VLMaps','VLMaps','vlmaps_today.pcd',f"{v['VLMaps']['feature_shape'][0]:,}×{v['VLMaps']['feature_shape'][1]}体素特征")]:
    rows.append(f'<tr><td>{label}</td><td>{description}</td><td><a href="../native_maps/preview/{file}">下载预览PCD</a></td></tr>')
retrieval=(rc.WORKSPACE/'semantic_keyframes/retrieval_preview.jpg').exists()
retrieval_html='<h2>抽样图像检索示例</h2><img src="../semantic_keyframes/retrieval_preview.jpg" alt="现场图像检索">' if retrieval else ''
assets=Path(__file__).resolve().parent/'web'
shutil.copy2(assets/'cloud_viewer.js',out/'cloud_viewer.js')
shutil.copytree(assets/'vendor',out/'vendor',dirs_exist_ok=True)
entries=[{'key':key,'title':title,'file':None,'display_points':0} for key,title in [
    ('lidar','LiDAR 几何 PCD 原文件全部点'),('colored','RGB–LiDAR 彩色 PCD 原文件全部点'),
    ('hmsg_today','HMSG 原生对象全部点'),('cg_today','ConceptGraphs 原生实例全部点'),
    ('vlmaps_today','VLMaps 原生特征体素全部点')]]
# The exporter reads native map paths from validation.json and preserves every saved point.
(out/'data.js').write_text('window.DATA='+json.dumps({'clouds':entries},ensure_ascii=False)+';\n')
subprocess.run([sys.executable,str(Path(__file__).with_name('export_full_clouds.py')),
                '--workspace',str(rc.WORKSPACE)],check=True)
interactive='''<h2>交互式三维点云：原文件全部点</h2>
<div class="controls"><select id="cloudChoice" aria-label="选择点云"></select>
<label>显示模式 <select id="cloudQuality" aria-label="点云显示模式"><option value="full">原文件全部点</option><option value="quick">快速预览</option></select></label>
<button id="cloudLoad">加载全部点</button><button id="cloudStop" disabled>停止加载</button><button id="cloudFit">重置视角</button></div>
<div id="cloud" style="width:100%;height:540px"></div><p id="cloudStatus"></p>
<label>点大小 <input id="pointSize" type="range" min="1" max="5" step="0.5" value="2"></label>
<p id="cloudNote" class="note"></p><a id="cloudDownload" hidden download></a>
<p class="note">网页不再额外降采样。文件本身的建图体素/抽帧仍保留，来源见每项说明；全部雷达扫描可另运行手册中的export_unvoxel_lidar.py导出。页面以Z向上显示，不将不同来源地图自动配准。</p>
<script type="importmap">{"imports":{"three":"./vendor/three.module.js"}}</script><script src="data.js"></script><script type="module" src="cloud_viewer.js"></script>'''
html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{escape(rc.SCENE_ID)} 三地图</title>
<style>body{{font:16px/1.7 system-ui,sans-serif;background:#f2f5f7;color:#193442;margin:0}}main{{max-width:1280px;margin:25px auto;padding:24px;background:white;border-radius:12px}}img{{max-width:100%}}table{{width:100%;border-collapse:collapse}}td,th{{text-align:left;padding:12px;border-bottom:1px solid #dde7ea}}a{{color:#096a83}}.note{{color:#596e79}}</style>
<main><p><a href="/search/">文字指令检索与人工标注</a></p><h1>{escape(rc.SCENE_ID)}：三种原生语义地图</h1><p>{v['sample_frames']}张抽样图像，来源于whichground_clean的可配置处理流程。</p><img src="../native_maps/preview/three_native_maps.png" alt="原生三地图"><table><tr><th>地图</th><th>实际输出</th><th>可视化</th></tr>{''.join(rows)}</table>
<p>本机使用PCD查看器旋转查看；这三种原生地图的世界坐标为Y向上，查看器使用<code>--y-up</code>。</p><p class="note">预测对象与自动房间区域不是人工真值。结构与数值检查通过；语义准确率、独立标定和导航可通行性未验证。</p>
<p>原相机每{rc.CAMERA_STRIDE}张取1张，CLIP在抽样后计算特征，未参与选帧。HMSG实例融合再每{rc.HMSG_STRIDE}张取1张，CG每{rc.CG_STRIDE}张，VLMaps每{rc.VLMAPS_STRIDE}张。</p>
<p><a href="../share/{escape(rc.SCENE_ID)}_native_three_maps.tar.gz">下载三图包</a> · <a href="../native_maps/validation.json">数据检查</a> · <a href="../RUN_REPORT.md">运行说明</a></p>{retrieval_html}</main></html>'''
html=html.replace('<table><tr>',interactive+'<table><tr>',1)
(out/'index.html').write_text(html,encoding='utf-8')
print('VIEWER',out/'index.html')
