# 生产机语义地图生成

从 ROS 2 MCAP 和录制对应的相机/LiDAR 标定生成 HMSG、ConceptGraphs、VLMaps 三张原生地图。流程沿用原来的串行阶段：采样与去畸变、时间和位姿对齐、DA3 深度、LiDAR 尺度拟合、RGB-D 生成、三图建图和交付校验。可选类别锁定结果独立输出。这个目录交付源码、安装脚本及模型清单，权重、缓存、数据和运行结果由生产机单独管理。

## 1. 重建环境

复现基线是 Linux x86_64（Ubuntu 22.04）、Python 3.10、NVIDIA RTX 4090、CUDA 12.4 兼容驱动和 PyTorch 2.5.1+cu124。构建 ConceptGraphs 的扩展还需要 CUDA **Toolkit** 12.4、GCC 和 Ninja。请先通过生产机的软件源安装 NVIDIA 驱动和 Toolkit，并确认 `nvidia-smi` 与 `nvcc --version` 可用。其他显卡需按架构设置 `TORCH_CUDA_ARCH_LIST`，默认是 4090 的 `8.9`。

以 Ubuntu 22.04 为例，在源码目录执行：

```bash
sudo apt-get install python3.10 python3.10-venv python3.10-dev \
  build-essential git ninja-build ffmpeg libgl1 libglib2.0-0 libgomp1 libpulse0
export CUDA_HOME=/usr/local/cuda-12.4
bash scripts/setup_env.sh
```

安装脚本建立 `package/.venv-maps` 和 `package/.venv-da3` 两套独立环境，分别可用 `bash scripts/setup_env.sh maps` 和 `bash scripts/setup_env.sh da3` 重建。可用 `SEMANTIC_GEN_PYTHON` 指定 Python 3.10，可用 `SEMANTIC_GEN_PYPI_INDEX` 和 `SEMANTIC_GEN_TORCH_INDEX` 指定可达的镜像。原快照中的 OpenCV 5.0.0 在公开 PyPI 的同版本路径不可用，目前安装脚本固定使用 4.11.0.86。这个变化需要在生产机 GPU 运行中核对取帧和深度对齐结果。GradSLAM 按冻结版本独立安装，其旧包元数据中的 Open3D/ChamferDist 约束由建图环境的对应固定版本提供。

## 2. 准备模型和缓存

```bash
export SEMANTIC_GEN_MODEL_ROOT=/data/semantic-mapping/models
export SEMANTIC_GEN_CACHE_ROOT=/data/semantic-mapping/cache
python3.10 scripts/prepare_models.py list
```

按输出中的 `source` 获取文件，写到对应的 `destination`。`bytes` 和 `sha256` 来自已使用的权重快照，`stages` 标明使用阶段。CLIP-H/14 的缓存链接和小型配置由下面的 `init-cache` 命令建立，无需下载两份。建议给这套流程使用独立的缓存根目录，避免与其他任务使用的 `refs/main` 冲突。SAM2、YOLO-World、MobileSAM、LSeg、DA3、CLIP 和 timm backbone 都有各自来源。受许可或访问限制的权重由有权限的维护者从批准的来源提供，按相同路径和哈希校验。下载失败或字节不一致时先核对来源，建图预检会报告具体文件与阶段。

```bash
python3.10 scripts/prepare_models.py init-cache
python3.10 scripts/prepare_models.py verify
```

校验过程需要读取约 10 GiB 模型文件。建图默认使用本地文件和 Hugging Face 离线缓存。可在联网机器上先准备模型与缓存，再整体传到生产机，保留模型目录与缓存的相对布局并重新校验。LSeg 的 Google Drive 大文件可用建图环境中的 `gdown` 获取。历史快照也可作为经授权的相同权重来源。 [模型清单](package/docs/mapping_provenance/weights.json)

## 3. 输入预检与正式建图

提供 ROS 2 MCAP 目录（含 `metadata.yaml`），或包含该录包的 ZIP。默认话题是 `/aima/hal/camera/front/image_raw`、`/aima/hal/lidar/front/point_cloud` 和 `/map_odometry`。原生录包解码器使用 NV12。标定须包含实际相机内参、尺寸、畸变和 `camera`、`rslidar_front` 到同一 base 的外参。预检会比对录包的帧和里程计 child frame，实际标定精度仍需用录制验证。

```bash
python3.10 run.py build \
  --bag /data/recordings/site-a \
  --calibration /data/calibration/site-a.json \
  --output-root /data/mapping-runs --prepare-only

python3.10 run.py build \
  --bag /data/recordings/site-a \
  --calibration /data/calibration/site-a.json \
  --output-root /data/mapping-runs
```

`--prepare-only` 写入 `prepared_not_mapped`。正式运行可以新建任务，也可通过输出中打印的 `AUTOBUILD_WORKSPACE` 恢复该次任务。恢复会重新检查输入、配置、模型、运行环境和各阶段产物签名。更换录包、标定或类别时新建任务。调试时可以用 `--max-samples` 限制采样数量，正式生成使用默认的全量关键帧。

```bash
python3.10 run.py build --resume /data/mapping-runs/<run>
python3.10 run.py stage --config /data/mapping-runs/<run>/run_config.yaml --stage doctor
```

按对应录制附加 `--label-yaml /data/labels/site-a.yaml` 可生成有限类别副本。已按现有无损原始观测协议导出的输入包可通过 `--prepared-input /data/observations/bundle.zip` 提供自带标定和消息。两种输入入口任选其一。处理其他录包的 topic 配置时，可从 `package/code/robot_mapping/configs/new_recording_defaults.yaml` 派生一份独立模板，用 `prepare_new_run.py --help` 查看自定义输入方法，再对生成的运行配置执行 `run.py stage --stage all`。

## 4. 检查交付

| 路径（相对运行目录） | 内容 |
|---|---|
| `delivery/MAPPING_RESULT.json` | 任务状态、产物定位和交付校验。正式运行以退出码与 `success` 一起判定。 |
| `native_maps/validation.json` | HMSG 图结构、CG 实例、VLMaps 特征与基本数值检查。 |
| `share/*_native_three_maps.tar.gz` | 原生三图归档和校验清单，可在解压目录运行 `sha256sum -c SHA256SUMS`。 |
| `label_locked/LABEL_ALIGNMENT.json` | 提供标签时生成的独立副本报告及缺少候选的类别。 |
| `logs/`、`RUN_STATUS.json`、`TIMING_REPORT.json` | 阶段日志、失败原因和耗时。 |

CPU 契约测试命令如下。测试用合成 MCAP、标定和权重字节检查输入及恢复逻辑。三图验证覆盖产物结构，语义精度、物体身份和导航仍需各自的验收。

```bash
PYTHONPATH=package/code/robot_mapping package/.venv-maps/bin/python \
  -m unittest discover -s tests -v
```

## 5. 与 semantic_map_builder 一起使用

现有 builder 使用本目录的 HMSG/ConceptGraphs 原生入口。更新其配置中的 `backend`，并在首次运行前使用上文命令核对模型：

```json
{
  "package": "/opt/semantic_map_gen_production/package",
  "python": "/opt/semantic_map_gen_production/package/.venv-maps/bin/python",
  "depth_python": "/opt/semantic_map_gen_production/package/.venv-da3/bin/python",
  "environment": {
    "SEMANTIC_GEN_MODEL_ROOT": "/data/semantic-mapping/models",
    "SEMANTIC_GEN_CACHE_ROOT": "/data/semantic-mapping/cache",
    "SEMANTIC_GEN_CLIP_ROOT": "/data/semantic-mapping/models/clip",
    "HF_HUB_CACHE": "/data/semantic-mapping/cache/huggingface",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1"
  }
}
```

`recording` 输入的旧配置如写死 `depth.model`，请将它指向相同模型根目录的 `DA3METRIC-LARGE`，或者删除该字段使用后端默认值。builder 导出 CG 对象和 HMSG 区域。

## 源码来源

- `runner/` 和 `package/code/robot_mapping/` 为命令行编排及原生入口。
- `package/code/third_party/` 保留三图及其依赖的冻结源码，DA3 源码在 `package/code/Depth-Anything-3/`。随附上游提交、许可证、原始补丁和文件哈希。 [来源说明](provenance/README.md)

从工作区根目录用下面的命令打包已经安装过的源码目录。录包和运行输出继续存放在单独的数据目录。

```bash
tar --exclude='__pycache__' --exclude='.venv-*' \
  -czf semantic_map_gen_production-source.tar.gz \
  semantic_map_gen_production/{.gitignore,AGENTS.md,README.md,run.py,runner,scripts,tests,provenance} \
  semantic_map_gen_production/package/{code,docs,kit}
```

本次实施记录见 [验证记录](provenance/VALIDATION.md)。
