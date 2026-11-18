# 2026-11-18 重构验证

- **源码核验**：新目录约 11 MiB（磁盘占用）。源码目录中没有环境、模型 checkpoint、历史录包或运行结果。逐项对比保留的第三方文件与旧基线 SHA-256，文件内容全部匹配。
- **CPU 契约与入口**：独立 Python 3.10 环境运行 `PYTHONPATH=package/code/robot_mapping python -m unittest discover -s tests -v`，8 项通过。用合成 MCAP 实际调用 `run.py build --prepare-only` 和 builder 的 `--prepare-only`，均返回 `prepared_not_mapped`。对预检任务用空模型目录执行正式恢复，doctor 报告 12 项缺失，交付状态为 `failed`。
- **真实录包输入**：使用隔离建图环境对旧基线 2026-10-21 的 MCAP 和对应标定执行 `run.py build --prepare-only`，结果为 `prepared_not_mapped`。以同一份真实标定单独比较 OpenCV 5.0.0 和 4.11.0 的新相机内参、ROI 和去畸变像素映射，最大绝对差为 0。
- **CPU 原生预处理**：`run.py stage --stage prepare` 在合成 MCAP 上完成 NV12 解码、去畸变和 LiDAR 投影。第一次测试夹具把点放在相机深度零平面，投影为空且原生预览报错。修正合成点后，重测得到 1 帧图像、4 个地图点和 4 个有效深度像素。
- **模型和缓存**：用基线模型文件校验 12 项模型或缓存资产。DA3 配置文件从清单固定 revision 获取，其 SHA-256 与基线相同。
- **DA3 安装与推理**：在隔离目录重建 Python 3.10 / PyTorch 2.5.1+cu124 环境，`pip check` 通过。离线加载固定的 DA3METRIC-LARGE，对合成 RGB 图做一次 CPU 推理，深度形状为 `1 × 112 × 168`，元素有限且为正。
- **建图环境**：从独立 Python 3.10 环境和 CUDA Toolkit 12.4 构建，重建了 ChamferDist 1.0.3 与固定提交的 PyTorch3D 0.7.8。系统 Python 缺少开发头文件时，`scikit-fmm` 的构建失败。换用带开发头文件的独立 Python 3.10 后，按脚本完成建图环境。网络下载较慢的 Open3D 和 SymPy 使用公开镜像的相同 wheel，并比对 PyPI SHA-256。两套环境准备后，`run.py stage --stage doctor` 在合成 MCAP 配置上 **43/43** 通过，覆盖运行时、CUDA、模型与缓存、原生源码路径和录包合同。
- **真实 GPU 三图**：本机 RTX 4090 验证期间约 0.8 GiB 空闲显存，本次没有运行完整录包三图。结构测试、CPU 模型推理与 GPU 端到端建图分别计入验收。
