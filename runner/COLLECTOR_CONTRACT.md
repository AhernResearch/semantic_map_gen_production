# 原生三图收集契约

`collect_outputs.py --workspace RUN --output NEW_DIR --require-complete` 只读取已生成数据，不执行模型、服务或机器人操作。`NEW_DIR`必须是新目录，位于当前运行目录内且在原生地图目录之外。`--check-only`只输出JSON，不创建tar或写源workspace。

成功要求三图真实存在且基本格式/结构可读、原`native_maps/validation.json`的`all_passed`及逐项检查均通过、没有未结束/失败的最新流水线阶段或源SHA不一致。`--require-complete`还要求all的14阶段记录、声明产物和三图总包/结果报告完整。历史失败在最新成功重跑后不再算当前失败；失败始终`success=false`、非零退出，并保留已找到产物路径。

`MAPPING_RESULT.json`的关键字段：

- `success/status/missing/warnings`：真实完成状态与中文缺口。
- `artifacts.hmsg_graph`：绝对graph路径、JSON/PLY结构计数、源tree SHA、独立`HMSG_graph_*.tar.gz`路径/SHA。
- `artifacts.conceptgraphs/vlmaps`：绝对`.pkl.gz/.h5df`路径、格式、源SHA；不执行未知pickle。
- `artifacts.pipeline_archives/reports/logs`：原总tar、报告、日志的路径/SHA。
- `generated_at`：收集报告生成时间，ISO8601 `Asia/Shanghai`。模型产物实际生成时间在阶段的`finished_at`/成功来源中，不能把旧产物检查时间当今天重跑时间。
- `stage_timing_s`：每阶段最新row的记录处理耗时秒；skip/缺时间为null。
- `stage_timing_details`：原start/end、exit、skipped与`successful_artifact_source`的原生成时间。
- `total_pipeline_stage_s`：只相加最新实际处理row，历史重试和复用来源耗时不重复加入。pipeline没有独立invocation ID，跨resume时该和不冒充本次端到端时间。
- `native_frame_count`：原validation的准备样本帧数；`map_observation_frames`分别表示原报告每图观测数量；`voxel_count`为原VLMaps特征行数。
- `input_volume`：若`run_preparation.json`存在，读取实际bag MCAP字节与提取记录的原archive字节。无元数据时留null，不估计。

这些耗时与输入规模可供producer的整体计时/2秒资源采样分析。collector不直接断言4090、CPU或磁盘是绝对瓶颈。模型未在本步骤训练；原生三图/合同检查不等于独立语义GT、完备表面、自动完整补全或真机成功。
