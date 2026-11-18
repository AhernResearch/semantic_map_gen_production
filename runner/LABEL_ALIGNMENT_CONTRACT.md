# 可选标签YAML：有限类别锁定

标签文件缺省时照常运行原生三图。提供version:3 annotations YAML时，本模块保留其原ID、名字、已有位姿字段及原文件字节；robot_pos是导航站位，camera字段可能是占位，均不进入物体中心/特征分类。重复名字的多个ID共享类别，不能据此绑定同一个物理对象。

准备命令：

```bash
python label_alignment.py prepare --yaml SOURCE.yaml --workspace NEW_WORKSPACE --config NEW_WORKSPACE/run_config.yaml
```

写`NEW_WORKSPACE/LABEL_PREPARATION.json`，含`success/config_path/yaml_snapshot/yaml_sha256/detector_classes`。只另存config覆盖`conceptgraphs.classes`，原config/YAML不写回。明确英译来自代码GLOSSARY或`--translations JSON`；当前aiday_1021新增叫法“黄色消防帽/黑色手提箱”分别映射yellow hard hat/black toolbox。未知中文不猜译。配置中的workspace原值不改；调用者先准备正确的新run config。

完成原生三图且native validation通过后：

```bash
python label_alignment.py postprocess --yaml SNAPSHOT.yaml --workspace WORKSPACE --output WORKSPACE/label_locked/new_version --encode-text
```

也可用`--text-cache`。`--encode-text`才显式加载本地H14/B32，串行短文本推理、释放模型，不训练/下载权重。默认阈值H=.25、B=.20、margin=.02；它们是未独立标定的余弦候选门槛，不是概率或精度保证。低分/低margin/无有效feature保留unknown；unknown不是YAML外新增目标。维度或文本cache身份/权重SHA错误直接失败。

稳定报告`OUTPUT/LABEL_ALIGNMENT.json`（保留RESULT alias）含：

- `success/artifacts{hmsg:{path},conceptgraphs:{path},vlmaps:{path}}/artifact_paths`。
- `missing_target_categories`与`yaml_target_status`：无类别支持写未检出候选；有候选也不证明某个YAML实例ID已定位。
- `source_hashes/parity_checks/feature_space_contract/text_provenance/claims`。
- `claims`明确未训练、未创建物理ID绑定、未独立精度验证、非完备表面、无导航授权、原生输入不改。

HMSG JSON/PLY与CG原生pkl.gz在独立目录。对象name/class_name为YAML类别候选或unknown；几何、原图像/文字feature、mask、bbox、颜色与帧来源不改。HMSG/CG使用真实H14图像feature对真实H14 text；VLMaps使用真实B32空间512原grid_feat对真实B32 text。

VLMaps为独立HDF5副本，保留每个原dataset值/类型/形状，追加`locked_label_index/locked_label_score/locked_label_margin/label_names/label_names_en/label_yaml_sha256/label_yaml_source_bytes/locked_text_features_B`。unknown索引=-1；同类多个实例或体素不获得YAML annotation_id定位真值。进度只打印`label objects n/N`、`label voxels n/N`，不把记录数比例描述为总时间进度。

原部署producer明确HMSG/CG为ViT-H-14，LSeg/VLMaps为ViT-B/32空间；代码记录producer SHA并核对原validation维度与cache encoder/已部署权重SHA。真实cache按英文描述选择，禁止按新旧annotation数组位置或ID猜匹配。

`robot_navigation/label_guided_detection.py`、`robot_mapping/segment_label_points.py`与`build_label_aligned_maps.py`已有源图/mask/真实LiDAR及明确核验的独立ID层流程。这里记录工具入口，但不启动长重建、不从最近物体/robot_pos强配；实例级绑定需另行核对原YAML、源图/mask/实测几何SHA和人工式核验。
