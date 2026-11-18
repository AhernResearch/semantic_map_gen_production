# 冻结基线与移植记录

源码从工作区历史 `semantic_map_gen` 提取。各第三方来源见 `../package/docs/mapping_provenance/source_inventory.json`，当时使用的补丁和环境快照见同目录的 `verified_run_20261015/`。`vendor_sha256.json` 记录保留文件的 SHA-256，可与原冻结树逐项比对。它包含 CLIP tokenizer 字典和 HMSG 映射桥实际使用的类别描述向量，两者是运行资产。文件清单不含模型权重。

`local_portability.patch` 保存上一次移植的历史记录，其中的旧机器用户目录已替换为占位路径。当前移植调整入口和运行路径，抽出生产 CLI、删去网页服务与历史场景耦合，新增显式模型身份校验和环境构建脚本。原生三图入口、算法文件、采样和关联默认参数继承冻结基线。公开 pip 安装的 OpenCV 版本差异见根 README。
