# 仓库整理与恢复

2026-10-05，活动代码统一为 **Qwen3.5-9B 英文 Wikipag 应用记录 → 五字段 card** 流程。唯一构建入口为 `scripts/build_wikipag_application_cards.py`。

## 本轮清理

从工作目录移出 87 个旧源码、配置、提示词、测试或说明文件，以及 9 个字节码缓存文件，覆盖：

- 旧单学科／批量直接提炼和基于题集的构建入口。
- 旧 smoke、评测、多语言生成、基线和卡库改写脚本及其专用模块。
- 新流程不调用的 Wiki 下载／FAISS 服务脚本。
- 旧流程专用配置、提示词、测试，以及混合旧方案和实验结果的入口文档。

新流程所需的模型配置、客户端和文件读写已经独立到 `src/construction/`；继续使用通用的凭据解析和缓存组件。安装依赖缩减为 httpx、pydantic，pytest 为测试依赖。

清理后 50 项离线检查通过；活动源码无缺失的本地模块引用，README 与当前说明无失效文件链接。归档中的 96 份文件及 6 份既有卡库／运行记录保护文件均核对过 SHA-256。

Qwen 服务已实测联通；单主题构建走完审核、一次修订及复核，最终因字段引用支持不足被拒绝，导出 0 张卡片。恢复新增调用为 0，7 份输出逐字节一致。本地运行记录位于 `runs/qwen35_pipeline_cleanup_20261005/README.md`，不随源码发布；这不是稳定产出合格卡片或答题收益的验证。

模型默认固定为 Qwen3.5-9B，API 协议由当前入口显式选择，不再继承开发工具配置中的模型或协议。构建协议版本为 `wikipag-application-cards-v3`，旧 run 应使用对应源码快照恢复，不能由新版直接续写。

## 恢复位置

本轮归档位于仓库之外：

```text
/home/work/migoo_ai_post-train/linxuan/CA_archive/pipeline_cleanup_20261005/
  before/                  # 清理前的 115 份源码和文档快照
  removed/                 # 按原相对路径保存的移出文件
  before_manifest.json     # 清理前文件 SHA-256
  plan.json / moved.json   # 移出清单、归档路径与 SHA-256
  protected_assets.json    # 既有卡库入口和主要产物校验值
  git_status_before.txt    # 操作前的完整工作区状态
  head.txt                 # 操作前 HEAD
```

归档包含此前尚未提交的修改及未跟踪文件。恢复时先核对清单与哈希，并在独立目录恢复；不要用旧快照覆盖当前源码。历史脚本依赖原相对目录布局，不能假设移动后仍可直接执行。

## 旧实验记录清理

按用户要求，当前版本移除旧实验汇总、基线与消融结果、评测输出、图表、论文材料及废弃脚本。远端新增的两份旧实验文档也一并移除；Git 提交历史正常合并并保留。

本地旧 `runs/` 条目和 `artifacts/` 共 39 项移至仓库外：

```text
/home/work/migoo_ai_post-train/linxuan/CA_archive/obsolete_experiments_20261005/
  runs/                 # 旧实验、历史卡库、缓存及此前的整理记录
  artifacts/            # 旧评测产物及历史资产
  plan.json / moved.json
  verification.json
```

移动保留文件内容及目录结构，核对了各移动入口的 inode、大小和修改时间，三个历史卡库相对软链接在归档内仍可解析。原始 Wikipag 语料和外部索引未改动。旧记录不改写来源、模型或结果，也不作为当前流程的产物。

本地 `runs/` 仅保留 `qwen35_pipeline_cleanup_20261005/` 的当前流程验证记录，继续由 Git 忽略。仓库只发布当前构建源码、提示词、说明和测试；生成数据、模型缓存、日志、向量索引及外部归档不随源码上传。
