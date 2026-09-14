# 当前实验与分析清单

更新时间：2026-09-14。

研究版本：英文 Wikipedia/Wikipag 概念用法库，冻结答题模型，无额外 SFT 或 GRPO。

核验仓库：[Traveler03/CA][repo]。本次核验时 `main` 为 `bda6641d818db682ba95e7dee024b04f7c864aa2`，下列远程证据均固定到该提交。

本清单整理已存在的报告、汇总数据和稿件，不运行新的答题实验，不改写预测，不将理论值补成实测值。未找到归档证据，不等于断言实验从未在其他位置运行。

## 1. 总览

| 类别 | 当前可确认内容 | 状态与计数方式 |
| --- | --- | --- |
| 主实验 | 3 个答题模型，11 个模型与方法配置 | 已归档实测，见第 3 节 |
| Qwen 运行阶段消融 | 完整 CA + 6 个变体 | 已归档实测；完整 CA 与主实验 M03 相同，不重复计数 |
| 主实验与消融合计 | 17 个不重复的低资源评测配置 | 每个配置覆盖 47,815 个实例；这是配置数，不是 17 次独立重复或多随机种子实验 |
| 英语补充评测 | Qwen CA 的 Global-MMLU/en、MMLU-ProX/en | 已归档实测，不混入五语言主实验均值 |
| 提示版本历史对照 | 本地化指令提示与旧英语指令提示 | 汇总中有记录；旧版本单独预测文件尚未定位，见第 5 节 |
| 分语言与跨模型比较 | 主实验的五语言展开、CA 相对 tCRAG 的差值 | 已完成的结果拆分，不是新增模型运行 |
| 运行统计 | 输出有效率、Qwen CA 调用与回退统计、Qwen tCRAG 延迟 | 已有记录；不等于完整公平的效率对比 |
| 知识库结构分析 | 全库规模、学科分布、字段数量、文本长度 | 已完成全库统计，不是下游消融 |
| 字段可见性分析 | 700/1,200 字符前缀下的字段可见性 | 已完成全库普查，不是语义质量或检索召回测量 |
| 条件理论分析 | 无效输出修复边界、选项变化边界、阈值不变性、审计样本量等 | 已完成推导，不计入实测实验数量 |
| 旧稿中的其他实验 | 内容消融、SFT、构建消融、成本、人工分析等 | 旧稿待核验，见第 8 节，不计入当前已完成实验 |

## 2. 统一口径

### 2.1 数据集与语言

| 数据集 | 每种语言的实例数 | 五种低资源语言合计 |
| --- | ---: | ---: |
| Global-MMLU | 4,341 | 21,705 |
| MMLU-ProX | 5,222 | 26,110 |
| 合计 | 9,563 | 47,815 |

五种低资源语言固定为 `bn`（孟加拉语）、`hi`（印地语）、`ne`（尼泊尔语）、`sw`（斯瓦希里语）、`te`（泰卢固语）。下文分语言表均采用这一顺序。

准确率单位为百分比。综合准确率由正确题数相加后除以实例数得到，不对已经四舍五入的分数再求平均。各语言在同一数据集内实例数相同。翻译版本可能共享源题，不能把所有语言行当成相互独立的样本来计算显著性。

### 2.2 模型与方法命名

| 清单简称 | 实际模型或实现 |
| --- | --- |
| Qwen3-8B | `Qwen3-8B`，答题模型 |
| Ministral-3-8B | `Ministral-3-8B-Instruct-2512`；归档运行目录标记 BF16 |
| Llama-3.1-8B | `Llama-3.1-8B-Instruct` |
| 检索编码器 | `Qwen3-Embedding-4B`，不是答题模型 |
| tCRAG | 归档报告标签；当前稿件称 tRAG，代码方法为 `trag`，即问题翻译后检索英语 Wiki |
| CORAL-Wikipag | 使用 Wikipag 的 CORAL 适配实现；已核对的启动配置是一轮检索，包含双查询与答案门控，不能直接称为原论文完整多轮复现 |
| CA | 概念查询、学科过滤检索、重排与检查、基于概念用法卡片答题 |

完整 CA 通常从 20 个候选中选择至多 5 张卡片；不是每题必定使用 5 张。检查阶段还会改写指导，因此“重排”不等于仅调整排序。

### 2.3 知识库与评测版本

按作者确认，完整知识库的 **60,653 张卡片均为 clean**，覆盖 57 个学科；统计不再按旧标签分割知识库。

已归档答题结果使用元数据规范化前的序列化文本。规范化保留知识正文与来源标识，但改变了部分文本头尾，因此当前导出需要重建索引；尚不能把归档成绩描述为规范化后的重新评测。更正细节保留在本地 `paper/source-ledger.md`，未随本次清单发布。

## 3. 已归档主实验

### 3.1 完整比较表

以下 11 个配置均有覆盖完整实例的归档汇总。每行低资源评测总数均为 47,815。

| 编号 | 答题模型 | 方法 | Global-MMLU | MMLU-ProX | 综合 | 正确题数 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| M01 | Qwen3-8B | 原语言零样本 | 47.24 | 29.32 | 37.45 | 17,909 |
| M02 | Qwen3-8B | tCRAG | 58.99 | 43.16 | 50.34 | 24,071 |
| M03 | Qwen3-8B | CA | 62.55 | 45.96 | 53.49 | 25,576 |
| M04 | Ministral-3-8B | 原语言零样本 | 54.44 | 29.83 | 41.00 | 19,604 |
| M05 | Ministral-3-8B | tCRAG | 63.57 | 41.93 | 51.75 | 24,746 |
| M06 | Ministral-3-8B | CORAL-Wikipag | 61.98 | 38.83 | 49.34 | 23,592 |
| M07 | Ministral-3-8B | CA | 72.13 | 58.46 | 64.67 | 30,921 |
| M08 | Llama-3.1-8B | 原语言零样本 | 31.83 | 19.70 | 25.21 | 12,052 |
| M09 | Llama-3.1-8B | tCRAG | 45.78 | 28.78 | 36.49 | 17,450 |
| M10 | Llama-3.1-8B | CORAL-Wikipag | 46.03 | 28.74 | 36.59 | 17,495 |
| M11 | Llama-3.1-8B | CA | 41.99 | 27.77 | 34.23 | 16,366 |

证据：M01 见[零样本汇总][zero-json]；M02 见[tCRAG 汇总][trag-json]；M03 见[Qwen CA 对比汇总][ca-json]；M04-M11 见[双模型汇总][two-json]。

**未在该归档中定位 Qwen3-8B 的完整 CORAL-Wikipag 结果，不应自行补一行分数。** 上表也不代表所有模型都优于所有基线。

### 3.2 主实验按语言展开

每个单元格合并该语言在两个数据集上的 9,563 个实例。

| 编号 | 模型 / 方法 | bn | hi | ne | sw | te |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| M01 | Qwen3-8B / 原语言零样本 | 41.98 | 43.66 | 41.28 | 20.30 | 40.05 |
| M02 | Qwen3-8B / tCRAG | 52.53 | 53.60 | 52.81 | 40.88 | 51.90 |
| M03 | Qwen3-8B / CA | 56.10 | 57.15 | 55.37 | 43.37 | 55.46 |
| M04 | Ministral-3-8B / 原语言零样本 | 42.53 | 44.25 | 42.87 | 33.57 | 41.78 |
| M05 | Ministral-3-8B / tCRAG | 52.27 | 53.63 | 52.90 | 46.97 | 52.99 |
| M06 | Ministral-3-8B / CORAL-Wikipag | 50.51 | 51.66 | 49.92 | 44.40 | 50.21 |
| M07 | Ministral-3-8B / CA | 66.86 | 67.79 | 67.26 | 54.24 | 67.19 |
| M08 | Llama-3.1-8B / 原语言零样本 | 24.70 | 27.74 | 27.55 | 24.76 | 21.27 |
| M09 | Llama-3.1-8B / tCRAG | 36.95 | 37.85 | 36.39 | 34.52 | 36.76 |
| M10 | Llama-3.1-8B / CORAL-Wikipag | 37.33 | 38.19 | 36.34 | 34.85 | 36.23 |
| M11 | Llama-3.1-8B / CA | 34.61 | 37.58 | 35.02 | 31.33 | 32.59 |

各模型的 CA 相对 tCRAG 的综合差值如下，均先按正确题数计算再四舍五入：

| 模型 | CA 正确题数 | tCRAG 正确题数 | 差值（百分点） |
| --- | ---: | ---: | ---: |
| Qwen3-8B | 25,576 | 24,071 | +3.15 |
| Ministral-3-8B | 30,921 | 24,746 | +12.91 |
| Llama-3.1-8B | 16,366 | 17,450 | -2.27 |

Qwen 和 Ministral 在五种语言上的变化方向均为正，Llama 均为负。这是描述性结果，不是显著性检验，也没有确定模型差异的原因。

### 3.3 已有学科拆分

零样本、tCRAG、Qwen CA 分片及双模型汇总中均保留了学科维度统计，可以继续用于学科差异分析。当前已确认“学科统计已归档”，不等于已经完成有配对对照和显著性检验的学科结论。

Qwen CA 的第一个分片混合了多个数据集与语言；使用其中 `by_subject` 时必须保留其混合范围，不能直接当作单一数据集的学科结果。主实验比较应优先使用已对齐的数据集与语言汇总。

## 4. 已归档消融实验

### 4.1 Qwen3-8B 运行阶段消融

三类干预：是否执行检查、卡片相关性、卡片数量。每个配置均覆盖五种低资源语言与两个数据集，总计 47,815 个实例，汇总缺失数均为 0。

| 编号 | 配置 | Global-MMLU | MMLU-ProX | 综合 | 正确题数 |
| --- | --- | ---: | ---: | ---: | ---: |
| 参考 | 完整 CA（M03 复用） | 62.55 | 45.96 | 53.49 | 25,576 |
| A01 | 去掉 rerank/check | 64.32 | 70.65 | 67.78 | 32,408 |
| A02 | 全局随机卡片 | 50.16 | 40.77 | 45.03 | 21,533 |
| A03 | 同学科随机卡片 | 50.36 | 41.11 | 45.31 | 21,666 |
| A04 | Top-1 | 63.33 | 45.92 | 53.83 | 25,737 |
| A05 | Top-3 | 62.86 | 45.88 | 53.59 | 25,622 |
| A06 | Top-10 | 62.28 | 45.95 | 53.36 | 25,516 |

证据：[消融报告][ablation-report]、[消融汇总 JSON][ablation-json]。报告中包含逐配置目录与预测归档索引。

| 编号 | 实际改变 | 可用于讨论的问题 | 解释边界 |
| --- | --- | --- | --- |
| A01 | 绕过候选重排、检查和指导改写，直接使用检索卡片 | 当前检查流水线是否改善效果 | 同时改变选择、改写、源文本截断与调用次数，不是纯排序消融 |
| A02 | 从全库随机提供卡片 | 任意背景知识是否能替代相关卡片 | 不是卡片字段或合成流程的消融 |
| A03 | 从同学科随机提供卡片 | 仅学科相关性是否足够 | 不能直接衡量某一字段的贡献 |
| A04-A06 | 改变卡片数量设置 | 更多卡片是否持续有益 | 不是训练消融，也没有据此测得完整成本曲线 |

这些结果仅确认在 Qwen3-8B 上归档了这一套消融，不能当作 Ministral 或 Llama 也完成了同样的实验。

### 4.2 Global-MMLU 消融按语言展开

每个单元格分母为 4,341。

| 配置 | bn | hi | ne | sw | te |
| --- | ---: | ---: | ---: | ---: | ---: |
| 完整 CA（M03 复用） | 67.40 | 69.29 | 65.24 | 46.35 | 64.48 |
| 去掉 rerank/check | 69.43 | 71.71 | 67.31 | 46.42 | 66.71 |
| 全局随机卡片 | 53.37 | 56.51 | 51.55 | 39.00 | 50.36 |
| 同学科随机卡片 | 53.61 | 56.69 | 52.02 | 39.28 | 50.22 |
| Top-1 | 68.44 | 70.40 | 66.39 | 46.05 | 65.38 |
| Top-3 | 67.68 | 70.15 | 65.68 | 46.14 | 64.66 |
| Top-10 | 67.52 | 68.99 | 64.87 | 45.84 | 64.18 |

### 4.3 MMLU-ProX 消融按语言展开

每个单元格分母为 5,222。

| 配置 | bn | hi | ne | sw | te |
| --- | ---: | ---: | ---: | ---: | ---: |
| 完整 CA（M03 复用） | 46.71 | 47.05 | 47.17 | 40.88 | 47.97 |
| 去掉 rerank/check | 72.83 | 73.61 | 73.54 | 60.13 | 73.17 |
| 全局随机卡片 | 41.11 | 41.98 | 42.55 | 36.27 | 41.96 |
| 同学科随机卡片 | 41.29 | 42.68 | 42.88 | 36.60 | 42.13 |
| Top-1 | 46.97 | 47.34 | 46.96 | 40.54 | 47.82 |
| Top-3 | 46.48 | 47.45 | 46.94 | 40.31 | 48.20 |
| Top-10 | 46.88 | 47.20 | 47.22 | 40.71 | 47.74 |

## 5. 其他已有评测与运行统计

### 5.1 英语补充评测

Qwen3-8B 的完整 CA 归档另外包含以下英语结果，证据为同一份 [CA 汇总][ca-json]。

| 数据集 / 语言 | 实例数 | 正确题数 | 准确率 |
| --- | ---: | ---: | ---: |
| global_mmlu / en | 4,341 | 3,568 | 82.19 |
| mmlu_prox / en | 5,222 | 2,581 | 49.43 |

加入这两个英语分层后，CA 归档共 57,378 个实例，其中 31,725 个正确。该总量与五语言主实验的 47,815 不能混用；英语补充评测不是五语言主实验的新基线。

### 5.2 零样本提示版本历史对照

[本地化零样本汇总][zero-json] 的 `comparison_old_english_prompt_full` 中记录了旧提示版本对照，两个版本的比较实例数为 47,815。

| 版本 | 正确题数 | 综合准确率 |
| --- | ---: | ---: |
| 旧英语指令提示（摘要内历史对照） | 18,782 | 39.28 |
| 本地化指令提示（M01） | 17,909 | 37.45 |

这里比较的是答题指令的提示语言版本，不是把所有题目换成英语。旧版本的单独预测路径尚未在本次仓库快照中定位，因此标记为“摘要内历史对照”，不并入第 3 节当前主表，也不据此宣称已完成受控的查询语言消融。

### 5.3 输出有效性与完成覆盖

各主实验配置的汇总覆盖均完整。无效输出与缺失输出是不同概念：即使预测文件完整，也可能存在无法解析的答案。

| 编号 | 模型 / 方法 | 记录为无效的输出数 | 记录的有效率 |
| --- | --- | ---: | ---: |
| M01 | Qwen3-8B / 原语言零样本 | 0 | 100.00 |
| M02 | Qwen3-8B / tCRAG | 9 | 99.98 |
| M03 | Qwen3-8B / CA | 3 | 99.99 |
| M04 | Ministral-3-8B / 原语言零样本 | 96 | 99.80 |
| M05 | Ministral-3-8B / tCRAG | 177 | 99.63 |
| M06 | Ministral-3-8B / CORAL-Wikipag | 6 | 99.99 |
| M07 | Ministral-3-8B / CA | 43 | 99.91 |
| M08 | Llama-3.1-8B / 原语言零样本 | 613 | 98.72 |
| M09 | Llama-3.1-8B / tCRAG | 50 | 99.90 |
| M10 | Llama-3.1-8B / CORAL-Wikipag | 0 | 100.00 |
| M11 | Llama-3.1-8B / CA | 0 | 100.00 |

这些数量来自 `parsed`/`valid` 汇总。已审查的部分汇总器信任存储标记，并将缺失标记默认视为有效；本清单没有独立重新解析全部原始答案。有效率高不等于答案解析一定正确。

### 5.4 Qwen CA 的调用、回退与最终卡片数

以下是两个实际运行分片的汇总，不是两种算法消融。

| 分片范围 | 实例数 | 记录的模型调用数 | 查询改写回退数 | 重排检查回退数 | 平均最终卡片数 | 运行错误数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Global-MMLU 六语言 + MMLU-ProX 英语 | 31,268 | 93,804 | 485 | 49 | 2.536 | 0 |
| MMLU-ProX 五种低资源语言 | 26,110 | 78,330 | 234 | 196 | 2.582 | 0 |

证据：[混合分片汇总][ca-run-a]、[MMLU-ProX 低资源分片汇总][ca-run-b]。两份汇总中的调用数均为实例数的三倍。调用数不能替代输入输出 token、GPU 时间或货币费用；回退计数也尚未与逐题得失关联起来。

### 5.5 Qwen tCRAG 的已有延迟记录

[tCRAG 汇总][trag-json] 的 `overall.latency_s` 包含实测延迟统计：

| 指标 | 秒 |
| --- | ---: |
| 平均 | 6.679 |
| P50 | 6.868 |
| P95 | 9.691 |
| 最大 | 14.126 |

因此不能笼统说“完全没有延迟数据”。准确说法是：**已有该运行的延迟记录，但尚无对齐硬件、并发与推理配置的全方法效率比较，也没有据此得到完整构建成本和摊销成本。**

## 6. 已完成的资源与接口分析

### 6.1 全库结构统计

统计输入为[归档完整卡片库][bank]。派生统计保留在本地 `paper/latex/figures/wikipag/bank_profile_source.json`，未随本次清单发布；本节列出已核对的结果。

| 统计项 | 结果 |
| --- | ---: |
| 完整 clean 卡片数 | 60,653 |
| 学科数 | 57 |
| 每学科卡片数中位数 | 1,007 |
| 每学科最少 / 最多卡片数 | 531 / 1,916 |
| 平均知识正文长度（字符） | 2,027.61 |
| 平均适用条件 / 规则 / 易错点条目数 | 3.03 / 4.83 / 3.61 |
| 满足当前字段数量下限的卡片数 | 47,948 |

知识正文长度不含元数据头尾。字段数量下限是至少两条适用条件、两条规则和一条易错点；不满足这一当前下限的卡片仍属于完整 clean 库。结构完整性与语义正确率不同，不能将这些统计写成人工质量通过率。

### 6.2 卡片前缀的字段可见性普查

范围为[归档完整卡片库][bank]的全部 60,653 张描述，卡片等权计数。派生统计保留在本地 `paper/latex/figures/wikipag/prefix_coverage_source.json`，未随本次清单发布。

| 源文本前缀上限 | 完整定义 | 至少一条完整适用条件 | 至少一条完整规则 | 至少一条完整易错点 |
| --- | --- | --- | --- | --- |
| 700 字符 | 58,808（96.96%） | 45,986（75.82%） | 8,162（13.46%） | 1,356（2.24%） |
| 1,200 字符 | 60,653（100.00%） | 60,653（100.00%） | 46,205（76.18%） | 14,964（24.67%） |

700 与 1,200 是归档运行的源文本字符上限，不是 token 数。普查衡量完整字段项及续行能否出现在前缀中，不衡量其语义正确性、问题相关性或检查器生成的文本。它提供了接口信息暴露的证据，但没有证明截断造成了去掉检查后的全部增益。

### 6.3 工程正确性验证

已有卡片元数据可逆更正检查、前缀解析与边界测试，以及中英文论文数值和引用一致性检查。上一轮仓库测试记录为 36 项通过。

这些属于工程验证，不计入主实验或消融数量。本次清单整理只读取并核对记录，没有重新执行答题、训练或索引构建。

## 7. 已完成的算术与理论分析

**本节均不是新增下游实验，不用于填充缺失的实测结果。**

| 编号 | 分析 | 已有结论 | 适用范围 |
| --- | --- | --- | --- |
| T01 | 仅修复记录为无效的输出 | CA 相对 tCRAG 的差值范围：Ministral 为 `[+12.54,+13.00]`，Llama 为 `[-2.37,-2.27]` 个百分点 | 固定全部已标记有效的输出，仅允许无效输出变正确；不是置信区间或实际修复结果 |
| T02 | 最终选项变化的敏感性 | 假设 47,815 题中至多 2,390 题的最终选项变化，则 Qwen、Ministral、Llama 的准确率范围分别为 `[48.49,58.49]`、`[59.67,69.67]`、`[29.23,39.23]` | 5% 是假设，不是任何消融已观测的变化比例；不适用于现有大幅提升的去检查变体 |
| T03 | 冗余综合分数阈值 | 当前硬门槛下 `Q >= 0.95`，删除默认 `Q >= 0.72` 测试改变 0 个卡片接受决定 | 固定草稿、接受字段和验证结果；不是实测下游准确率变化为 0 |
| T04 | 人工审计样本量规划 | 误差容限 10.0、7.5、5.0 个百分点，对应充分抽样次数 185、328、738 | Hoeffding 界；单指标、独立均匀有放回抽样、固定二元标签、95% 保证；不是已完成标注或人工通过率 |
| T05 | 指导收益与干扰 | `Delta = p*g - (1-p)*h`；既有假设参数在 `p=0.6` 下给出 `+3.2` 或 `-0.8` 个百分点 | 仅为条件说明，没有拟合具体模型、语言或消融 |
| T06 | 构建成本摊销 | `C_avg(N) = C_build/N + c_online` | 成本恒等式；没有代入完整实测构建费用，也不能宣称已达到收支平衡点 |

T01 的输入来自[双模型汇总][two-json]，T02 的参考正确题数来自[Qwen CA 汇总][ca-json]与[双模型汇总][two-json]。T03 对应[构建评分函数][builder]。其余假设和适用范围在本表中列出；完整推导保留在本地 `paper/latex/wikipag_acl_zh_draft.tex` 附录 G，未随本次清单发布。

## 8. 旧稿待核验的实验

以下条目确实在[旧版英文稿](https://github.com/Traveler03/CA/blob/bda6641d818db682ba95e7dee024b04f7c864aa2/paper/latex/acl_latex.tex)中出现，但没有在本次核验的当前归档中找到足以对应其数字、模型和知识库版本的完整运行证据。旧稿主要涉及 CA-SFT 和另一版记忆设置，不能直接搬入当前冻结模型的 Wikipag 主表。

本节是“待追溯条目索引”，不是认定这些实验已经完成，也不是认定旧稿数字是理论值。为避免误用，不在这里复制未核验的准确率、显著性或人工结果。

| 编号 | 旧稿中的实验或分析 | 旧稿位置 |
| --- | --- | --- |
| L01 | Base、Direct-SFT、CA-SFT 下的 Global-MMLU 主实验及 Zero-shot、CoT、Cross-CoT、Self-Translate、SoT、CA-only、CA-Mem 对照 | [主结果与模型设置](https://github.com/Traveler03/CA/blob/bda6641d818db682ba95e7dee024b04f7c864aa2/paper/latex/acl_latex.tex#L423) |
| L02 | INCLUDE 与 MGSM 的扩展评测 | [额外基准](https://github.com/Traveler03/CA/blob/bda6641d818db682ba95e7dee024b04f7c864aa2/paper/latex/acl_latex.tex#L701) |
| L03 | 无记忆、随机记忆、概念错配记忆、对齐记忆 | [旧稿检索质量组](https://github.com/Traveler03/CA/blob/bda6641d818db682ba95e7dee024b04f7c864aa2/paper/latex/acl_latex.tex#L491) |
| L04 | Description Only、Usage Summary Only、Full Memory | [旧稿内容消融组](https://github.com/Traveler03/CA/blob/bda6641d818db682ba95e7dee024b04f7c864aa2/paper/latex/acl_latex.tex#L496) |
| L05 | 原语言问题、英语问题、概念、概念加描述的检索查询对照 | [旧稿检索策略组](https://github.com/Traveler03/CA/blob/bda6641d818db682ba95e7dee024b04f7c864aa2/paper/latex/acl_latex.tex#L503) |
| L06 | 随源数据增加的记忆增长曲线与类别分布 | [增长实验](https://github.com/Traveler03/CA/blob/bda6641d818db682ba95e7dee024b04f7c864aa2/paper/latex/acl_latex.tex#L561)、[库统计](https://github.com/Traveler03/CA/blob/bda6641d818db682ba95e7dee024b04f7c864aa2/paper/latex/acl_latex.tex#L1415) |
| L07 | Raw Candidate、Curated、Minimal Update 的构建流程消融 | [构建消融](https://github.com/Traveler03/CA/blob/bda6641d818db682ba95e7dee024b04f7c864aa2/paper/latex/acl_latex.tex#L1496) |
| L08 | 旧版本 Top-1/2/3 敏感性 | [旧稿 Top-k](https://github.com/Traveler03/CA/blob/bda6641d818db682ba95e7dee024b04f7c864aa2/paper/latex/acl_latex.tex#L1519) |
| L09 | 分阶段输入输出 token 与相对推理成本 | [旧稿成本表](https://github.com/Traveler03/CA/blob/bda6641d818db682ba95e7dee024b04f7c864aa2/paper/latex/acl_latex.tex#L1548) |
| L10 | 英语与中文概念锚点 | [锚点语言分析](https://github.com/Traveler03/CA/blob/bda6641d818db682ba95e7dee024b04f7c864aa2/paper/latex/acl_latex.tex#L1574) |
| L11 | 检索阈值敏感性及开发集阈值选择 | [阈值敏感性](https://github.com/Traveler03/CA/blob/bda6641d818db682ba95e7dee024b04f7c864aa2/paper/latex/acl_latex.tex#L1602)、[开发集选择](https://github.com/Traveler03/CA/blob/bda6641d818db682ba95e7dee024b04f7c864aa2/paper/latex/acl_latex.tex#L1762) |
| L12 | 高资源英语评测、记忆注入与未注入样本拆分 | [英语结果](https://github.com/Traveler03/CA/blob/bda6641d818db682ba95e7dee024b04f7c864aa2/paper/latex/acl_latex.tex#L1630)、[注入拆分](https://github.com/Traveler03/CA/blob/bda6641d818db682ba95e7dee024b04f7c864aa2/paper/latex/acl_latex.tex#L1641) |
| L13 | 旧版跨模型泛化，包括不同于当前主表的模型设置 | [旧版跨模型实验](https://github.com/Traveler03/CA/blob/bda6641d818db682ba95e7dee024b04f7c864aa2/paper/latex/acl_latex.tex#L1663) |
| L14 | 配对 bootstrap、置信区间和显著性 | [统计可靠性](https://github.com/Traveler03/CA/blob/bda6641d818db682ba95e7dee024b04f7c864aa2/paper/latex/acl_latex.tex#L1702) |
| L15 | Raw QA、Raw Trajectory、Raw QA + Concept Key 与完整记忆 | [原始记忆对照](https://github.com/Traveler03/CA/blob/bda6641d818db682ba95e7dee024b04f7c864aa2/paper/latex/acl_latex.tex#L1733) |
| L16 | 旧版数据过滤与记忆安全检查 | [旧稿审计表](https://github.com/Traveler03/CA/blob/bda6641d818db682ba95e7dee024b04f7c864aa2/paper/latex/acl_latex.tex#L1789) |
| L17 | 人工失败分析、答案翻转、成功与失败案例、描述与用法对照案例 | [人工分析](https://github.com/Traveler03/CA/blob/bda6641d818db682ba95e7dee024b04f7c864aa2/paper/latex/acl_latex.tex#L1833)、[翻转分析](https://github.com/Traveler03/CA/blob/bda6641d818db682ba95e7dee024b04f7c864aa2/paper/latex/acl_latex.tex#L1860)、[案例](https://github.com/Traveler03/CA/blob/bda6641d818db682ba95e7dee024b04f7c864aa2/paper/latex/acl_latex.tex#L1888) |

特别说明：**“仅定义 vs 完整用法卡片”不是从未在旧稿中出现过；它属于 L04。当前缺的是与现行实验版本对得上的可核验结果，而不是表格标题。** 同理，旧稿 Top-k 与当前 A04-A06 不是同一组实验，旧稿成本表也不能替代第 5 节当前运行的成本证据。

## 9. 当前版本尚待补充或定位证据的项目

| 项目 | 当前状态 | 已有结果能否替代 |
| --- | --- | --- |
| 仅定义 vs 完整用法卡片 | 旧稿有 L04；当前冻结模型与当前库下未定位对应完整结果 | tCRAG/CORAL 系统对比不能单独隔离用法字段贡献 |
| 去掉规则、适用条件、易错点的单字段消融 | 当前归档中未定位实测结果 | Top-k、随机卡片和前缀普查均不能替代 |
| 同源原始文段、普通摘要与完整卡片的表示对照 | 当前稿中是实验方案，尚无可核验实测表 | 仅共享 Wikipedia 总库不等于同源证据匹配 |
| 当前构建验证与字段定向证据收集消融 | 未定位当前版本实测结果 | T03 的冗余阈值推导不能证明整个验证流程的作用 |
| 当前概念查询改写、学科过滤消融 | 未定位完整结果；旧稿相关查询表为 L05 | 提示指令语言历史对照不是同一干预 |
| 当前完整人工质量审计、来源蕴含与适用性标注 | 尚未找到可核验标注结果 | clean 标识、字段数量和 T04 样本量都不是人工正确率 |
| 固定源题依赖的配对不确定性、独立解析复核、逐题因果排查 | 当前汇总尚未提供这些验证 | 有效率和 T01 算术边界不能替代 |
| 对齐设置的全方法 token、延迟、GPU 成本和构建成本 | 只有部分调用、回退和 tCRAG 延迟统计 | 不能据此宣称 CA 比所有基线更省成本 |
| 元数据规范化后的重建索引与重新评测 | 当前记录显示仍需重建索引，未确认重新评测 | 不能将旧分数改称新版本实测 |
| 其他讨论过的方法与训练方案 | 本次快照未定位 D-RAG、DKM-RAG、QTT-RAG、MultiRAG 或当前 SFT/GRPO 的完整可核验结果 | 对话中讨论、已有代码或旧稿文字都不等于已完成实验 |

## 10. 论文中可以怎样引用这份清单

| 论文用途 | 应引用的内容 | 不应混入的内容 |
| --- | --- | --- |
| 主实验 | M01-M11，明确模型、基准与实现适配 | 缺失的 Qwen CORAL 分数、旧稿训练版本数字 |
| 运行阶段消融 | A01-A06 和同一份 M03 参考结果 | 把去检查叫作仅排序消融，或把重复展示 M03 当新实验 |
| 泛化与语言差异 | 第 3.2 节及单列的英语补充结果 | 未验证的“所有模型都提升”“语言无关”结论 |
| 资源分析 | 第 6 节的结构与字段可见性 | 将字段存在等同于内容正确、检索相关或下游有用 |
| 系统运行分析 | 第 5.3-5.5 节的已记录统计 | 未对齐的效率排名或不存在的 token 成本 |
| 条件分析附录 | T01-T06，连同假设和适用范围 | 将推导区间、计划样本量写成已完成实验 |
| 后续补实验 | 第 8 节先找旧日志，第 9 节再确定重跑范围 | 不核对旧模型与知识库版本就直接复用旧表 |

总结：**当前已归档的低资源主实验与运行消融共有 17 个不重复配置；另外有英语补充评测、部分历史对照与运行记录、全库分析及理论推导。内容表示、构建流程与人工质量等旧稿条目仍需追溯，不能算入当前版本已完成的实测证据。**

## 11. 来源索引与本次核验范围

本清单的主要数值来自 7 份远程归档 JSON：零样本、tCRAG、Qwen CA 合并汇总、双模型合并汇总、消融合并汇总及两个 Qwen CA 运行分片。本次逐一检查了本地缓存内容的 Git blob SHA-1 与固定提交目录树一致，再按整数正确题数重算表格。

这种检查确认了所读内容对应固定提交，并验证汇总层数值一致性；它不等于重新运行模型、认证所有历史运行配置一致，或逐题审查全部预测。

- 当前稿件保留在本地 `paper/latex/wikipag_acl_zh_draft.tex` 与 `paper/latex/wikipag_acl_en.tex`，未随本次清单发布。
- 版本与证据补充记录保留在本地 `paper/source-ledger.md`，未随本次清单发布。本页不为未发布文件提供失效的在线链接。
- 旧稿待核验入口：[旧版英文稿](https://github.com/Traveler03/CA/blob/bda6641d818db682ba95e7dee024b04f7c864aa2/paper/latex/acl_latex.tex)；此文件未在本次整理中修改。
- 远程可读报告：[Qwen tCRAG][trag-report]、[Qwen CA][ca-report]、[Qwen 消融][ablation-report]、[双模型主实验][two-report]。

[repo]: https://github.com/Traveler03/CA/tree/bda6641d818db682ba95e7dee024b04f7c864aa2
[bank]: https://github.com/Traveler03/CA/blob/bda6641d818db682ba95e7dee024b04f7c864aa2/artifacts/clean_cards/concept_card_qwen3_1024/concept_card_bank.jsonl.gz
[builder]: https://github.com/Traveler03/CA/blob/bda6641d818db682ba95e7dee024b04f7c864aa2/scripts/run_subject_concept_smoke.py
[zero-json]: https://github.com/Traveler03/CA/blob/bda6641d818db682ba95e7dee024b04f7c864aa2/runs/smoke_001/full_qwen3_8b_localized_zero_shot_2gpu/summary.json
[trag-json]: https://github.com/Traveler03/CA/blob/bda6641d818db682ba95e7dee024b04f7c864aa2/runs/smoke_001/full_tcrag_qwen3_8b_local_2gpu/summary.json
[trag-report]: https://github.com/Traveler03/CA/blob/bda6641d818db682ba95e7dee024b04f7c864aa2/runs/smoke_001/full_tcrag_qwen3_8b_local_2gpu/report.md
[ca-json]: https://github.com/Traveler03/CA/blob/bda6641d818db682ba95e7dee024b04f7c864aa2/artifacts/evaluation/full_global_mmlu_mmlu_prox_ca_eval/final_ca_vs_tcrag_summary.json
[ca-report]: https://github.com/Traveler03/CA/blob/bda6641d818db682ba95e7dee024b04f7c864aa2/artifacts/evaluation/full_global_mmlu_mmlu_prox_ca_eval/final_ca_vs_tcrag_report.md
[two-json]: https://github.com/Traveler03/CA/blob/bda6641d818db682ba95e7dee024b04f7c864aa2/artifacts/evaluation/two_model_full_low_resource_20260821/two_model_summary/metrics.json
[two-report]: https://github.com/Traveler03/CA/blob/bda6641d818db682ba95e7dee024b04f7c864aa2/artifacts/evaluation/two_model_full_low_resource_20260821/two_model_summary/report.md
[ablation-report]: https://github.com/Traveler03/CA/blob/bda6641d818db682ba95e7dee024b04f7c864aa2/artifacts/evaluation/full_ca_ablation_qwen3_8b/ablation_report.md
[ablation-json]: https://github.com/Traveler03/CA/blob/bda6641d818db682ba95e7dee024b04f7c864aa2/artifacts/evaluation/full_ca_ablation_qwen3_8b/ablation_summary.json
[ca-run-a]: https://github.com/Traveler03/CA/blob/bda6641d818db682ba95e7dee024b04f7c864aa2/artifacts/evaluation/full_global_mmlu_mmlu_prox_ca_eval/ca_eval_full_bank_qwen3_8b_all_test/summary.json
[ca-run-b]: https://github.com/Traveler03/CA/blob/bda6641d818db682ba95e7dee024b04f7c864aa2/artifacts/evaluation/full_global_mmlu_mmlu_prox_ca_eval/ca_eval_full_bank_qwen3_8b_mmlu_prox_low_resource_all_test/summary.json
