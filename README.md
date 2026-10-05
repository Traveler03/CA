# Wikipag Concept-Usage Bank

使用 **Qwen3.5-9B**，从英文 Wikipag 文本片段合成应用问题、求解并自检，再提炼可复用的英文概念用法卡片。

仓库只维护这一套构建流程，唯一入口为 [build_wikipag_application_cards.py](scripts/build_wikipag_application_cards.py)。

现有答题卡库固定为 **56,579 张卡片、57 个学科**。卡库内容和索引保持冻结，已有实验结果继续对应各自实际使用的卡库快照、模型和答题配置。

主实验、历史基线和 Qwen3.5-9B 消融汇总见 [实验结果](docs/RESULTS.md)。

下述 pipeline 是后续构建所维护的实现。更新构建代码不触发现有卡库重建或实验重跑；新运行仅写入独立目录。现有卡库的构建历史与新 pipeline 的验证记录分别保留。

```text
英文 Wikipag passages
  → 片段筛选与同文章材料补充
  → Qwen3.5-9B 生成英文应用问题
  → Qwen3.5-9B 求解
  → Qwen3.5-9B 来源约束自检
  → Qwen3.5-9B 提炼五字段 card
  → Qwen3.5-9B 审核 card
  → 未通过的最多修订一次，再次审核
  → 导出通过审核的卡片，合并完全重复项
```

自检不通过的组不进入提炼；卡片复核不通过不导出。代码在每阶段检查 JSON、来源身份及逐字引文，保存响应缓存、来源记录和断点。所有模型阶段共用一个 Qwen3.5-9B 服务。

## Card 格式

```json
{
  "concept_name": "Concept name",
  "definition": "Definition of the concept.",
  "trigger": ["Applicability conditions."],
  "decision": ["Reusable procedure, rule, or formula."],
  "pitfall": ["A source-supported limitation or mistake."]
}
```

`trigger / decision / pitfall` 为使用三元组；`pitfall` 可为空。卡片保留通用规则、多步操作和必要条件，具体问题、答案、来源引文、推导依据及审核记录位于旁文件。

## 运行

```bash
pip install -e '.[test]'

python scripts/build_wikipag_application_cards.py \
  --passage-groups runs/wiki_sources/passage_groups.jsonl \
  --validate-only

python scripts/build_wikipag_application_cards.py \
  --passage-groups runs/wiki_sources/passage_groups.jsonl \
  --output-dir runs/wikipag_qwen35_cards \
  --concurrency 8
```

默认模型 ID 为 `qwen3.5-9b`，默认使用 Chat Completions。服务地址和认证可沿用本地 provider，也可通过 `--base-url` 和 `--api-key-env` 指定；`--model` 用于指定同一 Qwen3.5-9B 部署的服务别名。模型选择不继承开发工具的模型配置。

[详细输入与运行说明](docs/CONSTRUCTION.md) 包含来源补检索参数、输出文件、恢复方式和审核规则。

## 代码与检查

| 路径 | 用途 |
|---|---|
| `scripts/build_wikipag_application_cards.py` | 唯一构建入口 |
| `src/construction/` | 来源准备、模型调用、构建与导出 |
| `prompts/wikipag_application/` | 造题、求解、自检、提炼、审核及修订提示词 |
| `src/clients/`、`src/runtime/`、`src/utils/` | 缓存、凭据解析和通用辅助组件 |
| `tests/` | 当前流程的离线检查 |
| `runs/` | 本地输入、生成产物、缓存与运行记录，Git 忽略 |

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
```

原始 Wikipag 语料和本地索引只读。生成内容仅写入独立 run，不写入生产数据库；模型审核用于筛选，逐字引用通过不等于独立证明语义正确。

旧流程的归档位置和恢复清单见 [仓库整理记录](docs/WORKSPACE.md)。
