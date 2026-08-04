# Wikipag Concept-Usage Bank

这是只保留「从 Wikipag/Wikipedia 英文材料构建 concept 与 usage card」的干净代码版本。

不包含：

- evaluation-specific 节点生成；
- 评测流水线；
- 生成数据、缓存、日志、向量索引或下载好的 Wiki 资产。

## 主流程

```text
subject
  -> subject profile
  -> concept discovery queries
  -> Wikipag passage retrieval
  -> candidate concept extraction
  -> concept filtering + grounding
  -> same-concept merge
  -> concept registry
  -> evidence pack per concept
  -> compact slot extraction
  -> slot evidence verification
  -> one runtime card per subject+concept
  -> runtime card embedding index
```

## Card 质量门槛

一个 runtime card 只有通过以下 gate 才会进入 `runtime_cards.jsonl`：

- definition 至少 1 条，且被 evidence 支持；
- trigger 至少 2 条，说明何时使用该 concept；
- rule 至少 2 条，且至少 2 条是可执行规则、公式、分类标准或判断步骤；
- pitfall 至少 1 条，必须来自 assumption / exception / limitation / confusable boundary 等 evidence；
- accepted claim 比例不低于 0.75；
- `quality_score >= min_card_quality_score`，默认 0.72；
- 每条保留 slot 都必须能追溯到 evidence source。
- trigger/rule/pitfall 必须是当前 subject 与 concept 的中心用法；过窄的研究、实验、医学、历史或高级边角材料会被拒绝或不生成。

未通过的 card 会进入：

- `runtime_card_quality.jsonl`
- `rejected_items.jsonl`

## 关键脚本

- `scripts/wiki_faiss/serve_sherlock_wiki.py`：启动本地 Wiki FAISS 检索服务。
- `scripts/run_subject_concept_smoke.py`：单 subject 的 compact runtime card 构建。
- `scripts/run_usage_bank_batch.py`：多 subject 顺序批处理。
- `scripts/combine_clean_shards.py`：合并多个 wiki-clean shard。
- `scripts/query_runtime_cards.py`：按题目/查询文本检索 top-k runtime cards。
- `scripts/export_concept_cards_to_ca_mem_bank.py`：把 `runtime_cards.jsonl` 转成 CA-Mem bank。
- `scripts/rebuild_ca_mem_index_sentence_transformers.py`：用本地 SentenceTransformer/Qwen embedding 重建运行时索引。

## 最小运行示例

先启动本地检索服务：

```bash
python scripts/wiki_faiss/serve_sherlock_wiki.py \
  --host 127.0.0.1 \
  --port 8897 \
  --index ivfpq.faiss
```

跑一个 subject：

```bash
python scripts/run_subject_concept_smoke.py \
  --subject high_school_microeconomics \
  --category social_sciences \
  --output-dir runs/smoke_001/high_school_microeconomics \
  --target-active-concepts 20 \
  --max-passages 32 \
  --max-card-concepts 10 \
  --min-card-quality-score 0.72 \
  --card-index-backend wikipag \
  --concurrency 4
```

检索 runtime cards：

```bash
python scripts/query_runtime_cards.py \
  "price change quantity demanded elasticity" \
  --index-dir runs/smoke_001/high_school_microeconomics \
  --subject high_school_microeconomics \
  --min-score 0.35 \
  --top-k 5
```

合并多个 shard 时，默认也会用 Wikipag service 的 `Qwen3-Embedding-4B` 重建 combined `runtime_card_index`；如需离线调试可显式传 `--card-index-backend hash`。

导出到 CA-Mem：

```bash
python scripts/export_concept_cards_to_ca_mem_bank.py \
  --source-dir runs/smoke_001/high_school_microeconomics \
  --output-dir runs/smoke_001/high_school_microeconomics_ca_mem \
  --runtime-unified
```

## 主要输出

构建目录会产生：

- `concept_registry.jsonl`
- `concept_evidence.jsonl`
- `evidence_packs.jsonl`
- `runtime_cards.raw.jsonl`
- `runtime_card_claims.jsonl`
- `runtime_card_quality.jsonl`
- `runtime_cards.jsonl`
- `runtime_card_index.jsonl`
- `runtime_card_index.npy`
- `runtime_card_index_meta.json`
- `bank_manifest.json`
- `summary.json`

导出到 CA-Mem 后会产生：

- `bank.jsonl`
- `concept_cards.jsonl`
- `concept_card_to_usage_trace.jsonl`
- `build_index.npy`
- `bank_manifest.json`
