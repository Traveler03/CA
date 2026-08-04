# Wikipag Concept-Usage Bank

这是只保留「从 Wikipag/Wikipedia 英文材料构建 concept 与 usage card」的干净代码版本。

不包含：

- evaluation-specific 节点生成；
- 评测流水线；
- 生成数据、缓存、日志、向量索引或下载好的 Wiki 资产。

## 主流程

```text
subject list
  -> seed queries
  -> Wikipag passage retrieval
  -> candidate concept extraction
  -> concept filtering + grounding
  -> same-concept merge
  -> canonical concept registry
  -> usage retrieval jobs
  -> usage-card extraction
  -> evidence-claim verification
  -> one CA-Mem concept card per subject+concept
```

## 关键脚本

- `scripts/wiki_faiss/serve_sherlock_wiki.py`：启动本地 Wiki FAISS 检索服务。
- `scripts/run_subject_concept_smoke.py`：单 subject 的 concept/card 构建。
- `scripts/run_usage_bank_batch.py`：多 subject 顺序批处理。
- `scripts/combine_clean_shards.py`：合并多个 wiki-clean shard。
- `scripts/export_concept_cards_to_ca_mem_bank.py`：把 `usage_cards.jsonl` 聚合成一 concept 一张 CA-Mem card。
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
  --max-passages 24 \
  --max-usage-concepts 5 \
  --max-usage-jobs 10 \
  --concurrency 4
```

导出一 concept 一张卡：

```bash
python scripts/export_concept_cards_to_ca_mem_bank.py \
  --source-dir runs/smoke_001/high_school_microeconomics \
  --output-dir runs/smoke_001/high_school_microeconomics_ca_mem \
  --runtime-unified \
  --group-by-concept-name
```

## 主要输出

构建目录会产生：

- `concept_registry.jsonl`
- `concept_evidence.jsonl`
- `concept_relations.jsonl`
- `usage_jobs.jsonl`
- `usage_materials.jsonl`
- `usage_cards.jsonl`
- `usage_card_claims.jsonl`
- `usage_index.jsonl`
- `bank_manifest.json`
- `summary.json`

导出到 CA-Mem 后会产生：

- `bank.jsonl`
- `concept_cards.jsonl`
- `concept_card_to_usage_trace.jsonl`
- `build_index.npy`
- `bank_manifest.json`
