# Qwen3.5-9B 英文 Wikipag 应用记录合成

唯一构建入口为 [build_wikipag_application_cards.py](../scripts/build_wikipag_application_cards.py)，当前协议为 `wikipag-application-cards-v3`。固定使用英文 Wikipag passages，先准备来源，再由 **Qwen3.5-9B** 造题、求解、自检、提炼和审核；必要时一次修订再复核。正文保持五字段接口。

本说明适用于后续构建运行。现有 56,579 张卡片及其索引保持冻结，已有实验保留原卡库快照和模型配置的对应关系。新运行的产物与验证记录写入独立目录，不替换现有卡库。

旧构建路径及其专用依赖已从活动源码移出，恢复方式见[仓库整理记录](WORKSPACE.md)。

## 输出接口保持不变

`cards.jsonl` 每行仍只有 `concept_name`、`definition`、`trigger`、`decision`、`pitfall`。前两项为字符串，后三项为字符串列表。Conditions 对应 `trigger`，Rules 对应 `decision`，Pitfalls 对应 `pitfall`；有依据的内容才写入，`pitfall` 可以为空。

`bank.jsonl` 沿用 `memory_id`、`subject`、`concept`、`description` 的运行格式，描述包含原有的 Description / Trigger / Decision / Pitfall 段落。来源和合成记录放在逐行对应的 `bank_metadata.jsonl` 中，模型看到的卡片正文不增加标签或题目答案。

## 输入

提供 UTF-8 JSONL，每行是一组主题相关的英文 Wikipag passages。原始语料和检索索引只读，分组后的输入文件放在独立 run 中。组内 1–5 个片段，每段至多 12,000 字符，每组至多 30,000 字符；过长输入需显式拆组，不静默截断。

以下仅为字段示意；ID 和 text 必须来自实际 Wikipag 记录：

```json
{
  "group_id": "unique-group-id",
  "subject": "subject-name",
  "corpus": "English Wikipag",
  "language": "en",
  "passages": [
    {
      "source_id": "original-source-id",
      "passage_id": "original-passage-id",
      "title": "Original title",
      "text": "Exact English passage text from Wikipag."
    }
  ]
}
```

写入 JSONL 时每个对象独占一行。不同组可以复用同一片段，同一 `(source_id, passage_id)` 必须对应相同文本。`corpus` 与 `language` 是输入来源声明，程序检查其格式及一致性，不会凭这些字符串独立认证语料来源。

## 执行

先检查输入，整个命令不读取模型凭据、不调用模型：

```bash
python scripts/build_wikipag_application_cards.py \
  --passage-groups runs/wiki_sources/passage_groups.jsonl \
  --validate-only
```

所有生成、审核和修订阶段共用同一 Qwen3.5-9B 服务。默认模型 ID 为 `qwen3.5-9b`；如果部署使用不同服务别名，通过 `--model` 指定：

```bash
python scripts/build_wikipag_application_cards.py \
  --passage-groups runs/wiki_sources/passage_groups.jsonl \
  --output-dir runs/wikipag_qwen35_cards \
  --concurrency 16
```

默认筛选输入片段，不额外检索。配置以下三个参数后，会只读访问本地原始语料和偏移索引，核对 seed 的 ID／文本，并在同文章内补充片段：

```bash
python scripts/build_wikipag_application_cards.py \
  --passage-groups runs/wiki_sources/passage_groups.jsonl \
  --output-dir runs/wikipag_qwen35_cards_with_sources \
  --offsets-db /path/to/wikipedia-en-faiss/offsets.sqlite \
  --corpus-root /path/to/wikipedia-en-passages \
  --source-id-prefix wikipag-en-2026-07-01: \
  --concurrency 16 --max-completion-tokens 6144
```

`source_id_prefix` 必须对应输入的真实来源标识。该后端要求 `article#passage` 形式的数字 ID；不会加载 FAISS 或嵌入模型。默认每篇文章最多取 64 个候选（按 ID 排序，是否截断写入记录），保留通过筛选的 seed 后，按通用操作词、条件与边界词的覆盖数量选择补充材料。每组仍至多 5 段、30,000 字符。可通过 `--max-source-candidates` 和 `--max-passages` 调整预算。词法排序只是候选选择，不能认定来源支持；当前也不支持跨文章语义检索。

筛选拦截过短片段、编码损坏、消歧义页面，以及明确缺失公式操作数的文本，例如 `time requirement of ,`。对受损片段整段弃用，不补写公式、不修改原文；缺少发音转写的空括号本身不触发公式拦截。规则只能识别部分损坏模式，后续模型仍需检查来源完整性。没有可用片段时，该组在调用模型前拒绝。

构建入口复用现有 provider 的地址和认证，但模型默认始终为 `qwen3.5-9b`，不会被开发工具中的模型设置或 `CHAT_MODEL` 覆盖。可通过 `--base-url` 指定 Qwen 服务地址，通过 `--api-key-env` 指定存放密钥的环境变量；不要把密钥作为命令参数传入。

默认 `--endpoint chat/completions`，发送 `max_tokens` 和 `reasoning_effort=none`，与当前 Qwen 服务的非思考模式参数一致；求解阶段仍显式生成解释和操作步骤。需要服务支持的其他思考设置时可用 `--reasoning-effort` 调整。服务明确支持 Responses 时可使用 `--endpoint responses`，发送 `max_output_tokens`。客户端只解析最终回答，不把独立 reasoning_content 或内联 `<think>…</think>` 当作 JSON；因长度上限中断的输出不通过校验。

阶段文件记录服务实际返回的模型名称、用量和耗时；`summary.json` 中的调用数表示本次执行新增调用，恢复执行可能为 0。默认输出预算为 6,144 tokens，可用 `--max-completion-tokens` 调整；实际是否足够取决于服务的思考输出和卡片长度。

并发范围为 1–32。同一命令重启会校验并复用已完成阶段，允许调整并发；输入、模型、提示词、实现、检索配置或来源文件的大小／修改时间变更需新建 run。输出目录加写锁，防止两个进程混写。模型请求与响应保存在该 run 的 `cache/model/`、`model_raw/`，检索候选及来源原始行哈希保存在 `cache/retrieval/`，阶段检查点原子写入 `stages/`。哈希用于检测文件不一致，不是语料语义认证。

## 阶段与筛选

1. [造题](../prompts/wikipag_application/application.txt)：构造一道应用问题，显式列出假设情境和数值，引用所需的来源知识。
2. [求解](../prompts/wikipag_application/solution.txt)：给出答案和简短、可检查的解答说明，保留引用。
3. [自检](../prompts/wikipag_application/self_check.txt)：检查知识应用、来源充分性、假设、答案和英文要求；任何条件不满足或存在 issue 就结束该组，不进入提炼。
4. [提炼](../prompts/wikipag_application/extraction.txt)：优先输出一张聚焦应用流程的五字段卡片，确需多个不同概念时至多三张。去除具体情境答案与数值代入，保留适用条件及可复用的多步操作。每个正文标量及列表项都需在旁文件中有来源引用，并标明 `support_type=direct/derived`、简短 `rationale`。允许从有引文的前提做基本代数、算术或逻辑推导；情境假设须转为显式适用条件，不能凭模型记忆补领域事实。零卡片输出记为明确拒绝。
5. [card 审核](../prompts/wikipag_application/card_review.txt)：检查来源是否支持内容、必要条件、是否遗留题目实例、是否能指导操作、是否保留有关概念的解题步骤以及英文要求。审核必须覆盖每个候选且仅一次；任何 false、issue 或 missing step 都拒绝该卡。
6. [一次修订](../prompts/wikipag_application/card_repair.txt)与复核：默认只修订被拒绝候选，支持无法修复时返回 null；不增添或合并候选。随后在不提供首次审核结论的上下文中重新审核修订后的候选集。原稿、理由与修订均保留；复核失败不再循环修订。`--max-card-repairs 0` 可关闭修订，直接导出首次通过审核的候选。

正常组为 5 次模型调用，有修订和复核时最多 7 个阶段调用，另有传输／格式重试。自检拒绝后不继续；一个候选失败不会使同组已通过最终审核的其他卡片自动失效。请求或格式错误单独记为 error，不当作质量拒绝。正文不加入任何审核标签，所有筛选信息都存于旁文件。

每阶段先校验严格 JSON schema，再校验引用的 source ID、passage ID 和逐字引文。卡片审核增加语义判断，但仍是同模型筛选，不能视为独立正确性认证。当前实现不执行模型生成的程序或自动证明计算正确。

字段引用必须完整覆盖，包括 concept_name 和 definition 两个独立条目。格式重试会提供原输出及具体缺失／重复的字段索引、错误引文，要求修正对应项；校验标准不会因重试而放宽。接口失败的原始响应保留，恢复重试不会覆盖此前失败记录。

归并仅合并同学科内全部字段一致的重复卡片，忽略 Unicode NFC 和空白差异，并合并其来源记录。不同条件、规则或边界的同名概念保留为不同卡片，不进行未经核验的语义合并。

主要旁文件：

| 文件 | 内容 |
|---|---|
| `passage_groups.jsonl` / `prepared_passage_groups.jsonl` | 原始输入与实际用于合成的证据包 |
| `source_preparation/*.json` | 每个候选的过滤理由、选择结果、原始行哈希及字节偏移（本地检索模式） |
| `synthesis_records.jsonl` | 造题、求解、自检、原始卡片、审核与修订的完整过程 |
| `rejected_records.jsonl` | 最终没有可导出卡片的拒绝组及拒绝阶段 |
| `card_rejections.jsonl` | 每次失败的 card 审核，包括后来被修复的卡片 |
| `errors.jsonl` | 请求、接口或文件错误导致的未完成组 |
| `summary.json` | 产量、过滤／修订计数、本次执行模型用量和输出哈希 |

存在请求、格式或引用错误时 `summary.json` 的 `complete` 为 false，命令返回非零状态；恢复后只重新处理未完成阶段。源片段或卡片被质量门槛拒绝不算执行错误，但始终留记录。`complete=true` 只表示构建执行完毕，不能解释为所有输入通过或所有内容正确。

当前流程完成到卡片及追溯记录导出。跨文章语义检索、多应用记录联合提炼、向量构建和发布不属于这个入口。
