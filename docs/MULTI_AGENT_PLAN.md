# 多 Agent 并行改造方案

## 一、当前瓶颈分析

| 阶段 | 当前行为 | LLM 调用次数 | 可并行度 |
|------|---------|-------------|---------|
| Ingest | 纯本地 I/O，无 LLM | 0 | 不需要 |
| Summarize | `for part in parts` 串行循环 | N（= part 数量） | **完全独立，可全并行** |
| Stage Synthesis | `for group in groups` 串行循环 | M（= group 数量） | **完全独立，可全并行** |
| Final | 单次调用 | 1 | 不可并行 |

config 里已经定义了 `batch_size: 4` 和 `max_workers: 4`，但代码中**完全未使用**。

---

## 二、分层改造（由浅到深 3 层）

### Layer 1：异步 LLM 层（基础设施）

当前 `OpenAICompatibleClient` 用 `httpx.Client`（同步），每次调用都阻塞等待。

**改动点：**

- `LLMClient` 基类新增 `acomplete()` 和 `acomplete_json()` 异步抽象方法
- `OpenAICompatibleClient` 内部用 `httpx.AsyncClient` 实现异步版本，保留同步方法做向后兼容
- `retry_llm_call` 装饰器新增异步版本 `retry_llm_call_async`，用 `asyncio.sleep` 替代 `time.sleep`

```
LLMClient (base.py)
├── complete()           # 保留，同步
├── complete_json()      # 保留，同步
├── async acomplete()        # 新增
└── async acomplete_json()   # 新增
```

### Layer 2：Stage 内并行（Fan-out）

这是**收益最大**的一层。Summarize 和 Stage Synthesis 的工作项完全独立。

**Summarize 阶段改造：**

```
当前：  part_0 -> part_1 -> part_2 -> ... -> part_N  (串行)

改为：  +-- part_0 --+
        +-- part_1 --+
        +-- part_2 --+  <- asyncio.Semaphore(max_workers) 控制并发
        +-- ...     --+
        +-- part_N --+
        asyncio.gather() 收集结果
```

- 用 `asyncio.Semaphore(max_workers)` 做并发控制（读现有的 `pipeline.max_workers` 配置）
- 每个 task 独立调用 `acomplete()`，独立 retry
- 结果收集后按 `part_index` 排序，保证顺序一致
- 失败的 task 记入 `ErrorAggregator`，其余继续（现有 Continue-with-Partial 策略不变）

**Stage Synthesis 同理**，fan-out 所有 group。

### Layer 3：Stage 间流水线化（Streaming Pipeline）

当前：必须等所有 summary 完成才开始 stage synthesis。

**改为 Producer-Consumer 流式：**

```
Summarize (producer)          Stage Synthesis (consumer)
    |                              |
    +-- summary_0 done --+         |
    +-- summary_1 done --+         |
    +-- summary_2 done --+         |
    +-- summary_3 done --+         |
    +-- summary_4 done --+--> 凑够 group_size=5 --> 立即启动 group_0 合成
    +-- summary_5 done --+         |
    +-- ...              --+       |
    +-- summary_9 done --+--> 凑够 group_size=5 --> 立即启动 group_1 合成
    ...
```

- 用 `asyncio.Queue` 连接两个阶段
- Summarize 完成一个就往 queue 里推
- Stage Synthesis 的 collector 从 queue 取，攒够 `group_size` 个就启动一个合成 task
- Final 阶段仍需等所有 stage synthesis 完成（这是数据依赖，无法绕过）

**效果**：假如有 20 个 part，`group_size=5`，现在不需要等 20 个 summary 全完成；前 5 个 summary 一完成，stage_0 就可以开始了。总延迟显著缩短。

---

## 三、真正的多 Agent（异构 Agent 角色）

以上是"并行化"，如果要做真正的**多 Agent 协作**，核心思路是：不同阶段用不同模型/角色。

### 3.1 异构模型配置

```yaml
agents:
  summarizer:
    model: "deepseek-chat"        # 便宜快速模型
    temperature: 0.3
    system_prompt: "你是一个精确的文本摘要专家..."

  synthesizer:
    model: "deepseek-chat"
    temperature: 0.5
    system_prompt: "你是一个擅长归纳整合的分析师..."

  analyst:
    model: "claude-sonnet-4-6"   # 贵但强的模型，只在 final 用
    temperature: 0.7
    system_prompt: "你是一个资深分析师..."

  auditor:
    model: "deepseek-chat"
    temperature: 0.1              # 低温度，审核更严谨
    system_prompt: "你是一个事实核查专家..."
```

每个 Agent 独立实例化 LLM client，可以指向不同 provider / model / temperature。当前的 `get_llm_client(model_config)` 已经支持这个能力，只需要让每个 stage 从 `agents.{role}` 取自己的配置而不是共用顶层 `model`。

### 3.2 多视角并行 Agent

```
同一份 stage summaries
        |
        +--> Agent A: 主题/趋势分析
        +--> Agent B: 实体/关系提取
        +--> Agent C: 情感/态度分析
        +--> Agent D: 时间线梳理
        |
        +--> Meta Agent: 整合四个视角 -> final_analysis
```

在 Final 阶段前插入一个 **parallel specialist** 层，多个 agent 并行处理同一份输入，各自输出专项分析，最后由一个 meta agent 做整合。

---

## 四、需要配套改动的模块

| 模块 | 改动 |
|------|------|
| `llm/base.py` | 新增 `acomplete` / `acomplete_json` 抽象方法 |
| `llm/openai_compatible.py` | 新增 `httpx.AsyncClient` 实现 + 共享连接池 |
| `utils/retry.py` | 新增 `retry_llm_call_async`，用 `asyncio.sleep` |
| `pipeline/summarize.py` | for 循环 -> `asyncio.gather` + `Semaphore` |
| `pipeline/stage_synthesis.py` | 同上 |
| `pipeline/orchestrator.py` | 顶层改为 `async def run()`，编排 stage 间的 queue 流水线 |
| `cli.py` | 入口加 `asyncio.run()` |
| `manifest.py` | 写入操作加锁（`asyncio.Lock`），多 task 并发写 manifest 时防竞争 |
| `config.yaml` | 新增 `agents` 配置段；`max_workers` / `batch_size` 生效 |
| `errors/continuation.py` | `ErrorAggregator` 加线程安全（或用 async-safe 收集） |

---

## 五、推荐实施顺序

```
Phase 1  --  异步 LLM 层 + Summarize 并行
             （改动最小，收益最大，20 个 part 从串行 20 次变为并发 4-8 次）

Phase 2  --  Stage Synthesis 并行 + Stage 间流水线
             （进一步压缩总延迟）

Phase 3  --  多 Agent 角色配置
             （不同阶段用不同模型/参数，性价比优化）

Phase 4  --  多视角并行 Agent
             （Final 前的 specialist 并行层，提升分析深度）
```

### 预期收益

**Phase 1 单独就能把 Summarize 阶段的耗时从 `N * avg_latency` 降到 `ceil(N/max_workers) * avg_latency`**。

对于 20 个 part、`max_workers=4` 的场景，约 **5x 加速**。

### 风险与注意事项

1. **Rate Limit**：并发过高会触发 API 429，需要 Semaphore 限流 + 现有的 retry_llm_call 指数退避配合
2. **Manifest 竞争写入**：多个并发 task 同时完成时会并发写 manifest，需要 `asyncio.Lock` 保护
3. **内存压力**：大量 part 同时在内存中持有 prompt + response，需要关注峰值内存
4. **向后兼容**：保留同步 `complete()` 方法，CLI 入口用 `asyncio.run()` 包装，不影响现有测试
5. **错误可观测性**：并发场景下错误日志会交错，建议给每条日志加 `[part_XX]` / `[group_XX]` 前缀
