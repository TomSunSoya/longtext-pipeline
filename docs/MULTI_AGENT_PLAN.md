# 多 Agent 并行架构说明

本文档描述当前代码库中已经落地的多 Agent / 并行分析架构，以及已经明确放弃的旧方案。

## 1. 当前真实架构

当前实现不是“全链路 queue 流水线”，而是：

1. `LongtextPipeline.run()` 保持同步入口。
2. `SummarizeStage.run()` 和 `StageSynthesisStage.run()` 是异步 stage，但只做 stage 内并发。
3. Orchestrator 在同步边界用 `asyncio.run()` 调用异步 stage。
4. `FinalAnalysisStage.run(..., multi_perspective=True)` 才会进入多 Agent 并行分析。

也就是说，当前系统是：

`sync orchestrator -> async summarize fan-out -> async stage fan-out -> optional multi-agent final analysis`

## 2. 已落地能力

### 2.1 异步 LLM 调用

- `LLMClient` 提供同步与异步两套接口：
  - `complete()`
  - `complete_json()`
  - `acomplete()`
  - `acomplete_json()`
- `OpenAICompatibleClient` 同时支持 `httpx.Client` 和 `httpx.AsyncClient`
- 异步重试通过 `retry_llm_call_async` 实现

### 2.2 Stage 内并发

`SummarizeStage` 和 `StageSynthesisStage` 均使用：

- `asyncio.Semaphore(max_workers)` 控制并发上限
- `asyncio.gather()` 扇出执行
- Continue-with-Partial 策略处理局部失败

这两层都保留了输入顺序：

- `SummarizeStage.run()` 返回的 `Summary` 顺序与 `parts` 顺序一致
- `StageSynthesisStage.run()` 返回的 `StageSummary` 顺序与分组顺序一致

### 2.3 Final 阶段多 Agent 并行

`FinalAnalysisStage` 内部支持四个 specialist agent 并行执行：

- `topic_analyst`
- `entity_analyst`
- `sentiment_analyst`
- `timeline_analyst`

然后由 `analyst` 作为 meta-agent 做整合。

内部调用链如下：

1. `_run_multi_perspective()`
2. `_generate_specialist_analysis(...)` 并发执行四次
3. `_aggregate_with_meta_agent(...)` 聚合成功结果
4. 若成功 specialist 少于 3 个，则回退到 `_run_single_pass()`

阈值规则：

- `>= 3/4` specialist 成功：继续多视角结果整合
- `< 3/4` specialist 成功：回退到单次分析

## 3. 配置契约

当前统一使用嵌套 `model` 结构，不再使用扁平旧格式。

多视角 specialist 数量现在可以显式指定：

- CLI: `longtext run input.txt -mp --agent-count 2`
- 配置: `pipeline.specialist_count: 2`

规则：

- 取值范围为 `1..4`
- 未指定时默认使用全部 4 个 specialist
- `--agent-count` 会自动开启多视角模式，即使没有显式传 `-mp`
- specialist 选择顺序固定为：
  1. `topic_analyst`
  2. `entity_analyst`
  3. `sentiment_analyst`
  4. `timeline_analyst`
- 成功阈值为 `min(3, specialist_count)`

示例：

```yaml
model:
  provider: openai
  name: gpt-4o-mini
  api_key: ${OPENAI_API_KEY}
  timeout: 120.0

pipeline:
  max_workers: 4

stages:
  stage:
    group_size: 5

agents:
  summarizer:
    model: null
  stage_synthesizer:
    model: null
  analyst:
    model:
      provider: openai
      name: gpt-4o
  topic_analyst:
    model:
      provider: openai
      name: gpt-4o
  entity_analyst:
    model:
      provider: openai
      name: gpt-4o
  sentiment_analyst:
    model:
      provider: openai
      name: gpt-4o
  timeline_analyst:
    model:
      provider: openai
      name: gpt-4o
```

说明：

- `summarizer` / `stage_synthesizer` / `analyst` / `auditor` 是通用 agent
- `topic_analyst` / `entity_analyst` / `sentiment_analyst` / `timeline_analyst` 是 Final 阶段 specialist
- 某个 agent 的 `model: null` 表示回退到顶层 `model`

## 4. 明确废弃的旧方案

以下方案已不属于当前实现：

### 4.1 Stage 间 queue 流水线

当前没有：

- `asyncio.Queue` 串联 summarize 和 stage synthesis
- producer / consumer 模式
- `queue` 参数传入 `SummarizeStage.run()` 或 `StageSynthesisStage.run()`

这条路线已经从实现和测试中移除。

### 4.2 异步 orchestrator 公共接口

当前没有把 orchestrator 的公共 `run()` 改成异步。

保留同步 orchestrator 的原因：

- CLI 和现有调用路径更稳定
- sync / async 边界清晰
- 避免在未知调用环境中嵌套 event loop

### 4.3 旧的 FinalAnalysis 私有接口

以下旧接口已删除，不应再被测试或调用：

- `run_multi_perspective_analysis`
- `_generate_topic_analysis`
- `_generate_entity_analysis`
- `_generate_sentiment_analysis`
- `_generate_timeline_analysis`
- `_run_single_pass_analysis`

当前对应接口为：

- `_generate_specialist_analysis`
- `_run_multi_perspective`
- `_run_single_pass`
- `run(..., multi_perspective=True)`

## 5. Manifest 并发策略

`ManifestManager.save_manifest()` 当前保持同步方法。

并发保护使用：

- `threading.Lock`

没有使用 `asyncio.Lock`，原因是 manifest 写入点跨同步/异步边界存在共享，使用线程锁更直接，也避免把同步调用链错误改成 coroutine。

## 6. 设计结论

当前代码库的选择是：

- 保留同步 orchestrator
- 在 stage 内做真正有收益的并发
- 在 Final 阶段做真正有意义的多 Agent 专家分析
- 不引入 queue streaming 这类额外复杂度
- 不保留已经删除的旧接口

如果后续继续扩展，推荐优先级如下：

1. 继续优化 specialist prompt 与 meta-agent 聚合质量
2. 为不同 specialist 引入更精细的模型路由策略
3. 补充性能基准而不是恢复 queue 流水线
