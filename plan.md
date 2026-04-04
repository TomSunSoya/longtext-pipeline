# longtext-pipeline 项目方案说明书

## 1. 项目概述

`longtext-pipeline` 是一个面向超长文本的分层分析流水线工具，定位为 **Python CLI 项目**。它的核心目标不是“一次性把长文本塞给大模型”，而是把长文本处理拆解为多个层次，使模型在每一层只处理自己更擅长的尺度，从而提高整体分析质量、可控性与可审计性。

它适用于以下类型的任务：

- 超长聊天记录分析
- 长会议纪要、访谈记录分析
- 长篇项目文档、知识库文档总结
- 多文件文本合集的分层归纳
- 需要中间产物保留、支持回溯与审计的长文本处理场景

项目采用如下总体流程：

1. 将原始长文本切分为多个 part
2. 为每个 part 生成第一层 summary
3. 将多个 summary 按顺序分组，生成第二层 stage summary
4. 基于全部 stage summary 生成最终总分析
5. 可选增加 audit 流程，对输出中的幻觉、过度推断、时间线错位等进行审计

整个系统强调：

- **可分层**
- **可落盘**
- **可回溯**
- **可配置**
- **可复现**
- **模型无关**

---

## 2. 项目定位

本项目不是单纯的 prompt 集合，也不是依赖某个特定 agent 平台的 skill，而是一个 **脚本主导流程、模型负责理解与写作** 的通用型长文本分析框架。

### 核心定位

- 脚本负责：
  - 文件扫描
  - 文本切分
  - 分组
  - 输出目录管理
  - 合并与状态记录
  - 断点续跑
  - 审计流程调度

- 模型负责：
  - part-level summary
  - stage-level synthesis
  - final cross-stage analysis
  - 可选审计分析

### 设计思想

与把所有事情交给 agent 自己调度相比，这种设计更稳定，因为：

- 机械任务不消耗模型 token
- 中间文件可见，不是黑盒
- 更容易控制流程一致性
- 更适合开源和 GitHub 发布
- 更利于模型替换与成本优化

---

## 3. 使用场景

第一版主要面向以下输入类型：

- `.txt`
- `.md`

这些是最适合该流水线的输入类型，因为它们容易稳定提取为线性文本序列。

### 第一优先级适合场景

- 微信聊天导出文本
- 长 Markdown 笔记
- 访谈文本
- 日志文件
- 长篇会议纪要
- issue/工单记录导出

### 后续扩展支持

- `.jsonl`
- `.docx`
- `.pdf`（通过预处理转文本）
- 多文件目录输入

需要注意的是，项目更适合**纯文本或可稳定还原为纯文本的文档**。对结构复杂、强依赖版式的 PDF 或复杂 DOCX，必须先做预处理，否则后续分析质量会受明显影响。

---

## 4. 总体架构

项目采用典型的多阶段流水线设计：

### 第一阶段：Ingest
读取原始输入，进行文本清洗和切分，输出多个 `part_*.txt` 文件。

### 第二阶段：Summarize
逐个处理 `part_*.txt`，为每个分片生成第一层摘要 `summary_*.md`。

### 第三阶段：Stage
将多个 `summary_*.md` 按顺序分组，生成阶段级分析文件 `stage_*.md`。

### 第四阶段：Final
读取全部 `stage_*.md`，生成一份最终总分析文档。

### 第五阶段：Audit（可选）
对 summary / stage / final 输出做审计，识别幻觉、过强推断、时间线错误与证据边界问题。

### 核心思想

整个流程不是“一次总结”，而是逐层压缩与提升抽象层级：

**原始文本 → 分片摘要 → 阶段摘要 → 最终分析 → 可选审计**

这样做的核心收益在于：

- 模型更容易在局部范围内保持稳定
- 最终分析能建立在中间层之上
- 任何结论都可以向下回溯
- 可以对不同层级输出分别做质量控制

---

## 5. 仓库目录结构设计

建议项目仓库采用如下结构：

```text
longtext-pipeline/
├─ README.md
├─ LICENSE
├─ pyproject.toml
├─ .gitignore
├─ examples/
│  ├─ config.general.yaml
│  ├─ config.relationship.yaml
│  └─ sample_input/
│     ├─ input.txt
│     └─ input.md
├─ src/
│  └─ longtext_pipeline/
│     ├─ __init__.py
│     ├─ cli.py
│     ├─ config.py
│     ├─ models.py
│     ├─ errors.py
│     ├─ splitter.py
│     ├─ grouper.py
│     ├─ manifest.py
│     ├─ renderer.py
│     ├─ prompts/
│     │  ├─ summary_relationship.txt
│     │  ├─ stage_relationship.txt
│     │  ├─ final_relationship.txt
│     │  ├─ audit_relationship.txt
│     │  ├─ summary_general.txt
│     │  ├─ stage_general.txt
│     │  ├─ final_general.txt
│     │  └─ audit_general.txt
│     ├─ llm/
│     │  ├─ __init__.py
│     │  ├─ base.py
│     │  ├─ openai_compatible.py
│     │  └─ factory.py
│     ├─ pipeline/
│     │  ├─ ingest.py
│     │  ├─ summarize.py
│     │  ├─ stage.py
│     │  ├─ final.py
│     │  └─ audit.py
│     └─ utils/
│        ├─ io.py
│        ├─ hashing.py
│        ├─ token_estimator.py
│        ├─ text_clean.py
│        └─ retry.py
├─ tests/
│  ├─ test_splitter.py
│  ├─ test_grouper.py
│  ├─ test_manifest.py
│  └─ test_config.py
└─ docs/
   ├─ architecture.md
   ├─ prompt-design.md
   └─ roadmap.md

---

## 6. 核心特性（同步自 README.md）

- **分层处理**: 4阶段流水线（Ingest → Summarize → Stage → Final）
- **可恢复**: 基于 SHA-256 校验的断点续跑机制
- **双模式**: 通用分析模式 + 关系分析模式（实验性）
- **模型无关**: 支持 OpenAI 兼容 API（OpenAI、OpenRouter、Ollama 等）
- **审计追踪**: 中间产物保留，支持回溯验证

---

## 7. 快速开始（同步自 README.md）

### 安装

```bash
# 克隆项目
cd longtext-pipeline

# 可编辑模式安装
pip install -e .

# 验证安装
longtext --version
```

### 运行示例

```bash
# 设置 API Key
export OPENAI_API_KEY="sk-your-api-key-here"

# 运行流水线
longtext run sample_input.txt --config examples/config.general.yaml

# 查看状态
longtext status sample_input.txt

# 查看结果
cat output/final_analysis.md
```

---

## 8. Git 提交历史

### 2026-04-04 - 首次提交（初始化）

**提交信息**: `Initial commit: longtext-pipeline project initialization`

**变更内容**:
- 初始化 Python CLI 项目结构
- 添加核心模块框架（cli.py, config.py, models.py 等）
- 添加流水线模块（ingest, summarize, stage, final, audit）
- 添加 LLM 兼层（OpenAI-compatible API 支持）
- 添加测试框架和配置示例
- 添加完整文档（README.md, plan.md, docs/）

**远端仓库**: GitHub（待创建）

---

## 9. 下一步计划

1. ✅ 项目初始化完成
2. ⏳ 创建 GitHub 远端仓库
3. ⏳ 推送代码到远端
4. 待定：补充单元测试覆盖率
5. 待定：完善 prompt 设计文档
6. 待定：添加 CI/CD 配置