# longtext-pipeline MVP Spec

## 1. Project Overview

`longtext-pipeline` is a hierarchical analysis pipeline for ultra-large texts, positioned as a **Python CLI project**. Rather than "one-time feeding of large text to large models," this project decomposes long text processing into multiple levels where models handle only what they are better suited for at each scale, thus improving overall analysis quality, controllability, and audibility.

Suitable for:
- Ultra-long chat record analysis
- Long meeting minutes, interview record analysis  
- Long project document, knowledge base document summarization
- Multi-file text collection stratified inductive analysis
- Long-text processing scenarios requiring intermediate product retention, supporting traceability and auditing

Project adopts the following general flow:

1. Split original long text into multiple parts
2. Generate first-layer summaries for each part
3. Group multiple summaries sequentially to generate second-layer stage summaries
4. Generate final overall analysis based on all stage summaries
5. Optional addition of audit processes to scrutinize hallucinations, excessive inferences, timeline mismatches, etc. in outputs

The whole system emphasizes:
- **Stratifiable**
- **Disk-creatable** 
- **Tractable**
- **Configurable**
- **Reproducible**
- **Model-Agnostic**

---

## 2. MVP Scope

### Core Features:
- Support `.txt` and `.md` input formats
- 4-stage processing pipeline: Ingest → Summarize → Stage → Final
- YAML-based configuration system (`config.general.yaml`)
- OpenAI-compatible LLM interface
- Manifest-based tracking with file hash validation
- Resume functionality with checkpoint mechanics
- Status command for monitoring progress
- Dual-mode prompts (general and relationship-focused)
- Local-first processing (no networking overhead)
- Single-file, single-provider, sequential processing strategy

### Non-Goals:
- PDF/DOCX support (future extensions)
- Web UI or dashboard interface
- Advanced caching strategies
- Concurrent/multi-threaded processing
- Native multi-LLM provider support
- Full audit implementation
- Batch processing
- Stream processing

---

## 3. Input/Output Requirements

### Supported Input Formats:
- `.txt` - Plain text files (UTF-8 encoding)
- `.md` - Markdown files (UTF-8 encoding)

### Processing Flow:
```
input.txt/input.md → [Ingest] → parts/part_*.txt → [Summarize] → summaries/summary_*.md → [Stage] → stages/stage_*.md → [Final] → final/final_analysis.md
```

### Output Files Structure:
- `parts/` - Individual text segments
- `summaries/` - Part-level summaries
- `stages/` - Stage-level syntheses  
- `final/` - Final cross-stage analysis
- `manifest.json` - Tracking and state management
- `status.log` - Processing state log

---

## 4. Error Handling Philosophy

### Continue-with-Partial Strategy:
The system implements a resilient "continue-with-partial" error strategy that ensures partial progress is preserved even when individual processing steps fail:

1. **Individual Part Failure**: If a specific part fails to summarize, the system logs the error and continues processing remaining parts. Failed parts are tracked separately for retry attempts.

2. **Stage Generation Continuity**: If some part summaries fail but others succeed, stage generation continues with available summaries. The system adjusts grouping based on available summaries rather than failing entirely.

3. **Component-Level Recovery**: Each processing stage (Ingest, Summarize, Stage, Final) generates partial outputs that can be reused if subsequent stages fail, preventing loss of completed work.

4. **LLM Error Resilience**: Network timeouts, rate limits, and LLM errors are handled with exponential backoff retry logic. Failed LLM calls are logged with timestamps for manual intervention.

5. **Graceful Degradation**: When errors occur, the system continues with reduced functionality rather than stopping completely, always preserving existing progress.

---

## 5. Resume/Checkpoint Mechanics

### Manifest-Based Processing:
The system implements robust resume capabilities through a manifest-based approach with hash validation:

1. **State Tracking**: Each processing step writes state information to `manifest.json` including:
   - File paths of processed inputs
   - SHA-256 hashes of input content
   - Processing stage status (pending, in-progress, completed, failed)
   - Timestamps for all operations
   - Retry attempts for failed operations

2. **Hash Validation**: Before processing, the system validates that input file hashes match their originally recorded values, ensuring content integrity between runs.

3. **Incremental Processing**: On resume, the system identifies completed tasks via manifest inspection and skips these, focusing only on pending or failed items.

4. **Checkpoint Recovery**: During long-running processes, checkpoints are periodically saved to allow recovery from interruptions at multiple points rather than only from the beginning.

---

## 6. Edge Cases and Handling

### Empty Input Handling:
When encountering empty or nearly empty input files:
- Log warning but continue gracefully
- Create placeholders to maintain processing sequence integrity
- Update manifest appropriately

### Tiny Input Handling:
For input files below minimum recommended thresholds:
- Apply special processing path with adjusted prompts
- Generate notices indicating processing adaptations
- Maintain consistent output format despite small input

### Concurrent Run Prevention:
To prevent race conditions during concurrent runs:
- Implement file locking mechanism based on manifest
- Check for active lock files before starting processing
- Fail with guidance message for concurrent access

### Disk Space Monitoring:
Continuously monitor available disk space during processing:
- Check before initiating operations that create significant temporary data
- Fail gracefully with informative message if space constraints detected
- Clean up temporary data on failure

---

## 7. Hard Constraints

These fundamental limitations define the MVP's scope and operational parameters:

### Technical Constraints:
- **Single File Limit**: Process exactly one input file per pipeline run
- **Single Provider**: Connect to one LLM provider at a time
- **Sequential Processing**: Execute pipeline stages sequentially, not in parallel
- **Local Storage Only**: Operate solely with local files and manifests
- **No Streaming**: Use full text chunks without streaming input/output
- **UTF-8 Exclusivity**: Accept only valid UTF-8 encoded text input
- **CLI Interface Only**: No GUI, web interface, or interactive mode options

### Architectural Constraints:
- **Non-Distributed**: Designed as single machine application, no distributed processing
- **Linear Pipeline**: Fixed 4-stage processing path with no branching variations
- **Model Agnostic**: Must work with minimal configuration for OpenAI-compatible endpoints
- **Atomic Operations**: Each stage must complete entirely or roll back to avoid partial states
- **No External Storage**: All state maintained locally in working directory
- **Static Configuration**: Config loaded once at pipeline start, no runtime changes

### Functional Constraints:
- **Unidirectional Flow**: Process follows strict Ingest→Summarize→Stage→Final sequence
- **No Realtime Updates**: Batch processing only, no ability to add content during processing
- **Memory Conservative**: Operate within reasonable memory bounds without streaming
- **Deterministic**: Same input and configuration must produce identical manifest and outputs

---