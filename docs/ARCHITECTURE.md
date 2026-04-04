# Architecture Overview

## Module Structure

The `longtext-pipeline` follows a layered modular architecture with clear responsibilities and interfaces:

- **CLI Layer** (`cli.py`) - Command-line interface
- **Configuration Layer** (`config.py`) - Configuration management  
- **Model Layer** (`models.py`, `errors.py`) - Data structures and error definitions
- **Processing Layer** (`splitter.py`, `grouper.py`) - Text processing and grouping
- **Manifest Layer** (`manifest.py`) - State tracking and resume capability
- **Rendering Layer** (`renderer.py`) - Output formatting
- **LLM Layer** (`llm/`) - LLM abstraction and provider integration
- **Pipeline Layer** (`pipeline/`) - Core stages of processing
- **Utilities Layer** (`utils/`) - General-purpose utility functions
- **Prompt Resources** (`prompts/`) - Static resources for LLMs

---

## Module Responsibilities and Interfaces

### 1. CLI Module (`src/longtext_pipeline/cli.py`)
**Responsibilities:**
- Entry point for the application
- Parse command-line arguments and options
- Initialize and wire together components based on user inputs
- Handle basic configuration loading and validation
- Manage the overall execution flow orchestration
- Dispatch processing tasks to appropriate pipeline stages

**Anti-Responsibilities (strictly must NOT do):**
- Perform actual text processing or analysis beyond parsing user input
- Direct implementation of LLM integration or communication
- Detailed configuration validation (passes to config module)
- File I/O operations (delegates to utility functions)
- Business logic implementation (orchestrates but doesn't process)
- Error recovery beyond basic command-line argument validation

**Interface Contract:**
- Accepts command-line arguments and returns exit codes
- Interacts with Config module for configuration loading
- Calls Pipeline stages with processed inputs

---

### 2. Configuration Module (`src/longtext_pipeline/config.py`)
**Responsibilities:**
- Load configuration from YAML files, environment variables, and CLI arguments
- Validate configuration against predefined schema
- Provide runtime configuration objects with defaults
- Handle configuration merging and precedence
- Supply LLM client configuration settings

**Anti-Responsibilities (strictly must NOT do):**
- Perform file I/O beyond configuration loading and saving
- Initiate text processing, LLM operations, or pipeline tasks
- Implement business logic unrelated to configuration
- Store or manage application state beyond configuration data
- Handle user interaction beyond configuration setup
- Execute external processes or system commands

**Interface Contract:**
- Accepts paths to config files and environment vars
- Returns validated configuration objects
- Provides LLM-specific settings to LLM module

---

### 3. Models Module (`src/longtext_pipeline/models.py` and `errors.py`)
**Responsibilities:**
- Define data structures (Pydantic models) for all pipeline operations
- Represent input files, split parts, summaries, stages, manifests
- Define pipeline state and intermediate representations  
- Define custom Exceptions for the application

**Anti-Responsibilities (strictly must NOT do):**
- Perform I/O operations on disk, network, or any storage
- Interact with external services, APIs, databases, or filesystems
- Handle business logic or perform actual data processing
- Implement algorithms or manipulate data content
- Initiate LLM calls or communicate with models
- Control program flow or make conditional decisions based on data

**Interface Contract:**
- Provides structured data types across all modules
- Handles serialization/deserialization of structured data

---

### 4. Errors Module (`src/longtext_pipeline/errors.py`)
**Responsibilities:**
- Define all custom exceptions for the pipeline
- Provide error categories (validation, processing, LLM, etc.)
- Establish error hierarchies and codes

**Anti-Responsibilities (strictly must NOT do):**
- Implement error recovery or mitigation logic
- Log errors or handle exception output (logging handled elsewhere)
- Handle exception catching globally across the application
- Control program execution or manage retry attempts
- Perform diagnostic operations beyond error classification
- Store runtime information or application state

**Interface Contract:**
- Provides exception classes for other modules to raise and catch
- Supports error categorization for exception handling strategies

---

### 5. Splitter Module (`src/longtext_pipeline/splitter.py`)
**Responsibilities:**
- Divide input texts into manageable segments according to settings
- Ensure segment boundaries align with meaningful text boundaries
- Generate unique identifiers for each segment
- Handle large file processing efficiently

**Anti-Responsibilities (strictly must NOT do):**
- Interact with LLMs or initiate external service calls
- Perform summarization, analysis, or content manipulation tasks
- Handle persistence of outputs or file writing operations
- Apply business rules beyond text segmentation logic
- Maintain complex application state or manage processing contexts
- Perform validation beyond segmentation parameters

**Interface Contract:**
- Accepts raw text string and configuration
- Returns array of text segments with identifiers
- Integrates with I/O utilities for file reading

---

### 6. Grouper Module (`src/longtext_pipeline/grouper.py`)
**Responsibilities:**
- Group processed segments into meaningful stages for analysis
- Apply grouping logic based on sequence and semantic considerations
- Ensure coherent analysis of related content
- Maintain grouping history for traceability

**Anti-Responsibilities (strictly must NOT do):**
- Generate summaries or initiate LLM-based content generation
- Interact with external APIs or service calls
- Modify original content or alter the semantics of segments
- Perform content validation or quality assessments
- Handle file I/O operations for persisting results
- Manage pipeline execution or coordinate processing stages

**Interface Contract:**
- Accepts arrays of processed segments
- Returns grouped collections optimized for next-stage processing

---

### 7. Manifest Module (`src/longtext_pipeline/manifest.py`)
**Responsibilities:**
- Track processing state of individual files and parts
- Enable resumable operations using SHA-256 hash validation  
- Track which parts have been processed and which remain
- Manage progress reporting and status validation
- Coordinate checkpoint creation and restoration

**Anti-Responsibilities (strictly must NOT do):**
- Perform actual file content processing or analysis
- Interact with LLMs or initiate AI model requests  
- Handle business logic of the processing pipeline
- Modify or analyze the content stored in processed files
- Perform I/O with external services beyond local state files
- Make decisions about what processing should occur

**Interface Contract:**
- Persists processing state to disk
- Provides resume functionality for pipeline operations
- Coordinates with I/O utilities for file-based state tracking

---

### 8. Renderer Module (`src/longtext_pipeline/renderer.py`)
**Responsibilities:**
- Format pipeline outputs in desired presentation formats
- Convert structured data models to text/markdown/binary outputs
- Support different output templates and styles
- Handle document assembly from components

**Anti-Responsibilities (strictly must NOT do):**
- Modify content semantics or change the factual meaning of processed data
- Apply business logic transformations or editorial decisions
- Interact directly with LLMs or initiate model requests
- Perform validation beyond formatting and structure checks
- Manage file persistence or handle I/O operations (delegates to utilities)
- Store runtime application state or pipeline progress information

**Interface Contract:**
- Accepts structured pipeline data as input
- Returns formatted output strings or files for persistence

---

### 9. LLM Layer (`src/longtext_pipeline/llm/`)
#### `base.py` - Abstract base class layer:
**Responsibilities:**
- Define abstract interfaces for LLM operations
- Establish common LLM interaction patterns
- Provide base class for all LLM integrations

#### `openai_compatible.py` - Concrete implementations:
**Responsibilities:**
- Implement specific LLM provider integrations
- Handle API communication, retries, and error propagation
- Perform token estimation and rate limiting  
- Manage conversation state and request templating

#### `factory.py` - Factory for instantiation:
**Responsibilities:**  
- Create appropriate LLM client based on configuration
- Manage LLM client lifecycle
- Provide centralized client creation with proper setup

**Anti-Responsibilities (any of the LLM modules - strictly must NOT do):**
- Determine when LLM operations should happen (pipeline modules decide execution timing)
- Store processed content permanently or manage persistence
- Make high-level architectural decisions about processing flow
- Implement error recovery beyond LLM communication failure
- Handle application state or coordinate pipeline execution
- Manage authentication, billing or usage outside of API calls

**Interface Contract:**
- Accepts prompts and configuration settings
- Returns processed responses or raises exceptions
- Integrates with Configuration and Pipeline modules for settings and orchestration

---

### 10. pipeline/ Submodules
#### `ingest.py` - Ingestion stage:
**Responsibilities:**
- Load input files and prepare for text splitting
- Perform pre-processing on content (encoding, cleaning)
- Prepare initial pipeline state

#### `summarize.py` - Summarization stage:
**Responsibilities:**
- Apply LLM-based summarization to individual text parts
- Manage prompt templating for summarization tasks
- Handle error recovery for failed summary attempts

#### `stage.py` - Stage aggregation stage:
**Responsibilities:**
- Combine multiple summaries into higher-level stage summaries
- Maintain relationship context across groupings  
- Apply staged summarization methodology

#### `final.py` - Final composition stage:
**Responsibilities:**
- Generate final comprehensive analysis from stage summaries
- Maintain synthesis coherence and completeness
- Apply final formatting and validation checks

#### `audit.py` - Optional audit stage:
**Responsibilities:**
- Audit generated content for hallucinations, timeline misalignment
- Verify evidence consistency with source materials
- Flag problematic assertions and weak inferences

**Anti-Responsibilities (any of the pipeline modules - strictly must NOT do):**
- Perform complex file I/O beyond their immediate scope (use utilities for that)
- Define core data models (use models module)
- Implement basic utility functions (depend on utilities module)
- Handle direct user interfaces (use CLI module for that)
- Store application configuration or manage state directly
- Manage LLM connections directly (use LLM module for that)

**Interface Contract:**
- Accept configured pipeline components via initialization
- Execute specific stage of processing
- Return structured results for subsequent stages

---

### 11. Utilities (`src/longtext_pipeline/utils/`)
#### `io.py` - I/O operations:
**Responsibilities:**
- Handle file reading and writing operations
- Manage path manipulation
- Coordinate disk-based caching and temporary files

#### `hashing.py` - Hash calculations:
**Responsibilities:**  
- Calculate SHA-256 hashes for content
- Support cache key generation
- Provide content identity verification

#### `token_estimator.py` - Token calculations:  
**Responsibilities:**
- Estimate token usage for cost management
- Support input validation against length limits
- Assist with chunk size decisions

#### `text_clean.py` - Text preprocessing:
**Responsibilities:**
- Standardize text formatting
- Remove extraneous whitespace
- Normalization for predictable processing

#### `retry.py` - Retry logic:
**Responsibilities:**
- Handle retry policies for transient failures
- Coordinate exponential backoff patterns  
- Provide resilience for API calls

**Anti-Responsibilities (strictly must NOT do):**
- Implement domain-specific business logic that violates separation of concerns
- Change meaning or substance of content being processed
- Handle error cases beyond generic utility-level failures (like I/O errors)
- Manage external dependencies or system-specific configurations
- Perform operations that require application state awareness
- Handle application-level concerns like coordination or orchestration

**Interface Contract:**
- Accept specific input with clear specifications  
- Return enhanced version of input or extracted values
- Enable other modules to focus on their core concerns

---

## Data Flow Between Modules

The system follows a strict unidirectional data flow pattern from top to bottom:

```
CLI → Config → Splitter → Grouper → [Summary Stage] → [Stage Stage] → [Final Stage] → Renderer
                        ↓
                   Manifest (tracks state)
                        ↓
                    Utilities (I/O, hashing, etc.) ← LLM clients ← Prompts
```

**Ingest Stage:**
1. CLI parses arguments and loads config
2. Config creates configuration objects
3. Splitter processes input files based on config
4. Manifest tracks processed segments
5. Results passed to Summarize stage

**Summarize Stage:**
1. Pipeline receives prepared segments 
2. Uses LLM module to apply summarization
3. Manifest tracks progress of summary processing
4. Results passed to Stage stage

**Stage Stage:**
1. Grouper assembles relevant summaries into groups
2. Stage pipeline applies group-level analysis 
3. Results tracked via Manifest
4. Passed to Final stage  

**Final Stage:**
1. Final pipeline synthesizes results from all stages
2. Applies renderer for final format
3. Outputs completed analysis

Throughout all stages, error handling and logging modules coordinate to provide resilience and observability, while utilities support all layers with infrastructure functions.