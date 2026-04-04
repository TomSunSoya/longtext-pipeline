# Data Model Specification for longtext-pipeline

## Overview

This document specifies the data structures and file formats used by the longtext-pipeline, a hierarchical analysis tool for processing super-long text. The pipeline follows a multi-stage approach with checkpoint tracking through manifests and structured intermediate artifacts.

## Pipeline Flow and Relationship Map

```
Raw Input File
      ↓
Ingest → part_*.txt files
      ↓
Summarize → summary_*.md files  
      ↓
Stage Synthesis → stage_*.md files
      ↓
Final Analysis → final_analysis.md
      ↓ [Optional]
Audit → audit_*.md files
```

## 1. Manifest Schema (manifest.json)

Stores pipeline state and enables resumable execution. Located at `.longtext/manifest.json` by default.

### Schema Definition

```json
{
  "session_id": "string, unique ID for this pipeline run - generated as YYYYMMDD_HHMMSS_[random_suffix]",
  "input_path": "string, path to original input file",
  "input_hash": "string, SHA-256 hash of input file content to detect modifications",
  "created_at": "string, timestamp when session began (ISO 8601)",
  "updated_at": "string, timestamp of last update (ISO 8601)", 
  "total_parts": "integer, total number of parts after ingestion",
  "total_stages": "integer, total number of stages after grouping",
  "estimated_tokens": "integer [optional], rough token count of input",
  "status": "string, overall status ('not_started', 'ingesting', 'summarizing', 'staging', 'finalizing', 'completed', 'failed', 'partial_success')",
  "stages": {
    "ingest": {
      "status": "string ('not_started', 'running', 'successful', 'failed')",
      "completed_at": "string [optional], when stage completed",
      "error": "string [optional], error message if failed",
      "stats": {
        "parts_created": "integer, number of part files created",
        "estimated_tokens": "integer, estimated tokens in all parts combined"
      }
    },
    "summarize": {
      "status": "string ('not_started', 'running', 'successful', 'failed', 'skipped')",
      "completed_at": "string [optional], timestamp",
      "error": "string [optional], error message",
      "stats": {
        "summaries_completed": "integer, count of successfully processed summaries", 
        "summaries_total": "integer, total summaries to process",
        "errors": [
          {
            "index": "integer, 0-based part index",
            "file": "string, path to part file that failed",
            "error": "string, error message"
          }
        ]
      }
    },
    "stage": {
      "status": "string ('not_started', 'running', 'successful', 'failed', 'skipped')",
      "completed_at": "string [optional], timestamp",
      "error": "string [optional], error message",
      "stats": {
        "stages_completed": "integer, count of created stage files",
        "stages_total": "integer, total stages to process",
        "errors": [
          {
            "index": "integer, 0-based stage index",
            "summary_files": "string[], list of summary files processed",
            "error": "string, error message"
          }
        ]
      }
    },
    "final": {
      "status": "string ('not_started', 'running', 'successful', 'failed', 'skipped')",
      "completed_at": "string [optional], timestamp",
      "error": "string [optional], error message",
      "stats": {
        "completed_at": "string [optional], when final analysis completed"
      }
    },
    "audit": {
      "status": "string ('not_started', 'skipped', 'running', 'successful', 'failed'), defaults to 'skipped'",
      "completed_at": "string [optional], timestamp",
      "error": "string [optional], error message",
      "stats": {
        "audits_run": "integer [optional], number of audit operations"
      }
    }
  }
}
```

### Required Fields
- `session_id` - Unique identifier for the pipeline session
- `input_path` - Absolute path to original input file
- `input_hash` - SHA-256 hash to verify input hasn't changed since last run
- `created_at` - Initial session creation timestamp
- `updated_at` - Last update timestamp 
- `status` - Overall pipeline status
- `stages` - Individual stage tracking

### Example JSON

```json
{
  "session_id": "20260403_153045_ab7f2d",
  "input_path": "/home/user/documents/chat_log.txt",
  "input_hash": "a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef1234567",
  "created_at": "2026-04-03T15:30:45Z",
  "updated_at": "2026-04-03T15:45:22Z",
  "total_parts": 5,
  "total_stages": 2,
  "estimated_tokens": 25000,
  "status": "summarizing",
  "stages": {
    "ingest": {
      "status": "successful",
      "completed_at": "2026-04-03T15:32:12Z",
      "error": null,
      "stats": {
        "parts_created": 5,
        "estimated_tokens": 25000
      }
    },
    "summarize": {
      "status": "running",
      "completed_at": null,
      "error": null,
      "stats": {
        "summaries_completed": 3,
        "summaries_total": 5,
        "errors": []
      }
    },
    "stage": {
      "status": "not_started",
      "completed_at": null,
      "error": null,
      "stats": {
        "stages_completed": 0,
        "stages_total": 2,
        "errors": []
      }
    },
    "final": {
      "status": "not_started",
      "completed_at": null,
      "error": null,
      "stats": {}
    },
    "audit": {
      "status": "skipped",
      "completed_at": null,
      "error": null,
      "stats": {}
    }
  }
}
```

## 2. Part Files Format (part_*.txt)

Output of the **Ingest** stage, containing text chunks split for processing.

### Format Structure

```
INPUT_PATH: {absolute path to original input}
PART_INDEX: {0-based index of this part}
TOKEN_COUNT: {approximate token count for this chunk}
CHUNK_SIZE: {character count of content}
CONTENT_TYPE: {detected based on extension: 'text/plain' or 'text/markdown'}
METADATA_END: ---END---

{Raw text content of this chunk}
```

### Example File (`part_01.txt`)

```
INPUT_PATH: /home/user/documents/chat_log.txt
PART_INDEX: 0
TOKEN_COUNT: 1250
CHUNK_SIZE: 7432
CONTENT_TYPE: text/plain
METADATA_END: ---END---

Alice: Hey, how's the project coming along?

Bob: It's going well, though we hit a slight roadbump with the authentication flow.

Alice: What kind of roadbump? 

Bob: Well, the OAuth integration took longer than expected...

[more chat content here...]
```

### Required Components
- Metadata header with file tracking information
- Separator marking end of metadata (`---END---`)
- Raw chunk content without embedded structural changes

## 3. Summary Files Format (summary_*.md)

Output of the **Summarize** stage, containing summaries of corresponding parts.

### Format Structure

```markdown
# Summary for Part {:02d}

**Generated:** {ISO 8601 timestamp}  
**Part File:** {link to corresponding part file}  
**Tokens:** {approximate token count of summary}  

## Key Points
- List of main ideas extracted from the corresponding part
- Bullet points summarizing important aspects

## Entities Identified  
- Person: Named entities found in text
- Organization: Companies, institutions, etc.
- Location: Geographic places mentioned
- Other: Dates, events, concepts of significance

## Themes
- Higher-level themes emerging from text
- Recurring topics or narratives

## Action Items (if applicable)
- Tasks, responsibilities, deadlines mentioned in part

## Additional Notes
[Any other salient information not captured elsewhere]

---

_Summary generated by [{LLM model}]({session timestamp})_
```

### Example File (`summary_01.md`)

```markdown
# Summary for Part 01

**Generated:** 2026-04-03T15:35:21Z  
**Part File:** ../part_01.txt
**Tokens:** 250  

## Key Points
- Alice and Bob discussing project status 
- Authentication flow presented challenges
- OAuth integration took longer than expected

## Entities Identified
- Person: Alice, Bob
- Topic: Project, authentication flow, OAuth integration

## Themes
- Project management and communication
- Technical implementation challenges  
- Timeline adjustments due to complexity

## Action Items (if applicable)
- No specific action items assigned in this segment

## Additional Notes
- Bob seems optimistic despite roadbump, frames challenge positively

---

_Summary generated by gpt-4-turbo (2026-04-03T15:35:21Z)_  
```

## 4. Stage Files Format (stage_*.md)

Output of the **Stage** stage, synthesizing multiple part summaries into organized insights.

### Format Structure

```markdown
# Stage Summary {:02d}

**Generated:** {ISO 8601 timestamp}  
**Combined From:** {links to summary files in this stage}
**Summary Count:** {integer, how many summary files combined}
**Tokens:** {approximate token count of overall stage summary}

## Executive Summary  
Brief synthesis of the themes covered in this stage's constituent summaries.

## Consolidated Key Points
- Aggregated insights from multiple parts
- Cross-part connections and developments
- Thematic progression across parts

## Entity Synthesis
### Persons
- Combined person mentions with frequency/count
- Role identification across conversations

### Organizations & Concepts  
- Organizational information consolidated
- Technical concepts clarified across contexts  

## Theme Evolution
- How major topics developed across this stage
- Relationships between themes that emerge when summaries are viewed together

## Consistency Checks
[Optional section - highlights discrepancies or contradictions]
- Inconsistent information across parts (if any)  
- Timeline issues (if any)
- Contradictory statements (if any)

## Action Items Tracking
- Cumulative list of all action items from constituents
- Responsibility assignments observed

---

_Staged synthesized by [{LLM model}]({session timestamp})_
```

### Example File (`stage_01.md`)

```markdown
# Stage Summary 01

**Generated:** 2026-04-03T15:40:15Z  
**Combined From:** [summary_01.md](./summary_01.md), [summary_02.md](./summary_02.md), [summary_03.md](./summary_03.md)
**Summary Count:** 3
**Tokens:** 524

## Executive Summary
Three consecutive segments show a project progressing from initial challenges detection through implementation details. Team communication remains positive despite obstacles.

## Consolidated Key Points
- Project facing authentication complexity challenges
- Original timeline likely needs adjustment 
- Team maintaining good morale during difficulties
- Clear division of responsibilities observed

## Entity Synthesis  
### Persons
- Alice: Project coordinator (mentioned 8 times)
- Bob: Development lead (mentioned 9 times) 
- Sarah: Brief mention as stakeholder (mentioned 1 time)

### Organizations & Concepts
- Project Team: Small team of 2-3 core members
- Authentication Flow: Technical focus area requiring OAuth integration  
- Timeline Expectations: Under review due to implementation complexity

## Theme Evolution  
- Challenges: Evolves from discovery (summary_01) → details (summary_02) → implications (summary_03)
- Communication: Remains consistently constructive across all segments
- Complexity: Becomes clearer from estimation → implementation → timeline impact sequence

## Consistency Checks
- No apparent inconsistencies across analyzed segments
- Timeline progression internally consistent  

## Action Items Tracking
- Review OAuth integration approach (to discuss with Sarah)
- Update timeline estimate for authentication component

---

_Staged synthesized by gpt-4-turbo (2026-04-03T15:40:15Z)_  
```

## 5. Final Analysis Format (final_analysis.md)

Output of the **Final** stage, providing comprehensive analysis across all stages.

### Format Structure

```markdown
# Final Analysis for {input filename}

**Generated:** {ISO 8601 timestamp}
**Source File:** [{input filename}]({original path})
**Analysis Scope:** {total parts} parts, {total stages} stages analyzed
**Processing Time:** {estimated duration in seconds/minutes}
**Tokens Analyzed:** {rough approximate token count}
**Models Used:** {LLM models used for processing}

## Executive Summary
A high-level, holistic overview integrating insights from all stages. This should highlight the three to five most significant insights about the text corpus.

## Comprehensive Theme Analysis
### Primary Themes  
- Most prominent themes identified throughout the entire document
- How these themes interrelate and influence each other

### Secondary Themes  
- Less dominant but still significant thematic content
- Niche concepts that appear occasionally but meaningfully

### Thematic Evolution  
- How themes developed over the course of the document
- Progression or shifts in focus

## Participant Analysis  
### Key Participants
- Comprehensive list of all significant people/entities identified
- Their roles, influence, and activities throughout the document
- Frequency of mention and centrality to discourse

### Interaction Dynamics
- How participants communicate and respond to each other
- Leadership roles, decision-making patterns, consensus styles
- Relationship dynamics and communication effectiveness

## Timeline Reconstruction 
[If applicable] Chronological ordering of significant events and developments, especially useful for conversation logs, meeting minutes, or process documentation.

## Anomalies and Edge Cases
### Inconsistencies
- Any contradictions or temporal inconsistencies identified
- Conflicting information that needs human review

### Outliers  
- Unusual events, communications, or content that stands out
- Departures from typical patterns in communication or topic

## Implications and Recommendations  
### Immediate Implications
- Consequences likely to follow from the documented situation

### Strategic Considerations
- Longer-term impacts, opportunities, or risks identified

### Recommendations
- Specific action recommendations based on the analysis
- Areas meriting focused attention

## Confidence Assessment
### Highly Confident Elements
- Information verified through multiple sources/contextual consistency

### Tentative Elements
- Interpretations based on limited context or ambiguous language

### Uncertain Elements
- Areas that may require clarification or additional information

---

_Final analysis generated by [{LLM model}]({session timestamp})_

**Confidence Level:** [] High [] Medium [] Low  
**Review Needed:** [ ] Yes [ ] No [ ] Critical Elements Only
```

### Example File (`final_analysis.md`) 

```markdown
# Final Analysis for chat_log.txt

**Generated:** 2026-04-03T15:45:10Z
**Source File:** [chat_log.txt](file:///home/user/documents/chat_log.txt)  
**Analysis Scope:** 5 parts, 2 stages analyzed
**Processing Time:** 285s
**Tokens Analyzed:** 24500
**Models Used:** gpt-4-turbo

## Executive Summary
Team is effectively managing a software development project through communication challenges during authentication implementation phase. Strong team cohesion maintained despite timeline uncertainties. Key focus shift toward OAuth integration complexity.

## Comprehensive Theme Analysis  
### Primary Themes
- **Project Management Resilience:** Team adapts to implementation challenges while maintaining momentum
- **Technical Complexity Navigation:** Specific challenges with OAuth requiring expert approach  
- **Communication Stability under Stress:** Interpersonal relations strong during difficult periods

### Secondary Themes  
- **Timeline Adaptation:** Natural response to complexity emergence
- **Stakeholder Coordination:** Mention of Sarah suggests multi-level engagement
- **Risk Mitigation Awareness:** Proactive problem identification

### Thematic Evolution
Document moves from challenge identification through detailed problem scoping to solution adaptation. No evidence of escalating tension or blame attribution.

## Participant Analysis  
### Key Participants
- **Alice (Project Coordinator):** Primarily information-seeking, maintains oversight stance (24 mentions)
- **Bob (Development Lead):** Technical problem ownership, solution-oriented perspective (26 mentions) 
- **Sarah (Stakeholder):** Referenced but infrequently appearing (1 mention)

### Interaction Dynamics
Collaborative without conflict. Alice asks clarifying questions, Bob provides technical details. Balanced communication flow without dominance.

## Timeline Reconstruction
1. Problem identification phase  
2. Technical detail extraction phase
3. Timeline impact assessment phase
4. Solution pathway establishment

## Anomalies and Edge Cases
### Inconsistencies
None identified at analysis level.
### Outliers
One stakeholder reference suggesting broader team awareness than direct dialogue indicates.

## Implications and Recommendations
### Immediate Implications
OAuth implementation delay affects overall timeline; need stakeholder communication.

### Strategic Considerations  
Robust technical foundation being built; longer-term stability may result.

### Recommendations  
- Schedule Sarah alignment meeting regarding approach
- Formal timeline reassessment for overall project tracking

## Confidence Assessment
### Highly Confident Elements
Participant roles and communication patterns
### Tentative Elements  
Specific technical challenges beyond described details
### Uncertain Elements
Sarah's full role and expectations

---

_Final analysis generated by gpt-4-turbo (2026-04-03T15:45:10Z)_

**Confidence Level:** [x] High [ ] Medium [ ] Low  
**Review Needed:** [ ] Yes [x] No [ ] Critical Elements Only
```

## 6. Audit Files Format (audit_*.md) [Experimental]

Output of the **Audit** stage (if performed), reviewing analysis quality and accuracy.

### Format Structure

```markdown
# {stage_to_audit} Audit Report - {timestamp}

**Analysis File Audited:** {path/to/file/being/augited}  
**Audit Performed:** {YYYY-MM-DDTHH:MM:SSZ}
**Quality Score:** {numeric scale e.g., 1-10 or % confidence}
**Model:** {model used for auditing}

## Issue Categories Identified

### Factual Accuracy ({score}/10)
- Verified facts matching source material
- Flagged potentially erroneous interpretations
- Areas requiring factual rechecking

### Thematic Coherence ({score}/10) 
- Consistency of theme identification
- Logical progression of ideas
- Apparent omissions or mischaracterizations

### Temporal Sequencing ({score}/10) [if applicable]
- Chronological accuracy in timeline reconstruction
- Event sequence integrity
- Temporal assumption validity

### Entity Tracking ({score}/10)
- Consistent participant identification
- Accurate role assignment and relationship mapping
- Entity disambiguation quality

## Specific Quality Notes

### Confidence Alerts
- {List elements the auditor considers most questionable}

### Verification Needs
- {List where external verification would improve reliability} 

### Enhancement Suggestions  
{Specific suggestions for improving future analysis of similar documents}

---

_Audit performed by [{model_used}]({completion_timestamp})_
```

## 7. File Relationships and Referencing

### Cross-File References

Files in the pipeline reference each other through:

1. **Metadata Headers**: Each file contains references to its parent/source
2. **Filename Conventions**: Sequential naming reveals dependency chains
3. **Manifest Tracking**: Central manifest links all related assets

### File Resolution Path

When reconstructing the analysis pathway:
```
manifest.json → stages[] → stage_name → output_file
              → input requirements → part_*.txt or summary_*.md
```

### Resume Capabilities 
The manifest.json, validated against input_hash, determines which stages can be skipped during subsequent runs.

### Output Directory Structure

```
.original_input_basename/
├── manifest.json                 # Main checkpoint tracker
├── part_01.txt                   # First chunk after ingestion
├── part_02.txt                   # Second chunk after ingestion
├── ...
├── summary_01.md                 # Summary for part_01
├── summary_02.md                 # Summary for part_02
├── ...
├── stage_01.md                   # Stage summary of summaries 1-3
├── stage_02.md                   # Stage summary of remaining summaries  
├── ...
├── final_analysis.md             # Cross-stage synthesis
├── audit_final.md                # [Optional] Quality audit
└── .metadata/
    ├── generation.log            # Generation timestamp and details
    ├── token_estimates.json      # Running token counts and estimates
    └── error_reports.json        # Collection of partial failure information
```

Note: Basename refers to the input filename without extension, and all numbered files use zero-padded numbering.