# RegDelta — Regulatory Change Impact Analyzer

## A Production-Grade Multi-Agent RAG System for Financial Compliance

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [The Real-World Problem](#2-the-real-world-problem)
3. [Why AI / LLMs Are Necessary](#3-why-ai--llms-are-necessary)
4. [Why RAG Is Necessary](#4-why-rag-is-necessary)
5. [Why Agentic RAG Is Necessary](#5-why-agentic-rag-is-necessary)
6. [Why Multi-Agent Architecture Is Justified](#6-why-multi-agent-architecture-is-justified)
7. [System Architecture](#7-system-architecture)
8. [RAG Architecture (Deep Dive)](#8-rag-architecture-deep-dive)
9. [Agent Architecture (Deep Dive)](#9-agent-architecture-deep-dive)
10. [Data Sources and Datasets](#10-data-sources-and-datasets)
11. [LLM and Model Choices](#11-llm-and-model-choices)
12. [Technology Stack](#12-technology-stack)
13. [Evaluation Framework](#13-evaluation-framework)
14. [Baselines and Ablation Plan](#14-baselines-and-ablation-plan)
15. [Research Questions](#15-research-questions)
16. [Failure Mode Analysis](#16-failure-mode-analysis)
17. ["Why Not Just Use a Single LLM?"](#17-why-not-just-use-a-single-llm)
18. [Build Roadmap (Phased)](#18-build-roadmap-phased)
19. [Learning Outcomes](#19-learning-outcomes)
20. [Interview Preparation](#20-interview-preparation)
21. [Appendix A: API Reference](#appendix-a-api-reference)
22. [Appendix B: Synthetic Policy Generation Guide](#appendix-b-synthetic-policy-generation-guide)
23. [Appendix C: Glossary](#appendix-c-glossary)

---

# 1. Project Overview

## What Is RegDelta?

RegDelta is a multi-agent AI system that automatically analyzes the impact of new regulatory changes on an organization's internal compliance policies. When a regulator (SEC, FINRA, CFPB, OCC, or any financial regulatory body) publishes a new rule, guidance update, or enforcement action, RegDelta:

1. **Parses** the regulatory text to extract structured requirements (what changed, who is affected, when it takes effect, what is required).
2. **Maps** those requirements against internal compliance policies to identify which policies are affected.
3. **Detects conflicts** between the new requirement and existing regulatory obligations from other jurisdictions or regulators.
4. **Assesses impact** by performing gap analysis — comparing what the regulation requires versus what the current policy provides.
5. **Generates a structured impact report** with severity ratings, recommended actions, and citations to specific regulatory sections and policy provisions.

## One-Line Description

> An agentic RAG system that reads new financial regulations and tells you exactly which of your internal policies need to change, why, and how urgently — with citations.

## What Makes This Project Exceptional

- **Multi-hop retrieval is genuinely necessary**, not bolted on. The impact chain (new rule → existing regulation it modifies → internal policy implementing that regulation) is a 3+ hop retrieval problem.
- **Multi-agent decomposition is naturally motivated.** Parsing regulatory text, searching internal policies, detecting cross-regulatory conflicts, and performing gap analysis are fundamentally different tasks requiring different tools and prompting strategies.
- **Evaluation is unambiguous.** Either the system correctly identified the affected policies or it didn't. Either the citations point to real text or they don't.
- **The domain commands respect.** Compliance is a high-stakes, well-understood business problem where incorrect answers have financial and legal consequences.

---

# 2. The Real-World Problem

## Who Has This Problem?

- **Compliance officers** at banks, insurance companies, asset managers, broker-dealers, and fintech firms
- **Legal teams** at financial institutions responsible for regulatory interpretation
- **Regulatory change management (RCM) teams** — dedicated functions at large banks (JPMorgan, Goldman Sachs, HSBC all have RCM teams of 20-50+ people)
- **RegTech companies** building compliance automation products
- **Consulting firms** (Big 4) advising clients on regulatory compliance

## What Do They Currently Do?

When a new regulation is published:

1. An analyst reads the full regulatory text (often 50-200 pages for a major rule)
2. They manually identify the key requirements, definitions, effective dates, and affected entities
3. They search through the organization's policy library (often hundreds of documents totaling thousands of pages) to find relevant policies
4. They compare the regulatory requirements against current policy provisions
5. They check whether the new rule conflicts with or modifies obligations from other regulators
6. They write a regulatory change assessment with recommendations
7. A senior compliance officer reviews and approves the assessment
8. Action items are created for policy owners to update their policies

**Timeline:** A single significant regulatory update takes 2-4 weeks for a team of 2-3 analysts.

## Why Is This Inefficient?

- **Volume:** The US financial regulatory corpus produces thousands of updates per year across SEC, FINRA, OCC, CFPB, FDIC, Federal Reserve, state regulators, and others. A large bank monitors 20+ regulators.
- **Cross-reference complexity:** Regulations reference other regulations, which reference other regulations. A single new SEC rule might modify 5 existing rules, each of which is referenced by 10 internal policies.
- **Heterogeneous sources:** Regulatory text comes in different formats (Federal Register XML, PDF guidance letters, HTML enforcement actions, Word document compliance bulletins).
- **Institutional knowledge dependency:** Experienced compliance professionals who understand the relationships between regulations and internal policies are expensive and hard to replace.
- **Lag time:** The gap between regulatory publication and completed impact assessment creates compliance risk.

## Why Is This Worth Solving?

- Non-compliance fines in financial services exceeded **$10 billion globally** in recent years
- A single regulatory enforcement action can result in **$100M+ penalties** (Wells Fargo, Deutsche Bank, etc.)
- The regulatory change management market is estimated at **$2-4 billion** annually
- Banks spend an estimated **$270 billion annually** on compliance overall (Thomson Reuters)
- The demand for compliance professionals far exceeds supply

## What Would Happen If Solved?

- Impact assessments completed in **hours** instead of weeks
- 10x more regulatory updates analyzed per analyst
- Reduced risk of missed regulatory changes
- Consistent, auditable analysis with citations
- Institutional knowledge captured in the system rather than in analysts' heads

---

# 3. Why AI / LLMs Are Necessary

This problem **cannot** be solved with:

### Not SQL / Traditional Databases
Regulatory impact assessment is not a lookup problem. You cannot write a SQL query for "does this new SEC climate disclosure rule conflict with our existing ESG reporting policy?" The relationship is semantic, not structural.

### Not Traditional Search (Elasticsearch / keyword matching)
Regulatory text uses precise legal language where semantically equivalent concepts use different terms. "Material adverse change" relates to "significant negative impact," but they share no keywords. BM25 alone misses these connections. (However, BM25 is valuable as *part* of the solution — hence hybrid retrieval.)

### Not Classical ML / NLP
Older NER/classification models can extract entity types but cannot reason about the *implications* of a regulatory change across multiple documents. The task requires understanding, comparison, and judgment — not classification.

### Not Deterministic Workflows
Every regulatory update is different. Some affect one policy area; others span multiple domains. Some modify existing rules; others create entirely new requirements. The analysis path cannot be predetermined.

### Why LLMs Specifically?
- **Legal language understanding:** LLMs can interpret nuanced regulatory language including defined terms, cross-references, exceptions, and carve-outs
- **Multi-document reasoning:** LLMs can compare requirements from one document against provisions in another and identify gaps
- **Structured extraction:** LLMs can extract structured fields (requirements, effective dates, affected entities) from unstructured regulatory prose
- **Synthesis:** LLMs can produce coherent impact assessments that explain the relationship between regulatory changes and policy gaps

---

# 4. Why RAG Is Necessary

Providing the full context to an LLM without retrieval fails for these specific reasons:

### Knowledge Is Private
Internal compliance policies are proprietary. They cannot be included in LLM training data. They must be retrieved at inference time.

### Knowledge Changes Constantly
Regulations are published daily. Internal policies are updated monthly. The system must always reason over the current version of both regulatory text and internal policies.

### Volume Exceeds Context Windows
Even with 200K token context windows:
- The US Code of Federal Regulations (CFR) for financial services alone is millions of tokens
- A large bank's internal policy library is 500-2000 documents totaling millions of tokens
- A single analysis might need to reference 20-50 specific sections across 10-15 documents

### Citations Are Non-Negotiable
In compliance, every claim must be traceable to a specific regulatory section and policy provision. RAG provides the provenance chain; a pure LLM generation does not.

### Multiple Sources Must Be Reconciled
A single analysis might draw from:
- The new regulatory text (Federal Register)
- The existing regulation being modified (eCFR)
- Related guidance from the same regulator
- Related regulations from other regulators
- 5-10 internal policies
- Historical enforcement actions (for understanding regulatory intent)

RAG provides the mechanism to search, retrieve, and present the right information from the right source at the right time.

---

# 5. Why Agentic RAG Is Necessary

A standard (static pipeline) RAG system fails here because:

### Multi-Hop Retrieval
The impact chain requires chained retrieval:
1. **Hop 1:** Parse the new regulation → identify which existing regulations it modifies (e.g., "This rule amends 17 CFR § 229.402")
2. **Hop 2:** Retrieve the existing regulation being modified → understand what it currently requires
3. **Hop 3:** Find internal policies that implement that existing regulation → understand current compliance posture
4. **Hop 4:** Assess the gap between new requirements and current policy provisions

A static retrieve-then-generate pipeline does not support this chain. An agent must execute retrieval iteratively, using results from one hop to formulate queries for the next.

### Query Decomposition
A single regulatory update may affect multiple compliance domains simultaneously. The new SEC climate disclosure rule affects:
- ESG reporting policies
- Risk disclosure policies
- Internal controls policies
- Board governance policies

The agent must decompose the analysis into domain-specific sub-queries rather than attempting a single monolithic retrieval.

### Dynamic Source Selection
Different queries require different data sources:
- "What does the new rule require?" → Federal Register API
- "What does the existing regulation say?" → eCFR (Electronic Code of Federal Regulations)
- "Which internal policies are affected?" → Internal policy vector store
- "Is there related enforcement guidance?" → SEC enforcement actions database

The agent must decide which sources to query based on the nature of each sub-question.

### Iterative Retrieval with Sufficiency Assessment
Initial retrieval may not return all relevant policies. The agent must:
1. Retrieve initial results
2. Assess whether coverage is sufficient (did we find policies for all affected domains?)
3. Reformulate queries and retrieve more if needed
4. Determine when enough evidence has been gathered to make an assessment

This is the Self-RAG / Corrective RAG pattern — the system must evaluate its own retrieval quality before generating.

### Evidence Verification
The agent must verify that retrieved documents actually support its analysis:
- Do the cited policy sections actually address the regulatory requirement in question?
- Is the retrieved version of the regulation current?
- Are the cross-references between regulations correctly resolved?

---

# 6. Why Multi-Agent Architecture Is Justified

### The Decomposition Argument

RegDelta's workflow decomposes into tasks that are genuinely different in their tools, retrieval strategies, and reasoning:

| Agent | Primary Task | Tools Needed | Retrieval Strategy | Reasoning Type |
|-------|-------------|-------------|-------------------|---------------|
| **Regulatory Parser** | Extract structured requirements from regulatory text | PDF parser, Federal Register API, structured output schema | Retrieve full regulatory text from API | Extraction (structured output) |
| **Policy Mapper** | Find internal policies affected by each requirement | Vector DB, BM25 index, metadata filters | Hybrid retrieval over internal policy store | Similarity + relevance judgment |
| **Conflict Detector** | Find conflicts with existing obligations from other regulators | Regulatory graph DB, cross-regulatory search | Graph traversal + vector search across regulatory corpus | Logical comparison + conflict identification |
| **Impact Analyst** | Perform gap analysis and severity scoring | Comparison templates, severity rubrics | Uses results from Policy Mapper and Conflict Detector | Multi-step reasoning + judgment |
| **Report Generator** | Compile findings into a structured report | Report template engine, citation formatter | No retrieval (consumes upstream results) | Synthesis + formatting |

### Why Not a Single Agent?

A single agent with all these tools would face:

1. **Context window pressure:** The Regulatory Parser generates structured output that the Policy Mapper needs, which generates retrieved policies that the Impact Analyst needs. Carrying all of this in a single agent's context leads to information loss ("Lost in the Middle") and hallucination.

2. **Tool confusion:** An agent with 10+ tools (regulatory API, policy vector search, BM25 search, graph traversal, PDF parser, metadata filter, severity scorer, citation formatter, template engine, web scraper) is empirically more likely to select the wrong tool than an agent with 2-3 tools.

3. **Prompt interference:** The prompting strategy for structured extraction (Regulatory Parser) is fundamentally different from the prompting strategy for gap analysis (Impact Analyst). Combining them in one system prompt degrades both.

4. **Parallelization:** The Policy Mapper and Conflict Detector can run simultaneously once parsing is complete. A single agent executes sequentially.

### When Multi-Agent Is NOT Justified (Honesty Check)

For simple regulatory updates that affect a single, obvious policy area (e.g., "FINRA updates the margin requirement percentage from 25% to 30%"), a single agent could handle the analysis competently. The multi-agent architecture shows its value on complex, cross-domain updates. The evaluation framework should measure this explicitly.

---

# 7. System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE                            │
│            (Streamlit / React Frontend)                      │
│                                                             │
│  Input: Regulatory URL, pasted text, or uploaded PDF        │
│  Output: Structured impact report with citations            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI BACKEND                           │
│                                                             │
│  POST /api/analyze     — Submit new regulatory analysis     │
│  GET  /api/analysis/{id} — Get analysis status/results      │
│  POST /api/feedback    — Submit human feedback              │
│  GET  /api/health      — Health check                       │
│  GET  /api/metrics     — Prometheus metrics                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              ORCHESTRATOR (LangGraph Supervisor)             │
│                                                             │
│  - Manages agent execution order and state                  │
│  - Handles parallel execution (Policy Mapper ‖ Conflict     │
│    Detector)                                                │
│  - Implements human-in-the-loop checkpoint                  │
│  - Enforces iteration limits and cost ceilings              │
│  - Logs every decision for observability                    │
│                                                             │
│  State: RegDeltaState (shared across all agents)            │
│  Checkpointing: SQLite / PostgreSQL persistence             │
└──────┬──────────┬──────────┬──────────┬─────────┬───────────┘
       │          │          │          │         │
       ▼          ▼          ▼          ▼         ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│REGULATORY│ │ POLICY   │ │ CONFLICT │ │ IMPACT   │ │ REPORT   │
│ PARSER   │ │ MAPPER   │ │ DETECTOR │ │ ANALYST  │ │GENERATOR │
│  Agent   │ │  Agent   │ │  Agent   │ │  Agent   │ │  Agent   │
└────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
     │            │            │            │            │
     ▼            ▼            ▼            ▼            ▼
┌─────────────────────────────────────────────────────────────┐
│                    TOOL LAYER                                │
│                                                             │
│  Federal Register API    Internal Policy Vector Store       │
│  eCFR API                BM25 Index (Elasticsearch)         │
│  SEC EDGAR API           Regulatory Reference Graph (Neo4j) │
│  Web Scraper             Severity Scoring Schema            │
│  PDF Parser              Citation Formatter                 │
│  Structured Extractor    Report Template Engine             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                 DATA & INFRASTRUCTURE                        │
│                                                             │
│  PostgreSQL — Analysis history, metadata, user feedback     │
│  Qdrant     — Vector embeddings for policies & regulations  │
│  Redis      — Semantic cache, embedding cache               │
│  Neo4j      — Regulatory cross-reference graph (Phase 8)    │
│  LangSmith/Langfuse — Tracing, observability                │
│  Prometheus + Grafana — Metrics and dashboards              │
└─────────────────────────────────────────────────────────────┘
```

## Execution Flow

```
1. User submits regulatory text (URL, paste, or PDF upload)
          │
          ▼
2. Orchestrator receives input, initializes RegDeltaState
          │
          ▼
3. Regulatory Parser Agent
   ├── Fetches full regulatory text (Federal Register API or PDF parser)
   ├── Extracts: requirements[], affected_entities[], effective_dates[],
   │   definitions[], cross_references[], exemptions[]
   ├── Validates extraction (checks for required fields)
   └── Returns structured ParsedRegulation object
          │
          ▼
4. Orchestrator routes to parallel execution:
   ┌──────────────────────────┬──────────────────────────┐
   │                          │                          │
   ▼                          ▼                          │
5a. Policy Mapper Agent    5b. Conflict Detector Agent   │
   ├── For each requirement:  ├── For each requirement:  │
   │   ├── Generate queries   │   ├── Search regulatory  │
   │   │   (original +        │   │   corpus for related  │
   │   │    rewritten)        │   │   obligations         │
   │   ├── Hybrid retrieval   │   ├── Compare new req     │
   │   │   (dense + BM25)     │   │   against existing    │
   │   ├── Rerank results     │   │   obligations         │
   │   ├── Assess sufficiency │   ├── Identify conflicts  │
   │   │   (enough policies   │   │   or overlaps         │
   │   │    found?)           │   └── Return conflicts[]  │
   │   ├── Retry if needed    │                          │
   │   └── Return matched     │                          │
   │       policies[]         │                          │
   └──────────────────────────┴──────────────────────────┘
          │
          ▼
6. Impact Analyst Agent
   ├── For each requirement + matched policies:
   │   ├── Compare regulatory requirement against policy provisions
   │   ├── Identify gaps (what the regulation requires but policy doesn't address)
   │   ├── Assess severity (Critical / High / Medium / Low)
   │   ├── Consider conflicts detected by Conflict Detector
   │   └── Generate recommended actions
   └── Returns impact_assessments[]
          │
          ▼
7. ═══ HUMAN-IN-THE-LOOP CHECKPOINT ═══
   ├── Present preliminary findings to compliance officer
   ├── Officer can: approve, modify, request deeper analysis, or reject
   └── Analysis resumes after human input (LangGraph checkpoint/resume)
          │
          ▼
8. Report Generator Agent
   ├── Compiles all findings into structured report
   ├── Formats citations (regulatory section → policy section)
   ├── Generates executive summary
   ├── Creates action item list with owners and deadlines
   └── Returns final ComplianceImpactReport
          │
          ▼
9. Report stored in PostgreSQL, served to frontend
```

## Shared State Schema (RegDeltaState)

```python
from typing import TypedDict, Annotated, Literal
from operator import add
from langgraph.graph import MessagesState

class ParsedRequirement(TypedDict):
    id: str
    text: str
    requirement_type: Literal["prohibition", "obligation", "disclosure", "reporting", "procedural"]
    affected_entities: list[str]
    effective_date: str | None
    cross_references: list[str]
    exemptions: list[str]

class MatchedPolicy(TypedDict):
    requirement_id: str
    policy_id: str
    policy_title: str
    relevant_sections: list[str]
    relevance_score: float
    match_type: Literal["direct", "indirect", "potential"]

class RegulatoryConflict(TypedDict):
    requirement_id: str
    conflicting_regulation: str
    conflict_type: Literal["contradiction", "overlap", "ambiguity"]
    description: str
    severity: Literal["critical", "high", "medium", "low"]

class ImpactAssessment(TypedDict):
    requirement_id: str
    affected_policies: list[str]
    gap_description: str
    severity: Literal["critical", "high", "medium", "low"]
    recommended_actions: list[str]
    confidence: float
    citations: list[dict]

class RegDeltaState(MessagesState):
    # Input
    regulatory_input: str                    # URL, text, or file path
    input_type: Literal["url", "text", "pdf"]

    # Phase 1: Parsed regulation
    parsed_regulation: dict | None           # Full parsed regulation metadata
    requirements: list[ParsedRequirement]    # Extracted requirements

    # Phase 2: Policy mapping
    matched_policies: Annotated[list[MatchedPolicy], add]
    retrieval_sufficient: bool

    # Phase 3: Conflict detection
    conflicts: Annotated[list[RegulatoryConflict], add]

    # Phase 4: Impact analysis
    impact_assessments: list[ImpactAssessment]

    # Phase 5: Report
    final_report: dict | None

    # Metadata
    analysis_id: str
    status: Literal["parsing", "mapping", "detecting", "analyzing", "review", "reporting", "complete", "failed"]
    errors: Annotated[list[str], add]
    total_tokens_used: int
    total_cost_usd: float
```

---

# 8. RAG Architecture (Deep Dive)

## Retrieval Strategy Overview

RegDelta uses different retrieval strategies for different data sources because each source has different characteristics:

| Data Source | Retrieval Method | Why |
|---|---|---|
| Internal policies | Hybrid (dense + BM25) + cross-encoder reranking | Policies contain both precise legal terms (BM25) and semantic concepts (dense). Reranking critical because many policy sections sound similar but have different legal implications. |
| Federal Register / eCFR | API-first, then vector search for related regulations | New regulations are best retrieved via their document number or citation. Related regulations are found via semantic search. |
| Regulatory cross-references | Graph traversal (Phase 8: Neo4j) or citation parsing | Cross-references are structural (Section X cites Section Y), not semantic. Graph traversal is the natural retrieval method. |
| Historical analyses | Metadata-filtered vector search | Retrieve past analyses by regulator, domain, date range. |

## Internal Policy Retrieval Pipeline

This is the most architecturally complex retrieval in the system.

### Ingestion (Offline)

```
Raw Policy Documents (PDF, DOCX, MD)
        │
        ▼
   Document Loader
   (extract text, preserve structure: sections, subsections, headers)
        │
        ▼
   Semantic Chunker
   ├── Split on section boundaries (not fixed token count)
   ├── Preserve section hierarchy (Chapter > Section > Subsection)
   ├── Target: 300-500 tokens per chunk
   ├── Overlap: 50 tokens at section boundaries
   └── Each chunk retains: document_id, section_path, page_number
        │
        ▼
   Contextual Enrichment
   ├── For each chunk, generate a 1-2 sentence context summary
   │   using an LLM: "This chunk is from [document title],
   │   [section path], discussing [topic summary]."
   ├── Prepend context to chunk text before embedding
   └── Store both original chunk and contextualized chunk
        │
        ▼
   Embedding Generation
   ├── Model: Qwen3-Embedding-0.6B (local) or text-embedding-3-large (API)
   ├── Dimension: 1024 (Qwen3) or 3072 (OpenAI, can be reduced to 1024)
   └── Batch processing with progress tracking
        │
        ▼
   Dual Indexing
   ├── Vector Index: Qdrant collection with metadata payload
   │   Metadata: {document_id, section_path, regulator, jurisdiction,
   │              policy_domain, effective_date, version, chunk_index}
   │
   └── BM25 Index: Elasticsearch or rank_bm25 (Python library)
       Fields: original_text, section_path, document_title
```

### Retrieval (Online — per query)

```
Incoming Query (from Policy Mapper Agent)
e.g., "internal policies implementing SEC climate disclosure requirements"
        │
        ▼
   Query Rewriting
   ├── Original query retained
   ├── LLM generates 2-3 reformulations:
   │   - "ESG reporting policy climate risk disclosure"
   │   - "environmental sustainability reporting requirements SEC"
   │   - "greenhouse gas emissions disclosure internal controls"
   └── All queries executed in parallel
        │
        ▼
   ┌────────────────────┬────────────────────┐
   │  Dense Retrieval    │  Sparse Retrieval   │
   │  (Qdrant)           │  (BM25)             │
   │                     │                     │
   │  top_k = 20         │  top_k = 20         │
   │  per query variant  │  per query variant  │
   │                     │                     │
   │  + metadata filters:│  + field filters:   │
   │  policy_domain IN   │  document_title,    │
   │  [relevant domains] │  section_path       │
   └────────┬───────────┴────────┬───────────┘
            │                    │
            ▼                    ▼
   Reciprocal Rank Fusion (RRF)
   ├── Merge ranked lists from dense + sparse
   ├── RRF formula: score = Σ 1/(k + rank_i)  where k = 60
   ├── Deduplicate by chunk_id
   └── Output: top 30-40 candidate chunks
            │
            ▼
   Cross-Encoder Reranking
   ├── Model: BGE-reranker-v2-m3 or Qwen3-Reranker-0.6B
   ├── Score each (query, chunk) pair
   ├── Re-sort by cross-encoder score
   └── Output: top 10 chunks with relevance scores
            │
            ▼
   Sufficiency Check (Agent Decision)
   ├── Are there chunks from at least N distinct policies?
   ├── Do the top results cover the regulatory domains identified
   │   in the parsed requirements?
   ├── Is the minimum relevance score above threshold (0.6)?
   │
   ├── If sufficient → return results
   └── If insufficient → reformulate query → retry (max 3 attempts)
```

## Chunking Strategy: Why Semantic Over Fixed-Size

Fixed-size chunking (e.g., 500 tokens) is inappropriate for policy documents because:

1. **It splits mid-section.** A policy provision about "Prohibited Activities" might be split between two chunks, making neither chunk independently useful.
2. **It separates conditions from consequences.** "If the counterparty is located in a sanctioned jurisdiction (defined in Section 3.2), then the following enhanced due diligence procedures apply: [procedures]" — fixed chunking might put the condition in one chunk and the procedures in another.
3. **It loses hierarchy.** Which chapter and section a provision belongs to is critical metadata for compliance analysis.

**Semantic chunking splits on section boundaries**, keeping each section as a coherent unit. Sections that are too long (>800 tokens) are split at paragraph boundaries within the section.

## Contextual Chunking: Why It Matters

Following Anthropic's contextual retrieval approach, each chunk is prepended with LLM-generated context before embedding. This reduces retrieval failures by 35-49% in benchmarks.

**Without context:**
```
Chunk: "The covered entity shall submit quarterly reports to the Commission
no later than 45 calendar days after the end of each fiscal quarter."
```
This chunk is about quarterly reporting, but it lacks critical context: reporting about what? under which regulation? for which type of entity?

**With context:**
```
Context: "This chunk is from the 'Periodic Reporting Requirements' section
(Section 7.3) of the Internal Securities Reporting Policy, which implements
SEC Rule 13a-13 quarterly reporting obligations for registered investment
advisers."

Chunk: "The covered entity shall submit quarterly reports to the Commission
no later than 45 calendar days after the end of each fiscal quarter."
```

Now the retriever can match this chunk to queries about "SEC quarterly reporting for investment advisers" even if the chunk text doesn't contain those terms.

---

# 9. Agent Architecture (Deep Dive)

## Agent 1: Regulatory Parser

**Responsibility:** Transform raw regulatory text into structured data.

**Input:** Raw regulatory text (from URL, paste, or PDF)

**Output:** `ParsedRegulation` object with structured fields

**Tools:**
- `fetch_federal_register(document_number)` — Fetch full text via Federal Register API
- `fetch_ecfr_section(title, part, section)` — Fetch current CFR text
- `parse_pdf(file_path)` — Extract text from uploaded PDF
- `extract_structured_regulation(text)` — LLM-powered structured extraction

**Prompting strategy:** System prompt optimized for legal text extraction with structured output (JSON schema). Few-shot examples of parsed regulations.

**Model choice:** Claude Haiku 4.5 or GPT-4o-mini (extraction is simpler than reasoning; cheaper model suffices)

**Failure modes:**
- Regulatory text is poorly formatted (OCR artifacts in older PDFs)
- Ambiguous requirements that could be interpreted multiple ways
- Cross-references to regulations not in the database

**Termination criteria:** Returns when all required fields are populated and validated, or after 2 retry attempts.

## Agent 2: Policy Mapper

**Responsibility:** Find internal policies affected by each parsed requirement.

**Input:** `requirements[]` from Regulatory Parser

**Output:** `matched_policies[]` with relevance scores

**Tools:**
- `hybrid_search(query, filters)` — Hybrid dense+BM25 search over policy store
- `rerank(query, documents)` — Cross-encoder reranking
- `rewrite_query(query)` — LLM-powered query reformulation
- `get_policy_metadata(policy_id)` — Fetch policy metadata from PostgreSQL

**Prompting strategy:** System prompt focused on relevance assessment. The agent must judge whether retrieved policies *actually* address the regulatory requirement or are merely semantically similar.

**Model choice:** Claude Sonnet 4.6 or GPT-4o (requires judgment about relevance)

**Retrieval strategy:** Full hybrid pipeline described in Section 8.

**Iterative behavior:**
1. Search with original query
2. If < 3 relevant policies found, rewrite query and search again
3. If still insufficient, broaden metadata filters and search again
4. Maximum 3 retrieval iterations per requirement

**Termination criteria:** At least 1 policy matched per requirement with relevance score > 0.6, OR 3 retrieval attempts exhausted (flags as "low coverage — human review needed").

## Agent 3: Conflict Detector

**Responsibility:** Identify conflicts between new requirements and existing regulatory obligations.

**Input:** `requirements[]` from Regulatory Parser

**Output:** `conflicts[]` with conflict type and severity

**Tools:**
- `search_regulatory_corpus(query)` — Search existing regulatory text for related obligations
- `compare_requirements(new_req, existing_req)` — LLM-powered comparison for conflicts
- `traverse_regulatory_graph(regulation_id)` — (Phase 8) Traverse knowledge graph for related regulations

**Prompting strategy:** System prompt focused on identifying contradictions, overlaps, and ambiguities between regulatory requirements. Explicit instructions to distinguish between true conflicts (cannot comply with both simultaneously) and overlaps (both require similar but not identical actions).

**Model choice:** Claude Sonnet 4.6 or GPT-4o (requires careful reasoning about legal compatibility)

**Parallel execution:** Runs simultaneously with Policy Mapper.

**Termination criteria:** All requirements checked against the regulatory corpus, or timeout reached.

## Agent 4: Impact Analyst

**Responsibility:** Perform gap analysis and severity assessment.

**Input:** `requirements[]` + `matched_policies[]` + `conflicts[]`

**Output:** `impact_assessments[]` with severity, gaps, recommended actions, and citations

**Tools:**
- `compare_requirement_to_policy(requirement, policy_section)` — LLM-powered gap analysis
- `score_severity(gap_description, requirement_type)` — Severity scoring based on rubric
- `generate_recommendations(gap, requirement)` — Generate specific action items

**Prompting strategy:** The most complex prompt in the system. Includes a severity scoring rubric:
- **Critical:** Regulation requires something the policy explicitly prohibits, or policy is entirely silent on a required obligation
- **High:** Significant gaps in policy coverage that require substantial revision
- **Medium:** Policy addresses the topic but needs updates to align with new specifics
- **Low:** Minor wording or procedural changes needed

**Model choice:** Claude Sonnet 4.6 (requires the strongest reasoning; this is where model quality matters most)

**Self-verification:** After generating each assessment, the agent re-reads the cited regulatory text and policy text to verify the gap analysis is supported by the evidence.

**Termination criteria:** All matched requirement-policy pairs assessed.

## Agent 5: Report Generator

**Responsibility:** Compile findings into a structured compliance impact report.

**Input:** All upstream outputs

**Output:** `ComplianceImpactReport` with executive summary, detailed findings, citations, and action items

**Tools:**
- `format_citation(source, section, text)` — Standardize citation format
- `generate_executive_summary(assessments)` — LLM-powered summary
- `create_action_items(assessments)` — Generate prioritized action list

**Model choice:** Claude Haiku 4.5 or GPT-4o-mini (synthesis from existing analysis; cheaper model sufficient)

**Termination criteria:** Report generated with all required sections populated.

## Orchestrator Configuration

```python
# LangGraph supervisor pattern (conceptual)
from langgraph.graph import StateGraph, END

workflow = StateGraph(RegDeltaState)

# Add agent nodes
workflow.add_node("regulatory_parser", regulatory_parser_node)
workflow.add_node("policy_mapper", policy_mapper_node)
workflow.add_node("conflict_detector", conflict_detector_node)
workflow.add_node("impact_analyst", impact_analyst_node)
workflow.add_node("human_review", human_review_node)
workflow.add_node("report_generator", report_generator_node)

# Define edges
workflow.add_edge("regulatory_parser", "parallel_analysis")

# Parallel execution: Policy Mapper and Conflict Detector run simultaneously
workflow.add_node("parallel_analysis", fan_out_to_parallel)
workflow.add_edge("parallel_analysis", "impact_analyst")

workflow.add_edge("impact_analyst", "human_review")

# Human-in-the-loop: conditional edge based on human decision
workflow.add_conditional_edges(
    "human_review",
    route_after_review,
    {
        "approve": "report_generator",
        "modify": "impact_analyst",  # Re-analyze with human corrections
        "reject": END,
    }
)

workflow.add_edge("report_generator", END)

# Set entry point
workflow.set_entry_point("regulatory_parser")
```

---

# 10. Data Sources and Datasets

## External Regulatory Data (Publicly Available, Free)

### Federal Register API
- **URL:** `https://www.federalregister.gov/api/v1/`
- **Coverage:** All US federal regulations since 1994
- **Format:** JSON (structured metadata + full text in HTML)
- **No API key required**
- **Key endpoints:**
  - `GET /documents/{document_number}` — Single document
  - `GET /documents?conditions[agencies][]={agency}&conditions[type][]={type}` — Search
  - Fields: title, abstract, full_text_xml_url, agencies, cfr_references, effective_on, publication_date
- **Python library:** `fr-toolbelt` (`pip install fr-toolbelt`)

### Electronic Code of Federal Regulations (eCFR)
- **URL:** `https://www.ecfr.gov/api/v1/`
- **Coverage:** Current text of all federal regulations
- **No API key required**
- **Key endpoints:**
  - `GET /full/{date}/title-{title}.xml` — Full title text
  - `GET /versioner/v1/versions/title-{title}` — Version history

### SEC EDGAR
- **URL:** `https://efts.sec.gov/LATEST/search-index` (full-text search)
- **URL:** `https://data.sec.gov/submissions/CIK{number}.json` (structured filings)
- **Coverage:** All SEC filings since 1993
- **No API key required** (but requires User-Agent header with your name and email)
- **Rate limit:** 10 requests/second
- **Useful for:** SEC rules, enforcement actions, no-action letters, interpretive releases

### FINRA Regulatory Notices
- **URL:** `https://www.finra.org/rules-guidance/notices` (web scraping required)
- **Coverage:** FINRA regulatory notices, guidance, and rule amendments

### Regulations.gov
- **URL:** `https://api.regulations.gov/v4/` 
- **API key required** (free, from api.data.gov)
- **Coverage:** Proposed and final rules with public comments

## Internal Policy Data (Synthetic — You Will Create This)

Since real internal compliance policies are proprietary, you will generate a realistic synthetic policy corpus. This is acceptable because:
1. The interesting part is the *architecture* (multi-hop retrieval, conflict detection, impact analysis), not the specific policy content
2. Synthetic policies that reference real regulations create realistic retrieval challenges
3. Many publicly available compliance frameworks (NIST, SOX, Basel III) provide templates

### Synthetic Policy Corpus Specification

**Target:** 200-500 synthetic internal policies across these domains:

| Domain | Count | Example Policies |
|--------|:-----:|-----------------|
| Securities Regulation | 40-60 | Trading compliance, insider trading prevention, market manipulation controls |
| Anti-Money Laundering | 30-50 | KYC procedures, suspicious activity reporting, sanctions screening |
| Consumer Protection | 30-50 | Fair lending, UDAAP compliance, complaint handling |
| Data Privacy | 20-30 | Data retention, breach notification, customer consent |
| Risk Management | 30-40 | Enterprise risk framework, stress testing, model risk |
| Corporate Governance | 20-30 | Board oversight, conflicts of interest, whistleblower |
| Financial Reporting | 20-30 | Internal controls (SOX), financial statement preparation |
| Capital & Liquidity | 20-30 | Capital adequacy (Basel III), liquidity risk management |
| Technology & Cyber | 20-30 | Information security, vendor management, business continuity |
| Operational Risk | 20-30 | Incident management, fraud prevention, outsourcing |

**Each policy should include:**
- Title and document ID
- Effective date and version number
- Responsible department/owner
- Regulatory references (cite specific CFR sections, SEC rules, FINRA rules)
- Scope and applicability
- Definitions section with defined terms
- Requirements/procedures (the substantive content)
- Exceptions and exemptions
- Related policies (cross-references to other internal policies)

**Generation approach:** Use an LLM to generate realistic policies. Provide it with:
- A real regulation as input (from Federal Register)
- A template structure (sections listed above)
- Instructions to reference specific CFR/SEC rule sections
- Instructions to include defined terms that override common meanings

See Appendix B for the detailed generation prompt.

## Evaluation Dataset (You Will Create This)

**Gold-Standard Test Set:** 50-100 regulatory change scenarios with known correct answers.

**Structure of each test case:**

```json
{
    "test_id": "TC-001",
    "regulatory_update": {
        "document_number": "2024-12345",
        "title": "Climate-Related Disclosures for Investors",
        "regulator": "SEC",
        "type": "final_rule",
        "summary": "Requires registrants to disclose climate-related risks...",
        "source_url": "https://www.federalregister.gov/d/2024-12345"
    },
    "expected_affected_policies": [
        {"policy_id": "POL-ESG-001", "section": "Section 4.2", "severity": "high"},
        {"policy_id": "POL-RISK-012", "section": "Section 7.1", "severity": "medium"},
        {"policy_id": "POL-GOV-005", "section": "Section 3.4", "severity": "low"}
    ],
    "expected_conflicts": [
        {"conflicting_regulation": "State AG ESG Anti-Boycott Rule", "conflict_type": "contradiction"}
    ],
    "expected_not_affected": ["POL-AML-001", "POL-CYBER-003"],
    "difficulty": "hard",
    "multi_hop_required": true,
    "domains_spanned": ["ESG", "Risk Management", "Corporate Governance"]
}
```

**Creating the test set:**
1. Select 50-100 real regulatory updates from the Federal Register (mix of SEC, FINRA, CFPB, OCC)
2. For each update, manually determine which synthetic policies should be affected
3. Annotate severity, affected sections, and expected conflicts
4. Include negative examples (policies that should NOT be flagged)
5. Tag difficulty level and whether multi-hop retrieval is required

---

# 11. LLM and Model Choices

## Model Selection Strategy

Use the cheapest model that performs adequately for each task. Reserve expensive models for reasoning-intensive components.

### Recommended Configuration

| Component | Model | Cost (per 1M tokens) | Why This Model |
|---|---|---|---|
| Regulatory Parser | Claude Haiku 4.5 | In: $0.80, Out: $4.00 | Structured extraction doesn't need frontier reasoning; Haiku handles JSON extraction well |
| Policy Mapper (relevance judgment) | Claude Sonnet 4.6 | In: $3.00, Out: $15.00 | Needs judgment about whether a policy *actually* addresses a requirement vs. being superficially similar |
| Conflict Detector | Claude Sonnet 4.6 | In: $3.00, Out: $15.00 | Reasoning about legal compatibility requires strong inference |
| Impact Analyst | Claude Sonnet 4.6 | In: $3.00, Out: $15.00 | Most reasoning-intensive component; gap analysis requires careful comparison and judgment |
| Report Generator | Claude Haiku 4.5 | In: $0.80, Out: $4.00 | Synthesis from existing analysis; cheaper model sufficient |
| Query Rewriting | Claude Haiku 4.5 | In: $0.80, Out: $4.00 | Query reformulation is straightforward |
| Contextual Chunk Enrichment (offline) | GPT-4o-mini | In: $0.15, Out: $0.60 | Run once per chunk during ingestion; cheapest option for context generation |
| Embeddings | OpenAI text-embedding-3-large | $0.13 per 1M tokens | Good quality, easy integration, well-documented |
| Embeddings (cost-optimized) | Qwen3-Embedding-0.6B (local) | Free (GPU cost) | Leads MTEB benchmarks, zero per-token cost, requires GPU |
| Reranking | BGE-reranker-v2-m3 (local) | Free (GPU cost) | Strong cross-encoder, runs locally |

### Cost Estimate Per Analysis

| Step | Tokens (approx.) | Model | Cost |
|------|---|---|---|
| Parse regulation (50-page rule) | ~30K in, ~5K out | Haiku | $0.04 |
| Query rewriting (5 requirements × 3 variants) | ~5K in, ~2K out | Haiku | $0.01 |
| Policy Mapper relevance assessment | ~50K in, ~10K out | Sonnet | $0.30 |
| Conflict detection | ~30K in, ~5K out | Sonnet | $0.17 |
| Impact analysis | ~40K in, ~15K out | Sonnet | $0.35 |
| Report generation | ~20K in, ~10K out | Haiku | $0.06 |
| **Total per analysis** | **~175K in, ~47K out** | | **~$0.93** |

This is approximately $1 per regulatory analysis — extremely cost-effective compared to $5,000-15,000 for manual analysis.

### Open-Source Alternative Stack (Lower Cost, Higher Setup Effort)

| Component | Model | Notes |
|---|---|---|
| Main reasoning | Llama 3.1 70B or Qwen2.5 72B (via Ollama or vLLM) | Requires 40GB+ VRAM or quantized version |
| Parsing/routing | Llama 3.1 8B or Qwen2.5 7B | Runs on consumer GPU |
| Embeddings | Qwen3-Embedding-0.6B | Runs on consumer GPU |
| Reranking | BGE-reranker-v2-m3 | Runs on consumer GPU |

---

# 12. Technology Stack

## Complete Stack with Justifications

| Layer | Technology | Version | Why This Choice |
|---|---|---|---|
| **Language** | Python | 3.11+ | Standard for AI/ML; async support for parallel agents |
| **API Framework** | FastAPI | 0.110+ | Async, typed, auto-documentation, production-proven |
| **Agent Orchestration** | LangGraph | 0.2+ | Production-grade state management, checkpointing, human-in-the-loop. Largest production footprint among agent frameworks. |
| **RAG Framework** | LlamaIndex | 0.10+ | Best document loaders (PDF, DOCX); clean ingestion pipeline. Use for ingestion only; custom retrieval logic. |
| **Vector Database** | Qdrant | 1.8+ | Advanced metadata filtering, payload indexing, hybrid search support. Runs in Docker. |
| **BM25 Search** | rank_bm25 (Python) or Elasticsearch | — | rank_bm25 for simplicity (in-memory); Elasticsearch if you need persistence and scale |
| **Structured Database** | PostgreSQL | 16+ | Analysis history, policy metadata, evaluation results, user feedback. Use pgvector extension if you want vectors in Postgres. |
| **Cache** | Redis | 7+ | Semantic cache for repeated queries; embedding cache to avoid redundant API calls |
| **Graph Database** | Neo4j (Phase 8 only) | 5+ | Regulatory cross-reference graph. Only add when demonstrating GraphRAG experiment. |
| **Observability** | Langfuse (open-source) or LangSmith | — | Langfuse is self-hostable and free. LangSmith is easier but requires LangChain account. Trace every LLM call, retrieval, agent step. |
| **Metrics** | Prometheus + Grafana | — | Track latency, cost, retrieval quality, agent performance over time |
| **Evaluation** | DeepEval + custom metrics | — | Task completion, faithfulness, citation correctness, trajectory quality |
| **Testing** | pytest + pytest-asyncio | — | Unit tests for components, integration tests for pipeline |
| **Containerization** | Docker + Docker Compose | — | All infrastructure (Qdrant, Redis, PostgreSQL, Prometheus, Grafana) runs in containers |
| **Frontend** | Streamlit | — | Rapid prototyping for demo. Not the focus of the project. |
| **Version Control** | Git + GitHub | — | Standard |

### What I Am NOT Including (and Why)

| Technology | Why Excluded |
|---|---|
| **MCP** | RegDelta is a self-contained system, not integrating with an external tool ecosystem. MCP adds protocol overhead without solving a real problem here. |
| **Kubernetes** | Overkill for a portfolio project. Docker Compose is sufficient. |
| **Kafka / message queues** | The analysis workflow is request-response, not event-driven. No need for async message passing. |
| **Pinecone / Weaviate** | Qdrant is free, self-hosted, and has superior metadata filtering. No reason to use a managed service for a portfolio project. |
| **LangChain (core)** | Using LangGraph for orchestration and LlamaIndex for ingestion. LangChain core adds abstraction without clear benefit here. |
| **Fine-tuning** | Not justified until evaluation shows that prompting is the bottleneck. Start with prompt engineering. |

---

# 13. Evaluation Framework

## Evaluation Tiers

### Tier 1: Retrieval Evaluation

Measures whether the system retrieves the right documents.

| Metric | What It Measures | Target | How to Compute |
|---|---|---|---|
| Recall@10 | Of all relevant policies, what fraction is in the top 10 results? | > 0.8 | Gold-standard test set with known relevant policies |
| Precision@5 | Of the top 5 results, what fraction is actually relevant? | > 0.6 | Manual annotation or LLM-as-judge |
| MRR (Mean Reciprocal Rank) | How high is the first relevant result? | > 0.7 | 1/rank of first relevant result, averaged |
| NDCG@10 | Quality of the full ranking | > 0.7 | Discounted cumulative gain |
| Multi-hop success rate | For questions requiring chained retrieval, does the system complete the full chain? | > 0.6 | Test cases tagged as multi-hop |

### Tier 2: RAG / Generation Evaluation

Measures the quality of the generated analysis.

| Metric | What It Measures | Target | How to Compute |
|---|---|---|---|
| Faithfulness | Are claims in the analysis supported by retrieved documents? | > 0.85 | LLM-as-judge: "Is this claim supported by the provided context?" |
| Citation correctness | Do citations point to text that actually supports the claim? | > 0.90 | Extract cited text, compare against claim |
| Answer relevance | Does the analysis address the regulatory change? | > 0.85 | LLM-as-judge |
| Hallucination rate | Percentage of claims not grounded in evidence | < 0.10 | Inverse of faithfulness on claim level |
| Completeness | Are all affected policies identified? | > 0.75 | Compare against gold-standard affected policies |

### Tier 3: Agent Evaluation

Measures agent behavior quality.

| Metric | What It Measures | Target | How to Compute |
|---|---|---|---|
| Task success rate | Percentage of analyses completed successfully | > 0.85 | Binary: did the system produce a complete report? |
| Tool selection accuracy | Did agents use the right tools? | > 0.90 | Compare tool calls against expected tools per test case |
| Trajectory efficiency | Number of steps vs. minimum necessary | < 2x optimal | Count LLM calls, tool calls, retrieval operations |
| Failure recovery rate | When a tool call fails, does the system recover? | > 0.80 | Inject tool failures and measure recovery |
| Cost per analysis | Total tokens × price | < $2.00 | Sum all LLM calls |
| Latency | End-to-end time per analysis | < 5 min | Measure wall-clock time |

### Tier 4: Domain-Specific Evaluation

Measures compliance-specific quality.

| Metric | What It Measures | Target | How to Compute |
|---|---|---|---|
| Severity accuracy | Does the system correctly classify impact severity? | > 0.70 | Compare predicted severity against gold-standard |
| False positive rate | Policies flagged as affected that should not be | < 0.20 | Count false alarms in test set |
| False negative rate | Affected policies that were missed | < 0.10 | Count misses in test set (more dangerous than false positives) |
| Conflict detection F1 | Precision and recall of conflict identification | > 0.60 | Compare against annotated conflicts |
| Cross-reference resolution accuracy | Are regulatory cross-references correctly followed? | > 0.80 | Test cases with known cross-reference chains |

## Evaluation Pipeline

```
1. Load test cases from data/eval/test_cases.json
2. For each test case:
   a. Run full RegDelta pipeline
   b. Collect: retrieved policies, generated analysis, citations, severity scores
   c. Compare against gold-standard expected results
   d. Compute all metrics
   e. Log trace to Langfuse/LangSmith for trajectory analysis
3. Aggregate metrics across all test cases
4. Generate evaluation report with per-metric scores, per-category breakdowns,
   and failure analysis
5. Store results in PostgreSQL for historical comparison
```

---

# 14. Baselines and Ablation Plan

## Progressive Baselines

Each baseline adds one architectural capability, allowing you to measure the marginal value of each component.

### Baseline 1: Vanilla LLM (No Retrieval)

```
Input: Full regulatory text → LLM → "What is the impact?"
```
- No retrieval, no internal policies
- Expected result: Vague, generic analysis. No policy-specific insights. No citations.
- Purpose: Establishes the floor — how much can a frontier LLM do without any supporting architecture?

### Baseline 2: Naive RAG

```
Input: Query → Dense retrieval (top 5) → LLM → Answer
```
- Single dense embedding search over all documents (regulatory + policies mixed together)
- Fixed top-k = 5, no reranking, no query rewriting, no metadata filtering
- Single-pass generation
- Expected result: Retrieves some relevant documents but with poor precision. Misses multi-hop relationships. May retrieve regulatory text instead of policy text.

### Baseline 3: Advanced RAG

```
Input: Query → Hybrid retrieval (dense + BM25) → RRF → Reranking → LLM → Answer
```
- Hybrid search with RRF fusion
- Cross-encoder reranking
- Query rewriting (3 variants)
- Metadata filtering (separate regulatory and policy collections)
- Contextual chunking
- Still single-pass generation (no iterative retrieval)
- Expected result: Significantly better retrieval. Still misses multi-hop reasoning chains.

### Baseline 4: Agentic RAG (Single Agent)

```
Input: Query → Single agent with tools → Iterative retrieval → Self-assessment → Answer
```
- One agent with all tools (regulatory API, policy search, conflict detection)
- Iterative retrieval with sufficiency checking
- Query decomposition
- Self-correction (re-reads sources to verify claims)
- Expected result: Handles multi-hop better. May struggle with long analyses due to context pressure.

### Baseline 5: Multi-Agent System (Full RegDelta)

```
Input → Orchestrator → Parser → (Mapper ‖ Conflict Detector) → Analyst → Report
```
- Full architecture as described
- Expected result: Best quality on complex, cross-domain regulatory changes. The improvement over Baseline 4 should be most visible on analyses that span multiple regulatory domains.

### What to Measure Across Baselines

| Metric | B1 | B2 | B3 | B4 | B5 |
|---|---|---|---|---|---|
| Policy identification recall | Very low | Low | Medium | High | Highest |
| Citation correctness | N/A | Low | Medium | High | Highest |
| Multi-hop success | N/A | Very low | Low | Medium | High |
| Hallucination rate | Very high | High | Medium | Low | Lowest |
| Cost per analysis | Lowest | Low | Medium | High | Highest |
| Latency | Fastest | Fast | Medium | Slow | Slowest |

The key insight to demonstrate: **each architectural upgrade improves quality metrics but increases cost and latency. The question is whether the quality improvement justifies the added complexity.**

---

# 15. Research Questions

These are questions you can investigate experimentally, turning RegDelta from an engineering project into a research contribution.

### RQ1: Does GraphRAG outperform vector retrieval for cross-regulatory conflict detection?

**Hypothesis:** Graph traversal catches regulatory conflicts that embedding similarity misses, because conflicts often involve structurally related but semantically distant regulations.

**Experiment:**
- Build a regulatory knowledge graph in Neo4j (regulations as nodes, citations/amendments as edges)
- Implement graph-based retrieval for the Conflict Detector
- Compare conflict detection F1 between vector-only and graph-augmented retrieval
- Measure: F1 score, precision, recall on conflict detection test set
- Also measure: indexing cost, query latency, maintenance overhead

### RQ2: Does multi-agent decomposition reduce hallucination compared to single-agent?

**Hypothesis:** Specialized agents with narrower scope produce more grounded outputs because each agent has fewer tools and a more focused prompt, reducing the chance of confusion.

**Experiment:**
- Run the same 50 test cases with Baseline 4 (single agent) and Baseline 5 (multi-agent)
- Measure: faithfulness score, hallucination rate, citation correctness
- Control for: total tokens consumed (multi-agent uses more tokens; ensure the comparison is fair)
- Also measure: cost difference — is the quality improvement worth the cost increase?

### RQ3: What is the minimum retrieval recall required for reliable compliance analysis?

**Hypothesis:** There is a retrieval recall threshold below which the system's impact assessments become unreliable, and this threshold is higher for compliance than for general Q&A (because missing a relevant policy is more dangerous than missing a relevant Wikipedia paragraph).

**Experiment:**
- Artificially degrade retrieval quality by randomly removing correct results at varying rates
- Measure how impact assessment accuracy degrades as retrieval recall decreases
- Plot the accuracy-recall curve
- Identify the "cliff" where accuracy drops sharply

---

# 16. Failure Mode Analysis

| # | Failure Mode | Likelihood | Impact | Detection | Mitigation |
|---|---|---|---|---|---|
| 1 | **Hallucinated regulatory requirements** — Parser extracts a requirement that doesn't exist in the source text | Medium | Critical | Citation verification: extract cited text and compare | Re-read source text after extraction; require exact quotes for key requirements |
| 2 | **Missed affected policy** — Policy Mapper fails to find a relevant policy | High | Critical | Compare against gold-standard; monitor Recall@K | Multi-query retrieval, iterative search, low minimum relevance threshold |
| 3 | **False positive policy match** — Policy flagged as affected when it's not relevant | Medium | Medium | Monitor Precision@K; human review checkpoint | Cross-encoder reranking; agent assesses relevance before including |
| 4 | **Wrong severity classification** — Impact rated as "low" when it should be "critical" | Medium | High | Compare against gold-standard severity labels | Severity rubric in prompt; require justification for each severity rating |
| 5 | **Incorrect cross-reference resolution** — System follows a cross-reference to the wrong section | Medium | High | Verify cross-reference targets exist and are correctly cited | Validate CFR section numbers against eCFR API |
| 6 | **Agent infinite loop** — Agent keeps retrying retrieval without finding sufficient results | Low | Medium | Max iteration limits; cost ceiling | Hard limit of 3 retrieval iterations per requirement; timeout per agent |
| 7 | **Context pollution** — Information from one requirement's analysis leaks into another's | Medium | Medium | Separate state per requirement | Clear state boundaries in RegDeltaState; reset per-requirement context |
| 8 | **Stale regulatory data** — System analyzes against outdated version of a regulation | Medium | High | Timestamp all retrieved regulatory text | Check effective dates; warn if retrieved regulation has been amended since |
| 9 | **Prompt injection via regulatory text** — Malicious content in regulatory text manipulates agent behavior | Low | Critical | Input sanitization | Separate data from instructions; never embed raw regulatory text in system prompts |
| 10 | **Cost explosion** — Complex regulation triggers excessive LLM calls | Medium | Medium | Per-analysis cost tracking | Token budgets per agent; route simple updates to cheaper pipeline |
| 11 | **Incorrect citations** — Report cites a section that doesn't contain the supporting text | High | Critical | Citation verification pipeline | After report generation, extract each citation and verify against source |
| 12 | **Conflicting agent outputs** — Policy Mapper and Conflict Detector produce contradictory findings | Low | Medium | Consistency check in Impact Analyst | Impact Analyst prompt explicitly handles contradictions |

---

# 17. "Why Not Just Use a Single LLM?"

> "Why can't I just paste the regulatory text and all my internal policies into Claude's 200K context window?"

1. **Scale:** Internal policies total millions of tokens across hundreds of documents. They don't fit in any context window, no matter how large.

2. **No citation trail.** The model generates an answer, but you can't audit which specific policy section it referenced. In compliance, every claim must be traceable.

3. **"Lost in the Middle."** Research (Liu et al., 2024) shows models struggle with information placed in the middle of long contexts. A critical policy provision buried at position 80,000 is likely to be missed.

4. **No temporal awareness.** The model doesn't know which version of a regulation was active at a specific point in time, or whether a regulation has been superseded.

5. **No structured conflict detection.** Identifying conflicts between regulations from different jurisdictions requires systematic comparison, not a single-pass generation.

6. **No incrementality.** When a new regulatory update arrives, you would have to re-process everything. With RAG, you only retrieve what's relevant to the new update.

7. **Cost:** Sending 200K tokens per analysis at frontier model pricing is $0.60-3.00 just for input tokens. RegDelta's targeted retrieval uses ~175K total tokens because it retrieves only what's needed.

8. **No evaluation/debugging.** With a single LLM call, you can't inspect intermediate reasoning. With agents and retrieval, you can trace exactly which documents were retrieved, which tools were called, and where the reasoning went wrong.

---

# 18. Build Roadmap (Phased)

## Phase 0: Learn the Fundamentals (1-2 weeks)

Before writing project code, build small standalone experiments to understand each concept:

### Week 1: RAG Foundations
- [ ] **Embeddings:** Generate embeddings with OpenAI API and Qwen3. Compute cosine similarity between sentences. Understand what makes two texts "similar" in embedding space.
- [ ] **Vector search:** Set up Qdrant in Docker. Insert 100 embeddings. Query and understand top-k retrieval.
- [ ] **BM25:** Implement BM25 with `rank_bm25` library. Search the same 100 documents. Compare results against vector search.
- [ ] **Chunking:** Take a 20-page PDF. Try fixed-size (500 tokens) and semantic chunking. Compare retrieval quality.
- [ ] **Basic RAG:** Build a minimal retrieve-then-generate pipeline over 10 documents.

### Week 2: Agents and LangGraph
- [ ] **LangGraph basics:** Build a simple 3-node graph (input → process → output). Understand state, edges, conditional routing.
- [ ] **Tool calling:** Define 2-3 tools (web search, calculator). Build an agent that selects and calls tools.
- [ ] **LangGraph agent:** Build a ReAct agent in LangGraph that iteratively retrieves and reasons.
- [ ] **Supervisor pattern:** Build a minimal 2-agent supervisor (researcher + writer) to understand the pattern.
- [ ] **Human-in-the-loop:** Add a checkpoint node where execution pauses and resumes after human input.

## Phase 1: Minimal Prototype (2-3 weeks)

**Goal:** A working system that takes regulatory text and retrieves potentially affected policies.

- [ ] Set up project structure and Docker Compose (Qdrant, Redis, PostgreSQL)
- [ ] Build Federal Register API client (fetch regulations by document number)
- [ ] Generate 50 synthetic internal policies (see Appendix B)
- [ ] Build document ingestion pipeline: load → chunk → embed → store in Qdrant
- [ ] Build basic dense retrieval: query → embed → search Qdrant → return top-10
- [ ] Build simple LLM generation: retrieved chunks + query → impact assessment
- [ ] Build 10 test cases with known correct answers
- [ ] Measure Baseline 2 (naive RAG) metrics
- [ ] Set up Langfuse or LangSmith for tracing

**Deliverable:** A working naive RAG system with 10 evaluated test cases.

## Phase 2: Advanced Retrieval (2 weeks)

**Goal:** Significantly improve retrieval quality.

- [ ] Add BM25 index alongside Qdrant
- [ ] Implement Reciprocal Rank Fusion to merge dense + sparse results
- [ ] Add cross-encoder reranking (BGE-reranker-v2)
- [ ] Implement query rewriting (LLM generates 3 query variants per input)
- [ ] Add metadata filtering (regulator, policy_domain, effective_date)
- [ ] Implement contextual chunking (prepend LLM-generated context before embedding)
- [ ] Generate 100 more synthetic policies (total: 150)
- [ ] Build 30 more test cases (total: 40)
- [ ] Measure Baseline 3 (advanced RAG) and compare against Baseline 2

**Deliverable:** Quantified improvement in Recall@10 and Precision@5 from hybrid retrieval + reranking.

## Phase 3: Agentic Behavior (2-3 weeks)

**Goal:** Add iterative retrieval, query decomposition, and self-correction.

- [ ] Build Regulatory Parser as a LangGraph node with structured output
- [ ] Build Policy Mapper as a LangGraph node with iterative retrieval
- [ ] Implement query decomposition (regulatory update → multiple sub-queries)
- [ ] Implement sufficiency checking (does the agent have enough evidence?)
- [ ] Implement self-verification (agent re-reads sources to verify claims)
- [ ] Add multi-hop retrieval (follow cross-references between regulations)
- [ ] Build the single-agent pipeline (one agent, all tools)
- [ ] Measure Baseline 4 (agentic RAG) and compare against Baseline 3

**Deliverable:** Quantified improvement from iterative retrieval and multi-hop on test cases that require chained reasoning.

## Phase 4: Multi-Agent Architecture (2-3 weeks)

**Goal:** Decompose into specialized agents and build the supervisor.

- [ ] Build Conflict Detector Agent
- [ ] Build Impact Analyst Agent
- [ ] Build Report Generator Agent
- [ ] Build LangGraph supervisor (orchestrator)
- [ ] Implement parallel execution (Policy Mapper ‖ Conflict Detector)
- [ ] Implement human-in-the-loop checkpoint
- [ ] Add shared state management (RegDeltaState)
- [ ] Generate 50 more synthetic policies (total: 200)
- [ ] Build 30 more test cases (total: 70)
- [ ] Measure Baseline 5 (multi-agent) and compare against Baseline 4

**Deliverable:** Full multi-agent system with quantified comparison across all 5 baselines.

## Phase 5: Evaluation Framework (1-2 weeks)

**Goal:** Build a rigorous, automated evaluation pipeline.

- [ ] Implement all retrieval metrics (Recall@K, MRR, NDCG)
- [ ] Implement RAG metrics (faithfulness, citation correctness, groundedness) using DeepEval
- [ ] Implement agent metrics (task success, trajectory quality, cost per analysis)
- [ ] Build automated evaluation pipeline (run all test cases, compute all metrics, generate report)
- [ ] Finalize gold-standard test set (expand to 100 test cases)
- [ ] Run all 5 baselines on the complete test set
- [ ] Generate comparison tables and charts

**Deliverable:** Complete evaluation report with metrics across all baselines.

## Phase 6: Reliability and Hardening (1-2 weeks)

**Goal:** Handle failures, hallucinations, and edge cases.

- [ ] Implement citation verification pipeline
- [ ] Add confidence scoring to all assessments
- [ ] Implement retry logic with exponential backoff for API calls
- [ ] Add iteration limits and cost ceilings per agent
- [ ] Implement input sanitization (prompt injection protection)
- [ ] Handle edge cases: empty regulatory text, no matching policies, API timeouts
- [ ] Build error reporting and graceful degradation
- [ ] Add semantic caching for repeated queries

**Deliverable:** System handles all documented failure modes gracefully.

## Phase 7: Production Polish (1-2 weeks)

**Goal:** Production-ready API, observability, and demo interface.

- [ ] Build complete FastAPI backend with all endpoints
- [ ] Add Prometheus metrics (latency, cost, retrieval quality, agent performance)
- [ ] Build Grafana dashboards
- [ ] Add analysis history and feedback storage in PostgreSQL
- [ ] Build Streamlit frontend for demo
- [ ] Containerize everything with Docker Compose
- [ ] Write README with setup instructions

**Deliverable:** Fully containerized, observable system with demo interface.

## Phase 8: Research Experiments (1-2 weeks)

**Goal:** Investigate research questions.

- [ ] **Experiment 1 (RQ1):** Build regulatory knowledge graph in Neo4j. Implement graph-based retrieval. Compare against vector-only on conflict detection.
- [ ] **Experiment 2 (RQ2):** Run single-agent vs. multi-agent comparison on full test set. Compare faithfulness, hallucination rate, cost.
- [ ] **Experiment 3 (RQ3):** Degrade retrieval quality systematically and measure impact on analysis accuracy.
- [ ] Document experiment methodology, results, and conclusions

**Deliverable:** Research results with charts showing experimental findings.

## Phase 9: Final Presentation (1 week)

- [ ] Create architecture diagram (clean, professional)
- [ ] Record demo video analyzing a recent real regulatory update
- [ ] Compile evaluation results into presentation-ready tables and charts
- [ ] Write technical blog post explaining the system
- [ ] Clean up GitHub repository (README, documentation, reproducible setup)
- [ ] Prepare for interview discussions (review Section 20)

**Deliverable:** Portfolio-ready project with demo, documentation, and evaluation results.

---

# 19. Learning Outcomes

By completing RegDelta, you will have hands-on experience with:

### LLM Engineering
- Structured output extraction (JSON mode, function calling)
- Prompt engineering for domain-specific tasks (legal text)
- Model routing (cheap models for extraction, expensive for reasoning)
- Token budgeting and cost optimization
- Handling non-determinism in LLM outputs

### Retrieval-Augmented Generation
- Dense retrieval (vector similarity search)
- Sparse retrieval (BM25)
- Hybrid retrieval (dense + sparse + Reciprocal Rank Fusion)
- Cross-encoder reranking
- Semantic chunking and contextual chunking
- Multi-query retrieval (query rewriting)
- Metadata filtering
- Multi-hop retrieval (chained queries)
- Retrieval evaluation (Recall@K, MRR, NDCG)
- GraphRAG (Phase 8)

### Agent Engineering
- LangGraph state graphs (nodes, edges, conditional routing)
- Supervisor pattern (orchestrator managing specialist agents)
- Parallel agent execution
- Human-in-the-loop (checkpoint/resume)
- Tool design and function calling
- Agent state management
- Iterative retrieval with sufficiency checking
- Self-correction and verification
- Agent evaluation (trajectory analysis, tool selection accuracy)
- Failure recovery and retry logic
- Cost ceilings and iteration limits

### Production AI Engineering
- Observability (LangSmith/Langfuse tracing)
- Metrics collection (Prometheus) and visualization (Grafana)
- Caching (Redis semantic cache)
- API design (FastAPI)
- Containerization (Docker Compose)
- Database design (PostgreSQL for structured data, Qdrant for vectors)
- Evaluation pipelines (automated testing against gold-standard)
- Citation verification
- Error handling and graceful degradation

---

# 20. Interview Preparation

## What Makes This Project Impressive

1. **It solves a real, expensive problem.** Regulatory compliance costs $270B annually. Every financial institution has this problem.
2. **Every architectural choice is justified.** You can explain why hybrid retrieval, why multi-agent, why not GraphRAG (until Phase 8), why not MCP.
3. **Rigorous evaluation.** 5 progressive baselines with quantified improvements at each step. Ablation studies showing the marginal value of each component.
4. **Honest failure analysis.** You documented failure modes before building, not after.
5. **Research contribution.** 3 experimental research questions with testable hypotheses.

## 15 Difficult Interview Questions and How to Answer Them

1. **"Walk me through what happens when a new SEC rule is published."**
2. **"You used 5 agents. Prove to me that multi-agent outperforms a single agent."**
3. **"How do you know your citations are correct?"**
4. **"Why LangGraph over CrewAI? Over building your own orchestrator?"**
5. **"What happens when the system encounters a regulation type it's never seen before?"**
6. **"How do you handle a regulation that modifies another regulation you haven't indexed?"**
7. **"Your cost is $1 per analysis. How would you reduce it to $0.20?"**
8. **"What was your biggest failure mode and how did you mitigate it?"**
9. **"If I gave you 10x more policies (2000 instead of 200), what breaks?"**
10. **"How do you handle regulatory text that is genuinely ambiguous?"**
11. **"What is your retrieval recall on multi-hop questions specifically?"**
12. **"How would you deploy this for a bank that operates in 15 countries?"**
13. **"A compliance officer says your system missed a critical policy. How do you debug?"**
14. **"Why did you use contextual chunking? What improvement did it provide?"**
15. **"If context windows grow to 10M tokens, does your architecture still make sense?"**

---

# Appendix A: API Reference

## Federal Register API

```
Base URL: https://www.federalregister.gov/api/v1

# Search for recent SEC final rules
GET /documents?conditions[agencies][]=securities-and-exchange-commission
    &conditions[type][]=RULE
    &conditions[publication_date][gte]=2024-01-01
    &per_page=20
    &order=newest

# Get a specific document
GET /documents/{document_number}

# Useful fields in response:
# - title, abstract, body_html_url, full_text_xml_url
# - agencies[], cfr_references[], effective_on
# - regulation_id_numbers[], docket_ids[]
# - type (RULE, PRORULE, NOTICE, PRESDOCU)
```

## eCFR API

```
Base URL: https://www.ecfr.gov/api

# Get current text of a CFR section
GET /versioner/v1/full/{date}/title-{title}.xml?subtitle={sub}&chapter={ch}&part={pt}

# Get version history
GET /versioner/v1/versions/title-{title}?part={pt}
```

## SEC EDGAR Full-Text Search

```
Base URL: https://efts.sec.gov/LATEST/search-index

# Search (requires User-Agent header: "YourName your@email.com")
GET ?q="climate+disclosure"&dateRange=custom&startdt=2024-01-01&enddt=2024-12-31
    &forms=RULE,RELEASE

# Response includes: _id (accession number), form_type, entity_name, file_date
```

---

# Appendix B: Synthetic Policy Generation Guide

## Prompt Template for Generating Internal Policies

```
You are an expert compliance writer at a large US bank. Generate a realistic
internal compliance policy document based on the following specifications.

The policy should:
1. Reference specific real regulations (cite actual CFR sections, SEC rules,
   FINRA rules by their real numbers)
2. Include defined terms in a Definitions section (some terms should override
   their common meaning, as is typical in legal documents)
3. Include cross-references to other internal policies (use the format
   "POL-{DOMAIN}-{NUMBER}", e.g., "POL-AML-003")
4. Include specific procedures, thresholds, and requirements
5. Be realistic in length (3-10 pages equivalent, 1500-5000 words)

Specifications:
- Domain: {domain}  (e.g., "Anti-Money Laundering")
- Policy ID: {policy_id}  (e.g., "POL-AML-003")
- Title: {title}  (e.g., "Customer Due Diligence and KYC Procedures")
- Primary regulation: {primary_regulation}  (e.g., "31 CFR § 1020.220 - CDD Rule")
- Related regulations: {related_regulations}
- Effective date: {effective_date}
- Version: {version}

Structure the policy with these sections:
1. Purpose and Scope
2. Regulatory Authority (cite specific regulations)
3. Definitions
4. Applicability
5. Policy Requirements (the substantive content — be specific)
6. Procedures
7. Exceptions and Exemptions
8. Reporting Requirements
9. Record Retention
10. Related Policies (cross-reference 2-3 other internal policies by ID)
11. Review and Update Schedule
12. Document Control (version history)
```

---

# Appendix C: Glossary

| Term | Definition |
|---|---|
| **CFR** | Code of Federal Regulations — the codification of general and permanent rules published by federal agencies |
| **eCFR** | Electronic Code of Federal Regulations — the current, online version of the CFR |
| **EDGAR** | Electronic Data Gathering, Analysis, and Retrieval — SEC's filing system |
| **Federal Register** | The daily journal of the US Government; where new rules are published |
| **BM25** | Best Matching 25 — a probabilistic sparse retrieval algorithm based on term frequency |
| **RRF** | Reciprocal Rank Fusion — algorithm for merging ranked lists from different retrieval methods |
| **Cross-encoder** | A model that scores a (query, document) pair jointly, more accurate than bi-encoder but slower |
| **Bi-encoder** | A model that embeds query and document independently, enabling fast approximate nearest neighbor search |
| **Multi-hop retrieval** | Retrieval where results from one query inform the next query, forming a chain |
| **Contextual chunking** | Prepending LLM-generated context to each chunk before embedding to improve retrieval |
| **PICO** | Population, Intervention, Comparison, Outcome — framework for clinical evidence (used in EvidenceGrid) |
| **UDAAP** | Unfair, Deceptive, or Abusive Acts or Practices — consumer protection standard |
| **KYC** | Know Your Customer — identity verification requirement |
| **SOX** | Sarbanes-Oxley Act — corporate governance and financial reporting law |
| **Basel III** | International banking regulation framework for capital adequacy |
| **FINRA** | Financial Industry Regulatory Authority — self-regulatory organization for broker-dealers |
| **CFPB** | Consumer Financial Protection Bureau |
| **OCC** | Office of the Comptroller of the Currency |

---

*Document version: 1.0 | Created: August 2026 | Project: RegDelta*
