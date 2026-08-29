# RegDelta — Regulatory Change Impact Analyzer

**A RAG system that reads new RBI regulations and identifies which internal compliance policies need to change, why, and how urgently — with citations.**

## The Problem

When the Reserve Bank of India publishes a new circular, notification, or master direction, compliance teams at banks and financial institutions must manually:

1. Read the full regulatory text
2. Identify what changed and who is affected
3. Search through hundreds of internal policies to find relevant ones
4. Compare new requirements against current policy provisions
5. Write an impact assessment with recommendations

This takes **2–4 weeks per significant update**, and RBI publishes hundreds of updates per year.

## What RegDelta Does

Given a new RBI regulatory update, RegDelta:

1. **Parses** the regulatory text into structured requirements (what changed, who is affected, effective date)
2. **Retrieves** potentially affected internal compliance policies using hybrid search
3. **Detects conflicts** between new requirements and existing regulatory obligations
4. **Performs gap analysis** — what the regulation requires vs. what the policy currently provides
5. **Generates an impact report** with severity ratings, recommended actions, and traceable citations

## Architecture

*The architecture evolves incrementally. See [docs/](docs/) for architecture decision records.*

**Current phase:** MVP — basic ingestion and retrieval pipeline.

## Data Sources

- **Regulatory data:** Real, publicly available RBI circulars, notifications, and master directions from [rbi.org.in](https://www.rbi.org.in)
- **Internal policies:** Synthetic (LLM-generated) compliance policies that reference real RBI regulations. Clearly labelled as synthetic.

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| Language | Python 3.11+ | Standard for AI/ML |
| Data fetching | requests + BeautifulSoup | RBI provides HTML; no API needed |
| Vector database | Qdrant | Free, self-hosted, metadata filtering |
| Embeddings | OpenAI / local models | Configurable |
| LLM | Anthropic Claude / OpenAI | Configurable per task |
| Agent orchestration | LangGraph | Production-grade state management (added in Phase 3) |
| Evaluation | Custom + DeepEval | Retrieval and generation metrics |
| Observability | Langfuse | Trace every LLM call and retrieval |

## Project Status

- [x] Project structure and specification
- [x] RBI data ingestion
- [x] Document preprocessing
- [x] EDA on regulatory corpus
- [x] Chunking and embedding
- [ ] Basic retrieval pipeline
- [ ] Synthetic policy generation
- [ ] LLM-powered impact analysis
- [ ] Evaluation framework
- [ ] Advanced retrieval (hybrid + reranking)
- [ ] Agentic retrieval
- [ ] Multi-agent architecture

## Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/regdelta.git
cd regdelta

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -e .

# Copy environment template
cp .env.example .env
# Edit .env with your API keys

# Start infrastructure (when needed, later phases)
# docker compose up -d
```

## License

MIT
