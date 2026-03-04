# Layer10 Take-Home: Grounded Long-Term Memory System

## Write-Up Document

### 1. Executive Summary

This project implements a **grounded long-term memory system** that extracts structured knowledge from the Enron Email Dataset (2,000 emails), deduplicates entities, stores them in a Neo4j knowledge graph, and provides retrieval APIs plus interactive visualization.

**Pipeline Results:**
- **670 emails processed** (99.9% success rate; 671 unique bodies from 2,000 rows after body-hash dedup)
- **2,119 deduplicated entities** (from 5,905 raw extractions — 2.8× reduction)
- **2,411 grounded claims** (from 4,292 raw — 1.8× reduction)
- **5 entity types**: Person (744), Organization (608), Topic (396), Project (273), Meeting (98)
- **10 relationship types**: MENTIONED_IN (1,016), COMMUNICATED_WITH (639), DISCUSSED_IN (252), WORKS_ON (141), WORKS_AT (133), PART_OF (76), ATTENDED (74), ASSIGNED_TO (50), DECIDED (24), REPORTS_TO (6)

---

### 2. Architecture Overview

```
┌──────────────────┐
│  Enron CSV        │  2,000 emails (5.4 MB)
│  data/            │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  pipeline/        │  LLM-based structured extraction
│  extraction.py    │  Llama 3.1 8B via OpenRouter
│  10 workers       │  Body-hash dedup → 671 unique
└────────┬─────────┘
         │  5,905 entities, 4,292 claims
         ▼
┌──────────────────┐
│  pipeline/        │  3-pass entity deduplication
│  dedup.py         │  (UnionFind) + claim consolidation
└────────┬─────────┘
         │  2,119 entities, 2,411 claims
         ▼
┌──────────────────┐
│  pipeline/        │  Neo4j ingestion
│  graph_builder.py │  Nodes + Relationships
└────────┬─────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────┐
│api/    │ │viz/      │
│FastAPI │ │Streamlit │
│REST API│ │Graph Viz │
└────────┘ └──────────┘
```

---

### 3. Ontology Design

**Entity Types** (schema.py):
| Type | Description | Example |
|------|-------------|---------|
| person | Individual people | Tim Belden, Kate Symes |
| organization | Companies, departments | Enron, Portland General Electric |
| project | Work initiatives, products | TruOrange, EnPower |
| topic | Discussion subjects, concepts | Gas Market, Power Trading |
| meeting | Specific meetings or events | Weekly Staff Meeting |

**Claim Types** (12 relationship types):
| Type | Subject → Object | Example |
|------|-------------------|---------|
| works_at | person → organization | Tim Belden works_at Enron |
| reports_to | person → person | Kate Symes reports_to Tim Belden |
| works_on | person → project | Mark Guzman works_on TruOrange |
| part_of | organization → organization | ENA part_of Enron |
| discussed_in | topic → meeting | Gas Prices discussed_in Weekly Meeting |
| decided | person → topic | Tim Belden decided Power Purchase |
| communicated_with | person → person | Kate Symes communicated_with Amy |
| attended | person → meeting | Mark Guzman attended Team Standup |
| assigned_to | person → project | Anna Mehrer assigned_to Cal ISO |
| mentioned_in | entity → meeting | Enron mentioned_in Energy Report |
| contradicts | claim → claim | For temporal conflicts |
| supersedes | claim → claim | For updated information |

**Every claim is grounded** with:
- Source artifact (email message-id)
- Evidence excerpt (relevant quote from email body)
- Confidence score (0.0–1.0)

---

### 4. Design Decisions & Tradeoffs

#### 4a. LLM Selection
- **Chosen**: Llama 3.1 8B via OpenRouter (paid API)
- **Why not Gemini**: Severe rate limits (~15 requests/minute), quota exhaustion
- **Why not Ollama local**: RTX 3050 4GB VRAM too constrained — qwen2.5:7b at 2.5 tok/s was impractical for 671 emails
- **Why not Groq**: 500K tokens/day cap would require multi-day extraction
- **Result**: 10 parallel workers, ~3 seconds/email, full extraction in 35 minutes

#### 4b. Body-Hash Deduplication
- **Discovery**: Enron stores identical emails in multiple folders (discussion_threads, info, all_documents)
- **Impact**: 2,000 rows → only 671 unique email bodies
- **Solution**: SHA-256 hash on cleaned body text before LLM calls
- **Savings**: ~66% reduction in API costs and extraction time

#### 4c. Entity Deduplication Strategy
Three-pass approach using UnionFind:
1. **Exact match** (case-insensitive): "enron" = "ENRON" = "Enron"
2. **Alias absorption**: If entity A's canonical name appears as entity B's alias (same type), merge
3. **Fuzzy match**: token_sort_ratio ≥ 90% via rapidfuzz, same entity type, with prefix matching heuristic

**Result**: 5,905 → 2,119 entities (64% reduction)

#### 4d. Claim Consolidation
- Claims referencing merged entities are remapped to canonical entity IDs
- Duplicate claims (same subject + object + type) are merged, combining evidence lists
- 934 claims with unresolved entities (neither subject nor object matched any entity group) were dropped — acceptable loss for data quality

#### 4e. Graph Storage
- **Neo4j 5** chosen for native graph queries (shortest paths, traversals, pattern matching)
- Entity nodes carry: canonical_name, entity_type, aliases[], mention_count
- Relationship edges carry: confidence, mention_count, detail, excerpts[]
- Indexes on id (unique), canonical_name, entity_type for fast lookups

---

### 5. Evaluation & Quality Analysis

#### 5a. Extraction Quality
- **Success rate**: 670/671 (99.9%) — only 1 permanent failure (likely malformed email)
- **Empty-extraction retry**: For emails with >100 chars body but 0 entities/claims on first pass, up to 3 retries (handles transient LLM failure)
- **Average per email**: ~8.8 entities, ~6.4 claims extracted

#### 5b. Entity Resolution Quality
- **Enron** correctly absorbed 32 aliases (Corp, ENA, Enron North America, Portland General Electric, etc.)
- **Person names** merged email addresses with display names (e.g., "tim.belden@enron.com" → Tim Belden)
- **Fuzzy matching** catches variations like "Tim Belden/HOU/ECT@ECT" → "Tim Belden"

#### 5c. Graph Connectivity
- **Most connected**: Mark Guzman (198), Kate Symes (164), Enron (158), TruOrange (86)
- **Isolated nodes**: 625 (entities mentioned only in metadata, not linked to others)
- **Average degree**: ~2.3 relationships per connected entity

#### 5d. Known Limitations
1. **Entity type misclassification**: Some topics classified as organizations (e.g., "Texas" classified as organization)
2. **Temporal granularity**: Claims lack temporal ordering beyond email date
3. **Single-hop extraction**: LLM extracts from individual emails; cross-email inference not performed
4. **8B model limitations**: Smaller model occasionally misses nuanced relationships vs. larger models

---

### 6. Retrieval API

FastAPI endpoints at `http://localhost:8000`:

![API Swagger UI](Screenshots/api_swagger.png)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness check |
| `/stats` | GET | Graph-level statistics |
| `/search?q=...&limit=10` | GET | Fuzzy entity search |
| `/entity/{name}` | GET | Entity detail + relationships |
| `/query` | POST | Structured graph query |
| `/path` | POST | Shortest path between entities |
| `/neighbourhood/{name}?depth=2` | GET | k-hop subgraph |

Example queries:
```bash
# Search for entities
curl "http://localhost:8000/search?q=Tim%20Belden"

# Get entity context with all relationships
curl "http://localhost:8000/entity/Tim%20Belden"

# Find path between two entities
curl -X POST "http://localhost:8000/path" \
  -H "Content-Type: application/json" \
  -d '{"source": "Tim Belden", "target": "Enron"}'
```

---

### 7. Visualization

Two visualization approaches:

**1. Streamlit Interactive App** (`viz/viz_app.py`, launched via `python run_viz.py`):

| Overview | Entity Search + Evidence |
|----------|-------------------------|
| ![Overview](Screenshots/viz_overview.png) | ![Search](Screenshots/viz_search.png) |

| Graph Explorer | Path Finder |
|----------------|-------------|
| ![Graph](Screenshots/viz_graph.png) | ![Path](Screenshots/viz_pathfinder.png) |

**2. Static HTML** (`viz/generate_static_viz.py`): Pyvis-generated standalone HTML files:

| Full Graph (Top 60 Entities) | Tim Belden Spotlight |
|------------------------------|---------------------|
| ![Overview](Screenshots/viz_pyvis_overview.png) | ![Tim Belden](Screenshots/viz_pyvis_tim_belden.png) |

| Enron Organization | Kate Symes | Mark Guzman |
|--------------------|------------|-------------|
| ![Enron](Screenshots/vis_pyvis_enron.png) | ![Kate Symes](Screenshots/viz_pyvis_kate_symes.png) | ![Mark Guzman](Screenshots/viz_pyvis_mark_guzman.png) |

---

### 8. Example Context Packs

Five pre-generated context packs demonstrate how the retrieval API assembles grounded context for downstream LLM consumption. Each pack is a self-contained JSON file in `output/context_packs/`:

| Pack | Query Strategy | Contents | Size |
|------|---------------|----------|------|
| `pack_1_tim_belden.json` | Entity lookup | Tim Belden's entity data + all relationships | 13.6 KB |
| `pack_2_belden_symes_path.json` | Shortest path + entity contexts | Path between Tim Belden and Kate Symes, plus both entity contexts | 62.3 KB |
| `pack_3_decisions.json` | Claim-type filter | All claims of type DECIDED | 4.5 KB |
| `pack_4_enron_org.json` | Entity + neighbourhood | Enron entity context + 2-hop subgraph | 101.3 KB |
| `pack_5_reporting_structure.json` | Claim-type filter | All REPORTS_TO claims (organizational hierarchy) | 1.2 KB |

These packs show how different retrieval strategies (entity lookup, path traversal, claim filtering, neighbourhood expansion) can be composed to build rich, grounded context for an LLM agent.

---

### 9. Reproducibility

#### Prerequisites
- Python 3.10+
- Docker (for Neo4j)
- OpenRouter API key (or other LLM backend)

#### Steps
```bash
# 1. Clone and setup
git clone <repo-url> && cd LayerAI
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env with your API keys

# 3. Dataset (already included as data/enron_subset.csv)

# 4. Start Neo4j
docker run -d --name neo4j-memory -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/layer10memory neo4j:5-community

# 5. Run full pipeline (extract → dedup → graph build)
python run_pipeline.py
# Or individually:
#   python -m pipeline.extraction
#   python -m pipeline.dedup
#   python -m pipeline.graph_builder

# 6. Run API
python run_api.py  # http://localhost:8000

# 7. Run visualization
python run_viz.py  # http://localhost:8501

# 8. Generate static HTML visualizations
python -m viz.generate_static_viz
```

---

### 10. File Structure

```
├── config.py                  # Central configuration (.env loader)
├── schema.py                  # Ontology definitions (Pydantic models)
├── run_pipeline.py            # Run extraction → dedup → graph in sequence
├── run_api.py                 # Start FastAPI server
├── run_viz.py                 # Start Streamlit app
├── requirements.txt           # Python dependencies
├── .env.example               # Template for API keys
│
├── pipeline/
│   ├── extraction.py          # LLM extraction (parallel + checkpointing)
│   ├── dedup.py               # 3-pass entity dedup + claim consolidation
│   └── graph_builder.py       # Neo4j graph ingestion
│
├── api/
│   └── retrieval_api.py       # FastAPI REST API (7 endpoints)
│
├── viz/
│   ├── viz_app.py             # Streamlit interactive graph explorer
│   └── generate_static_viz.py # Pyvis standalone HTML generation
│
├── scripts/
│   ├── eda_enron.py           # Exploratory data analysis
│   └── select_subset.py       # Dataset subset selection
│
├── Screenshots/               # UI screenshots (10 images)
│
├── data/
│   └── enron_subset.csv       # 2,000 email subset (5.4 MB)
│
└── output/
    ├── deduped_graph.json     # Serialized unified graph (2.5 MB)
    ├── graph_stats.json       # Graph statistics
    ├── context_packs/         # 5 example context packs
    ├── visualizations/        # 5 static HTML graph visualizations
    └── extractions/           # 670 per-email JSON extraction files
```

---

### 11. Summary of Metrics

| Metric | Value |
|--------|-------|
| Emails processed | 670/671 (99.9%) |
| Unique email bodies | 671 (from 2,000 rows) |
| Raw entities extracted | 5,905 |
| Raw claims extracted | 4,292 |
| Deduplicated entities | 2,119 (64% reduction) |
| Consolidated claims | 2,411 (44% reduction) |
| Entity types | 5 |
| Relationship types | 10 active (12 defined) |
| Neo4j nodes | 2,119 |
| Neo4j relationships | 2,411 |
| Most connected entity | Mark Guzman (198 connections) |
| Extraction time | ~35 minutes (10 parallel workers) |
| LLM model | Llama 3.1 8B (via OpenRouter) |

---

### 12. Adaptation to Layer10 Production

This prototype demonstrates the core pattern that adapts to Layer10's production context:

**Incremental Updates**
- The extraction pipeline already uses body-hash checkpointing — new emails skip already-processed bodies. For production, an append-only ingestion queue feeds new artifacts through the same extract → dedup → merge pipeline without reprocessing the full corpus.

**Update Semantics (contradicts / supersedes)**
- The schema defines `contradicts` and `supersedes` claim types for temporal conflict resolution. A production system would compare new claims against existing ones for the same subject-object pair and automatically flag contradictions or chain supersession links, giving the retrieval layer a temporal ordering of beliefs.

**Scaling the Ontology**
- `schema.py` is a single-file ontology that can be extended by adding new `EntityType` and `ClaimType` enum values. The extraction prompt is generated from the schema, so changes propagate automatically to the LLM contract.

**Multi-Source Adaptation**
- While this demo uses emails, the `Artifact` model already abstracts the source (message-id, subject, participants). Adapting to Slack, docs, or meeting transcripts requires only a new loader that maps source fields to the `Artifact` schema — the extraction, dedup, and graph layers remain unchanged.

**Retrieval for LLM Agents**
- The context packs demonstrate the retrieval pattern: entity lookup, path traversal, claim filtering, and neighbourhood expansion compose into grounded context windows that an LLM agent can consume. The FastAPI layer serves as the agent's memory interface.
