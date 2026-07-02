# Self-Corrective Agentic RAG Pipeline (LangGraph)

This project implements a self-corrective Agentic RAG workflow using LangGraph.

It combines:
- Query routing (vector store vs web search)
- Retrieval and document relevance grading
- Answer generation
- Hallucination and answer-quality grading
- Iterative correction path when answers are not supported or not useful

## Graph

The latest generated graph image is available at:
- [graph/graph.png](graph/graph.png)

You can also regenerate it by running:

```bash
uv run python -m graph.graph
```

## High-Level Flow

Pipeline entrypoint is in [main.py](main.py), and the graph is defined in [graph/graph.py](graph/graph.py).

Flow:
1. Route question to `retrieve` or `websearch`
2. If `retrieve`:
	1. Retrieve candidate chunks from Chroma
	2. Grade each chunk for relevance
	3. If weak retrieval, branch to `websearch`
3. Generate answer from current context
4. Grade hallucination/support against retrieved facts
5. Grade whether answer resolves the question
6. If not useful/supported, loop back through correction path

## Project Structure

- [main.py](main.py): App entrypoint for invoking graph
- [ingestion.py](ingestion.py): Builds/loads vector store retriever
- [graph/graph.py](graph/graph.py): LangGraph workflow and routing logic
- [graph/state.py](graph/state.py): Typed graph state schema
- [graph/consts.py](graph/consts.py): Node name constants
- [graph/chains/router.py](graph/chains/router.py): Route classifier chain
- [graph/chains/retrieval_grader.py](graph/chains/retrieval_grader.py): Doc relevance grader
- [graph/chains/generation_chain.py](graph/chains/generation_chain.py): Answer generation chain
- [graph/chains/hallucination_grader.py](graph/chains/hallucination_grader.py): Grounding grader
- [graph/chains/answer_grader.py](graph/chains/answer_grader.py): Final usefulness grader
- [graph/nodes/retrieve.py](graph/nodes/retrieve.py): Retrieval node
- [graph/nodes/grade_documents.py](graph/nodes/grade_documents.py): Retrieval filtering node
- [graph/nodes/web_search.py](graph/nodes/web_search.py): Tavily augmentation node
- [graph/nodes/generate.py](graph/nodes/generate.py): Generation node
- [graph/chains/tests/test_chains.py](graph/chains/tests/test_chains.py): Chain-level tests

## Prerequisites

- Python >= 3.11.3
- uv
- API keys in environment

Install dependencies:

```bash
uv sync
```

## Environment Variables

Create a `.env` file in project root with at least:

```env
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
LANGCHAIN_API_KEY=your_langsmith_key_optional
LANGCHAIN_PROJECT=reflexion-agent
HUGGINGFACEHUB_API_TOKEN=your_hf_token_optional
```

Notes:
- `GROQ_API_KEY` is required for routing, grading, and generation chains.
- `TAVILY_API_KEY` is required for web search node.
- `HUGGINGFACEHUB_API_TOKEN` is optional for the current embedding model (`sentence-transformers/all-mpnet-base-v2`), but recommended.

## Ingestion and Retriever

Ingestion is handled in [ingestion.py](ingestion.py):
- Loads 3 Lilian Weng posts
- Splits into chunks
- Uses Hugging Face embeddings (`sentence-transformers/all-mpnet-base-v2`)
- Seeds Chroma collection (`rag-chroma`) in `.chroma` directory if empty

Run ingestion:

```bash
uv run python ingestion.py
```

## Run the Pipeline

```bash
uv run python main.py
```

Default question in main is:
- `agent memory`

You can modify that in [main.py](main.py) as needed.

## Run Tests

```bash
uv run pytest graph/chains/tests/test_chains.py -s -v
```

## Rate Limit Guidance

If you get provider rate limits:
1. Avoid full `app.invoke(...)` while debugging graph shape.
2. Render graph only via:

```bash
uv run python -m graph.graph
```

3. Validate individual chains selectively rather than running end-to-end repeatedly.

## Important Current Behavior

- Graph image generation currently happens in [graph/graph.py](graph/graph.py) when run as module/script.
- Retriever is initialized at import time in [ingestion.py](ingestion.py), which can make startup heavier.

## Common Troubleshooting

### 1) `python: command not found`
Your shell may not have the venv active. Use `uv run ...` commands.

### 2) `USER_AGENT environment variable not set`
This is a warning from web loaders. It is already defaulted in ingestion.

### 3) Empty retrieval results
Re-run ingestion once:

```bash
uv run python ingestion.py
```

### 4) Import path errors when running from subfolders
Prefer running commands from project root.

## Persistence and Git

Local persistence paths are ignored in git via [.gitignore](.gitignore):
- `.chroma/`
- `db/`
- `rag_embeddings/`
- `*.sqlite3`

