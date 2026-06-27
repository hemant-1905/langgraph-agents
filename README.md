# LangGraph Reflection Agent

A **tweet refinement agent** built with [LangGraph](https://github.com/langchain-ai/langgraph) that iteratively improves a tweet using a generate → reflect → generate loop powered by a Groq-hosted LLM.

## What it does

1. **Generate** — a Twitter influencer assistant LLM writes an improved version of your tweet.
2. **Reflect** — a second LLM persona (a viral Twitter critic) critiques the generated tweet for tone, length, virality, and style.
3. **Loop** — the critique is fed back to the generator for another pass. This repeats until a quality threshold (3 messages) is reached.
4. **Output** — the final polished tweet is printed to the terminal, along with an ASCII graph of the agent's state machine.

The agent graph looks like this:

start → generate ⇄ reflect → end

## Project structure

├── main.py # LangGraph state machine definition and entry point
├── chains.py # LLM prompt chains (generation + reflection)
├── pyproject.toml # Project dependencies managed by uv
├── .python-version # Pins Python 3.14.3
└── .env # API keys (not committed to git)


## Prerequisites

- [uv](https://docs.astral.sh/uv/) — Python package manager
- Python 3.14.3 (managed automatically by uv via `.python-version`)
- A [Groq](https://console.groq.com/) API key
- A [LangSmith](https://smith.langchain.com/) API key (for tracing)

## Setup

### 1. Clone and enter the repo

```bash
git clone <your-repo-url>
cd langgraph-agents

uv venv .venv --clear
uv sync

GROQ_API_KEY="your_groq_api_key_here"
LANGCHAIN_API_KEY="your_langsmith_api_key_here"
LANGCHAIN_PROJECT="reflect-agent"
OPENAI_API_KEY=""                        # Optional, not used by default
HUGGINGFACEHUB_API_TOKEN=""              # Optional, not used by default

Running
# Activate the virtual environment
source .venv/bin/activate

# Run the agent
python main.py

Tracing
LangSmith tracing is enabled by default. View runs at https://eu.api.smith.langchain.com under the project name set in LANGCHAIN_PROJECT.
