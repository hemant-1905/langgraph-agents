# LangGraph Studio Setup Guide

This guide documents the exact setup flow used in this repository to run LangGraph Studio successfully.

## What This File Is For

Use this file when you want to:
- install LangGraph CLI in this existing project,
- run the local LangGraph API server,
- connect it to LangGraph Studio / LangSmith,
- find local API documentation endpoints.

## 1) Prerequisites

- You are inside this repository root.
- You are using `uv` (not plain `pip`) for dependency management.
- You have a valid LangSmith API key.
- Your `.env` file contains required provider keys (for example Groq, Tavily).

## 2) Install LangGraph CLI

```bash
uv add "langgraph-cli[inmem]"
```

## 3) Install This Project in Editable Mode

```bash
uv pip install -e .
```

Note:
- This repository is an existing project.
- Do **not** run `langgraph new ...` here. That command is only for creating a fresh scaffold.

## 4) Validate LangGraph App Config

`langgraph.json` should point to your graph app and env file.

Current config in this repo:

```json
{
  "graphs": {
    "agent": "./graph/graph.py:app"
  },
  "env": ".env",
  "dependencies": ["."]
}
```

## 5) Configure `.env`

Make sure `.env` has at least:

- `LANGSMITH_API_KEY`
- `LANGCHAIN_PROJECT`
- `LANGCHAIN_ENDPOINT` (EU endpoint if applicable)
- `LANGCHAIN_TRACING_V2`
- your model/tool provider keys used by the app

Important: keep secrets private and rotate keys if they were ever exposed.

## 6) Launch LangGraph Studio Dev Server

```bash
uv run langgraph dev --studio-url https://eu.smith.langchain.com
```

If Safari blocks localhost/http, use tunnel mode:

```bash
uv run langgraph dev --studio-url https://eu.smith.langchain.com --tunnel
```

## 7) Restart vs Hot Reload

Hot reload is automatic for normal code changes.

Restart `langgraph dev` when you change:
- `.env`
- `langgraph.json`
- dependencies (`uv add ...`)

## 8) Local API Documentation

When the server is running, open:

- `http://localhost:2024/docs`
- `http://localhost:2024/openapi.json`

If your server starts on a different port, replace `2024` with that port from terminal logs.

## 9) Quick Command Cheat Sheet

```bash
# Install CLI
uv add "langgraph-cli[inmem]"

# Editable install
uv pip install -e .

# Start local server + Studio
uv run langgraph dev --studio-url https://eu.smith.langchain.com

# Safari-safe mode
uv run langgraph dev --studio-url https://eu.smith.langchain.com --tunnel
```
