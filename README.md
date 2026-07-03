# LangGraph ReACT Agent with SQLite Memory

This project is a LangGraph-based ReACT agent that can:

- reason with an LLM,
- call tools (Tavily search + custom temperature tool),
- persist conversation state on disk using SQLite,
- resume memory by session ID across runs.

## Architecture

The graph has two nodes:

- agent_reason: calls the tool-enabled LLM
- act: executes tool calls via ToolNode

Flow:

1. Start at agent_reason
2. If tool calls exist, route to act
3. Return to agent_reason
4. Stop when no tool calls remain

Memory persistence is enabled through LangGraph SqliteSaver and stored in a local database file named memory.db.

## Project Files

- main.py: graph creation, SQLite checkpointer wiring, invocation
- nodes.py: agent reasoning node + tool node
- react.py: LLM setup and tools
- pyproject.toml: dependencies
- memory.db: auto-created SQLite checkpoint store

## Prerequisites

- Python 3.11+
- uv package manager
- API keys:
	- GROQ_API_KEY (required)
	- TAVILY_API_KEY (required for web search tool)
	- LANGCHAIN_API_KEY, LANGCHAIN_PROJECT (optional; for tracing)

## Setup

1. Install dependencies:

```bash
uv sync
```

2. Create a .env file in the project root:

```env
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key

# Optional tracing
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=your_project_name
```

## How to Run

Run the agent:

```bash
uv run python main.py
```

You will see:

- a printed Session ID
- a final model response

## Session ID and Memory Persistence

The app uses thread_id for memory scoping:

- If SESSION_ID is provided, the same conversation thread is resumed.
- If SESSION_ID is not provided, a new UUID is generated.

### Continue the same memory session

```bash
SESSION_ID=my-session-1 uv run python main.py
```

Run the same command again with the same SESSION_ID and memory context is reused from SQLite.

### Start a fresh session

```bash
SESSION_ID=my-session-2 uv run python main.py
```

This creates a separate memory thread in the same memory.db file.

## How SQLite Memory Works Here

- SQLite DB file: memory.db
- Checkpointer: SqliteSaver
- Key for memory partitioning: configurable.thread_id
- Persisted data includes messages and tool call history for each session

## Verify Stored Session Memory

Use this command to inspect the stored messages for a session:

```bash
SESSION_ID=my-session-1 uv run python -c "
from main import app
config = {'configurable': {'thread_id': 'my-session-1'}}
state = app.get_state(config)
for msg in state.values.get('messages', []):
		print(msg.__class__.__name__, ':', str(msg.content)[:140])
"
```

## Common Issues

- Error: command not found: python
	- Use uv run python main.py instead of python main.py.

- No temperature/search answer
	- Ensure GROQ_API_KEY and TAVILY_API_KEY are present in .env.

- Memory not resuming
	- Reuse exactly the same SESSION_ID value.

## Quick Execution Examples

```bash
# New session (random UUID)
uv run python main.py

# Fixed session (persistent thread)
SESSION_ID=000002 uv run python main.py

# Re-run same session later
SESSION_ID=000002 uv run python main.py
```

## Notes

- The graph image is exported to agent_reason.png.
- SQLite file memory.db is local, simple, and good for single-machine persistence.
