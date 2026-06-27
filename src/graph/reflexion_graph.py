from __future__ import annotations

import json
from typing import Literal

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.chains.responder import ResponderChain
from src.chains.revisor import RevisorChain
from src.config.settings import Settings
from src.models.schemas import CritiqueModel
from src.models.state import ReflexionState
from src.tools.search import TavilySearchTools


class ReflexionGraphBuilder:
    def __init__(
        self,
        settings: Settings,
        model_name: str = "qwen/qwen3-32b",
        max_tool_calls: int = 4,
    ) -> None:
        self._responder_chain = ResponderChain(settings.groq_api_key, model_name)
        self._revisor_chain = RevisorChain(settings.groq_api_key, model_name)
        self._search_tools = TavilySearchTools(settings.tavily_api_key)
        self._tool_node = ToolNode(self._search_tools.tools())
        self._max_tool_calls = max_tool_calls

    def _start_node(self, state: ReflexionState) -> ReflexionState:
        user_input = state.get(
            "user_input",
            "Write a short summary about AI-powered SOC and the leading companies there.",
        )
        return {
            "user_input": user_input,
            "responder_output": None,
            "revised_output": None,
            "pending_search_queries": [],
            "search_traces": [],
            "tool_call_count": state.get("tool_call_count", 0),
            "max_tool_calls": state.get("max_tool_calls", self._max_tool_calls),
            "tool_phase": "responder",
        }

    def _responder_node(self, state: ReflexionState) -> ReflexionState:
        responder_output = self._responder_chain.invoke(state["user_input"])
        return {
            "responder_output": responder_output,
            "pending_search_queries": responder_output.search,
            "tool_phase": "responder",
        }

    def _execute_tools_node(self, state: ReflexionState) -> ReflexionState:
        remaining = state["max_tool_calls"] - state["tool_call_count"]
        queries = state.get("pending_search_queries", [])

        if remaining <= 0 or not queries:
            return {"pending_search_queries": []}

        phase: Literal["responder", "revisor"] = state.get("tool_phase", "responder")
        tool_name = (
            "responder_tavily_search"
            if phase == "responder"
            else "revisor_tavily_search"
        )

        selected_queries = queries[:remaining]
        tool_calls = []
        query_lookup: dict[str, str] = {}

        for idx, query in enumerate(selected_queries):
            call_id = f"{tool_name}_{state['tool_call_count'] + idx + 1}"
            tool_calls.append(
                {
                    "id": call_id,
                    "name": tool_name,
                    "args": {"query": query},
                }
            )
            query_lookup[call_id] = query

        ai_message = AIMessage(content="Execute web research", tool_calls=tool_calls)
        result = self._tool_node.invoke({"messages": [ai_message]})
        tool_messages = result.get("messages", [])

        new_traces = []
        for msg in tool_messages:
            if not isinstance(msg, ToolMessage):
                continue

            query = query_lookup.get(msg.tool_call_id, "")
            try:
                parsed = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
            except Exception:
                parsed = {"raw": str(msg.content)}

            new_traces.append(
                {
                    "phase": phase,
                    "tool_name": msg.name or tool_name,
                    "tool_call_id": msg.tool_call_id,
                    "query": query,
                    "result": parsed,
                }
            )

        return {
            "search_traces": [*state["search_traces"], *new_traces],
            "tool_call_count": state["tool_call_count"] + len(selected_queries),
            "pending_search_queries": [],
        }

    def _revisor_node(self, state: ReflexionState) -> ReflexionState:
        base_output = state.get("revised_output") or state.get("responder_output")
        if not base_output:
            raise ValueError("Missing base output for revision step.")

        critique = base_output.critique
        critique_text = (
            f"Superfluous: {critique.superfluous}\nAssumptions: {critique.assumptions}"
        )

        search_context = self._format_search_context(state.get("search_traces", []))

        revised_output = self._revisor_chain.invoke(
            user_input=state["user_input"],
            draft_response=base_output.response,
            critique=critique_text,
            search_context=search_context,
        )

        return {
            "revised_output": revised_output,
            "pending_search_queries": revised_output.search,
            "tool_phase": "revisor",
        }

    @staticmethod
    def _format_search_context(search_traces: list[dict]) -> str:
        if not search_traces:
            return "No web evidence collected yet."

        chunks: list[str] = []
        for idx, trace in enumerate(search_traces, start=1):
            chunks.append(
                (
                    f"{idx}. phase={trace.get('phase')} tool={trace.get('tool_name')} "
                    f"query={trace.get('query')} result={json.dumps(trace.get('result'))[:1200]}"
                )
            )
        return "\n".join(chunks)

    @staticmethod
    def _end_node(state: ReflexionState) -> ReflexionState:
        return state

    @staticmethod
    def _route_after_revisor(state: ReflexionState) -> Literal["execute_tools", "end_node"]:
        if state["tool_call_count"] >= state["max_tool_calls"]:
            return "end_node"
        if state.get("pending_search_queries"):
            return "execute_tools"
        return "end_node"

    def build(self):
        graph = StateGraph(ReflexionState)

        graph.add_node("start_node", self._start_node)
        graph.add_node("responder", self._responder_node)
        graph.add_node("execute_tools", self._execute_tools_node)
        graph.add_node("revisor", self._revisor_node)
        graph.add_node("end_node", self._end_node)

        graph.add_edge(START, "start_node")
        graph.add_edge("start_node", "responder")
        graph.add_edge("responder", "execute_tools")
        graph.add_edge("execute_tools", "revisor")
        graph.add_conditional_edges(
            "revisor",
            self._route_after_revisor,
            {
                "execute_tools": "execute_tools",
                "end_node": "end_node",
            },
        )
        graph.add_edge("end_node", END)

        return graph.compile()
