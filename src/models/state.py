from typing import Any, Literal, TypedDict

from src.models.schemas import ResponderOutputModel, RevisedOutputModel


class ReflexionState(TypedDict, total=False):
    user_input: str
    responder_output: ResponderOutputModel | None
    revised_output: RevisedOutputModel | None
    pending_search_queries: list[str]
    search_traces: list[dict[str, Any]]
    tool_call_count: int
    max_tool_calls: int
    tool_phase: Literal["responder", "revisor"]
