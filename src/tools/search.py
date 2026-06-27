import json
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from tavily import TavilyClient


class TavilySearchInput(BaseModel):
    query: str = Field(..., description="Search query to run on Tavily")


class TavilySearchTools:
    def __init__(self, tavily_api_key: str, max_results: int = 3) -> None:
        self._client = TavilyClient(api_key=tavily_api_key)
        self._max_results = max_results

        self.responder_search_tool = StructuredTool.from_function(
            func=self._responder_search,
            name="responder_tavily_search",
            description="Tavily search used after responder output.",
            args_schema=TavilySearchInput,
        )
        self.revisor_search_tool = StructuredTool.from_function(
            func=self._revisor_search,
            name="revisor_tavily_search",
            description="Tavily search used after revisor output.",
            args_schema=TavilySearchInput,
        )

    def _search(self, query: str) -> str:
        try:
            result: dict[str, Any] = self._client.search(
                query=query,
                max_results=self._max_results,
                include_answer=True,
            )
            return json.dumps(result)
        except Exception as exc:
            return json.dumps({"error": str(exc), "query": query})

    def _responder_search(self, query: str) -> str:
        return self._search(query)

    def _revisor_search(self, query: str) -> str:
        return self._search(query)

    def tools(self) -> list[StructuredTool]:
        return [self.responder_search_tool, self.revisor_search_tool]
