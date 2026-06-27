import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    tavily_api_key: str
    langchain_api_key: str | None = None
    langchain_project: str = "reflexion-agent"
    langchain_endpoint: str = "https://eu.api.smith.langchain.com"

    @classmethod
    def from_env(cls) -> "Settings":
        groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        tavily_api_key = os.getenv("TAVILY_API_KEY", "").strip()

        if not groq_api_key:
            raise ValueError("Missing GROQ_API_KEY in environment.")
        if not tavily_api_key:
            raise ValueError("Missing TAVILY_API_KEY in environment.")

        settings = cls(
            groq_api_key=groq_api_key,
            tavily_api_key=tavily_api_key,
            langchain_api_key=os.getenv("LANGCHAIN_API_KEY", "").strip() or None,
            langchain_project=os.getenv("LANGCHAIN_PROJECT", "reflexion-agent").strip(),
        )

        if settings.langchain_api_key:
            os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
            os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project

        return settings
