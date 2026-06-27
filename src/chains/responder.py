from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.models.schemas import ResponderOutputModel


class ResponderChain:
    def __init__(self, groq_api_key: str, model_name: str) -> None:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert analyst for AI security operations products. "
                    "Generate an initial short summary and self-critique. "
                    "Return structured output with fields: response, critique, search. "
                    "Keep response under 170 words.",
                ),
                (
                    "human",
                    "Task: {user_input}\n"
                    "Provide:\n"
                    "1) response: a concise summary\n"
                    "2) critique.superfluous: list of unnecessary claims\n"
                    "3) critique.assumptions: list of assumptions lacking evidence\n"
                    "4) search: 1-3 focused web search queries",
                ),
            ]
        )

        llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_name, temperature=0.2)
        self._chain = prompt | llm.with_structured_output(ResponderOutputModel)

    def invoke(self, user_input: str) -> ResponderOutputModel:
        return self._chain.invoke({"user_input": user_input})
