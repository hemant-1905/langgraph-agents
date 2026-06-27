from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.models.schemas import RevisedOutputModel


class RevisorChain:
    def __init__(self, groq_api_key: str, model_name: str) -> None:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a strict reviser. Improve factual grounding using tool evidence. "
                    "Return structured output with fields: response, critique, search. "
                    "If evidence is enough, return an empty search list.",
                ),
                (
                    "human",
                    "Original task: {user_input}\n\n"
                    "Current draft:\n{draft_response}\n\n"
                    "Current critique:\n{critique}\n\n"
                    "Tool evidence:\n{search_context}\n\n"
                    "Revise the draft into a short summary about AI-powered SOC and leading companies.",
                ),
            ]
        )

        llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_name, temperature=0.1)
        self._chain = prompt | llm.with_structured_output(RevisedOutputModel)

    def invoke(
        self,
        *,
        user_input: str,
        draft_response: str,
        critique: str,
        search_context: str,
    ) -> RevisedOutputModel:
        return self._chain.invoke(
            {
                "user_input": user_input,
                "draft_response": draft_response,
                "critique": critique,
                "search_context": search_context,
            }
        )
