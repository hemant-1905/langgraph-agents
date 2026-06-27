from typing import List
from pydantic import BaseModel, Field

class Reflection(BaseModel):
    """
    Schema for reflection data.
    """
    missing: str = Field(..., description="The missing information in the reflection.")
    superfluous: str = Field(..., description="The superfluous information in the reflection.")

class AnswerQuestion(BaseModel):
    """
    Schema for the request to answer a question.
    """
    answer: str = Field(..., description="The answer to the question.")
    reflection: Reflection = Field(..., description="The reflection data for the previous answer.")
    search_queries: List[str] = Field(..., description="Recommended search queries to research information and improve the answer.")

class ReviseAnswer(AnswerQuestion):
    """
    Schema for the request to revise an answer.
    """
    references: List[str] = Field(..., description="List of references used in the revised answer.")

