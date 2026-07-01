import os
import dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
dotenv.load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")

from langchain_core.prompts import ChatPromptTemplate

llm=ChatGroq(groq_api_key=groq_api_key,model_name="qwen/qwen3-32b")

class HallucinationGrader(BaseModel):
    '''
    Binary Score for hallucination check on generated answer.
    '''
    binary_score: str = Field(description="Binary score indicating whether the answer is hallucinated or not. Acceptable values are 'yes' or 'no'.")


system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

hallucination_grade_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("user", "Set of facts: {documents}\n\Generated Answer: {generation}"),
    ]
)

hallucination_structured_llm_grader = llm.with_structured_output(HallucinationGrader)

hallucination_grader = hallucination_grade_prompt_template | hallucination_structured_llm_grader
