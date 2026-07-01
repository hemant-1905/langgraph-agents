import os
import dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
dotenv.load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")

from langchain_core.prompts import ChatPromptTemplate

llm=ChatGroq(groq_api_key=groq_api_key,model_name="qwen/qwen3-32b")

class GradeDocuments(BaseModel):
    '''
    Binary Score for relevance check on retrieved documents.
    '''
    binary_score: str = Field(description="Binary score indicating whether the document is relevant or not. Acceptable values are 'yes' or 'no'.")


system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("user", "Question: {question}\n\nDocument: {document}\n\n Please provide a binary score indicating whether the document is relevant to the question. Acceptable values are 'yes' or 'no'."),
    ]
)

structured_llm_grader = llm.with_structured_output(GradeDocuments)

retrieval_grader = grade_prompt_template | structured_llm_grader
