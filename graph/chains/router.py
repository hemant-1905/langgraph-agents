from pydantic import BaseModel, Field
from typing import Literal
import os
import dotenv
from langchain_groq import ChatGroq
dotenv.load_dotenv()
from langchain_core.prompts import ChatPromptTemplate

groq_api_key=os.getenv("GROQ_API_KEY")

llm=ChatGroq(groq_api_key=groq_api_key,model_name="qwen/qwen3-32b")


class RouteQuery(BaseModel):
    
    data_source: Literal["websearch", "vectorstore"] = Field(description="The data source to use for the query. Acceptable values are 'websearch' or 'vectorstore'.")
    


system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. For all else, use web-search."""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

structured_llm_router = llm.with_structured_output(RouteQuery)

router = route_prompt | structured_llm_router
