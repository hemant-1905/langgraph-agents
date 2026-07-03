
from langchain.tools import tool
from langchain_groq import ChatGroq
import os
import dotenv
dotenv.load_dotenv()
from langchain_tavily import TavilySearch

groq_api_key=os.getenv("GROQ_API_KEY")

@tool
def triple_temperature(float_value: float) -> float:
    """
    Triple the input float value.

    Args:
        float_value (float): The input float value to be tripled.

    Returns:
        float: The tripled value.
    """
    return float_value * 3

tools = [TavilySearch(max_results=1), triple_temperature]

llm=ChatGroq(groq_api_key=groq_api_key,model_name="qwen/qwen3-32b").bind_tools(tools)


      

