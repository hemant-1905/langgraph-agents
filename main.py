import os

from dotenv import load_dotenv
load_dotenv()

from graph.graph import app
from graph import graph
import os

os.environ["LANGCHAIN_ENDPOINT"] = "https://eu.api.smith.langchain.com"
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

if __name__ == "__main__":
    print("Starting the LangGraph application...")
    print(app.invoke({"question": "agent memory"}))

    ##try this question for the other flow: "Will India dominate the world order?"
