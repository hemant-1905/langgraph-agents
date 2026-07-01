from dotenv import load_dotenv
load_dotenv()

from graph.graph import app
from graph import graph

if __name__ == "__main__":
    print("Starting the LangGraph application...")
    print(app.invoke({"question": "agent memory"}))
