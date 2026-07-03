import os
import uuid
import sqlite3

from  dotenv import load_dotenv
from langchain.messages import HumanMessage
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from nodes import agent_reason, tool_node
load_dotenv()
import os

os.environ["LANGCHAIN_ENDPOINT"] = "https://eu.api.smith.langchain.com"
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

AGENT_REASON = "agent_reason"
ACT = "act"
LAST = -1

def should_continue(state: MessagesState) -> str:
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT

flow = StateGraph(MessagesState)

flow.add_node(AGENT_REASON, agent_reason)
flow.add_node(ACT, tool_node)
flow.add_edge(ACT, AGENT_REASON)
flow.add_conditional_edges(AGENT_REASON, should_continue, 
              {ACT: ACT, 
               END: END})

flow.set_entry_point(AGENT_REASON)

# SQLite on-disk memory persistence
db_conn = sqlite3.connect("memory.db", check_same_thread=False)
memory = SqliteSaver(db_conn)
app = flow.compile(checkpointer=memory)
app.get_graph().draw_mermaid_png(output_file_path="agent_reason.png")

if __name__ == "__main__":
    # Use a fixed thread_id to continue the same session, or generate a new one
    session_id = os.getenv("SESSION_ID", str(uuid.uuid4()))
    config = {"configurable": {"thread_id": session_id}}
    print(f"Session ID: {session_id}")

    res = app.invoke(
        {"messages": [HumanMessage(content="What is the temperature in tokyo? List it and triple it.")]},
        config=config,
    )
    print(res["messages"][LAST].content)