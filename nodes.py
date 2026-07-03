from langgraph.graph import MessagesState
from react import llm, tools
from langgraph.prebuilt import ToolNode

SYSTEM_MESSAGE = """
You are a helpful assistant that can use tools to answer questions.
"""

def agent_reason(state: MessagesState) -> MessagesState:
    
    res = llm.invoke([{"role": "system", "content": SYSTEM_MESSAGE}, *state["messages"]])
    
    return {"messages": [res]}

tool_node = ToolNode(tools)






