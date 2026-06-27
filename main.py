
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from IPython.display import display, Image
import os

load_dotenv()

os.environ["LANGCHAIN_ENDPOINT"] = "https://eu.api.smith.langchain.com"
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")


from langgraph.graph import StateGraph, END, START, add_messages
from chains import generation_chain, reflection_chain
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

REFLECTION = "reflect"
GENERATION = "generate"

def generation_node(state: MessageGraph):
    return {"messages": [generation_chain.invoke({"messages": state["messages"]})]}

def reflection_node(state: MessageGraph):
    res = reflection_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}

builder = StateGraph(MessageGraph)
builder.add_node(GENERATION, generation_node)
builder.add_node(REFLECTION, reflection_node)
##builder.set_entry_point(GENERATION)
builder.add_edge(START, GENERATION)


def should_continue(state: MessageGraph):
    if len(state["messages"]) > 3:
        return END
    return REFLECTION

builder.add_edge(REFLECTION, GENERATION)
builder.add_conditional_edges(
    GENERATION,
    should_continue,
    {
        REFLECTION: REFLECTION,
        END: END,
    },
)

graph = builder.compile()
print(graph.get_graph().draw_mermaid())
graph.get_graph().print_ascii()

# try:
#     display(Image(graph.get_graph().draw_mermaid_png()))
# except Exception as e:
#     print("Could not display graph as PNG. Please ensure you have the required dependencies installed.")
#     print(f"Error: {e}")


if __name__ == "__main__":
    print("Hello LangGraph")
    inputs = {
        "messages": [
            HumanMessage(
                content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post

                                  """
            )
        ]
    }
    # response = graph.invoke(inputs)
    # print(response)

    for event in graph.stream(inputs):
        print("=" * 80)
        print(event)