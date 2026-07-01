from typing import Any
from ingestion import retriever
from graph.state import GraphState


def retrieve(state: GraphState) -> dict[str, Any]:
    """
    Retrieve relevant information from the knowledge base based on the query.

    Attributes:
        state (GraphState): The current state of the graph, including the question.
    Returns:
        dict: A dictionary containing the retrieved information.
    """
    
    print("RETRIEVE STARTED")
    question = state["question"]
    retrieved_docs = retriever.invoke(question)
    print("RETRIEVE END")
    return {"documents": retrieved_docs, "question": question}