from typing import Any
from graph.state import GraphState
from graph.chains.generation_chain import generation_chain


def generate(state: GraphState) -> dict[str, Any]:
    """
    Generate a response based on the question and retrieved documents.

    Attributes:
        state (GraphState): The current state of the graph, including the question.
    Returns:
        dict: A dictionary containing the graded documents.
    """
    print("GENERATE STARTED")

    question = state["question"]
    documents = state["documents"]

    # Generate a response based on the question and retrieved documents
    generated_response = generation_chain.invoke({"context": documents, "question": question})

    print("GENERATE END")

    return {"generation": generated_response, "question": question, "documents": documents}
    