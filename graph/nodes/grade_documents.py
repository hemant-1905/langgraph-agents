from typing import Any
from graph.state import GraphState
from graph.chains.retrieval_grader import retrieval_grader


def grade_documents(state: GraphState) -> dict[str, Any]:
    """
    Grade the retrieved documents based on their relevance to the question.

    Attributes:
        state (GraphState): The current state of the graph, including the question.
    Returns:
        dict: A dictionary containing the graded documents.
    """
    print("GRADE DOCUMENTS STARTED")
    question = state["question"]
    documents = state["documents"]
    web_search = False  # Initialize web_search variable
    filtered_documents = []

    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_documents.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = True
            continue
    return {"documents": filtered_documents, "question": question, "web_search": web_search}
