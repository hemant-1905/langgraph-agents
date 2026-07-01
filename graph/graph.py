from langgraph.graph import END, StateGraph
from graph.chains import answer_grader, router, hallucination_grader
from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH
from graph.nodes.grade_documents import grade_documents
from graph.nodes.generate import generate
from graph.nodes.retrieve import retrieve
from graph.nodes.web_search import web_search
from graph.state import GraphState

def route_question(state: GraphState) -> str:
    """
    Route the question to the appropriate node based on the current state.

    Args:
        state (GraphState): The current state of the graph.

    Returns:
        str: The next node to route the question to.
    """
    print("ROUTING QUESTION")
    question = state["question"]
    res = router.invoke({"question": question})

    if res.data_source == "websearch":
        print("ROUTING TO WEBSEARCH")
        return WEBSEARCH
    elif res.data_source == "vectorstore":
        print("ROUTING TO RETRIEVE")
        return RETRIEVE
    


def decide_to_generate(state: GraphState) -> str:
    """
    Decide whether to generate a new document based on the current state.

    Args:
        state (GraphState): The current state of the graph.

    Returns:
        str: The next node to route the question to.
    """
    print("DECIDING WHETHER TO GENERATE")
    if state["web_search"]:
        print("DECIDING TO ROUTE TO WEBSEARCH")
        return WEBSEARCH
    else:
        print("DECIDING TO ROUTE TO GENERATE")
        return GENERATE
    


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    """
    Grade the generated document based on its relevance to the question and the retrieved documents.

    Args:
        state (GraphState): The current state of the graph. 

    Returns:
        str: The next node to route the question to.
    """
    print("CHECK HALLUCINATION")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    hallucination_score = hallucination_grader.invoke({"documents": documents, "generation": generation})

    if hallucination_score.binary_score == "yes":   
        print("NOT HALLUNICATING") 
        score = answer_grader.invoke({"question": question, "generation": generation})
        if score.binary_score == "yes":
            print("USEFUL ANSWER")
            return "useful"
        else:
            print("NOT USEFUL ANSWER")
            return "not useful"
    else:
        print("HALLUCINATING")
        return "not supported"
    


workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

workflow.set_conditional_entry_point(route_question, {RETRIEVE: RETRIEVE, WEBSEARCH: WEBSEARCH})
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(GRADE_DOCUMENTS, decide_to_generate, {WEBSEARCH: WEBSEARCH, GENERATE: GENERATE})
workflow.add_conditional_edges(GENERATE, grade_generation_grounded_in_documents_and_question, {"useful": END, "not useful": WEBSEARCH, "not supported": GENERATE})
workflow.add_edge(WEBSEARCH, GENERATE)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")