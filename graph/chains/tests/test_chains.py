from dotenv import load_dotenv
from graph.chains.router import RouteQuery, question_router
from graph.chains import generation_chain, hallucination_grader
from graph.chains.retrieval_grader import retrieval_grader, GradeDocuments
from ingestion import retriever

load_dotenv()


def test_retrival_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )

    assert res.binary_score == "yes"


def test_retrival_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": "Will India dominate the world order?", "document": doc_txt}
    )
    assert res.binary_score == "no"


def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})

def hallucination_retrival_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    generation = generation_chain.invoke({"context": docs, "question": question})
    response = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )
    
    assert response.binary_score == "yes"


def hallucination_retrival_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    res = hallucination_grader.invoke(
        {
            "documents": docs,
            "generation": "In order to make pizza we need to first start with the dough",
        }
    )
    
    assert res.binary_score == "no"

def test_router_to_websearch() -> None:
    question = "Will India dominate the world order?"
    query = question_router.invoke({"question": question})
    assert query.data_source == "websearch"

def test_router_to_vectorstore() -> None:
    question = "What is agent memory?"
    query = question_router.invoke({"question": question})
    assert query.data_source == "vectorstore"