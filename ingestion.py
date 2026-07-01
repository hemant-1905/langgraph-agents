from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
import os
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

load_dotenv()
os.environ.setdefault("USER_AGENT", "langgraph-agents-ingestion/1.0")

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
# all_chunks = []

# for url in urls:
#     loader = WebBaseLoader(url)
#     documents = loader.load()
#     chunks = text_splitter.split_documents(documents)
#     all_chunks.extend(chunks)



# if __name__ == "__main__":
#     chunks = run_ingestion()
#     print(f"Ingestion completed. Total chunks: {len(chunks)}") 
#     print(f"Type of chunks: {type(chunks)}")  # Print the type of the first chunk

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# vectorstore = Chroma(
#     collection_name="lilianweng",
#     embedding_function=embeddings,
#     persist_directory="./rag_embeddings/db",
# )

# # Index once so retrieval tests have data without duplicating documents on every import.
# if vectorstore._collection.count() == 0 and all_chunks:
#     vectorstore.add_documents(all_chunks)

# retriever = vectorstore.as_retriever()

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# vectorstore = Chroma.from_documents(
#     documents=doc_splits,
#     collection_name="rag-chroma",
#     embedding=OpenAIEmbeddings(),
#     persist_directory="./.chroma",
# )

retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory="./.chroma",
    embedding_function=embeddings,
).as_retriever()

# Seed the collection once if empty so tests have retrievable data.
_vectorstore = retriever.vectorstore
if _vectorstore._collection.count() == 0 and doc_splits:
    _vectorstore.add_documents(doc_splits)