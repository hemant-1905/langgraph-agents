import os
import dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

dotenv.load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")

llm=ChatGroq(groq_api_key=groq_api_key,model_name="qwen/qwen3-32b")

# Inline equivalent of hub.pull("rlm/rag-prompt")
prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "You are an assistant for question-answering tasks. "
         "Use the following pieces of retrieved context to answer the question. "
         "If you don't know the answer, just say that you don't know. "
         "Use three sentences maximum and keep the answer concise.\n"
         "Question: {question}\nContext: {context}\nAnswer:"),
    ]
)

generation_chain = prompt | llm | StrOutputParser()
