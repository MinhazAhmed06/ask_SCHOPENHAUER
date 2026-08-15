from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openrouter import ChatOpenRouter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_experimental.text_splitter import SemanticChunker
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import json
import tempfile

load_dotenv()

with open('rag_chapters_output.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

KNOWLEDGE_BASE = list(chapter["chapter_content"] for chapter in data)
embedding_model = HuggingFaceEmbeddings()

def rec_knowbase():
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    doc = Document(page_content=KNOWLEDGE_BASE[0])
    chunks = splitter.split_documents([doc])

    vector_store = Chroma.from_documents(
        documents = chunks,
        embedding = embedding_model,
        persist_directory = tempfile.mkdtemp()
    )

    return vector_store

def basic_rag():
    vector_store = rec_knowbase()
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':2})
    llm = ChatOpenRouter(
        model = 'nvidia/nemotron-3.5-lightning:free',
        temperature = 0.2
    )

    prompt = ChatPromptTemplate.from_template(
"""
Answer the question based on the following context:
{context}

Question: {question}

Answer:

make sure to answer in a precise manner and if you dont know the answer just say "I don't know".
"""        
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context":retriever | format_docs, "question":RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    question = "what does Voltaire say?"
    answer = rag_chain.invoke(question)
    print(answer)

basic_rag()