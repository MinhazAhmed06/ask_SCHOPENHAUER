from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openrouter import ChatOpenRouter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

import json
import tempfile
from dotenv import load_dotenv
load_dotenv()

with open('rag_chapters_output.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

documents = [
    Document(
        page_content=item["chapter_content"],
        metadata=item["metadata"], 
    )
    for item in data
]

embedding_model = HuggingFaceEmbeddings()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", " ", ""])
chunks = splitter.split_documents(documents)

vector_store = Chroma.from_documents(
    documents = chunks,
    embedding = embedding_model,
    persist_directory = tempfile.mkdtemp()
)

vector_retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)

bm25_retriever = BM25Retriever.from_documents(
    documents=chunks,
    k=5
)

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever,vector_retriever],
    weights=[0.5,0.5]
)


def test(retriever):
    question = "what does Voltaire say?"
    results = retriever.invoke(question)
    for i, doc in enumerate(results):
        preview = doc.page_content[0:100] + "..."
        print(f'{i+1} = {preview}')
    return results

# test(vector_retriever)
# test(bm25_retriever)
test(ensemble_retriever)