import os
import tempfile
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader

from dotenv import load_dotenv
load_dotenv()

def pdf_loader(pdf_path:str):
    try:
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        
        print(f"Loaded {len(documents)} document(s) from PDF")
        for i,doc in enumerate(documents):
            print(f"document {i+1} content preview: {doc.page_content[:100]}...")
            print(f"Metadata: {doc.metadata}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    pdf_loader("docs/wisdomoflife01scho.pdf")