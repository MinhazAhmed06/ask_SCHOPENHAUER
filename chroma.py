import chromadb
import uuid
import json

with open('rag_chapters_output.json', 'r') as file:
    rag_chapters_output = json.load(file)

chroma_client = chromadb.PersistentClient(path="./chroma_data")
collection = chroma_client.get_or_create_collection(name="WisdomOfLife")

collection.add(
    ids = [str(uuid.uuid4()) for chapter in rag_chapters_output],
    documents = [chapter['chapter_content'] for chapter in rag_chapters_output],
    metadatas = [chapter['metadata'] for chapter in rag_chapters_output]
)

def chroma_search(query):
    results = collection.query(
        query_texts = [query],
        n_results = 2
    )
    for doc, dist in zip(results["documents"][0], results["distances"][0]):
        print(f"\n\n{doc} - distance {dist:.4f}")

def metadata_filter(query):
    results = collection.query(
        query_texts = [query],
        n_results = 3,
        # where = {'chapter':'On Suicide'},
        # where_document = {'$contains':'suicide'}
    )
    for doc in zip(results["documents"][0]):
        print(f"\n\n{doc}")

chroma_search('why do people commit suicide')
# metadata_filter(' ')
    