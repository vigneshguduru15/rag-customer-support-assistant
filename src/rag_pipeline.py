from src.loader import load_pdf
from src.chunker import chunk_documents
from src.embeddings import load_or_create_vectorstore
from src.retriever import retrieve_chunks
from src.llm import call_llm


def build_rag_pipeline(file_path: str):
    documents = load_pdf(file_path)
    chunks = chunk_documents(documents)
    vectorstore = load_or_create_vectorstore(chunks)
    return vectorstore


def generate_answer(vectorstore, query):
    docs = retrieve_chunks(vectorstore, query)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
Answer based on the context below.
If the answer is partially available, try to explain.

Context:
{context}

Question: {query}
"""

    return call_llm(prompt)