from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def load_or_create_vectorstore(chunks=None, persist_directory="./chroma_db"):
    embedding = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # ✅ If DB exists → load it
    try:
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding
        )

        if vectorstore._collection.count() > 0:
            print("✅ Loaded existing ChromaDB")
            return vectorstore

    except Exception:
        pass

    # ❌ Else create new
    print("⚡ Creating new ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=persist_directory
    )

    vectorstore.persist()
    return vectorstore