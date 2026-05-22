# Existing function (KEEP this)
def retrieve_chunks(vectorstore, query, k=10):
    """
    Retrieve top-k similar chunks
    """
    results = vectorstore.similarity_search(query, k=k)
    return results


# ✅ ADD THIS (for LangGraph compatibility)
def get_retriever(vectorstore, k=10):
    """
    Returns a retriever object (LangChain style)
    """
    return vectorstore.as_retriever(search_kwargs={"k": k})