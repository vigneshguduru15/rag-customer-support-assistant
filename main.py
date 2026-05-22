from src.rag_pipeline import build_rag_pipeline
from src.langgraph_flow import build_graph
from src.retriever import get_retriever
from src.llm import get_llm

def main():
    vectorstore = build_rag_pipeline(
    "data/customer_support_faq.pdf"
    )

    retriever = get_retriever(vectorstore)
    llm = get_llm()

    graph = build_graph(retriever, llm)

    while True:
        query = input("\nAsk a question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        result = graph.invoke({
            "query": query,
            "context": "",
            "answer": "",
            "feedback": "",
            "score": 0,
            "iteration": 0
        })

        print("\nAnswer:\n", result["answer"])
        print("Score:", result["score"])

if __name__ == "__main__":
    main()