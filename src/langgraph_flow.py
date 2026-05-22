from typing import TypedDict
from langgraph.graph import StateGraph, END
import json


# ================================
# 1. STATE DEFINITION
# ================================
class GraphState(TypedDict):
    query: str
    context: str
    answer: str
    feedback: str
    score: float
    iteration: int


# ================================
# 2. GRAPH BUILDER
# ================================
def build_graph(retriever, llm):

    builder = StateGraph(GraphState)

    # ----------------------------
    # 🔍 RETRIEVE
    # ----------------------------
    def retrieve(state: GraphState):
        docs = retriever.invoke(state["query"])  # ✅ fixed

        if not docs:
            context = "No relevant documents found."
        else:
            context = "\n\n".join([doc.page_content for doc in docs])

        return {**state, "context": context}

    # ----------------------------
    # ✍️ GENERATE
    # ----------------------------
    def generate(state: GraphState):
        prompt = f"""
You are an expert AI assistant.

Answer the question ONLY using the provided context.
If the answer is not in the context, say "I don't know".

Context:
{state['context']}

Question:
{state['query']}
"""

        answer = llm.predict(prompt).strip()

        return {**state, "answer": answer}

    # ----------------------------
    # 🧪 EVALUATE
    # ----------------------------
    def evaluate(state: GraphState):
        eval_prompt = f"""
You are an evaluator.

Evaluate the answer based on:
- correctness
- completeness
- relevance to the question

Question:
{state['query']}

Answer:
{state['answer']}

Return STRICT JSON ONLY:
{{
  "score": float (0 to 1),
  "feedback": "specific improvement suggestion"
}}
"""

        response = llm.predict(eval_prompt)

        try:
            parsed = json.loads(response)
            score = float(parsed.get("score", 0.5))
            feedback = parsed.get("feedback", "No feedback provided")
        except Exception:
            score = 0.5
            feedback = "Failed to parse evaluation output"

        return {
            **state,
            "score": score,
            "feedback": feedback,
            "iteration": state["iteration"] + 1
        }

    # ----------------------------
    # 🔁 IMPROVE
    # ----------------------------
    def improve(state: GraphState):
        improve_prompt = f"""
Improve the answer using the feedback.

Original Question:
{state['query']}

Context:
{state['context']}

Previous Answer:
{state['answer']}

Feedback:
{state['feedback']}

Return a better, more accurate answer.
"""

        improved_answer = llm.predict(improve_prompt).strip()

        return {**state, "answer": improved_answer}

    # ----------------------------
    # 🧠 DECISION
    # ----------------------------
    def decide(state: GraphState):
        if state["score"] >= 0.8:
            return "end"

        if state["iteration"] >= 3:
            return "end"

        return "improve"

    # ================================
    # BUILD GRAPH
    # ================================
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_node("evaluate", evaluate)
    builder.add_node("improve", improve)

    builder.set_entry_point("retrieve")

    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "evaluate")

    builder.add_conditional_edges(
        "evaluate",
        decide,
        {
            "improve": "improve",
            "end": END
        }
    )

    builder.add_edge("improve", "evaluate")

    return builder.compile()