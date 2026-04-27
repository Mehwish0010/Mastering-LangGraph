# pattern_4.py

import os
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)

class State(TypedDict):
    task: str
    draft: str
    feedback: str
    iteration: int
    final: str

def generate_draft(state: State):
    iteration = state.get('iteration', 0) + 1
    if iteration == 1:
        prompt = f"Write a response to: {state['task']}"
    else:
        prompt = f"Improve this:\nDraft: {state['draft']}\nFeedback: {state['feedback']}"
    response = llm.invoke(prompt)
    print(f"Iteration {iteration}: Draft done")
    return {"draft": response.content.strip(), "iteration": iteration}

def evaluate_draft(state: State):
    prompt = f"""Task: {state['task']}
    Draft: {state['draft']}
    If excellent reply: APPROVED — otherwise give feedback."""
    response = llm.invoke(prompt)
    print(f"Feedback: {response.content[:80]}")
    return {"feedback": response.content.strip()}

def finalize_output(state: State):
    return {"final": state["draft"]}

def decide_next(state: State) -> str:
    if "APPROVED" in state["feedback"].upper():
        return "finalize"
    elif state.get('iteration', 0) >= 3:
        return "finalize"
    else:
        return "refine"

workflow = StateGraph(State)
workflow.add_node("generate", generate_draft)
workflow.add_node("evaluate", evaluate_draft)
workflow.add_node("finalize", finalize_output)
workflow.set_entry_point("generate")
workflow.add_edge("generate", "evaluate")
workflow.add_conditional_edges(
    "evaluate",
    decide_next,
    {"refine": "generate", "finalize": "finalize"}
)
workflow.add_edge("finalize", END)
graph = workflow.compile()


# ── 3 simple tests ──

@traceable(name="test_full_loop")
def test_full_loop():
    result = graph.invoke({"task": "Explain AI in 2 sentences.", "iteration": 0})
    assert result["final"] != ""
    print("Test 1 passed!")

@traceable(name="test_generate_node")
def test_generate_node():
    result = generate_draft({"task": "Explain AI.", "draft": "", "feedback": "", "iteration": 0, "final": ""})
    assert result["draft"] != ""
    assert result["iteration"] == 1
    print("Test 2 passed!")

@traceable(name="test_decide_next")
def test_decide_next():
    approved = {"task": "", "draft": "", "feedback": "APPROVED", "iteration": 1, "final": ""}
    refine   = {"task": "", "draft": "", "feedback": "Needs work", "iteration": 1, "final": ""}
    maxed    = {"task": "", "draft": "", "feedback": "Needs work", "iteration": 3, "final": ""}
    assert decide_next(approved) == "finalize"
    assert decide_next(refine)   == "refine"
    assert decide_next(maxed)    == "finalize"
    print("Test 3 passed!")


if __name__ == "__main__":
    print("=" * 40)
    print("Tests — Pattern 4 Reflection")
    print("=" * 40)

    test_full_loop()
    test_generate_node()
    test_decide_next()

    print("\nAll tests done!")
    print("Check: https://smith.langchain.com")