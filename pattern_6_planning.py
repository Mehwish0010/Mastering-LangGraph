# pattern_6.py

import os
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# Custom reducer
def merge_dicts(left: dict, right: dict) -> dict:
    return {**left, **right}

# State
class State(TypedDict):
    task: str
    plan: list[str]
    research_results: Annotated[dict, merge_dicts]
    final_output: str

# Node 1 - Plan
def create_plan(state: State):
    response = llm.invoke(f"Create 3-5 step plan for: {state['task']}")
    steps = [
        line.strip()
        for line in response.content.split('\n')
        if line.strip() and any(c.isdigit() for c in line[:3])
    ]
    print(f"Plan: {steps}")
    return {"plan": steps}

# Node 2 - Technical research
def research_technology(state: State):
    response = llm.invoke(f"Research technical aspects of: {state['task']}")
    print("Technical research done")
    return {"research_results": {"technical": response.content}}

# Node 3 - Market research
def research_market(state: State):
    response = llm.invoke(f"Research market aspects of: {state['task']}")
    print("Market research done")
    return {"research_results": {"market": response.content}}

# Node 4 - Synthesize
def synthesize_report(state: State):
    prompt = f"""Write a 3 paragraph report:
    Task: {state['task']}
    Technical: {state['research_results'].get('technical', '')}
    Market: {state['research_results'].get('market', '')}"""
    response = llm.invoke(prompt)
    print("Report done!")
    return {"final_output": response.content}

# Build graph
workflow = StateGraph(State)
workflow.add_node("planner", create_plan)
workflow.add_node("research_tech", research_technology)
workflow.add_node("research_market", research_market)
workflow.add_node("synthesize", synthesize_report)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "research_tech")
workflow.add_edge("planner", "research_market")
workflow.add_edge("research_tech", "synthesize")
workflow.add_edge("research_market", "synthesize")
workflow.add_edge("synthesize", END)

graph = workflow.compile()

# Run
if __name__ == "__main__":
    result = graph.invoke({
        "task": "Analyze impact of AI on software development",
        "research_results": {}
    })

    print("\n--- FINAL REPORT ---")
    print(result["final_output"])