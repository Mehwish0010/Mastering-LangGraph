# pattern_3.py

import os
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)

# State
class State(TypedDict):
    text: str
    outputs: Annotated[list, operator.add]

# Node 1
def summarize(state: State):
    response = llm.invoke(f"Summarize in 1-2 sentences: {state['text']}")
    print("Summary done")
    return {"outputs": [f"Summary: {response.content.strip()}"]}

# Node 2
def critique(state: State):
    response = llm.invoke(f"Give brief critique: {state['text']}")
    print("Critique done")
    return {"outputs": [f"Critique: {response.content.strip()}"]}

# Node 3
def extract_keywords(state: State):
    response = llm.invoke(f"Extract 5 keywords: {state['text']}")
    print("Keywords done")
    return {"outputs": [f"Keywords: {response.content.strip()}"]}

# Node 4
def combine_results(state: State):
    all_outputs = "\n\n".join(state['outputs'])
    response = llm.invoke(f"Synthesize these:\n{all_outputs}")
    print("Combined!")
    return {"outputs": [f"Final: {response.content.strip()}"]}

# Build graph
workflow = StateGraph(State)
workflow.add_node("summarize", summarize)
workflow.add_node("critique", critique)
workflow.add_node("keywords", extract_keywords)
workflow.add_node("combine", combine_results)

workflow.add_edge(START, "summarize")
workflow.add_edge(START, "critique")
workflow.add_edge(START, "keywords")

workflow.add_edge("summarize", "combine")
workflow.add_edge("critique", "combine")
workflow.add_edge("keywords", "combine")
workflow.add_edge("combine", END)

graph = workflow.compile()

# Run
if __name__ == "__main__":
    result = graph.invoke({
        "text": "AI is transforming software development.",
        "outputs": []
    })

    print("\n--- RESULTS ---")
    for output in result["outputs"]:
        print(f"\n{output}")