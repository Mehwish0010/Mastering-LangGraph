import os
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()

# Setup LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7
)

# State - shared memory between nodes
class State(TypedDict):
    text: str
    topics: str
    title: str

# Node 1
def extract_topics(state: State) -> State:
    response = llm.invoke(f"Extract 2-3 key topics from: {state['text']}")
    state["topics"] = response.content.strip()
    print(f"Topics: {state['topics']}")
    return state

# Node 2
def generate_titles(state: State) -> State:
    response = llm.invoke(f"Generate 3 blog titles for: {state['topics']}")
    state["title"] = response.content.strip()
    print(f"Titles done!")
    return state

# Build graph
workflow = StateGraph(State)
workflow.add_node("extract_topics", extract_topics)
workflow.add_node("generate_titles", generate_titles)
workflow.set_entry_point("extract_topics")
workflow.add_edge("extract_topics", "generate_titles")
workflow.add_edge("generate_titles", END)
graph = workflow.compile()

# Tests with LangSmith tracing
@traceable(name="test_full_graph")
def test_full_graph():
    result = graph.invoke({"text": "LangGraph builds powerful AI workflows."})
    assert result["topics"] != ""
    assert result["title"] != ""
    print("Test 1 passed!")

@traceable(name="test_node1_only")
def test_node1_only():
    result = extract_topics({"text": "AI is changing the world.", "topics": "", "title": ""})
    assert result["topics"] != ""
    print("Test 2 passed!")

@traceable(name="test_node2_only")
def test_node2_only():
    result = generate_titles({"text": "", "topics": "AI, Machine Learning", "title": ""})
    assert result["title"] != ""
    print("Test 3 passed!")

# Run all tests
if __name__ == "__main__":
    print("Starting tests...\n")
    test_full_graph()
    test_node1_only()
    test_node2_only()
    print("\nAll tests done!")
    print("Check: https://smith.langchain.com")