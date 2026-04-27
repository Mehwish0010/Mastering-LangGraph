# pattern_2.py

import os
from typing import Literal
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

# State
class State(TypedDict):
    text: str
    sentiment: str
    response: str

# Node 1 - Classify
def classify_sentiment(state: State) -> State:
    response = llm.invoke(
        f"Analyze sentiment. Reply ONLY 'positive' or 'negative'.\nText: {state['text']}"
    )
    state["sentiment"] = response.content.strip().lower()
    print(f"Sentiment: {state['sentiment']}")
    return state

# Node 2 - Positive handler
def handle_positive(state: State) -> State:
    response = llm.invoke(
        f"Give an enthusiastic encouraging response to: {state['text']}"
    )
    state["response"] = response.content.strip()
    return state

# Node 3 - Negative handler
def handle_negative(state: State) -> State:
    response = llm.invoke(
        f"Give a supportive empathetic response to: {state['text']}"
    )
    state["response"] = response.content.strip()
    return state

# Router - traffic police
def router(state: State) -> Literal["positive", "negative"]:
    return "positive" if "positive" in state["sentiment"] else "negative"

# Build graph
workflow = StateGraph(State)
workflow.add_node("classify", classify_sentiment)
workflow.add_node("positive", handle_positive)
workflow.add_node("negative", handle_negative)
workflow.set_entry_point("classify")
workflow.add_conditional_edges(
    "classify",
    router,
    {
        "positive": "positive",
        "negative": "negative"
    }
)
workflow.add_edge("positive", END)
workflow.add_edge("negative", END)
graph = workflow.compile()


# ══════════════════════════════
#        LANGSMITH TESTS
# ══════════════════════════════

@traceable(name="test_positive_routing")
def test_positive_routing():
    result = graph.invoke({
        "text": "I just got promoted! Best day ever!",
        "sentiment": "",
        "response": ""
    })
    assert result["sentiment"] == "positive"
    assert result["response"] != ""
    print("Test 1 passed!")

@traceable(name="test_negative_routing")
def test_negative_routing():
    result = graph.invoke({
        "text": "I failed my exam and feel terrible.",
        "sentiment": "",
        "response": ""
    })
    assert result["sentiment"] == "negative"
    assert result["response"] != ""
    print("Test 2 passed!")

@traceable(name="test_router_direct")
def test_router_direct():
    # Test router without LLM
    pos: State = {"text": "", "sentiment": "positive", "response": ""}
    neg: State = {"text": "", "sentiment": "negative", "response": ""}
    assert router(pos) == "positive"
    assert router(neg) == "negative"
    print("Test 3 passed!")

@traceable(name="test_classify_node")
def test_classify_node():
    result = classify_sentiment({
        "text": "This is amazing!",
        "sentiment": "",
        "response": ""
    })
    assert result["sentiment"] in ["positive", "negative"]
    print(f"Test 4 passed! Sentiment: {result['sentiment']}")

@traceable(name="test_response_length")
def test_response_length():
    pos = graph.invoke({"text": "I love this!", "sentiment": "", "response": ""})
    neg = graph.invoke({"text": "I hate this.", "sentiment": "", "response": ""})
    assert len(pos["response"]) > 10
    assert len(neg["response"]) > 10
    print("Test 5 passed!")


# ══════════════════════════════
#            MAIN
# ══════════════════════════════

if __name__ == "__main__":
    print("=" * 50)
    print("  LangSmith Tests — Pattern 2 Routing")
    print("=" * 50)
    print(f"  Tracing : {os.getenv('LANGCHAIN_TRACING_V2')}")
    print(f"  Project : {os.getenv('LANGCHAIN_PROJECT')}")
    print("=" * 50)

    tests = [
        test_positive_routing,
        test_negative_routing,
        test_router_direct,
        test_classify_node,
        test_response_length,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"  Results: {passed} pass  |  {failed} fail")
    print("=" * 50)
    print("\nCheck: https://smith.langchain.com")