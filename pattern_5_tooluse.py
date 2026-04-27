# pattern_5.py

import os
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

# State
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Tools
@tool
def calculator(expression: str) -> str:
    """Evaluate mathematical expressions"""
    result = eval(expression, {"__builtins__": {}}, {})
    return f"Result: {result}"

@tool
def get_weather(city: str) -> str:
    """Get weather for a city"""
    data = {
        "San Francisco": "Sunny 72F",
        "London": "Cloudy 60F",
        "Tokyo": "Partly cloudy 68F"
    }
    return data.get(city, f"No data for {city}")

# LLM with tools
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)
tools = [calculator, get_weather]
model_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)

# Agent node
def call_model(state: State):
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Router
def should_continue(state: State) -> str:
    last = state["messages"][-1]
    if hasattr(last, 'tool_calls') and last.tool_calls:
        print(f"Tool called: {last.tool_calls[0]['name']}")
        return "tools"
    return END

# Build graph
workflow = StateGraph(State)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_edge("tools", "agent")
graph = workflow.compile()


# ── Simple tests ──

def test_calculator():
    result = graph.invoke({
        "messages": [HumanMessage(content="What is 25 * 4?")]
    })
    print("Test 1 passed!")
    print(f"Response: {result['messages'][-1].content}")

def test_weather():
    result = graph.invoke({
        "messages": [HumanMessage(content="Weather in London?")]
    })
    print("Test 2 passed!")
    print(f"Response: {result['messages'][-1].content}")

def test_no_tool():
    result = graph.invoke({
        "messages": [HumanMessage(content="Say hello!")]
    })
    print("Test 3 passed!")
    print(f"Response: {result['messages'][-1].content}")


if __name__ == "__main__":
    print("=" * 40)
    print("Tests — Pattern 5 Tool Use")
    print("=" * 40)

    test_calculator()
    test_weather()
    test_no_tool()

    print("\nAll done!")