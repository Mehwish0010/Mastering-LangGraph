# pattern_7.py

import os
from typing import Literal, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# State
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    next_agent: str

# Router model
class Router(BaseModel):
    next_agent: Literal["weather_agent", "news_agent", "calculator_agent", "__end__"]
    reasoning: str

supervisor_model = llm.with_structured_output(Router)

# Supervisor
def supervisor(state: State):
    last = state['messages'][-1]
    if isinstance(last, HumanMessage):
        prompt = f"""You manage these agents:
        - weather_agent: weather questions
        - news_agent: news questions
        - calculator_agent: math questions

        User: "{last.content}"
        Pick the right agent or __end__ if done."""

        decision = supervisor_model.invoke(prompt)
        print(f"Routing to: {decision.next_agent}")
        print(f"Reason: {decision.reasoning}")
        return {"next_agent": decision.next_agent}
    return {"next_agent": "__end__"}

# Router function
def route_to_agent(state: State):
    return state.get("next_agent", "__end__")

# Weather agent
def weather_agent(state: State):
    response = llm.invoke(
        f"Answer this weather question: {state['messages'][-1].content}"
    )
    print("Weather agent done")
    return {"messages": [AIMessage(content=response.content)]}

# News agent
def news_agent(state: State):
    response = llm.invoke(
        f"Answer this news question: {state['messages'][-1].content}"
    )
    print("News agent done")
    return {"messages": [AIMessage(content=response.content)]}

# Calculator agent
def calculator_agent(state: State):
    response = llm.invoke(
        f"Solve this math problem: {state['messages'][-1].content}"
    )
    print("Calculator agent done")
    return {"messages": [AIMessage(content=response.content)]}

# Build graph
workflow = StateGraph(State)
workflow.add_node("supervisor", supervisor)
workflow.add_node("weather_agent", weather_agent)
workflow.add_node("news_agent", news_agent)
workflow.add_node("calculator_agent", calculator_agent)

workflow.add_edge(START, "supervisor")
workflow.add_conditional_edges(
    "supervisor",
    route_to_agent,
    {
        "weather_agent": "weather_agent",
        "news_agent": "news_agent",
        "calculator_agent": "calculator_agent",
        "__end__": END
    }
)
workflow.add_edge("weather_agent", END)
workflow.add_edge("news_agent", END)
workflow.add_edge("calculator_agent", END)

graph = workflow.compile()


# ── Simple tests ──

def test_weather():
    result = graph.invoke({
        "messages": [HumanMessage(content="Weather in Tokyo?")],
        "next_agent": ""
    })
    print(f"\nTest 1: {result['messages'][-1].content[:80]}")

def test_calculator():
    result = graph.invoke({
        "messages": [HumanMessage(content="What is 234 * 67?")],
        "next_agent": ""
    })
    print(f"\nTest 2: {result['messages'][-1].content[:80]}")

def test_news():
    result = graph.invoke({
        "messages": [HumanMessage(content="Any tech news today?")],
        "next_agent": ""
    })
    print(f"\nTest 3: {result['messages'][-1].content[:80]}")

def test_end():
    result = graph.invoke({
        "messages": [HumanMessage(content="Thanks goodbye!")],
        "next_agent": ""
    })
    # __end__ pe jaega — sirf 1 message hoga
    print(f"\nTest 4: next_agent = {result['next_agent']}")


if __name__ == "__main__":
    print("=" * 40)
    print("Tests — Pattern 7 Multi Agent")
    print("=" * 40)

    test_weather()
    test_calculator()
    test_news()
    test_end()

    print("\nAll done!")
    