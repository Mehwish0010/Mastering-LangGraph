#  Pattern 01 _Prompt Chaining with LangGraph

A beginner-friendly example of the **Prompt Chaining** pattern using [LangGraph](https://github.com/langchain-ai/langgraph) and Google Gemini.

## What is Prompt Chaining?

Prompt chaining is an agentic design pattern where you break a complex task into a series of smaller steps. Each step (node) processes the output of the previous one, passing data through a shared state.

```
Input Text --> [Extract Topics] --> [Generate Titles] --> Final Output
```

In this example:
1. **Node 1 — Extract Topics**: Takes input text and extracts 2-3 key topics using an LLM.
2. **Node 2 — Generate Titles**: Takes those topics and generates blog title suggestions.

## How LangGraph Works Here

LangGraph lets you build workflows as **graphs** — with nodes (functions) and edges (connections between them).

```python
# Define shared state
class State(TypedDict):
    text: str       # input text
    topics: str     # output from node 1
    title: str      # output from node 2

# Build the graph
workflow = StateGraph(State)
workflow.add_node("extract_topics", extract_topics)   # step 1
workflow.add_node("generate_titles", generate_titles)  # step 2
workflow.set_entry_point("extract_topics")             # start here
workflow.add_edge("extract_topics", "generate_titles") # connect steps
workflow.add_edge("generate_titles", END)              # finish
graph = workflow.compile()
```

**Key concepts:**
- **State** — A shared dictionary that all nodes can read from and write to.
- **Nodes** — Python functions that take the state, do something (like call an LLM), and return the updated state.
- **Edges** — Define the flow from one node to the next.
- **Entry Point** — Which node runs first.

## Project Structure

```
prompt-chaining/
├── pattern_1.py    # Main code with graph + tests
├── .env            # Your API keys (not committed)
├── .gitignore      # Ignores .env and cache files
└── README.md       # This file
```

## Setup

### 1. Install dependencies

```bash
pip install langgraph langchain-google-genai langsmith python-dotenv
```

### 2. Create a `.env` file

```
GOOGLE_API_KEY=your-google-api-key-here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-api-key-here
LANGCHAIN_PROJECT=pattern-1-test
```

- **Google API Key**: Get one from [Google AI Studio](https://aistudio.google.com/apikey)
- **LangSmith API Key** (optional): Get one from [smith.langchain.com](https://smith.langchain.com) for tracing/debugging

### 3. Run the code

```bash
python pattern_1.py
```

### Expected Output

```
Starting tests...

Topics: 1. AI Workflows  2. Workflow Building  3. Powerful Capabilities
Titles done!
Test 1 passed!
Topics: 1. Transformative Power of AI  2. Societal Impact  3. Future Implications
Test 2 passed!
Titles done!
Test 3 passed!

All tests done!
Check: https://smith.langchain.com
```

## Tests Included

The script includes 3 tests using LangSmith tracing:

| Test | What it does |
|------|-------------|
| `test_full_graph` | Runs the entire graph end-to-end |
| `test_node1_only` | Tests only the topic extraction node |
| `test_node2_only` | Tests only the title generation node |

You can view detailed traces at [smith.langchain.com](https://smith.langchain.com) if you have LangSmith set up.

## What to Try Next

- Add a **third node** (e.g., generate a blog outline from the titles)
- Add **conditional edges** to route based on the number of topics found
- Try the **routing pattern** or **parallel pattern** with LangGraph
- Explore [LangGraph documentation](https://langchain-ai.github.io/langgraph/)

## Requirements

- Python 3.10+
- Google Gemini API key
- (Optional) LangSmith API key for tracing
