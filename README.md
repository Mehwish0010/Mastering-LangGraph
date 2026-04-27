# LangGraph Agentic Patterns

Beginner-friendly examples of **Agentic Design Patterns** using [LangGraph](https://github.com/langchain-ai/langgraph) and Google Gemini.

---

## Pattern 1 — Prompt Chaining (`pattern_1_prompt-chaining.py`)

Prompt chaining breaks a complex task into a series of smaller steps. Each step (node) processes the output of the previous one, passing data through a shared state.

```
Input Text --> [Extract Topics] --> [Generate Titles] --> Final Output
```

**How it works:**
1. **Node 1 — Extract Topics**: Takes input text and extracts 2-3 key topics using an LLM.
2. **Node 2 — Generate Titles**: Takes those topics and generates blog title suggestions.

```python
workflow = StateGraph(State)
workflow.add_node("extract_topics", extract_topics)
workflow.add_node("generate_titles", generate_titles)
workflow.set_entry_point("extract_topics")
workflow.add_edge("extract_topics", "generate_titles")
workflow.add_edge("generate_titles", END)
graph = workflow.compile()
```

**Key concepts:**
- **State** — A shared dictionary that all nodes can read from and write to.
- **Nodes** — Python functions that take the state, call an LLM, and return the updated state.
- **Edges** — Define the flow from one node to the next.

**Tests:**

| Test | What it does |
|------|-------------|
| `test_full_graph` | Runs the entire graph end-to-end |
| `test_node1_only` | Tests only the topic extraction node |
| `test_node2_only` | Tests only the title generation node |

```bash
python pattern_1_prompt-chaining.py
```

---

## Pattern 2 — Routing (`pattern_2_routing.py`)

The routing pattern classifies input and sends it down different paths based on the classification.

```
Input Text --> [Classify Sentiment] --positive--> [Positive Handler] --> Output
                                    --negative--> [Negative Handler] --> Output
```

**How it works:**
1. **Classify Node** — Uses the LLM to determine if the input text is positive or negative.
2. **Router** — A conditional edge that directs the flow based on the sentiment.
3. **Positive Handler** — Generates an enthusiastic, encouraging response.
4. **Negative Handler** — Generates a supportive, empathetic response.

```python
workflow.add_conditional_edges(
    "classify",
    router,
    {"positive": "positive", "negative": "negative"}
)
```

**Key concepts:**
- **Conditional Edges** — Route to different nodes based on a function's return value.
- **Router Function** — A simple function that inspects state and returns the next node name.

**Tests:**

| Test | What it does |
|------|-------------|
| `test_positive_routing` | Verifies positive text routes correctly |
| `test_negative_routing` | Verifies negative text routes correctly |
| `test_router_direct` | Tests the router function without LLM |
| `test_classify_node` | Tests the classification node in isolation |
| `test_response_length` | Checks both paths produce meaningful responses |

```bash
python pattern_2_routing.py
```

---

## Project Structure

```
prompt-chaining/
├── pattern_1_prompt-chaining.py  # Pattern 1: Prompt chaining
├── pattern_2_routing.py          # Pattern 2: Routing
├── .env                          # Your API keys (not committed)
├── .gitignore                    # Ignores .env and cache files
└── README.md                     # This file
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

## What to Try Next

- Add a **third node** (e.g., generate a blog outline from the titles)
- Try the **parallel pattern** with LangGraph
- Combine routing + chaining in a single graph
- Explore [LangGraph documentation](https://langchain-ai.github.io/langgraph/)

## Requirements

- Python 3.10+
- Google Gemini API key
- (Optional) LangSmith API key for tracing
