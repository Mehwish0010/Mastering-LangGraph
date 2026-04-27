# LangGraph Agentic Patterns

Beginner-friendly examples of **Agentic Design Patterns** using [LangGraph](https://github.com/langchain-ai/langgraph) and Google Gemini and Groq.

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

## Pattern 3 — Parallelization (`pattern_3_parallelization.py`)

The parallelization pattern runs multiple LLM calls at the same time (fan-out), then combines the results in a single step (fan-in).

```
              ┌--> [Summarize]  --┐
Input Text -->├--> [Critique]   --├--> [Combine] --> Final Output
              └--> [Keywords]   --┘
```

**How it works:**
1. **Summarize Node** — Generates a 1-2 sentence summary of the input.
2. **Critique Node** — Provides a brief critique of the input.
3. **Keywords Node** — Extracts 5 keywords from the input.
4. **Combine Node** — Synthesizes all three outputs into a final result.

```python
# Fan-out: all three run in parallel from START
workflow.add_edge(START, "summarize")
workflow.add_edge(START, "critique")
workflow.add_edge(START, "keywords")

# Fan-in: all three feed into combine
workflow.add_edge("summarize", "combine")
workflow.add_edge("critique", "combine")
workflow.add_edge("keywords", "combine")
```

**Key concepts:**
- **Fan-out / Fan-in** — Multiple nodes start from the same point and converge at a single node.
- **Annotated State with `operator.add`** — Allows multiple nodes to append to the same list without overwriting each other.

```bash
python pattern_3_parallelization.py
```

---

## Pattern 4 — Reflection (`pattern_4_reflection.py`)

The reflection pattern generates a draft, evaluates it, and iteratively refines it based on feedback — looping until the output is approved or a max iteration limit is reached.

```
[Generate Draft] --> [Evaluate] --APPROVED--> [Finalize] --> Output
                        |                         ↑
                        └──needs work──> [Generate Draft] (retry)
```

**How it works:**
1. **Generate Node** — Writes a draft (or improves it using previous feedback).
2. **Evaluate Node** — Reviews the draft and either approves it or gives feedback.
3. **Decide Next** — Routes to finalize if approved or max iterations (3) reached, otherwise loops back.
4. **Finalize Node** — Returns the final approved draft.

```python
workflow.add_conditional_edges(
    "evaluate",
    decide_next,
    {"refine": "generate", "finalize": "finalize"}
)
```

**Key concepts:**
- **Iterative Looping** — The graph can loop back to a previous node based on conditions.
- **Max Iterations** — A safety cap to prevent infinite loops.

**Tests:**

| Test | What it does |
|------|-------------|
| `test_full_loop` | Runs the full generate-evaluate loop end-to-end |
| `test_generate_node` | Tests the generate node in isolation |
| `test_decide_next` | Tests the routing logic without LLM |

```bash
python pattern_4_reflection.py
```

---

## Pattern 5 — Tool Use (`pattern_5_tooluse.py`)

The tool use pattern gives the LLM access to external tools (functions) that it can call when needed, creating a ReAct-style agent loop.

```
[User Message] --> [Agent] --tool call--> [Tool Node] --> [Agent] --> Output
                      |                                      ↑
                      └──no tool needed──> Output             |
                                                              └── (loop if more tools needed)
```

**How it works:**
1. **Agent Node** — The LLM decides whether to call a tool or respond directly.
2. **Tool Node** — Executes the tool and returns the result to the agent.
3. **Router** — Checks if the LLM made a tool call; if yes, routes to tools, otherwise ends.

```python
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_edge("tools", "agent")
```

**Key concepts:**
- **Tool Binding** — `llm.bind_tools(tools)` gives the LLM awareness of available tools.
- **ToolNode** — LangGraph's built-in node that executes tool calls automatically.
- **ReAct Loop** — Agent reasons, acts (calls tool), observes result, and repeats if needed.

**Tools included:**
- `calculator` — Evaluates math expressions
- `get_weather` — Returns mock weather data for cities

**Tests:**

| Test | What it does |
|------|-------------|
| `test_calculator` | Asks a math question, expects tool use |
| `test_weather` | Asks for weather, expects tool use |
| `test_no_tool` | Simple greeting, expects direct response |

```bash
python pattern_5_tooluse.py
```

---

## Pattern 6 — Planning (`pattern_6_planning.py`)

The planning pattern creates a plan first, then executes research tasks in parallel, and synthesizes everything into a final report.

```
[Create Plan] --> [Research Tech]   --┐
              --> [Research Market] --├--> [Synthesize Report] --> Output
```

**How it works:**
1. **Planner Node** — Creates a 3-5 step plan for the given task.
2. **Research Tech Node** — Researches technical aspects (runs in parallel).
3. **Research Market Node** — Researches market aspects (runs in parallel).
4. **Synthesize Node** — Combines all research into a final 3-paragraph report.

```python
workflow.add_edge("planner", "research_tech")
workflow.add_edge("planner", "research_market")
workflow.add_edge("research_tech", "synthesize")
workflow.add_edge("research_market", "synthesize")
```

**Key concepts:**
- **Plan-then-Execute** — The LLM creates a plan before doing the work.
- **Custom Reducer (`merge_dicts`)** — Merges parallel research results into a single dict without overwriting.
- **Parallel Research** — Multiple research nodes fan out from the planner and converge at synthesis.

```bash
python pattern_6_planning.py
```

---

## Project Structure

```
prompt-chaining/
├── pattern_1_prompt-chaining.py  # Pattern 1: Prompt chaining (Gemini)
├── pattern_2_routing.py          # Pattern 2: Routing (Gemini)
├── pattern_3_parallelization.py  # Pattern 3: Parallelization (Gemini)
├── pattern_4_reflection.py       # Pattern 4: Reflection (Groq)
├── pattern_5_tooluse.py          # Pattern 5: Tool use (Groq)
├── pattern_6_planning.py         # Pattern 6: Planning (Groq)
├── .env                          # Your API keys (not committed)
├── .gitignore                    # Ignores .env and cache files
└── README.md                     # This file
```

## Setup

### 1. Install dependencies

```bash
pip install langgraph langchain-google-genai langchain-groq langsmith python-dotenv
```

### 2. Create a `.env` file

```
GOOGLE_API_KEY=your-google-api-key-here
GROQ_API_KEY=your-groq-api-key-here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-api-key-here
LANGCHAIN_PROJECT=pattern-1-test
```

- **Google API Key**: Get one from [Google AI Studio](https://aistudio.google.com/apikey)
- **Groq API Key**: Get one from [console.groq.com](https://console.groq.com)
- **LangSmith API Key** (optional): Get one from [smith.langchain.com](https://smith.langchain.com) for tracing/debugging

## Requirements

- Python 3.10+
- Google Gemini API key (patterns 1-3)
- Groq API key (patterns 4-6)
- (Optional) LangSmith API key for tracing
