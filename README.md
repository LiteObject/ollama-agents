# Ollama Multi-Agent Framework

A configurable multi-agent orchestration framework built on Ollama, featuring YAML-based agent configuration, 17+ orchestration patterns, and comprehensive observability.

## What is Ollama?

Ollama is a tool that allows you to run open-source large language models (LLMs) locally on your machine. It supports a variety of models, including Llama 2, Code Llama, Qwen, and others, bundling model weights, configuration, and data into a single package defined by a Modelfile.

## Features

### Core Framework
- **YAML Configuration**: Define agents declaratively in YAML files
- **Plugin System**: Create custom tools with the `@tool` decorator
- **Shared Context**: Inter-agent communication and variable passing
- **17+ Orchestration Patterns**: Sequential, parallel, hierarchical, and more

### Interactive Agent (agent.py)
- **Multi-turn conversations**: Engage in ongoing dialogues with the AI
- **Web search integration**: Agent can search the web and fetch content
- **Robust error handling**: Graceful handling of connection issues
- **Environment variable support**: Configurable via `.env` file

### Observability
- **Structured Logging**: JSON and human-readable log formats
- **Event Hooks**: Subscribe to agent lifecycle events
- **Context Tracking**: Session IDs and duration metrics

## Quick Start

### Installation

1. **Install Ollama**: Visit [ollama.com/download](https://ollama.com/download)

2. **Start Ollama service**:
   ```bash
   ollama serve
   ```

3. **Install required model**:
   ```bash
   ollama pull qwen3:14b
   ```

4. **Set up configuration**:
   ```bash
   cp .env.example .env
   ```
   Edit `.env`:
   ```env
   OLLAMA_API_KEY=your_api_key_here
   OLLAMA_MODEL=qwen3:14b
   ```

5. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Run the Interactive Agent

```bash
python agent.py
```

### Run the Multi-Agent Demo

```bash
python main.py
```

## Architecture

```
ollama-demo/
├── agent.py                    # Interactive single-agent CLI
├── main.py                     # Multi-agent demo entry point
├── core/
│   ├── __init__.py             # Package exports
│   ├── config.py               # AgentConfig dataclass
│   ├── agent.py                # Reusable Agent class
│   ├── context.py              # SharedContext for inter-agent comms
│   ├── tools.py                # @tool decorator and ToolRegistry
│   ├── orchestrator.py         # Multi-agent orchestration patterns
│   └── observability/
│       ├── __init__.py
│       ├── logging.py          # Structured logging
│       └── hooks.py            # Event hook system
├── config/
│   └── agents/
│       ├── researcher.yaml     # Research specialist
│       ├── analyst.yaml        # Analysis specialist
│       └── writer.yaml         # Writing specialist
├── tools/
│   ├── builtin/                # Ollama built-in tools
│   └── custom/                 # Custom tool plugins
├── requirements.txt
└── docker-compose.yml
```

## Usage Examples

### 1. Define an Agent (YAML)

```yaml
# config/agents/researcher.yaml
name: researcher
model: qwen3:14b
description: Research specialist with web access
system_prompt: |
  You are a research specialist. Use web search to find accurate information.
tools:
  - web_search
  - web_fetch
think: true
max_iterations: 10
```

### 2. Create Custom Tools

```python
from core import tool

@tool(name="calculate", description="Evaluate math expression")
def calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression."""
    return str(eval(expression))  # Use ast.literal_eval in production
```

### 3. Orchestrate Multiple Agents

```python
from core import Orchestrator, AgentConfig

# Create orchestrator
orchestrator = Orchestrator()

# Add agents from YAML
orchestrator.add_agent_from_yaml("config/agents/researcher.yaml")
orchestrator.add_agent_from_yaml("config/agents/writer.yaml")

# Run sequential workflow
result = orchestrator.run_sequential([
    ("researcher", "Research the topic: {topic}"),
    ("writer", "Write an article based on: {researcher_result}"),
], initial_vars={"topic": "AI trends 2025"})
```

### 4. Use Event Hooks

```python
from core import AgentEvent, default_hook_registry

def on_agent_complete(event_data):
    print(f"Agent {event_data.agent_name} completed in {event_data.duration_ms}ms")

default_hook_registry.on(AgentEvent.AGENT_END, on_agent_complete)
```

### 5. Configure Logging

```python
from core import configure_logging

# JSON logs to file, colored console output
configure_logging(
    level="INFO",
    log_file="logs/agent.log",
    json_format=False,
    use_colors=True,
)
```

## Orchestration Patterns

| Pattern | Method | Description |
|---------|--------|-------------|
| Sequential | `run_sequential()` | Chain agents, pass outputs |
| Parallel | `run_parallel()` | Run agents concurrently |
| Conditional | `run_conditional()` | Branch based on output |
| Loop | `run_loop()` | Repeat until condition met |
| Hierarchical | `run_hierarchical()` | Supervisor delegates to workers |
| Map-Reduce | `run_map_reduce()` | Process items, then aggregate |
| Pipeline | `run_pipeline()` | Chain with optional filters |
| Voting | `run_voting()` | Multiple agents vote |
| Debate | `run_debate()` | Agents argue, moderator concludes |
| Supervisor | `run_supervisor()` | Review and request revisions |
| Router | `run_router()` | Classify and route to specialist |
| Ensemble | `run_ensemble()` | Combine with weighting |
| State Machine | `run_state_machine()` | FSM with transitions |
| Event-Driven | `run_event_driven()` | Process via handlers |
| Critic | `run_critic()` | Creator-critic loop |
| Planner-Executor | `run_planner_executor()` | Plan then execute |
| Chain-of-Thought | `run_chain_of_thought()` | Explicit reasoning steps |

## Event Types

Subscribe to these events via `EventHookRegistry`:

| Event | When Triggered |
|-------|----------------|
| `AGENT_CREATED` | Agent instantiated |
| `AGENT_START` | Agent begins processing |
| `AGENT_END` | Agent completes |
| `AGENT_ERROR` | Agent encounters error |
| `TOOL_CALL_START` | Tool execution begins |
| `TOOL_CALL_END` | Tool execution completes |
| `TOOL_CALL_ERROR` | Tool execution fails |
| `ORCHESTRATION_START` | Workflow begins |
| `ORCHESTRATION_END` | Workflow completes |
| `ORCHESTRATION_STEP` | Individual step in workflow |
| `MESSAGE_SENT` | Message sent to model |
| `MESSAGE_RECEIVED` | Response received |

## Dependencies

### Required
- `ollama>=0.6.0` - Ollama Python SDK
- `httpx>=0.28.1` - HTTP client
- `pydantic>=2.11.9` - Data validation
- `PyYAML>=6.0.2` - YAML configuration
- `python-dotenv` - Environment variables

## Docker Setup

```bash
# Start Ollama container
docker-compose up -d

# Pull model
docker exec -it ollama ollama pull qwen3:14b

# Run agent
python agent.py
```

## Troubleshooting

### Common Issues

1. **Connection Error**: Ensure Ollama is running
   ```bash
   ollama serve
   ```

2. **Model Not Found**: Install the required model
   ```bash
   ollama pull qwen3:14b
   ```

3. **Web Search Auth Error**: Add API key to `.env`
   ```env
   OLLAMA_API_KEY=your_api_key
   ```

4. **Import Errors**: Install all dependencies
   ```bash
   pip install -r requirements.txt
   ```

## References

- [Ollama Official Repository](https://github.com/ollama/ollama)
- [Ollama Python SDK](https://github.com/ollama/ollama-python)
- [Available Models](https://ollama.com/library)