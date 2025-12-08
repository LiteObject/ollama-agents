# Ollama Demo - Interactive Agent with Web Search

This repository demonstrates how to build an intelligent agent using Ollama that can interact with users and perform web searches to answer questions.

## What is Ollama?

Ollama is a tool that allows you to run open-source large language models (LLMs) locally on your machine. It supports a variety of models, including Llama 2, Code Llama, Qwen, and others, bundling model weights, configuration, and data into a single package defined by a Modelfile.

## Project Features

### Interactive AI Agent (agent.py)
- **Multi-turn conversations**: Engage in ongoing dialogues with the AI
- **Web search integration**: Agent can search the web and fetch content to answer questions
- **Robust error handling**: Graceful handling of connection issues and API errors
- **Type-safe implementation**: Full type hints and proper error handling
- **User-friendly interface**: Clear progress indicators and formatted output
- **Context management**: Smart truncation of results to prevent token overflow
- **Environment variable support**: Built-in `.env` file parser (works even without python-dotenv)

### Available Tools
- **Web Search**: Search the internet for current information
- **Web Fetch**: Retrieve and analyze content from specific URLs
- **Thinking Mode**: See the AI's reasoning process

## Quick Start

### Option 1: Using Docker (Recommended)

1. **Start Ollama container**:
   ```bash
   docker-compose up -d
   ```

2. **Access the container**:
   ```bash
   docker exec -it ollama bash
   ```

3. **Install required model**:
   ```bash
   ollama pull gpt-oss:20b
   ```

4. **Set up web search (optional)**:
   Follow the web search setup instructions in Option 2, step 4.

5. **Run the agent**:
   ```bash
   python agent.py
   ```

### Option 2: Local Installation

1. **Install Ollama**: Visit [ollama.com/download](https://ollama.com/download)

2. **Start Ollama service**:
   ```bash
   ollama serve
   ```

3. **Install required model**:
   ```bash
   ollama pull gpt-oss:20b
   ```

4. **Set up configuration**:
   Create a `.env` file in the project root:
   ```bash
   cp .env.example .env
   ```
   Edit the `.env` file to configure your settings:
   ```env
   # Required for web search functionality
   OLLAMA_API_KEY=your_actual_api_key_here
   
   # Optional: Specify which model to use (defaults to gpt-oss:20b)
   OLLAMA_MODEL=gpt-oss:20b
   ```
   Sign up at [ollama.com](https://ollama.com/) to get an API key for web search functionality.

5. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Note: `python-dotenv` is optional. The agent includes a fallback `.env` parser that works without it.

6. **Run the agent**:
   ```bash
   python agent.py
   ```

## Usage Examples

### Basic Interaction
```
Enter your question (or press Enter for default): What is the latest news about AI?

--- Iteration 1 ---
Thinking: I need to search for recent AI news to provide current information...
Tool calls: 1
  Tool 1: web_search
    Arguments: {'query': 'latest AI news 2024'}
    Result (first 200 chars): Recent developments in artificial intelligence include...
Response: Based on my search, here are the latest developments in AI...
```

### Multi-turn Conversation
```
Final response received. Would you like to ask another question?

Enter your next question (or 'quit'/'exit' to stop): Tell me more about that company

--- Iteration 2 ---
Thinking: The user wants more details about the company mentioned...
```

### Web Search Not Configured
```
Note: Web search requires OLLAMA_API_KEY environment variable
   Sign up at https://ollama.com/ to get an API key for web search

--- Iteration 1 ---
Tool calls: 1
  Tool 1: web_search
    Error executing tool web_search: Authorization header with Bearer token is required
    Web search requires an Ollama API key. To enable web search:
       1. Sign up at https://ollama.com/
       2. Create an API key from your account
       3. Set environment variable: export OLLAMA_API_KEY="your_api_key"
       4. Restart this application
```

## Dependencies

See [requirements.txt](requirements.txt) for the complete list of Python dependencies:

### Required
- `ollama>=0.6.0` - Official Ollama Python SDK (includes web search support)
- `httpx>=0.28.1` - HTTP client for web requests
- `pydantic>=2.11.9` - Data validation and type safety

### Optional
- `python-dotenv` - Automatic `.env` file loading (agent includes fallback parser)

## Project Structure

```
├── agent.py              # Main interactive agent with web search
├── docker-compose.yml    # Docker setup for Ollama service
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (API keys) - not tracked in git
├── .env.example          # Example environment file
├── README.md             # This file
└── .gitignore            # Git ignore rules
```

## Key Features of the Agent

- **Robust Error Handling**: Handles connection issues, API errors, and tool failures
- **Interactive Loop**: Supports multi-turn conversations with context retention (max 10 iterations per question)
- **Web Integration**: Can search and fetch web content in real-time
- **Thinking Mode**: Shows the AI's reasoning process
- **Performance Optimized**: Smart context management (8000 char limit for tool results)
- **User-Friendly**: Clear feedback with emojis, progress indicators, and easy exit options
- **Fallback Support**: Works even without python-dotenv using built-in `.env` parser

## Troubleshooting

### Common Issues

1. **Connection Error**: Ensure Ollama is running
   ```bash
   ollama serve
   ```

2. **Model Not Found**: Install the required model
   ```bash
   ollama pull gpt-oss:20b
   ```

3. **Web Search Authentication Error**: Web search requires an API key
   - Sign up at [ollama.com](https://ollama.com/)
   - Create an API key from your account dashboard
   - Add it to your `.env` file: `OLLAMA_API_KEY=your_api_key`
   - Or set environment variable: `export OLLAMA_API_KEY="your_api_key"`
   - The agent will work without web search if no API key is provided

4. **Different Model**: To use a different model
   - Set `OLLAMA_MODEL=your_model_name` in your `.env` file
   - Or set environment variable: `export OLLAMA_MODEL="your_model_name"`
   - Make sure the model is available: `ollama pull your_model_name`

5. **Import Warning**: If you see "python-dotenv not installed" warning
   - Install it with: `pip install python-dotenv`
   - Or ignore it - the agent includes a fallback `.env` parser

### Getting Help

- Verify Ollama installation at [ollama.com](https://ollama.com)
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Ollama service status: `ollama list` to see available models
- For Docker users: `docker-compose logs` to check container status

## How It Works

The agent follows this conversation flow:

1. **User Input**: Accept question from user
2. **LLM Processing**: Send to GPT-OSS model with thinking enabled
3. **Tool Detection**: Model decides if web search/fetch is needed
4. **Tool Execution**: Run requested tools (with authentication if configured)
5. **Result Integration**: Add tool results to conversation context
6. **Response Generation**: Model generates final answer using all information
7. **Iteration Check**: Continue if more tools needed (max 10 iterations)
8. **User Continuation**: Ask if user wants to continue conversation

## References

- [Ollama Official Repository](https://github.com/ollama/ollama)
- [Ollama Python SDK Documentation](https://github.com/ollama/ollama-python)
- [Available Models](https://ollama.com/library)