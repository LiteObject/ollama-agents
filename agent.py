"""
Ollama Agent with Web Search Capabilities.

This module provides an interactive agent that can use Ollama language models
with web search and web fetch tools to answer questions and perform research.

The agent maintains conversation context, handles tool calls automatically,
and provides a user-friendly interface for multi-turn conversations.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, cast

try:
    from dotenv import load_dotenv  # type: ignore[import]
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore[assignment]

import ollama
from ollama import chat, ChatResponse, Message, ResponseError, RequestError

# Define available tools as a constant for easier configuration
AVAILABLE_TOOLS = [ollama.web_search, ollama.web_fetch]


def _load_env_file_fallback(env_path: Path) -> bool:
    """Load key=value pairs from a .env file when python-dotenv is unavailable."""

    if not env_path.exists():
        return False

    loaded_any = False
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value
            loaded_any = True

    return loaded_any


if load_dotenv is not None:
    load_dotenv(override=True)
else:
    ENV_FILE = Path(__file__).resolve().parent / ".env"
    if _load_env_file_fallback(ENV_FILE):
        print("Loaded environment variables from .env using fallback parser.")
    else:
        print("Warning: python-dotenv not installed and no .env file could be loaded.")
        print("Install with: pip install python-dotenv")


def message_to_dict(message) -> Dict[str, Any]:
    """Convert a Message object to a dictionary for the conversation history."""

    result = {
        "role": getattr(message, "role", "assistant"),
        "content": getattr(message, "content", "") or "",
    }

    if hasattr(message, "thinking") and message.thinking:
        result["thinking"] = message.thinking

    if hasattr(message, "tool_calls") and message.tool_calls:
        result["tool_calls"] = [
            {
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
            }
            for tc in message.tool_calls
        ]

    if hasattr(message, "tool_name") and message.tool_name:
        result["tool_name"] = message.tool_name

    return result


def get_chat_response(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    tools,
    think: bool = True,
) -> ChatResponse:
    """Call ollama.chat ensuring a concrete ChatResponse is returned.

    The Ollama chat API can return either a ChatResponse (non-streaming) or an
    iterator of ChatResponse chunks (streaming). Some static type checkers get
    confused and think it always returns a generator. This wrapper ensures we
    always get a single ChatResponse object.

    Args:
        model: Name of the Ollama model to use (e.g., "gpt-oss")
        messages: List of conversation messages in chat format
        tools: List of available tools (web_search, web_fetch, etc.)
        think: Whether to enable internal reasoning/thinking mode

    Returns:
        Single ChatResponse object with message and metadata

    Raises:
        RuntimeError: If chat() returns unexpected type or no message
    """
    # Call Ollama chat API with explicit non-streaming mode
    result = chat(
        model=model,
        messages=messages,
        tools=tools,
        think=think,
        stream=False,  # Force non-streaming to get single response
    )

    # Normal case: result is a ChatResponse with .message attribute
    if hasattr(result, "message"):
        return cast(ChatResponse, result)

    # Defensive fallback: if result is unexpectedly an iterator (streaming mode)
    # This shouldn't happen with stream=False, but handle it gracefully
    try:
        iterator = iter(result)  # type: ignore[assignment]
    except TypeError as exc:  # pragma: no cover
        raise RuntimeError("Unexpected chat() return type without 'message'") from exc

    # Consume iterator to find the last chunk containing a message
    last_with_message: ChatResponse | None = None
    for chunk in iterator:  # type: ignore[assignment]
        if hasattr(chunk, "message"):
            last_with_message = chunk  # type: ignore[assignment]

    # Ensure we found at least one chunk with a message
    if last_with_message is None:  # pragma: no cover
        raise RuntimeError("No chunk with 'message' produced by chat()")

    return cast(ChatResponse, last_with_message)


def main():
    """Main function to run the Ollama agent with web search capabilities.

    This function orchestrates the entire conversation flow:
    1. Sets up available tools (web search and web fetch)
    2. Gets initial user question
    3. Runs conversation loop with LLM and tool calls
    4. Handles errors gracefully with user-friendly messages

    Returns:
        int: Exit code (0 for success, 1 for errors)
    """
    # Dictionary mapping tool names to their callable functions
    # These tools allow the LLM to search and fetch web content
    available_tools = {tool.__name__: tool for tool in AVAILABLE_TOOLS}

    # Display welcome message and instructions
    print("Ollama Agent with Web Search")
    print("Type 'quit' or 'exit' to stop the conversation")

    # Display model configuration
    model_name = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
    print(f"Using model: {model_name}")

    # Check if API key is available for web search
    if not os.getenv("OLLAMA_API_KEY"):
        print("Note: Web search requires OLLAMA_API_KEY environment variable")
        print("   Sign up at https://ollama.com/ to get an API key for web search")
    else:
        print("Web search enabled with API key")

    print("-" * 50)

    # Get the initial question from user (or use default)
    initial_question = input(
        "Enter your question (or press Enter for default): "
    ).strip()
    if not initial_question:
        initial_question = "what are the latest developments in AI?"

    # Initialize conversation history with the user's first message
    # This list maintains the full conversation context for the LLM
    messages: List[Dict[str, Any]] = [{"role": "user", "content": initial_question}]

    try:
        # Main conversation loop control variables
        conversation_active = True
        max_iterations = 10  # Safety limit to prevent infinite tool calling loops
        iteration_count = 0

        # Continue conversation until user quits or max iterations reached
        while conversation_active and iteration_count < max_iterations:
            iteration_count += 1

            try:
                print(f"\n--- Iteration {iteration_count} ---")

                # Get model name from environment variable with fallback
                model_name = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")

                # Get LLM response using our normalized chat wrapper
                # This handles the model's response and any tool calls it wants to make
                response = get_chat_response(
                    model=model_name,  # Ollama model name from env or default
                    messages=messages,  # Full conversation history
                    tools=AVAILABLE_TOOLS,  # Available tools for LLM
                    think=True,  # Enable internal reasoning
                )

                # Extract the message from the response
                message: Message = response.message  # type: ignore[attr-defined]

                # Show the model's internal thinking process (if enabled)
                # This helps users understand how the model reasons through problems
                if hasattr(message, "thinking") and message.thinking:
                    print(f"Thinking: {message.thinking}")

                # Display the model's main response to the user
                if hasattr(message, "content") and message.content:
                    print(f"Response: {message.content}")

                # Add the assistant's message to conversation history
                # This maintains context for future turns in the conversation
                messages.append(message_to_dict(message))

                # Process any tool calls the model wants to make
                # Tools allow the model to search web, fetch content, etc.
                if hasattr(message, "tool_calls") and message.tool_calls:
                    print(f"Tool calls: {len(message.tool_calls)}")

                    # Execute each tool call requested by the model
                    for i, tool_call in enumerate(message.tool_calls, 1):
                        print(f"  Tool {i}: {tool_call.function.name}")

                        # Look up the actual function to call
                        function_to_call = available_tools.get(tool_call.function.name)

                        if function_to_call:
                            try:
                                # Extract arguments and call the tool function
                                args = tool_call.function.arguments
                                print(f"    Arguments: {args}")
                                result = function_to_call(**args)

                                # Show user a preview of the tool result
                                result_str = str(result)
                                print(
                                    f"    Result (first 200 chars): {result_str[:200]}..."
                                )

                                # Add tool result to conversation for model to use
                                # Limit length to prevent context window overflow
                                messages.append(
                                    {
                                        "role": "tool",  # Mark as tool response
                                        "content": result_str[
                                            :8000
                                        ],  # Truncate long results
                                        "tool_name": tool_call.function.name,  # Which tool
                                    }
                                )

                            except (
                                ConnectionError,  # Network issues
                                TimeoutError,  # Request timeouts
                                ValueError,  # Invalid arguments
                                RuntimeError,  # Other runtime issues
                            ) as tool_error:
                                # Handle tool execution errors gracefully
                                error_msg = str(tool_error)

                                # Provide specific help for web search authentication errors
                                if (
                                    "Authorization header with Bearer token is required"
                                    in error_msg
                                ):
                                    print(
                                        f"    Error executing tool "
                                        f"{tool_call.function.name}: {error_msg}"
                                    )
                                    print(
                                        "    Web search requires an Ollama API key. "
                                        "To enable web search:"
                                    )
                                    print("       1. Sign up at https://ollama.com/")
                                    print(
                                        "       2. Create an API key from your account"
                                    )
                                    print(
                                        "       3. Set environment variable: "
                                        'export OLLAMA_API_KEY="your_api_key"'
                                    )
                                    print("       4. Restart this application")
                                    error_msg = (
                                        "Web search unavailable - requires API key setup. "
                                        "See instructions above."
                                    )
                                else:
                                    print(
                                        f"    Error executing tool "
                                        f"{tool_call.function.name}: {error_msg}"
                                    )

                                # Still add error to conversation so model knows what happened
                                messages.append(
                                    {
                                        "role": "tool",
                                        "content": error_msg,
                                        "tool_name": tool_call.function.name,
                                    }
                                )
                        else:
                            # Handle case where model requests unknown tool
                            error_msg = f"Tool {tool_call.function.name} not found"
                            print(f"    {error_msg}")
                            messages.append(
                                {
                                    "role": "tool",
                                    "content": error_msg,
                                    "tool_name": tool_call.function.name,
                                }
                            )
                else:
                    # No tool calls means model gave a final answer
                    # Ask user if they want to continue the conversation
                    print(
                        "\nFinal response received. Would you like to ask another question?"
                    )

                    # Get user's next input or exit
                    user_input = input(
                        "\nEnter your next question (or 'quit'/'exit' to stop): "
                    ).strip()

                    # Check if user wants to quit
                    if user_input.lower() in ["quit", "exit", "q", ""]:
                        conversation_active = False
                        print("Conversation ended.")
                    else:
                        # Add new user message and reset iteration counter
                        # Reset counter allows new question to have full iterations
                        messages.append({"role": "user", "content": user_input})
                        iteration_count = 0  # Reset for new question

            except (ResponseError, RequestError) as api_error:
                # Handle Ollama API specific errors (model not found, etc.)
                print(f"Ollama API Error: {api_error}")
                print("Please check that Ollama is running and the model is available.")
                break  # Exit conversation loop but not the program

    except ConnectionError:
        # Handle case where Ollama service is not running
        print("Connection Error: Failed to connect to Ollama.")
        print("Please ensure Ollama is installed and running.")
        print("Visit https://ollama.com/download for installation instructions.")
        return 1  # Exit with error code

    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\nConversation interrupted by user.")
        return 0  # Normal exit

    except (RuntimeError, ValueError) as e:
        # Handle other runtime errors
        print(f"Runtime error: {e}")
        return 1  # Exit with error code

    return 0  # Normal exit - conversation completed successfully


if __name__ == "__main__":
    # Entry point: run main() and exit with its return code
    # This allows the script to be run directly or imported as a module
    sys.exit(main())
