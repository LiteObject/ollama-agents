"""
Agent class for AI-powered conversational agents.

This module provides a reusable Agent class that wraps Ollama models
with tool-calling capabilities and shared context support.
"""

# pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-positional-arguments, too-many-locals, too-many-branches, too-many-statements

import time
import uuid
from typing import Any, Callable, Dict, List, Mapping, Optional, cast

import ollama
from ollama import (
    chat,
    ChatResponse,
    Message as OllamaMessage,
    ResponseError,
    RequestError,
)

from core.config import AgentConfig
from core.context import SharedContext
from core.observability.hooks import (
    AgentEvent,
    EventHookRegistry,
    default_hook_registry,
)
from core.observability.logging import AgentLogger, get_logger
from core.tools import ToolRegistry, default_registry


class Agent:
    """A configurable AI agent powered by Ollama.

    This class provides a high-level interface for creating conversational
    agents with tool-calling capabilities. Agents can be configured via
    YAML files or programmatically.

    Usage:
        # From config
        config = AgentConfig.from_yaml("config/agents/researcher.yaml")
        agent = Agent(config)
        response = agent.run("What are the latest AI trends?")

        # Programmatic
        agent = Agent(
            config=AgentConfig(
                name="researcher",
                model="gpt-oss:20b",
                system_prompt="You are a research specialist.",
                tools=["web_search", "web_fetch"],
            )
        )
    """

    def __init__(
        self,
        config: AgentConfig,
        tool_registry: Optional[ToolRegistry] = None,
        context: Optional[SharedContext] = None,
        hook_registry: Optional[EventHookRegistry] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """Initialize the agent.

        Args:
            config: Agent configuration
            tool_registry: Tool registry to use (defaults to global registry)
            context: Shared context for inter-agent communication
            hook_registry: Event hook registry (defaults to global registry)
            session_id: Session identifier for logging/tracing
        """
        self.config = config
        self.name = config.name
        self.model = config.model
        self.system_prompt = config.system_prompt
        self.think = config.think
        self.max_iterations = config.max_iterations
        self.session_id = session_id or str(uuid.uuid4())

        self._registry = tool_registry or default_registry
        self._context = context or SharedContext()
        self._hooks = hook_registry or default_hook_registry
        self._conversation: List[Dict[str, Any]] = []

        # Set up logging
        self._logger: AgentLogger = get_logger(
            name=f"agent.{self.name}",
            session_id=self.session_id,
            agent_name=self.name,
        )

        # Load tools from registry
        self._tools: List[Callable[..., Any]] = []
        self._tool_map: Dict[str, Callable[..., Any]] = {}
        self._load_tools()

        # Trigger agent created event
        self._hooks.trigger(
            AgentEvent.AGENT_CREATED,
            agent_name=self.name,
            session_id=self.session_id,
            data={"model": self.model, "tools": self.config.tools},
        )
        self._logger.info(
            "Agent created",
            extra={"model": self.model, "tools": self.config.tools},
        )

    def _load_tools(self) -> None:
        """Load tools from the registry based on config."""
        # Add builtin Ollama tools
        builtin_tools = {
            "web_search": ollama.web_search,
            "web_fetch": ollama.web_fetch,
        }

        for tool_name in self.config.tools:
            # Check builtin first
            if tool_name in builtin_tools:
                tool_func = builtin_tools[tool_name]
                self._tools.append(tool_func)
                self._tool_map[tool_name] = tool_func
                self._logger.debug(f"Loaded builtin tool: {tool_name}")
            else:
                # Check registry
                tool_func = self._registry.get_function(tool_name)
                if tool_func:
                    self._tools.append(tool_func)
                    self._tool_map[tool_name] = tool_func
                    self._logger.debug(f"Loaded registry tool: {tool_name}")
                else:
                    self._logger.warning(f"Tool '{tool_name}' not found")

    def _message_to_dict(self, message: OllamaMessage) -> Dict[str, Any]:
        """Convert an Ollama Message to a dictionary."""
        result: Dict[str, Any] = {
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

    def _get_chat_response(self, messages: List[Dict[str, Any]]) -> ChatResponse:
        """Call Ollama chat API and return response.

        Args:
            messages: Conversation messages

        Returns:
            ChatResponse from Ollama
        """
        result = chat(
            model=self.model,
            messages=messages,
            tools=self._tools if self._tools else None,
            think=self.think,
            stream=False,
        )

        if hasattr(result, "message"):
            return cast(ChatResponse, result)

        # Handle iterator case (shouldn't happen with stream=False)
        try:
            iterator = iter(result)  # type: ignore
        except TypeError as exc:
            raise RuntimeError("Unexpected chat() return type") from exc

        last_with_message: ChatResponse | None = None
        for chunk in iterator:  # type: ignore
            if hasattr(chunk, "message"):
                last_with_message = chunk  # type: ignore

        if last_with_message is None:
            raise RuntimeError("No message produced by chat()")

        return cast(ChatResponse, last_with_message)

    def _execute_tool(self, tool_name: str, arguments: Mapping[str, Any]) -> str:
        """Execute a tool and return the result.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            Tool result as string
        """
        tool_func = self._tool_map.get(tool_name)

        if not tool_func:
            self._logger.error(f"Tool '{tool_name}' not found")
            return f"Error: Tool '{tool_name}' not found"

        # Trigger tool start event
        self._hooks.trigger(
            AgentEvent.TOOL_CALL_START,
            agent_name=self.name,
            session_id=self.session_id,
            data={"tool": tool_name, "arguments": dict(arguments)},
        )
        self._logger.debug(
            f"Executing tool: {tool_name}",
            extra={"arguments": dict(arguments)},
        )

        start_time = time.perf_counter()

        try:
            result = tool_func(**arguments)
            result_str = str(result)[:8000]  # Truncate to prevent context overflow

            duration_ms = (time.perf_counter() - start_time) * 1000

            # Trigger tool end event
            self._hooks.trigger(
                AgentEvent.TOOL_CALL_END,
                agent_name=self.name,
                session_id=self.session_id,
                duration_ms=duration_ms,
                data={"tool": tool_name, "result_length": len(result_str)},
            )
            self._logger.debug(
                f"Tool completed: {tool_name}",
                duration_ms=duration_ms,
                extra={"result_length": len(result_str)},
            )

            return result_str

        except (TypeError, ValueError, RuntimeError) as e:
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Trigger tool error event
            self._hooks.trigger(
                AgentEvent.TOOL_CALL_ERROR,
                agent_name=self.name,
                session_id=self.session_id,
                duration_ms=duration_ms,
                error=e,
                data={"tool": tool_name},
            )
            self._logger.error(
                f"Tool error: {tool_name} - {e}",
                duration_ms=duration_ms,
            )

            return f"Error executing {tool_name}: {e}"

    def run(
        self,
        prompt: str,
        context: Optional[SharedContext] = None,
        verbose: bool = False,
    ) -> str:
        """Run the agent with a prompt and return the final response.

        This method handles the full conversation loop including tool calls.

        Args:
            prompt: The user's prompt/question
            context: Optional shared context (uses instance context if not provided)
            verbose: Whether to print progress information

        Returns:
            The agent's final response
        """
        ctx = context or self._context
        start_time = time.perf_counter()

        # Trigger agent start event
        self._hooks.trigger(
            AgentEvent.AGENT_START,
            agent_name=self.name,
            session_id=self.session_id,
            data={"prompt_length": len(prompt)},
        )
        self._logger.info(
            "Agent starting",
            extra={"prompt_length": len(prompt)},
        )

        # Initialize conversation with system prompt if set
        messages: List[Dict[str, Any]] = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # Interpolate variables in prompt
        interpolated_prompt = ctx.interpolate(prompt)
        messages.append({"role": "user", "content": interpolated_prompt})

        # Add to shared context history
        ctx.add_message("user", interpolated_prompt, agent_name=self.name)

        # Trigger message sent event
        self._hooks.trigger(
            AgentEvent.MESSAGE_SENT,
            agent_name=self.name,
            session_id=self.session_id,
            data={"role": "user", "content_length": len(interpolated_prompt)},
        )

        iteration = 0
        final_response = ""

        while iteration < self.max_iterations:
            iteration += 1

            if verbose:
                print(f"\n[{self.name}] Iteration {iteration}")

            self._logger.debug(f"Iteration {iteration}", step=iteration)

            try:
                response = self._get_chat_response(messages)
                message = response.message  # type: ignore

                # Show thinking if verbose
                if verbose and hasattr(message, "thinking") and message.thinking:
                    print(f"[{self.name}] Thinking: {message.thinking[:200]}...")

                # Get content
                content = getattr(message, "content", "") or ""

                if verbose and content:
                    print(f"[{self.name}] Response: {content[:200]}...")

                # Add to conversation
                messages.append(self._message_to_dict(message))

                # Trigger message received event
                self._hooks.trigger(
                    AgentEvent.MESSAGE_RECEIVED,
                    agent_name=self.name,
                    session_id=self.session_id,
                    data={
                        "role": "assistant",
                        "content_length": len(content),
                        "has_tool_calls": bool(
                            hasattr(message, "tool_calls") and message.tool_calls
                        ),
                    },
                )

                # Handle tool calls
                if hasattr(message, "tool_calls") and message.tool_calls:
                    if verbose:
                        print(f"[{self.name}] Tool calls: {len(message.tool_calls)}")

                    self._logger.debug(
                        f"Processing {len(message.tool_calls)} tool calls",
                        step=iteration,
                    )

                    for tool_call in message.tool_calls:
                        tool_name = tool_call.function.name
                        arguments = tool_call.function.arguments

                        if verbose:
                            print(f"[{self.name}]   Executing: {tool_name}")

                        result = self._execute_tool(tool_name, arguments)

                        messages.append(
                            {
                                "role": "tool",
                                "content": result,
                                "tool_name": tool_name,
                            }
                        )

                        ctx.add_message(
                            "tool",
                            result,
                            agent_name=self.name,
                            tool_name=tool_name,
                        )
                else:
                    # No tool calls - we have a final response
                    final_response = content
                    break

            except (ResponseError, RequestError) as e:
                duration_ms = (time.perf_counter() - start_time) * 1000

                # Trigger agent error event
                self._hooks.trigger(
                    AgentEvent.AGENT_ERROR,
                    agent_name=self.name,
                    session_id=self.session_id,
                    duration_ms=duration_ms,
                    error=e,
                )
                self._logger.error(
                    f"API Error: {e}",
                    duration_ms=duration_ms,
                    exc_info=True,
                )

                if verbose:
                    print(f"[{self.name}] API Error: {e}")
                return f"API Error: {e}"

            except (TypeError, ValueError, RuntimeError) as e:
                duration_ms = (time.perf_counter() - start_time) * 1000

                # Trigger agent error event
                self._hooks.trigger(
                    AgentEvent.AGENT_ERROR,
                    agent_name=self.name,
                    session_id=self.session_id,
                    duration_ms=duration_ms,
                    error=e,
                )
                self._logger.error(
                    f"Error: {e}",
                    duration_ms=duration_ms,
                    exc_info=True,
                )

                if verbose:
                    print(f"[{self.name}] Error: {e}")
                return f"Error: {e}"

        # Store output in context
        ctx.set_agent_output(self.name, final_response)
        ctx.add_message("assistant", final_response, agent_name=self.name)

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Trigger agent end event
        self._hooks.trigger(
            AgentEvent.AGENT_END,
            agent_name=self.name,
            session_id=self.session_id,
            duration_ms=duration_ms,
            data={
                "iterations": iteration,
                "response_length": len(final_response),
            },
        )
        self._logger.info(
            "Agent completed",
            duration_ms=duration_ms,
            extra={"iterations": iteration, "response_length": len(final_response)},
        )

        return final_response

    def chat(self, message: str) -> str:
        """Single-turn chat without resetting conversation.

        Useful for multi-turn conversations where you want to maintain
        the full conversation history.

        Args:
            message: User message

        Returns:
            Agent response
        """
        self._conversation.append({"role": "user", "content": message})

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(self._conversation)

        try:
            response = self._get_chat_response(messages)
            content = getattr(response.message, "content", "") or ""  # type: ignore
            self._conversation.append({"role": "assistant", "content": content})
            return content
        except (ResponseError, RequestError, TypeError, ValueError, RuntimeError) as e:
            self._logger.error(f"Chat error: {e}")
            return f"Error: {e}"

    def reset(self) -> None:
        """Reset the conversation history."""
        self._conversation.clear()

    def __repr__(self) -> str:
        """String representation of the agent."""
        return (
            f"Agent(name='{self.name}', model='{self.model}', "
            f"tools={len(self._tools)})"
        )
