"""
Shared context for inter-agent communication.

This module provides a SharedContext class that allows agents to share
state, variables, and artifacts during orchestration.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class Message:
    """A message in the conversation history.

    Attributes:
        role: The role of the message sender (user, assistant, tool, system)
        content: The message content
        agent_name: Name of the agent that sent the message (if applicable)
        timestamp: When the message was created
        metadata: Additional metadata about the message
    """

    role: str
    content: str
    agent_name: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format for Ollama API."""
        result: Dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }

        if self.metadata:
            result.update(self.metadata)

        return result


class SharedContext:
    """Shared context for inter-agent communication.

    This class provides a central store for:
    - Variables that can be read/written by any agent
    - Conversation history shared across agents
    - Artifacts (files, data) produced by agents

    Usage:
        context = SharedContext()
        context.set("research_data", some_data)
        data = context.get("research_data")
    """

    def __init__(self) -> None:
        """Initialize an empty shared context."""
        self._variables: Dict[str, Any] = {}
        self._history: List[Message] = []
        self._artifacts: Dict[str, Any] = {}
        self._agent_outputs: Dict[str, str] = {}

    # --- Variable Management ---

    def set(self, key: str, value: Any) -> None:
        """Set a variable in the shared context.

        Args:
            key: Variable name
            value: Variable value
        """
        self._variables[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a variable from the shared context.

        Args:
            key: Variable name
            default: Default value if key not found

        Returns:
            The variable value or default
        """
        return self._variables.get(key, default)

    def has(self, key: str) -> bool:
        """Check if a variable exists in the context.

        Args:
            key: Variable name

        Returns:
            True if the variable exists
        """
        return key in self._variables

    def delete(self, key: str) -> None:
        """Delete a variable from the context.

        Args:
            key: Variable name
        """
        self._variables.pop(key, None)

    def get_all_variables(self) -> Dict[str, Any]:
        """Get all variables in the context.

        Returns:
            Dictionary of all variables
        """
        return dict(self._variables)

    # --- Conversation History ---

    def add_message(
        self,
        role: str,
        content: str,
        agent_name: Optional[str] = None,
        **metadata: Any,
    ) -> None:
        """Add a message to the conversation history.

        Args:
            role: Message role (user, assistant, tool, system)
            content: Message content
            agent_name: Name of the agent that sent the message
            **metadata: Additional metadata
        """
        message = Message(
            role=role,
            content=content,
            agent_name=agent_name,
            metadata=metadata,
        )
        self._history.append(message)

    def get_history(self, agent_name: Optional[str] = None) -> List[Message]:
        """Get conversation history.

        Args:
            agent_name: Filter by agent name (None for all)

        Returns:
            List of messages
        """
        if agent_name is None:
            return list(self._history)

        return [m for m in self._history if m.agent_name == agent_name]

    def get_history_as_dicts(self) -> List[Dict[str, Any]]:
        """Get conversation history as list of dictionaries.

        Returns:
            List of message dictionaries for Ollama API
        """
        return [m.to_dict() for m in self._history]

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self._history.clear()

    # --- Artifacts ---

    def store_artifact(
        self, name: str, data: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store an artifact (file, data, etc.) in the context.

        Args:
            name: Artifact name
            data: Artifact data
            metadata: Optional metadata about the artifact
        """
        self._artifacts[name] = {
            "data": data,
            "metadata": metadata or {},
            "timestamp": datetime.now(),
        }

    def get_artifact(self, name: str) -> Optional[Any]:
        """Get an artifact by name.

        Args:
            name: Artifact name

        Returns:
            Artifact data or None if not found
        """
        artifact = self._artifacts.get(name)
        return artifact["data"] if artifact else None

    def list_artifacts(self) -> List[str]:
        """Get list of artifact names.

        Returns:
            List of artifact names
        """
        return list(self._artifacts.keys())

    # --- Agent Outputs ---

    def set_agent_output(self, agent_name: str, output: str) -> None:
        """Store the output of an agent.

        Args:
            agent_name: Name of the agent
            output: The agent's output
        """
        self._agent_outputs[agent_name] = output

    def get_agent_output(self, agent_name: str) -> Optional[str]:
        """Get the output of a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            The agent's output or None
        """
        return self._agent_outputs.get(agent_name)

    def get_last_agent_output(self) -> Optional[str]:
        """Get the most recent agent output.

        Returns:
            The last agent's output or None
        """
        if not self._agent_outputs:
            return None

        # Return the last added output
        return list(self._agent_outputs.values())[-1]

    # --- Utility Methods ---

    def interpolate(self, template: str) -> str:
        """Interpolate variables into a template string.

        Replaces {variable_name} with variable values from context.

        Args:
            template: Template string with {variable} placeholders

        Returns:
            Interpolated string
        """
        result = template

        # Replace variables
        for key, value in self._variables.items():
            result = result.replace(f"{{{key}}}", str(value))

        # Replace agent outputs
        for agent_name, output in self._agent_outputs.items():
            result = result.replace(f"{{{agent_name}}}", str(output))

        return result

    def clear(self) -> None:
        """Clear all context data."""
        self._variables.clear()
        self._history.clear()
        self._artifacts.clear()
        self._agent_outputs.clear()

    def __repr__(self) -> str:
        """String representation of the context."""
        return (
            f"SharedContext("
            f"variables={len(self._variables)}, "
            f"history={len(self._history)}, "
            f"artifacts={len(self._artifacts)}, "
            f"agent_outputs={len(self._agent_outputs)})"
        )
