"""
Agent configuration management with YAML support.

This module provides dataclasses for agent configuration and utilities
for loading configurations from YAML files.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class AgentConfig:
    """Configuration for an AI agent.

    Attributes:
        name: Unique identifier for the agent
        model: Ollama model name to use (e.g., "gpt-oss:20b", "llama3.2:latest")
        system_prompt: Instructions that define the agent's behavior and role
        tools: List of tool names the agent can use
        think: Whether to enable reasoning/thinking mode
        max_iterations: Maximum tool-calling iterations per run
        temperature: Model temperature for response randomness (0.0-1.0)
        description: Human-readable description of the agent's purpose
    """

    name: str
    model: str = "gpt-oss:20b"
    system_prompt: str = ""
    tools: List[str] = field(default_factory=list)
    think: bool = True
    max_iterations: int = 10
    temperature: float = 0.7
    description: str = ""

    @classmethod
    def from_yaml(cls, path: Path | str) -> "AgentConfig":
        """Load agent configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file

        Returns:
            AgentConfig instance with values from the file

        Raises:
            FileNotFoundError: If the YAML file doesn't exist
            yaml.YAMLError: If the YAML is malformed
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Agent config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = yaml.safe_load(f)

        return cls(**data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentConfig":
        """Create agent configuration from a dictionary.

        Args:
            data: Dictionary with configuration values

        Returns:
            AgentConfig instance
        """
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary.

        Returns:
            Dictionary representation of the configuration
        """
        return {
            "name": self.name,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "tools": self.tools,
            "think": self.think,
            "max_iterations": self.max_iterations,
            "temperature": self.temperature,
            "description": self.description,
        }

    def to_yaml(self, path: Path | str) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Path where the YAML file will be saved
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator.

    Attributes:
        agents: List of agent configurations
        default_model: Default model for agents without explicit model
        shared_context_enabled: Whether agents share context
    """

    agents: List[AgentConfig] = field(default_factory=list)
    default_model: str = "gpt-oss:20b"
    shared_context_enabled: bool = True

    @classmethod
    def from_yaml(cls, path: Path | str) -> "OrchestratorConfig":
        """Load orchestrator configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file

        Returns:
            OrchestratorConfig instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Orchestrator config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = yaml.safe_load(f)

        agents = [AgentConfig.from_dict(a) for a in data.get("agents", [])]

        return cls(
            agents=agents,
            default_model=data.get("default_model", "gpt-oss:20b"),
            shared_context_enabled=data.get("shared_context_enabled", True),
        )
