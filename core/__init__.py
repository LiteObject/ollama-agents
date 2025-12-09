"""
Core module for the multi-agent framework.

This module provides the foundational components for building and orchestrating
AI agents with configurable tools and shared context.
"""

from core.agent import Agent
from core.config import AgentConfig
from core.context import SharedContext
from core.observability import (
    AgentEvent,
    AgentLogger,
    EventHookRegistry,
    configure_logging,
    default_hook_registry,
    get_logger,
)
from core.orchestrator import Orchestrator
from core.tools import ToolRegistry, tool

__all__ = [
    # Core components
    "AgentConfig",
    "Agent",
    "SharedContext",
    "tool",
    "ToolRegistry",
    "Orchestrator",
    # Observability
    "AgentEvent",
    "AgentLogger",
    "EventHookRegistry",
    "configure_logging",
    "default_hook_registry",
    "get_logger",
]
