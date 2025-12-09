"""
Observability module for multi-agent framework.

This module provides structured logging and event hooks for monitoring
and debugging agent orchestration.
"""

from .hooks import (
    AgentEvent,
    EventHookRegistry,
    default_hook_registry,
)
from .logging import AgentLogger, configure_logging, get_logger

__all__ = [
    # Logging
    "AgentLogger",
    "configure_logging",
    "get_logger",
    # Hooks
    "AgentEvent",
    "EventHookRegistry",
    "default_hook_registry",
]
