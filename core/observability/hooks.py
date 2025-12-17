"""
Event hooks for multi-agent framework.

This module provides an event hook system that allows external code to
subscribe to agent lifecycle events for monitoring, logging, and custom
integrations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class AgentEvent(Enum):
    """Event types that can be hooked into.

    Events are triggered at key points in agent and orchestration lifecycle.
    """

    # Agent lifecycle events
    AGENT_CREATED = "agent_created"
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    AGENT_ERROR = "agent_error"

    # Message events
    MESSAGE_SENT = "message_sent"
    MESSAGE_RECEIVED = "message_received"

    # Tool events
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"
    TOOL_CALL_ERROR = "tool_call_error"

    # Orchestration events
    ORCHESTRATION_START = "orchestration_start"
    ORCHESTRATION_END = "orchestration_end"
    ORCHESTRATION_STEP = "orchestration_step"
    ORCHESTRATION_ERROR = "orchestration_error"

    # Context events
    CONTEXT_UPDATED = "context_updated"

    # Custom events
    CUSTOM = "custom"


@dataclass
class EventData:
    """Data payload for an event.

    Attributes:
        event: The event type
        timestamp: When the event occurred
        agent_name: Name of the agent (if applicable)
        session_id: Session identifier
        data: Additional event-specific data
        error: Error information (if applicable)
        duration_ms: Duration of the operation (if applicable)
    """

    event: AgentEvent
    timestamp: datetime = field(default_factory=datetime.now)
    agent_name: Optional[str] = None
    session_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Exception] = None
    duration_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert event data to dictionary.

        Returns:
            Dictionary representation of the event
        """
        result: Dict[str, Any] = {
            "event": self.event.value,
            "timestamp": self.timestamp.isoformat(),
        }

        if self.agent_name:
            result["agent_name"] = self.agent_name
        if self.session_id:
            result["session_id"] = self.session_id
        if self.data:
            result["data"] = self.data
        if self.error:
            result["error"] = str(self.error)
        if self.duration_ms is not None:
            result["duration_ms"] = self.duration_ms

        return result


# Type alias for hook callbacks
HookCallback = Callable[[EventData], None]


class EventHookRegistry:
    """Registry for event hooks.

    Allows subscribing to events and triggering callbacks when events occur.

    Usage:
        registry = EventHookRegistry()

        # Subscribe to events
        registry.on(AgentEvent.AGENT_START, my_callback)
        registry.on_all(my_universal_callback)

        # Trigger events
        registry.trigger(AgentEvent.AGENT_START, EventData(...))

        # Unsubscribe
        registry.off(AgentEvent.AGENT_START, my_callback)
    """

    def __init__(self) -> None:
        """Initialize an empty hook registry."""
        self._hooks: Dict[AgentEvent, List[HookCallback]] = {}
        self._global_hooks: List[HookCallback] = []
        self._enabled: bool = True

    def on(
        self,
        event: AgentEvent,
        callback: HookCallback,
    ) -> None:
        """Subscribe to a specific event.

        Args:
            event: Event type to subscribe to
            callback: Function to call when event occurs
        """
        if event not in self._hooks:
            self._hooks[event] = []
        if callback not in self._hooks[event]:
            self._hooks[event].append(callback)

    def on_all(self, callback: HookCallback) -> None:
        """Subscribe to all events.

        Args:
            callback: Function to call for any event
        """
        if callback not in self._global_hooks:
            self._global_hooks.append(callback)

    def off(
        self,
        event: AgentEvent,
        callback: HookCallback,
    ) -> None:
        """Unsubscribe from a specific event.

        Args:
            event: Event type to unsubscribe from
            callback: Callback to remove
        """
        if event in self._hooks and callback in self._hooks[event]:
            self._hooks[event].remove(callback)

    def off_all(self, callback: HookCallback) -> None:
        """Unsubscribe from all events.

        Args:
            callback: Callback to remove from global hooks
        """
        if callback in self._global_hooks:
            self._global_hooks.remove(callback)

    def clear(self, event: Optional[AgentEvent] = None) -> None:
        """Clear hooks for an event or all events.

        Args:
            event: Specific event to clear, or None for all
        """
        if event:
            self._hooks[event] = []
        else:
            self._hooks.clear()
            self._global_hooks.clear()

    def trigger(
        self,
        event: AgentEvent,
        data: Optional[EventData] = None,
        **kwargs: Any,
    ) -> None:
        """Trigger an event and call all registered callbacks.

        Args:
            event: Event type to trigger
            data: Event data (created from kwargs if not provided)
            **kwargs: Arguments to create EventData if data not provided.
                      Known fields (agent_name, session_id, error, duration_ms)
                      are passed directly; all others go into the 'data' dict.
        """
        if not self._enabled:
            return

        # Create event data if not provided
        if data is None:
            # Separate known EventData fields from extra data
            known_fields = {"agent_name", "session_id", "error", "duration_ms"}
            event_kwargs: Dict[str, Any] = {"event": event}
            extra_data: Dict[str, Any] = {}

            for key, value in kwargs.items():
                if key in known_fields:
                    event_kwargs[key] = value
                else:
                    extra_data[key] = value

            # Add extra data to the 'data' dict field
            if extra_data:
                event_kwargs["data"] = extra_data

            data = EventData(**event_kwargs)

        # Call event-specific hooks
        for callback in self._hooks.get(event, []):
            try:
                callback(data)
            except (TypeError, ValueError, RuntimeError, AttributeError):
                # Don't let hook errors break the main flow
                pass

        # Call global hooks
        for callback in self._global_hooks:
            try:
                callback(data)
            except (TypeError, ValueError, RuntimeError, AttributeError):
                pass

    def enable(self) -> None:
        """Enable event triggering."""
        self._enabled = True

    def disable(self) -> None:
        """Disable event triggering (hooks won't be called)."""
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        """Check if event triggering is enabled."""
        return self._enabled

    def list_hooks(self, event: Optional[AgentEvent] = None) -> Dict[str, int]:
        """List registered hooks.

        Args:
            event: Specific event to list, or None for all

        Returns:
            Dictionary of event -> hook count
        """
        if event:
            return {event.value: len(self._hooks.get(event, []))}

        result = {e.value: len(hooks) for e, hooks in self._hooks.items()}
        result["_global"] = len(self._global_hooks)
        return result


# Default global hook registry
default_hook_registry = EventHookRegistry()


# Convenience functions for the default registry
def on(event: AgentEvent, callback: HookCallback) -> None:
    """Subscribe to an event on the default registry."""
    default_hook_registry.on(event, callback)


def on_all(callback: HookCallback) -> None:
    """Subscribe to all events on the default registry."""
    default_hook_registry.on_all(callback)


def off(event: AgentEvent, callback: HookCallback) -> None:
    """Unsubscribe from an event on the default registry."""
    default_hook_registry.off(event, callback)


def trigger(
    event: AgentEvent,
    data: Optional[EventData] = None,
    **kwargs: Any,
) -> None:
    """Trigger an event on the default registry."""
    default_hook_registry.trigger(event, data, **kwargs)
