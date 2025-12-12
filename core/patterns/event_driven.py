"""
Event-driven orchestration pattern handler.
"""

# pylint: disable=too-few-public-methods, arguments-differ, too-many-arguments, too-many-positional-arguments, protected-access, keyword-arg-before-vararg, too-many-locals

from typing import Any, Dict, List
from core.patterns.base import PatternHandler


class EventDrivenHandler(PatternHandler):
    """Handler for event-driven orchestration pattern."""

    def execute(
        self,
        event_handlers: Dict[str, tuple[str, str]],
        events: List[Dict[str, Any]],
        verbose: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run event-driven pattern processing a stream of events.

        Args:
            event_handlers: Dict of event_type -> (agent_name, prompt_template)
            events: List of event dictionaries with 'type' and 'data' keys
            verbose: Whether to print progress
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary with 'processed_events', 'unhandled_events', 'outputs'
        """
        if verbose:
            print(f"\n[EventDriven] Processing {len(events)} events...")
            print(f"[EventDriven] Registered handlers: {list(event_handlers.keys())}")

        processed_events: List[Dict[str, Any]] = []
        unhandled_events: List[Dict[str, Any]] = []
        outputs: List[str] = []

        for event in events:
            event_type = event.get("type", "unknown")
            event_data = event.get("data", {})

            if verbose:
                print(f"\n[EventDriven] Event: {event_type}")

            if event_type in event_handlers:
                agent_name, prompt_template = event_handlers[event_type]

                # Set event data in context
                self.orchestrator.context.set("event_type", event_type)
                self.orchestrator.context.set("event_data", str(event_data))
                for key, value in event_data.items():
                    self.orchestrator.context.set(f"event_{key}", str(value))

                prompt = self.orchestrator.context.interpolate(prompt_template)
                output = self.orchestrator.run_agent(agent_name, prompt, verbose)

                processed_events.append(
                    {
                        "event": event,
                        "handler": agent_name,
                        "output": output,
                    }
                )
                outputs.append(output)
            else:
                unhandled_events.append(event)
                if verbose:
                    print(f"[EventDriven] No handler for event type: {event_type}")

        return {
            "processed_events": processed_events,
            "unhandled_events": unhandled_events,
            "outputs": outputs,
            "total_processed": len(processed_events),
            "total_unhandled": len(unhandled_events),
        }
