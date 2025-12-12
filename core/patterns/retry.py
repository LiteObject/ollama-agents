"""
Retry orchestration pattern handler.
"""

# pylint: disable=too-few-public-methods, arguments-differ, too-many-arguments, too-many-positional-arguments, protected-access, keyword-arg-before-vararg

from typing import List, Any, Optional, Callable
from core.patterns.base import PatternHandler


class RetryHandler(PatternHandler):
    """Handler for retry orchestration pattern."""

    def execute(
        self,
        primary_agent: str,
        prompt: str,
        validator_fn: Callable[[str], bool],
        fallback_agents: Optional[List[str]] = None,
        max_retries: int = 3,
        verbose: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[str, str]:
        """Run with retry and optional fallback agents.

        Attempts the primary agent, retries on validation failure,
        then falls back to alternative agents.

        Args:
            primary_agent: Primary agent to try first
            prompt: Prompt for the agents
            validator_fn: Function(output) -> True if valid
            fallback_agents: List of fallback agent names
            max_retries: Maximum retries per agent
            verbose: Whether to print progress
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Tuple of (successful_agent_name, output)
        """
        agents_to_try = [primary_agent] + (fallback_agents or [])

        for agent_name in agents_to_try:
            if verbose:
                print(f"\n[Retry] Trying agent: {agent_name}")

            for attempt in range(1, max_retries + 1):
                if verbose:
                    print(f"[Retry] Attempt {attempt}/{max_retries}")

                self.orchestrator.context.set("retry_attempt", attempt)
                interpolated_prompt = self.orchestrator.context.interpolate(prompt)
                output = self.orchestrator.run_agent(
                    agent_name, interpolated_prompt, verbose
                )

                if validator_fn(output):
                    if verbose:
                        print("[Retry] Validation passed!")
                    return agent_name, output

                if verbose:
                    print("[Retry] Validation failed, retrying...")

        # All agents failed
        raise RuntimeError(
            f"All agents failed validation after retries: {agents_to_try}"
        )
