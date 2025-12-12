"""
Parallel orchestration pattern handler.
"""

# pylint: disable=too-few-public-methods, arguments-differ, too-many-arguments, too-many-positional-arguments, protected-access, keyword-arg-before-vararg, duplicate-code

import concurrent.futures
import time
from typing import Dict

from core.observability.hooks import AgentEvent
from core.patterns.base import PatternHandler


class ParallelHandler(PatternHandler):
    """Handler for parallel execution pattern."""

    def execute(
        self,
        agent_prompts: Dict[str, str],
        verbose: bool = False,
        max_workers: int = 4,
    ) -> Dict[str, str]:
        """Run multiple agents in parallel.

        Args:
            agent_prompts: Dictionary of agent_name -> prompt
            verbose: Whether to print progress
            max_workers: Maximum parallel workers

        Returns:
            Dictionary of agent_name -> response
        """
        start_time = time.perf_counter()
        pattern = "parallel"

        # Trigger orchestration start event
        self.orchestrator._hooks.trigger(
            AgentEvent.ORCHESTRATION_START,
            session_id=self.orchestrator.session_id,
            data={"pattern": pattern, "agents": list(agent_prompts.keys())},
        )
        self.orchestrator._logger.info(
            f"Starting {pattern} orchestration",
            pattern=pattern,
            extra={"agents": list(agent_prompts.keys())},
        )

        results: Dict[str, str] = {}

        def run_single(name: str, prompt: str) -> tuple[str, str]:
            agent = self.orchestrator.get_agent(name)
            if not agent:
                return name, f"Error: Agent '{name}' not found"
            interpolated = self.orchestrator._context.interpolate(prompt)
            return name, agent.run(
                interpolated, context=self.orchestrator._context, verbose=verbose
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_single, name, prompt): name
                for name, prompt in agent_prompts.items()
            }

            for future in concurrent.futures.as_completed(futures):
                name, result = future.result()
                results[name] = result

                if verbose:
                    print(f"\n[{name}] Completed: {result[:200]}...")

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Trigger orchestration end event
        self.orchestrator._hooks.trigger(
            AgentEvent.ORCHESTRATION_END,
            session_id=self.orchestrator.session_id,
            duration_ms=duration_ms,
            data={"pattern": pattern, "agents_completed": len(results)},
        )
        self.orchestrator._logger.info(
            f"Completed {pattern} orchestration",
            duration_ms=duration_ms,
            pattern=pattern,
        )

        return results
