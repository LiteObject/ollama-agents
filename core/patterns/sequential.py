"""
Sequential orchestration pattern handler.
"""

# pylint: disable=too-few-public-methods, arguments-differ, too-many-arguments, too-many-positional-arguments, protected-access, keyword-arg-before-vararg, duplicate-code

import time
from typing import Any, Dict, List, Optional

from core.observability.hooks import AgentEvent
from core.patterns.base import PatternHandler


class SequentialHandler(PatternHandler):
    """Handler for sequential execution pattern."""

    def execute(
        self,
        steps: List[tuple[str, str]],
        initial_vars: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ) -> str:
        """Run agents sequentially, passing outputs between them.

        Args:
            steps: List of (agent_name, prompt_template) tuples
            initial_vars: Initial variables for the context
            verbose: Whether to print progress

        Returns:
            Final agent's response
        """
        start_time = time.perf_counter()
        pattern = "sequential"

        # Trigger orchestration start event
        self.orchestrator._hooks.trigger(
            AgentEvent.ORCHESTRATION_START,
            session_id=self.orchestrator.session_id,
            data={"pattern": pattern, "steps": len(steps)},
        )
        self.orchestrator._logger.info(
            f"Starting {pattern} orchestration",
            pattern=pattern,
            extra={"steps": len(steps)},
        )

        # Set initial variables
        if initial_vars:
            for key, value in initial_vars.items():
                self.orchestrator._context.set(key, value)

        last_output = ""

        for step_num, (agent_name, prompt_template) in enumerate(steps, 1):
            if verbose:
                print(f"\n{'='*50}")
                print(f"Running agent: {agent_name}")
                print(f"{'='*50}")

            # Trigger step event
            self.orchestrator._hooks.trigger(
                AgentEvent.ORCHESTRATION_STEP,
                session_id=self.orchestrator.session_id,
                agent_name=agent_name,
                data={"step": step_num, "pattern": pattern},
            )
            self.orchestrator._logger.debug(
                f"Executing step {step_num}: {agent_name}",
                step=step_num,
                agent_name=agent_name,
                pattern=pattern,
            )

            agent = self.orchestrator.get_agent(agent_name)
            if not agent:
                raise ValueError(f"Agent '{agent_name}' not found")

            # Interpolate prompt with context
            prompt = self.orchestrator._context.interpolate(prompt_template)

            # Run agent
            last_output = agent.run(
                prompt, context=self.orchestrator._context, verbose=verbose
            )

            # Store output in context so next agent can access it
            self.orchestrator._context.set(agent_name, last_output)

            if verbose:
                print(f"\n[{agent_name}] Output: {last_output[:500]}...")

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Trigger orchestration end event
        self.orchestrator._hooks.trigger(
            AgentEvent.ORCHESTRATION_END,
            session_id=self.orchestrator.session_id,
            duration_ms=duration_ms,
            data={"pattern": pattern, "steps_completed": len(steps)},
        )
        self.orchestrator._logger.info(
            f"Completed {pattern} orchestration",
            duration_ms=duration_ms,
            pattern=pattern,
        )

        return last_output
