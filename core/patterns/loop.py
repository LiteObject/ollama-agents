"""
Loop orchestration pattern handler.
"""

# pylint: disable=too-few-public-methods, arguments-differ, too-many-arguments, too-many-positional-arguments, protected-access, keyword-arg-before-vararg

from typing import Callable

from core.patterns.base import PatternHandler


class LoopHandler(PatternHandler):
    """Handler for loop execution pattern."""

    def execute(
        self,
        agent_name: str,
        prompt_template: str,
        condition_fn: Callable[[str, int], bool],
        max_iterations: int = 5,
        verbose: bool = False,
    ) -> str:
        """Run an agent in a loop until a condition is met.

        Args:
            agent_name: Agent to run
            prompt_template: Prompt template (can include {iteration} placeholder)
            condition_fn: Function(output, iteration) -> bool to continue
            max_iterations: Maximum iterations
            verbose: Whether to print progress

        Returns:
            Final response
        """
        agent = self.orchestrator.get_agent(agent_name)
        if not agent:
            raise ValueError(f"Agent '{agent_name}' not found")

        last_output = ""
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Set iteration variable
            self.orchestrator._context.set("iteration", iteration)
            self.orchestrator._context.set("last_output", last_output)

            prompt = self.orchestrator._context.interpolate(prompt_template)

            if verbose:
                print(f"\n[Loop] Iteration {iteration}")

            last_output = agent.run(
                prompt, context=self.orchestrator._context, verbose=verbose
            )

            # Check condition
            if not condition_fn(last_output, iteration):
                if verbose:
                    print("[Loop] Condition met, stopping")
                break

        return last_output
