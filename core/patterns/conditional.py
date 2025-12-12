"""
Conditional orchestration pattern handler.
"""

# pylint: disable=too-few-public-methods, arguments-differ, too-many-arguments, too-many-positional-arguments, protected-access, keyword-arg-before-vararg

from typing import Dict, Optional, Tuple

from core.patterns.base import PatternHandler


class ConditionalHandler(PatternHandler):
    """Handler for conditional execution pattern."""

    def execute(
        self,
        condition_agent: str,
        condition_prompt: str,
        branches: Dict[str, Tuple[str, str]],
        default_branch: Optional[Tuple[str, str]] = None,
        verbose: bool = False,
    ) -> str:
        """Run conditional workflow based on an agent's output.

        Args:
            condition_agent: Agent that determines the branch
            condition_prompt: Prompt for the condition agent
            branches: Dictionary of condition_value -> (agent_name, prompt)
            default_branch: Default (agent_name, prompt) if no match
            verbose: Whether to print progress

        Returns:
            Final response
        """
        # Get condition result
        condition_result = self.orchestrator.run_agent(
            condition_agent, condition_prompt, verbose
        )
        condition_result_lower = condition_result.lower().strip()

        if verbose:
            print(f"\n[Conditional] Condition result: {condition_result_lower}")

        # Find matching branch
        for condition, (agent_name, prompt) in branches.items():
            if condition.lower() in condition_result_lower:
                if verbose:
                    print(f"[Conditional] Taking branch: {condition}")
                return self.orchestrator.run_agent(agent_name, prompt, verbose)

        # Use default branch if provided
        if default_branch:
            agent_name, prompt = default_branch
            if verbose:
                print("[Conditional] Taking default branch")
            return self.orchestrator.run_agent(agent_name, prompt, verbose)

        return condition_result
