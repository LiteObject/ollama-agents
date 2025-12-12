"""
Chain of thought orchestration pattern handler.
"""

# pylint: disable=too-few-public-methods, arguments-differ, too-many-arguments, too-many-positional-arguments, protected-access, keyword-arg-before-vararg

from typing import List, Dict, Any
from core.patterns.base import PatternHandler


class ChainOfThoughtHandler(PatternHandler):
    """Handler for chain-of-thought orchestration pattern."""

    def execute(
        self,
        agent_name: str,
        problem: str,
        thinking_steps: int = 3,
        verbose: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run chain-of-thought reasoning with explicit steps.

        Args:
            agent_name: Agent to perform reasoning
            problem: Problem to solve
            thinking_steps: Number of thinking steps
            verbose: Whether to print progress
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary with 'steps' and 'final_answer'
        """
        if verbose:
            print(f"\n[Chain-of-Thought] Problem: {problem[:100]}...")

        steps: List[str] = []
        previous_thinking = ""

        for step in range(1, thinking_steps + 1):
            if step == 1:
                prompt = (
                    f"Problem: {problem}\n\n"
                    f"Step {step}/{thinking_steps}: Break down this problem. "
                    f"What are the key components we need to address?"
                )
            elif step == thinking_steps:
                prompt = (
                    f"Problem: {problem}\n\n"
                    f"Previous thinking:\n{previous_thinking}\n\n"
                    f"Step {step}/{thinking_steps}: Based on your analysis, "
                    f"provide the final answer or solution."
                )
            else:
                prompt = (
                    f"Problem: {problem}\n\n"
                    f"Previous thinking:\n{previous_thinking}\n\n"
                    f"Step {step}/{thinking_steps}: Continue your analysis. "
                    f"What insights can you derive?"
                )

            if verbose:
                print(f"\n[CoT] Step {step}/{thinking_steps}")

            response = self.orchestrator.run_agent(agent_name, prompt, verbose)
            steps.append(response)
            previous_thinking += f"\nStep {step}: {response}"

        return {"steps": steps, "final_answer": steps[-1]}
