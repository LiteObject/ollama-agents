"""
Supervisor orchestration pattern handler.
"""

# pylint: disable=too-few-public-methods, arguments-differ, too-many-arguments, too-many-positional-arguments, protected-access, keyword-arg-before-vararg, too-many-locals

from typing import List, Dict, Any, Optional, Callable
from core.patterns.base import PatternHandler


class SupervisorHandler(PatternHandler):
    """Handler for supervisor orchestration pattern."""

    def execute(
        self,
        supervisor_agent: str,
        worker_agent: str,
        task: str,
        max_revisions: int = 3,
        quality_threshold: Optional[Callable[[str], float]] = None,
        min_quality: float = 0.8,
        verbose: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run supervisor pattern with reflection and revision requests.

        The supervisor reviews worker output and can request revisions
        until quality is acceptable.

        Args:
            supervisor_agent: Agent that reviews and provides feedback
            worker_agent: Agent that produces the work
            task: Initial task description
            max_revisions: Maximum revision rounds
            quality_threshold: Optional function(output) -> quality score (0-1)
            min_quality: Minimum quality score to accept
            verbose: Whether to print progress
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary with 'final_output', 'revision_history', 'total_revisions'
        """
        if verbose:
            print(f"\n[Supervisor] Task: {task[:100]}...")
            print(
                f"[Supervisor] Worker: {worker_agent}, Max revisions: {max_revisions}"
            )

        revision_history: List[Dict[str, str]] = []
        current_output = ""
        feedback = ""

        for revision in range(max_revisions + 1):
            # Worker creates or revises
            if revision == 0:
                worker_prompt = f"Complete this task:\n\n{task}"
            else:
                worker_prompt = (
                    f"Original task: {task}\n\n"
                    f"Your previous work:\n{current_output}\n\n"
                    f"Supervisor feedback:\n{feedback}\n\n"
                    f"Please revise your work based on this feedback."
                )

            if verbose:
                print(f"\n[Supervisor] Revision {revision}: Worker producing output...")

            current_output = self.orchestrator.run_agent(
                worker_agent, worker_prompt, verbose
            )

            # Check quality threshold if provided
            if quality_threshold:
                quality = quality_threshold(current_output)
                if verbose:
                    print(f"[Supervisor] Quality score: {quality:.2f}")

                if quality >= min_quality:
                    if verbose:
                        print("[Supervisor] Quality threshold met!")
                    return {
                        "final_output": current_output,
                        "revision_history": revision_history,
                        "total_revisions": revision,
                    }

            # If last revision, stop
            if revision == max_revisions:
                if verbose:
                    print("[Supervisor] Max revisions reached.")
                break

            # Supervisor reviews
            if verbose:
                print("[Supervisor] Supervisor reviewing...")

            review_prompt = (
                f"Original task: {task}\n\n"
                f"Worker output:\n{current_output}\n\n"
                f"Provide constructive feedback on how to improve this work. "
                f"Be specific about what is missing or needs correction."
            )
            feedback = self.orchestrator.run_agent(
                supervisor_agent, review_prompt, verbose
            )

            revision_history.append(
                {
                    "revision": str(revision),
                    "output": current_output,
                    "feedback": feedback,
                }
            )

        return {
            "final_output": current_output,
            "revision_history": revision_history,
            "total_revisions": max_revisions,
        }
