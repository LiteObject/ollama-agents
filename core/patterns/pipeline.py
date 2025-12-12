"""
Pipeline orchestration pattern handler.
"""

# pylint: disable=too-few-public-methods, arguments-differ, too-many-arguments, too-many-positional-arguments, protected-access, keyword-arg-before-vararg

from typing import List, Any, Optional, Callable
from core.patterns.base import PatternHandler


class PipelineHandler(PatternHandler):
    """Handler for pipeline orchestration pattern."""

    def execute(
        self,
        stages: List[tuple[str, str, Optional[Callable[[str], bool]]]],
        initial_input: str,
        verbose: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Run a pipeline with optional filter stages.

        Each stage can have a filter function that determines if the
        pipeline should continue.

        Args:
            stages: List of (agent_name, prompt_template, filter_fn) tuples.
                    filter_fn(output) -> True to continue, False to stop
            initial_input: Initial input to the pipeline
            verbose: Whether to print progress
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Final output or last successful stage output
        """
        if verbose:
            print(f"\n[Pipeline] Starting with {len(stages)} stages")

        self.orchestrator.context.set("pipeline_input", initial_input)
        current_output = initial_input

        for i, stage in enumerate(stages):
            agent_name = stage[0]
            prompt_template = stage[1]
            filter_fn = stage[2] if len(stage) > 2 else None

            if verbose:
                print(f"\n[Pipeline] Stage {i+1}: {agent_name}")

            self.orchestrator.context.set("stage_input", current_output)
            self.orchestrator.context.set("stage_number", i + 1)
            prompt = self.orchestrator.context.interpolate(prompt_template)

            current_output = self.orchestrator.run_agent(agent_name, prompt, verbose)
            self.orchestrator.context.set(f"stage_{i+1}_output", current_output)

            # Apply filter if present
            if filter_fn and not filter_fn(current_output):
                if verbose:
                    print(f"[Pipeline] Filter stopped at stage {i+1}")
                break

        return current_output
