"""
Hierarchical orchestration pattern handler.
"""

# pylint: disable=too-few-public-methods, arguments-differ, too-many-arguments, too-many-positional-arguments, protected-access, keyword-arg-before-vararg, too-many-locals

from typing import List, Any
from core.patterns.base import PatternHandler


class HierarchicalHandler(PatternHandler):
    """Handler for hierarchical orchestration pattern."""

    def execute(
        self,
        supervisor_agent: str,
        worker_agents: List[str],
        task_prompt: str,
        aggregation_prompt: str,
        verbose: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Run hierarchical orchestration with a supervisor delegating to workers.

        Args:
            supervisor_agent: Agent that coordinates the work
            worker_agents: List of worker agent names
            task_prompt: Initial task description
            aggregation_prompt: Prompt for aggregating results
            verbose: Whether to print progress
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Final aggregated response
        """
        if verbose:
            print(f"\n[Hierarchical] Supervisor: {supervisor_agent}")
            print(f"[Hierarchical] Workers: {worker_agents}")

        # Supervisor analyzes and creates sub-tasks
        delegation_prompt = (
            f"Analyze this task and create specific sub-tasks for each worker.\n"
            f"Workers available: {', '.join(worker_agents)}\n"
            f"Task: {task_prompt}\n\n"
            f"Output format: For each worker, provide their specific sub-task."
        )
        delegation = self.orchestrator.run_agent(
            supervisor_agent, delegation_prompt, verbose
        )
        self.orchestrator.context.set("delegation", delegation)

        if verbose:
            print(f"\n[Hierarchical] Delegation: {delegation[:300]}...")

        # Workers execute in parallel
        worker_prompts = {
            worker: f"Your assignment from the supervisor:\n{delegation}\n\n"
            f"Original task: {task_prompt}\n\n"
            f"Complete your part of the task."
            for worker in worker_agents
        }
        worker_results = self.orchestrator.run_parallel(worker_prompts, verbose=verbose)

        # Store worker results
        for worker, result in worker_results.items():
            self.orchestrator.context.set(f"{worker}_result", result)

        # Supervisor aggregates results
        results_summary = "\n\n".join(
            f"[{worker}]: {result}" for worker, result in worker_results.items()
        )
        self.orchestrator.context.set("worker_results", results_summary)

        final_prompt = self.orchestrator.context.interpolate(aggregation_prompt)
        return self.orchestrator.run_agent(supervisor_agent, final_prompt, verbose)
