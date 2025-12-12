"""
Ensemble orchestration pattern handler.
"""

# pylint: disable=too-few-public-methods, arguments-differ, too-many-arguments, too-many-positional-arguments, protected-access, keyword-arg-before-vararg

from typing import Any, Dict, List, Optional
from core.patterns.base import PatternHandler


class EnsembleHandler(PatternHandler):
    """Handler for ensemble orchestration pattern."""

    def execute(
        self,
        agents: List[str],
        prompt: str,
        weights: Optional[Dict[str, float]] = None,
        combiner_agent: Optional[str] = None,
        combination_strategy: str = "synthesize",
        verbose: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run ensemble pattern combining multiple agent responses.

        Args:
            agents: List of agent names to run
            prompt: Prompt for all agents
            weights: Optional weights for each agent (for ranking)
            combiner_agent: Optional agent to combine responses
            combination_strategy: How to combine ('synthesize', 'best', 'merge')
            verbose: Whether to print progress
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary with 'individual_responses', 'combined_response', 'weights'
        """
        if verbose:
            print(f"\n[Ensemble] Running {len(agents)} agents...")

        # Get all responses in parallel
        agent_prompts = {agent: prompt for agent in agents}
        individual_responses = self.orchestrator.run_parallel(
            agent_prompts, verbose=verbose
        )

        # Apply weights if provided
        weights = weights or {agent: 1.0 for agent in agents}

        result: Dict[str, Any] = {
            "individual_responses": individual_responses,
            "weights": weights,
        }

        # Combine responses
        if combiner_agent:
            if combination_strategy == "synthesize":
                combine_prompt = (
                    f"Multiple experts have provided responses to: {prompt}\n\n"
                    + "\n\n".join(
                        f"[Expert {i+1} (weight: {weights.get(agent, 1.0)})]: {resp}"
                        for i, (agent, resp) in enumerate(individual_responses.items())
                    )
                    + "\n\nSynthesize these responses into a single comprehensive answer, "
                    "giving more weight to higher-weighted experts."
                )
            elif combination_strategy == "best":
                combine_prompt = (
                    f"Multiple experts have provided responses to: {prompt}\n\n"
                    + "\n\n".join(
                        f"[Expert {i+1}]: {resp}"
                        for i, resp in enumerate(individual_responses.values())
                    )
                    + "\n\nSelect the BEST response and explain why it's superior."
                )
            else:  # merge
                combine_prompt = (
                    f"Multiple experts have provided responses to: {prompt}\n\n"
                    + "\n\n".join(
                        f"[Expert {i+1}]: {resp}"
                        for i, resp in enumerate(individual_responses.values())
                    )
                    + "\n\nMerge all unique information from these responses."
                )

            result["combined_response"] = self.orchestrator.run_agent(
                combiner_agent, combine_prompt, verbose
            )
        else:
            # Just concatenate if no combiner
            result["combined_response"] = "\n\n---\n\n".join(
                f"[{agent}]: {resp}" for agent, resp in individual_responses.items()
            )

        return result
