"""
Voting orchestration pattern handler.
"""

# pylint: disable=too-few-public-methods, arguments-differ, too-many-arguments, too-many-positional-arguments, protected-access, keyword-arg-before-vararg

from typing import List, Dict, Any, Optional
from core.patterns.base import PatternHandler


class VotingHandler(PatternHandler):
    """Handler for voting orchestration pattern."""

    def execute(
        self,
        voter_agents: List[str],
        prompt: str,
        aggregator_agent: Optional[str] = None,
        aggregation_prompt: Optional[str] = None,
        verbose: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run voting pattern where multiple agents provide responses.

        All agents answer the same prompt, then optionally an aggregator
        synthesizes the consensus.

        Args:
            voter_agents: List of agent names to vote
            prompt: Prompt for all voters
            aggregator_agent: Optional agent to aggregate votes
            aggregation_prompt: Prompt for aggregation with {votes} placeholder
            verbose: Whether to print progress
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary with 'votes' and optionally 'consensus'
        """
        if verbose:
            print(f"\n[Voting] {len(voter_agents)} agents voting")

        # Get all votes in parallel
        voter_prompts = {agent: prompt for agent in voter_agents}
        votes = self.orchestrator.run_parallel(voter_prompts, verbose=verbose)

        result: Dict[str, Any] = {"votes": votes}

        # Aggregate if aggregator is provided
        if aggregator_agent and aggregation_prompt:
            votes_summary = "\n\n".join(
                f"[{agent}] voted: {vote}" for agent, vote in votes.items()
            )
            self.orchestrator.context.set("votes", votes_summary)
            consensus_prompt = self.orchestrator.context.interpolate(aggregation_prompt)
            result["consensus"] = self.orchestrator.run_agent(
                aggregator_agent, consensus_prompt, verbose
            )

            if verbose:
                print(f"\n[Voting] Consensus: {result['consensus'][:200]}...")

        return result
