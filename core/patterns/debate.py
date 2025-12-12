"""
Debate orchestration pattern handler.
"""

# pylint: disable=too-few-public-methods, arguments-differ, too-many-arguments, too-many-positional-arguments, protected-access, keyword-arg-before-vararg, too-many-locals

from typing import List, Dict, Any, Optional
from core.patterns.base import PatternHandler


class DebateHandler(PatternHandler):
    """Handler for debate orchestration pattern."""

    def execute(
        self,
        debater_agents: List[str],
        topic: str,
        moderator_agent: Optional[str] = None,
        rounds: int = 3,
        verbose: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run a debate between agents with optional moderator.

        Agents take turns responding to each other's arguments.

        Args:
            debater_agents: List of debating agent names (usually 2)
            topic: Debate topic/question
            moderator_agent: Optional moderator to guide and conclude
            rounds: Number of debate rounds
            verbose: Whether to print progress
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary with 'debate_history' and optionally 'conclusion'
        """
        if verbose:
            print(f"\n[Debate] Topic: {topic}")
            print(f"[Debate] Debaters: {debater_agents}, Rounds: {rounds}")

        debate_history: List[Dict[str, str]] = []
        previous_arguments = ""

        for round_num in range(1, rounds + 1):
            if verbose:
                print(f"\n[Debate] === Round {round_num} ===")

            for debater in debater_agents:
                if round_num == 1 and debater == debater_agents[0]:
                    prompt = (
                        f"You are debating the topic: {topic}\n\n"
                        f"Present your opening argument."
                    )
                else:
                    prompt = (
                        f"Topic: {topic}\n\n"
                        f"Previous arguments:\n{previous_arguments}\n\n"
                        f"Respond to the previous arguments and strengthen your position."
                    )

                response = self.orchestrator.run_agent(debater, prompt, verbose)
                debate_history.append(
                    {"round": round_num, "agent": debater, "argument": response}
                )
                previous_arguments += f"\n\n[{debater}]: {response}"

                if verbose:
                    print(f"\n[{debater}]: {response[:300]}...")

        result: Dict[str, Any] = {"debate_history": debate_history}

        # Moderator conclusion
        if moderator_agent:
            conclusion_prompt = (
                f"You are moderating a debate on: {topic}\n\n"
                f"Full debate:\n{previous_arguments}\n\n"
                f"Provide a balanced summary and declare the stronger arguments."
            )
            result["conclusion"] = self.orchestrator.run_agent(
                moderator_agent, conclusion_prompt, verbose
            )

            if verbose:
                print(f"\n[Moderator Conclusion]: {result['conclusion'][:300]}...")

        self.orchestrator.context.set("debate_history", previous_arguments)
        return result
