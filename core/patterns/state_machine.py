"""
State machine orchestration pattern handler.
"""

# pylint: disable=too-few-public-methods, arguments-differ, too-many-arguments, too-many-positional-arguments, protected-access, keyword-arg-before-vararg, too-many-locals

from typing import Any, Dict, List
from core.patterns.base import PatternHandler


class StateMachineHandler(PatternHandler):
    """Handler for state machine orchestration pattern."""

    def execute(
        self,
        states: Dict[str, Dict[str, Any]],
        initial_state: str,
        input_data: str,
        max_transitions: int = 10,
        verbose: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run state machine pattern with defined states and transitions.

        Each state has:
            - agent: Agent to run in this state
            - prompt: Prompt template
            - transitions: Dict of condition -> next_state

        Args:
            states: State definitions {state_name: {agent, prompt, transitions}}
            initial_state: Starting state name
            input_data: Initial input data
            max_transitions: Maximum state transitions
            verbose: Whether to print progress
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary with 'final_state', 'state_history', 'final_output'
        """
        if verbose:
            print(f"\n[StateMachine] Starting in state: {initial_state}")
            print(f"[StateMachine] Available states: {list(states.keys())}")

        current_state = initial_state
        state_history: List[Dict[str, Any]] = []
        current_output = ""
        self.orchestrator.context.set("input_data", input_data)

        for transition in range(max_transitions):
            if current_state not in states:
                raise ValueError(f"Unknown state: {current_state}")

            state_config = states[current_state]
            agent_name = state_config["agent"]
            prompt_template = state_config["prompt"]
            transitions = state_config.get("transitions", {})

            if verbose:
                print(
                    f"\n[StateMachine] State: {current_state} (transition {transition + 1})"
                )

            # Run agent for current state
            self.orchestrator.context.set("current_state", current_state)
            self.orchestrator.context.set("previous_output", current_output)
            prompt = self.orchestrator.context.interpolate(prompt_template)
            current_output = self.orchestrator.run_agent(agent_name, prompt, verbose)

            state_history.append(
                {
                    "state": current_state,
                    "agent": agent_name,
                    "output": current_output,
                }
            )

            # Determine next state
            next_state = None
            output_lower = current_output.lower()

            for condition, target_state in transitions.items():
                if condition == "*":  # Wildcard - always matches
                    next_state = target_state
                    break
                if condition.lower() in output_lower:
                    next_state = target_state
                    break

            if next_state is None or next_state == "END":
                if verbose:
                    print("[StateMachine] Reached terminal state or END")
                break

            current_state = next_state

        return {
            "final_state": current_state,
            "state_history": state_history,
            "final_output": current_output,
            "total_transitions": len(state_history),
        }
