"""
Router orchestration pattern handler.
"""

# pylint: disable=too-few-public-methods, arguments-differ, too-many-arguments, too-many-positional-arguments, protected-access, keyword-arg-before-vararg, too-many-locals

from typing import Any, Dict, Optional
from core.patterns.base import PatternHandler


class RouterHandler(PatternHandler):
    """Handler for router orchestration pattern."""

    def execute(
        self,
        router_agent: str,
        routes: Dict[str, tuple[str, str]],
        input_prompt: str,
        default_route: Optional[tuple[str, str]] = None,
        verbose: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run router pattern that classifies input and routes to specialists.

        Args:
            router_agent: Agent that classifies the input
            routes: Dictionary of category -> (agent_name, prompt_template)
            input_prompt: User input to classify and route
            default_route: Default (agent_name, prompt_template) if no match
            verbose: Whether to print progress
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary with 'classification', 'route_taken', 'response'
        """
        if verbose:
            print("\n[Router] Classifying input...")

        # Build classification prompt
        categories = list(routes.keys())
        classify_prompt = (
            f"Classify the following input into one of these categories: "
            f"{', '.join(categories)}\n\n"
            f"Input: {input_prompt}\n\n"
            f"Respond with ONLY the category name, nothing else."
        )

        classification = self.orchestrator.run_agent(
            router_agent, classify_prompt, verbose
        )
        classification = classification.strip().lower()

        if verbose:
            print(f"[Router] Classification: {classification}")

        # Find matching route
        route_taken = None
        response = ""

        for category, (agent_name, prompt_template) in routes.items():
            if category.lower() in classification:
                route_taken = category
                self.orchestrator.context.set("user_input", input_prompt)
                self.orchestrator.context.set("classification", classification)
                prompt = self.orchestrator.context.interpolate(prompt_template)
                response = self.orchestrator.run_agent(agent_name, prompt, verbose)
                break
        else:
            # No match, use default
            if default_route:
                route_taken = "default"
                agent_name, prompt_template = default_route
                self.orchestrator.context.set("user_input", input_prompt)
                prompt = self.orchestrator.context.interpolate(prompt_template)
                response = self.orchestrator.run_agent(agent_name, prompt, verbose)
            else:
                response = f"Unable to route input. Classification: {classification}"

        return {
            "classification": classification,
            "route_taken": route_taken,
            "response": response,
        }
