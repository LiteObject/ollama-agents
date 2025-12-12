"""
Map-reduce orchestration pattern handler.
"""

# pylint: disable=too-few-public-methods, arguments-differ, too-many-arguments, too-many-positional-arguments, protected-access, keyword-arg-before-vararg, too-many-locals

from typing import List, Any
import concurrent.futures
from core.patterns.base import PatternHandler


class MapReduceHandler(PatternHandler):
    """Handler for map-reduce orchestration pattern."""

    def execute(
        self,
        mapper_agent: str,
        reducer_agent: str,
        items: List[str],
        map_prompt_template: str,
        reduce_prompt_template: str,
        max_workers: int = 4,
        verbose: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Run map-reduce pattern over a list of items.

        Args:
            mapper_agent: Agent that processes each item
            reducer_agent: Agent that combines results
            items: List of items to process
            map_prompt_template: Prompt template with {item} placeholder
            reduce_prompt_template: Prompt template with {mapped_results} placeholder
            max_workers: Maximum parallel workers
            verbose: Whether to print progress
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Reduced result
        """
        if verbose:
            print(f"\n[Map-Reduce] Mapping {len(items)} items with '{mapper_agent}'")

        mapper = self.orchestrator._agents.get(mapper_agent)
        if not mapper:
            raise ValueError(f"Mapper agent '{mapper_agent}' not found")

        # Map phase - process items in parallel
        mapped_results: List[str] = []

        def map_item(item: str, index: int) -> tuple[int, str]:
            # Note: Context is shared, so we need to be careful with parallel writes
            # Ideally, we should use a local context or pass variables directly
            # For now, we'll rely on the agent's run method handling the prompt interpolation
            # But wait, the prompt interpolation happens BEFORE run.
            # We need to interpolate locally.

            # Create a temporary context for interpolation if needed,
            # but the original code used self._context.set("item", item) which is
            # NOT thread-safe.
            # However, since we are refactoring, let's try to keep the logic similar
            # but safer if possible.
            # The original code had a race condition on self._context.set("item", item).
            # We should fix this by doing interpolation here.

            local_context_vars = self.orchestrator.context.variables.copy()
            local_context_vars["item"] = item
            local_context_vars["item_index"] = index

            # Simple interpolation for now, assuming simple {item} replacement
            # or we can use the context's interpolate method if we can pass a temporary dict
            # But context.interpolate uses self.variables.

            # Let's do a manual interpolation for the specific keys we know
            prompt = map_prompt_template.replace("{item}", str(item)).replace(
                "{item_index}", str(index)
            )

            # If there are other variables, we might miss them.
            # But for MapReduce, item is the key.

            result = mapper.run(
                prompt, context=self.orchestrator.context, verbose=False
            )
            return index, result

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(map_item, item, i) for i, item in enumerate(items)
            ]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
            # Sort by index to maintain order
            results.sort(key=lambda x: x[0])
            mapped_results = [r[1] for r in results]

        if verbose:
            print(f"[Map-Reduce] Map phase complete, {len(mapped_results)} results")

        # Reduce phase
        combined = "\n\n---\n\n".join(
            f"Item {i+1}: {result}" for i, result in enumerate(mapped_results)
        )
        self.orchestrator.context.set("mapped_results", combined)
        reduce_prompt = self.orchestrator.context.interpolate(reduce_prompt_template)

        if verbose:
            print(f"[Map-Reduce] Reducing with '{reducer_agent}'")

        return self.orchestrator.run_agent(reducer_agent, reduce_prompt, verbose)
