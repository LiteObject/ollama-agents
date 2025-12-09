"""
Orchestrator for coordinating multiple agents.

This module provides an Orchestrator class that manages the execution
of multiple agents in various patterns: sequential, parallel, conditional, etc.
"""

import concurrent.futures
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from core.agent import Agent
from core.config import AgentConfig
from core.context import SharedContext
from core.observability.hooks import (
    AgentEvent,
    EventHookRegistry,
    default_hook_registry,
)
from core.observability.logging import AgentLogger, get_logger
from core.tools import ToolRegistry, default_registry


class ExecutionPattern(Enum):
    """Supported execution patterns for orchestration."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    HIERARCHICAL = "hierarchical"
    MAP_REDUCE = "map_reduce"
    PIPELINE = "pipeline"
    VOTING = "voting"
    DEBATE = "debate"
    # Phase 2 patterns
    SUPERVISOR = "supervisor"
    STATE_MACHINE = "state_machine"
    ROUTER = "router"
    ENSEMBLE = "ensemble"
    EVENT_DRIVEN = "event_driven"


@dataclass
class WorkflowStep:
    """A single step in a workflow.

    Attributes:
        agent: Name of the agent to execute
        prompt: Prompt template (can include {variable} placeholders)
        output_var: Variable name to store the output
        condition: Optional condition for execution (for conditional patterns)
        max_loops: Maximum iterations (for loop patterns)
        loop_condition: Condition to continue looping
    """

    agent: str
    prompt: str
    output_var: Optional[str] = None
    condition: Optional[str] = None
    max_loops: int = 5
    loop_condition: Optional[str] = None


@dataclass
class Workflow:
    """A workflow definition with multiple steps.

    Attributes:
        name: Workflow name
        description: Human-readable description
        steps: List of workflow steps
        pattern: Execution pattern
    """

    name: str
    steps: List[WorkflowStep]
    description: str = ""
    pattern: ExecutionPattern = ExecutionPattern.SEQUENTIAL

    @classmethod
    def from_yaml(cls, path: Path | str) -> "Workflow":
        """Load workflow from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            Workflow instance
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        steps = [
            WorkflowStep(
                agent=s["agent"],
                prompt=s["prompt"],
                output_var=s.get("output_var"),
                condition=s.get("condition"),
                max_loops=s.get("max_loops", 5),
                loop_condition=s.get("loop_condition"),
            )
            for s in data.get("steps", [])
        ]

        pattern = ExecutionPattern(data.get("pattern", "sequential"))

        return cls(
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            steps=steps,
            pattern=pattern,
        )


class Orchestrator:
    """Orchestrator for coordinating multiple agents.

    This class manages the creation and execution of agents, handling
    various execution patterns and inter-agent communication via shared context.

    Usage:
        # Create orchestrator
        orchestrator = Orchestrator()

        # Add agents
        orchestrator.add_agent(AgentConfig(name="researcher", ...))
        orchestrator.add_agent(AgentConfig(name="writer", ...))

        # Run sequential workflow
        result = orchestrator.run_sequential([
            ("researcher", "Research {topic}"),
            ("writer", "Write article based on: {researcher}"),
        ], initial_vars={"topic": "AI trends"})
    """

    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        context: Optional[SharedContext] = None,
        hook_registry: Optional[EventHookRegistry] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            tool_registry: Tool registry for agents
            context: Shared context for inter-agent communication
            hook_registry: Event hook registry for observability
            session_id: Session identifier for logging/tracing
        """
        self._registry = tool_registry or default_registry
        self._context = context or SharedContext()
        self._hooks = hook_registry or default_hook_registry
        self.session_id = session_id or str(uuid.uuid4())
        self._agents: Dict[str, Agent] = {}
        self._configs: Dict[str, AgentConfig] = {}

        # Set up logging
        self._logger: AgentLogger = get_logger(
            name="orchestrator",
            session_id=self.session_id,
        )
        self._logger.info("Orchestrator initialized")

    @property
    def context(self) -> SharedContext:
        """Get the shared context."""
        return self._context

    def add_agent(self, config: AgentConfig) -> Agent:
        """Add an agent to the orchestrator.

        Args:
            config: Agent configuration

        Returns:
            The created Agent instance
        """
        agent = Agent(
            config=config,
            tool_registry=self._registry,
            context=self._context,
            hook_registry=self._hooks,
            session_id=self.session_id,
        )
        self._agents[config.name] = agent
        self._configs[config.name] = config
        self._logger.info(f"Added agent: {config.name}", agent_name=config.name)
        return agent

    def add_agent_from_yaml(self, path: Path | str) -> Agent:
        """Add an agent from a YAML configuration file.

        Args:
            path: Path to YAML config file

        Returns:
            The created Agent instance
        """
        config = AgentConfig.from_yaml(path)
        return self.add_agent(config)

    def get_agent(self, name: str) -> Optional[Agent]:
        """Get an agent by name.

        Args:
            name: Agent name

        Returns:
            Agent instance or None
        """
        return self._agents.get(name)

    def list_agents(self) -> List[str]:
        """Get list of agent names.

        Returns:
            List of agent names
        """
        return list(self._agents.keys())

    def run_agent(
        self,
        name: str,
        prompt: str,
        verbose: bool = False,
    ) -> str:
        """Run a single agent.

        Args:
            name: Agent name
            prompt: Prompt for the agent
            verbose: Whether to print progress

        Returns:
            Agent response
        """
        agent = self._agents.get(name)
        if not agent:
            raise ValueError(f"Agent '{name}' not found")

        return agent.run(prompt, context=self._context, verbose=verbose)

    def run_sequential(
        self,
        steps: List[tuple[str, str]],
        initial_vars: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ) -> str:
        """Run agents sequentially, passing outputs between them.

        Args:
            steps: List of (agent_name, prompt_template) tuples
            initial_vars: Initial variables for the context
            verbose: Whether to print progress

        Returns:
            Final agent's response
        """
        start_time = time.perf_counter()
        pattern = "sequential"

        # Trigger orchestration start event
        self._hooks.trigger(
            AgentEvent.ORCHESTRATION_START,
            session_id=self.session_id,
            data={"pattern": pattern, "steps": len(steps)},
        )
        self._logger.info(
            f"Starting {pattern} orchestration",
            pattern=pattern,
            extra={"steps": len(steps)},
        )

        # Set initial variables
        if initial_vars:
            for key, value in initial_vars.items():
                self._context.set(key, value)

        last_output = ""

        for step_num, (agent_name, prompt_template) in enumerate(steps, 1):
            if verbose:
                print(f"\n{'='*50}")
                print(f"Running agent: {agent_name}")
                print(f"{'='*50}")

            # Trigger step event
            self._hooks.trigger(
                AgentEvent.ORCHESTRATION_STEP,
                session_id=self.session_id,
                agent_name=agent_name,
                data={"step": step_num, "pattern": pattern},
            )
            self._logger.debug(
                f"Executing step {step_num}: {agent_name}",
                step=step_num,
                agent_name=agent_name,
                pattern=pattern,
            )

            agent = self._agents.get(agent_name)
            if not agent:
                raise ValueError(f"Agent '{agent_name}' not found")

            # Interpolate prompt with context
            prompt = self._context.interpolate(prompt_template)

            # Run agent
            last_output = agent.run(prompt, context=self._context, verbose=verbose)

            if verbose:
                print(f"\n[{agent_name}] Output: {last_output[:500]}...")

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Trigger orchestration end event
        self._hooks.trigger(
            AgentEvent.ORCHESTRATION_END,
            session_id=self.session_id,
            duration_ms=duration_ms,
            data={"pattern": pattern, "steps_completed": len(steps)},
        )
        self._logger.info(
            f"Completed {pattern} orchestration",
            duration_ms=duration_ms,
            pattern=pattern,
        )

        return last_output

    def run_parallel(
        self,
        agent_prompts: Dict[str, str],
        verbose: bool = False,
        max_workers: int = 4,
    ) -> Dict[str, str]:
        """Run multiple agents in parallel.

        Args:
            agent_prompts: Dictionary of agent_name -> prompt
            verbose: Whether to print progress
            max_workers: Maximum parallel workers

        Returns:
            Dictionary of agent_name -> response
        """
        start_time = time.perf_counter()
        pattern = "parallel"

        # Trigger orchestration start event
        self._hooks.trigger(
            AgentEvent.ORCHESTRATION_START,
            session_id=self.session_id,
            data={"pattern": pattern, "agents": list(agent_prompts.keys())},
        )
        self._logger.info(
            f"Starting {pattern} orchestration",
            pattern=pattern,
            extra={"agents": list(agent_prompts.keys())},
        )

        results: Dict[str, str] = {}

        def run_single(name: str, prompt: str) -> tuple[str, str]:
            agent = self._agents.get(name)
            if not agent:
                return name, f"Error: Agent '{name}' not found"
            interpolated = self._context.interpolate(prompt)
            return name, agent.run(interpolated, context=self._context, verbose=verbose)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_single, name, prompt): name
                for name, prompt in agent_prompts.items()
            }

            for future in concurrent.futures.as_completed(futures):
                name, result = future.result()
                results[name] = result

                if verbose:
                    print(f"\n[{name}] Completed: {result[:200]}...")

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Trigger orchestration end event
        self._hooks.trigger(
            AgentEvent.ORCHESTRATION_END,
            session_id=self.session_id,
            duration_ms=duration_ms,
            data={"pattern": pattern, "agents_completed": len(results)},
        )
        self._logger.info(
            f"Completed {pattern} orchestration",
            duration_ms=duration_ms,
            pattern=pattern,
        )

        return results

    def run_conditional(
        self,
        condition_agent: str,
        condition_prompt: str,
        branches: Dict[str, tuple[str, str]],
        default_branch: Optional[tuple[str, str]] = None,
        verbose: bool = False,
    ) -> str:
        """Run conditional workflow based on an agent's output.

        Args:
            condition_agent: Agent that determines the branch
            condition_prompt: Prompt for the condition agent
            branches: Dictionary of condition_value -> (agent_name, prompt)
            default_branch: Default (agent_name, prompt) if no match
            verbose: Whether to print progress

        Returns:
            Final response
        """
        # Get condition result
        condition_result = self.run_agent(condition_agent, condition_prompt, verbose)
        condition_result_lower = condition_result.lower().strip()

        if verbose:
            print(f"\n[Conditional] Condition result: {condition_result_lower}")

        # Find matching branch
        for condition, (agent_name, prompt) in branches.items():
            if condition.lower() in condition_result_lower:
                if verbose:
                    print(f"[Conditional] Taking branch: {condition}")
                return self.run_agent(agent_name, prompt, verbose)

        # Use default branch if provided
        if default_branch:
            agent_name, prompt = default_branch
            if verbose:
                print("[Conditional] Taking default branch")
            return self.run_agent(agent_name, prompt, verbose)

        return condition_result

    def run_loop(
        self,
        agent_name: str,
        prompt_template: str,
        condition_fn: Callable[[str, int], bool],
        max_iterations: int = 5,
        verbose: bool = False,
    ) -> str:
        """Run an agent in a loop until a condition is met.

        Args:
            agent_name: Agent to run
            prompt_template: Prompt template (can include {iteration} placeholder)
            condition_fn: Function(output, iteration) -> bool to continue
            max_iterations: Maximum iterations
            verbose: Whether to print progress

        Returns:
            Final response
        """
        agent = self._agents.get(agent_name)
        if not agent:
            raise ValueError(f"Agent '{agent_name}' not found")

        last_output = ""
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Set iteration variable
            self._context.set("iteration", iteration)
            self._context.set("last_output", last_output)

            prompt = self._context.interpolate(prompt_template)

            if verbose:
                print(f"\n[Loop] Iteration {iteration}")

            last_output = agent.run(prompt, context=self._context, verbose=verbose)

            # Check condition
            if not condition_fn(last_output, iteration):
                if verbose:
                    print("[Loop] Condition met, stopping")
                break

        return last_output

    def run_workflow(self, workflow: Workflow, verbose: bool = False) -> str:
        """Run a complete workflow.

        Args:
            workflow: Workflow definition
            verbose: Whether to print progress

        Returns:
            Final output
        """
        if verbose:
            print(f"\n{'#'*60}")
            print(f"# Running workflow: {workflow.name}")
            print(f"# Pattern: {workflow.pattern.value}")
            print(f"{'#'*60}")

        if workflow.pattern == ExecutionPattern.SEQUENTIAL:
            steps = [(step.agent, step.prompt) for step in workflow.steps]
            return self.run_sequential(steps, verbose=verbose)

        elif workflow.pattern == ExecutionPattern.PARALLEL:
            prompts = {step.agent: step.prompt for step in workflow.steps}
            results = self.run_parallel(prompts, verbose=verbose)
            # Return all results combined
            return "\n\n".join(
                f"[{name}]: {result}" for name, result in results.items()
            )

        elif workflow.pattern == ExecutionPattern.PIPELINE:
            stages = [(step.agent, step.prompt, None) for step in workflow.steps]
            initial_input = self._context.get("pipeline_input", "")
            return self.run_pipeline(stages, str(initial_input), verbose=verbose)

        elif workflow.pattern == ExecutionPattern.VOTING:
            if not workflow.steps:
                raise ValueError("Voting workflow requires at least one step")
            # First step defines voters, last step (if different) is aggregator
            voter_agents = [step.agent for step in workflow.steps[:-1]] or [
                workflow.steps[0].agent
            ]
            aggregator = workflow.steps[-1].agent if len(workflow.steps) > 1 else None
            prompt = workflow.steps[0].prompt
            result = self.run_voting(
                voter_agents=voter_agents,
                prompt=prompt,
                aggregator_agent=aggregator,
                aggregation_prompt=(workflow.steps[-1].prompt if aggregator else None),
                verbose=verbose,
            )
            return result.get("consensus", str(result.get("votes", "")))

        else:
            raise NotImplementedError(f"Pattern {workflow.pattern} not yet implemented")

    def run_hierarchical(
        self,
        supervisor_agent: str,
        worker_agents: List[str],
        task_prompt: str,
        aggregation_prompt: str,
        verbose: bool = False,
    ) -> str:
        """Run hierarchical orchestration with a supervisor delegating to workers.

        The supervisor analyzes the task and delegates to workers, then
        aggregates their results.

        Args:
            supervisor_agent: Agent that coordinates the work
            worker_agents: List of worker agent names
            task_prompt: Initial task description
            aggregation_prompt: Prompt for aggregating results
            verbose: Whether to print progress

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
        delegation = self.run_agent(supervisor_agent, delegation_prompt, verbose)
        self._context.set("delegation", delegation)

        if verbose:
            print(f"\n[Hierarchical] Delegation: {delegation[:300]}...")

        # Workers execute in parallel
        worker_prompts = {
            worker: f"Your assignment from the supervisor:\n{delegation}\n\n"
            f"Original task: {task_prompt}\n\n"
            f"Complete your part of the task."
            for worker in worker_agents
        }
        worker_results = self.run_parallel(worker_prompts, verbose=verbose)

        # Store worker results
        for worker, result in worker_results.items():
            self._context.set(f"{worker}_result", result)

        # Supervisor aggregates results
        results_summary = "\n\n".join(
            f"[{worker}]: {result}" for worker, result in worker_results.items()
        )
        self._context.set("worker_results", results_summary)

        final_prompt = self._context.interpolate(aggregation_prompt)
        return self.run_agent(supervisor_agent, final_prompt, verbose)

    def run_map_reduce(
        self,
        mapper_agent: str,
        reducer_agent: str,
        items: List[str],
        map_prompt_template: str,
        reduce_prompt_template: str,
        max_workers: int = 4,
        verbose: bool = False,
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

        Returns:
            Reduced result
        """
        if verbose:
            print(f"\n[Map-Reduce] Mapping {len(items)} items with '{mapper_agent}'")

        mapper = self._agents.get(mapper_agent)
        if not mapper:
            raise ValueError(f"Mapper agent '{mapper_agent}' not found")

        # Map phase - process items in parallel
        mapped_results: List[str] = []

        def map_item(item: str, index: int) -> tuple[int, str]:
            self._context.set("item", item)
            self._context.set("item_index", index)
            prompt = self._context.interpolate(map_prompt_template)
            result = mapper.run(prompt, context=self._context, verbose=False)
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
        self._context.set("mapped_results", combined)
        reduce_prompt = self._context.interpolate(reduce_prompt_template)

        if verbose:
            print(f"[Map-Reduce] Reducing with '{reducer_agent}'")

        return self.run_agent(reducer_agent, reduce_prompt, verbose)

    def run_pipeline(
        self,
        stages: List[tuple[str, str, Optional[Callable[[str], bool]]]],
        initial_input: str,
        verbose: bool = False,
    ) -> str:
        """Run a pipeline with optional filter stages.

        Each stage can have a filter function that determines if the
        pipeline should continue.

        Args:
            stages: List of (agent_name, prompt_template, filter_fn) tuples.
                    filter_fn(output) -> True to continue, False to stop
            initial_input: Initial input to the pipeline
            verbose: Whether to print progress

        Returns:
            Final output or last successful stage output
        """
        if verbose:
            print(f"\n[Pipeline] Starting with {len(stages)} stages")

        self._context.set("pipeline_input", initial_input)
        current_output = initial_input

        for i, stage in enumerate(stages):
            agent_name = stage[0]
            prompt_template = stage[1]
            filter_fn = stage[2] if len(stage) > 2 else None

            if verbose:
                print(f"\n[Pipeline] Stage {i+1}: {agent_name}")

            self._context.set("stage_input", current_output)
            self._context.set("stage_number", i + 1)
            prompt = self._context.interpolate(prompt_template)

            current_output = self.run_agent(agent_name, prompt, verbose)
            self._context.set(f"stage_{i+1}_output", current_output)

            # Apply filter if present
            if filter_fn and not filter_fn(current_output):
                if verbose:
                    print(f"[Pipeline] Filter stopped at stage {i+1}")
                break

        return current_output

    def run_voting(
        self,
        voter_agents: List[str],
        prompt: str,
        aggregator_agent: Optional[str] = None,
        aggregation_prompt: Optional[str] = None,
        verbose: bool = False,
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

        Returns:
            Dictionary with 'votes' and optionally 'consensus'
        """
        if verbose:
            print(f"\n[Voting] {len(voter_agents)} agents voting")

        # Get all votes in parallel
        voter_prompts = {agent: prompt for agent in voter_agents}
        votes = self.run_parallel(voter_prompts, verbose=verbose)

        result: Dict[str, Any] = {"votes": votes}

        # Aggregate if aggregator is provided
        if aggregator_agent and aggregation_prompt:
            votes_summary = "\n\n".join(
                f"[{agent}] voted: {vote}" for agent, vote in votes.items()
            )
            self._context.set("votes", votes_summary)
            consensus_prompt = self._context.interpolate(aggregation_prompt)
            result["consensus"] = self.run_agent(
                aggregator_agent, consensus_prompt, verbose
            )

            if verbose:
                print(f"\n[Voting] Consensus: {result['consensus'][:200]}...")

        return result

    def run_debate(
        self,
        debater_agents: List[str],
        topic: str,
        moderator_agent: Optional[str] = None,
        rounds: int = 3,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Run a debate between agents with optional moderator.

        Agents take turns responding to each other's arguments.

        Args:
            debater_agents: List of debating agent names (usually 2)
            topic: Debate topic/question
            moderator_agent: Optional moderator to guide and conclude
            rounds: Number of debate rounds
            verbose: Whether to print progress

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

                response = self.run_agent(debater, prompt, verbose)
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
            result["conclusion"] = self.run_agent(
                moderator_agent, conclusion_prompt, verbose
            )

            if verbose:
                print(f"\n[Moderator Conclusion]: {result['conclusion'][:300]}...")

        self._context.set("debate_history", previous_arguments)
        return result

    def run_retry(
        self,
        primary_agent: str,
        prompt: str,
        validator_fn: Callable[[str], bool],
        fallback_agents: Optional[List[str]] = None,
        max_retries: int = 3,
        verbose: bool = False,
    ) -> tuple[str, str]:
        """Run with retry and optional fallback agents.

        Attempts the primary agent, retries on validation failure,
        then falls back to alternative agents.

        Args:
            primary_agent: Primary agent to try first
            prompt: Prompt for the agents
            validator_fn: Function(output) -> True if valid
            fallback_agents: List of fallback agent names
            max_retries: Maximum retries per agent
            verbose: Whether to print progress

        Returns:
            Tuple of (successful_agent_name, output)
        """
        agents_to_try = [primary_agent] + (fallback_agents or [])

        for agent_name in agents_to_try:
            if verbose:
                print(f"\n[Retry] Trying agent: {agent_name}")

            for attempt in range(1, max_retries + 1):
                if verbose:
                    print(f"[Retry] Attempt {attempt}/{max_retries}")

                self._context.set("retry_attempt", attempt)
                interpolated_prompt = self._context.interpolate(prompt)
                output = self.run_agent(agent_name, interpolated_prompt, verbose)

                if validator_fn(output):
                    if verbose:
                        print("[Retry] Validation passed!")
                    return agent_name, output

                if verbose:
                    print("[Retry] Validation failed, retrying...")

        # All agents failed
        raise RuntimeError(
            f"All agents failed validation after retries: {agents_to_try}"
        )

    def run_chain_of_thought(
        self,
        agent_name: str,
        problem: str,
        thinking_steps: int = 3,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Run chain-of-thought reasoning with explicit steps.

        Args:
            agent_name: Agent to perform reasoning
            problem: Problem to solve
            thinking_steps: Number of thinking steps
            verbose: Whether to print progress

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

            response = self.run_agent(agent_name, prompt, verbose)
            steps.append(response)
            previous_thinking += f"\nStep {step}: {response}"

        return {"steps": steps, "final_answer": steps[-1]}

    # =========================================================================
    # Phase 2: Advanced Orchestration Patterns
    # =========================================================================

    def run_supervisor(
        self,
        supervisor_agent: str,
        worker_agent: str,
        task: str,
        max_revisions: int = 3,
        quality_threshold: Optional[Callable[[str], float]] = None,
        min_quality: float = 0.8,
        verbose: bool = False,
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

            current_output = self.run_agent(worker_agent, worker_prompt, verbose)

            # Check quality threshold if provided
            if quality_threshold:
                quality = quality_threshold(current_output)
                if verbose:
                    print(f"[Supervisor] Quality score: {quality:.2f}")
                if quality >= min_quality:
                    revision_history.append(
                        {
                            "revision": revision,
                            "output": current_output,
                            "feedback": "Passed quality threshold",
                            "quality": quality,
                        }
                    )
                    break

            # Supervisor reviews
            review_prompt = (
                f"Review this work for the task: {task}\n\n"
                f"Work submitted:\n{current_output}\n\n"
                f"Provide feedback. If the work is satisfactory, respond with 'APPROVED'. "
                f"Otherwise, provide specific feedback for improvement."
            )

            if verbose:
                print("[Supervisor] Supervisor reviewing...")

            feedback = self.run_agent(supervisor_agent, review_prompt, verbose)

            revision_history.append(
                {
                    "revision": revision,
                    "output": current_output,
                    "feedback": feedback,
                }
            )

            # Check if approved
            if "APPROVED" in feedback.upper():
                if verbose:
                    print("[Supervisor] Work approved!")
                break

            if verbose:
                print(f"[Supervisor] Feedback: {feedback[:200]}...")

        return {
            "final_output": current_output,
            "revision_history": revision_history,
            "total_revisions": len(revision_history) - 1,
        }

    def run_router(
        self,
        router_agent: str,
        routes: Dict[str, tuple[str, str]],
        input_prompt: str,
        default_route: Optional[tuple[str, str]] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Run router pattern that classifies input and routes to specialists.

        Args:
            router_agent: Agent that classifies the input
            routes: Dictionary of category -> (agent_name, prompt_template)
            input_prompt: User input to classify and route
            default_route: Default (agent_name, prompt_template) if no match
            verbose: Whether to print progress

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

        classification = self.run_agent(router_agent, classify_prompt, verbose)
        classification = classification.strip().lower()

        if verbose:
            print(f"[Router] Classification: {classification}")

        # Find matching route
        route_taken = None
        for category, (agent_name, prompt_template) in routes.items():
            if category.lower() in classification:
                route_taken = category
                self._context.set("user_input", input_prompt)
                self._context.set("classification", classification)
                prompt = self._context.interpolate(prompt_template)
                response = self.run_agent(agent_name, prompt, verbose)
                break
        else:
            # No match, use default
            if default_route:
                route_taken = "default"
                agent_name, prompt_template = default_route
                self._context.set("user_input", input_prompt)
                prompt = self._context.interpolate(prompt_template)
                response = self.run_agent(agent_name, prompt, verbose)
            else:
                response = f"Unable to route input. Classification: {classification}"

        return {
            "classification": classification,
            "route_taken": route_taken,
            "response": response,
        }

    def run_ensemble(
        self,
        agents: List[str],
        prompt: str,
        weights: Optional[Dict[str, float]] = None,
        combiner_agent: Optional[str] = None,
        combination_strategy: str = "synthesize",
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Run ensemble pattern combining multiple agent responses.

        Args:
            agents: List of agent names to run
            prompt: Prompt for all agents
            weights: Optional weights for each agent (for ranking)
            combiner_agent: Optional agent to combine responses
            combination_strategy: How to combine ('synthesize', 'best', 'merge')
            verbose: Whether to print progress

        Returns:
            Dictionary with 'individual_responses', 'combined_response', 'weights'
        """
        if verbose:
            print(f"\n[Ensemble] Running {len(agents)} agents...")

        # Get all responses in parallel
        agent_prompts = {agent: prompt for agent in agents}
        individual_responses = self.run_parallel(agent_prompts, verbose=verbose)

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

            result["combined_response"] = self.run_agent(
                combiner_agent, combine_prompt, verbose
            )
        else:
            # Just concatenate if no combiner
            result["combined_response"] = "\n\n---\n\n".join(
                f"[{agent}]: {resp}" for agent, resp in individual_responses.items()
            )

        return result

    def run_state_machine(
        self,
        states: Dict[str, Dict[str, Any]],
        initial_state: str,
        input_data: str,
        max_transitions: int = 10,
        verbose: bool = False,
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

        Returns:
            Dictionary with 'final_state', 'state_history', 'final_output'
        """
        if verbose:
            print(f"\n[StateMachine] Starting in state: {initial_state}")
            print(f"[StateMachine] Available states: {list(states.keys())}")

        current_state = initial_state
        state_history: List[Dict[str, Any]] = []
        current_output = ""
        self._context.set("input_data", input_data)

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
            self._context.set("current_state", current_state)
            self._context.set("previous_output", current_output)
            prompt = self._context.interpolate(prompt_template)
            current_output = self.run_agent(agent_name, prompt, verbose)

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
                elif condition.lower() in output_lower:
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

    def run_event_driven(
        self,
        event_handlers: Dict[str, tuple[str, str]],
        events: List[Dict[str, Any]],
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Run event-driven pattern processing a stream of events.

        Args:
            event_handlers: Dict of event_type -> (agent_name, prompt_template)
            events: List of event dictionaries with 'type' and 'data' keys
            verbose: Whether to print progress

        Returns:
            Dictionary with 'processed_events', 'unhandled_events', 'outputs'
        """
        if verbose:
            print(f"\n[EventDriven] Processing {len(events)} events...")
            print(f"[EventDriven] Registered handlers: {list(event_handlers.keys())}")

        processed_events: List[Dict[str, Any]] = []
        unhandled_events: List[Dict[str, Any]] = []
        outputs: List[str] = []

        for event in events:
            event_type = event.get("type", "unknown")
            event_data = event.get("data", {})

            if verbose:
                print(f"\n[EventDriven] Event: {event_type}")

            if event_type in event_handlers:
                agent_name, prompt_template = event_handlers[event_type]

                # Set event data in context
                self._context.set("event_type", event_type)
                self._context.set("event_data", str(event_data))
                for key, value in event_data.items():
                    self._context.set(f"event_{key}", str(value))

                prompt = self._context.interpolate(prompt_template)
                output = self.run_agent(agent_name, prompt, verbose)

                processed_events.append(
                    {
                        "event": event,
                        "handler": agent_name,
                        "output": output,
                    }
                )
                outputs.append(output)
            else:
                unhandled_events.append(event)
                if verbose:
                    print(f"[EventDriven] No handler for event type: {event_type}")

        return {
            "processed_events": processed_events,
            "unhandled_events": unhandled_events,
            "outputs": outputs,
            "total_processed": len(processed_events),
            "total_unhandled": len(unhandled_events),
        }

    def run_critic(
        self,
        creator_agent: str,
        critic_agent: str,
        task: str,
        max_rounds: int = 3,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Run creator-critic pattern for iterative improvement.

        Unlike supervisor, both agents are equal peers - the critic provides
        constructive criticism and the creator improves.

        Args:
            creator_agent: Agent that creates content
            critic_agent: Agent that critiques and suggests improvements
            task: Task to complete
            max_rounds: Maximum critique rounds
            verbose: Whether to print progress

        Returns:
            Dictionary with 'final_output', 'critique_rounds', 'improvement_history'
        """
        if verbose:
            print(f"\n[Critic] Task: {task[:100]}...")

        improvement_history: List[Dict[str, str]] = []
        current_work = ""

        for round_num in range(1, max_rounds + 1):
            if verbose:
                print(f"\n[Critic] === Round {round_num}/{max_rounds} ===")

            # Creator creates or improves
            if round_num == 1:
                creator_prompt = f"Complete this task:\n\n{task}"
            else:
                creator_prompt = (
                    f"Task: {task}\n\n"
                    f"Your current work:\n{current_work}\n\n"
                    f"Critic's feedback:\n{critique}\n\n"
                    f"Improve your work based on this feedback."
                )

            current_work = self.run_agent(creator_agent, creator_prompt, verbose)

            if verbose:
                print(f"\n[Creator]: {current_work[:300]}...")

            # Critic provides feedback
            critic_prompt = (
                f"Task: {task}\n\n"
                f"Work to critique:\n{current_work}\n\n"
                f"Provide constructive criticism. What could be improved? "
                f"Be specific and actionable. If the work is excellent and "
                f"needs no improvement, say 'EXCELLENT - no changes needed'."
            )

            critique = self.run_agent(critic_agent, critic_prompt, verbose)

            if verbose:
                print(f"\n[Critic]: {critique[:300]}...")

            improvement_history.append(
                {
                    "round": round_num,
                    "work": current_work,
                    "critique": critique,
                }
            )

            # Check if critic is satisfied
            if (
                "EXCELLENT" in critique.upper()
                or "NO CHANGES NEEDED" in critique.upper()
            ):
                if verbose:
                    print("[Critic] Critic satisfied, ending rounds.")
                break

        return {
            "final_output": current_work,
            "critique_rounds": len(improvement_history),
            "improvement_history": improvement_history,
        }

    def run_planner_executor(
        self,
        planner_agent: str,
        executor_agent: str,
        goal: str,
        max_steps: int = 5,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Run planner-executor pattern for goal decomposition.

        Planner breaks down the goal into steps, executor handles each step.

        Args:
            planner_agent: Agent that creates and updates the plan
            executor_agent: Agent that executes each step
            goal: High-level goal to achieve
            max_steps: Maximum execution steps
            verbose: Whether to print progress

        Returns:
            Dictionary with 'plan', 'execution_log', 'final_result'
        """
        if verbose:
            print(f"\n[PlannerExecutor] Goal: {goal}")

        # Planner creates initial plan
        plan_prompt = (
            f"Create a step-by-step plan to achieve this goal: {goal}\n\n"
            f"List each step numbered (1, 2, 3, etc.). "
            f"Keep steps atomic and actionable. Maximum {max_steps} steps."
        )

        plan = self.run_agent(planner_agent, plan_prompt, verbose)
        self._context.set("plan", plan)

        if verbose:
            print(f"\n[Planner] Plan:\n{plan}")

        execution_log: List[Dict[str, str]] = []
        completed_steps = ""

        for step_num in range(1, max_steps + 1):
            # Ask planner what's next
            next_step_prompt = (
                f"Goal: {goal}\n\n"
                f"Original plan:\n{plan}\n\n"
                f"Completed steps:\n{completed_steps or 'None yet'}\n\n"
                f"What is the next step to execute? "
                f"If all steps are complete, say 'GOAL ACHIEVED'."
            )

            next_step = self.run_agent(planner_agent, next_step_prompt, verbose)

            if "GOAL ACHIEVED" in next_step.upper():
                if verbose:
                    print("[PlannerExecutor] Goal achieved!")
                break

            if verbose:
                print(f"\n[Step {step_num}] {next_step[:200]}...")

            # Executor handles the step
            execute_prompt = (
                f"Execute this step: {next_step}\n\n"
                f"Context - Goal: {goal}\n"
                f"Previous work:\n{completed_steps or 'Starting fresh'}"
            )

            result = self.run_agent(executor_agent, execute_prompt, verbose)

            execution_log.append(
                {
                    "step_number": step_num,
                    "step_description": next_step,
                    "result": result,
                }
            )

            completed_steps += f"\nStep {step_num}: {next_step}\nResult: {result}\n"

        return {
            "plan": plan,
            "execution_log": execution_log,
            "final_result": completed_steps,
            "steps_executed": len(execution_log),
        }

    def clear_context(self) -> None:
        """Clear the shared context."""
        self._context.clear()

    def __repr__(self) -> str:
        """String representation."""
        return f"Orchestrator(agents={len(self._agents)})"
