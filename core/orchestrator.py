"""
Orchestrator for coordinating multiple agents.

This module provides an Orchestrator class that manages the execution
of multiple agents in various patterns: sequential, parallel, conditional, etc.
"""

# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-public-methods, too-many-instance-attributes

import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, cast

from core.agent import Agent
from core.config import AgentConfig
from core.context import SharedContext
from core.observability.hooks import (
    EventHookRegistry,
    default_hook_registry,
)
from core.observability.logging import AgentLogger, get_logger
from core.patterns.conditional import ConditionalHandler
from core.patterns.loop import LoopHandler
from core.patterns.parallel import ParallelHandler
from core.patterns.sequential import SequentialHandler
from core.patterns.hierarchical import HierarchicalHandler
from core.patterns.map_reduce import MapReduceHandler
from core.patterns.pipeline import PipelineHandler
from core.patterns.voting import VotingHandler
from core.patterns.debate import DebateHandler
from core.patterns.retry import RetryHandler
from core.patterns.chain_of_thought import ChainOfThoughtHandler
from core.patterns.supervisor import SupervisorHandler
from core.patterns.state_machine import StateMachineHandler
from core.patterns.router import RouterHandler
from core.patterns.ensemble import EnsembleHandler
from core.patterns.event_driven import EventDrivenHandler
from core.tools import ToolRegistry, default_registry
from core.workflow import ExecutionPattern, Workflow


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

        # Initialize pattern handlers
        self._handlers = {
            ExecutionPattern.SEQUENTIAL: SequentialHandler(self),
            ExecutionPattern.PARALLEL: ParallelHandler(self),
            ExecutionPattern.CONDITIONAL: ConditionalHandler(self),
            ExecutionPattern.LOOP: LoopHandler(self),
            ExecutionPattern.HIERARCHICAL: HierarchicalHandler(self),
            ExecutionPattern.MAP_REDUCE: MapReduceHandler(self),
            ExecutionPattern.PIPELINE: PipelineHandler(self),
            ExecutionPattern.VOTING: VotingHandler(self),
            ExecutionPattern.DEBATE: DebateHandler(self),
            ExecutionPattern.SUPERVISOR: SupervisorHandler(self),
            ExecutionPattern.STATE_MACHINE: StateMachineHandler(self),
            ExecutionPattern.ROUTER: RouterHandler(self),
            ExecutionPattern.ENSEMBLE: EnsembleHandler(self),
            ExecutionPattern.EVENT_DRIVEN: EventDrivenHandler(self),
        }
        # Note: Retry and ChainOfThought are not in ExecutionPattern enum yet or are helper methods
        # But I can add them to handlers if I want to use them via run_workflow or similar
        # For now, I'll just instantiate them when needed or add them to a separate dict
        # if they are not in Enum

        self._retry_handler = RetryHandler(self)
        self._cot_handler = ChainOfThoughtHandler(self)

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
        handler = self._handlers[ExecutionPattern.SEQUENTIAL]
        return handler.execute(steps=steps, initial_vars=initial_vars, verbose=verbose)

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
        handler = self._handlers[ExecutionPattern.PARALLEL]
        return handler.execute(
            agent_prompts=agent_prompts, verbose=verbose, max_workers=max_workers
        )

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
        handler = self._handlers[ExecutionPattern.CONDITIONAL]
        return handler.execute(
            condition_agent=condition_agent,
            condition_prompt=condition_prompt,
            branches=branches,
            default_branch=default_branch,
            verbose=verbose,
        )

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
        handler = self._handlers[ExecutionPattern.LOOP]
        return handler.execute(
            agent_name=agent_name,
            prompt_template=prompt_template,
            condition_fn=condition_fn,
            max_iterations=max_iterations,
            verbose=verbose,
        )

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

        if workflow.pattern == ExecutionPattern.PARALLEL:
            prompts = {step.agent: step.prompt for step in workflow.steps}
            results = self.run_parallel(prompts, verbose=verbose)
            # Return all results combined
            return "\n\n".join(
                f"[{name}]: {result}" for name, result in results.items()
            )

        if workflow.pattern == ExecutionPattern.PIPELINE:
            stages_list: List[tuple[str, str, None]] = [
                (step.agent, step.prompt, None) for step in workflow.steps
            ]
            initial_input = self._context.get("pipeline_input", "")
            # Cast to satisfy type checker (None is valid for Optional[Callable])
            return self.run_pipeline(
                cast(
                    List[tuple[str, str, Optional[Callable[[str], bool]]]], stages_list
                ),
                str(initial_input),
                verbose=verbose,
            )

        if workflow.pattern == ExecutionPattern.VOTING:
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
        handler = self._handlers[ExecutionPattern.HIERARCHICAL]
        return handler.execute(
            supervisor_agent=supervisor_agent,
            worker_agents=worker_agents,
            task_prompt=task_prompt,
            aggregation_prompt=aggregation_prompt,
            verbose=verbose,
        )

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
        handler = self._handlers[ExecutionPattern.MAP_REDUCE]
        return handler.execute(
            mapper_agent=mapper_agent,
            reducer_agent=reducer_agent,
            items=items,
            map_prompt_template=map_prompt_template,
            reduce_prompt_template=reduce_prompt_template,
            max_workers=max_workers,
            verbose=verbose,
        )

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
        handler = self._handlers[ExecutionPattern.PIPELINE]
        return handler.execute(
            stages=stages,
            initial_input=initial_input,
            verbose=verbose,
        )

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
        handler = self._handlers[ExecutionPattern.VOTING]
        return handler.execute(
            voter_agents=voter_agents,
            prompt=prompt,
            aggregator_agent=aggregator_agent,
            aggregation_prompt=aggregation_prompt,
            verbose=verbose,
        )

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
        handler = self._handlers[ExecutionPattern.DEBATE]
        return handler.execute(
            debater_agents=debater_agents,
            topic=topic,
            moderator_agent=moderator_agent,
            rounds=rounds,
            verbose=verbose,
        )

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
        return self._retry_handler.execute(
            primary_agent=primary_agent,
            prompt=prompt,
            validator_fn=validator_fn,
            fallback_agents=fallback_agents,
            max_retries=max_retries,
            verbose=verbose,
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
        return self._cot_handler.execute(
            agent_name=agent_name,
            problem=problem,
            thinking_steps=thinking_steps,
            verbose=verbose,
        )

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
        handler = self._handlers[ExecutionPattern.SUPERVISOR]
        return handler.execute(
            supervisor_agent=supervisor_agent,
            worker_agent=worker_agent,
            task=task,
            max_revisions=max_revisions,
            quality_threshold=quality_threshold,
            min_quality=min_quality,
            verbose=verbose,
        )

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
        handler = self._handlers[ExecutionPattern.ROUTER]
        return handler.execute(
            router_agent=router_agent,
            routes=routes,
            input_prompt=input_prompt,
            default_route=default_route,
            verbose=verbose,
        )

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
        handler = self._handlers[ExecutionPattern.ENSEMBLE]
        return handler.execute(
            agents=agents,
            prompt=prompt,
            weights=weights,
            combiner_agent=combiner_agent,
            combination_strategy=combination_strategy,
            verbose=verbose,
        )

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
        handler = self._handlers[ExecutionPattern.STATE_MACHINE]
        return handler.execute(
            states=states,
            initial_state=initial_state,
            input_data=input_data,
            max_transitions=max_transitions,
            verbose=verbose,
        )

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
        handler = self._handlers[ExecutionPattern.EVENT_DRIVEN]
        return handler.execute(
            event_handlers=event_handlers,
            events=events,
            verbose=verbose,
        )

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
        critique = ""  # Initialize critique for first round

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
