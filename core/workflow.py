"""
Workflow definitions for the orchestration framework.

This module contains the data structures and enums used to define
workflows and execution patterns.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

import yaml


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
