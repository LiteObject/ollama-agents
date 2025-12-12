#!/usr/bin/env python3
"""
Multi-Agent Orchestration Demo.

This script demonstrates how to use the multi-agent framework to:
1. Create agents from YAML configurations
2. Run agents individually
3. Orchestrate agents in sequence
4. Use shared context between agents

Usage:
    python main.py
"""
# pylint: disable=wrong-import-position

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    print("Error: python-dotenv is required. Install with: pip install python-dotenv")
    sys.exit(1)

# Load environment variables
load_dotenv(override=True)

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from core import Agent, AgentConfig, Orchestrator


def demo_single_agent() -> None:
    """Demonstrate using a single agent."""
    print("\n" + "=" * 60)
    print("DEMO 1: Single Agent")
    print("=" * 60)

    # Create agent from YAML config
    config_path = PROJECT_ROOT / "config" / "agents" / "researcher.yaml"

    if config_path.exists():
        config = AgentConfig.from_yaml(config_path)
        print(f"Loaded config: {config.name}")
    else:
        # Fallback to programmatic config
        config = AgentConfig(
            name="researcher",
            model=os.getenv("OLLAMA_MODEL", "gpt-oss:20b"),
            system_prompt="You are a helpful research assistant.",
            tools=["web_search"],
            max_iterations=3,
        )

    agent = Agent(config)
    print(f"Created agent: {agent}")

    # Run the agent
    prompt = input("\nEnter your question (or press Enter for default): ").strip()
    if not prompt:
        prompt = "What are the latest developments in AI agents?"

    print(f"\nRunning agent with prompt: {prompt[:50]}...")
    response = agent.run(prompt, verbose=True)

    print(f"\n{'─' * 40}")
    print("FINAL RESPONSE:")
    print(f"{'─' * 40}")
    print(response)


def demo_orchestrator() -> None:
    """Demonstrate using the orchestrator with multiple agents."""
    print("\n" + "=" * 60)
    print("DEMO 2: Multi-Agent Orchestration")
    print("=" * 60)

    # Create orchestrator
    orchestrator = Orchestrator()

    # Load agents from YAML configs
    config_dir = PROJECT_ROOT / "config" / "agents"
    model = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")

    if config_dir.exists():
        for yaml_file in config_dir.glob("*.yaml"):
            try:
                agent = orchestrator.add_agent_from_yaml(yaml_file)
                print(f"Loaded agent: {agent.name}")
            except (FileNotFoundError, ValueError, KeyError) as e:
                print(f"Failed to load {yaml_file}: {e}")
    else:
        # Fallback to programmatic configs
        orchestrator.add_agent(
            AgentConfig(
                name="researcher",
                model=model,
                system_prompt="You research topics thoroughly.",
                tools=["web_search"],
                max_iterations=3,
            )
        )
        orchestrator.add_agent(
            AgentConfig(
                name="analyst",
                model=model,
                system_prompt="You analyze information and provide insights.",
                max_iterations=2,
            )
        )
        orchestrator.add_agent(
            AgentConfig(
                name="writer",
                model=model,
                system_prompt="You write clear, engaging content.",
                max_iterations=2,
            )
        )

    print(f"\nOrchestrator ready with agents: {orchestrator.list_agents()}")

    # Get topic from user
    topic = input("\nEnter a topic to research (or press Enter for default): ").strip()
    if not topic:
        topic = "the future of AI agents in enterprise"

    # Set initial context
    orchestrator.context.set("topic", topic)

    print(f"\nRunning sequential workflow on topic: {topic}")
    print("-" * 40)

    # Run sequential workflow: Research → Analyze → Write
    result = orchestrator.run_sequential(
        steps=[
            ("researcher", "Research the following topic thoroughly: {topic}"),
            (
                "analyst",
                "Analyze this research and identify key insights:\n\n{researcher}",
            ),
            (
                "writer",
                "Write a brief executive summary based on this analysis:\n\n{analyst}",
            ),
        ],
        verbose=True,
    )

    print(f"\n{'─' * 40}")
    print("FINAL OUTPUT:")
    print(f"{'─' * 40}")
    print(result)


def demo_parallel() -> None:
    """Demonstrate running agents in parallel."""
    print("\n" + "=" * 60)
    print("DEMO 3: Parallel Agent Execution")
    print("=" * 60)

    orchestrator = Orchestrator()
    model = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")

    # Create specialized research agents
    orchestrator.add_agent(
        AgentConfig(
            name="tech_researcher",
            model=model,
            system_prompt="You research technology topics.",
            tools=["web_search"],
            max_iterations=2,
        )
    )
    orchestrator.add_agent(
        AgentConfig(
            name="business_researcher",
            model=model,
            system_prompt="You research business and market topics.",
            tools=["web_search"],
            max_iterations=2,
        )
    )
    orchestrator.add_agent(
        AgentConfig(
            name="summarizer",
            model=model,
            system_prompt="You combine and summarize information.",
            max_iterations=2,
        )
    )

    topic = input("\nEnter a topic (or press Enter for default): ").strip()
    if not topic:
        topic = "AI agents in customer service"

    print(f"\nRunning parallel research on: {topic}")

    # Run two researchers in parallel
    results = orchestrator.run_parallel(
        agent_prompts={
            "tech_researcher": f"Research the technology behind: {topic}",
            "business_researcher": f"Research the business impact of: {topic}",
        },
        verbose=True,
    )

    # Combine results
    combined = "\n\n".join(f"**{name}**:\n{result}" for name, result in results.items())
    orchestrator.context.set("research", combined)

    # Summarize
    summary = orchestrator.run_agent(
        "summarizer",
        "Combine and summarize these research findings:\n\n{research}",
        verbose=True,
    )

    print(f"\n{'─' * 40}")
    print("COMBINED SUMMARY:")
    print(f"{'─' * 40}")
    print(summary)


def main() -> int:
    """Main entry point."""
    print("=" * 60)
    print("Multi-Agent Framework Demo")
    print("=" * 60)

    print("\nAvailable demos:")
    print("1. Single Agent - Use one agent with web search")
    print("2. Orchestrator - Sequential multi-agent workflow")
    print("3. Parallel - Run agents in parallel")
    print("q. Quit")

    choice = input("\nSelect demo (1/2/3/q): ").strip().lower()

    try:
        if choice == "1":
            demo_single_agent()
        elif choice == "2":
            demo_orchestrator()
        elif choice == "3":
            demo_parallel()
        elif choice in ("q", "quit", "exit"):
            print("Goodbye!")
            return 0
        else:
            print("Invalid choice. Please select 1, 2, 3, or q.")
            return 1
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        return 0
    except (ValueError, RuntimeError, IOError) as e:
        print(f"\nError: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
