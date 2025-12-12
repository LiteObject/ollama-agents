"""
Example custom tool demonstrating the @tool decorator.

This module shows how to create custom tools that agents can use.
"""

import random
from datetime import datetime

# Import from parent to avoid circular imports
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.tools import tool


@tool(name="get_current_time", description="Get the current date and time")
def get_current_time() -> str:
    """Return the current date and time as a formatted string."""
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


@tool(name="calculate", description="Perform basic arithmetic calculations")
def calculate(expression: str) -> str:
    """Evaluate a simple arithmetic expression.

    Args:
        expression: A math expression like "2 + 2" or "10 * 5"

    Returns:
        The result of the calculation
    """
    try:
        # Only allow safe operations
        allowed = set("0123456789+-*/.(). ")
        if not all(c in allowed for c in expression):
            return "Error: Only basic arithmetic operations are allowed"

        result = eval(expression)  # Safe because we validated input
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool(name="generate_id", description="Generate a unique identifier")
def generate_id(prefix: str = "ID") -> str:
    """Generate a unique ID with an optional prefix.

    Args:
        prefix: Prefix for the ID (default: "ID")

    Returns:
        A unique identifier string
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_part = random.randint(1000, 9999)
    return f"{prefix}-{timestamp}-{random_part}"


@tool(name="word_count", description="Count words in text")
def word_count(text: str) -> str:
    """Count the number of words in the provided text.

    Args:
        text: The text to count words in

    Returns:
        Word count as a string
    """
    words = text.split()
    return f"Word count: {len(words)}"


@tool(name="summarize_list", description="Summarize a list of items")
def summarize_list(items: str, separator: str = ",") -> str:
    """Parse and summarize a list of items.

    Args:
        items: Items separated by the separator
        separator: The separator between items (default: comma)

    Returns:
        Summary of the list
    """
    item_list = [item.strip() for item in items.split(separator) if item.strip()]
    return f"Found {len(item_list)} items: {', '.join(item_list[:5])}{'...' if len(item_list) > 5 else ''}"
