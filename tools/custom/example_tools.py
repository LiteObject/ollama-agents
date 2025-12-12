"""
Example custom tool demonstrating the @tool decorator.

This module shows how to create custom tools that agents can use.
"""

import ast
import operator as op
import random
from collections.abc import Callable
from datetime import datetime

from core.tools import tool


BinaryOp = Callable[[float, float], float]
UnaryOp = Callable[[float], float]


_BIN_OPS: dict[type[ast.operator], BinaryOp] = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
}

_UNARY_OPS: dict[type[ast.unaryop], UnaryOp] = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}


def _safe_eval_arithmetic(expression: str) -> float:
    parsed = ast.parse(expression, mode="eval")

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _BIN_OPS:
                raise ValueError("Unsupported operator")
            return _BIN_OPS[op_type](_eval(node.left), _eval(node.right))

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _UNARY_OPS:
                raise ValueError("Unsupported unary operator")
            return _UNARY_OPS[op_type](_eval(node.operand))

        raise ValueError("Unsupported expression")

    return _eval(parsed)


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
        result = _safe_eval_arithmetic(expression)
        return str(result)
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError) as exc:
        return f"Error: {exc}"


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
    preview = ", ".join(item_list[:5])
    suffix = "..." if len(item_list) > 5 else ""
    return f"Found {len(item_list)} items: {preview}{suffix}"
