"""
Tool registration and plugin system.

This module provides a decorator-based system for registering tools that
agents can use, and a registry for discovering and managing tools.
"""

import importlib
import importlib.util
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ToolDefinition:
    """Definition of a tool that agents can use.

    Attributes:
        name: Unique identifier for the tool
        description: Human-readable description of what the tool does
        function: The actual callable that implements the tool
        parameters: Dictionary describing the tool's parameters
    """

    name: str
    description: str
    function: Callable[..., Any]
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool function."""
        return self.function(*args, **kwargs)


# Global registry of tools
_TOOL_REGISTRY: Dict[str, ToolDefinition] = {}


def tool(
    name: Optional[str] = None,
    description: str = "",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to register a function as a tool.

    Usage:
        @tool(name="search_web", description="Search the web for information")
        def search_web(query: str) -> str:
            ...

    Args:
        name: Tool name (defaults to function name)
        description: Description of what the tool does

    Returns:
        Decorator function
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or ""

        # Extract parameter information from function signature
        sig = inspect.signature(func)
        parameters: Dict[str, Any] = {}

        for param_name, param in sig.parameters.items():
            param_info: Dict[str, Any] = {"name": param_name}

            # Get type annotation if available
            if param.annotation != inspect.Parameter.empty:
                param_info["type"] = param.annotation.__name__

            # Get default value if available
            if param.default != inspect.Parameter.empty:
                param_info["default"] = param.default

            parameters[param_name] = param_info

        # Create and register the tool definition
        tool_def = ToolDefinition(
            name=tool_name,
            description=tool_description,
            function=func,
            parameters=parameters,
        )

        _TOOL_REGISTRY[tool_name] = tool_def

        # Return the original function (not wrapped)
        return func

    return decorator


class ToolRegistry:
    """Registry for managing and discovering tools.

    This class provides methods to register, discover, and retrieve tools
    from both the global registry and custom tool directories.
    """

    def __init__(self) -> None:
        """Initialize the tool registry with builtin tools."""
        self._tools: Dict[str, ToolDefinition] = {}
        self._load_global_registry()

    def _load_global_registry(self) -> None:
        """Load tools from the global registry."""
        self._tools.update(_TOOL_REGISTRY)

    def register(self, tool_def: ToolDefinition) -> None:
        """Register a tool definition.

        Args:
            tool_def: The tool definition to register
        """
        self._tools[tool_def.name] = tool_def

    def register_function(
        self,
        func: Callable[..., Any],
        name: Optional[str] = None,
        description: str = "",
    ) -> None:
        """Register a function as a tool.

        Args:
            func: The function to register
            name: Tool name (defaults to function name)
            description: Description of what the tool does
        """
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or ""

        tool_def = ToolDefinition(
            name=tool_name,
            description=tool_description,
            function=func,
        )

        self._tools[tool_name] = tool_def

    def get(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool by name.

        Args:
            name: The tool name

        Returns:
            ToolDefinition if found, None otherwise
        """
        return self._tools.get(name)

    def get_function(self, name: str) -> Optional[Callable[..., Any]]:
        """Get a tool's function by name.

        Args:
            name: The tool name

        Returns:
            The tool function if found, None otherwise
        """
        tool_def = self._tools.get(name)
        return tool_def.function if tool_def else None

    def list_tools(self) -> List[str]:
        """Get a list of all registered tool names.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def get_all(self) -> Dict[str, ToolDefinition]:
        """Get all registered tools.

        Returns:
            Dictionary of tool name to ToolDefinition
        """
        return dict(self._tools)

    def discover_from_directory(self, directory: Path | str) -> int:
        """Discover and load tools from Python files in a directory.

        This method imports all .py files in the directory and registers
        any functions decorated with @tool.

        Args:
            directory: Path to the directory containing tool modules

        Returns:
            Number of tools discovered
        """
        directory = Path(directory)
        if not directory.exists():
            return 0

        initial_count = len(self._tools)

        for py_file in directory.glob("*.py"):
            if py_file.name.startswith("_"):
                continue

            try:
                spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    # Tools are auto-registered via the @tool decorator
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f"Warning: Failed to load tool module {py_file}: {e}")

        # Reload global registry after discovery
        self._load_global_registry()

        return len(self._tools) - initial_count

    def get_tools_for_ollama(self, tool_names: List[str]) -> List[Callable[..., Any]]:
        """Get tool functions in the format expected by Ollama.

        Args:
            tool_names: List of tool names to retrieve

        Returns:
            List of callable functions
        """
        tools: List[Callable[..., Any]] = []

        for name in tool_names:
            tool_def = self._tools.get(name)
            if tool_def:
                tools.append(tool_def.function)

        return tools


# Create a default registry instance
default_registry = ToolRegistry()
