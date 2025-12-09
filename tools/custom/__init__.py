"""
Custom tools package initialization.

Place your custom tool modules here. Tools decorated with @tool
will be automatically discovered and registered.

Example:
    # my_tool.py
    from core.tools import tool

    @tool(name="my_custom_tool", description="Does something useful")
    def my_custom_tool(param1: str, param2: int = 10) -> str:
        return f"Result: {param1} with {param2}"
"""
