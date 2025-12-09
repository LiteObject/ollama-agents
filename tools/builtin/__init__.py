"""
Builtin tools package initialization.

This module registers the builtin Ollama tools (web_search, web_fetch)
with the tool registry.
"""

from core.tools import tool

# Note: web_search and web_fetch are builtin Ollama tools
# They are handled specially in the Agent class
# This file is here for future builtin tools
