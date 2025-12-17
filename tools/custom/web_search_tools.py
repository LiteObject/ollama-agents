"""
Smart web search tools with automatic freshness detection.

This module provides enhanced web search wrappers that automatically detect
time-sensitive queries (current events, recent news, "who is the current...")
and modify search queries to prioritize recent results.
"""

from datetime import datetime
from typing import Any

import ollama


# Keywords that indicate user wants current/recent information
TIME_SENSITIVE_PATTERNS = [
    "current",
    "latest",
    "recent",
    "now",
    "today",
    "this week",
    "this month",
    "this year",
    "who is the",
    "who are the",
    "what is the current",
    "breaking",
    "news",
    "update",
    "new",
]


def _is_time_sensitive(query: str) -> bool:
    """Check if a query is likely asking about current/recent information.

    Args:
        query: The search query to analyze

    Returns:
        True if the query appears to be time-sensitive
    """
    query_lower = query.lower()
    return any(pattern in query_lower for pattern in TIME_SENSITIVE_PATTERNS)


def _get_date_suffix() -> str:
    """Get a date suffix for search queries to prioritize recent results.

    Returns:
        String like "December 2025" for the current month/year
    """
    now = datetime.now()
    return now.strftime("%B %Y")


def web_search_smart(query: str, max_results: int = 10) -> Any:
    """Search the web with automatic freshness detection.

    For time-sensitive queries (current events, news, "who is the current..."),
    this function automatically appends the current month/year to the query
    to help prioritize recent results.

    Args:
        query: The search query
        max_results: Maximum number of results to return (default: 10)

    Returns:
        Search results from Ollama's web search

    Examples:
        >>> web_search_smart("current president of the US")
        # Automatically searches: "current president of the US December 2025"

        >>> web_search_smart("python tutorial")
        # Searches as-is: "python tutorial" (not time-sensitive)
    """
    # Check if query is time-sensitive
    if _is_time_sensitive(query):
        date_suffix = _get_date_suffix()
        enhanced_query = f"{query} {date_suffix}"
    else:
        enhanced_query = query

    # Call Ollama's web search
    return ollama.web_search(query=enhanced_query, max_results=max_results)


def web_fetch_smart(url: str) -> Any:
    """Fetch content from a URL.

    This is a passthrough to ollama.web_fetch for consistency.

    Args:
        url: The URL to fetch content from

    Returns:
        Content from the URL
    """
    return ollama.web_fetch(url=url)
