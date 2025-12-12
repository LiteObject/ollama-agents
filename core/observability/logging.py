"""
Structured logging for multi-agent framework.

This module provides structured logging with context injection,
making it easy to trace agent interactions and debug issues.
"""

# pylint: disable=too-many-arguments, too-many-positional-arguments

import json
import logging
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union


class LogLevel(Enum):
    """Log levels for the agent framework."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class StructuredFormatter(logging.Formatter):
    """Custom formatter that outputs structured JSON logs."""

    def __init__(self, include_timestamp: bool = True, pretty: bool = False) -> None:
        """Initialize the formatter.

        Args:
            include_timestamp: Whether to include timestamp in output
            pretty: Whether to pretty-print JSON output
        """
        super().__init__()
        self._include_timestamp = include_timestamp
        self._pretty = pretty

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as structured JSON.

        Args:
            record: The log record to format

        Returns:
            JSON-formatted log string
        """
        log_data: Dict[str, Any] = {
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }

        if self._include_timestamp:
            log_data["timestamp"] = datetime.now().isoformat()

        # Add extra context if present
        if getattr(record, "agent_name", None):
            log_data["agent"] = getattr(record, "agent_name")
        if getattr(record, "session_id", None):
            log_data["session_id"] = getattr(record, "session_id")
        if getattr(record, "step", None) is not None:
            log_data["step"] = getattr(record, "step")
        if getattr(record, "pattern", None):
            log_data["pattern"] = getattr(record, "pattern")
        if getattr(record, "duration_ms", None) is not None:
            log_data["duration_ms"] = getattr(record, "duration_ms")
        if getattr(record, "extra_data", None):
            log_data["data"] = getattr(record, "extra_data")

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        indent = 2 if self._pretty else None
        return json.dumps(log_data, indent=indent, default=str)


class HumanReadableFormatter(logging.Formatter):
    """Formatter that outputs human-readable logs with context."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",
    }

    def __init__(self, use_colors: bool = True) -> None:
        """Initialize the formatter.

        Args:
            use_colors: Whether to use ANSI colors in output
        """
        super().__init__()
        self._use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as human-readable text.

        Args:
            record: The log record to format

        Returns:
            Formatted log string
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level = record.levelname

        # Build context string
        context_parts: list[str] = []
        if getattr(record, "agent_name", None):
            context_parts.append(f"agent={getattr(record, 'agent_name')}")
        if getattr(record, "session_id", None):
            session_id = getattr(record, "session_id")
            context_parts.append(f"session={session_id[:8]}")
        if getattr(record, "step", None) is not None:
            context_parts.append(f"step={getattr(record, 'step')}")
        if getattr(record, "pattern", None):
            context_parts.append(f"pattern={getattr(record, 'pattern')}")
        if getattr(record, "duration_ms", None) is not None:
            context_parts.append(f"duration={getattr(record, 'duration_ms')}ms")

        context = f" [{', '.join(context_parts)}]" if context_parts else ""

        # Apply colors
        if self._use_colors:
            color = self.COLORS.get(level, "")
            reset = self.COLORS["RESET"]
            level_str = f"{color}{level:8}{reset}"
        else:
            level_str = f"{level:8}"

        message = record.getMessage()
        base = f"{timestamp} | {level_str} |{context} {message}"

        # Add exception if present
        if record.exc_info:
            base += f"\n{self.formatException(record.exc_info)}"

        return base


class AgentLogger:
    """Logger with context injection for agent operations.

    Usage:
        logger = AgentLogger("my_agent", session_id="abc123")
        logger.info("Processing request", extra={"tokens": 150})
        logger.error("Failed to process", exc_info=True)
    """

    def __init__(
        self,
        name: str,
        session_id: Optional[str] = None,
        agent_name: Optional[str] = None,
    ) -> None:
        """Initialize the agent logger.

        Args:
            name: Logger name (typically module name)
            session_id: Session identifier for correlation
            agent_name: Agent name for context
        """
        self._logger = logging.getLogger(f"agent.{name}")
        self._session_id = session_id
        self._agent_name = agent_name

    def _log(
        self,
        level: int,
        message: str,
        agent_name: Optional[str] = None,
        step: Optional[int] = None,
        pattern: Optional[str] = None,
        duration_ms: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
        exc_info: bool = False,
    ) -> None:
        """Internal logging method with context injection.

        Args:
            level: Log level
            message: Log message
            agent_name: Override agent name
            step: Current step number
            pattern: Orchestration pattern name
            duration_ms: Operation duration in milliseconds
            extra: Additional data to include
            exc_info: Whether to include exception info
        """
        # Create extra dict for context
        log_extra: Dict[str, Any] = {}

        if agent_name or self._agent_name:
            log_extra["agent_name"] = agent_name or self._agent_name
        if self._session_id:
            log_extra["session_id"] = self._session_id
        if step is not None:
            log_extra["step"] = step
        if pattern:
            log_extra["pattern"] = pattern
        if duration_ms is not None:
            log_extra["duration_ms"] = round(duration_ms, 2)
        if extra:
            log_extra["extra_data"] = extra

        self._logger.log(level, message, extra=log_extra, exc_info=exc_info)

    def debug(
        self,
        message: str,
        **kwargs: Any,
    ) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, message, **kwargs)

    def info(
        self,
        message: str,
        **kwargs: Any,
    ) -> None:
        """Log info message."""
        self._log(logging.INFO, message, **kwargs)

    def warning(
        self,
        message: str,
        **kwargs: Any,
    ) -> None:
        """Log warning message."""
        self._log(logging.WARNING, message, **kwargs)

    def error(
        self,
        message: str,
        **kwargs: Any,
    ) -> None:
        """Log error message."""
        self._log(logging.ERROR, message, **kwargs)

    def critical(
        self,
        message: str,
        **kwargs: Any,
    ) -> None:
        """Log critical message."""
        self._log(logging.CRITICAL, message, **kwargs)

    def with_context(
        self,
        agent_name: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> "AgentLogger":
        """Create a new logger with additional context.

        Args:
            agent_name: Agent name to add
            session_id: Session ID to add

        Returns:
            New AgentLogger with merged context
        """
        return AgentLogger(
            name=self._logger.name.replace("agent.", ""),
            session_id=session_id or self._session_id,
            agent_name=agent_name or self._agent_name,
        )


def configure_logging(
    level: Union[LogLevel, str] = LogLevel.INFO,
    log_file: Optional[Union[str, Path]] = None,
    json_format: bool = False,
    use_colors: bool = True,
    pretty_json: bool = False,
) -> None:
    """Configure logging for the agent framework.

    Args:
        level: Minimum log level
        log_file: Optional file path for log output
        json_format: Whether to use JSON format (vs human-readable)
        use_colors: Whether to use colors in console output
        pretty_json: Whether to pretty-print JSON logs
    """
    # Convert string level to LogLevel
    if isinstance(level, str):
        level = LogLevel[level.upper()]

    # Get root agent logger
    root_logger = logging.getLogger("agent")
    root_logger.setLevel(level.value)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create formatter
    if json_format:
        formatter: logging.Formatter = StructuredFormatter(pretty=pretty_json)
    else:
        formatter = HumanReadableFormatter(use_colors=use_colors)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (always JSON for easier parsing)
    if log_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(StructuredFormatter(pretty=False))
        root_logger.addHandler(file_handler)


def get_logger(
    name: str,
    session_id: Optional[str] = None,
    agent_name: Optional[str] = None,
) -> AgentLogger:
    """Get an agent logger instance.

    Args:
        name: Logger name
        session_id: Optional session ID
        agent_name: Optional agent name

    Returns:
        Configured AgentLogger instance
    """
    return AgentLogger(name=name, session_id=session_id, agent_name=agent_name)


# Default logger for the framework
default_logger = get_logger("framework")
