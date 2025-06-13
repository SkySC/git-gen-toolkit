#!/usr/bin/env python3

import sys
from typing import Any, Dict, Optional, Union

from loguru import logger
from rich.console import Console

# * Initialize Rich console for colored output
console = Console()

# * Default log format
DEFAULT_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level.name:<7}</level> |"
    " <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> -"
    " <level>{message}</level>"
)

# * Map log levels to names and numeric values
LOG_LEVELS = {
    "trace": 5,
    "debug": 10,
    "info": 20,
    "success": 25,
    "warning": 30,
    "error": 40,
    "critical": 50,
}


def setup_logging(
    level: Union[str, int] = "info",
    config: Optional[Dict[str, Any]] = None,
    show_time: bool = True,
) -> None:
    """Configure the logger with the specified level and format.

    Args:
        level: The log level (debug, info, warning, error, critical)
        config: Configuration dictionary that may contain logging settings
        show_time: Whether to show timestamp in logs (useful to disable in tests)
    """
    # * Remove default handlers
    logger.remove()

    # * Get log level from config if provided
    if config and "output" in config and "log_level" in config["output"]:
        level = config["output"]["log_level"]

    # * Convert string level to numeric if needed
    if isinstance(level, str):
        level = level.lower()
        level = LOG_LEVELS.get(level, LOG_LEVELS["info"])

    # * Determine format based on whether we want to show time
    log_format = DEFAULT_LOG_FORMAT
    if not show_time:
        log_format = log_format.replace(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | ", ""
        )

    # * Add stdout handler with proper format and level
    logger.add(
        sys.stdout,
        format=log_format,
        level=level,
        colorize=True,
    )


def get_logger(name: str = "git_gen_toolkit"):
    """Get a logger instance with the specified name.

    Args:
        name: The logger name, usually the module name

    Returns:
        A logger instance
    """
    return logger.bind(name=name)


# * Helper functions for common logging patterns


def info(message: str, *args, **kwargs):
    """Log an info message."""
    logger.info(message, *args, **kwargs)


def debug(message: str, *args, **kwargs):
    """Log a debug message."""
    logger.debug(message, *args, **kwargs)


def success(message: str, *args, **kwargs):
    """Log a success message."""
    logger.success(message, *args, **kwargs)


def warning(message: str, *args, **kwargs):
    """Log a warning message."""
    logger.warning(message, *args, **kwargs)


def error(message: str, *args, **kwargs):
    """Log an error message."""
    logger.error(message, *args, **kwargs)


def critical(message: str, *args, **kwargs):
    """Log a critical message."""
    logger.critical(message, *args, **kwargs)


# * Rich console output helpers for special formatting cases


def print_diff(diff_content: str, title: str = "Staged Diff"):
    """Print a git diff with syntax highlighting."""
    console.print(f"\n[bold white]--- {title} ---[/bold white]")
    console.print(diff_content, highlight=True)
    console.print(f"[bold white]--- End {title} ---[/bold white]")
    console.print(f"Total diff lines: {len(diff_content.splitlines())}")


def print_commit_message(message: str):
    """Print a generated commit message with formatting."""
    console.print("\n[bold green]--- Generated Commit Message ---[/bold green]")
    console.print(message)
    console.print("[bold green]--- End Commit Message ---[/bold green]")


def print_reasoning(reasoning: str):
    """Print model reasoning with formatting."""
    console.print("\n[bold blue]--- Model Reasoning ---[/bold blue]")
    console.print(reasoning)
    console.print("[bold blue]--- End Model Reasoning ---[/bold blue]")


def print_section(title: str, content: str):
    """Print a generic section with a title."""
    console.print(f"\n[bold yellow]--- {title} ---[/bold yellow]")
    if content:
        console.print(content)
        console.print(f"[bold yellow]--- End {title} ---[/bold yellow]")


def print_error(message: str):
    """Print an error message prominently."""
    console.print(f"\n[bold red]ERROR:[/bold red] {message}")
