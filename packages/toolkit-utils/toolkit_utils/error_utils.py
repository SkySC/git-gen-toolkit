"""Error handling utilities for git-gen-toolkit.

This module provides a centralized error handling system with custom exceptions
and helper functions for consistent error reporting across all toolkit components.
"""

import functools
import sys
import traceback
from typing import Any, Callable, Optional, Type, TypeVar, cast

from .logging_utils import error

# * Type variable for decorator return type preservation
F = TypeVar("F", bound=Callable[..., Any])


class ToolkitError(Exception):
    """Base exception class for all toolkit errors."""

    def __init__(self, message: str, cause: Optional[Exception] = None):
        self.message = message
        self.cause = cause
        super().__init__(message)

    def __str__(self) -> str:
        if self.cause:
            return (
                f"{self.message} (Caused by: {type(self.cause).__name__}:"
                f" {str(self.cause)})"
            )
        return self.message


class ConfigError(ToolkitError):
    """Raised when there is an issue with configuration."""

    pass


class GitError(ToolkitError):
    """Raised when a Git operation fails."""

    pass


class LLMProviderError(ToolkitError):
    """Base class for LLM provider errors."""

    pass


class LLMConnectionError(LLMProviderError):
    """Raised when a connection to an LLM provider fails."""

    pass


class LLMModelNotFoundError(LLMProviderError):
    """Raised when a specified model is not found."""

    pass


class LLMModelNotLoadedError(LLMProviderError):
    """Raised when a model is found but not loaded."""

    pass


class LLMResponseError(LLMProviderError):
    """Raised when there is an issue with an LLM response."""

    pass


class ChunkingError(ToolkitError):
    """Raised when there is an issue with the chunking process."""

    pass


class TemplateError(ToolkitError):
    """Raised when there is an issue with templates."""

    pass


def handle_error(
    e: Exception,
    error_type: Type[Exception] = Exception,
    message: Optional[str] = None,
    debug: bool = False,
    exit_code: Optional[int] = None,
) -> None:
    """Handle an exception with consistent logging and optional exit.

    Args:
        e: The exception to handle
        error_type: The expected type of exception for more specific error messages
        message: Custom error message prefix
        debug: Whether to print debug information
        exit_code: If provided, exit the program with this code
    """
    # * Ensure we start error output on a clean line
    # * This helps when spinners or other output might interfere
    sys.stdout.write("\n")
    sys.stdout.flush()

    # * Determine message based on exception type
    if isinstance(e, ToolkitError):
        err_message = str(e)
    elif message:
        err_message = f"{message}: {str(e)}"
    else:
        err_message = str(e)

    # * Log the error
    error(err_message)

    # * Print debug information if requested
    if debug:
        error("\nDebug information:")
        traceback.print_exc()

    # * Exit if requested
    if exit_code is not None:
        sys.exit(exit_code)


def error_handler(
    error_type: Optional[Type[Exception]] = None,
    message: Optional[str] = None,
    reraise: bool = False,
    default_return: Any = None,
    debug_env_var: str = "DEBUG",
) -> Callable[[F], F]:
    """Decorator for handling errors in functions.

    Args:
        error_type: The expected type of exception
        message: Custom error message prefix
        reraise: Whether to re-raise the exception after handling
        default_return: Value to return on error if not reraising
        debug_env_var: Environment variable to check for debug mode

    Returns:
        Decorated function
    """
    error_type = error_type or Exception

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import os

            debug = os.environ.get(debug_env_var, "").lower() in ("true", "1", "yes")
            try:
                return func(*args, **kwargs)
            except error_type as e:
                func_name = getattr(func, "__qualname__", func.__name__)
                custom_message = message or f"Error in {func_name}"
                handle_error(e, error_type, custom_message, debug)
                if reraise:
                    raise
                return default_return

        return cast(F, wrapper)

    return decorator


def convert_exception(
    from_error_type: Type[Exception], to_error_type: Type[ToolkitError]
) -> Callable[[F], F]:
    """Convert a specific exception type to a toolkit exception type.

    Args:
        from_error_type: The original exception type to catch
        to_error_type: The toolkit exception type to convert to

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except from_error_type as e:
                raise to_error_type(str(e), cause=e)

        return cast(F, wrapper)

    return decorator
