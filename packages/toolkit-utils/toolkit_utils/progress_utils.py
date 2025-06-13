"""Progress feedback utilities for Git-Gen-Toolkit.

This module provides consistent progress indicators, spinners, and progress bars
for long-running operations across the toolkit.
"""

import time
from typing import Any, Callable, List, Optional, TypeVar

from halo import Halo
from tqdm import tqdm

# * Type variable for generic function return types
T = TypeVar("T")


def progress_bar(
    iterable=None,
    total: Optional[int] = None,
    desc: str = "",
    unit: str = "it",
    leave: bool = True,
    position: int = 0,
) -> tqdm:
    """Create a consistent progress bar using tqdm.

    Args:
        iterable: Iterable to decorate with a progressbar
        total: Total number of items if iterable is not provided
        desc: Description to show alongside the progress bar
        unit: String to use as the unit of iteration
        leave: Whether to leave the progress bar on screen after completion
        position: Position of the progress bar (vertical offset)

    Returns:
        A tqdm instance for iteration or manual updates
    """
    return tqdm(
        iterable=iterable,
        total=total,
        desc=desc,
        unit=unit,
        leave=leave,
        position=position,
        dynamic_ncols=True,  # ? Adapt to terminal width
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {unit} [{elapsed}<{remaining}]",
    )


def spinner(
    text: str = "Processing",
    spinner_type: str = "dots",
    color: str = "cyan",
    text_color: Optional[str] = None,
) -> Halo:
    """Create a spinner for indeterminate progress operations.

    Args:
        text: Text to display alongside the spinner
        spinner_type: Type of spinner animation ('dots', 'dots12', etc.)
        color: Color of the spinner
        text_color: Color of the text (same as spinner if None)

    Returns:
        A Halo spinner instance
    """
    return Halo(
        text=text,
        spinner=spinner_type,
        color=color,
        text_color=text_color,
    )


def with_spinner(
    func: Callable[..., T],
    text: str = "Processing",
    spinner_type: str = "dots",
    success_text: Optional[str] = None,
    error_text: Optional[str] = None,
) -> Callable[..., T]:
    """Decorator to run a function with a spinner.

    Args:
        func: Function to decorate
        text: Text to display alongside the spinner
        spinner_type: Type of spinner animation
        success_text: Text to show on success (default: same as text + " complete")
        error_text: Text to show on error (default: same as text + " failed")

    Returns:
        Decorated function that shows a spinner while running
    """

    def wrapper(*args, **kwargs) -> T:
        s = spinner(text, spinner_type)
        s.start()
        try:
            result = func(*args, **kwargs)
            s.succeed(success_text or f"{text} complete")
            return result
        except Exception as e:
            s.fail(error_text or f"{text} failed: {str(e)}")
            raise
        finally:
            s.stop()

    return wrapper


def with_progress(
    items: List[Any],
    func: Callable[[Any], T],
    desc: str = "Processing",
    unit: str = "item",
) -> List[T]:
    """Process a list of items with a progress bar.

    Args:
        items: List of items to process
        func: Function to apply to each item
        desc: Description for the progress bar
        unit: Unit label for the items

    Returns:
        List of results from applying the function to each item
    """
    results = []
    with progress_bar(total=len(items), desc=desc, unit=unit) as pbar:
        for item in items:
            result = func(item)
            results.append(result)
            pbar.update(1)
    return results


def indeterminate_progress(
    check_func: Callable[[], bool],
    success_message: str = "Operation complete",
    timeout: float = 60.0,
    check_interval: float = 0.5,
    spinner_text: str = "Processing",
) -> bool:
    """Run a spinner until a condition is met or timeout occurs.

    Args:
        check_func: Function that returns True when operation is complete
        success_message: Message to display when operation succeeds
        timeout: Maximum time to wait in seconds
        check_interval: How often to check the condition in seconds
        spinner_text: Text to display alongside the spinner

    Returns:
        True if operation completed successfully, False if timed out
    """
    s = spinner(spinner_text)
    s.start()

    start_time = time.time()
    try:
        while time.time() - start_time < timeout:
            if check_func():
                s.succeed(success_message)
                return True
            time.sleep(check_interval)

        s.fail(f"Operation timed out after {timeout:.1f} seconds")
        return False
    except Exception as e:
        s.fail(f"Operation failed: {str(e)}")
        raise
    finally:
        s.stop()
