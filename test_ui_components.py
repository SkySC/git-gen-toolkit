#!/usr/bin/env python3

"""
Test script for progress and logging utilities.
"""

import time

from toolkit_utils.logging_utils import error, info, success, warning
from toolkit_utils.progress_utils import create_progress_bar, create_spinner


def test_spinner():
    """Test spinner functionality."""
    print("\nTesting spinner...")

    spinner = create_spinner("Processing data...")
    spinner.start()
    time.sleep(2)
    spinner.stop()

    success("Spinner test completed!")


def test_progress_bar():
    """Test progress bar functionality."""
    print("\nTesting progress bar...")

    progress = create_progress_bar("Processing items", total=10)

    for i in range(10):
        time.sleep(0.1)
        progress.update(1)

    progress.close()
    success("Progress bar test completed!")


def test_logging():
    """Test logging functionality."""
    print("\nTesting logging utilities...")

    info("This is an info message")
    success("This is a success message")
    warning("This is a warning message")

    success("Logging test completed!")


if __name__ == "__main__":
    print("Testing toolkit-utils progress and logging functionality...")

    test_logging()
    test_spinner()
    test_progress_bar()

    print("\n🎉 All tests completed successfully!")
