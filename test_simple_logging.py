#!/usr/bin/env python3

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "packages", "toolkit-utils"))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "packages", "llm-connection")
)

print("Testing logging fixes...")

# Test 1: Check that logging format is correct
from toolkit_utils.logging_utils import (
    DEFAULT_LOG_FORMAT,
    error,
    info,
    setup_logging,
    warning,
)

print(f"Default log format: {DEFAULT_LOG_FORMAT}")
print("Should show 'level.name:<7' instead of 'level: <8'")

# Initialize logging
setup_logging(level="info")

print("\n=== Testing log level formatting ===")
error("This is an error message")
warning("This is a warning message")
info("This is an info message")

print("\n=== Testing completed ===")
print(
    "Check above: ERROR, WARNING, and INFO should be properly aligned without excessive spacing"
)
