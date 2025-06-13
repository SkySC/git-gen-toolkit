#!/usr/bin/env python3

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "packages", "toolkit-utils"))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "packages", "llm-connection")
)

import time

from toolkit_utils.logging_utils import error, info, setup_logging, warning
from toolkit_utils.progress_utils import spinner

print("Testing logging initialization and formatting...")
print()

# Initialize logging
setup_logging(level="info")

print("=== Test 1: Spinner fail + error (should be properly separated now) ===")
s = spinner("Testing model loading")
s.start()
time.sleep(0.5)
s.fail("Model not found")
print()  # Manual line break
error("Cannot connect to service")
print()

print("=== Test 2: Regular logging ===")
error("Error message")
warning("Warning message")
info("Info message")
print()

print("=== Test completed ===")
