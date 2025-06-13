#!/usr/bin/env python3

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "packages", "toolkit-utils"))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "packages", "llm-connection")
)

from toolkit_utils.logging_utils import error, warning, info
from toolkit_utils.progress_utils import spinner
import time

print("Testing spinner and error formatting...")
print()

# Test case 1: Spinner fail followed by error (this was the main issue)
print("=== Test 1: Spinner fail + error ===")
s = spinner("Testing model loading")
s.start()
time.sleep(0.5)  # Short sleep to see spinner
s.fail("Model not found")
print()  # This should create a line break
error("Cannot connect to service")
print()

# Test case 2: Just logging without spinner
print("=== Test 2: Regular logging ===")
error("Error message")
warning("Warning message")
info("Info message")
print()

print("=== Test completed ===")
