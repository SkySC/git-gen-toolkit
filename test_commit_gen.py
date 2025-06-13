#!/usr/bin/env python3

"""
Test script for debugging commit-gen functionality.
"""

import sys
import traceback


def test_basic_functionality():
    """Test basic functionality step by step."""
    try:
        print("Testing commit-gen basic functionality...")

        # Test imports
        print("1. Testing imports...")
        from commit_gen.main import main, parse_arguments

        print("   ✓ Successfully imported main functions")

        # Test argument parsing
        print("2. Testing argument parsing...")
        import sys

        # Simulate help argument
        old_argv = sys.argv
        sys.argv = ["cgen", "--help"]
        try:
            args = parse_arguments()
        except SystemExit as e:
            # Help exits with code 0, this is expected
            print(f"   ✓ Help command exited with code: {e.code}")
        finally:
            sys.argv = old_argv

        # Test basic parsing without arguments
        print("3. Testing basic argument parsing...")
        sys.argv = ["cgen"]
        try:
            args = parse_arguments()
            print(f"   ✓ Arguments parsed successfully: {vars(args)}")
        except Exception as e:
            print(f"   ✗ Error parsing arguments: {e}")
        finally:
            sys.argv = old_argv

        print("All basic tests completed!")

    except Exception as e:
        print(f"Error during testing: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    test_basic_functionality()
