#!/usr/bin/env python3

"""
Direct functionality test for git-gen-toolkit packages.
"""

import sys
import traceback


def test_toolkit_utils():
    """Test toolkit-utils package functionality."""
    print("Testing toolkit-utils package...")

    try:
        # Test imports
        from toolkit_utils import (
            error_utils,
            logging_utils,
            progress_utils,
            template_utils,
        )

        print("✓ All modules imported successfully")

        # Test error utilities
        from toolkit_utils.error_utils import ToolkitError, error_handler

        @error_handler(message="Test error", default_return="handled")
        def test_error_func():
            raise ValueError("Test error")

        result = test_error_func()
        assert result == "handled"
        print("✓ Error handling works correctly")

        # Test template utilities
        from toolkit_utils.template_utils import enrich_template

        template = "Hello {{name}}!"
        context = {"name": "World"}
        result = enrich_template(template, context)
        assert result == "Hello World!"
        print("✓ Template enrichment works correctly")

        # Test progress utilities
        from toolkit_utils.progress_utils import create_spinner

        spinner = create_spinner("Testing...")
        assert spinner is not None
        print("✓ Progress utilities work correctly")

        print("✓ toolkit-utils package test passed!\n")
        return True

    except Exception as e:
        print(f"✗ toolkit-utils test failed: {e}")
        traceback.print_exc()
        return False


def test_llm_connection():
    """Test llm-connection package functionality."""
    print("Testing llm-connection package...")

    try:
        # Test imports
        from llm_connection import message_builder, message_cleanup, providers

        print("✓ All modules imported successfully")

        # Test provider creation
        from llm_connection.providers import LMStudioProvider, OllamaProvider

        ollama = OllamaProvider(model="llama3")
        assert ollama.name == "Ollama"
        assert ollama.model == "llama3"
        print("✓ OllamaProvider creation works")

        lmstudio = LMStudioProvider(model="llama3")
        assert lmstudio.name == "LM Studio"
        assert lmstudio.model == "llama3"
        print("✓ LMStudioProvider creation works")

        # Test message building
        from llm_connection.message_builder import build_messages

        messages = build_messages("System prompt", "User message")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        print("✓ Message building works correctly")

        # Test message cleanup
        from llm_connection.message_cleanup import clean_response

        dirty = "  response with whitespace  \n"
        clean = clean_response(dirty)
        assert clean == "response with whitespace"
        print("✓ Message cleanup works correctly")

        print("✓ llm-connection package test passed!\n")
        return True

    except Exception as e:
        print(f"✗ llm-connection test failed: {e}")
        traceback.print_exc()
        return False


def test_commit_gen():
    """Test commit-gen package functionality."""
    print("Testing commit-gen package...")

    try:
        # Test imports
        from commit_gen import chunking_utils, config, template_utils

        print("✓ All modules imported successfully")

        # Test configuration
        from commit_gen.config import get_config

        config_data = get_config()
        assert "llm" in config_data
        assert "templates" in config_data
        print("✓ Configuration loading works")

        # Test chunking
        from commit_gen.chunking_utils import calculate_chunk_size, chunk_diff

        chunk_size = calculate_chunk_size(4096)
        assert isinstance(chunk_size, int)
        assert chunk_size > 0
        print("✓ Chunk size calculation works")

        # Test template loading (with defaults)
        from commit_gen.template_utils import load_commit_template

        template = load_commit_template()  # Should load default
        assert template is not None
        print("✓ Template loading works")

        print("✓ commit-gen package test passed!\n")
        return True

    except Exception as e:
        print(f"✗ commit-gen test failed: {e}")
        traceback.print_exc()
        return False


def test_pr_gen():
    """Test pr-gen package functionality."""
    print("Testing pr-gen package...")

    try:
        # Test imports
        from pr_gen import constants, main

        print("✓ All modules imported successfully")

        # Test constants
        from pr_gen.constants import DEFAULT_MODEL, DEFAULT_PROVIDER

        assert DEFAULT_MODEL is not None
        assert DEFAULT_PROVIDER is not None
        print("✓ Constants loaded correctly")

        print("✓ pr-gen package test passed!\n")
        return True

    except Exception as e:
        print(f"✗ pr-gen test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("Running comprehensive functionality tests for git-gen-toolkit...\n")

    results = []
    results.append(test_toolkit_utils())
    results.append(test_llm_connection())
    results.append(test_commit_gen())
    results.append(test_pr_gen())

    passed = sum(results)
    total = len(results)

    print(f"Test Results: {passed}/{total} packages passed")

    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
