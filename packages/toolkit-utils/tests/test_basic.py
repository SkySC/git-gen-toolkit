"""
Basic unit tests for toolkit-utils package.
"""

from unittest.mock import MagicMock, patch

import pytest


def test_imports():
    """Test that all modules can be imported without errors."""
    from toolkit_utils import error_utils, logging_utils, progress_utils, template_utils
    from toolkit_utils.error_utils import LLMProviderError, ToolkitError
    from toolkit_utils.logging_utils import error, info, setup_logging
    from toolkit_utils.progress_utils import create_progress_bar, create_spinner
    from toolkit_utils.template_utils import enrich_template, load_template_file

    assert True  # All imports successful


def test_error_utils():
    """Test error utilities."""
    from toolkit_utils.error_utils import LLMProviderError, ToolkitError, error_handler

    # Test custom exceptions
    with pytest.raises(ToolkitError):
        raise ToolkitError("Test error")

    with pytest.raises(LLMProviderError):
        raise LLMProviderError("Test LLM error")

    # Test error handler decorator
    @error_handler(message="Test error handler", default_return="default")
    def failing_function():
        raise ValueError("Test error")

    result = failing_function()
    assert result == "default"


def test_template_utils():
    """Test template utilities."""
    from toolkit_utils.template_utils import enrich_template

    # Test template enrichment
    template = "Hello {{name}}, welcome to {{project}}!"
    context = {"name": "World", "project": "Test"}

    result = enrich_template(template, context)
    assert result == "Hello World, welcome to Test!"


def test_progress_utils():
    """Test progress utilities."""
    from toolkit_utils.progress_utils import create_progress_bar, create_spinner

    # Test spinner creation
    spinner = create_spinner("Testing...")
    assert spinner is not None

    # Test progress bar creation
    progress_bar = create_progress_bar("Processing", total=100)
    assert progress_bar is not None
