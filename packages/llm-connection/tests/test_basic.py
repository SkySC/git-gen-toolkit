"""
Basic unit tests for llm-connection package.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest


def test_imports():
    """Test that all modules can be imported without errors."""
    from llm_connection import constants, message_builder, message_cleanup, providers
    from llm_connection.message_builder import build_messages
    from llm_connection.message_cleanup import clean_response
    from llm_connection.providers import LLMProvider, LMStudioProvider, OllamaProvider

    assert True  # All imports successful


def test_llm_provider_base_class():
    """Test the base LLMProvider class."""
    from llm_connection.providers import LLMProvider

    # Test that LLMProvider is abstract
    with pytest.raises(TypeError):
        LLMProvider()


def test_ollama_provider():
    """Test OllamaProvider functionality."""
    from llm_connection.providers import OllamaProvider

    provider = OllamaProvider(model="llama3", base_url="http://localhost:11434")

    assert provider.name == "Ollama"
    assert provider.model == "llama3"
    assert provider.base_url == "http://localhost:11434"


def test_lmstudio_provider():
    """Test LMStudioProvider functionality."""
    from llm_connection.providers import LMStudioProvider

    provider = LMStudioProvider(model="llama3", base_url="http://localhost:1234")

    assert provider.name == "LM Studio"
    assert provider.model == "llama3"
    assert provider.base_url == "http://localhost:1234"


def test_message_builder():
    """Test message building functionality."""
    from llm_connection.message_builder import build_messages

    system_msg = "You are a helpful assistant"
    user_msg = "Hello world"

    messages = build_messages(system_msg, user_msg)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == system_msg
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == user_msg


def test_message_cleanup():
    """Test message cleanup functionality."""
    from llm_connection.message_cleanup import clean_response

    # Test basic cleanup
    dirty_response = "  Here is a response with extra whitespace  \n\n"
    clean = clean_response(dirty_response)
    assert clean == "Here is a response with extra whitespace"

    # Test markdown code block cleanup
    markdown_response = "```\nsome code\n```"
    clean = clean_response(markdown_response)
    assert clean == "some code"
