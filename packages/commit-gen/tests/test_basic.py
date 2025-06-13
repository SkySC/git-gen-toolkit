"""
Basic unit tests for commit-gen package.
"""

from unittest.mock import MagicMock, mock_open, patch

import pytest


def test_imports():
    """Test that all modules can be imported without errors."""
    from commit_gen import chunking_utils, config, core, git_utils, template_utils
    from commit_gen.chunking_utils import chunk_diff
    from commit_gen.config import get_config, update_config
    from commit_gen.core import generate_commit_message_for_diff
    from commit_gen.git_utils import get_current_branch_name, get_staged_diff
    from commit_gen.template_utils import load_commit_template, load_prompt_template

    assert True  # All imports successful


def test_config_functionality():
    """Test configuration loading and updating."""
    from commit_gen.config import get_config, update_config

    # Test default config
    config = get_config()
    assert "llm" in config
    assert "templates" in config
    assert "output" in config
    assert "git" in config

    # Test config update
    updated_config = update_config({"debug": True})
    assert "debug" in updated_config or updated_config["output"]["debug"] == True


@patch("commit_gen.git_utils.subprocess.run")
def test_git_utils(mock_run):
    """Test git utility functions."""
    from commit_gen.git_utils import get_current_branch_name, get_staged_diff

    # Mock git diff output
    mock_run.return_value = MagicMock(
        stdout="diff --git a/file.py b/file.py\n+added line", returncode=0
    )

    diff = get_staged_diff()
    assert "diff --git" in diff

    # Mock git branch output
    mock_run.return_value = MagicMock(stdout="main", returncode=0)

    branch = get_current_branch_name()
    assert branch == "main"


def test_chunking_utils():
    """Test diff chunking functionality."""
    from commit_gen.chunking_utils import calculate_chunk_size, chunk_diff

    # Test chunk size calculation
    chunk_size = calculate_chunk_size(context_size=4096, buffer=0.2)
    assert isinstance(chunk_size, int)
    assert chunk_size > 0

    # Test diff chunking
    large_diff = "\n".join([f"line {i}" for i in range(100)])
    chunks = chunk_diff(large_diff, max_lines=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.split("\n")) <= 20


@patch("builtins.open", new_callable=mock_open, read_data="Template content: {{diff}}")
def test_template_utils(mock_file):
    """Test template loading functionality."""
    from commit_gen.template_utils import load_commit_template, load_prompt_template

    # Test template loading
    template = load_commit_template("test_template.md")
    assert template == "Template content: {{diff}}"

    prompt_template = load_prompt_template("test_prompt.md")
    assert prompt_template == "Template content: {{diff}}"


def test_core_chunking_logic():
    """Test the core chunking decision logic."""
    from commit_gen.core import should_chunk_diff

    # Small diff should not be chunked
    small_diff = "\n".join([f"line {i}" for i in range(10)])
    assert not should_chunk_diff(small_diff, chunk_size=50)

    # Large diff should be chunked
    large_diff = "\n".join([f"line {i}" for i in range(100)])
    assert should_chunk_diff(large_diff, chunk_size=50)
