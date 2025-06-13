"""
Constants used throughout the pr-gen package.
This module contains default templates and other constants.
"""

import os

# * Default templates
DEFAULT_PR_TEMPLATE = """
## Description
[Brief description of the changes]

## Changes Made
[List of main changes]

## Testing Done
[Describe tests that were performed]
"""

DEFAULT_PROMPT_TEMPLATE = """
Based on these git commits from branch '{{branch_name}}', please create a professional pull request description in Markdown format.
Use this template:

{{pr_template}}

## Commits
{{commit_list}}

Please fill in the template sections with appropriate content derived from the commit messages.
"""

# * Default configuration values
DEFAULT_SYSTEM_MESSAGE = (
    "You are a helpful AI assistant specializing in creating professional pull request"
    " descriptions."
)
DEFAULT_LLM_TEMPERATURE = 0.7
DEFAULT_LLM_TIMEOUT = 300
DEFAULT_LLM_MAX_TOKENS = -1  # * Use model default
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_LMSTUDIO_BASE_URL = "http://localhost:1234"
DEFAULT_OLLAMA_MODEL = "llama3"
DEFAULT_LMSTUDIO_MODEL = "llama3"

# * Chunking parameters
DEFAULT_CHUNK_SIZE = 30
MIN_CHUNK_SIZE = 10
MAX_CHUNK_SIZE = 100


def get_default_template_path(filename):
    """Get the path to a template file included in the package."""
    try:
        # * Try to find the package resources using importlib
        try:
            # * Python 3.9+
            from importlib.resources import files

            return str(files("pr_gen.templates").joinpath(filename))
        except ImportError:
            # * Python 3.7-3.8
            import importlib.resources as pkg_resources

            return pkg_resources.path("pr_gen.templates", filename)
    except (ImportError, ModuleNotFoundError):
        # * Fall back to looking in the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        template_dir = os.path.join(script_dir, "templates")

        return os.path.join(template_dir, filename)
