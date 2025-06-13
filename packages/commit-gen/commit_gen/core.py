#!/usr/bin/env python3

"""
Core API functionality for commit-gen.

This module provides the core functionality of the commit-gen package as a clean API
that can be imported and used by other applications.
"""

from typing import Dict, Optional, Tuple

from llm_connection.message_cleanup import clean_message
from llm_connection.providers import LLMProvider
from toolkit_utils import LLMProviderError, ToolkitError, error_handler

from .chunking_utils import handle_diff_chunking, process_chunks
from .git_utils import get_current_branch_name, get_recent_commits
from .template_utils import enrich_template_with_git_context


@error_handler(
    message="Error generating commit message",
    default_return=("", "Error generating commit message"),
)
def generate_commit_message_for_diff(
    diff_content: str,
    provider_instance: LLMProvider,
    commit_template_content: str,
    prompt_template_content: str,
    chunk_size: Optional[int] = None,
    disable_chunking: bool = False,
    debug: bool = False,
) -> Tuple[str, str]:
    """
    Generate a commit message for a git diff using an LLM.

    Core API function that handles the commit message generation process.
    This function doesn't print any output or interact with the user.

    Args:
        diff_content: The git diff content as a string
        provider_instance: An initialized LLM provider instance
        commit_template_content: The commit message template content
        prompt_template_content: The prompt template content
        chunk_size: Optional maximum number of lines per chunk
        disable_chunking: Whether to disable diff chunking
        debug: Whether to enable debug mode

    Returns:
        Tuple containing (commit_message, reasoning)
    """
    if not diff_content:
        return "", "No diff content provided"

    if not provider_instance:
        raise ToolkitError("No valid LLM provider instance")

    if not commit_template_content or not prompt_template_content:
        raise ToolkitError("Missing template content")

    # * Create a simple args-like object for chunking
    class Args:
        def __init__(self):
            self.debug = debug
            self.max_lines_per_chunk = chunk_size
            self.disable_chunking = disable_chunking

    args = Args()

    # * Handle chunking of the diff
    diff_chunks = handle_diff_chunking(
        diff_content,
        args,
        provider_instance,
        commit_template_content,
        prompt_template_content,
    )

    # * Generate the commit message
    def generate_message(provider, diff, commit_tpl, prompt_tpl):
        return provider.generate_with_template(
            diff,
            primary_template_content=commit_tpl,
            prompt_template_content=prompt_tpl,
        )

    raw_message = process_chunks(
        provider_instance,
        diff_chunks,
        commit_template_content,
        prompt_template_content,
        debug,
        generate_commit_message_func=generate_message,
    )

    # * Clean up the message and extract reasoning
    if raw_message:
        message, reasoning = clean_message(raw_message)
        return message, reasoning

    return "", "Failed to generate commit message"


@error_handler(message="Error setting up LLM provider", default_return=None)
def setup_provider_from_config(config: Dict) -> Optional[LLMProvider]:
    """
    Set up and return an LLM provider based on configuration.

    Pure function version of the provider setup logic, without interactive components.

    Args:
        config: Configuration dictionary

    Returns:
        An initialized LLM provider instance or None
    """
    from llm_connection.providers import LMStudioProvider, OllamaProvider

    # * Check if LLM is disabled
    if config["llm"]["provider"] == "none":
        return None

    provider_type = config["llm"]["provider"]
    provider_instance: Optional[LLMProvider] = None

    # * Provider options from config
    provider_options = {
        "temperature": config["llm"]["temperature"],
        "max_tokens": config["llm"]["max_tokens"],
        "system_message": config["llm"]["system_message"],
    }

    # * Remove max_tokens if set to -1 (use model default)
    if provider_options["max_tokens"] == -1:
        del provider_options["max_tokens"]

    # * Initialize the specified provider
    if provider_type == "auto":
        # * For API usage, auto should try providers in order: ollama, lmstudio
        # * and use the first available one
        try:
            # * Try Ollama first
            provider_config = config["providers"]["ollama"]
            provider_instance = OllamaProvider(
                model=config["llm"]["model"] or provider_config["default_model"],
                base_url=provider_config["base_url"],
                timeout=config["llm"]["timeout"],
                **provider_options,
            )

            if not provider_instance.is_available():
                provider_instance = None
        except Exception:
            provider_instance = None

        if provider_instance is None:
            try:
                # * Try LM Studio if Ollama failed
                provider_config = config["providers"]["lmstudio"]
                provider_instance = LMStudioProvider(
                    model=config["llm"]["model"] or provider_config["default_model"],
                    base_url=provider_config["base_url"],
                    auto_start_server=provider_config.get(
                        "auto_start_server", False
                    ),  # ? Default to False for API
                    timeout=config["llm"]["timeout"],
                    **provider_options,
                )

                if not provider_instance.is_available():
                    provider_instance = None
            except Exception:
                provider_instance = None

    elif provider_type == "ollama":
        provider_config = config["providers"]["ollama"]
        try:
            provider_instance = OllamaProvider(
                model=config["llm"]["model"] or provider_config["default_model"],
                base_url=provider_config["base_url"],
                timeout=config["llm"]["timeout"],
                **provider_options,
            )
        except Exception as e:
            raise LLMProviderError(
                f"Failed to initialize Ollama provider: {str(e)}", cause=e
            )

    elif provider_type == "lmstudio":
        provider_config = config["providers"]["lmstudio"]
        try:
            provider_instance = LMStudioProvider(
                model=config["llm"]["model"] or provider_config["default_model"],
                base_url=provider_config["base_url"],
                auto_start_server=provider_config.get(
                    "auto_start_server", False
                ),  # ? Default to False for API
                timeout=config["llm"]["timeout"],
                **provider_options,
            )
        except Exception as e:
            raise LLMProviderError(
                f"Failed to initialize LM Studio provider: {str(e)}", cause=e
            )
    else:
        raise LLMProviderError(f"Unknown provider type: {provider_type}")

    return provider_instance


@error_handler(message="Error enriching templates with context", default_return="")
def enrich_templates_with_context(
    prompt_template_content: str, repo_path: str = "."
) -> str:
    """
    Enrich the prompt template with git context information.

    Args:
        prompt_template_content: The original template content
        repo_path: Path to the git repository

    Returns:
        The enriched template content
    """
    if not prompt_template_content:
        return ""

    branch_name = get_current_branch_name(repo_path)
    recent_commits = get_recent_commits(repo_path)

    return enrich_template_with_git_context(
        prompt_template_content, branch_name, recent_commits
    )


@error_handler(
    message="Error creating commit message",
    default_return=("", "Error creating commit message"),
)
def create_commit_message(
    diff_content: Optional[str] = None,
    config_dict: Optional[Dict] = None,
    repo_path: str = ".",
    custom_commit_template: Optional[str] = None,
    custom_prompt_template: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Generate a commit message from a diff with minimal setup.

    This is a high-level function that combines all the necessary steps
    to generate a commit message with minimal configuration.

    Args:
        diff_content: Git diff content as string, will use staged diff if None
        config_dict: Optional configuration dictionary, uses default if None
        repo_path: Path to git repository for diff and context
        custom_commit_template: Optional custom commit template content
        custom_prompt_template: Optional custom prompt template content

    Returns:
        A tuple of (commit_message, reasoning)
    """
    from .config import get_config
    from .git_utils import get_staged_diff
    from .template_utils import load_commit_template, load_prompt_template

    # * Use default config if none provided
    if config_dict is None:
        config_dict = get_config()

    # * Get diff content if not provided
    if diff_content is None:
        diff_content = get_staged_diff(repo_path)
        if not diff_content:
            return "", "No staged changes to process"

    # * Set up the LLM provider
    provider = setup_provider_from_config(config_dict)
    if not provider:
        return "", "No valid LLM provider available"

    # * Check if provider is ready
    if not provider.is_model_loaded():
        return "", f"Model '{provider.model}' is not loaded"

    # * Load templates
    template_paths = config_dict["templates"]

    # * Use custom templates if provided, otherwise load from config
    if custom_commit_template:
        commit_template_content = custom_commit_template
    else:
        commit_template_content = load_commit_template(
            template_paths.get("commit_template")
        )
        if not commit_template_content:
            return "", "Failed to load commit template"

    if custom_prompt_template:
        prompt_template_content = custom_prompt_template
    else:
        prompt_template_content = load_prompt_template(
            template_paths.get("prompt_template")
        )
        if not prompt_template_content:
            return "", "Failed to load prompt template"

    # * Enrich prompt template with git context
    prompt_template_content = enrich_templates_with_context(
        prompt_template_content, repo_path
    )
    if not prompt_template_content:
        return "", "Failed to enrich template with git context"

    # * Generate the commit message
    return generate_commit_message_for_diff(
        diff_content,
        provider,
        commit_template_content,
        prompt_template_content,
        chunk_size=None,
        disable_chunking=config_dict.get("chunking", {}).get("enabled", True) is False,
        debug=config_dict.get("output", {}).get("debug", False),
    )


# ? Example of API usage:
"""
# Import the API
from commit_gen.core import create_commit_message
from commit_gen.config import load_config

# Load configuration (or create your own)
my_config = load_config()
my_config["llm"]["provider"] = "ollama"
my_config["llm"]["model"] = "llama3"

# Get a commit message for the staged changes
message, reasoning = create_commit_message(config_dict=my_config)

# Or get a commit message for a specific diff
with open("my_diff.patch", "r") as f:
    diff_content = f.read()

message, reasoning = create_commit_message(
  diff_content=diff_content, config_dict=my_config
)

# You can then use the message programmatically
print(f"Generated commit message: {message}")
if reasoning:
    print(f"Reasoning: {reasoning}")
"""
