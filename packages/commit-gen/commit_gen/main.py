#!/usr/bin/env python3

"""
Command-line interface for commit-gen.

This module provides the CLI functionality that wraps around the core API.
"""

import argparse
import sys

from llm_connection.providers import LLMProvider, LMStudioProvider, OllamaProvider
from toolkit_utils import LLMProviderError, ToolkitError, error_handler
from toolkit_utils.logging_utils import (
    error,
    info,
    print_commit_message,
    print_diff,
    print_reasoning,
    setup_logging,
)

from .config import get_config, update_config
from .core import generate_commit_message_for_diff
from .git_utils import (
    apply_commit_message,
    get_current_branch_name,
    get_recent_commits,
    get_staged_diff,
)
from .interactive_utils import (
    select_model_interactively,
)
from .template_utils import (
    enrich_template_with_git_context,
    load_commit_template,
    load_prompt_template,
)

# --- LLM Interaction ---


@error_handler(message="Error setting up LLM provider", default_return=None)
def setup_provider(args=None):
    """Set up the LLM provider based on configuration.

    Args:
        args: Command line arguments (optional, will use config if not provided)

    Returns:
        An initialized provider instance or None
    """
    # * Load configuration with CLI args if provided
    if args:
        config = update_config(vars(args))
    else:
        config = get_config()

    # * Check if LLM is disabled
    if config["llm"]["provider"] == "none":
        return None

    provider_type = config["llm"]["provider"]
    provider_instance = None

    # * Auto-detect available providers
    if provider_type == "auto":
        providers = LLMProvider.get_available_providers(timeout=3)

        if not providers:
            raise LLMProviderError(
                "No LLM providers detected. Please ensure Ollama or LM Studio is"
                " running."
            )

        model_name = config["llm"]["model"]
        if model_name:
            # * If a specific model was requested, find a provider that has it
            model_found = False
            for provider_name, provider in providers.items():
                models = provider.list_models()
                for model in models:
                    if model["id"] == model_name:
                        provider.model = model_name

                        # * Check if the model is actually loaded and ready
                        if provider.is_model_loaded():
                            provider_instance = provider
                            model_found = True
                            info(f"Using {provider_name} with model {model_name}")
                            break
                        else:
                            error(
                                f"Model {model_name} is available in"
                                f" {provider_name} but not loaded yet."
                            )
                            error(
                                f"Please load the model in {provider_name} and try"
                                " again."
                            )
                if model_found:
                    break

            if not model_found:
                error(
                    f"Model {model_name} not found or not loaded in any available"
                    " provider."
                )
                info("Available models will be listed for selection.")
                provider_instance = None

        # * If no provider_instance yet, use interactive selection
        if not provider_instance:
            provider_instance, selected_model = select_model_interactively(providers)

    # * Explicit provider specified
    else:
        # * Prepare provider options
        provider_options = {
            "temperature": config["llm"]["temperature"],
            "max_tokens": config["llm"]["max_tokens"],
            "system_message": config["llm"]["system_message"],
        }
        if provider_options["max_tokens"] == -1:
            del provider_options["max_tokens"]

        # * Initialize the specified provider
        if provider_type == "ollama":
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

            # * If no model specified, prompt interactively
            if not config["llm"]["model"]:
                if provider_instance.is_available():
                    _, selected_model = select_model_interactively(
                        {"ollama": provider_instance}
                    )
                    if not selected_model:
                        return None
                else:
                    raise LLMProviderError(
                        "Ollama provider not available. Please check if it's running."
                    )
            # * Check if the specified model is loaded
            elif not provider_instance.is_model_loaded():
                raise LLMProviderError(
                    f"Model {config['llm']['model']} is available in Ollama but not"
                    " loaded yet. Please load the model in Ollama and try again."
                )

        elif provider_type == "lmstudio":
            provider_config = config["providers"]["lmstudio"]
            try:
                provider_instance = LMStudioProvider(
                    model=config["llm"]["model"] or provider_config["default_model"],
                    base_url=provider_config["base_url"],
                    auto_start_server=provider_config.get("auto_start_server", True),
                    timeout=config["llm"]["timeout"],
                    **provider_options,
                )
            except Exception as e:
                raise LLMProviderError(
                    f"Failed to initialize LM Studio provider: {str(e)}", cause=e
                )

            # * If no model specified, prompt interactively
            if not config["llm"]["model"]:
                if provider_instance.is_available():
                    _, selected_model = select_model_interactively(
                        {"lmstudio": provider_instance}
                    )
                    if not selected_model:
                        return None
                else:
                    raise LLMProviderError(
                        "LM Studio provider not available. Please check if it's"
                        " running."
                    )
            # * Check if the specified model is loaded
            elif not provider_instance.is_model_loaded():
                raise LLMProviderError(
                    f"Model {config['llm']['model']} is available in LM Studio but not"
                    " loaded yet. Please load the model in LM Studio and try again."
                )
        else:
            raise LLMProviderError(f"Unknown provider type: {provider_type}")

    return provider_instance


def parse_arguments():
    """Parse command line arguments for the commit generation tool."""
    from . import __version__

    parser = argparse.ArgumentParser(
        description="Generate commit messages from staged git diffs."
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=f"cgen {__version__}",
        help="Show the version of the commit-gen tool",
    )
    parser.add_argument(
        "--model",
        help="LLM model (if not specified, will detect and select interactively)",
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "lmstudio", "none", "auto"],
        default="auto",
        help="LLM provider to use (default: auto-detect)",
    )
    parser.add_argument("--base-url", help="API base URL (optional)")
    parser.add_argument(
        "--commit-template",
        help="Path to a custom commit template file",
    )
    parser.add_argument(
        "--prompt-template",
        help="Path to a custom prompt template file",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Request timeout in seconds",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="LLM temperature",
    )
    parser.add_argument(
        "--max-tokens", type=int, help="LLM max tokens (-1 for model default)"
    )
    parser.add_argument(
        "--system-message",
        type=str,
        help="LLM system message",
    )
    parser.add_argument(
        "--show-reasoning",
        action="store_true",
        help="Show model reasoning/thought process if present",
    )
    parser.add_argument(
        "--max-lines-per-chunk",
        type=int,
        default=None,
        help=(
            "Maximum number of diff lines per chunk (overrides automatic calculation"
            " based on model context size)"
        ),
    )
    parser.add_argument(
        "--disable-chunking",
        action="store_true",
        help="Disable diff chunking for large changes",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the generated commit message after generation",
    )
    parser.add_argument(
        "--config",
        help="Path to a configuration file",
    )

    return parser.parse_args()


# --- Main Execution ---


@error_handler(message="Error in commit-gen main function", default_return=1)
def main():
    # * Parse command line arguments
    args = parse_arguments()

    # * Load the configuration with CLI args
    config = update_config(vars(args))

    # * Setup logging based on config
    setup_logging(level="debug" if config["output"]["debug"] else "info", config=config)

    # * Early exit if provider is none
    if config["llm"]["provider"] == "none":
        info("Skipping LLM generation.")
        return 0

    # * Get staged diff
    diff = get_staged_diff()
    if not diff:
        raise ToolkitError("No staged changes to process")

    if config["output"]["debug"]:
        print_diff(diff)

    # * Get git context for template enrichment
    branch_name = get_current_branch_name()
    recent_commits = get_recent_commits()

    # * Load templates based on configuration
    template_paths = config["templates"]
    commit_template_path = template_paths["commit_template"]
    prompt_template_path = template_paths["prompt_template"]

    # * Load templates (defaults if not specified)
    commit_template_content = load_commit_template(commit_template_path)
    if not commit_template_content:
        raise ToolkitError("Failed to load commit template")

    prompt_template_content = load_prompt_template(prompt_template_path)
    if not prompt_template_content:
        raise ToolkitError("Failed to load prompt template")

    # * Enrich prompt template with git context
    prompt_template_content = enrich_template_with_git_context(
        prompt_template_content, branch_name, recent_commits
    )
    if not prompt_template_content:
        raise ToolkitError("Failed to enrich prompt template with git context")

    # * Initialize provider - use interactive setup for CLI
    provider_instance = setup_provider(args)
    if not provider_instance:
        raise ToolkitError("No valid LLM provider available")

    info(f"Generating commit message using {provider_instance.name}...")

    # * Check if the model is loaded before proceeding
    if not provider_instance.is_model_loaded():
        raise ToolkitError(
            f"Model '{provider_instance.model}' is not loaded. Please load the model in"
            " the provider UI before continuing."
        )

    # * Generate the commit message using the core API
    message, reasoning = generate_commit_message_for_diff(
        diff,
        provider_instance,
        commit_template_content,
        prompt_template_content,
        chunk_size=args.max_lines_per_chunk,
        disable_chunking=args.disable_chunking,
        debug=config["output"]["debug"],
    )

    if not message:
        raise ToolkitError("Failed to generate commit message")

    # * Display the message
    print_commit_message(message)

    # * Show reasoning if requested
    if config["output"]["show_reasoning"] and reasoning:
        print_reasoning(reasoning)

    # * Apply the commit message if requested
    if config["git"]["auto_apply"]:
        apply_commit_message(message)

    return 0


if __name__ == "__main__":
    # * When running directly, add the parent directory to path
    if __package__ is None:
        from pathlib import Path

        # * Add the parent directory to sys.path
        file = Path(__file__).resolve()
        parent = file.parent.parent
        sys.path.append(str(parent))

        # * Set the package name
        __package__ = "commit_gen"

    main()
