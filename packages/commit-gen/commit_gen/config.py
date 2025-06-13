"""
Configuration management for commit-gen.

This module provides a centralized configuration system that loads settings from
multiple sources with proper precedence:
1. Default values
2. Configuration file values
3. Environment variables
4. Command line arguments
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import tomli
from toolkit_utils import ConfigError, error_handler
from toolkit_utils.logging_utils import warning

# * Define the configuration schema with default values
DEFAULT_CONFIG = {
    "llm": {
        "provider": "auto",  # ? "auto", "ollama", "lmstudio", "none"
        "model": None,  # ? Will be provider-specific if not specified
        "temperature": 0.7,
        "max_tokens": -1,  # ? Use model default
        "timeout": 300,  # ? seconds
        "system_message": (
            "You are a helpful AI assistant specializing in writing clear, concise, and"
            " informative git commit messages."
        ),
    },
    "providers": {
        "ollama": {
            "base_url": "http://localhost:11434",
            "default_model": "llama3",
        },
        "lmstudio": {
            "base_url": "http://localhost:1234",
            "default_model": "llama3",
            "auto_start_server": True,
        },
    },
    "templates": {
        "commit_template": None,  # ? Will use default from package if None
        "prompt_template": None,  # ? Will use default from package if None
    },
    "chunking": {
        "enabled": True,
        "default_chunk_size": 30,
        "min_chunk_size": 10,
        "max_chunk_size": 100,
        "characters_per_token": 3.5,  # ? Approximate
        "reserved_response_tokens": 500,
        "safety_margin_percent": 15,
        "min_available_percent": 30,
    },
    "output": {
        "show_reasoning": False,
        "debug": False,
        "log_level": (
            "info"
        ),  # ? "trace", "debug", "info", "success", "warning", "error", "critical"
        "show_timestamps": True,  # ? Whether to show timestamps in log messages
        "color": True,  # ? Enable/disable colored output
    },
    "git": {
        "auto_apply": False,
    },
}


def _merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries, with override values taking precedence."""
    result = base.copy()

    for key, value in override.items():
        # * If both values are dictionaries, merge them recursively
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_configs(result[key], value)
        # * Otherwise, override or add the value
        else:
            result[key] = value

    return result


@error_handler(message="Error loading configuration file", default_return={})
def _load_config_file(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file if it exists.

    Args:
        config_path: Optional path to a specific config file

    Returns:
        Dict containing configuration values or empty dict if no file found
    """
    # * Check if a specific path was provided
    if config_path:
        path = Path(config_path)
        if path.exists():
            # * Load based on file extension
            if path.suffix.lower() == ".json":
                with open(path, "r") as f:
                    try:
                        return json.load(f)
                    except json.JSONDecodeError as e:
                        raise ConfigError(
                            f"Invalid JSON in config file {path}: {str(e)}", cause=e
                        )
            elif path.suffix.lower() == ".toml":
                if tomli is None:
                    warning(f"Cannot load {path}. tomli package is not installed.")
                    return {}
                with open(path, "rb") as f:
                    try:
                        return tomli.load(f)
                    except tomli.TOMLDecodeError as e:
                        raise ConfigError(
                            f"Invalid TOML in config file {path}: {str(e)}", cause=e
                        )
            else:
                warning(f"Unsupported file format: {path.suffix}")
                return {}
        else:
            warning(f"Config file not found: {config_path}")
            return {}

    # * Look in standard locations for config files
    config_paths = [
        # ? Current directory
        Path("commit-gen.json"),
        Path("commit-gen.toml"),
        # ? User's home directory
        Path.home() / ".config" / "commit-gen" / "config.json",
        Path.home() / ".config" / "commit-gen" / "config.toml",
        Path.home() / ".commit-gen.json",
        Path.home() / ".commit-gen.toml",
    ]

    for path in config_paths:
        if path.exists():
            # * Load based on file extension
            if path.suffix.lower() == ".json":
                with open(path, "r") as f:
                    try:
                        return json.load(f)
                    except json.JSONDecodeError as e:
                        warning(f"Invalid JSON in config file {path}: {str(e)}")
                        continue
            elif path.suffix.lower() == ".toml":
                if tomli is None:
                    warning(f"Cannot load {path}. tomli package is not installed.")
                    continue
                with open(path, "rb") as f:
                    try:
                        return tomli.load(f)
                    except Exception as e:
                        warning(f"Invalid TOML in config file {path}: {str(e)}")
                        continue

    # * No config file found
    return {}


@error_handler(message="Error loading environment variables", default_return={})
def _load_env_vars() -> Dict[str, Any]:
    """Load configuration from environment variables."""
    env_config: Dict[str, Any] = {}

    # * Define the mapping of environment variables to config keys
    env_mappings = {
        "COMMIT_GEN_PROVIDER": ["llm", "provider"],
        "COMMIT_GEN_MODEL": ["llm", "model"],
        "COMMIT_GEN_TEMPERATURE": ["llm", "temperature"],
        "COMMIT_GEN_MAX_TOKENS": ["llm", "max_tokens"],
        "COMMIT_GEN_TIMEOUT": ["llm", "timeout"],
        "COMMIT_GEN_SYSTEM_MESSAGE": ["llm", "system_message"],
        "COMMIT_GEN_OLLAMA_URL": ["providers", "ollama", "base_url"],
        "COMMIT_GEN_LMSTUDIO_URL": ["providers", "lmstudio", "base_url"],
        "COMMIT_GEN_COMMIT_TEMPLATE": ["templates", "commit_template"],
        "COMMIT_GEN_PROMPT_TEMPLATE": ["templates", "prompt_template"],
        "COMMIT_GEN_DISABLE_CHUNKING": ["chunking", "enabled"],
        "COMMIT_GEN_SHOW_REASONING": ["output", "show_reasoning"],
        "COMMIT_GEN_DEBUG": ["output", "debug"],
        "COMMIT_GEN_AUTO_APPLY": ["git", "auto_apply"],
    }

    # * Process each environment variable if it exists
    for env_var, config_path in env_mappings.items():
        if env_var in os.environ:
            # * Get the value and convert to appropriate type
            env_value: str = os.environ[env_var]
            value: Union[str, bool, int, float]

            # * Handle boolean values
            if env_value.lower() in ("true", "yes", "1"):
                value = True
            elif env_value.lower() in ("false", "no", "0"):
                value = False
            # * Handle numeric values
            elif env_value.replace(".", "", 1).isdigit():
                if "." in env_value:
                    value = float(env_value)
                else:
                    value = int(env_value)
            # * Keep string values as is
            else:
                value = env_value

            # * Special handling for disable_chunking (invert the value)
            if env_var == "COMMIT_GEN_DISABLE_CHUNKING" and isinstance(value, bool):
                value = not value

            # * Build nested dictionary structure
            current = env_config
            for part in config_path[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            # * Set the value at the final level
            current[config_path[-1]] = value

    return env_config


@error_handler(message="Error applying CLI arguments", default_return={})
def _apply_cli_args(config: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    """Apply command line arguments to the configuration."""
    cli_config: Dict[str, Any] = {}

    # * Define mapping of CLI args to config keys
    cli_mappings = {
        "provider": ["llm", "provider"],
        "model": ["llm", "model"],
        "temperature": ["llm", "temperature"],
        "max_tokens": ["llm", "max_tokens"],
        "timeout": ["llm", "timeout"],
        "system_message": ["llm", "system_message"],
        "base_url": None,  # ? Special handling needed
        "commit_template": ["templates", "commit_template"],
        "prompt_template": ["templates", "prompt_template"],
        "disable_chunking": ["chunking", "enabled"],
        "show_reasoning": ["output", "show_reasoning"],
        "debug": ["output", "debug"],
        "apply": ["git", "auto_apply"],
    }

    # * Process each CLI argument if it exists and is not None
    for arg_name, config_path in cli_mappings.items():
        if arg_name in args and args[arg_name] is not None:
            value = args[arg_name]

            # * Special handling for base_url based on provider
            if arg_name == "base_url" and value is not None:
                provider = args.get("provider") or config.get("llm", {}).get(
                    "provider", "auto"
                )
                if provider in ("ollama", "lmstudio"):
                    if "providers" not in cli_config:
                        cli_config["providers"] = {}
                    if provider not in cli_config["providers"]:
                        cli_config["providers"][provider] = {}
                    cli_config["providers"][provider]["base_url"] = value
                continue

            # * Special handling for disable_chunking (invert the value)
            if arg_name == "disable_chunking" and isinstance(value, bool):
                value = not value

            # * Build nested dictionary structure
            if config_path:
                current = cli_config
                for part in config_path[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]

                # * Set the value at the final level
                current[config_path[-1]] = value

    # * Handle the config file argument separately
    if "config" in args and args["config"] is not None:
        file_config = _load_config_file(args["config"])
        if file_config:
            # * Apply file config before CLI args
            config = _merge_configs(config, file_config)

    # * Apply CLI config (highest precedence)
    return _merge_configs(config, cli_config)


@error_handler(
    message="Error loading configuration", default_return=DEFAULT_CONFIG.copy()
)
def load_config(cli_args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load configuration from all sources with proper precedence:
    1. Default values
    2. Configuration file
    3. Environment variables
    4. Command line arguments

    Args:
        cli_args: Dictionary of command line arguments

    Returns:
        Dict[str, Any]: Complete configuration dictionary
    """
    # * Start with default config
    config = DEFAULT_CONFIG.copy()

    # * Load and merge config file (if exists in standard locations)
    # ? CLI-specified config file will be handled in _apply_cli_args
    if not cli_args or "config" not in cli_args or cli_args["config"] is None:
        file_config = _load_config_file()
        if file_config:
            config = _merge_configs(config, file_config)

    # * Load and merge environment variables
    env_config = _load_env_vars()
    if env_config:
        config = _merge_configs(config, env_config)

    # * Apply command line arguments (highest precedence)
    if cli_args:
        config = _apply_cli_args(config, cli_args)

    return config


# * Global configuration singleton
_config = None


def get_config() -> Dict[str, Any]:
    """
    Get the current configuration.
    This is a singleton that ensures we only load the config once.

    Returns:
        Dict[str, Any]: The complete configuration
    """
    global _config

    if _config is None:
        _config = load_config()

    return _config


@error_handler(
    message="Error updating configuration", default_return=DEFAULT_CONFIG.copy()
)
def update_config(cli_args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update the current configuration with CLI arguments.

    Args:
        cli_args: Dictionary of command line arguments

    Returns:
        Dict[str, Any]: The updated configuration
    """
    global _config

    if _config is None:
        _config = load_config(cli_args)
    else:
        _config = _apply_cli_args(_config, cli_args)

    return _config
