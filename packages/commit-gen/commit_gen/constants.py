"""
Constants used throughout the commit-gen package.
This module contains template utilities and other non-configurable constants.
Configuration values have been moved to the config module.
"""

import os


# * Constants for template handling
def get_default_template_path(filename):
    """Get the path to a template file included in the package."""
    try:
        # * Try to find the package resources using importlib
        try:
            # ? Python 3.9+
            from importlib.resources import files

            return str(files("commit_gen.templates").joinpath(filename))
        except ImportError:
            # ? Python 3.7-3.8
            import importlib.resources as pkg_resources

            return pkg_resources.path("commit_gen.templates", filename)
    except (ImportError, ModuleNotFoundError):
        # * Fall back to looking in the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        template_dir = os.path.join(script_dir, "templates")
        return os.path.join(template_dir, filename)


DEFAULT_COMMIT_TEMPLATE_PATH = get_default_template_path("commit-template.md")
DEFAULT_PROMPT_TEMPLATE_PATH = get_default_template_path("prompt-template.md")
