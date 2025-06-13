"""Template handling utilities for git-gen-toolkit.

This module provides centralized template handling functions for loading, enriching,
and managing templates across all toolkit components. It ensures consistent error
handling and provides a unified interface for working with various template types.
"""

import os
from typing import Dict, List, Optional, Union

from .error_utils import TemplateError, error_handler
from .logging_utils import warning


@error_handler(message="Error loading template from file", default_return=None)
def load_template(
    template_path: str,
    default_template_path: Optional[str] = None,
    default_template_content: Optional[str] = None,
    template_type: str = "template",
) -> Optional[str]:
    """Load a template from file with fallback options.

    This function attempts to load a template from the specified path,
    falling back to the default template path or content if provided.
    It includes proper error handling and logging.

    Args:
        template_path: Path to the template file
        default_template_path: Path to the default template file if primary path fails
        default_template_content: Default template content as a string if both path fail
        template_type: Description of template type for logging (e.g., "commit", "PR")

    Returns:
        str: The template content or None if loading failed

    Raises:
        TemplateError: If the template cannot be loaded from any source
    """
    # * Try to load from the provided path
    if template_path and os.path.exists(template_path):
        try:
            with open(template_path, "r") as f:
                return f.read()
        except Exception as e:
            warning(f"Error reading {template_type} template from {template_path}: {e}")
    elif template_path:
        warning(f"{template_type.capitalize()} template not found: {template_path}")

    # * Try to load from the default path
    if default_template_path and os.path.exists(default_template_path):
        try:
            warning(
                f"Using default {template_type} template from {default_template_path}"
            )
            with open(default_template_path, "r") as f:
                return f.read()
        except Exception as e:
            warning(f"Error reading default {template_type} template: {e}")
    elif default_template_path:
        warning(f"Default {template_type} template not found: {default_template_path}")

    # * Use the default content if provided
    if default_template_content:
        warning(f"Using built-in default {template_type} template")
        return default_template_content

    # * If we get here, we couldn't load the template from any source
    raise TemplateError(f"Failed to load {template_type} template from any source")


@error_handler(message="Error enriching template with Git context", default_return=None)
def enrich_template_with_git_context(
    template_content: str,
    branch_name: Optional[str] = None,
    recent_commits: Optional[List[Union[Dict, object]]] = None,
    placeholder: str = "{{git_context}}",
    content_placeholder: Optional[str] = None,
) -> Optional[str]:
    """Enrich a template with Git context information.

    Adds branch and commit information to a template, either at a specific
    placeholder or in a sensible location. Handles both dictionary-based commit
    objects and Git commit objects.

    Args:
        template_content: The original template content
        branch_name: The current branch name
        recent_commits: List of recent commits (as dictionaries with 'hash', 'date',
                       'message' keys or as git.Commit objects)
        placeholder: The placeholder string to replace with Git context
        content_placeholder: Optional content placeholder (like {{diff}}
                             or {{commit_list}}) to place context before

    Returns:
        str: The enriched template content or None if enrichment failed
    """
    if not template_content:
        return template_content

    # * Create a git context section to add to the template
    git_context = []

    # * Add branch information if available
    if branch_name:
        git_context.append(f"Current branch: {branch_name}")

    # * Add recent commits if available
    if recent_commits and len(recent_commits) > 0:
        git_context.append("\nRecent commits:")

        for commit in recent_commits:
            # * Handle both dict-style commits and git.Commit objects
            if isinstance(commit, dict):
                commit_hash = commit.get("hash", "")[:7] if commit.get("hash") else ""
                commit_date = commit.get("date", "")
                commit_message = commit.get("message", "").strip()
                git_context.append(f"• {commit_hash} ({commit_date}): {commit_message}")
            else:
                # * Assume git.Commit object
                try:
                    commit_hash = commit.hexsha[:7] if hasattr(commit, "hexsha") else ""
                    commit_date = (
                        str(commit.authored_datetime)
                        if hasattr(commit, "authored_datetime")
                        else ""
                    )
                    commit_message = (
                        commit.message.strip() if hasattr(commit, "message") else ""
                    )
                    git_context.append(
                        f"• {commit_hash} ({commit_date}): {commit_message}"
                    )
                except AttributeError:
                    # * Skip commits that don't have the expected attributes
                    continue

    # * If we have context to add
    if git_context:
        context_section = "\n\n## Git Context\n" + "\n".join(git_context)

        # * Look for a specific place to insert it, or add at the end
        if placeholder in template_content:
            return template_content.replace(placeholder, context_section)
        elif content_placeholder and content_placeholder in template_content:
            # * Add before the content if content placeholder exists
            return template_content.replace(
                content_placeholder, f"{context_section}\n\n{content_placeholder}"
            )
        else:
            # * Just append to the end
            return template_content + context_section

    return template_content
