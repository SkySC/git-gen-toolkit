"""Template utilities for commit-gen."""

from toolkit_utils import enrich_template_with_git_context as enrich_template
from toolkit_utils import error_handler, load_template

from .constants import DEFAULT_COMMIT_TEMPLATE_PATH, DEFAULT_PROMPT_TEMPLATE_PATH


@error_handler(message="Error loading commit template", default_return=None)
def load_commit_template(template_path):
    """Load commit template from file or use default.

    Args:
        template_path: Path to the commit template file

    Returns:
        str: The commit template content

    Raises:
        TemplateError: If the template can't be loaded
    """
    return load_template(
        template_path=template_path,
        default_template_path=DEFAULT_COMMIT_TEMPLATE_PATH,
        template_type="commit",
    )


@error_handler(message="Error loading prompt template", default_return=None)
def load_prompt_template(prompt_template_path):
    """Load prompt template from file.

    Args:
        prompt_template_path: Path to the prompt template file

    Returns:
        str: The prompt template content

    Raises:
        TemplateError: If the template can't be loaded
    """
    return load_template(
        template_path=prompt_template_path,
        default_template_path=DEFAULT_PROMPT_TEMPLATE_PATH,
        template_type="prompt",
    )


@error_handler(message="Error enriching template with Git context", default_return=None)
def enrich_template_with_git_context(
    template_content, branch_name=None, recent_commits=None
):
    """Enrich the prompt template with git context information.

    Args:
        template_content: The original template content
        branch_name: The current branch name
        recent_commits: List of recent commits

    Returns:
        str: The enriched template content
    """
    return enrich_template(
        template_content=template_content,
        branch_name=branch_name,
        recent_commits=recent_commits,
        content_placeholder="{{diff}}",
    )
