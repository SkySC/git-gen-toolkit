"""Git utilities for commit-gen."""

from datetime import datetime

from git import GitCommandError, InvalidGitRepositoryError, Repo
from toolkit_utils import GitError, error_handler, handle_error
from toolkit_utils.logging_utils import info, success


def get_staged_diff(repo_path="."):
    """Get the staged diff content.

    Args:
        repo_path: Path to the git repository

    Returns:
        str: The staged diff content, or None if there are no staged changes

    Raises:
        GitError: If there's an error accessing the git repository
    """
    try:
        repo = Repo(repo_path)
        # * Use --staged for staged changes,
        # * or None for all working tree changes vs HEAD
        diff_output = repo.git.diff("--staged")
        if not diff_output:
            # * No staged changes found
            return None

        return diff_output
    except InvalidGitRepositoryError:
        raise GitError(f"Not a valid git repository: {repo_path}")
    except GitCommandError as e:
        raise GitError(f"Git command error: {str(e)}", cause=e)
    except Exception as e:
        raise GitError("Unexpected error getting git diff", cause=e)


@error_handler(message="Failed to get current branch name", default_return=None)
def get_current_branch_name(repo_path="."):
    """Get the name of the current git branch.

    Args:
        repo_path: Path to the git repository

    Returns:
        str: The current branch name, or None if there's an error
    """
    repo = Repo(repo_path)
    return repo.active_branch.name


@error_handler(message="Failed to get recent commits", default_return=[])
def get_recent_commits(repo_path=".", count=5):
    """Get recent commits from the current branch.

    Args:
        repo_path: Path to the git repository
        count: Number of recent commits to retrieve

    Returns:
        list: List of dicts with commit information, or empty list if there's an error
    """
    repo = Repo(repo_path)
    commits = []

    for commit in repo.iter_commits(max_count=count):
        # * Format the commit date nicely
        commit_date = datetime.fromtimestamp(commit.committed_date).strftime("%Y-%m-%d")

        # * Get the short commit hash
        short_hash = repo.git.rev_parse(commit.hexsha, short=True)

        # * Add to our list of commits
        commits.append(
            {
                "hash": short_hash,
                "message": commit.message.strip().split("\n")[0],  # ? Just first line
                "author": str(commit.author),
                "date": commit_date,
            }
        )

    return commits


def apply_commit_message(message, repo_path=".", debug=False):
    """Apply the generated commit message to the repository.

    Args:
        message: The commit message to apply
        repo_path: Path to the git repository
        debug: Whether to print debug information

    Returns:
        bool: True if the commit was applied successfully, False otherwise
    """
    try:
        repo = Repo(repo_path)

        # * Ask for confirmation
        info("\nDo you want to apply this commit message? (y/n)")
        choice = input().lower().strip()

        if choice.startswith("y"):
            # * Apply the commit message
            repo.git.commit("-m", message)
            success("Commit applied successfully!")
            return True
        else:
            info("Commit message not applied.")
            return False
    except Exception as e:
        handle_error(e, message="Error applying commit message", debug=debug)
        return False
