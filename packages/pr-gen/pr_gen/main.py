#!/usr/bin/env python3

import argparse
import os
import os.path
import re
import sys
from datetime import datetime

import questionary
from git import Repo
from llm_connection.providers import LMStudioProvider, OllamaProvider
from toolkit_utils.logging_utils import setup_logging
from tqdm import tqdm

# Import constants from constants module
from .constants import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LLM_TIMEOUT,
    DEFAULT_LMSTUDIO_BASE_URL,
    DEFAULT_LMSTUDIO_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_PR_TEMPLATE,
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_SYSTEM_MESSAGE,
    MAX_CHUNK_SIZE,
    MIN_CHUNK_SIZE,
    get_default_template_path,
)

# --- Helper Functions ---


def load_template(template_path):
    """Load template from file if it exists, otherwise use default template."""
    if not template_path:
        return DEFAULT_PR_TEMPLATE

    if not os.path.exists(template_path):
        print(f"Template file not found: {template_path}")
        print("Using default template instead.")
        return DEFAULT_PR_TEMPLATE

    try:
        with open(template_path, "r") as file:
            return file.read()
    except Exception as e:
        print(f"Error reading template file {template_path}: {e}")
        print("Using default template instead.")
        return DEFAULT_PR_TEMPLATE


def load_prompt_template(prompt_template_path):
    """Load prompt template from file if it exists, otherwise use default template."""
    if not prompt_template_path:
        return DEFAULT_PROMPT_TEMPLATE

    if not os.path.exists(prompt_template_path):
        print(f"Prompt template file not found: {prompt_template_path}")
        print("Using default prompt template instead.")
        return DEFAULT_PROMPT_TEMPLATE

    try:
        with open(prompt_template_path, "r") as file:
            return file.read()
    except Exception as e:
        print(f"Error reading prompt template file {prompt_template_path}: {e}")
        print("Using default prompt template instead.")
        return DEFAULT_PROMPT_TEMPLATE


def generate_pr_description(
    provider_instance,
    commits,
    branch_name,
    template_content,
    prompt_template_content,
):
    """Generate a PR description using the LLM provider."""
    # * First check if the model is loaded before proceeding
    if not provider_instance.is_model_loaded():
        print(f"\nERROR: Model '{provider_instance.model}' is not loaded.")
        print("Please load the model in the provider UI before continuing.")
        return None

    commit_text = "\n".join([f"- {c.hexsha[:7]}: {c.message.strip()}" for c in commits])

    # * Check if we need to split commits into chunks (for very large PRs)
    commit_lines = len(commits)
    total_chars = len(commit_text)

    # * If we have a lot of commits, consider chunking
    if commit_lines > 50 or total_chars > 5000:
        # * Simple chunking for large commit lists
        return process_commits_in_chunks(
            provider_instance,
            commits,
            branch_name,
            template_content,
            prompt_template_content,
        )
    else:
        # * Standard processing for normal-sized commit lists
        # * Replace placeholders in the prompt template content
        prompt = prompt_template_content.replace("{{branch_name}}", branch_name)
        prompt = prompt.replace("{{pr_template}}", template_content)
        prompt = prompt.replace("{{commit_list}}", commit_text)

        return provider_instance._call_api(prompt)


def process_commits_in_chunks(
    provider_instance,
    commits,
    branch_name,
    pr_template_content,
    prompt_template_content,
    wait_for_model_load=True,
    max_load_retries=3,
):
    """Process large commit lists in chunks using template caching.

    Args:
        provider_instance: The LLM provider instance
        commits: List of commit objects
        branch_name: Name of the branch
        pr_template_content: PR template content
        prompt_template_content: Prompt template content
        wait_for_model_load: If True, wait for models to load rather than immediately failing
        max_load_retries: Maximum number of retries when waiting for model to load

    Returns:
        Generated PR description or None if failed
    """
    # * Check if model is loaded before starting
    model_loaded = provider_instance.is_model_loaded()
    retry_count = 0

    # * If model isn't loaded but we're configured to wait, retry a few times
    while not model_loaded and wait_for_model_load and retry_count < max_load_retries:
        print(
            f"\nWaiting for model '{provider_instance.model}' to load... (attempt"
            f" {retry_count+1}/{max_load_retries})"
        )
        print("This might take some time for larger models.")

        # * Wait longer with each retry
        import time

        time.sleep(10 + retry_count * 5)  # * Progressive backoff: 10s, 15s, 20s...
        model_loaded = provider_instance.is_model_loaded()
        retry_count += 1

    if not model_loaded:
        print(
            f"\nERROR: Model '{provider_instance.model}' is not loaded after"
            f" {max_load_retries} attempts."
        )
        print(
            "Please ensure the model is properly loaded in the provider UI before"
            " continuing."
        )
        print(
            "Note: For LM Studio and Ollama, a test request has been sent which may"
            " have"
        )
        print("      triggered the model to start loading in the background.")
        return None

    # * Split commits into chunks of reasonable size
    chunk_size = 30  # ? Note: Adjust based on typical commit message size
    commit_chunks = [
        commits[i : i + chunk_size] for i in range(0, len(commits), chunk_size)
    ]

    total_chunks = len(commit_chunks)

    # * Create progress bar for multiple chunks
    chunk_progress = tqdm(
        total=total_chunks,
        desc="Processing commits in chunks",
        unit="chunk",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} chunks [{elapsed}<{remaining}]",
    )

    result = None

    for i, chunk in enumerate(commit_chunks):
        # * Verify the model is still loaded before each chunk processing
        if i > 0:
            # * For subsequent chunks, check model state but don't wait as long
            model_loaded = provider_instance.is_model_loaded()
            retry_count = 0

            # * Allow one retry for continued processing
            if not model_loaded and wait_for_model_load:
                print(
                    f"\nWaiting for model '{provider_instance.model}' to be ready"
                    " again..."
                )
                import time

                time.sleep(5)
                model_loaded = provider_instance.is_model_loaded()

            if not model_loaded:
                print(
                    f"\nERROR: Model '{provider_instance.model}' is no longer loaded."
                )
                print("Returning partial results generated so far.")
                chunk_progress.close()
                return result

        # * Create commit text for this chunk
        commit_text = "\n".join(
            [f"- {c.hexsha[:7]}: {c.message.strip()}" for c in chunk]
        )

        # * Update progress description to show current chunk
        chunk_progress.set_description(f"Processing chunk {i+1}/{total_chunks}")

        # * For first chunk, use the full template approach
        if hasattr(provider_instance, "call_api_with_templates"):
            # * Create branch context only for the first chunk
            context_prompt = prompt_template_content.replace(
                "{{branch_name}}", branch_name
            )

            # * Use the generic template caching API
            chunk_result = provider_instance.call_api_with_templates(
                content=commit_text,
                primary_template_content=pr_template_content,
                prompt_template_content=context_prompt,
                chunk_index=i,
                total_chunks=total_chunks,
                primary_template_key="{{pr_template}}",
                content_key="{{commit_list}}",
            )
        else:
            # * Fall back to standard approach if template caching not available
            prompt = prompt_template_content.replace("{{branch_name}}", branch_name)
            prompt = prompt.replace("{{pr_template}}", pr_template_content)
            prompt = prompt.replace("{{commit_list}}", commit_text)
            chunk_result = provider_instance._call_api(prompt)

        # * For first chunk, just save the result
        if i == 0:
            result = chunk_result
        else:
            # * For subsequent chunks, ask the model to update/improve the previous result
            consolidation_prompt = f"""You previously created a PR description based on some commits.
Here are additional commits from the same branch that weren't included in the original analysis.
Please update your PR description to incorporate insights from these new commits:

Additional commits:
{commit_text}

Your previous PR description:
{result}

Please provide an updated, comprehensive PR description that covers all commits.
"""
            result = provider_instance._call_api(consolidation_prompt)

        # * Update the progress bar
        chunk_progress.update(1)

    # * Close the progress bar
    chunk_progress.close()

    return result


def save_pr_description(content, branch_name):
    """Save PR description to a Markdown file."""
    timestamp = datetime.now().strftime("%Y-%m-%d")
    branch_name_cleaned = branch_name.replace("/", "-")
    filename = f"PR_{branch_name_cleaned}_{timestamp}.md"

    with open(filename, "w") as f:
        f.write(content)

    print(f"PR description saved to: {filename}")
    return filename


def extract_reasoning(content):
    """Extract reasoning tags from content and return clean content and reasoning."""
    # * Check if there's any content to process
    if not content:
        return "", ""

    # * Improved patterns using [\s\S]*? instead of .*? to match across lines
    patterns = [
        (r"<think>[\s\S]*?</think>", ""),
        (r"<reasoning>[\s\S]*?</reasoning>", ""),
        (r"<thoughts>[\s\S]*?</thoughts>", ""),
        (r"<thought>[\s\S]*?</thought>", ""),
    ]

    reasoning = []
    clean_content = content

    # * Process each pattern
    for pattern, _ in patterns:
        # * Find all matches
        matches = re.findall(pattern, clean_content, re.DOTALL)

        if matches:
            for match in matches:
                # * Extract just the content inside the tags
                inner_content = re.sub(r"<[^>]+>([\s\S]*)</[^>]+>", r"\1", match)
                reasoning.append(inner_content)

            # * Remove the entire tag and its contents from the clean content
            clean_content = re.sub(pattern, "", clean_content)

    # * Clean up any excessive newlines from tag removal
    clean_content = re.sub(r"\n{3,}", "\n\n", clean_content)

    return clean_content.strip(), "\n\n".join(reasoning).strip()


# --- Git Helper Functions ---


def find_repo():
    cwd = os.getcwd()
    # * Check if a '.git' folder exists in the current working directory
    if os.path.exists(os.path.join(cwd, ".git")):
        return cwd
    else:
        print("This is not a git repository")
        sys.exit(1)


def find_current_branch_name(repo_path):
    repo = Repo(repo_path)
    # * Check if the repository is in a 'detached HEAD' state
    if repo.head.is_detached:
        print("The repository is in a 'detached HEAD' state")
        sys.exit(1)
    else:
        return repo.active_branch.name


def find_diverging_point(repo_path, branch_name):
    """
    Find the best common ancestor commit (merge base) between the current branch and
    potential parent branches.
    """
    repo = Repo(repo_path)
    current_commit = repo.head.commit

    # * Candidates for the parent branch
    parent_candidates = ["main", "master", "development"]
    potential_parents = []
    for name in parent_candidates:
        try:
            if name in repo.refs and repo.refs[name].commit != current_commit:
                potential_parents.append(repo.refs[name].commit)
        except (KeyError, AttributeError):
            continue

    # * If no standard parents found, consider other branches
    if not potential_parents:
        for branch in repo.branches:
            if branch.name != branch_name and branch.commit != current_commit:
                potential_parents.append(branch.commit)

    if not potential_parents:
        print("Could not find any potential parent branches to compare against.")
        sys.exit(1)

    try:
        # * Find the merge base(s) between the current commit and potential parents
        merge_bases = repo.merge_base(current_commit, *potential_parents)
        if not merge_bases:
            print(
                "Could not find a common ancestor (merge base). Are branches unrelated?"
            )
            sys.exit(1)

        # * Typically, there's one merge base. If multiple, the first is sufficient.
        diverging_commit = merge_bases[0]
        print(f"Found diverging point (merge base): {diverging_commit.hexsha[:7]}")
        return diverging_commit.hexsha
    except Exception as e:
        print(f"Error finding merge base: {e}")
        sys.exit(1)


def iterate_commits(repo_path, branch_name, args):
    repo = Repo(repo_path)

    # * Find the actual diverging point commit hash
    diverging_point_hash = find_diverging_point(repo_path, branch_name)

    try:
        # * Use the diverging point hash in the rev-list command
        commits = list(
            repo.git.rev_list(
                f"{diverging_point_hash}..{branch_name}",
                "--first-parent",
                "--no-merges",
            ).splitlines()
        )

        # * Convert commit hashes to commit objects
        commit_objects = [repo.commit(commit_hash) for commit_hash in commits]

        # * Exclude merge commits (commits with multiple parents)
        commit_objects = [c for c in commit_objects if len(c.parents) <= 1]

        # * Print the commit count
        print(
            "Found"
            f" {len(commit_objects)} commit{'s' if len(commit_objects) != 1 else ''} in"
            f" branch '{branch_name}'"
        )

        # * Print each commit's information if debug mode is enabled
        if args.debug:
            for commit in commit_objects:
                print_commit(commit)

        return commit_objects
    except Exception as e:
        print(f"Error getting branch-specific commits: {e}")
        return []


def print_commit(commit):
    # * Strip trailing whitespace from the commit message
    message = commit.message.rstrip()
    print(
        f"""---
Commit: {commit.hexsha}
Message: "{message}"
Date: {commit.authored_datetime}
"""
    )


def debug_info():
    import subprocess

    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")

    try:
        subprocess.run(["which", "python3"], check=True)
    except Exception as e:
        print(f"Error running 'which python3': {e}")


def select_commits_interactively(commits):
    """Prompt the user to interactively select which commits to include."""
    choices = [
        questionary.Choice(
            title=f"{c.hexsha[:7]} - {c.message.splitlines()[0]}",
            value=c,
            checked=True,
        )
        for c in commits
    ]
    selected = questionary.checkbox("Select commits to include:", choices=choices).ask()

    return selected or []


# --- Main Execution Logic ---


def main():
    from . import __version__

    parser = argparse.ArgumentParser(
        description="Generate PR descriptions from git commits."
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show the version and exit",
    )
    parser.add_argument(
        "--model",
        default="llama3",
        help="Specify the model to use (default: llama3)",
    )
    parser.add_argument(
        "--provider",
        default="ollama",
        choices=["ollama", "lmstudio", "none"],
        help="LLM provider to use (default: ollama)",
    )
    parser.add_argument(
        "--base-url",
        help="Base URL for the API (default depends on provider)",
    )
    parser.add_argument(
        "--template",
        default=get_default_template_path("pr-template.md"),
        help="Path to a custom PR template file",
    )
    parser.add_argument(
        "--prompt-template",
        default=get_default_template_path("prompt-template.md"),
        help="Path to a custom prompt template file",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Request timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--select",
        "-s",
        action="store_true",
        help="Interactively select which commits to include",
    )
    parser.add_argument(
        "--show-reasoning",
        action="store_true",
        help="Show model reasoning/thought process if present",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM temperature setting (default: 0.7)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=-1,
        help="LLM max tokens setting (default: -1)",
    )
    parser.add_argument(
        "--system-message",
        type=str,
        default=(
            "You are a helpful AI assistant specializing in creating"
            " professional pull request descriptions."
        ),
        help="System message/prompt for the LLM",
    )

    args = parser.parse_args()  # * Setup logging based on debug flag
    setup_logging(level="debug" if args.debug else "info")

    try:
        if args.debug:
            debug_info()

        repo = find_repo()
        print(f"Found git repository at: {repo}")

        branch = find_current_branch_name(repo)
        print(f"Current branch: {branch}")

        commits = iterate_commits(repo, branch, args)
        if args.select:
            commits = select_commits_interactively(commits)

        if args.provider == "none" or not commits:
            print("Skipping LLM generation.")
            return 0

        pr_template_content = load_template(args.template)
        prompt_template_content = load_prompt_template(args.prompt_template)

        provider_options = {
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "system_message": args.system_message,
        }
        if provider_options["max_tokens"] == -1:
            del provider_options["max_tokens"]

        provider_instance = None
        if args.provider == "ollama":
            base_url = args.base_url or "http://localhost:11434"
            provider_instance = OllamaProvider(
                model=args.model,
                base_url=base_url,
                timeout=args.timeout,
                template_path=args.template,
                prompt_template_path=args.prompt_template,
                **provider_options,
            )
        elif args.provider == "lmstudio":
            base_url = args.base_url or "http://localhost:1234"
            provider_instance = LMStudioProvider(
                model=args.model,
                base_url=base_url,
                timeout=args.timeout,
                template_path=args.template,
                prompt_template_path=args.prompt_template,
                **provider_options,
            )

        if provider_instance:
            print(f"\nGenerating PR description using {provider_instance.name}...")
            pr_content = generate_pr_description(
                provider_instance,
                commits,
                branch,
                pr_template_content,
                prompt_template_content,
            )

            if pr_content:
                if args.debug:
                    print(f"Original content length: {len(pr_content)}")

                clean_content, reasoning = extract_reasoning(pr_content)

                if args.debug:
                    print(f"Cleaned content length: {len(clean_content)}")

                if args.show_reasoning and reasoning:
                    print("\n=== Model Reasoning ===")
                    print(reasoning)
                    print("=== End of Reasoning ===\n")

                filename = save_pr_description(clean_content, branch)
                print(f"You can view your PR description with: open {filename}")
            else:
                print("Failed to generate PR description.")
                return 1
        else:
            print(f"Unknown provider: {args.provider}")
            return 1
    except KeyboardInterrupt:
        print("\nOperation cancelled by user. Exiting gracefully.")
        return 1
    except Exception as e:
        if args.debug:
            import traceback

            traceback.print_exc()
        else:
            print(f"Error: {str(e)}")

        return 1

    return 0


if __name__ == "__main__":
    if __package__ is None:
        from pathlib import Path

        file = Path(__file__).resolve()
        parent = file.parent.parent
        sys.path.append(str(parent))

        __package__ = "prgen"

    main()
