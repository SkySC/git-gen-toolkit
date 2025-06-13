"""Message building utilities for LLM providers."""

from .constants import DEFAULT_SYSTEM_MESSAGE


def build_simple_prompt(content, system_message=None):
    """Build a simple text prompt with optional system message prepended.

    Args:
        content: The main content/prompt
        system_message: Optional system message to prepend

    Returns:
        str: The formatted prompt
    """
    if system_message:
        return f"{system_message}\n\n{content}"

    return content


def build_messages_format(content, system_message=DEFAULT_SYSTEM_MESSAGE):
    """Build messages in the chat format used by most modern LLMs.

    Args:
        content: The user message content
        system_message: The system message (instructions for the AI)

    Returns:
        list: List of message objects with role and content
    """
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": content},
    ]


def build_chunked_prompt(chunk_content, chunk_index, total_chunks, content_type="diff"):
    """Build a prompt specifically for handling a chunk in a multi-chunk workflow.

    Args:
        chunk_content: The content of the current chunk
        chunk_index: Current chunk index (0-based)
        total_chunks: Total number of chunks
        content_type: Type of content ("diff" or "commits")

    Returns:
        str: Formatted prompt for the chunk
    """
    if content_type == "diff":
        return f"""This is chunk {chunk_index+1} of {total_chunks} of the same diff.
Based on this additional diff content, analyze it and generate a commit message part:

```diff
{chunk_content}
```

Generate a focused commit message for just this part of the change.
"""
    else:
        return f"""This is chunk {chunk_index+1} of {total_chunks} of the same commit list.
Based on this additional commit information, analyze it and update your PR description:

{chunk_content}

Continue generating a focused PR description for these additional commits.
"""


def build_template_prompt(
    content,
    primary_template_content=None,
    prompt_template_content=None,
    primary_template_key="{{primary_template}}",
    content_key="{{content}}",
):
    """Build a prompt using templates.

    Args:
        content: The main content to include (diff or commit list)
        primary_template_content: The primary template content
        prompt_template_content: The prompt template content
        primary_template_key: The key for the primary template in the prompt template
        content_key: The key for the content in the prompt template

    Returns:
        str: The formatted prompt with templates applied
    """
    if not prompt_template_content:
        return content

    # * Replace template keys
    prompt = prompt_template_content
    if primary_template_content:
        prompt = prompt.replace(primary_template_key, primary_template_content)
    prompt = prompt.replace(content_key, content)

    return prompt
