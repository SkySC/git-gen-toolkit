"""Message cleanup utilities for LLM-generated content.

This module contains functions for cleaning up and extracting meaningful content
from LLM-generated messages, separating the actual content from explanations,
reasoning, and other metadata.
"""

import re


def extract_tag_content(content, tag_patterns):
    """Extract content enclosed in specific tags.

    Args:
        content: The text content to process
        tag_patterns: List of regex patterns for tags to extract

    Returns:
        tuple: (content with tags removed, extracted tag content)
    """
    if not content:
        return "", ""

    extracted_content = []
    clean_content = content

    # * Process each pattern
    for pattern in tag_patterns:
        # * Find all matches
        matches = re.findall(pattern, clean_content, re.DOTALL)

        if matches:
            for match in matches:
                # * Extract just the content inside the tags
                inner_content = re.sub(r"<[^>]+>([\s\S]*)</[^>]+>", r"\1", match)
                extracted_content.append(inner_content)

            # * Remove the entire tag and its contents from the clean content
            clean_content = re.sub(pattern, "", clean_content)

    return clean_content, "\n\n".join(extracted_content)


def extract_markdown_blocks(content):
    """Extract content from markdown code blocks.

    Args:
        content: The text content to process

    Returns:
        tuple: (extracted block content, surrounding text)
    """
    if not content:
        return "", ""

    # * Look for markdown code blocks with the content
    code_block_pattern = r"```(?:markdown)?\n([\s\S]*?)\n```"
    code_blocks = re.findall(code_block_pattern, content)

    # * If no code blocks found, return original content
    if not code_blocks:
        return "", content

    # * Get everything outside the first code block as surrounding text
    surrounding_text = []
    full_match = re.search(code_block_pattern, content)
    if full_match:
        before_block = content[: full_match.start()].strip()
        after_block = content[full_match.end() :].strip()

        if before_block:
            surrounding_text.append(before_block)
        if after_block:
            surrounding_text.append(after_block)

    # * Use the content of the first code block as the clean content
    return code_blocks[0].strip(), "\n\n".join(surrounding_text)


def extract_by_separators(content):
    """Extract content by splitting on separator lines.

    Args:
        content: The text content to process

    Returns:
        tuple: (main content, other parts)
    """
    if not content:
        return "", ""

    # * Look for separator lines which often indicate explanations
    separator_pattern = r"-{3,}"
    parts = re.split(separator_pattern, content)

    if len(parts) <= 1:
        return content, ""

    # * Try to find the part that looks most like a conventional formatted message
    # * (with [type] for commits or # for PR titles)
    message_parts = [
        part
        for part in parts
        if re.search(r"^\s*\[[a-z]+\]", part, re.MULTILINE)
        or re.search(r"^\s*#\s+\w+", part, re.MULTILINE)
    ]

    other_parts = []
    main_content = ""

    if message_parts:
        # * Use the first part that looks like a formatted message
        main_content = message_parts[0].strip()

        # * Add the rest as other content
        for part in parts:
            if part.strip() != main_content:
                other_parts.append(part.strip())
    else:
        # * If we can't identify a message part, use the longest part
        main_content = max(parts, key=len).strip()

        # * Add the rest as other content
        for part in parts:
            if part.strip() != main_content:
                other_parts.append(part.strip())

    return main_content, "\n\n".join(other_parts)


def extract_content(content):
    """Extract the primary content from LLM output, removing explanations and metadata.

    This function handles multiple formats:
    1. Explicit reasoning/thinking tags
    2. Markdown code blocks
    3. Separator lines

    Args:
        content: The content to clean up

    Returns:
        str: The clean extracted content
    """
    if not content:
        return ""

    # * Try different extraction methods in order of preference
    # * First, try to extract tag content (reasoning, thinking, etc.)
    tag_patterns = [
        r"<think>[\s\S]*?</think>",
        r"<reasoning>[\s\S]*?</reasoning>",
        r"<thoughts>[\s\S]*?</thoughts>",
        r"<thought>[\s\S]*?</thought>",
    ]
    clean_content, _ = extract_tag_content(content, tag_patterns)

    # * Next, try to extract from markdown code blocks
    markdown_content, _ = extract_markdown_blocks(clean_content)
    if markdown_content:
        clean_content = markdown_content
    else:
        # * If no markdown blocks, try separator-based extraction
        clean_content, _ = extract_by_separators(clean_content)

    # * Clean up any excessive newlines
    clean_content = re.sub(r"\n{3,}", "\n\n", clean_content)

    return clean_content.strip()


def extract_reasoning(content):
    """Extract reasoning and metadata from LLM output.

    Args:
        content: The content to process

    Returns:
        str: The extracted reasoning and metadata
    """
    if not content:
        return ""

    reasoning_parts = []

    # * Extract reasoning tags
    tag_patterns = [
        r"<think>[\s\S]*?</think>",
        r"<reasoning>[\s\S]*?</reasoning>",
        r"<thoughts>[\s\S]*?</thoughts>",
        r"<thought>[\s\S]*?</thought>",
    ]
    _, tag_content = extract_tag_content(content, tag_patterns)
    if tag_content:
        reasoning_parts.append(tag_content)

    # * Extract text outside markdown blocks
    _, markdown_surrounding = extract_markdown_blocks(content)
    if markdown_surrounding:
        reasoning_parts.append(markdown_surrounding)

    # * Extract text from separators
    _, separator_parts = extract_by_separators(content)
    if separator_parts:
        reasoning_parts.append(separator_parts)

    return "\n\n".join(reasoning_parts).strip()


def clean_message(content):
    """Clean up a message for final output, separating content from reasoning.

    Args:
        content: The content to clean up

    Returns:
        tuple: (clean_content, reasoning)
    """
    clean_content = extract_content(content)
    reasoning = extract_reasoning(content)

    return clean_content, reasoning
