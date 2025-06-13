"""Chunking utilities for commit-gen.

This module contains functions for splitting large diffs into manageable chunks
for processing by LLMs with limited context windows.
"""

from llm_connection.message_cleanup import clean_message
from toolkit_utils import ChunkingError, error_handler
from toolkit_utils.logging_utils import debug, error, info
from toolkit_utils.progress_utils import progress_bar, spinner

from .config import get_config

# * Get chunking configuration from the config system
config = get_config()
chunking_config = config["chunking"]
CHARACTERS_PER_TOKEN = chunking_config["characters_per_token"]
DEFAULT_CHUNK_SIZE = chunking_config["default_chunk_size"]
MIN_CHUNK_SIZE = chunking_config["min_chunk_size"]
MAX_CHUNK_SIZE = chunking_config["max_chunk_size"]
MIN_AVAILABLE_PERCENT = chunking_config["min_available_percent"]
RESERVED_RESPONSE_TOKENS = chunking_config["reserved_response_tokens"]
SAFETY_MARGIN_PERCENT = chunking_config["safety_margin_percent"]


def estimate_token_count(text):
    """Roughly estimate token count based on characters.

    This is a very rough estimate: ~4 chars = 1 token for code/text.
    A more accurate estimation would require a tokenizer from the LLM's model.

    Args:
        text: The text to estimate tokens for

    Returns:
        int: Estimated token count
    """
    if not text:
        return 0

    return len(text) // CHARACTERS_PER_TOKEN


@error_handler(message="Error chunking diff content", default_return=[])
def chunk_diff(diff_content, lines_per_chunk=200):
    """Split a large diff into manageable chunks based on line count.

    Args:
        diff_content: The diff content to split
        lines_per_chunk: Maximum number of lines per chunk

    Returns:
        list: A list of diff chunks
    """
    if not diff_content:
        return []

    # * Split the diff into lines
    diff_lines = diff_content.splitlines()
    total_lines = len(diff_lines)

    # * If the diff is small enough, return it as a single chunk
    if total_lines <= lines_per_chunk:
        return [diff_content]

    # * Calculate how many chunks we need
    chunk_count = (total_lines + lines_per_chunk - 1) // lines_per_chunk
    chunks = []

    for i in range(chunk_count):
        start_idx = i * lines_per_chunk
        end_idx = min((i + 1) * lines_per_chunk, total_lines)

        # * Create a chunk with context information
        chunk_header = (
            f"DIFF CHUNK {i+1} OF {chunk_count} (lines {start_idx+1}-{end_idx} of"
            f" {total_lines})\n\n"
        )
        chunk_content = "\n".join(diff_lines[start_idx:end_idx])
        chunks.append(chunk_header + chunk_content)

    return chunks


@error_handler(
    message="Error calculating optimal chunk size", default_return=DEFAULT_CHUNK_SIZE
)
def calculate_optimal_chunk_size(
    provider_instance,
    prompt_template_content=None,
    commit_template_content=None,
    debug=False,
    include_templates=True,
):
    """Calculate the optimal chunk size based on the model's context window.

    This calculation accounts for:
    1. The model's total context window
    2. Tokens used by the prompt template (only if include_templates=True)
    3. Tokens used by the commit template (only if include_templates=True)
    4. Reserved tokens for the model's response
    5. A safety margin

    Args:
        provider_instance: The LLM provider instance
        prompt_template_content: The content of the prompt template
        (only required if include_templates=True)
        commit_template_content: The content of the commit template
        (only required if include_templates=True)
        debug: Whether to print debug information
        include_templates: Whether to include template tokens in calculation
        (set to False for subsequent chunks that don't include templates)

    Returns:
        int: The recommended number of lines per chunk
    """
    # * Get the context size from the provider
    context_size = provider_instance.get_context_size()

    if debug:
        debug(f"Retrieved context size from model: {context_size} tokens")

    # * Calculate tokens used by templates (if needed)
    template_tokens = 0
    if include_templates:
        if prompt_template_content is None or commit_template_content is None:
            raise ChunkingError("Template content required when include_templates=True")

        # * First, create a sample prompt with the templates but empty diff
        sample_prompt = prompt_template_content.replace(
            "{{commit_template}}", commit_template_content
        )
        sample_prompt = sample_prompt.replace("{{diff}}", "")
        template_tokens = estimate_token_count(sample_prompt)

    # * Reserve tokens for the response (rough estimate)
    response_tokens = RESERVED_RESPONSE_TOKENS

    # * Add a safety margin (percentage of context)
    safety_margin = context_size * (SAFETY_MARGIN_PERCENT / 100)

    # * Calculate available tokens for diff content
    available_tokens = context_size - template_tokens - response_tokens - safety_margin

    # * Ensure we have at least some tokens available
    available_tokens = max(
        available_tokens, context_size * (MIN_AVAILABLE_PERCENT / 100)
    )

    # * Estimate lines based on tokens (rough estimate based on chars per token)
    avg_chars_per_line = 100  # * Approximate average
    tokens_per_line = avg_chars_per_line / CHARACTERS_PER_TOKEN
    lines_per_chunk = int(available_tokens / tokens_per_line)

    if debug:
        debug(f"Template tokens: ~{template_tokens}")
        debug(f"Reserved for response: ~{response_tokens}")
        debug(f"Safety margin: ~{int(safety_margin)}")
        debug(f"Available tokens for diff: ~{int(available_tokens)}")
        debug(f"Calculated optimal lines per chunk: {lines_per_chunk}")

    # * Set reasonable bounds
    lines_per_chunk = max(MIN_CHUNK_SIZE, min(MAX_CHUNK_SIZE, lines_per_chunk))

    return lines_per_chunk


@error_handler(message="Error creating dynamic diff chunks", default_return=[])
def dynamic_chunk_diff(diff_content, first_chunk_size, subsequent_chunk_size):
    """
    Split a large diff into chunks with different sizes for first vs subsequent chunks.

    Args:
        diff_content: The diff content to split
        first_chunk_size: Maximum lines for the first chunk (smaller due to templates)
        subsequent_chunk_size: Maximum lines for subsequent chunks (larger)

    Returns:
        list: A list of diff chunks with optimized sizes
    """
    if not diff_content:
        return []

    # * Split the diff into lines
    diff_lines = diff_content.splitlines()
    total_lines = len(diff_lines)

    # * Ensure the chunk sizes are within acceptable limits
    first_chunk_size = max(MIN_CHUNK_SIZE, min(MAX_CHUNK_SIZE, first_chunk_size))
    subsequent_chunk_size = max(
        MIN_CHUNK_SIZE, min(MAX_CHUNK_SIZE, subsequent_chunk_size)
    )

    # * If the diff fits in the first chunk, return it as a single chunk
    if total_lines <= first_chunk_size:
        return [diff_content]

    # * Initialize chunks list and calculate how many we'll need
    chunks = []
    remaining_lines = total_lines - first_chunk_size
    # * Add 1 for the first chunk + the number of subsequent chunks needed
    chunk_count = (
        1 + (remaining_lines + subsequent_chunk_size - 1) // subsequent_chunk_size
    )

    # * Create the first chunk (smaller size due to templates)
    first_chunk_header = (
        f"DIFF CHUNK 1 OF {chunk_count} (lines 1-{first_chunk_size} of"
        f" {total_lines})\n\n"
    )
    first_chunk_content = "\n".join(diff_lines[:first_chunk_size])
    chunks.append(first_chunk_header + first_chunk_content)

    # * Create subsequent chunks (larger size since no templates)
    for i in range(1, chunk_count):
        start_idx = first_chunk_size + (i - 1) * subsequent_chunk_size
        end_idx = min(first_chunk_size + i * subsequent_chunk_size, total_lines)

        chunk_header = (
            f"DIFF CHUNK {i+1} OF {chunk_count} (lines {start_idx+1}-{end_idx} of"
            f" {total_lines})\n\n"
        )
        chunk_content = "\n".join(diff_lines[start_idx:end_idx])
        chunks.append(chunk_header + chunk_content)

    return chunks


@error_handler(message="Error handling diff chunking", default_return=[])
def handle_diff_chunking(
    diff, args, provider_instance, commit_template_content, prompt_template_content
):
    """Handle chunking of the diff content based on command-line arguments.

    Args:
        diff: The diff content to chunk
        args: Command-line arguments
        provider_instance: The LLM provider instance
        commit_template_content: The commit template content
        prompt_template_content: The prompt template content

    Returns:
        list: A list of diff chunks
    """
    if args.disable_chunking:
        if args.debug:
            debug("Chunking disabled. Processing entire diff as one unit.")
        return [diff]

    # * Determine if we need to chunk at all based on estimated tokens
    diff_lines = len(diff.splitlines())
    estimated_tokens = estimate_token_count(diff)

    # * For very small diffs, skip chunking entirely
    if diff_lines < 100 or estimated_tokens < 1000:
        if args.debug:
            debug(
                f"Diff is small ({diff_lines} lines, ~{estimated_tokens} tokens),"
                " processing as a single chunk."
            )
        return [diff]

    # * Get user-specified chunk size if provided
    user_lines_per_chunk = args.max_lines_per_chunk

    # * If user specifies a chunk size, use that for all chunks
    if user_lines_per_chunk is not None:
        if args.debug:
            debug(
                f"Using user-specified chunk size: {user_lines_per_chunk} lines per"
                " chunk"
            )
        diff_chunks = chunk_diff(diff, user_lines_per_chunk)
    else:
        # * Otherwise, use optimized dynamic chunking strategy
        # * Calculate first chunk size with templates included
        first_chunk_size = calculate_optimal_chunk_size(
            provider_instance,
            prompt_template_content,
            commit_template_content,
            args.debug,
            include_templates=True,
        )

        # * Calculate subsequent chunk size without templates
        # * No need to pass the template parameters when not using them
        subsequent_chunk_size = calculate_optimal_chunk_size(
            provider_instance, debug=args.debug, include_templates=False
        )

        if args.debug:
            debug("Optimized chunking strategy:")
            debug(f"  First chunk size: {first_chunk_size} lines (including templates)")
            debug(
                f"  Subsequent chunk size: {subsequent_chunk_size} lines (without"
                " templates)"
            )

        # * Custom chunking with different sizes for first vs. subsequent chunks
        diff_chunks = dynamic_chunk_diff(diff, first_chunk_size, subsequent_chunk_size)

    # * Print debug information if requested
    if args.debug:
        debug(f"Total diff: {diff_lines} lines, ~{estimated_tokens} tokens")
        debug(f"Split into {len(diff_chunks)} chunks")

        for i, chunk in enumerate(diff_chunks):
            chunk_tokens = estimate_token_count(chunk)
            debug(
                f"  Chunk {i+1}: {len(chunk.splitlines())} lines,"
                f" ~{chunk_tokens} tokens"
            )

    return diff_chunks


@error_handler(message="Error processing diff chunks", default_return=None)
def process_chunks(
    provider_instance,
    diff_chunks,
    commit_template_content,
    prompt_template_content,
    debug=False,
    generate_commit_message_func=None,
):
    """Process each diff chunk with the LLM and collect the results.

    Args:
        provider_instance: The LLM provider instance
        diff_chunks: A list of diff chunks to process
        commit_template_content: The commit template content
        prompt_template_content: The prompt template content
        debug: Whether to print debug information
        generate_commit_message_func: Function to generate commit message

    Returns:
        The generated commit message or None if failed
    """
    if not diff_chunks:
        return None

    # * If there's only one chunk, process it directly
    if len(diff_chunks) == 1:
        if debug:
            debug(
                "Processing single diff chunk"
                f" ({len(diff_chunks[0].splitlines())} lines)"
            )

        # * Use the standard approach for a single chunk
        return generate_commit_message_func(
            provider_instance,
            diff_chunks[0],
            commit_template_content,
            prompt_template_content,
        )

    # * Process multiple chunks with template caching optimization
    partial_results = []
    total_chunks = len(diff_chunks)

    # * Create progress bar for multiple chunks using our utility
    with progress_bar(
        total=total_chunks,
        desc="Processing diff chunks",
        unit="chunk",
    ) as chunk_progress:
        for i, chunk in enumerate(diff_chunks):
            if debug:
                debug(
                    f"\nProcessing diff chunk {i+1} of"
                    f" {total_chunks} ({len(chunk.splitlines())} lines)"
                )
            else:
                # * Update the progress bar description to show current chunk
                chunk_progress.set_description(f"Processing chunk {i+1}/{total_chunks}")

            # * Generate a commit message for this chunk
            try:
                # * Generate commit message for this chunk
                chunk_message = generate_commit_message_func(
                    provider_instance,
                    chunk,
                    commit_template_content,
                    prompt_template_content,
                )

                if chunk_message:
                    # * Extract reasoning if present
                    clean_message_content, _ = clean_message(chunk_message)
                    partial_results.append(clean_message_content)

                    if debug:
                        debug(f"Generated partial commit message for chunk {i+1}:")
                        debug(clean_message_content)
                else:
                    error(f"Warning: Failed to generate commit message for chunk {i+1}")

            except Exception as e:
                error(f"Error processing chunk {i+1}: {e}")
                error(f"Warning: Failed to generate commit message for chunk {i+1}")

            # * Update the progress bar
            chunk_progress.update(1)

    # * If we have multiple results, consolidate them
    if len(partial_results) > 1:
        with spinner("Consolidating partial commit messages") as s:
            result = consolidate_commit_messages(
                provider_instance,
                partial_results,
                commit_template_content,
                prompt_template_content,
            )
            s.succeed("Consolidated commit messages")
            return result
    elif partial_results:
        return partial_results[0]
    else:
        return None


@error_handler(message="Error consolidating commit messages", default_return=None)
def consolidate_commit_messages(
    provider_instance,
    partial_messages,
    commit_template_content,
    prompt_template_content,
):
    """Combine multiple partial commit messages into a coherent final message.

    Args:
        provider_instance: The LLM provider instance
        partial_messages: List of partial commit messages to consolidate
        commit_template_content: The commit template content
        prompt_template_content: The prompt template content

    Returns:
        str: The consolidated commit message
    """
    info("\nConsolidating partial commit messages using the prompt template...")

    # * Create a "diff-like" content from the partial messages
    consolidated_input = "# CONSOLIDATED PARTIAL COMMIT MESSAGES\n\n"
    for i, message in enumerate(partial_messages):
        consolidated_input += f"## Partial Commit Message {i+1}\n\n{message}\n\n"

    # * Use the standard prompt template with our consolidated input instead of the diff
    prompt = prompt_template_content.replace(
        "{{commit_template}}", commit_template_content
    )
    prompt = prompt.replace("{{diff}}", consolidated_input)

    # * Call the LLM with the prompt template
    return provider_instance._call_api(prompt)
