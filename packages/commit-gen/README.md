# commit-gen

A robust, configurable tool for generating high-quality Git commit messages using Large Language Models (LLMs). This tool analyzes the diffs in your repository changes and produces structured, convention-compliant commit messages.

## Features

- Works with various LLM providers (Ollama, LM Studio, etc.)
- Handles large diffs through intelligent chunking
- Customizable commit templates
- Supports conventional commits format
- Optimized token usage
- Flexible configuration system via files, environment variables, and CLI

## Installation

```bash
# From the project root:
pip install -e ./packages/commit-gen
```

## Usage

```bash
# Generate a commit message for your staged changes
commit-gen

# Use a specific model
commit-gen --model llama3

# Specify a different provider
commit-gen --provider lmstudio

# Use a custom configuration file
commit-gen --config path/to/config.json
```

## Configuration

commit-gen uses a hierarchical configuration system with the following precedence (highest to lowest):

1. Command-line arguments
2. Environment variables
3. Configuration files (JSON, TOML)
4. Default values

### Configuration File

You can create either a JSON or TOML configuration file:

**config.json example:**

```json
{
  "llm": {
    "provider": "ollama",
    "model": "codellama",
    "temperature": 0.7,
    "max_tokens": 2000,
    "timeout": 60
  },
  "output": {
    "debug": false,
    "show_reasoning": true
  },
  "git": {
    "auto_apply": false
  },
  "chunking": {
    "min_chunk_size": 50,
    "max_chunk_size": 2000
  }
}
```

**config.toml example:**

```toml
[llm]
provider = "ollama"
model = "codellama"
temperature = 0.7
max_tokens = 2000
timeout = 60

[output]
debug = false
show_reasoning = true

[git]
auto_apply = false

[chunking]
min_chunk_size = 50
max_chunk_size = 2000
```

### Environment Variables

Environment variables are prefixed with `COMMIT_GEN_` and use underscores to separate sections:

```bash
# Set the LLM provider
export COMMIT_GEN_LLM_PROVIDER="ollama"

# Set the model
export COMMIT_GEN_LLM_MODEL="codellama"

# Enable debug output
export COMMIT_GEN_OUTPUT_DEBUG="true"
```

### Command Line Options

| Option                  | Description                               |
| ----------------------- | ----------------------------------------- |
| `--provider`            | LLM provider to use (ollama, lmstudio)    |
| `--model`               | Model name to use                         |
| `--base-url`            | Base URL for the provider API             |
| `--timeout`             | Timeout for API requests (seconds)        |
| `--temperature`         | Temperature setting for the LLM (0.0-1.0) |
| `--max-tokens`          | Maximum tokens for LLM response           |
| `--system-message`      | Custom system message for the LLM         |
| `--commit-template`     | Path to custom commit template            |
| `--prompt-template`     | Path to custom prompt template            |
| `--max-lines-per-chunk` | Maximum lines per chunk for large diffs   |
| `--disable-chunking`    | Process entire diff as one unit           |
| `--debug`               | Enable debug output                       |
| `--show-reasoning`      | Show model reasoning/thought process      |
| `--apply`               | Apply generated message to staged changes |
| `--config`              | Path to a configuration file              |

## Advanced Features

### Intelligent Diff Chunking

For large changes, `commit-gen` uses an advanced chunking strategy to handle diffs that exceed the model's context window. Here's how it works:

#### Dynamic Chunk Sizing

The tool automatically calculates optimal chunk sizes based on:

1. **First chunk size**: Smaller to accommodate the prompt and commit templates
2. **Subsequent chunk sizes**: Larger because they don't need to include the full templates

```txt
┌─────────────────────┐     ┌─────────────────────┐
│  First Chunk        │     │  Subsequent Chunks  │
├─────────────────────┤     ├─────────────────────┤
│ - Prompt Template   │     │ - Simplified Prompt │
│ - Commit Template   │     │ - More Space for    │
│ - Less Space for    │     │   Diff Content      │
│   Diff Content      │     │                     │
└─────────────────────┘     └─────────────────────┘
```

#### Prompt Template Handling

The prompt template is handled differently depending on the chunk:

- **First chunk**: Receives the full prompt template with commit template
- **Subsequent chunks**: Receives a simplified prompt that refers back to the original context

This approach:

- Maintains context across chunks so the model doesn't lose track of the task
- Optimizes token usage by not repeating the same instructions
- Ensures each chunk gets processed with the relevant context
- Balances between context preservation and maximizing space for diff content

The system calculates chunk sizes dynamically based on:

- The model's context window size
- Estimated token count for templates
- Reserved tokens for model response
- A safety margin to prevent context overflow

### Chunking Configuration

You can fine-tune the chunking behavior with these settings:

```json
{
  "chunking": {
    "characters_per_token": 4,
    "default_chunk_size": 200,
    "min_chunk_size": 50,
    "max_chunk_size": 2000,
    "min_available_percent": 10,
    "reserved_response_tokens": 2000,
    "safety_margin_percent": 10
  }
}
```

### Error Handling Strategy

commit-gen implements a dual-layer error handling strategy to accommodate both CLI and API usage:

#### API Layer Error Handling

The core API functions (in `core.py`) use error handlers that return default values instead of raising exceptions:

- Return tuples with `(result, error_message)` pattern for consistent API consumption
- Empty results with descriptive error messages on failure
- No exceptions raised to calling code, making it easier to use in library contexts

#### CLI Layer Error Handling

The CLI layer (in `main.py`) provides user-focused error handling:

- Converts API layer errors into user-friendly messages
- Provides appropriate exit codes for command-line usage
- Indicates success/failure appropriately for shell scripts

This separation ensures that library users have a consistent return value pattern to check, while CLI users get a friendly experience with clear error messages and standard shell behavior.

## Customization

### Custom Templates

You can provide your own templates to customize the commit message format:

```bash
cgen --commit-template path/to/custom-commit-template.md --prompt-template path/to/custom-prompt.md
```

### Template Variables

The prompt template supports the following variables:

- `{{commit_template}}`: Your commit template
- `{{diff}}`: The Git diff content

## How It Works

1. Analyzes the current Git diff
2. Determines if chunking is needed based on size
3. Calculates optimal chunk sizes if necessary
4. Sends chunks to the LLM with appropriate prompts
5. Consolidates responses into a unified commit message
6. Formats the output according to the commit template

## API Usage

As of version 0.3.0, commit-gen can also be used as a Python library in your own code. This allows you to integrate commit message generation into your own applications or workflows.

### Basic API Usage

```python
from commit_gen.core import create_commit_message
from commit_gen.config import load_config

# Get a commit message using default settings
message, reasoning = create_commit_message()

# Print results
print(f"Generated commit message: {message}")
if reasoning:
    print(f"Reasoning: {reasoning}")
```

### Advanced API Usage

```python
from commit_gen.core import create_commit_message, setup_provider_from_config
from commit_gen.config import load_config
from commit_gen.git_utils import get_staged_diff

# Load or create custom configuration
config = load_config()
config["llm"]["provider"] = "ollama"
config["llm"]["model"] = "llama3"

# Get diff content
diff_content = get_staged_diff()

# Use a custom commit template
custom_commit_template = """
type:
scope:
subject:

body:

BREAKING CHANGES:

footer:
"""

# Generate a commit message with custom options
message, reasoning = create_commit_message(
    diff_content=diff_content,
    config_dict=config,
    repo_path=".",
    custom_commit_template=custom_commit_template
)
```

### API Reference

The core API provides several functions:

#### `create_commit_message`

High-level function to generate a commit message with minimal setup.

```python
def create_commit_message(
    diff_content=None,
    config_dict=None,
    repo_path=".",
    custom_commit_template=None,
    custom_prompt_template=None
)
```

- **diff_content**: Git diff content as string, uses staged diff if None
- **config_dict**: Optional configuration dictionary, uses default config if None
- **repo_path**: Path to git repository for diff and context
- **custom_commit_template**: Optional custom commit template content
- **custom_prompt_template**: Optional custom prompt template content
- **Returns**: Tuple of (commit_message, reasoning)

#### `generate_commit_message_for_diff`

Lower-level function for more direct control over the generation process.

```python
def generate_commit_message_for_diff(
    diff_content,
    provider_instance,
    commit_template_content,
    prompt_template_content,
    chunk_size=None,
    disable_chunking=False,
    debug=False
)
```

- **diff_content**: The git diff content as a string
- **provider_instance**: An initialized LLM provider instance
- **commit_template_content**: The commit message template content
- **prompt_template_content**: The prompt template content
- **chunk_size**: Optional maximum number of lines per chunk
- **disable_chunking**: Whether to disable diff chunking
- **debug**: Whether to enable debug mode
- **Returns**: Tuple of (commit_message, reasoning)

#### `setup_provider_from_config`

Creates an LLM provider instance from configuration.

```python
def setup_provider_from_config(config)
```

- **config**: Configuration dictionary
- **Returns**: An initialized LLM provider instance or None

#### `enrich_templates_with_context`

Enriches a prompt template with git context information.

```python
def enrich_templates_with_context(
    prompt_template_content,
    repo_path="."
)
```

- **prompt_template_content**: The original template content
- **repo_path**: Path to the git repository
- **Returns**: The enriched template content

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Todos

Based on the refactoring we've done and reviewing the current state of the codebase, I can offer some thoughts on the commit-gen and llm-connection packages.

### Current Strengths

1. **Modular Design**: The separation between llm-connection and commit-gen creates clean boundaries of responsibility.

2. **Centralized Message Cleanup**: Moving the message cleanup functionality to llm-connection was a good improvement that follows the DRY principle.

3. **Provider Abstraction**: The LLMProvider base class with specific implementations (Ollama, LMStudio) allows for easy extension with new providers.

4. **Smart Chunking**: The diff chunking strategy automatically adapts to model context sizes, which is essential for handling large changes.

5. **Interactive Fallbacks**: The tool has good UX with interactive provider/model selection when automatic choices fail.

6. **Hierarchical Configuration**: The new configuration system provides flexibility with multiple configuration methods and proper precedence.

### Potential Improvements

1. ✅ **Configuration Management**: Implemented a proper config system with hierarchical precedence:

   - Default values
   - Config file values (JSON, TOML)
   - Environment variables
   - Command line arguments

2. **Error Handling**: The error handling is somewhat scattered and inconsistent. A more centralized approach to error handling with proper error classes could improve the robustness.

3. **Async Support**: The current implementation is synchronous, which can be limiting when dealing with multiple models or providers. Adding async support would allow for more efficient processing, especially for chunked operations.

4. **Testing**: While there's a test directory, expanding the test coverage would ensure reliability as the codebase evolves.

5. ✅ **Logging**: The code primarily uses print statements for output. A proper logging system would provide more flexibility for debugging and operation.

6. **Caching**: Consider implementing a caching layer for LLM responses to save tokens during development or when running similar commands repeatedly.

7. ✅ **Progress Feedback**: The chunking process shows a progress bar, but other long-running operations don't provide feedback. Consistent progress indicators would improve the user experience.

8. ✅ **Better Type Hints**: Added comprehensive type hints to improve IDE support and make the code more self-documenting.

9. ✅ **API vs. CLI Separation**: The current code mixes CLI-specific logic with the core functionality. Separating these would make it easier to use the core functionality as a library.

10. **Provider Auto-Configuration**: The provider setup could be more automatic. For example, it could auto-detect the models available in Ollama and offer intelligent defaults.
