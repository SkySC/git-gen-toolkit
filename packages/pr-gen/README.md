# PR-Gen

Generate pull request descriptions from git commit history using large language models.

## Prerequisites

- Python 3.7+
- Git

## Installation

```bash
pip install prgen
```

## Usage

```bash
# Basic usage
prgen

# With a specific provider
prgen --provider ollama

# With a specific model
prgen --model llama3

# With a custom template
prgen --template path/to/template.md

# Show version
prgen --version
```

## Features

- Automatically finds the repository and current branch
- Extracts commits specific to the current branch
- Generates a professional PR description using LLMs
- Supports multiple LLM providers (Ollama, LM Studio)
- Customizable templates and prompts

## LLM Providers

### Ollama

```bash
# Default: uses Ollama running on localhost:11434
prgen --provider ollama --model llama3
```

### LM Studio

```bash
prgen --provider lmstudio --model mistral
```

## Templates

The package comes with default templates, but you can provide your own:

```bash
# Use custom templates
prgen --template my-pr-template.md --prompt-template my-prompt.md
```

To see examples of templates, check the templates included in the package.
