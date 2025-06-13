# LLM Connection Package

This package provides shared connection logic for interacting with various Large Language Model (LLM) providers.

It contains base classes and specific implementations (e.g., for Ollama, LM Studio) that other packages within the `git-gen-toolkit` monorepo can use to make API calls to LLMs. This centralizes the connection details and allows for consistent LLM interaction across different tools like `pr-gen` and `commit-gen`.
