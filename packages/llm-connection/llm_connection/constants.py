"""Constants for LLM connection package."""

# * Default timeout values
DEFAULT_API_TIMEOUT = 300  # ? seconds
API_CHECK_TIMEOUT = 2  # ? seconds for availability checks
MODEL_CHECK_TIMEOUT = 5  # ? seconds for model checking
MODEL_LOAD_TIMEOUT = 30  # ? seconds to wait for model loading

# * Default URL values
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_LMSTUDIO_BASE_URL = "http://localhost:1234"

# * Default model names
DEFAULT_OLLAMA_MODEL = "llama3"
DEFAULT_LMSTUDIO_MODEL = "default"

# * API endpoints
OLLAMA_API_ENDPOINTS = {
    "generate": "/api/generate",
    "models": "/api/models",
    "status": "/api/status",
    "tags": "/api/tags",
    "show": "/api/show",
}

LMSTUDIO_API_ENDPOINTS = {
    "beta_chat": "/api/v0/chat/completions",
    "beta_models": "/api/v0/models",
    "beta_model_info": "/api/v0/models/{}",
    "openai_chat": "/v1/chat/completions",
    "openai_models": "/v1/models",
    "openai_model_info": "/v1/models/{}",
}

# * Default context window sizes
DEFAULT_CONTEXT_WINDOW_SIZE = 4096  # ? tokens

# * System message templates
DEFAULT_SYSTEM_MESSAGE = "You are a helpful AI assistant."

# * Test prompts
MODEL_TEST_PROMPT = "Say OK"
