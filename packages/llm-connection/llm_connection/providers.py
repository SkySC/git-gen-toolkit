import json
from abc import ABC, abstractmethod

import requests
from toolkit_utils import (
    LLMConnectionError,
    LLMModelNotFoundError,
    LLMModelNotLoadedError,
    LLMProviderError,
    LLMResponseError,
    error_handler,
)
from toolkit_utils.logging_utils import error, info, success, warning
from toolkit_utils.progress_utils import spinner

from .constants import (
    API_CHECK_TIMEOUT,
    DEFAULT_API_TIMEOUT,
    DEFAULT_CONTEXT_WINDOW_SIZE,
    DEFAULT_LMSTUDIO_BASE_URL,
    DEFAULT_LMSTUDIO_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    LMSTUDIO_API_ENDPOINTS,
    MODEL_CHECK_TIMEOUT,
    MODEL_LOAD_TIMEOUT,
    MODEL_TEST_PROMPT,
    OLLAMA_API_ENDPOINTS,
)
from .message_builder import (
    build_chunked_prompt,
    build_messages_format,
    build_simple_prompt,
    build_template_prompt,
)
from .message_cleanup import clean_message


class LLMProvider(ABC):
    """Base class for LLM providers.

    This class provides the abstract interface for all LLM providers.

    Return Value Pattern:
    - API methods that generate content will return content string or None on error
    - Methods with @error_handler will return their default value on error
    - Boolean methods (is_available, is_model_loaded) return True/False
    - List methods return empty lists on failure rather than None
    """

    def __init__(
        self,
        model=DEFAULT_LMSTUDIO_MODEL,
        base_url=DEFAULT_LMSTUDIO_BASE_URL,
        template_path=None,
        prompt_template_path=None,
        timeout=300,
        **kwargs,
    ):
        self.model = model
        self.base_url = base_url
        self.template_path = template_path
        self.prompt_template_path = prompt_template_path
        self.timeout = timeout
        self.provider_options = kwargs
        # * Cache for storing context between calls
        self.context_cache = {}
        # * Flag to control whether to automatically clean responses
        self.auto_clean_responses = kwargs.get("auto_clean_responses", True)

    @property
    @abstractmethod
    def name(self):
        """Return the name of the provider."""
        pass

    @abstractmethod
    def _call_api_raw(self, prompt):
        """Make the HTTP request to the LLM API and return the raw response."""
        pass

    def _call_api(self, prompt):
        """Make the HTTP request to the LLM API with automatic response cleaning.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            The response from the LLM, automatically cleaned
            if auto_clean_responses is True
        """
        with spinner("Processing request..."):
            # * Get the raw response from the LLM
            raw_response = self._call_api_raw(prompt)

        # * If auto-cleaning is enabled, clean the response
        if self.auto_clean_responses and raw_response:
            clean_content, reasoning = clean_message(raw_response)
            # * Store the reasoning in the context cache for later retrieval if needed
            self.context_cache["last_reasoning"] = reasoning
            return clean_content

        # * Return raw response if auto-cleaning is disabled or the response is empty
        return raw_response

    def get_last_reasoning(self):
        """Get the reasoning from the last API call if available.

        Returns:
            str: The reasoning text or empty string if not available
        """
        return self.context_cache.get("last_reasoning", "")

    @abstractmethod
    def get_context_size(self):
        """Get the maximum token context size for the loaded model."""
        pass

    @abstractmethod
    def is_available(self):
        """Check if the provider is available and running.

        Returns:
            bool: True if the provider is available and responding to API calls
        """
        pass

    @abstractmethod
    def list_models(self):
        """List all available models for this provider.

        Returns:
            list: A list of model information dictionaries with at least 'id' and 'name'
            keys
        """
        pass

    @abstractmethod
    def is_model_loaded(self, model_name=None):
        """Check if a specific model is loaded and ready to use.

        Args:
            model_name: The name of the model to check, or None to check the currently
            set model

        Returns:
            bool: True if the model is loaded and ready to use
        """
        pass

    @classmethod
    def get_available_providers(cls, timeout=2):
        """Check which providers are available and return initialized instances.

        Args:
            timeout: Timeout in seconds for checking provider availability

        Returns:
            dict: Dictionary of available provider instances by type
        """
        providers = {}

        # * Try Ollama
        try:
            ollama = OllamaProvider(timeout=timeout)
            if ollama.is_available():
                providers["ollama"] = ollama
        except Exception:
            pass

        # * Try LM Studio
        try:
            lmstudio = LMStudioProvider(timeout=timeout, auto_start_server=True)
            if lmstudio.is_available():
                providers["lmstudio"] = lmstudio
        except Exception:
            pass

        return providers

    def call_api_with_templates(
        self,
        content,
        primary_template_content=None,
        prompt_template_content=None,
        chunk_index=None,
        total_chunks=None,
        primary_template_key="{{primary_template}}",
        content_key="{{content}}",
    ):
        """Call the API with templates and content.

        This method enables caching template content when processing multiple chunks
        by using a different approach for first chunk vs subsequent chunks.

        This generic implementation works for both commit-gen and pr-gen:
        - For commit-gen: content is the diff, primary_template is commit_template
        - For pr-gen: content is the commit list, primary_template is pr_template

        Args:
            content: The main content to include in the prompt (diff or commit list)
            primary_template_content: The primary template content
            (commit or PR template)
            prompt_template_content: The prompt template content
            chunk_index: Current chunk index (0-based) for multi-chunk processing
            total_chunks: Total number of chunks for multi-chunk processing
            primary_template_key: The placeholder key for the primary template
            (default: "{{primary_template}}")
            content_key: The placeholder key for the content (default: "{{content}}")

        Returns:
            The response from the LLM
        """
        # * For first chunk or single chunk, use the full templates
        if chunk_index is None or chunk_index == 0:
            # * Use message_builder to create the prompt with templates
            prompt = build_template_prompt(
                content,
                primary_template_content,
                prompt_template_content,
                primary_template_key,
                content_key,
            )

            # * Cache the templates for future chunks
            if chunk_index == 0 and total_chunks > 1:
                self.context_cache["primary_template"] = primary_template_content
                self.context_cache["has_template_context"] = True
                self.context_cache["content_type"] = (
                    "diff" if content_key == "{{diff}}" else "commits"
                )

            return self._call_api(prompt)

        # * For subsequent chunks, use a simplified prompt to save tokens
        else:
            content_type = self.context_cache.get("content_type", "diff")
            # * Use message_builder's chunked prompt function
            prompt = build_chunked_prompt(
                content, chunk_index, total_chunks, content_type
            )
            return self._call_api(prompt)


class OllamaProvider(LLMProvider):
    def __init__(
        self,
        model=DEFAULT_OLLAMA_MODEL,
        base_url=DEFAULT_OLLAMA_BASE_URL,
        template_path=None,
        prompt_template_path=None,
        timeout=DEFAULT_API_TIMEOUT,
        **kwargs,
    ):
        super().__init__(
            model,
            base_url,
            template_path,
            prompt_template_path,
            timeout=timeout,
            **kwargs,
        )

    @property
    def name(self):
        return f"Ollama ({self.model})"

    @error_handler(message="Error calling Ollama API", default_return=None)
    def _call_api_raw(self, prompt):
        api_url = f"{self.base_url}{OLLAMA_API_ENDPOINTS['generate']}"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        payload.update(self.provider_options)

        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(
                api_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=self.timeout,
            )
            response.raise_for_status()
            response_data = response.json()

            return response_data.get("response", "")
        except requests.exceptions.ConnectionError as e:
            raise LLMConnectionError(
                f"Cannot connect to Ollama at {self.base_url}", cause=e
            )
        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                raise LLMModelNotFoundError(f"Model '{self.model}' not found", cause=e)
            else:
                error_response = (
                    response.json() if response.text else {"error": "Unknown error"}
                )
                error_msg = error_response.get("error", str(e))
                raise LLMProviderError(
                    f"HTTP error from Ollama API: {error_msg}", cause=e
                )
        except requests.exceptions.Timeout as e:
            raise LLMConnectionError(
                f"Timeout connecting to Ollama API: {str(e)}", cause=e
            )
        except requests.exceptions.RequestException as e:
            raise LLMConnectionError(f"Error calling Ollama API: {str(e)}", cause=e)
        except ValueError as e:
            raise LLMResponseError(
                f"Invalid JSON response from Ollama API: {str(e)}", cause=e
            )

    def get_context_size(self):
        """Get the maximum token context size for the loaded model.

        Returns:
            int: The maximum context size in tokens, or None if unable to retrieve
        """
        api_url = f"{self.base_url}{OLLAMA_API_ENDPOINTS['show']}"
        params = {"name": self.model}

        try:
            response = requests.post(api_url, json=params, timeout=self.timeout)
            response.raise_for_status()
            model_info = response.json()

            # * The context length is in the model's parameters
            # * Default to DEFAULT_CONTEXT_WINDOW_SIZE if we can't find it
            context_size = model_info.get("parameters", {}).get(
                "context_length", DEFAULT_CONTEXT_WINDOW_SIZE
            )

            return int(context_size)
        except (requests.exceptions.RequestException, ValueError, KeyError) as e:
            error(f"Error getting context size from Ollama: {e}")
            # * Return a safe default if we can't get the actual value
            return DEFAULT_CONTEXT_WINDOW_SIZE

    def is_available(self):
        """Check if the Ollama provider is available and running."""
        try:
            response = requests.get(
                f"{self.base_url}{OLLAMA_API_ENDPOINTS['status']}",
                timeout=API_CHECK_TIMEOUT,
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def list_models(self):
        """List all available models for the Ollama provider."""
        try:
            response = requests.get(
                f"{self.base_url}{OLLAMA_API_ENDPOINTS['models']}", timeout=self.timeout
            )
            response.raise_for_status()
            models = response.json()
            return [{"id": model["id"], "name": model["name"]} for model in models]
        except requests.exceptions.RequestException as e:
            error(f"Error listing models from Ollama: {e}")
            return []

    @error_handler(message="Error checking if model is loaded", default_return=False)
    def is_model_loaded(self, model_name=None, auto_load=True):
        """Check if a specific model is loaded and ready for inference in Ollama.

        Ollama will automatically begin loading models when API requests are made,
        so by default this method will trigger the model to start loading if it's not
        already loaded.

        Args:
            model_name: The name of the model to check, or None to check the currently
                set model
            auto_load: If True, allow the test request to trigger model loading

        Returns:
            bool: True if the model is loaded and ready to use
        """
        model_to_check = model_name or self.model

        try:
            # * First try the list models endpoint which has explicit model state
            # * information
            response = requests.get(
                f"{self.base_url}{OLLAMA_API_ENDPOINTS['tags']}",
                headers={"Content-Type": "application/json"},
                timeout=MODEL_CHECK_TIMEOUT,
            )

            if response.status_code == 200:
                models_info = response.json().get("models", [])
                found_model = False

                for model in models_info:
                    if model.get("name") == model_to_check:
                        found_model = True
                        # * Ollama tags API doesn't expose loading state directly
                        # * but we can check if the model is in the list
                        info(f"Model '{model_to_check}' is available in Ollama")
                        break

                # * If model isn't in the list, it may need to be pulled or is loading
                if not found_model and not auto_load:
                    warning(
                        f"Model '{model_to_check}' is not available. You may need to"
                        f" run 'ollama pull {model_to_check}'"
                    )
                    return False

            # * If we couldn't determine from the API or need to trigger loading,
            # * make a test request
            if auto_load:
                api_url = f"{self.base_url}{OLLAMA_API_ENDPOINTS['generate']}"
                payload = {
                    "model": model_to_check,
                    "prompt": MODEL_TEST_PROMPT,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 3},
                }

                info(f"Checking if model '{model_to_check}' is ready...")

                with spinner(f"Loading model '{model_to_check}'...") as s:
                    try:
                        response = requests.post(
                            api_url,
                            headers={"Content-Type": "application/json"},
                            json=payload,
                            timeout=MODEL_LOAD_TIMEOUT,
                        )

                        if response.status_code == 200:
                            s.succeed(f"Model '{model_to_check}' is loaded and ready")
                            return True
                        else:
                            if response.status_code == 404:
                                s.fail(f"Model '{model_to_check}' not found")
                                print()  # Add line break after spinner
                                warning(f"Try 'ollama pull {model_to_check}'")
                                raise LLMModelNotFoundError(
                                    f"Model '{model_to_check}' not found in Ollama. "
                                    f"Try 'ollama pull {model_to_check}'"
                                )
                            elif response.status_code == 500:
                                s.fail(
                                    f"Model '{model_to_check}' may be loading or has an"
                                    " error"
                                )
                                if "error" in response.json():
                                    error_msg = response.json()["error"]
                                    print()  # Add line break after spinner
                                    error(f"Error: {error_msg}")
                                    raise LLMModelNotLoadedError(
                                        f"Error with model '{model_to_check}':"
                                        f" {error_msg}"
                                    )
                            return False
                    except requests.exceptions.Timeout:
                        s.fail(f"Model '{model_to_check}' is still loading")
                        print()  # Add line break after spinner
                        info("Please wait for the model to finish loading in Ollama")
                        raise LLMModelNotLoadedError(
                            f"Model '{model_to_check}' is still loading. Please wait"
                            " for the model to finish loading in Ollama"
                        )
            else:
                info(
                    f"Not checking if model '{model_to_check}' can be loaded via API"
                    " call"
                )
                return False

        except requests.exceptions.ConnectionError:
            error(f"Cannot connect to Ollama at {self.base_url}")
            error("Please make sure Ollama is running")
            raise LLMConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Please make sure Ollama is running."
            )
        except requests.exceptions.RequestException as e:
            error(f"Error connecting to Ollama: {e}")
            raise LLMConnectionError(f"Error connecting to Ollama: {str(e)}", cause=e)


class LMStudioProvider(LLMProvider):
    def __init__(
        self,
        model=DEFAULT_LMSTUDIO_MODEL,
        base_url=DEFAULT_LMSTUDIO_BASE_URL,
        template_path=None,
        prompt_template_path=None,
        timeout=DEFAULT_API_TIMEOUT,
        auto_start_server=True,
        **kwargs,
    ):
        super().__init__(
            model,
            base_url,
            template_path,
            prompt_template_path,
            timeout=timeout,
            **kwargs,
        )
        self.auto_start_server = auto_start_server

    @property
    def name(self):
        return f"LM Studio ({self.model})"

    @error_handler(message="Error calling LM Studio API", default_return=None)
    def _call_api_raw(self, prompt):
        beta_api_url = f"{self.base_url}{LMSTUDIO_API_ENDPOINTS['beta_chat']}"

        system_message_content = self.provider_options.get(
            "system_message", "You are a helpful AI assistant."
        )
        messages = [
            {"role": "system", "content": system_message_content},
            {"role": "user", "content": prompt},
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": -1,
            "stream": False,
        }
        options_to_update = {
            k: v for k, v in self.provider_options.items() if k != "system_message"
        }
        payload.update(options_to_update)

        headers = {"Content-Type": "application/json"}

        try:
            # * First try Beta REST API
            response = requests.post(
                beta_api_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=self.timeout,
            )
            if response.status_code == 200:
                response_data = response.json()
                if response_data.get("choices"):
                    return response_data["choices"][0]["message"]["content"]

            # * Fall back to OpenAI compatibility API if Beta API fails
            openai_api_url = f"{self.base_url}{LMSTUDIO_API_ENDPOINTS['openai_chat']}"
            response = requests.post(
                openai_api_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=self.timeout,
            )
            response.raise_for_status()
            response_data = response.json()
            if response_data.get("choices"):
                return response_data["choices"][0]["message"]["content"]

            # * If we reach here with no return,
            # * it means the response format is unexpected
            raise LLMResponseError(
                f"Unexpected response format from LM Studio API: {response_data}"
            )
        except requests.exceptions.ConnectionError as e:
            raise LLMConnectionError(
                f"Cannot connect to LM Studio at {self.base_url}", cause=e
            )
        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                raise LLMModelNotFoundError(f"Model '{self.model}' not found", cause=e)
            else:
                error_response = (
                    response.json() if response.text else {"error": "Unknown error"}
                )
                error_msg = error_response.get("error", str(e))
                raise LLMProviderError(
                    f"HTTP error from LM Studio API: {error_msg}", cause=e
                )
        except requests.exceptions.Timeout as e:
            raise LLMConnectionError(
                f"Timeout connecting to LM Studio API: {str(e)}", cause=e
            )
        except requests.exceptions.RequestException as e:
            raise LLMConnectionError(f"Error calling LM Studio API: {str(e)}", cause=e)
        except ValueError as e:
            raise LLMResponseError(
                f"Invalid JSON response from LM Studio API: {str(e)}", cause=e
            )

    def get_context_size(self):
        """Get the maximum token context size for the loaded model.

        Returns:
            int: The maximum context size in tokens, or default value if unable to
            retrieve
        """
        # * Try Beta REST API first for specific model info
        api_url = f"{self.base_url}/api/v0/models/{self.model}"
        headers = {"Content-Type": "application/json"}

        try:
            # * Try to get model-specific information from Beta REST API
            response = requests.get(api_url, headers=headers, timeout=self.timeout)

            if response.status_code == 200:
                model_info = response.json()
                # * Try to get max context length from model info
                if "max_context_length" in model_info:
                    return int(model_info["max_context_length"])

            # * If model-specific request fails, try the models list endpoint
            response = requests.get(
                f"{self.base_url}/api/v0/models", headers=headers, timeout=self.timeout
            )

            if response.status_code == 200:
                models_info = response.json().get("data", [])
                # * Find our model in the list
                for model in models_info:
                    if model.get("id") == self.model:
                        if "max_context_length" in model:
                            return int(model["max_context_length"])

            # * Fall back to OpenAI compatibility API
            api_url = f"{self.base_url}/v1/models/{self.model}"
            response = requests.get(api_url, headers=headers, timeout=self.timeout)

            if response.status_code == 200:
                model_info = response.json()
                # * Try to get context length from model info
                if "context_length" in model_info:
                    return int(model_info["context_length"])
                if "max_tokens" in model_info:
                    return int(model_info["max_tokens"])

            # * If model-specific request fails, try the models list endpoint
            response = requests.get(
                f"{self.base_url}/v1/models", headers=headers, timeout=self.timeout
            )

            if response.status_code == 200:
                models_info = response.json()
                # * Find our model in the list
                for model in models_info.get("data", []):
                    if model.get("id") == self.model:
                        if "context_length" in model:
                            return int(model["context_length"])
                        if "max_tokens" in model:
                            return int(model["max_tokens"])

            # * Default fallback
            return 4096

        except (requests.exceptions.RequestException, ValueError, KeyError) as e:
            error(f"Error getting context size from LM Studio: {e}")
            # * Return a safe default if we can't get the actual value
            return 4096

    def is_available(self):
        """Check if the LM Studio provider is available and running."""
        try:
            # * Always try Beta REST API first for models
            response = requests.get(
                f"{self.base_url}/api/v0/models", timeout=self.timeout
            )
            if response.status_code == 200:
                return True

            # * If Beta REST API fails, fall back to OpenAI API
            response = requests.get(f"{self.base_url}/v1/models", timeout=self.timeout)
            if response.status_code == 200:
                return True

            # * If neither API is available and auto_start_server is enabled,
            # * try to start the server
            if self.auto_start_server:
                return self._start_server()

            return False
        except requests.exceptions.RequestException:
            # * If connection fails and auto_start_server is enabled, try to start the
            # * server
            if self.auto_start_server:
                return self._start_server()

            return False

    def _start_server(self):
        """Attempt to start the LM Studio server if not running.

        Returns:
            bool: True if server started successfully, False otherwise
        """
        try:
            import subprocess
            import time

            info("LM Studio server not detected. Attempting to start it...")

            # * Start the LM Studio server using the CLI command
            subprocess.Popen(
                ["lms", "server", "start"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # * Wait for a moment to let the server initialize
            time.sleep(5)

            # * Check if the server is now available
            for _ in range(3):
                response = requests.get(
                    f"{self.base_url}/api/v0/models", timeout=self.timeout
                )
                if response.status_code == 200:
                    success("LM Studio server started successfully.")
                    return True

                # * Wait before trying again
                time.sleep(2)

            warning("Failed to verify if LM Studio server started correctly.")
            return False

        except Exception as e:
            error(f"Error starting LM Studio server: {e}")
            warning(
                "Please make sure LM Studio is installed and the 'lms' command is"
                " available."
            )
            return False

    def list_models(self):
        """List all available models for the LM Studio provider."""
        try:
            # * Always try Beta REST API first
            response = requests.get(
                f"{self.base_url}/api/v0/models", timeout=self.timeout
            )

            if response.status_code == 200:
                models = response.json().get("data", [])
                return [
                    {
                        "id": model["id"],
                        "name": model["id"],
                        "state": model.get("state", "unknown"),
                        "max_context_length": model.get("max_context_length", 4096),
                    }
                    for model in models
                ]

            # * Fall back to OpenAI API if Beta API fails
            response = requests.get(f"{self.base_url}/v1/models", timeout=self.timeout)
            if response.status_code == 200:
                models = response.json().get("data", [])
                return [{"id": model["id"], "name": model["id"]} for model in models]

            return []
        except requests.exceptions.RequestException as e:
            error(f"Error listing models from LM Studio: {e}")
            return []

    @error_handler(message="Error checking if model is loaded", default_return=False)
    def is_model_loaded(self, model_name=None):
        """Check if a specific model is loaded and ready for inference in LM Studio.

        Args:
            model_name: The name of the model to check, or None to check the current
            model

        Returns:
            bool: True if the model is loaded and ready to use
        """
        model_to_check = model_name or self.model

        # * Get models list first, before starting spinner, to avoid log conflicts
        try:
            models = self.list_models()
        except Exception:
            # * If we can't get models list, we'll handle it in the spinner context
            models = []

        with spinner(
            f"Checking if model '{model_to_check}' is available in LM Studio"
        ) as s:
            try:
                # * For LM Studio API, we need to get the model's state from the list
                if not models:
                    # * If we couldn't get models earlier, the service is likely down
                    s.fail(f"Cannot connect to LM Studio at {self.base_url}")
                    print()  # Add line break after spinner
                    raise LLMConnectionError(
                        f"Cannot connect to LM Studio at {self.base_url}. "
                        "Please make sure LM Studio is running."
                    )

                for model_info in models:
                    if model_info["id"] == model_to_check:
                        # * Check for the model state if available
                        state = model_info.get("state", "unknown")
                        if state == "loaded":
                            s.succeed(f"Model '{model_to_check}' is loaded and ready")
                            return True
                        elif state == "loading":
                            s.text = f"Model '{model_to_check}' is still loading..."
                            s.fail(f"Model '{model_to_check}' is still loading")
                            print()  # Add line break after spinner
                            raise LLMModelNotLoadedError(
                                f"Model '{model_to_check}' is still loading in LM"
                                " Studio"
                            )
                        else:
                            # * If no state information, we assume it's ready
                            # * (older LM Studio versions)
                            s.succeed(
                                f"Model '{model_to_check}' appears to be available"
                            )
                            return True

                s.fail(f"Model '{model_to_check}' not found in LM Studio's model list")
                print()  # Add line break after spinner
                raise LLMModelNotFoundError(
                    f"Model '{model_to_check}' not found in LM Studio"
                )

            except requests.exceptions.ConnectionError:
                s.fail(f"Cannot connect to LM Studio at {self.base_url}")
                print()  # Add line break after spinner
                raise LLMConnectionError(
                    f"Cannot connect to LM Studio at {self.base_url}. "
                    "Please make sure LM Studio is running."
                )
            except requests.exceptions.RequestException as e:
                s.fail(f"Error checking model: {str(e)}")
                print()  # Add line break after spinner
                raise LLMConnectionError(
                    f"Error connecting to LM Studio: {str(e)}", cause=e
                )

    def generate_completion(self, prompt, use_message_format=True, system_message=None):
        """Generate a completion from the LLM provider.

        Args:
            prompt: The prompt to send to the LLM
            use_message_format: Whether to format the prompt using the messages format
            system_message: Optional system message to include

        Returns:
            The generated text from the LLM
        """
        if use_message_format:
            # * Use message_builder to create a properly formatted messages array
            formatted_prompt = build_messages_format(prompt, system_message)
            return self._call_api(formatted_prompt)
        else:
            # * Use simple prompt formatting
            formatted_prompt = build_simple_prompt(prompt, system_message)
            return self._call_api(formatted_prompt)

    def generate_with_template(
        self, content, primary_template_content=None, prompt_template_content=None
    ):
        """Generate a completion using templates.

        Args:
            content: The main content to send to the LLM
            primary_template_content: The primary template content as a string
            prompt_template_content: The prompt template content as a string

        Returns:
            The generated text from the LLM
        """
        # * Use message_builder to create a properly formatted template prompt
        formatted_prompt = build_template_prompt(
            content, primary_template_content, prompt_template_content
        )
        return self._call_api(formatted_prompt)
