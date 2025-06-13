"""Interactive utilities for commit-gen."""

import questionary
from toolkit_utils import ToolkitError, error_handler
from toolkit_utils.logging_utils import info, warning


@error_handler(
    message="Error selecting model interactively", default_return=(None, None)
)
def select_model_interactively(providers):
    """Prompt the user to select a provider and model interactively.

    Args:
        providers (dict): Dictionary of available provider instances

    Returns:
        tuple: (provider_instance, model_name) or (None, None) if cancelled
    """
    if not providers:
        raise ToolkitError(
            "No LLM providers available. Please ensure Ollama or LM Studio is running."
        )

    # * First, select a provider if multiple are available
    provider_instance = None
    if len(providers) > 1:
        provider_choices = []
        for provider_type, provider in providers.items():
            model_count = len(provider.list_models())
            provider_choices.append(
                questionary.Choice(
                    title=(
                        f"{provider_type.capitalize()} ({model_count} models available)"
                    ),
                    value=provider,
                )
            )

        provider_instance = questionary.select(
            "Select an LLM provider:", choices=provider_choices
        ).ask()

        # * User cancelled the selection
        if provider_instance is None:
            return None, None
    else:
        # * Only one provider available, use it
        provider_instance = list(providers.values())[0]
        provider_type = list(providers.keys())[0]
        info(f"Using {provider_type.capitalize()} (only available provider)")

    # * Then select a model from the chosen provider
    models = provider_instance.list_models()

    if not models:
        raise ToolkitError(f"No models available for {provider_instance.name}.")

    model_choices = [
        questionary.Choice(title=model["name"], value=model["id"]) for model in models
    ]

    selected_model = questionary.select("Select a model:", choices=model_choices).ask()

    # * User cancelled the selection
    if selected_model is None:
        return None, None

    # * Update the provider instance with the selected model
    provider_instance.model = selected_model

    # * Check if the model is loaded and warn the user if it's not
    info(f"Checking if model '{selected_model}' is loaded...")
    if not provider_instance.is_model_loaded():
        warning(f"\nModel '{selected_model}' is not yet fully loaded.")
        warning("Please make sure the model is loaded in the UI before proceeding.")

        # * Ask the user if they want to continue anyway
        continue_anyway = questionary.confirm(
            "Continue anyway? (Not recommended if model is still loading)",
            default=False,
        ).ask()

        if not continue_anyway:
            info("Operation cancelled. Please load the model and try again.")
            return None, None
        else:
            warning("Continuing with partially loaded model. This may cause errors.")

    return provider_instance, selected_model
