import ollama

from .config.datasets import LLM_DEFAULT_PARAMS, TRANSLATION_MODELS
from .config.logging import setup_logger

_LOGGER = setup_logger("llm_client", log_to_file=True, log_prefix="llm")


def get_model_config(model_key: str = None):
    """Get model configuration by key or return default."""
    if model_key and model_key in TRANSLATION_MODELS:
        return TRANSLATION_MODELS[model_key]

    # Return default model
    for config in TRANSLATION_MODELS.values():
        if config.get("default", False):
            return config

    # Fallback to first model
    return list(TRANSLATION_MODELS.values())[0]


def get_ollama_name(model_key: str = None) -> str:
    """Get Ollama model name from config key."""
    config = get_model_config(model_key)
    return config["ollama_name"]


def init_ollama(model_name: str = None):
    """Initialize Ollama client and validate model."""
    client = ollama.Client()
    _LOGGER.info("Connected to Ollama")

    # Use config if no model specified
    if not model_name:
        model_name = get_ollama_name()

    models = client.list()
    if hasattr(models, "models") and models.models:
        model_names = [model.model for model in models.models]
        if model_name not in model_names:
            raise Exception(f"{model_name} model not found")
        _LOGGER.info(f"{model_name} model found")
    else:
        raise Exception("No models found")

    return client


def translate_text(text: str, client, model_name: str, prompt: str) -> str:
    """Translate text using specified model with config parameters."""
    try:
        # Format the prompt with the actual text to translate
        formatted_prompt = prompt.format(text=text)

        response = client.chat(
            model=model_name,
            messages=[{"role": "user", "content": formatted_prompt}],
            options=LLM_DEFAULT_PARAMS,
        )

        translation = response["message"]["content"].strip()

        # Clean up the response
        if '"""' in translation:
            parts = translation.split('"""')
            if len(parts) >= 2:
                translation = parts[-2].strip()

        return translation
    except Exception as e:
        _LOGGER.error(f"Error translating text: {e}")
        return text
