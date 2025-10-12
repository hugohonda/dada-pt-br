#!/usr/bin/env python3
"""
LLM client module for Ollama integration.
"""

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
        response = client.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
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


def get_model_info(model_name: str = None):
    """Get information about a specific model."""
    if not model_name:
        model_name = get_ollama_name()

    try:
        client = ollama.Client()
        models = client.list()

        # Find the specified model
        for model in models.models:
            if model.model == model_name:
                return {
                    "name": model.model,
                    "size": getattr(model, "size", "Unknown"),
                    "modified_at": str(getattr(model, "modified_at", "Unknown")),
                    "digest": getattr(model, "digest", "Unknown")[:12] + "..."
                    if hasattr(model, "digest")
                    else "Unknown",
                }

        return {"name": model_name, "status": "Model not found in Ollama list"}

    except Exception as e:
        _LOGGER.warning(f"Could not get model info: {e}")
        return {"name": model_name, "error": str(e)}


def get_available_models():
    """Get list of available models."""
    try:
        client = ollama.Client()
        models = client.list()
        if hasattr(models, "models") and models.models:
            return [model.model for model in models.models]
        return []
    except Exception as e:
        _LOGGER.error(f"Error getting available models: {e}")
        return []


def is_model_available(model_name: str) -> bool:
    """Check if a model is available."""
    available_models = get_available_models()
    return model_name in available_models


def get_model_display_name(model_name: str = None) -> str:
    """Get a human-readable name for the model."""
    if not model_name:
        config = get_model_config()
        return config["display_name"]

    # Try to find by Ollama name
    for config in TRANSLATION_MODELS.values():
        if config["ollama_name"] == model_name:
            return config["display_name"]

    # Fallback to original name
    return model_name


def list_configured_models():
    """List all configured models."""
    return list(TRANSLATION_MODELS.keys())


def get_default_model_key() -> str:
    """Get the default model key."""
    for key, config in TRANSLATION_MODELS.items():
        if config.get("default", False):
            return key
    return list(TRANSLATION_MODELS.keys())[0]
