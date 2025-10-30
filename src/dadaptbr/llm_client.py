import os

import httpx
import ollama

from .config.datasets import LLM_DEFAULT_PARAMS, TRANSLATION_MODELS
from .config.logging import setup_logger

_LOGGER = setup_logger("llm_client", log_to_file=True, log_prefix="llm")


def init_ollama(model_name: str = None, max_workers: int = None):
    """Initialize Ollama client with optimized connection pooling."""
    host = os.getenv("OLLAMA_HOST", "localhost")

    if max_workers is not None:
        max_connections = max(max_workers * 2, 20)
        max_keepalive = max(max_workers, 10)

        limits = httpx.Limits(
            max_keepalive_connections=max_keepalive,
            max_connections=max_connections,
        )

        timeout = httpx.Timeout(
            connect=10.0,
            read=300.0,
            write=10.0,
            pool=30.0,
        )

        client = ollama.Client(
            host=host,
            limits=limits,
            timeout=timeout,
        )

        _LOGGER.info(
            f"Ollama client configured with connection pool: "
            f"max_connections={max_connections}, max_keepalive={max_keepalive}"
        )
    else:
        client = ollama.Client(host=host)

    _LOGGER.info(f"Connected to Ollama at {host}")

    sched_spread = os.getenv("OLLAMA_SCHED_SPREAD")
    num_parallel = os.getenv("OLLAMA_NUM_PARALLEL")

    if sched_spread:
        _LOGGER.info(
            f"Ollama multi-GPU scheduling enabled (OLLAMA_SCHED_SPREAD={sched_spread})"
        )

    if num_parallel:
        _LOGGER.info(
            f"Ollama parallel requests configured (OLLAMA_NUM_PARALLEL={num_parallel})"
        )
        _LOGGER.info(
            f"Recommend setting --workers={num_parallel} to match Ollama capacity"
        )

    # Use config if no model specified
    if not model_name:
        # Get default model from config
        model_name = next(
            (
                config["ollama_name"]
                for config in TRANSLATION_MODELS.values()
                if config.get("default", False)
            ),
            next(iter(TRANSLATION_MODELS.values()), {}).get("ollama_name", None),
        )
        if not model_name:
            raise Exception("No default translation model configured")

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
        formatted_prompt = prompt.format(text=text)

        # Check if this model should disable thinking
        # Find the model config by reverse lookup on ollama_name
        think_param = None
        for _model_key, config in TRANSLATION_MODELS.items():
            if config["ollama_name"] == model_name:
                if "think" in config:
                    think_param = config["think"]
                break

        # Build chat call parameters
        chat_kwargs = {
            "model": model_name,
            "messages": [{"role": "user", "content": formatted_prompt}],
            "options": LLM_DEFAULT_PARAMS,
        }

        # Add think parameter if specified (for thinking models like qwen3)
        if think_param is not None:
            chat_kwargs["think"] = think_param

        response = client.chat(**chat_kwargs)
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
