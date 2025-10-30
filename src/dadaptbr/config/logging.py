import logging
import os
import sys
from datetime import datetime


def log_model_info(
    logger: logging.Logger,
    phase: str,
    model_key: str,
    model_config: dict,
    **kwargs
) -> None:
    """Log standardized model information for any phase.

    Args:
        logger: Logger instance to use
        phase: Phase name (e.g., "translation", "evaluation", "review")
        model_key: Model key from config (e.g., "tower", "xcomet-xl")
        model_config: Model configuration dict from config file
        **kwargs: Additional phase-specific parameters to log
    """
    display_name = model_config.get("display_name", model_key)

    # Get model identifier based on phase type
    if "ollama_name" in model_config:
        model_id = model_config["ollama_name"]
        model_type = "Ollama"
    elif "hf_model_id" in model_config:
        model_id = model_config["hf_model_id"]
        model_type = "HuggingFace"
    else:
        model_id = model_key
        model_type = "Unknown"

    # Build parameter strings
    params = []
    if "max_workers" in kwargs:
        params.append(f"workers={kwargs['max_workers']}")
    if "batch_size" in kwargs:
        params.append(f"batch_size={kwargs['batch_size']}")
    if "device" in kwargs:
        params.append(f"device={kwargs['device']}")
    if "think" in model_config:
        params.append(f"think={model_config['think']}")

    param_str = ", ".join(params) if params else "default parameters"

    logger.info(
        f"[{phase.upper()}] Model: {display_name} ({model_key}) | "
        f"{model_type}: {model_id} | {param_str}"
    )


def setup_logger(name: str = "dadaptbr", log_to_file: bool = True, log_prefix: str = "app") -> logging.Logger:
    """Setup logger with console and optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(console_handler)

    # File (optional)
    if log_to_file:
        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/{log_prefix}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        logger.addHandler(file_handler)
        logger.info(f"Logging to: {log_file}")

    return logger
