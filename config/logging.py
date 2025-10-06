import logging
import os
import sys
from datetime import datetime


def setup_logger(
    name: str = "dadaptbr",
    log_to_file: bool = True,
    log_prefix: str = "translation",
    custom_timestamp: str = None,
) -> logging.Logger:
    """
    Set up a comprehensive logger with file and console output.

    Args:
        name: Logger name
        log_to_file: Whether to log to file
        log_prefix: Prefix for log file name (default: "translation")
        custom_timestamp: Custom timestamp for log file (default: auto-generated)

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (if enabled)
    if log_to_file:
        os.makedirs("logs", exist_ok=True)
        if custom_timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            timestamp = custom_timestamp
        log_file = f"logs/{log_prefix}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_file}")

    return logger
