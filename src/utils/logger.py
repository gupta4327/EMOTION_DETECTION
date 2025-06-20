import logging
import os
from logging.handlers import RotatingFileHandler

def get_logger(name: str = "app_logger") -> logging.Logger:
    """
    Returns a production-ready logger that logs DEBUG+ to console,
    and WARNING+ to a rotating file in a 'logs/' directory.
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Master level

    if logger.hasHandlers():
        return logger  # Avoid adding handlers multiple times

    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File Handler (WARNING and above)
    file_handler = RotatingFileHandler(
        filename="logs/proj_logs.log",
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3
    )
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(formatter)

    # Console Handler (DEBUG and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
