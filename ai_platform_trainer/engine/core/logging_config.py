import logging
import sys

LOG_FILE = "game.log"


def setup_logging() -> logging.Logger:
    """
    Sets up a shared logger that writes to both file and console.
    Returns the configured logger instance for convenience.
    """
    # Configure root logger to capture all messages
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers from the root logger
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    # Application specific logger
    logger = logging.getLogger("ai_platform_trainer")
    logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all messages

    # Prevent adding handlers multiple times if this is called more than once
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(LOG_FILE, mode='w')  # 'w' mode to start fresh each time
        file_handler.setLevel(logging.DEBUG)  # Capture debug messages in file
        file_format = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
        file_handler.setFormatter(file_format)

        # Stream (console) handler
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.DEBUG)  # Show debug messages in console too
        stream_format = logging.Formatter("%(levelname)s - %(name)s - %(message)s")
        stream_handler.setFormatter(stream_format)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger
