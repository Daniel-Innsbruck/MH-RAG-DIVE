import logging
import logging.handlers
import os


def setup_logger():
    """
    Configures and initializes the application logger.

    Sets up a logger with two handlers:
    - A FileHandler that writes DEBUG and higher level logs to 'logs/app.log'.
    - A StreamHandler that outputs INFO and higher level logs to the console.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger("rag_app")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler("logs/app.log", encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


logger = setup_logger()