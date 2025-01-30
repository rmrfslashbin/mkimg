import logging
from pathlib import Path
from rich.logging import RichHandler
from typing import Optional

def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    verbose: bool = False
) -> None:
    """Configure application logging"""
    # Set up basic configuration
    logging.basicConfig(
        level=logging.DEBUG if verbose else level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logging.getLogger().addHandler(file_handler)
    
    # Create logger
    logger = logging.getLogger("sdprompt")
    logger.setLevel(logging.DEBUG if verbose else level) 