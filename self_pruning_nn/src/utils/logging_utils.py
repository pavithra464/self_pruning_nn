import logging
import sys

def get_logger(name: str = "SelfPruningNN", level: int = logging.INFO) -> logging.Logger:
    """
    Sets up a standardized logger for training and debugging information.
    
    Args:
        name: The name of the logger
        level: The logging level
        
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers if already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        
        # Format: [TIME] [LEVEL] MESSAGE
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        
    return logger
