from typing import Dict, Optional
import logging


def get_logger(name: str = 'rtgmff', logfile: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger.

    The logger outputs to both the console and an optional log file.  Use
    this function at the start of your script to obtain a preconfigured
    logger instead of using `print` statements directly.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured
    logger.setLevel(level)
    fmt = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    # File handler if logfile is provided
    if logfile:
        fh = logging.FileHandler(logfile)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def log_metrics(epoch: int, metrics: Dict[str, float], logger: Optional[logging.Logger] = None,
                prefix: str = '') -> None:
    """Log metrics via the provided logger or print if none is given."""
    msg_parts = [f"{prefix}Epoch {epoch}"]
    for k, v in metrics.items():
        msg_parts.append(f"{k}={v:.4f}")
    message = ' | '.join(msg_parts)
    if logger is not None:
        logger.info(message)
    else:
        print(message)
