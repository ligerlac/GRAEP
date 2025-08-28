import logging
from typing import Any, Dict, List, Optional, Tuple

from rich.logging import RichHandler
from rich.text import Text
from rich.console import Console
from rich.pretty import Pretty
from rich.theme import Theme
from rich.style import Style
from tabulate import tabulate

# ANSI escape codes for colors
BLUE = "\033[0;34m"
YELLOW = "\033[1;33m"
RED = "\033[0;31m"
MAGENTA = "\033[95m"
GREEN = "\033[0;32m"
RESET = "\033[0m"

def log_banner(text: str) -> None:
    """
    Logs a magenta-colored banner.

    Parameters
    ----------
    text : str
        The text to display in the banner.
    """
    console = Console(theme=Theme({"log.message": "magenta", "repr.path": "white"}))

    text = (f"\n{'=' * 80}\n"
            f"{ ' ' * ((80 - len(text)) // 2)}{text.upper()}\n"
            f"{ '=' * 80}"
        )
    console.print(text, style="log.message")

def setup_logging(level: str = "INFO") -> None:
    """
    Sets up logging with RichHandler.

    Parameters
    ----------
    level : str, optional
        The logging level, by default "INFO"
    """
    log = logging.getLogger()

    # Check if handlers already exist to avoid duplicate logging
    if log.handlers:
        return

    custom_theme = Theme({
        "repr.path": "default",   # no color for paths
        "repr.filename": "default",
    })

    console = Console(theme=custom_theme)
    handler = RichHandler(console=console, rich_tracebacks=True, show_time=False, markup=False) #, level_styles=level_styles)
    handler.setFormatter(
        logging.Formatter("%(message)s")
    )
    log.addHandler(handler)
    log.setLevel(level)
