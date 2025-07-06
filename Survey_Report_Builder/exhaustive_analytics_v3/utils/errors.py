"""
Error handling utilities for Exhaustive Analytics v3

Provides consistent error formatting and reporting functions.
"""

from typing import List
from .. import config


def print_error_block(title: str, details: List[str]) -> None:
    """
    Print a clearly formatted error or warning block.
    
    Args:
        title: Error/warning title
        details: List of detail messages
    """
    print(f"\n{config.ERROR_SEPARATOR}")
    print(f"ERROR: {title}")
    print(config.ERROR_SEPARATOR)
    for detail in details:
        print(f"  {detail}")
    print(f"{config.ERROR_SEPARATOR}\n")