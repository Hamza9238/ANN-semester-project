"""
helpers.py
==========
Reusable utility functions for common operations across the pipeline.

Functions provided
------------------
  • get_device()              - Get optimal PyTorch device (CUDA or CPU)
  • ensure_directory()        - Create directory if it doesn't exist
  • print_section_header()    - Print formatted section header with logger
  • count_model_parameters()  - Count trainable parameters in a model
  • format_metric()           - Format metric value for display
"""

import logging
import os

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DEVICE MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """
    Get the optimal device for PyTorch operations.

    Returns
    -------
    torch.device : CUDA if available, otherwise CPU
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    return device


# ─────────────────────────────────────────────────────────────────────────────
# DIRECTORY MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def ensure_directory(path: str) -> str:
    """
    Create directory if it doesn't exist.

    Parameters
    ----------
    path : directory path to create

    Returns
    -------
    path : the directory path (created or already exists)
    """
    os.makedirs(path, exist_ok=True)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def print_section_header(title: str, width: int = 60):
    """
    Print a formatted section header using the logger.

    Parameters
    ----------
    title : section title text
    width : total width of header (default: 60)
    """
    separator = "=" * width
    logger.info(separator)
    logger.info(title)
    logger.info(separator)


def print_step_header(step_number: int, step_name: str, width: int = 60):
    """
    Print a formatted step header for pipeline stages.

    Parameters
    ----------
    step_number : step number (1, 2, 3, etc.)
    step_name   : name of the step
    width       : total width of header (default: 60)
    """
    separator = "=" * width
    header = f"STEP {step_number} – {step_name}"
    logger.info(separator)
    logger.info(header)
    logger.info(separator)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def count_model_parameters(model: nn.Module, verbose: bool = False) -> int:
    """
    Count the total number of trainable parameters in a model.

    Parameters
    ----------
    model   : PyTorch model
    verbose : if True, log the parameter count

    Returns
    -------
    int : total number of trainable parameters
    """
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        logger.info("  Total trainable parameters: %s", f"{total:,}")
    return total


# ─────────────────────────────────────────────────────────────────────────────
# FORMATTING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def format_metric(value: float, decimal_places: int = 4) -> str:
    """
    Format a metric value for clean display.

    Parameters
    ----------
    value          : metric value to format
    decimal_places : number of decimal places (default: 4)

    Returns
    -------
    str : formatted value
    """
    return f"{value:.{decimal_places}f}"


def format_percentage(value: float, decimal_places: int = 2) -> str:
    """
    Format a value as percentage.

    Parameters
    ----------
    value          : value to format (0-1)
    decimal_places : number of decimal places (default: 2)

    Returns
    -------
    str : formatted percentage (e.g., "85.50%")
    """
    return f"{value * 100:.{decimal_places}f}%"


# ─────────────────────────────────────────────────────────────────────────────
# DATA UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def validate_data_shape(X, y_cls, y_reg, name: str = "dataset") -> bool:
    """
    Validate that data shapes are consistent and valid.

    Parameters
    ----------
    X      : feature array
    y_cls  : classification targets
    y_reg  : regression targets
    name   : name of dataset for logging

    Returns
    -------
    bool : True if valid, False otherwise
    """
    if X is None or y_cls is None or y_reg is None:
        logger.error("%s: One or more arrays are None", name)
        return False

    if len(X) != len(y_cls) or len(X) != len(y_reg):
        logger.error(
            "%s: Shape mismatch - X:%d, y_cls:%d, y_reg:%d",
            name, len(X), len(y_cls), len(y_reg),
        )
        return False

    logger.info("%s shape validated: X=%s, y_cls=%s, y_reg=%s", name, X.shape, y_cls.shape, y_reg.shape)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Test device detection
    device = get_device()
    assert isinstance(device, torch.device)

    # Test directory creation
    test_dir = ensure_directory("./test_dir")
    assert os.path.exists(test_dir)

    # Test section header
    print_section_header("TEST SECTION")

    # Test step header
    print_step_header(1, "DATA COLLECTION")

    # Test metric formatting
    print("Formatted metric:", format_metric(0.123456789))
    print("Formatted percentage:", format_percentage(0.8567))

    # Cleanup
    os.rmdir(test_dir)
    print("\n✓ All helper tests passed!")
