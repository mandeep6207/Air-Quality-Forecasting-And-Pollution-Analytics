"""
utils.py
--------
Shared utility functions for the Air Quality Forecasting project.
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Logging ───────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """Return a consistently formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


# ── Path helpers ──────────────────────────────────────────────────────────────

def project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def ensure_dir(path) -> Path:
    """Create directory (and parents) if it does not exist; return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── Persistence helpers ───────────────────────────────────────────────────────

def save_json(data: Dict[str, Any], filepath) -> None:
    """Serialise *data* to a JSON file, creating parent dirs as needed."""
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=4)


def load_json(filepath) -> Dict[str, Any]:
    """Load and return a JSON file as a Python dict."""
    with open(filepath, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ── Plotting helpers ──────────────────────────────────────────────────────────

def save_figure(fig: plt.Figure, filepath, dpi: int = 150) -> None:
    """Save a matplotlib figure to *filepath*, creating parent dirs."""
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def set_plot_style() -> None:
    """Apply a clean, consistent style to all matplotlib figures."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#f8f9fa",
            "axes.grid": True,
            "grid.color": "#dee2e6",
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
        }
    )


# ── Numeric helpers ───────────────────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return Root Mean Squared Error."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def safe_divide(a: float, b: float, default: float | None = None) -> float | None:
    """Safely divide two numbers, returning ``default`` on error (e.g., division by zero).

    Useful to avoid exceptions in aggregation pipelines where a missing or zero
    denominator might otherwise interrupt processing.
    """
    try:
        return a / b
    except Exception:
        return default


def format_percentage(value: float, digits: int = 1) -> str:
    """Format a ratio or percentage for display.

    Examples:
        >>> format_percentage(0.1234)
        '12.3%'
    """
    try:
        return f"{round(float(value) * 100, digits):.{digits}f}%"
    except Exception:
        return "N/A"


def timed(logger_name: str = "utils"):
    """Decorator to log execution time of a function under the given logger name.

    Example:
        @timed('train')
        def train_model(...):
            ...
    """

    def _decorator(func):
        def _wrapped(*args, **kwargs):
            log = get_logger(logger_name)
            start = __import__('time').time()
            result = func(*args, **kwargs)
            elapsed = __import__('time').time() - start
            log.info("%s completed in %.2fs", func.__name__, elapsed)
            return result

        return _wrapped

    return _decorator
