"""AIRLOCALIZE spot detection for Python, with a napari tuner and a batch CLI."""

__version__ = "0.1.0"

from .core import (
    DEFAULT_PARAMS,
    PARAM_KEYS,
    auto_measure_psf,
    detect_frame,
    detect_stack,
    load_params,
    save_params,
)

__all__ = [
    "DEFAULT_PARAMS",
    "PARAM_KEYS",
    "auto_measure_psf",
    "detect_frame",
    "detect_stack",
    "load_params",
    "save_params",
]
