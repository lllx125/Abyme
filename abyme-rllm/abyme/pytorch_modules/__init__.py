"""
PyTorch-dependent modules for Abyme.

This package contains model implementations that require PyTorch and transformers.
These are separated from the main package to allow lightweight usage without heavy dependencies.
"""

# Lazy imports to avoid loading PyTorch unless needed
__all__ = ["HuggingFaceModel"]


def __getattr__(name):
    """Lazy import mechanism for HuggingFaceModel."""
    if name == "HuggingFaceModel":
        from .huggingface_model import HuggingFaceModel
        return HuggingFaceModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
