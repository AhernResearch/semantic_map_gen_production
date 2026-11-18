"""Focused building blocks for FSRVLN scene-package view adaptation."""

from .conversion import convert
from .policy import SOURCE_VIEW_SELECTION_VERSION

__all__ = ["SOURCE_VIEW_SELECTION_VERSION", "convert"]
