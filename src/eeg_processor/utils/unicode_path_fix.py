"""
Unicode path normalization utility.

This module provides a monkeypatch for pathlib.Path to handle Unicode normalization
in filenames, preventing issues with French characters and other Unicode variants.
"""

import pathlib
import unicodedata


def apply_unicode_path_fix():
    """Apply Unicode normalization monkeypatch to pathlib.Path.
    
    This prevents the code from breaking if a French character is used in a filename
    by normalizing all path arguments to NFC (Canonical Decomposition, followed by
    Canonical Composition) form.
    """
    if hasattr(pathlib.Path, '_original_new'):
        # Already patched
        return
        
    # Store original method
    pathlib.Path._original_new = pathlib.Path.__new__
    
    def _normalized_path_new(cls, *args, **kwargs):
        """Normalized version of pathlib.Path.__new__ that handles Unicode."""
        if args:
            normalized_arg = unicodedata.normalize('NFC', str(args[0]))
            args = (normalized_arg,) + args[1:]
        return pathlib.Path._original_new(cls, *args, **kwargs)
    
    # Apply the monkeypatch
    pathlib.Path.__new__ = _normalized_path_new


def remove_unicode_path_fix():
    """Remove the Unicode normalization monkeypatch."""
    if hasattr(pathlib.Path, '_original_new'):
        pathlib.Path.__new__ = pathlib.Path._original_new
        delattr(pathlib.Path, '_original_new')