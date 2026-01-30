"""Tests for poolparty type definitions."""

import pytest
from collections.abc import Sequence, Callable


class TestTypeAliases:
    """Test that type aliases are properly defined and importable."""
    
    def test_mode_type_import(self):
        """Test ModeType can be imported and used."""
        from poolparty.types import ModeType
        # ModeType is a Literal type - verify it exists
        assert ModeType is not None
    
    def test_alphabet_type_import(self):
        """Test AlphabetType can be imported and used."""
        from poolparty.types import AlphabetType
        assert AlphabetType is not None
    
    def test_filter_func_import(self):
        """Test FilterFunc can be imported and used."""
        from poolparty.types import FilterFunc
        assert FilterFunc is not None
    
    def test_beartype_reexport(self):
        """Test beartype is re-exported from types module."""
        from poolparty.types import beartype
        assert beartype is not None
        assert callable(beartype)
    
    def test_typing_constructs_reexport(self):
        """Test typing constructs are re-exported."""
        from poolparty.types import Union, Optional, Sequence, Callable
        assert Union is not None
        assert Optional is not None
        assert Sequence is not None
        assert Callable is not None


class TestAllExports:
    """Test __all__ exports are complete."""
    
    def test_all_exports_defined(self):
        """Test that __all__ is defined and complete."""
        from poolparty import types
        
        expected = [
            'beartype',
            'Union',
            'Optional',
            'Sequence',
            'Callable',
            'ModeType',
            'AlphabetType',
            'FilterFunc',
        ]
        
        for name in expected:
            assert name in types.__all__, f"{name} missing from __all__"
            assert hasattr(types, name), f"{name} not exported"

