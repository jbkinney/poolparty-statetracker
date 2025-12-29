"""Tests for poolparty alphabet utilities."""

import pytest
from poolparty.alphabet import get_alphabet, validate_sequence, NAMED_ALPHABETS


class TestNamedAlphabets:
    """Test the NAMED_ALPHABETS dictionary."""
    
    def test_dna_alphabet(self):
        """Test DNA alphabet is correct."""
        assert NAMED_ALPHABETS['dna'] == ['A', 'C', 'G', 'T']
    
    def test_rna_alphabet(self):
        """Test RNA alphabet is correct."""
        assert NAMED_ALPHABETS['rna'] == ['A', 'C', 'G', 'U']
    
    def test_protein_alphabet(self):
        """Test protein alphabet has 20 amino acids."""
        assert len(NAMED_ALPHABETS['protein']) == 20
        assert 'A' in NAMED_ALPHABETS['protein']
        assert 'M' in NAMED_ALPHABETS['protein']
        assert 'W' in NAMED_ALPHABETS['protein']
    
    def test_binary_alphabet(self):
        """Test binary alphabet is correct."""
        assert NAMED_ALPHABETS['binary'] == ['0', '1']


class TestGetAlphabetNamedAlphabets:
    """Test get_alphabet with named alphabets."""
    
    def test_dna_lowercase(self):
        """Test getting DNA alphabet with lowercase name."""
        result = get_alphabet('dna')
        assert result == ['A', 'C', 'G', 'T']
    
    def test_dna_uppercase(self):
        """Test getting DNA alphabet with uppercase name."""
        result = get_alphabet('DNA')
        assert result == ['A', 'C', 'G', 'T']
    
    def test_dna_mixed_case(self):
        """Test getting DNA alphabet with mixed case name."""
        result = get_alphabet('Dna')
        assert result == ['A', 'C', 'G', 'T']
    
    def test_rna_alphabet(self):
        """Test getting RNA alphabet."""
        result = get_alphabet('rna')
        assert result == ['A', 'C', 'G', 'U']
    
    def test_protein_alphabet(self):
        """Test getting protein alphabet."""
        result = get_alphabet('protein')
        assert len(result) == 20
    
    def test_binary_alphabet(self):
        """Test getting binary alphabet."""
        result = get_alphabet('binary')
        assert result == ['0', '1']
    
    def test_returns_copy(self):
        """Test that get_alphabet returns a copy, not the original."""
        result = get_alphabet('dna')
        result.append('X')
        # Original should be unchanged
        assert NAMED_ALPHABETS['dna'] == ['A', 'C', 'G', 'T']


class TestGetAlphabetCustomString:
    """Test get_alphabet with custom string input."""
    
    def test_custom_string(self):
        """Test getting alphabet from custom string."""
        result = get_alphabet('XYZ')
        assert result == ['X', 'Y', 'Z']
    
    def test_custom_string_ab(self):
        """Test getting alphabet from AB string."""
        result = get_alphabet('AB')
        assert result == ['A', 'B']
    
    def test_custom_numeric_string(self):
        """Test getting alphabet from numeric string."""
        result = get_alphabet('0123456789')
        assert result == list('0123456789')


class TestGetAlphabetSequence:
    """Test get_alphabet with sequence input."""
    
    def test_list_of_characters(self):
        """Test getting alphabet from list of characters."""
        result = get_alphabet(['X', 'Y', 'Z'])
        assert result == ['X', 'Y', 'Z']
    
    def test_tuple_of_characters(self):
        """Test getting alphabet from tuple of characters."""
        result = get_alphabet(('A', 'B', 'C'))
        assert result == ['A', 'B', 'C']


class TestGetAlphabetErrors:
    """Test get_alphabet error handling."""
    
    def test_single_character_error(self):
        """Test error for alphabet with single character."""
        with pytest.raises(ValueError, match="at least 2 characters"):
            get_alphabet(['A'])
    
    def test_empty_list_error(self):
        """Test error for empty alphabet."""
        with pytest.raises(ValueError, match="at least 2 characters"):
            get_alphabet([])
    
    def test_duplicate_characters_error(self):
        """Test error for duplicate characters."""
        with pytest.raises(ValueError, match="duplicate characters"):
            get_alphabet(['A', 'B', 'A'])
    
    def test_multi_character_element_error(self):
        """Test error for multi-character element."""
        with pytest.raises(ValueError, match="single characters"):
            get_alphabet(['A', 'BC', 'D'])
    
    def test_non_string_element_error(self):
        """Test error for non-string element."""
        # Beartype may catch this before our validation
        with pytest.raises(Exception):  # Could be ValueError or BeartypeCallHintParamViolation
            get_alphabet(['A', 1, 'C'])


class TestValidateSequence:
    """Test validate_sequence function."""
    
    def test_valid_dna_sequence(self):
        """Test validation of valid DNA sequence."""
        alphabet = get_alphabet('dna')
        # Should not raise
        validate_sequence('ACGTACGT', alphabet)
    
    def test_valid_rna_sequence(self):
        """Test validation of valid RNA sequence."""
        alphabet = get_alphabet('rna')
        validate_sequence('ACGUACGU', alphabet)
    
    def test_valid_custom_sequence(self):
        """Test validation with custom alphabet."""
        alphabet = get_alphabet('XY')
        validate_sequence('XYXYXY', alphabet)
    
    def test_empty_sequence_valid(self):
        """Test that empty sequence is valid."""
        alphabet = get_alphabet('dna')
        validate_sequence('', alphabet)
    
    def test_invalid_character(self):
        """Test error for invalid character in sequence."""
        alphabet = get_alphabet('dna')
        with pytest.raises(ValueError, match="invalid characters"):
            validate_sequence('ACGTX', alphabet)
    
    def test_invalid_multiple_characters(self):
        """Test error shows all invalid characters."""
        alphabet = get_alphabet('dna')
        with pytest.raises(ValueError, match="invalid characters"):
            validate_sequence('ACXYZ', alphabet)
    
    def test_lowercase_invalid_for_dna(self):
        """Test that lowercase is invalid for DNA alphabet."""
        alphabet = get_alphabet('dna')
        with pytest.raises(ValueError, match="invalid characters"):
            validate_sequence('acgt', alphabet)
    
    def test_error_message_shows_valid_alphabet(self):
        """Test that error message shows valid alphabet."""
        alphabet = get_alphabet('dna')
        with pytest.raises(ValueError, match="Valid alphabet: ACGT"):
            validate_sequence('X', alphabet)


class TestAlphabetModuleExports:
    """Test that module exports are correct."""
    
    def test_get_alphabet_exported(self):
        """Test get_alphabet is exported."""
        from poolparty import get_alphabet
        assert callable(get_alphabet)
    
    def test_named_alphabets_exported(self):
        """Test NAMED_ALPHABETS is exported."""
        from poolparty import NAMED_ALPHABETS
        assert isinstance(NAMED_ALPHABETS, dict)

