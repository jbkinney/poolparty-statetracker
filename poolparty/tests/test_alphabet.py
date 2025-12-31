"""Tests for poolparty alphabet utilities."""

import pytest
from poolparty.alphabet import (
    Alphabet, get_alphabet, NAMED_ALPHABETS,
    DNA_COMPLEMENT, RNA_COMPLEMENT, BINARY_COMPLEMENT
)


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


class TestComplementMappings:
    """Test complement mapping constants."""
    
    def test_dna_complement(self):
        """Test DNA Watson-Crick complement."""
        assert DNA_COMPLEMENT == {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    
    def test_rna_complement(self):
        """Test RNA Watson-Crick complement."""
        assert RNA_COMPLEMENT == {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
    
    def test_binary_complement(self):
        """Test binary complement."""
        assert BINARY_COMPLEMENT == {'0': '1', '1': '0'}


class TestGetAlphabetNamedAlphabets:
    """Test get_alphabet with named alphabets."""
    
    def test_dna_lowercase(self):
        """Test getting DNA alphabet with lowercase name."""
        alph = get_alphabet('dna')
        assert alph.chars == ['A', 'C', 'G', 'T']
    
    def test_dna_uppercase(self):
        """Test getting DNA alphabet with uppercase name."""
        alph = get_alphabet('DNA')
        assert alph.chars == ['A', 'C', 'G', 'T']
    
    def test_dna_mixed_case(self):
        """Test getting DNA alphabet with mixed case name."""
        alph = get_alphabet('Dna')
        assert alph.chars == ['A', 'C', 'G', 'T']
    
    def test_rna_alphabet(self):
        """Test getting RNA alphabet."""
        alph = get_alphabet('rna')
        assert alph.chars == ['A', 'C', 'G', 'U']
    
    def test_protein_alphabet(self):
        """Test getting protein alphabet."""
        alph = get_alphabet('protein')
        assert len(alph.chars) == 20
    
    def test_binary_alphabet(self):
        """Test getting binary alphabet."""
        alph = get_alphabet('binary')
        assert alph.chars == ['0', '1']
    
    def test_invalid_name_error(self):
        """Test error for invalid alphabet name."""
        with pytest.raises(ValueError, match="Unknown alphabet"):
            get_alphabet('invalid')


class TestAlphabetClass:
    """Test Alphabet class."""
    
    def test_init_with_chars(self):
        """Test Alphabet initialization with chars."""
        alph = Alphabet(chars=['A', 'B', 'C'])
        assert alph.chars == ['A', 'B', 'C']
        assert alph.size == 3
    
    def test_mutation_map_built(self):
        """Test mutation_map is built correctly."""
        alph = Alphabet(chars=['A', 'B', 'C'])
        assert alph.mutation_map['A'] == ['B', 'C']
        assert alph.mutation_map['B'] == ['A', 'C']
        assert alph.mutation_map['C'] == ['A', 'B']
    
    def test_get_mutations(self):
        """Test get_mutations method."""
        alph = Alphabet(chars=['A', 'B', 'C'])
        assert alph.get_mutations('A') == ['B', 'C']
    
    def test_get_mutations_invalid_char(self):
        """Test get_mutations with invalid character."""
        alph = Alphabet(chars=['A', 'B', 'C'])
        with pytest.raises(ValueError, match="not in alphabet"):
            alph.get_mutations('X')
    
    def test_complement_default_none(self):
        """Test complement is None by default for custom alphabets."""
        alph = Alphabet(chars=['A', 'B', 'C'])
        assert alph.complement is None
    
    def test_complement_custom(self):
        """Test custom complement mapping."""
        alph = Alphabet(
            chars=['A', 'B', 'C'],
            complement={'A': 'C', 'B': 'B', 'C': 'A'}
        )
        assert alph.get_complement('A') == 'C'
        assert alph.get_complement('B') == 'B'
        assert alph.get_complement('C') == 'A'
    
    def test_complement_missing_raises(self):
        """Test error when complement mapping is incomplete."""
        with pytest.raises(ValueError, match="missing for character"):
            Alphabet(
                chars=['A', 'B', 'C'],
                complement={'A': 'C', 'B': 'B'}  # Missing C
            )
    
    def test_complement_invalid_target_raises(self):
        """Test error when complement target is not in alphabet."""
        with pytest.raises(ValueError, match="not in alphabet"):
            Alphabet(
                chars=['A', 'B'],
                complement={'A': 'X', 'B': 'A'}  # X not in alphabet
            )
    
    def test_get_complement_no_mapping(self):
        """Test get_complement raises when no complement defined."""
        alph = Alphabet(chars=['A', 'B', 'C'])
        with pytest.raises(ValueError, match="no complement mapping"):
            alph.get_complement('A')
    
    def test_ignore_chars_default(self):
        """Test default ignore chars."""
        alph = Alphabet(chars=['A', 'B'])
        assert alph.ignore_chars == {'-', '.', ' '}
    
    def test_ignore_chars_custom(self):
        """Test custom ignore chars."""
        alph = Alphabet(chars=['A', 'B'], ignore_chars=['X', 'Y'])
        assert alph.ignore_chars == {'X', 'Y'}
    
    def test_repr(self):
        """Test Alphabet repr."""
        alph = Alphabet(chars=['A', 'B', 'C'])
        assert repr(alph) == "Alphabet(ABC)"


class TestAlphabetValidation:
    """Test Alphabet validation."""
    
    def test_single_character_error(self):
        """Test error for alphabet with single character."""
        with pytest.raises(ValueError, match="at least 2 characters"):
            Alphabet(chars=['A'])
    
    def test_empty_list_error(self):
        """Test error for empty alphabet."""
        with pytest.raises(ValueError, match="at least 2 characters"):
            Alphabet(chars=[])
    
    def test_duplicate_characters_error(self):
        """Test error for duplicate characters."""
        with pytest.raises(ValueError, match="duplicate characters"):
            Alphabet(chars=['A', 'B', 'A'])
    
    def test_multi_character_element_error(self):
        """Test error for multi-character element."""
        with pytest.raises(ValueError, match="single characters"):
            Alphabet(chars=['A', 'BC', 'D'])


class TestAlphabetValidateSequence:
    """Test Alphabet.validate_sequence method."""
    
    def test_valid_sequence(self):
        """Test validation of valid sequence."""
        alph = get_alphabet('dna')
        alph.validate_sequence('ACGTACGT')  # Should not raise
    
    def test_valid_with_ignore_chars(self):
        """Test validation allows ignore characters."""
        alph = get_alphabet('dna')
        alph.validate_sequence('ACGT-ACGT')  # Should not raise
        alph.validate_sequence('A.C.G.T')  # Should not raise
    
    def test_empty_sequence_valid(self):
        """Test that empty sequence is valid."""
        alph = get_alphabet('dna')
        alph.validate_sequence('')
    
    def test_invalid_character(self):
        """Test error for invalid character in sequence."""
        alph = get_alphabet('dna')
        with pytest.raises(ValueError, match="invalid characters"):
            alph.validate_sequence('ACGTX')
    
    def test_lowercase_invalid_for_dna(self):
        """Test that lowercase is invalid for DNA alphabet."""
        alph = get_alphabet('dna')
        with pytest.raises(ValueError, match="invalid characters"):
            alph.validate_sequence('acgt')
    
    def test_error_message_shows_valid_alphabet(self):
        """Test that error message shows valid alphabet."""
        alph = get_alphabet('dna')
        with pytest.raises(ValueError, match="Valid alphabet: ACGT"):
            alph.validate_sequence('X')


class TestAlphabetSeqLength:
    """Test Alphabet.get_seq_length method."""
    
    def test_simple_sequence(self):
        """Test length of simple sequence."""
        alph = get_alphabet('dna')
        assert alph.get_seq_length('ACGT') == 4
    
    def test_sequence_with_gaps(self):
        """Test length ignores gap characters."""
        alph = get_alphabet('dna')
        assert alph.get_seq_length('AC-GT') == 4
        assert alph.get_seq_length('A-C-G-T') == 4
        assert alph.get_seq_length('---ACGT---') == 4
    
    def test_sequence_with_dots(self):
        """Test length ignores dot characters."""
        alph = get_alphabet('dna')
        assert alph.get_seq_length('AC.GT') == 4
        assert alph.get_seq_length('...ACGT') == 4
    
    def test_sequence_with_spaces(self):
        """Test length ignores space characters."""
        alph = get_alphabet('dna')
        assert alph.get_seq_length('AC GT') == 4
        assert alph.get_seq_length('A C G T') == 4
    
    def test_sequence_with_mixed_ignore_chars(self):
        """Test length with mixed ignore characters."""
        alph = get_alphabet('dna')
        assert alph.get_seq_length('A-C.G T') == 4
    
    def test_empty_sequence(self):
        """Test length of empty sequence."""
        alph = get_alphabet('dna')
        assert alph.get_seq_length('') == 0
    
    def test_only_gaps(self):
        """Test sequence with only gaps."""
        alph = get_alphabet('dna')
        assert alph.get_seq_length('---') == 0


class TestAlphabetValidSeqPositions:
    """Test Alphabet.get_valid_seq_positions method."""
    
    def test_simple_sequence(self):
        """Test positions in simple sequence."""
        alph = get_alphabet('dna')
        assert alph.get_valid_seq_positions('ACGT') == [0, 1, 2, 3]
    
    def test_sequence_with_gaps(self):
        """Test positions skip gap characters."""
        alph = get_alphabet('dna')
        assert alph.get_valid_seq_positions('AC-GT') == [0, 1, 3, 4]
        assert alph.get_valid_seq_positions('A-C-G-T') == [0, 2, 4, 6]
    
    def test_leading_gaps(self):
        """Test positions with leading gaps."""
        alph = get_alphabet('dna')
        assert alph.get_valid_seq_positions('---ACGT') == [3, 4, 5, 6]
    
    def test_trailing_gaps(self):
        """Test positions with trailing gaps."""
        alph = get_alphabet('dna')
        assert alph.get_valid_seq_positions('ACGT---') == [0, 1, 2, 3]
    
    def test_sequence_with_dots(self):
        """Test positions skip dot characters."""
        alph = get_alphabet('dna')
        assert alph.get_valid_seq_positions('AC.GT') == [0, 1, 3, 4]
    
    def test_sequence_with_spaces(self):
        """Test positions skip space characters."""
        alph = get_alphabet('dna')
        assert alph.get_valid_seq_positions('AC GT') == [0, 1, 3, 4]
    
    def test_empty_sequence(self):
        """Test positions of empty sequence."""
        alph = get_alphabet('dna')
        assert alph.get_valid_seq_positions('') == []
    
    def test_only_gaps(self):
        """Test sequence with only gaps."""
        alph = get_alphabet('dna')
        assert alph.get_valid_seq_positions('---') == []
    
    def test_custom_ignore_chars(self):
        """Test with custom ignore characters."""
        alph = Alphabet(chars=['A', 'B'], ignore_chars=['X', 'Y'])
        assert alph.get_valid_seq_positions('AXBYA') == [0, 2, 4]


class TestNamedAlphabetComplement:
    """Test complement mappings for named alphabets."""
    
    def test_dna_has_complement(self):
        """Test DNA alphabet has Watson-Crick complement."""
        alph = get_alphabet('dna')
        assert alph.complement is not None
        assert alph.get_complement('A') == 'T'
        assert alph.get_complement('T') == 'A'
        assert alph.get_complement('G') == 'C'
        assert alph.get_complement('C') == 'G'
    
    def test_rna_has_complement(self):
        """Test RNA alphabet has Watson-Crick complement."""
        alph = get_alphabet('rna')
        assert alph.complement is not None
        assert alph.get_complement('A') == 'U'
        assert alph.get_complement('U') == 'A'
        assert alph.get_complement('G') == 'C'
        assert alph.get_complement('C') == 'G'
    
    def test_binary_has_complement(self):
        """Test binary alphabet has complement."""
        alph = get_alphabet('binary')
        assert alph.complement is not None
        assert alph.get_complement('0') == '1'
        assert alph.get_complement('1') == '0'
    
    def test_protein_no_complement(self):
        """Test protein alphabet has no complement."""
        alph = get_alphabet('protein')
        assert alph.complement is None


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
