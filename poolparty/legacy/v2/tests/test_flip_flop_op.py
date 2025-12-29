"""Tests for flip_flop operation."""

import pytest
from poolparty.operations import from_seqs, flip_flop
from poolparty import Pool
from poolparty.utils import reverse_complement_iupac


class TestFlipFlopOp:
    """Tests for flip_flop factory function."""
    
    def test_basic_creation(self):
        """Test basic flip_flop pool creation."""
        parent = from_seqs(['AACC', 'GGTT'])
        pool = flip_flop(parent)
        assert isinstance(pool, Pool)
        assert pool.seq_length == 4
        assert pool.operation.num_states == 2
    
    def test_forward_probability_one(self):
        """Test forward_probability=1.0 keeps sequences unchanged."""
        parent = from_seqs(['AACC'])
        pool = flip_flop(parent, forward_probability=1.0)
        result_df = pool.generate_library(num_seqs=10, seed=42)
        seqs = list(result_df['seq'])
        
        for seq in seqs:
            assert seq == 'AACC'
    
    def test_forward_probability_zero(self):
        """Test forward_probability=0.0 always reverse complements."""
        parent = from_seqs(['AACC'])
        pool = flip_flop(parent, forward_probability=0.0)
        result_df = pool.generate_library(num_seqs=10, seed=42)
        seqs = list(result_df['seq'])
        
        # Reverse complement of AACC is GGTT
        for seq in seqs:
            assert seq == 'GGTT'
    
    def test_forward_probability_half(self):
        """Test forward_probability=0.5 gives mix of forward and reverse."""
        parent = from_seqs(['AACC'])
        pool = flip_flop(parent, forward_probability=0.5)
        result_df = pool.generate_library(num_seqs=100, seed=42)
        seqs = list(result_df['seq'])
        
        forward_count = sum(1 for seq in seqs if seq == 'AACC')
        reverse_count = sum(1 for seq in seqs if seq == 'GGTT')
        
        assert forward_count > 0
        assert reverse_count > 0
        assert forward_count + reverse_count == 100
    
    def test_forward_probability_biased(self):
        """Test that forward_probability works correctly for biased values."""
        parent = from_seqs(['AACC'])
        pool = flip_flop(parent, forward_probability=0.9)
        result_df = pool.generate_library(num_seqs=1000, seed=42)
        seqs = list(result_df['seq'])
        
        forward_count = sum(1 for seq in seqs if seq == 'AACC')
        
        # Should be close to 90%
        assert 0.85 < forward_count / 1000 < 0.95
    
    def test_reproducible_with_seed(self):
        """Test that same seed produces same sequences."""
        parent1 = from_seqs(['AACC', 'GGTT', 'ACGT'])
        parent2 = from_seqs(['AACC', 'GGTT', 'ACGT'])
        pool1 = flip_flop(parent1, forward_probability=0.5)
        pool2 = flip_flop(parent2, forward_probability=0.5)
        
        result_df1 = pool1.generate_library(num_seqs=20, seed=42)
        result_df2 = pool2.generate_library(num_seqs=20, seed=42)
        
        seqs1 = list(result_df1['seq'])
        seqs2 = list(result_df2['seq'])
        
        assert seqs1 == seqs2
    
    def test_different_seeds_different_sequences(self):
        """Test that different seeds produce different sequences."""
        parent = from_seqs(['AACC', 'GGTT', 'ACGT'])
        pool = flip_flop(parent, forward_probability=0.5)
        
        result_df1 = pool.generate_library(num_seqs=20, seed=42)
        result_df2 = pool.generate_library(num_seqs=20, seed=123)
        
        seqs1 = list(result_df1['seq'])
        seqs2 = list(result_df2['seq'])
        
        # Should have some different sequences
        assert seqs1 != seqs2


class TestFlipFlopOpSequentialMode:
    """Tests for flip_flop operation in sequential mode."""
    
    def test_sequential_mode_state_zero(self):
        """Test that sequential mode state 0 produces forward sequences."""
        parent = from_seqs(['AACC'], mode='sequential')
        pool = flip_flop(parent, mode='sequential')
        
        # Generate 1 sequence starting from state 0
        # State 0: parent state 0, flip state 0 (forward)
        result_df = pool.generate_library(num_seqs=1, init_state=0, seed=42)
        seqs = list(result_df['seq'])
        
        assert seqs[0] == 'AACC'
    
    def test_sequential_mode_state_one(self):
        """Test that sequential mode state 1 produces reverse complement."""
        parent = from_seqs(['AACC'], mode='sequential')
        pool = flip_flop(parent, mode='sequential')
        
        # Generate 1 sequence starting from state 1
        # State 1: parent state 0, flip state 1 (reverse complement)
        result_df = pool.generate_library(num_seqs=1, init_state=1, seed=42)
        seqs = list(result_df['seq'])
        
        assert seqs[0] == 'GGTT'
    
    def test_sequential_mode_iterates_both_states(self):
        """Test that sequential mode iterates through both forward and reverse."""
        parent = from_seqs(['AACC'], mode='sequential')
        pool = flip_flop(parent, mode='sequential')
        
        # Generate 2 sequences to get both states
        result_df = pool.generate_library(num_seqs=2, init_state=0, seed=42)
        seqs = list(result_df['seq'])
        
        # State 0: forward, State 1: reverse complement
        assert seqs[0] == 'AACC'
        assert seqs[1] == 'GGTT'
    
    def test_sequential_mode_num_states(self):
        """Test that flip_flop op has exactly 2 states."""
        parent = from_seqs(['AACC'])
        pool = flip_flop(parent, mode='sequential')
        
        assert pool.operation.num_states == 2


class TestFlipFlopOpIUPAC:
    """Tests for IUPAC alphabet handling."""
    
    def test_iupac_standard_bases(self):
        """Test standard DNA bases are complemented correctly."""
        parent = from_seqs(['ACGT'])
        pool = flip_flop(parent, forward_probability=0.0)
        result_df = pool.generate_library(num_seqs=1, seed=42)
        seqs = list(result_df['seq'])
        
        # ACGT reversed and complemented is ACGT (palindrome)
        assert seqs[0] == 'ACGT'
    
    def test_iupac_ambiguity_codes_ry(self):
        """Test R and Y ambiguity codes complement correctly."""
        # R (A,G) <-> Y (C,T)
        parent = from_seqs(['RY'])
        pool = flip_flop(parent, forward_probability=0.0)
        result_df = pool.generate_library(num_seqs=1, seed=42)
        seqs = list(result_df['seq'])
        
        assert seqs[0] == 'RY'  # YR reversed = RY
    
    def test_iupac_ambiguity_codes_sw(self):
        """Test S and W are self-complementary."""
        # S (G,C) <-> S, W (A,T) <-> W
        parent = from_seqs(['SW'])
        pool = flip_flop(parent, forward_probability=0.0)
        result_df = pool.generate_library(num_seqs=1, seed=42)
        seqs = list(result_df['seq'])
        
        assert seqs[0] == 'WS'  # SW reversed with complements = WS
    
    def test_iupac_ambiguity_codes_km(self):
        """Test K and M complement correctly."""
        # K (G,T) <-> M (A,C)
        parent = from_seqs(['KM'])
        pool = flip_flop(parent, forward_probability=0.0)
        result_df = pool.generate_library(num_seqs=1, seed=42)
        seqs = list(result_df['seq'])
        
        assert seqs[0] == 'KM'  # MK reversed = KM
    
    def test_iupac_ambiguity_codes_bv(self):
        """Test B and V complement correctly."""
        # B (C,G,T) <-> V (A,C,G)
        parent = from_seqs(['BV'])
        pool = flip_flop(parent, forward_probability=0.0)
        result_df = pool.generate_library(num_seqs=1, seed=42)
        seqs = list(result_df['seq'])
        
        assert seqs[0] == 'BV'  # VB reversed = BV
    
    def test_iupac_ambiguity_codes_dh(self):
        """Test D and H complement correctly."""
        # D (A,G,T) <-> H (A,C,T)
        parent = from_seqs(['DH'])
        pool = flip_flop(parent, forward_probability=0.0)
        result_df = pool.generate_library(num_seqs=1, seed=42)
        seqs = list(result_df['seq'])
        
        assert seqs[0] == 'DH'  # HD reversed = DH
    
    def test_iupac_n_self_complementary(self):
        """Test N is self-complementary."""
        parent = from_seqs(['NNN'])
        pool = flip_flop(parent, forward_probability=0.0)
        result_df = pool.generate_library(num_seqs=1, seed=42)
        seqs = list(result_df['seq'])
        
        assert seqs[0] == 'NNN'
    
    def test_non_iupac_chars_preserved(self):
        """Test that non-IUPAC characters are passed through unchanged."""
        parent = from_seqs(['AA-CC'])
        pool = flip_flop(parent, forward_probability=0.0)
        result_df = pool.generate_library(num_seqs=1, seed=42)
        seqs = list(result_df['seq'])
        
        # Reversed: CC-AA, then complement: GG-TT
        assert seqs[0] == 'GG-TT'
    
    def test_lowercase_preserved(self):
        """Test that lowercase is handled correctly."""
        parent = from_seqs(['aacc'])
        pool = flip_flop(parent, forward_probability=0.0)
        result_df = pool.generate_library(num_seqs=1, seed=42)
        seqs = list(result_df['seq'])
        
        assert seqs[0] == 'ggtt'


class TestFlipFlopOpValidation:
    """Tests for input validation."""
    
    def test_invalid_forward_probability_negative(self):
        """Test that negative forward_probability raises error."""
        parent = from_seqs(['ACGT'])
        with pytest.raises(ValueError, match="forward_probability must be between 0 and 1"):
            flip_flop(parent, forward_probability=-0.1)
    
    def test_invalid_forward_probability_over_one(self):
        """Test that forward_probability > 1 raises error."""
        parent = from_seqs(['ACGT'])
        with pytest.raises(ValueError, match="forward_probability must be between 0 and 1"):
            flip_flop(parent, forward_probability=1.5)


class TestFlipFlopOpAncestors:
    """Tests for ancestor tracking in flip_flop pools."""
    
    def test_has_parent_pool(self):
        """Test that flip_flop has exactly one parent pool."""
        parent = from_seqs(['ACGT'])
        pool = flip_flop(parent)
        assert len(pool.operation.parent_pools) == 1
        assert pool.operation.parent_pools[0] is parent


class TestReverseComplementIupac:
    """Tests for the reverse_complement_iupac utility function."""
    
    def test_standard_bases(self):
        """Test standard DNA bases."""
        assert reverse_complement_iupac('ACGT') == 'ACGT'
        assert reverse_complement_iupac('AAAA') == 'TTTT'
        assert reverse_complement_iupac('TTTT') == 'AAAA'
        assert reverse_complement_iupac('GGGG') == 'CCCC'
        assert reverse_complement_iupac('CCCC') == 'GGGG'
    
    def test_mixed_case(self):
        """Test mixed case handling."""
        assert reverse_complement_iupac('AcGt') == 'aCgT'
        assert reverse_complement_iupac('aacc') == 'ggtt'
    
    def test_uracil(self):
        """Test U (uracil) is treated as T for complement."""
        assert reverse_complement_iupac('UUUU') == 'AAAA'
        assert reverse_complement_iupac('uuuu') == 'aaaa'
    
    def test_ambiguity_codes(self):
        """Test all IUPAC ambiguity codes."""
        # R <-> Y
        assert reverse_complement_iupac('R') == 'Y'
        assert reverse_complement_iupac('Y') == 'R'
        
        # S <-> S, W <-> W (self-complementary)
        assert reverse_complement_iupac('S') == 'S'
        assert reverse_complement_iupac('W') == 'W'
        
        # K <-> M
        assert reverse_complement_iupac('K') == 'M'
        assert reverse_complement_iupac('M') == 'K'
        
        # B <-> V
        assert reverse_complement_iupac('B') == 'V'
        assert reverse_complement_iupac('V') == 'B'
        
        # D <-> H
        assert reverse_complement_iupac('D') == 'H'
        assert reverse_complement_iupac('H') == 'D'
        
        # N <-> N (self-complementary)
        assert reverse_complement_iupac('N') == 'N'
    
    def test_non_iupac_chars_preserved(self):
        """Test that non-IUPAC characters pass through unchanged."""
        assert reverse_complement_iupac('A-T') == 'A-T'
        assert reverse_complement_iupac('ACG123T') == 'A321CGT'
        assert reverse_complement_iupac('') == ''

