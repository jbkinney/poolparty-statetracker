"""Tests for from_iupac operation."""

import pytest
from poolparty.operations.from_iupac_op import from_iupac, FromIupacOp, IUPAC_TO_DNA
from poolparty import Pool


class TestFromIupac:
    """Tests for from_iupac factory function."""
    
    def test_basic_creation(self):
        """Test basic from_iupac pool creation."""
        pool = from_iupac('ACGT')
        assert isinstance(pool, Pool)
        assert pool.seq_length == 4
        # All fixed bases = 1 state
        assert pool.operation.num_states == 1
    
    def test_ambiguous_bases(self):
        """Test pool with ambiguous bases."""
        # R = A or G (2 options)
        pool = from_iupac('R')
        assert pool.operation.num_states == 2
        
        # N = A, C, G, or T (4 options)
        pool = from_iupac('N')
        assert pool.operation.num_states == 4
        
        # RN = 2 × 4 = 8 states
        pool = from_iupac('RN')
        assert pool.operation.num_states == 8
    
    def test_num_states_calculation(self):
        """Test that num_states is calculated correctly."""
        # Multiple ambiguous positions
        pool = from_iupac('RYN')  # 2 × 2 × 4 = 16
        assert pool.operation.num_states == 16
        
        # Mixed fixed and ambiguous
        pool = from_iupac('ARNT')  # 1 × 2 × 4 × 1 = 8
        assert pool.operation.num_states == 8
    
    def test_seq_length(self):
        """Test sequence length matches IUPAC string length."""
        pool = from_iupac('ACGTNNNNN')
        assert pool.seq_length == 9
    
    def test_sequential_mode_enumerates_all(self):
        """Test that sequential mode enumerates all sequences."""
        pool = from_iupac('RY', mode='sequential')  # 2 × 2 = 4 states
        result_df = pool.generate_library(num_seqs=4, seed=0)
        seqs = list(result_df['seq'])
        
        assert len(seqs) == 4
        # R = A or G, Y = C or T
        expected = {'AC', 'AT', 'GC', 'GT'}
        assert set(seqs) == expected
    
    def test_all_bases_in_output(self):
        """Test that output contains only valid DNA bases."""
        pool = from_iupac('NNNN', mode='sequential')
        result_df = pool.generate_library(num_seqs=256, seed=0)  # 4^4 = 256 states
        seqs = list(result_df['seq'])
        
        for seq in seqs:
            assert all(c in 'ACGT' for c in seq)
    
    def test_uracil_treated_as_thymine(self):
        """Test that U is treated as T."""
        pool = from_iupac('U')
        assert pool.operation.num_states == 1
        result_df = pool.generate_library(num_seqs=1, seed=0)
        seq = result_df['seq'].iloc[0]
        assert seq == 'T'
    
    def test_generate_library_with_seed(self):
        """Test that same seed produces same sequences."""
        pool1 = from_iupac('NNNN')
        pool2 = from_iupac('NNNN')
        
        result_df1 = pool1.generate_library(num_seqs=10, seed=42)
        result_df2 = pool2.generate_library(num_seqs=10, seed=42)
        
        seqs1 = list(result_df1['seq'])
        seqs2 = list(result_df2['seq'])
        
        assert seqs1 == seqs2
    
    def test_different_seeds_different_sequences(self):
        """Test that different seeds produce different sequences."""
        pool = from_iupac('NNNN')
        
        result_df1 = pool.generate_library(num_seqs=20, seed=42)
        result_df2 = pool.generate_library(num_seqs=20, seed=123)
        
        seqs1 = list(result_df1['seq'])
        seqs2 = list(result_df2['seq'])
        
        # Should have some different sequences
        assert seqs1 != seqs2
    
    def test_random_mode_uniform_sampling(self):
        """Test that random mode samples uniformly from allowed bases."""
        # R = A or G (should be roughly 50/50)
        pool = from_iupac('R')
        result_df = pool.generate_library(num_seqs=1000, seed=42)
        seqs = list(result_df['seq'])
        
        # Should only have A or G
        unique_bases = set(seqs)
        assert unique_bases == {'A', 'G'}
        
        # Should have roughly equal distribution
        a_count = sum(1 for s in seqs if s == 'A')
        assert 0.4 < a_count / 1000 < 0.6
    
    def test_empty_string_raises(self):
        """Test that empty string raises error."""
        with pytest.raises(ValueError, match="non-empty"):
            from_iupac('')
    
    def test_invalid_character_raises(self):
        """Test that invalid characters raise error."""
        with pytest.raises(ValueError, match="invalid IUPAC"):
            from_iupac('ACGTX')
    
    def test_lowercase_converted(self):
        """Test that lowercase is converted to uppercase."""
        pool = from_iupac('acgt')
        result_df = pool.generate_library(num_seqs=1, seed=0)
        seq = result_df['seq'].iloc[0]
        assert seq == 'ACGT'
    
    def test_all_iupac_codes(self):
        """Test that all valid IUPAC codes are accepted."""
        all_codes = 'ACGTURSYWKMBDHVN'
        pool = from_iupac(all_codes)
        assert pool.seq_length == len(all_codes)
    
    def test_stores_iupac_seq(self):
        """Test that the IUPAC string is stored."""
        pool = from_iupac('NNNN')
        assert pool.operation.iupac_seq == 'NNNN'
    
    def test_deterministic_sequence(self):
        """Test that deterministic IUPAC generates expected sequence."""
        pool = from_iupac('ACGT')
        result_df = pool.generate_library(num_seqs=10, seed=42)
        seqs = list(result_df['seq'])
        
        # All sequences should be 'ACGT' since these are deterministic bases
        for seq in seqs:
            assert seq == 'ACGT'


class TestFromIupacSequentialMode:
    """Tests for sequential mode enumeration."""
    
    def test_sequential_complete_enumeration(self):
        """Test that sequential mode enumerates all combinations."""
        pool = from_iupac('NN', mode='sequential')  # 4 × 4 = 16 states
        result_df = pool.generate_library(num_seqs=16, seed=0)
        seqs = list(result_df['seq'])
        
        # Should have all 16 combinations
        expected = set()
        for b1 in 'ACGT':
            for b2 in 'ACGT':
                expected.add(b1 + b2)
        
        assert set(seqs) == expected
    
    def test_sequential_mixed_radix_order(self):
        """Test that sequential enumeration follows mixed-radix ordering."""
        pool = from_iupac('RY', mode='sequential')  # R=A/G, Y=C/T
        
        # State 0 should give first base at each position
        # Using the IUPAC_TO_DNA order: R=['A','G'], Y=['C','T']
        result_df = pool.generate_library(num_seqs=4, seed=0)
        seqs = list(result_df['seq'])
        
        # Check we got all expected sequences
        assert set(seqs) == {'AC', 'AT', 'GC', 'GT'}
    
    def test_sequential_wraps_around(self):
        """Test that sequential state wraps around num_states."""
        pool = from_iupac('R', mode='sequential')  # 2 states
        # Request more sequences than states to test wrapping
        result_df = pool.generate_library(num_seqs=6, seed=0)
        seqs = list(result_df['seq'])
        
        # Should cycle through states: A, G, A, G, A, G
        assert len(seqs) == 6
        # Every pair should have both A and G
        assert set(seqs) == {'A', 'G'}


class TestFromIupacCodes:
    """Test specific IUPAC code expansions."""
    
    def test_purine_R(self):
        """Test R expands to A or G."""
        pool = from_iupac('R', mode='sequential')
        result_df = pool.generate_library(num_seqs=2, seed=0)
        seqs = set(result_df['seq'])
        assert seqs == {'A', 'G'}
    
    def test_pyrimidine_Y(self):
        """Test Y expands to C or T."""
        pool = from_iupac('Y', mode='sequential')
        result_df = pool.generate_library(num_seqs=2, seed=0)
        seqs = set(result_df['seq'])
        assert seqs == {'C', 'T'}
    
    def test_strong_S(self):
        """Test S expands to G or C."""
        pool = from_iupac('S', mode='sequential')
        result_df = pool.generate_library(num_seqs=2, seed=0)
        seqs = set(result_df['seq'])
        assert seqs == {'G', 'C'}
    
    def test_weak_W(self):
        """Test W expands to A or T."""
        pool = from_iupac('W', mode='sequential')
        result_df = pool.generate_library(num_seqs=2, seed=0)
        seqs = set(result_df['seq'])
        assert seqs == {'A', 'T'}
    
    def test_keto_K(self):
        """Test K expands to G or T."""
        pool = from_iupac('K', mode='sequential')
        result_df = pool.generate_library(num_seqs=2, seed=0)
        seqs = set(result_df['seq'])
        assert seqs == {'G', 'T'}
    
    def test_amino_M(self):
        """Test M expands to A or C."""
        pool = from_iupac('M', mode='sequential')
        result_df = pool.generate_library(num_seqs=2, seed=0)
        seqs = set(result_df['seq'])
        assert seqs == {'A', 'C'}
    
    def test_not_A_B(self):
        """Test B expands to C, G, or T (not A)."""
        pool = from_iupac('B', mode='sequential')
        result_df = pool.generate_library(num_seqs=3, seed=0)
        seqs = set(result_df['seq'])
        assert seqs == {'C', 'G', 'T'}
    
    def test_not_C_D(self):
        """Test D expands to A, G, or T (not C)."""
        pool = from_iupac('D', mode='sequential')
        result_df = pool.generate_library(num_seqs=3, seed=0)
        seqs = set(result_df['seq'])
        assert seqs == {'A', 'G', 'T'}
    
    def test_not_G_H(self):
        """Test H expands to A, C, or T (not G)."""
        pool = from_iupac('H', mode='sequential')
        result_df = pool.generate_library(num_seqs=3, seed=0)
        seqs = set(result_df['seq'])
        assert seqs == {'A', 'C', 'T'}
    
    def test_not_T_V(self):
        """Test V expands to A, C, or G (not T)."""
        pool = from_iupac('V', mode='sequential')
        result_df = pool.generate_library(num_seqs=3, seed=0)
        seqs = set(result_df['seq'])
        assert seqs == {'A', 'C', 'G'}
    
    def test_any_N(self):
        """Test N expands to A, C, G, or T."""
        pool = from_iupac('N', mode='sequential')
        result_df = pool.generate_library(num_seqs=4, seed=0)
        seqs = set(result_df['seq'])
        assert seqs == {'A', 'C', 'G', 'T'}


class TestFromIupacAncestors:
    """Tests for ancestor tracking in from_iupac pools."""
    
    def test_no_parent_pools(self):
        """Test that from_iupac has no parent pools."""
        pool = from_iupac('ACGT')
        assert pool.operation.parent_pools == []

