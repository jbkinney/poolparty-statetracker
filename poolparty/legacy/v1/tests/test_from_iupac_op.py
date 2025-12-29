"""Tests for from_iupac operation."""

import pytest
from poolparty.operations.from_iupac_op import from_iupac_op, IUPAC_TO_DNA_DICT
from poolparty import Pool


class TestFromIupac:
    """Tests for from_iupac factory function."""
    
    def test_basic_creation(self):
        """Test basic from_iupac pool creation."""
        pool = from_iupac_op('ACGT')
        assert isinstance(pool, Pool)
        assert pool.seq_length == 4
        # All fixed bases = 1 state
        assert pool.operation.num_states == 1
    
    def test_ambiguous_bases(self):
        """Test pool with ambiguous bases."""
        # R = A or G (2 options)
        pool = from_iupac_op('R')
        assert pool.operation.num_states == 2
        
        # N = A, C, G, or T (4 options)
        pool = from_iupac_op('N')
        assert pool.operation.num_states == 4
        
        # RN = 2 × 4 = 8 states
        pool = from_iupac_op('RN')
        assert pool.operation.num_states == 8
    
    def test_num_states_calculation(self):
        """Test that num_states is calculated correctly."""
        # Multiple ambiguous positions
        pool = from_iupac_op('RYN')  # 2 × 2 × 4 = 16
        assert pool.operation.num_states == 16
        
        # Mixed fixed and ambiguous
        pool = from_iupac_op('ARNT')  # 1 × 2 × 4 × 1 = 8
        assert pool.operation.num_states == 8
    
    def test_seq_length(self):
        """Test sequence length matches IUPAC string length."""
        pool = from_iupac_op('ACGTNNNNN')
        assert pool.seq_length == 9
        
        pool.set_state(0)
        assert len(pool.seq) == 9
    
    def test_sequential_mode_enumerates_all(self):
        """Test that sequential mode enumerates all sequences."""
        pool = from_iupac_op('RY', mode='sequential')  # 2 × 2 = 4 states
        seqs = pool.generate_library(num_complete_iterations=1)
        
        assert len(seqs) == 4
        # R = A or G, Y = C or T
        expected = {'AC', 'AT', 'GC', 'GT'}
        assert set(seqs) == expected
    
    def test_all_bases_in_output(self):
        """Test that output contains only valid DNA bases."""
        pool = from_iupac_op('NNNN', mode='sequential')
        seqs = pool.generate_library(num_complete_iterations=1)
        
        for seq in seqs:
            assert all(c in 'ACGT' for c in seq)
    
    def test_uracil_treated_as_thymine(self):
        """Test that U is treated as T."""
        pool = from_iupac_op('U')
        assert pool.operation.num_states == 1
        pool.set_state(0)
        assert pool.seq == 'T'
    
    def test_generate_library_with_seed(self):
        """Test that same seed produces same sequences."""
        pool1 = from_iupac_op('NNNN')
        pool2 = from_iupac_op('NNNN')
        
        seqs1 = pool1.generate_library(num_seqs=10, seed=42)
        seqs2 = pool2.generate_library(num_seqs=10, seed=42)
        
        assert seqs1 == seqs2
    
    def test_empty_string_raises(self):
        """Test that empty string raises error."""
        with pytest.raises(ValueError, match="non-empty"):
            from_iupac_op('')
    
    def test_invalid_character_raises(self):
        """Test that invalid characters raise error."""
        with pytest.raises(ValueError, match="invalid IUPAC"):
            from_iupac_op('ACGTX')
        
        with pytest.raises(ValueError, match="invalid IUPAC"):
            from_iupac_op('acgt')  # lowercase not valid
    
    def test_all_iupac_codes(self):
        """Test that all valid IUPAC codes are accepted."""
        all_codes = 'ACGTURSYWKMBDHVN'
        pool = from_iupac_op(all_codes)
        assert pool.seq_length == len(all_codes)
        
        # Verify it can generate a sequence
        pool.set_state(0)
        assert len(pool.seq) == len(all_codes)


class TestFromIupacCodes:
    """Test specific IUPAC code expansions."""
    
    def test_purine_R(self):
        """Test R expands to A or G."""
        pool = from_iupac_op('R', mode='sequential')
        seqs = set(pool.generate_library(num_complete_iterations=1))
        assert seqs == {'A', 'G'}
    
    def test_pyrimidine_Y(self):
        """Test Y expands to C or T."""
        pool = from_iupac_op('Y', mode='sequential')
        seqs = set(pool.generate_library(num_complete_iterations=1))
        assert seqs == {'C', 'T'}
    
    def test_strong_S(self):
        """Test S expands to G or C."""
        pool = from_iupac_op('S', mode='sequential')
        seqs = set(pool.generate_library(num_complete_iterations=1))
        assert seqs == {'G', 'C'}
    
    def test_weak_W(self):
        """Test W expands to A or T."""
        pool = from_iupac_op('W', mode='sequential')
        seqs = set(pool.generate_library(num_complete_iterations=1))
        assert seqs == {'A', 'T'}
    
    def test_keto_K(self):
        """Test K expands to G or T."""
        pool = from_iupac_op('K', mode='sequential')
        seqs = set(pool.generate_library(num_complete_iterations=1))
        assert seqs == {'G', 'T'}
    
    def test_amino_M(self):
        """Test M expands to A or C."""
        pool = from_iupac_op('M', mode='sequential')
        seqs = set(pool.generate_library(num_complete_iterations=1))
        assert seqs == {'A', 'C'}
    
    def test_not_A_B(self):
        """Test B expands to C, G, or T (not A)."""
        pool = from_iupac_op('B', mode='sequential')
        seqs = set(pool.generate_library(num_complete_iterations=1))
        assert seqs == {'C', 'G', 'T'}
    
    def test_not_C_D(self):
        """Test D expands to A, G, or T (not C)."""
        pool = from_iupac_op('D', mode='sequential')
        seqs = set(pool.generate_library(num_complete_iterations=1))
        assert seqs == {'A', 'G', 'T'}
    
    def test_not_G_H(self):
        """Test H expands to A, C, or T (not G)."""
        pool = from_iupac_op('H', mode='sequential')
        seqs = set(pool.generate_library(num_complete_iterations=1))
        assert seqs == {'A', 'C', 'T'}
    
    def test_not_T_V(self):
        """Test V expands to A, C, or G (not T)."""
        pool = from_iupac_op('V', mode='sequential')
        seqs = set(pool.generate_library(num_complete_iterations=1))
        assert seqs == {'A', 'C', 'G'}
    
    def test_any_N(self):
        """Test N expands to A, C, G, or T."""
        pool = from_iupac_op('N', mode='sequential')
        seqs = set(pool.generate_library(num_complete_iterations=1))
        assert seqs == {'A', 'C', 'G', 'T'}


class TestFromIupacAncestors:
    """Tests for ancestor tracking in from_iupac pools."""
    
    def test_no_parent_pools(self):
        """Test that from_iupac has no parent pools."""
        pool = from_iupac_op('ACGT')
        assert pool.operation.parent_pools == []
    
    def test_ancestors_include_self(self):
        """Test that ancestors include the pool itself."""
        pool = from_iupac_op('NNNN')
        assert pool in pool.ancestors
        assert len(pool.ancestors) == 1
