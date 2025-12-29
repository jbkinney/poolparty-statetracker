"""Tests for filter operation."""

import pytest
from poolparty import filter, from_seqs, get_kmers, Pool, reset_op_id_counter
from poolparty.filters import (
    gc_range_filter,
    max_homopolymer_filter,
    min_hamming_dist_filter,
    min_edit_distance_filter,
    avoid_seqs_filter,
)


class TestFilterBasic:
    """Basic tests for filter factory function."""
    
    def setup_method(self):
        """Reset op id counter before each test."""
        reset_op_id_counter()
    
    def test_basic_creation(self):
        """Test basic pool creation with filtering."""
        # Create a pool with some sequences in sequential mode for deterministic behavior
        pool = from_seqs(['AAAA', 'ATAT', 'TTTT', 'GCGC'], mode='sequential')
        
        # Filter to keep only sequences containing 'AT'
        def has_at(seq, filtered_seqs=None):
            return 'AT' in seq
        
        filtered = filter(
            pool, 
            num_candidate_seqs=4, 
            filter_funcs=[has_at], 
            min_num_filtered_seqs=1
        )
        
        assert isinstance(filtered, Pool)
        assert filtered.operation.num_states == 1  # Only 'ATAT' passes
    
    def test_filter_with_sequential_mode(self):
        """Test filtering in sequential mode."""
        pool = from_seqs(['AAAA', 'ATAT', 'GCGC', 'TATA'], mode='sequential')
        
        def has_at(seq, filtered_seqs=None):
            return 'AT' in seq
        
        filtered = filter(
            pool, 
            num_candidate_seqs=4,
            filter_funcs=[has_at], 
            mode='sequential'
        )
        
        # Generate library
        result_df = filtered.generate_library(num_complete_iterations=1)
        
        # Should only have sequences containing 'AT'
        for seq in result_df['seq']:
            assert 'AT' in seq
    
    def test_filter_with_random_mode(self):
        """Test filtering in random mode returns valid sequences."""
        pool = from_seqs(['AAAA', 'ATAT', 'GCGC', 'TATA'])
        
        def has_at(seq, filtered_seqs=None):
            return 'AT' in seq
        
        filtered = filter(
            pool, 
            num_candidate_seqs=4,
            filter_funcs=[has_at], 
            mode='random'
        )
        
        # Generate multiple sequences, all should pass the filter
        result_df = filtered.generate_library(num_seqs=10)
        for seq in result_df['seq']:
            assert 'AT' in seq
    
    def test_multiple_filters(self):
        """Test that multiple filter functions all must pass."""
        pool = from_seqs(['AAAA', 'ATAT', 'ATGC', 'GCGC', 'TATA'], mode='sequential')
        
        def has_at(seq, filtered_seqs=None):
            return 'AT' in seq
        
        def has_gc(seq, filtered_seqs=None):
            return 'GC' in seq
        
        filtered = filter(
            pool, 
            num_candidate_seqs=5,
            filter_funcs=[has_at, has_gc], 
        )
        
        # Only 'ATGC' should pass both filters
        assert filtered.operation.num_states == 1
    
    def test_cumulative_filtered_seqs(self):
        """Test that filter function receives cumulative filtered_seqs."""
        pool = from_seqs(['AAA', 'BBB', 'AAA', 'CCC', 'BBB'], mode='sequential')
        
        # Filter for uniqueness - reject if already in filtered_seqs
        def unique_only(seq, filtered_seqs=None):
            if filtered_seqs is None:
                return True
            return seq not in filtered_seqs
        
        filtered = filter(
            pool, 
            num_candidate_seqs=5,
            filter_funcs=[unique_only], 
            mode='sequential'  # Use sequential mode to iterate through all
        )
        
        # Should only have 3 unique sequences: AAA (first), BBB (first), CCC
        # The second AAA and second BBB should be rejected
        assert filtered.operation.num_states == 3
        
        # Verify uniqueness - generate all 3 unique sequences
        result_df = filtered.generate_library(num_complete_iterations=1)
        unique_seqs = set(result_df['seq'])
        assert unique_seqs == {'AAA', 'BBB', 'CCC'}


class TestFilterValidation:
    """Validation tests for filter."""
    
    def setup_method(self):
        """Reset op id counter before each test."""
        reset_op_id_counter()
    
    def test_no_filters_raises(self):
        """Test that no filters raises ValueError."""
        pool = from_seqs(['AAA', 'TTT'])
        
        with pytest.raises(ValueError, match="At least one filter"):
            filter(pool, num_candidate_seqs=2)
    
    def test_zero_num_candidate_seqs_raises(self):
        """Test that zero num_candidate_seqs raises ValueError."""
        pool = from_seqs(['AAA', 'TTT'])
        
        def always_true(seq, filtered_seqs=None):
            return True
        
        with pytest.raises(ValueError, match="positive"):
            filter(pool, num_candidate_seqs=0, filter_funcs=[always_true])
    
    def test_negative_num_candidate_seqs_raises(self):
        """Test that negative num_candidate_seqs raises ValueError."""
        pool = from_seqs(['AAA', 'TTT'])
        
        def always_true(seq, filtered_seqs=None):
            return True
        
        with pytest.raises(ValueError, match="positive"):
            filter(pool, num_candidate_seqs=-5, filter_funcs=[always_true])
    
    def test_zero_min_num_filtered_seqs_raises(self):
        """Test that zero min_num_filtered_seqs raises ValueError."""
        pool = from_seqs(['AAA', 'TTT'])
        
        def always_true(seq, filtered_seqs=None):
            return True
        
        with pytest.raises(ValueError, match="positive"):
            filter(
                pool, 
                num_candidate_seqs=2,
                filter_funcs=[always_true], 
                min_num_filtered_seqs=0
            )
    
    def test_not_enough_filtered_seqs_raises(self):
        """Test that not meeting min_num_filtered_seqs raises ValueError."""
        pool = from_seqs(['AAA', 'TTT', 'GGG'], mode='sequential')
        
        # Filter that rejects everything
        def always_false(seq, filtered_seqs=None):
            return False
        
        with pytest.raises(ValueError, match="passed filters"):
            filter(
                pool, 
                num_candidate_seqs=3,
                filter_funcs=[always_false], 
                min_num_filtered_seqs=1
            )
    
    def test_min_num_filtered_seqs_threshold(self):
        """Test min_num_filtered_seqs threshold behavior."""
        pool = from_seqs(['AAA', 'AAT', 'ATT', 'TTT'], mode='sequential')
        
        # Filter that only accepts sequences with 'A'
        def has_a(seq, filtered_seqs=None):
            return 'A' in seq
        
        # Should work - 3 sequences pass (AAA, AAT, ATT)
        filtered = filter(
            pool, 
            num_candidate_seqs=4,
            filter_funcs=[has_a], 
            min_num_filtered_seqs=3
        )
        assert filtered.operation.num_states == 3
        
        # Should fail - only 3 pass but need 4
        with pytest.raises(ValueError, match="passed filters"):
            filter(
                pool, 
                num_candidate_seqs=4,
                filter_funcs=[has_a], 
                min_num_filtered_seqs=4
            )


class TestFilterDesignCards:
    """Tests for design card preservation in filter."""
    
    def setup_method(self):
        """Reset op id counter before each test."""
        reset_op_id_counter()
    
    def test_design_cards_preserved(self):
        """Test that design card data from parent pool is preserved."""
        pool = from_seqs(['AAAA', 'ATAT', 'GCGC'], mode='sequential')
        
        def has_at(seq, filtered_seqs=None):
            return 'AT' in seq
        
        filtered = filter(
            pool, 
            num_candidate_seqs=3,
            filter_funcs=[has_at], 
            mode='sequential'
        )
        
        result_df = filtered.generate_library(num_complete_iterations=1)
        
        # Check that filtered_seq_idx and source_seq_idx are present
        assert 'filtered_seq_idx' in result_df.columns or any('filtered_seq_idx' in col for col in result_df.columns)
        assert 'source_seq_idx' in result_df.columns or any('source_seq_idx' in col for col in result_df.columns)


class TestFilterWithDifferentPools:
    """Tests for filter with different input pool types."""
    
    def setup_method(self):
        """Reset op id counter before each test."""
        reset_op_id_counter()
    
    def test_filter_kmers(self):
        """Test filtering with get_kmers pool."""
        # Create a pool of 4-mers
        pool = get_kmers(length=4, alphabet='dna')
        
        # Filter to palindromes
        def is_palindrome(seq, filtered_seqs=None):
            return seq == seq[::-1]
        
        filtered = filter(
            pool, 
            num_candidate_seqs=100,
            filter_funcs=[is_palindrome], 
            min_num_filtered_seqs=1
        )
        
        # Generate and verify
        result_df = filtered.generate_library(num_seqs=10)
        for seq in result_df['seq']:
            assert seq == seq[::-1]
    
    def test_variable_length_sequences(self):
        """Test filtering with variable length sequences."""
        pool = from_seqs(['A', 'AA', 'AAA', 'AAAA'], mode='sequential')
        
        # Filter to sequences with length >= 2
        def min_length_2(seq, filtered_seqs=None):
            return len(seq) >= 2
        
        filtered = filter(
            pool, 
            num_candidate_seqs=4,
            filter_funcs=[min_length_2], 
        )
        
        assert filtered.operation.num_states == 3
        assert filtered.seq_length is None  # Variable length


class TestFilterStateManagement:
    """Tests for state management in filter."""
    
    def setup_method(self):
        """Reset op id counter before each test."""
        reset_op_id_counter()
    
    def test_sequential_mode_state_wrapping(self):
        """Test that sequential mode wraps around correctly."""
        pool = from_seqs(['AAAA', 'ATAT', 'GCGC', 'TATA'], mode='sequential')
        
        def has_at(seq, filtered_seqs=None):
            return 'AT' in seq
        
        filtered = filter(
            pool, 
            num_candidate_seqs=4,
            filter_funcs=[has_at], 
            mode='sequential'
        )
        
        # Generate more sequences than available (should wrap)
        result_df = filtered.generate_library(num_seqs=5)
        assert len(result_df) == 5
        
        # First 2 should match, then wrap
        seqs = list(result_df['seq'])
        assert seqs[0] == seqs[2]  # Wrap around
    
    def test_name_attribute(self):
        """Test name attribute is on operation."""
        pool = from_seqs(['AAA', 'ATG', 'TTT'])
        
        def has_atg(seq, filtered_seqs=None):
            return 'ATG' in seq
        
        filtered = filter(
            pool, 
            num_candidate_seqs=3,
            filter_funcs=[has_atg], 
            name='my_filter'
        )
        
        assert filtered.operation.name == 'my_filter'
    
    def test_no_parent_pools(self):
        """Test that FilterOp has no parent pools (like MixOp)."""
        pool = from_seqs(['AAA', 'TTT'])
        
        def always_true(seq, filtered_seqs=None):
            return True
        
        filtered = filter(
            pool, 
            num_candidate_seqs=2,
            filter_funcs=[always_true], 
        )
        
        # FilterOp pre-generates sequences, so has no parent pools in DAG
        assert filtered.operation.parent_pools == []


#########################################################
# Tests for Built-in Filter Factory Functions
#########################################################

class TestGcRangeFilter:
    """Tests for gc_range_filter factory function."""
    
    def setup_method(self):
        reset_op_id_counter()
    
    def test_gc_range_filter_basic(self):
        """Test gc_range_filter accepts sequences in range."""
        filter_func = gc_range_filter((0.4, 0.6))
        
        # 50% GC - should pass
        assert filter_func("ACGT", None) is True
        assert filter_func("GCGC", None) is False  # 100% GC
        assert filter_func("AAAA", None) is False  # 0% GC
    
    def test_gc_range_filter_edge_cases(self):
        """Test gc_range_filter at boundary values."""
        filter_func = gc_range_filter((0.5, 0.5))
        
        assert filter_func("ACGT", None) is True  # Exactly 50%
        assert filter_func("AACG", None) is True  # Also 50% (A, A, C, G)
        assert filter_func("AAAG", None) is False  # 25% GC
    
    def test_gc_range_filter_with_filter(self):
        """Test gc_range_filter integration with filter."""
        pool = from_seqs(['AAAA', 'ACGT', 'GCGC', 'ATAT'], mode='sequential')
        
        filtered = filter(
            pool,
            num_candidate_seqs=4,
            gc_range=(0.4, 0.6),
            mode='sequential'
        )
        
        # ACGT (50% GC) and ATAT (0% GC won't pass, 50% will)
        result_df = filtered.generate_library(num_complete_iterations=1)
        for seq in result_df['seq']:
            gc = sum(1 for c in seq if c in 'GC') / len(seq)
            assert 0.4 <= gc <= 0.6


class TestMaxHomopolymerFilter:
    """Tests for max_homopolymer_filter factory function."""
    
    def setup_method(self):
        reset_op_id_counter()
    
    def test_max_homopolymer_filter_basic(self):
        """Test max_homopolymer_filter rejects long runs."""
        filter_func = max_homopolymer_filter(3)
        
        assert filter_func("AAACGT", None) is True   # max run = 3
        assert filter_func("AAAACGT", None) is False  # max run = 4
        assert filter_func("ACGT", None) is True      # max run = 1
    
    def test_max_homopolymer_filter_with_filter(self):
        """Test max_homopolymer_filter integration with filter."""
        pool = from_seqs(['AAAA', 'AAAC', 'ACGT', 'TTTT'], mode='sequential')
        
        filtered = filter(
            pool,
            num_candidate_seqs=4,
            max_homopolymer=3,
            mode='sequential'
        )
        
        # Only AAAC and ACGT should pass (max run <= 3)
        assert filtered.operation.num_states == 2


class TestMinHammingDistFilter:
    """Tests for min_hamming_dist_filter factory function."""
    
    def setup_method(self):
        reset_op_id_counter()
    
    def test_min_hamming_dist_filter_basic(self):
        """Test min_hamming_dist_filter checks distance."""
        filter_func = min_hamming_dist_filter(2)
        
        # No filtered_seqs - always passes
        assert filter_func("ACGT", None) is True
        
        # Distance 0 - fails
        assert filter_func("ACGT", ["ACGT"]) is False
        
        # Distance 1 - fails
        assert filter_func("ACGT", ["ACGA"]) is False
        
        # Distance 2 - passes
        assert filter_func("ACGT", ["ACAA"]) is True
        
        # Distance 4 - passes
        assert filter_func("ACGT", ["TGCA"]) is True
    
    def test_min_hamming_dist_filter_different_lengths(self):
        """Test min_hamming_dist_filter ignores different-length sequences."""
        filter_func = min_hamming_dist_filter(2)
        
        # Different lengths - not compared via Hamming
        assert filter_func("ACGT", ["ACG"]) is True
        assert filter_func("ACGT", ["ACGTA"]) is True
    
    def test_min_hamming_dist_filter_with_filter(self):
        """Test min_hamming_dist_filter integration with filter."""
        pool = from_seqs(['AAAA', 'AAAC', 'TTTT', 'CCCC'], mode='sequential')
        
        filtered = filter(
            pool,
            num_candidate_seqs=4,
            min_hamming_dist=3,
            mode='sequential'
        )
        
        # AAAA passes (first), AAAC fails (dist 1), TTTT passes (dist 4), CCCC passes (dist 4)
        assert filtered.operation.num_states == 3


class TestMinEditDistanceFilter:
    """Tests for min_edit_distance_filter factory function."""
    
    def setup_method(self):
        reset_op_id_counter()
    
    def test_min_edit_distance_filter_basic(self):
        """Test min_edit_distance_filter checks distance."""
        filter_func = min_edit_distance_filter(2)
        
        # No filtered_seqs - always passes
        assert filter_func("ACGT", None) is True
        
        # Distance 0 - fails
        assert filter_func("ACGT", ["ACGT"]) is False
        
        # Distance 1 (one substitution) - fails
        assert filter_func("ACGT", ["ACGA"]) is False
        
        # Distance 2 - passes
        assert filter_func("ACGT", ["ACAA"]) is True
    
    def test_min_edit_distance_filter_different_lengths(self):
        """Test min_edit_distance_filter works with different lengths."""
        filter_func = min_edit_distance_filter(2)
        
        # Distance 1 (one deletion) - fails
        assert filter_func("ACGT", ["ACG"]) is False
        
        # Distance 2 - passes
        assert filter_func("ACGT", ["AC"]) is True
    
    def test_min_edit_distance_filter_with_filter(self):
        """Test min_edit_distance_filter integration with filter."""
        pool = from_seqs(['AAAA', 'AAAC', 'TTTT', 'CCCC'], mode='sequential')
        
        filtered = filter(
            pool,
            num_candidate_seqs=4,
            min_edit_distance=3,
            mode='sequential'
        )
        
        # AAAA passes (first), AAAC fails (dist 1), TTTT passes, CCCC passes
        assert filtered.operation.num_states == 3


class TestAvoidSeqsFilter:
    """Tests for avoid_seqs_filter factory function."""
    
    def setup_method(self):
        reset_op_id_counter()
    
    def test_avoid_seqs_filter_exact_match(self):
        """Test avoid_seqs_filter rejects exact matches by default."""
        filter_func = avoid_seqs_filter(["ACGT", "TGCA"])
        
        assert filter_func("ACGT", None) is False  # Exact match
        assert filter_func("TGCA", None) is False  # Exact match
        assert filter_func("AAAA", None) is True   # No match
    
    def test_avoid_seqs_filter_with_hamming_dist(self):
        """Test avoid_seqs_filter with Hamming distance."""
        filter_func = avoid_seqs_filter(["ACGT"], avoid_min_hamming_dist=2)
        
        assert filter_func("ACGT", None) is False  # Distance 0
        assert filter_func("ACGA", None) is False  # Distance 1
        assert filter_func("ACAA", None) is True   # Distance 2
    
    def test_avoid_seqs_filter_with_edit_dist(self):
        """Test avoid_seqs_filter with edit distance."""
        filter_func = avoid_seqs_filter(["ACGT"], avoid_min_edit_dist=2)
        
        assert filter_func("ACGT", None) is False  # Distance 0
        assert filter_func("ACGA", None) is False  # Distance 1
        assert filter_func("ACAA", None) is True   # Distance 2
        assert filter_func("ACG", None) is False   # Distance 1 (deletion)
    
    def test_avoid_seqs_filter_with_both_distances(self):
        """Test avoid_seqs_filter with both Hamming and edit distance."""
        filter_func = avoid_seqs_filter(
            ["ACGT"], 
            avoid_min_hamming_dist=2, 
            avoid_min_edit_dist=2
        )
        
        assert filter_func("ACGT", None) is False  # Both fail
        assert filter_func("ACAA", None) is True   # Both pass
    
    def test_avoid_seqs_filter_with_filter(self):
        """Test avoid_seqs_filter integration with filter."""
        pool = from_seqs(['ACGT', 'TGCA', 'AAAA', 'TTTT'], mode='sequential')
        
        filtered = filter(
            pool,
            num_candidate_seqs=4,
            avoid_seqs=['ACGT', 'TGCA'],
            mode='sequential'
        )
        
        # Only AAAA and TTTT should pass
        assert filtered.operation.num_states == 2
        result_df = filtered.generate_library(num_complete_iterations=1)
        seqs = set(result_df['seq'])
        assert seqs == {'AAAA', 'TTTT'}


class TestBuiltInFiltersValidation:
    """Validation tests for built-in filter parameters."""
    
    def setup_method(self):
        reset_op_id_counter()
    
    def test_gc_range_invalid_values(self):
        """Test gc_range validation."""
        pool = from_seqs(['ACGT'])
        
        # gc_range out of bounds
        with pytest.raises(ValueError, match="gc_range"):
            filter(pool, num_candidate_seqs=1, gc_range=(-0.1, 0.5))
        
        with pytest.raises(ValueError, match="gc_range"):
            filter(pool, num_candidate_seqs=1, gc_range=(0.5, 1.5))
        
        # min > max
        with pytest.raises(ValueError, match="gc_range"):
            filter(pool, num_candidate_seqs=1, gc_range=(0.6, 0.4))
    
    def test_combined_builtin_filters(self):
        """Test combining multiple built-in filters."""
        pool = get_kmers(length=6, alphabet='dna')
        
        filtered = filter(
            pool,
            num_candidate_seqs=1000,
            gc_range=(0.4, 0.6),
            max_homopolymer=2,
            min_edit_distance=2,
            min_num_filtered_seqs=1
        )
        
        # Generate and verify all constraints
        result_df = filtered.generate_library(num_seqs=10)
        for seq in result_df['seq']:
            # GC content in range
            gc = sum(1 for c in seq if c in 'GC') / len(seq)
            assert 0.4 <= gc <= 0.6
            
            # Max homopolymer <= 2
            max_run = 1
            current_run = 1
            for i in range(1, len(seq)):
                if seq[i] == seq[i-1]:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 1
            assert max_run <= 2
    
    def test_builtin_with_custom_filters(self):
        """Test combining built-in and custom filters."""
        pool = from_seqs(['ATGC', 'ATCG', 'GCGC', 'TTTT'], mode='sequential')
        
        # Custom filter: must contain 'AT'
        def has_at(seq, filtered_seqs=None):
            return 'AT' in seq
        
        filtered = filter(
            pool,
            num_candidate_seqs=4,
            filter_funcs=[has_at],
            gc_range=(0.4, 0.6),  # Also require 40-60% GC
            mode='sequential'
        )
        
        # ATGC: has 'AT', 50% GC -> passes
        # ATCG: has 'AT', 50% GC -> passes
        # GCGC: no 'AT' -> fails custom filter
        # TTTT: no 'AT', 0% GC -> fails both
        assert filtered.operation.num_states == 2

