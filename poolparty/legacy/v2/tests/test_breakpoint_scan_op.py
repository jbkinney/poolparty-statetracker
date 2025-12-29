"""Tests for BreakpointScanOp and breakpoint_scan factory function."""

import pytest
from math import comb
from poolparty import (
    Pool,
    MultiPool,
    Operation,
    from_seqs,
    breakpoint_scan,
    BreakpointScanOp,
    mutation_scan,
    concatenate,
)
from poolparty.utils import reset_op_id_counter


@pytest.fixture(autouse=True)
def reset_counter():
    """Reset operation ID counter before each test."""
    reset_op_id_counter()


class TestBreakpointScanBasic:
    """Basic tests for breakpoint_scan."""
    
    def test_single_breakpoint(self):
        """Single breakpoint should produce 2 segments."""
        multi = breakpoint_scan("ACGTACGT", num_breakpoints=1, mode='sequential')
        
        assert isinstance(multi, MultiPool)
        assert len(multi) == 2
        assert multi.operation.num_outputs == 2
    
    def test_two_breakpoints(self):
        """Two breakpoints should produce 3 segments."""
        multi = breakpoint_scan("ACGTACGTACGT", num_breakpoints=2, mode='sequential')
        
        assert len(multi) == 3
        assert multi.operation.num_outputs == 3
    
    def test_string_input(self):
        """Should accept string input directly."""
        multi = breakpoint_scan("ACGTACGT", num_breakpoints=1, mode='sequential')
        assert isinstance(multi, MultiPool)
    
    def test_pool_input(self):
        """Should accept Pool input."""
        parent = from_seqs(["ACGTACGT"])
        multi = breakpoint_scan(parent, num_breakpoints=1, mode='sequential')
        assert isinstance(multi, MultiPool)


class TestBreakpointScanPositions:
    """Tests for breakpoint position specification."""
    
    def test_explicit_positions(self):
        """Explicit positions should be used for breakpoints."""
        multi = breakpoint_scan(
            "ACGTACGT", 
            num_breakpoints=1, 
            positions=[4],  # Single valid position
            mode='sequential'
        )
        
        left, right = multi
        assert left.seq == "ACGT"
        assert right.seq == "ACGT"
    
    def test_explicit_positions_multiple(self):
        """Multiple explicit positions should work."""
        multi = breakpoint_scan(
            "ACGTACGT", 
            num_breakpoints=1, 
            positions=[2, 4, 6],
            mode='sequential'
        )
        
        # Should have C(3, 1) = 3 states
        assert multi.operation.num_states == 3
    
    def test_range_based_positions(self):
        """Range-based position specification should work."""
        multi = breakpoint_scan(
            "ACGTACGTACGT",  # Length 12
            num_breakpoints=1,
            start=2,
            end=10,
            step_size=2,
            mode='sequential'
        )
        
        # Valid positions: 2, 4, 6, 8 (all interior and in range)
        # C(4, 1) = 4 states
        assert multi.operation.num_states == 4
    
    def test_positions_filtered_to_interior(self):
        """Positions at 0 or seq_length should be filtered out."""
        multi = breakpoint_scan(
            "ACGT",  # Length 4
            num_breakpoints=1,
            positions=[0, 1, 2, 3, 4],  # 0 and 4 should be filtered
            mode='sequential'
        )
        
        # Only positions 1, 2, 3 are valid (interior)
        # C(3, 1) = 3 states
        assert multi.operation.num_states == 3
    
    def test_not_enough_positions_raises(self):
        """Should raise if not enough valid positions for breakpoints."""
        with pytest.raises(ValueError, match="Not enough valid positions"):
            breakpoint_scan(
                "ACGT",  # Only 3 interior positions: 1, 2, 3
                num_breakpoints=5,  # Need 5 positions
                mode='sequential'
            )
    
    def test_invalid_num_breakpoints_raises(self):
        """num_breakpoints < 1 should raise."""
        with pytest.raises(ValueError, match="num_breakpoints must be >= 1"):
            breakpoint_scan("ACGTACGT", num_breakpoints=0, mode='sequential')


class TestBreakpointScanStates:
    """Tests for state counting and sequential iteration."""
    
    def test_num_states_is_combination(self):
        """num_states should equal C(num_positions, num_breakpoints)."""
        seq = "ACGTACGTACGT"  # Length 12, interior positions: 1-11
        
        for k in [1, 2, 3]:
            multi = breakpoint_scan(seq, num_breakpoints=k, mode='sequential')
            
            n = 11  # Number of interior positions (1 through 11)
            expected = comb(n, k)
            assert multi.operation.num_states == expected
    
    def test_sequential_mode_iterates_all_combinations(self):
        """Sequential mode should iterate over all breakpoint combinations."""
        multi = breakpoint_scan(
            "ABCDEFGH",  # Length 8
            num_breakpoints=1,
            positions=[2, 4, 6],  # 3 positions
            mode='sequential'
        )
        
        left, right = multi
        
        # Generate all 3 states
        df_left = left.generate_library(num_seqs=3)
        df_right = right.generate_library(num_seqs=3, init_state=0)
        
        # Should have 3 unique left segments
        assert len(df_left['seq'].unique()) == 3
        
        # Left segments should be: AB, ABCD, ABCDEF
        left_seqs = sorted(df_left['seq'].tolist())
        assert left_seqs == sorted(["AB", "ABCD", "ABCDEF"])
    
    def test_sequential_mode_too_many_states_raises(self):
        """Sequential mode should raise if too many states."""
        # Create a sequence with many positions
        long_seq = "A" * 100  # 99 interior positions
        
        # C(99, 10) = very large number > max_sequential_states
        with pytest.raises(ValueError, match="cannot be sequential"):
            breakpoint_scan(long_seq, num_breakpoints=10, mode='sequential')


class TestBreakpointScanRandomMode:
    """Tests for random mode."""
    
    def test_random_mode_works(self):
        """Random mode should generate valid segments."""
        multi = breakpoint_scan("ACGTACGT", num_breakpoints=1, mode='random')
        
        left, right = multi
        
        # Should be able to generate sequences
        left_seq = left.seq
        right_seq = right.seq
        
        assert isinstance(left_seq, str)
        assert isinstance(right_seq, str)
        # Note: In random mode, left and right are computed independently
        # so their lengths may not add up to 8. This is expected behavior.
    
    def test_random_mode_varies(self):
        """Random mode should produce different results with different seeds."""
        multi = breakpoint_scan(
            "ACGTACGTACGT", 
            num_breakpoints=1,
            mode='random'
        )
        
        left, right = multi
        
        # Generate with different seeds
        seqs1 = left.generate_library(num_seqs=10, seed=42)['seq'].tolist()
        seqs2 = left.generate_library(num_seqs=10, seed=123)['seq'].tolist()
        
        # At least some should differ (probabilistically almost certain)
        assert seqs1 != seqs2


class TestBreakpointScanSegments:
    """Tests for segment correctness."""
    
    def test_segments_concatenate_to_original(self):
        """Concatenated segments should equal original sequence."""
        original = "ACGTACGTACGT"
        
        for n_breaks in [1, 2, 3]:
            multi = breakpoint_scan(
                original, 
                num_breakpoints=n_breaks,
                positions=[3, 6, 9],
                mode='sequential'
            )
            
            # Get first state's segments
            segments = [pool.seq for pool in multi]
            reconstructed = ''.join(segments)
            assert reconstructed == original
    
    def test_segment_boundaries(self):
        """Segments should be split at exactly the breakpoint positions."""
        multi = breakpoint_scan(
            "ABCDEFGH",
            num_breakpoints=2,
            positions=[2, 6],
            mode='sequential'
        )
        
        left, middle, right = multi
        
        # With breakpoints at 2 and 6:
        # left = [0:2] = "AB"
        # middle = [2:6] = "CDEF"
        # right = [6:8] = "GH"
        assert left.seq == "AB"
        assert middle.seq == "CDEF"
        assert right.seq == "GH"


class TestBreakpointScanDesignCards:
    """Tests for design card output."""
    
    def test_breakpoint_positions_in_results(self):
        """Results should include breakpoint_positions."""
        multi = breakpoint_scan(
            "ACGTACGT",
            num_breakpoints=1,
            mode='sequential'
        )
        
        # The source operation should have breakpoint_positions in results
        left, right = multi
        
        # Generate and check the source op has the data
        left.generate_library(num_seqs=1)
        
        # Check the source operation's results
        source_results = multi.operation._results_df
        assert 'breakpoint_positions' in source_results.columns
    
    def test_seq_columns_in_results(self):
        """Results should include seq_0, seq_1, etc."""
        multi = breakpoint_scan(
            "ACGTACGT",
            num_breakpoints=2,
            mode='sequential'
        )
        
        left, middle, right = multi
        left.generate_library(num_seqs=1)
        
        source_results = multi.operation._results_df
        assert 'seq_0' in source_results.columns
        assert 'seq_1' in source_results.columns
        assert 'seq_2' in source_results.columns


class TestBreakpointScanIntegration:
    """Integration tests with other operations."""
    
    def test_segment_in_concatenation(self):
        """Segments should work in concatenation with other pools."""
        multi = breakpoint_scan(
            "ACGTACGT",
            num_breakpoints=1,
            positions=[4],
            mode='sequential'
        )
        
        left, right = multi
        
        # Concatenate segments with a spacer
        combined = concatenate([left, from_seqs(["NN"]), right])
        
        # Should work
        seq = combined.seq
        assert seq == "ACGTNNACGT"
    
    def test_reconstruct_with_modified_segment(self):
        """Should be able to reconstruct with one modified segment."""
        original = "ACGTACGT"
        multi = breakpoint_scan(
            original,
            num_breakpoints=1,
            positions=[4],
            mode='sequential'
        )
        
        left, right = multi
        
        # Create a modified version with reversed right segment
        reversed_right = from_seqs([right.seq[::-1]])
        reconstructed = concatenate([left, reversed_right])
        
        expected = "ACGT" + "TGCA"  # Left unchanged, right reversed
        assert reconstructed.seq == expected
    
    def test_multiple_breakpoints_with_operations(self):
        """Multiple segments should work with various operations."""
        multi = breakpoint_scan(
            "AAAABBBBCCCC",
            num_breakpoints=2,
            positions=[4, 8],
            mode='sequential'
        )
        
        left, middle, right = multi
        
        assert left.seq == "AAAA"
        assert middle.seq == "BBBB"
        assert right.seq == "CCCC"
        
        # Substitute middle with a different sequence
        new_middle = from_seqs(["XXXX"])
        
        # Reconstruct
        reconstructed = left + new_middle + right
        seq = reconstructed.seq
        
        # Should have same length
        assert len(seq) == 12
        # First and last parts unchanged, middle replaced
        assert seq[:4] == "AAAA"
        assert seq[4:8] == "XXXX"
        assert seq[8:] == "CCCC"

