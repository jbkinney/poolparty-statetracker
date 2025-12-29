"""Tests for the BreakpointScan operation."""

import pytest
import numpy as np
import poolparty as pp
from poolparty import reset_op_id_counter
from poolparty.operations.breakpoint_scan import BreakpointScanOp, breakpoint_scan


@pytest.fixture(autouse=True)
def reset_ids():
    """Reset operation ID counter before each test."""
    reset_op_id_counter()
    yield
    reset_op_id_counter()


class TestBreakpointScanFactory:
    """Test breakpoint_scan factory function."""
    
    def test_returns_tuple_of_pools(self):
        """Test that breakpoint_scan returns a tuple of Pools."""
        with pp.Party() as party:
            result = breakpoint_scan('ACGT', num_breakpoints=1)
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert all(hasattr(p, 'operation') for p in result)
    
    def test_creates_breakpoint_scan_op(self):
        """Test that breakpoint_scan creates a BreakpointScanOp."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ACGT', num_breakpoints=1)
            assert isinstance(left.operation, BreakpointScanOp)
            assert left.operation is right.operation  # Same operation
    
    def test_accepts_string_input(self):
        """Test that breakpoint_scan accepts string input."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ACGT', num_breakpoints=1)
            party.output(left, name='left')
            party.output(right, name='right')
        
        df = party.generate(num_seqs=1)
        assert df['left'].iloc[0] + df['right'].iloc[0] == 'ACGT'
    
    def test_accepts_pool_input(self):
        """Test that breakpoint_scan accepts Pool input."""
        with pp.Party() as party:
            seq = pp.from_seqs(['ACGT'])
            left, right = breakpoint_scan(seq, num_breakpoints=1)
            party.output(left, name='left')
            party.output(right, name='right')
        
        df = party.generate(num_seqs=1)
        assert df['left'].iloc[0] + df['right'].iloc[0] == 'ACGT'


class TestBreakpointScanSingleBreakpoint:
    """Test single breakpoint behavior."""
    
    def test_single_breakpoint_count(self):
        """Test correct number of breakpoint positions."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ACGT', num_breakpoints=1, mode='sequential')
            party.output(left, name='left')
            party.output(right, name='right')
        
        df = party.generate(num_complete_iterations=1)
        # 3 possible breakpoint positions (after 1, 2, 3)
        assert len(df) == 3
    
    def test_single_breakpoint_splits(self):
        """Test that splits are correct."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ACGT', num_breakpoints=1, mode='sequential')
            party.output(left, name='left')
            party.output(right, name='right')
        
        df = party.generate(num_complete_iterations=1)
        
        # All splits should reconstruct original
        for _, row in df.iterrows():
            assert row['left'] + row['right'] == 'ACGT'
    
    def test_single_breakpoint_positions(self):
        """Test specific breakpoint positions."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ABCD', num_breakpoints=1, mode='sequential')
            party.output(left, name='left')
            party.output(right, name='right')
        
        df = party.generate(num_complete_iterations=1)
        
        splits = set(zip(df['left'], df['right']))
        expected = {('A', 'BCD'), ('AB', 'CD'), ('ABC', 'D')}
        assert splits == expected


class TestBreakpointScanMultipleBreakpoints:
    """Test multiple breakpoints behavior."""
    
    def test_double_breakpoint_returns_three_pools(self):
        """Test that 2 breakpoints returns 3 pools."""
        with pp.Party() as party:
            result = breakpoint_scan('ACGTACGT', num_breakpoints=2)
            assert len(result) == 3
    
    def test_double_breakpoint_splits(self):
        """Test that double breakpoint splits are correct."""
        with pp.Party() as party:
            left, mid, right = breakpoint_scan('ACGTACGT', num_breakpoints=2, 
                                                mode='sequential')
            party.output(left, name='left')
            party.output(mid, name='mid')
            party.output(right, name='right')
        
        df = party.generate(num_seqs=10)
        
        for _, row in df.iterrows():
            assert row['left'] + row['mid'] + row['right'] == 'ACGTACGT'
    
    def test_triple_breakpoint(self):
        """Test triple breakpoint."""
        with pp.Party() as party:
            pools = breakpoint_scan('ABCDEFGH', num_breakpoints=3, mode='sequential')
            assert len(pools) == 4
            
            for i, p in enumerate(pools):
                party.output(p, name=f'seg{i}')
        
        df = party.generate(num_seqs=10)
        
        for _, row in df.iterrows():
            reconstructed = row['seg0'] + row['seg1'] + row['seg2'] + row['seg3']
            assert reconstructed == 'ABCDEFGH'


class TestBreakpointScanSequentialMode:
    """Test BreakpointScan in sequential mode."""
    
    def test_sequential_enumeration(self):
        """Test sequential enumeration of breakpoints."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ABCDE', num_breakpoints=1, mode='sequential')
            party.output(left, name='left')
        
        df = party.generate(num_complete_iterations=1)
        # 4 possible positions
        assert len(df) == 4
    
    def test_sequential_num_states(self):
        """Test num_states calculation."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ACGT', num_breakpoints=1, mode='sequential')
            # C(3, 1) = 3 (3 positions between 4 chars)
            assert left.operation.num_states == 3
    
    def test_sequential_num_states_double(self):
        """Test num_states for double breakpoint."""
        with pp.Party() as party:
            pools = breakpoint_scan('ABCDEF', num_breakpoints=2, mode='sequential')
            # 5 positions, choose 2: C(5,2) = 10
            assert pools[0].operation.num_states == 10


class TestBreakpointScanRandomMode:
    """Test BreakpointScan in random mode."""
    
    def test_random_sampling(self):
        """Test random sampling of breakpoints."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ACGTACGT', num_breakpoints=1, mode='random')
            party.output(left, name='left')
            party.output(right, name='right')
        
        df = party.generate(num_seqs=100, seed=42)
        
        # All should be valid splits
        for _, row in df.iterrows():
            assert row['left'] + row['right'] == 'ACGTACGT'
    
    def test_random_variability(self):
        """Test that random mode produces varied outputs."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ACGTACGT', num_breakpoints=1, mode='random')
            party.output(left, name='left')
        
        df = party.generate(num_seqs=100, seed=42)
        unique_lefts = df['left'].nunique()
        assert unique_lefts > 1
    
    def test_random_requires_rng(self):
        """Test that random mode requires RNG."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ACGT', num_breakpoints=1, mode='random')
        
        with pytest.raises(RuntimeError, match="Random mode requires RNG"):
            left.operation.compute(['ACGT'], 0, None)


class TestBreakpointScanPositions:
    """Test custom positions parameter."""
    
    def test_custom_positions(self):
        """Test breakpoint with custom positions."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ABCDE', num_breakpoints=1, 
                                           positions=[2, 4], mode='sequential')
            party.output(left, name='left')
            party.output(right, name='right')
        
        df = party.generate(num_complete_iterations=1)
        assert len(df) == 2  # Only 2 positions
        
        splits = set(zip(df['left'], df['right']))
        expected = {('AB', 'CDE'), ('ABCD', 'E')}
        assert splits == expected
    
    def test_positions_filters_invalid(self):
        """Test that positions outside range are filtered."""
        with pp.Party() as party:
            # Position 0 and 5 should be filtered (not valid for length 5)
            left, right = breakpoint_scan('ABCDE', num_breakpoints=1,
                                           positions=[0, 2, 5], mode='sequential')
            party.output(left, name='left')
        
        df = party.generate(num_complete_iterations=1)
        # Only position 2 is valid
        assert len(df) == 1


class TestBreakpointScanStartEndStep:
    """Test start/end/step_size parameters."""
    
    def test_start_parameter(self):
        """Test start parameter."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ABCDEF', num_breakpoints=1,
                                           start=3, mode='sequential')
            party.output(left, name='left')
        
        df = party.generate(num_complete_iterations=1)
        # Positions 3, 4, 5 (3 positions)
        assert len(df) == 3
        
        # All left segments should have at least 3 chars
        for left in df['left']:
            assert len(left) >= 3
    
    def test_end_parameter(self):
        """Test end parameter."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ABCDEF', num_breakpoints=1,
                                           end=2, mode='sequential')
            party.output(left, name='left')
        
        df = party.generate(num_complete_iterations=1)
        # Positions 1, 2 (2 positions)
        assert len(df) == 2
        
        # All left segments should have at most 2 chars
        for left in df['left']:
            assert len(left) <= 2
    
    def test_step_size_parameter(self):
        """Test step_size parameter."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ABCDEFGH', num_breakpoints=1,
                                           step_size=2, mode='sequential')
            party.output(left, name='left')
        
        df = party.generate(num_complete_iterations=1)
        # Positions 1, 3, 5, 7 (4 positions with step 2)
        assert len(df) == 4
    
    def test_combined_start_end_step(self):
        """Test combining start, end, and step_size."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ABCDEFGHIJ', num_breakpoints=1,
                                           start=2, end=8, step_size=2, 
                                           mode='sequential')
            party.output(left, name='left')
        
        df = party.generate(num_complete_iterations=1)
        # Positions 2, 4, 6, 8 (4 positions)
        assert len(df) == 4


class TestBreakpointScanDesignCards:
    """Test BreakpointScan design card output."""
    
    def test_breakpoints_in_output(self):
        """Test breakpoints are in output."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ACGT', num_breakpoints=1, mode='sequential')
            party.output(left, name='left')
        
        df = party.generate(num_seqs=3)
        assert 'breakpoint_scan.breakpoints' in df.columns
    
    def test_breakpoints_values(self):
        """Test breakpoint values are correct."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ABCD', num_breakpoints=1, mode='sequential')
            party.output(left, name='left')
            party.output(right, name='right')
        
        df = party.generate(num_complete_iterations=1)
        
        for _, row in df.iterrows():
            breakpoints = row['breakpoint_scan.breakpoints']
            # Breakpoint value should equal length of left segment
            assert breakpoints[0] == len(row['left'])


class TestBreakpointScanErrors:
    """Test BreakpointScan error handling."""
    
    def test_zero_breakpoints_error(self):
        """Test error for num_breakpoints=0."""
        with pytest.raises(ValueError, match="num_breakpoints must be >= 1"):
            breakpoint_scan('ACGT', num_breakpoints=0)
    
    def test_negative_breakpoints_error(self):
        """Test error for negative num_breakpoints."""
        with pytest.raises(ValueError, match="num_breakpoints must be >= 1"):
            breakpoint_scan('ACGT', num_breakpoints=-1)
    
    def test_not_enough_positions_error(self):
        """Test error when not enough valid positions."""
        with pytest.raises(ValueError, match="Not enough valid positions"):
            # Only 3 positions available, but need 5 breakpoints
            breakpoint_scan('ACGT', num_breakpoints=5)


class TestBreakpointScanMultiOutput:
    """Test BreakpointScan multi-output behavior."""
    
    def test_output_indices(self):
        """Test that output pools have correct indices."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ACGT', num_breakpoints=1)
            assert left.output_index == 0
            assert right.output_index == 1
    
    def test_num_outputs(self):
        """Test num_outputs attribute."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ACGT', num_breakpoints=1)
            assert left.operation.num_outputs == 2
    
    def test_three_outputs(self):
        """Test three outputs with 2 breakpoints."""
        with pp.Party() as party:
            seg0, seg1, seg2 = breakpoint_scan('ACGTACGT', num_breakpoints=2)
            assert seg0.output_index == 0
            assert seg1.output_index == 1
            assert seg2.output_index == 2
            assert seg0.operation.num_outputs == 3


class TestBreakpointScanCompute:
    """Test BreakpointScan compute method directly."""
    
    def test_compute_sequential(self):
        """Test compute in sequential mode."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ACGT', num_breakpoints=1, mode='sequential')
        
        result = left.operation.compute(['ACGT'], 0, None)
        assert 'seq_0' in result
        assert 'seq_1' in result
        assert result['seq_0'] + result['seq_1'] == 'ACGT'
    
    def test_compute_random(self):
        """Test compute in random mode."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ACGT', num_breakpoints=1, mode='random')
        
        rng = np.random.default_rng(42)
        result = left.operation.compute(['ACGT'], 0, rng)
        assert result['seq_0'] + result['seq_1'] == 'ACGT'


class TestBreakpointScanWithOtherOperations:
    """Test BreakpointScan combined with other operations."""
    
    def test_with_mutation_scan(self):
        """Test breakpoint scan followed by mutation."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ACGT', num_breakpoints=1, mode='sequential')
            mutated_right = pp.mutation_scan(right, k=1, mode='sequential')
            combined = left + mutated_right
            party.output(combined, name='seq')
        
        df = party.generate(num_seqs=10)
        assert len(df) == 10
    
    def test_with_concatenation(self):
        """Test breakpoint segments with concatenation."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ACGT', num_breakpoints=1, mode='sequential')
            combined = left + '---' + right
            party.output(combined, name='seq')
        
        df = party.generate(num_seqs=3)
        for seq in df['seq']:
            # Remove separator and check reconstruction
            parts = seq.split('---')
            assert parts[0] + parts[1] == 'ACGT'


class TestBreakpointScanCustomName:
    """Test BreakpointScan name parameter."""
    
    def test_default_name(self):
        """Test default operation name."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ACGT', num_breakpoints=1)
            assert left.operation.name == 'breakpoint_scan'
    
    def test_custom_name(self):
        """Test custom operation name."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ACGT', num_breakpoints=1, name='my_split')
            assert left.operation.name == 'my_split'
    
    def test_custom_name_in_design_card(self):
        """Test custom name appears in design card columns."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ACGT', num_breakpoints=1, name='split')
            party.output(left, name='left')
        
        df = party.generate(num_seqs=1)
        assert 'split.breakpoints' in df.columns


class TestBreakpointScanSpacing:
    """Test min_spacing and max_spacing constraints."""
    
    def test_min_spacing_filters_close_breakpoints(self):
        """Test that min_spacing filters out breakpoints that are too close."""
        # Sequence: ABCDEFGHIJ (10 chars), positions 1-9
        # With 2 breakpoints and min_spacing=3, only combinations where
        # the two breakpoints are at least 3 apart are valid
        with pp.Party() as party:
            seg0, seg1, seg2 = breakpoint_scan(
                'ABCDEFGHIJ', num_breakpoints=2, 
                min_spacing=3, mode='sequential'
            )
            party.output(seg0, name='seg0')
            party.output(seg1, name='seg1')
            party.output(seg2, name='seg2')
        
        df = party.generate(num_complete_iterations=1)
        
        # Verify all generated combinations have spacing >= 3
        for _, row in df.iterrows():
            breakpoints = row['breakpoint_scan.breakpoints']
            assert len(breakpoints) == 2
            spacing = breakpoints[1] - breakpoints[0]
            assert spacing >= 3
    
    def test_max_spacing_filters_far_breakpoints(self):
        """Test that max_spacing filters out breakpoints that are too far apart."""
        # With 2 breakpoints and max_spacing=2, only combinations where
        # the two breakpoints are at most 2 apart are valid
        with pp.Party() as party:
            seg0, seg1, seg2 = breakpoint_scan(
                'ABCDEFGHIJ', num_breakpoints=2,
                max_spacing=2, mode='sequential'
            )
            party.output(seg0, name='seg0')
            party.output(seg1, name='seg1')
            party.output(seg2, name='seg2')
        
        df = party.generate(num_complete_iterations=1)
        
        # Verify all generated combinations have spacing <= 2
        for _, row in df.iterrows():
            breakpoints = row['breakpoint_scan.breakpoints']
            assert len(breakpoints) == 2
            spacing = breakpoints[1] - breakpoints[0]
            assert spacing <= 2
    
    def test_combined_min_max_spacing(self):
        """Test combined min_spacing and max_spacing constraints."""
        # With min_spacing=2 and max_spacing=4, breakpoints must be 2-4 apart
        with pp.Party() as party:
            seg0, seg1, seg2 = breakpoint_scan(
                'ABCDEFGHIJ', num_breakpoints=2,
                min_spacing=2, max_spacing=4, mode='sequential'
            )
            party.output(seg0, name='seg0')
            party.output(seg1, name='seg1')
            party.output(seg2, name='seg2')
        
        df = party.generate(num_complete_iterations=1)
        
        # Verify all generated combinations have 2 <= spacing <= 4
        for _, row in df.iterrows():
            breakpoints = row['breakpoint_scan.breakpoints']
            spacing = breakpoints[1] - breakpoints[0]
            assert 2 <= spacing <= 4
    
    def test_num_states_reflects_filtered_count(self):
        """Test that num_states is the filtered count, not the unfiltered count."""
        # Positions 1-9, choose 2: C(9,2) = 36 unfiltered combinations
        # With min_spacing=5, only pairs with spacing >= 5 are valid
        # Valid pairs: (1,6), (1,7), (1,8), (1,9), (2,7), (2,8), (2,9),
        #              (3,8), (3,9), (4,9) = 10 combinations
        with pp.Party() as party:
            pools = breakpoint_scan(
                'ABCDEFGHIJ', num_breakpoints=2,
                min_spacing=5, mode='sequential'
            )
            # Count should be 10, not 36
            assert pools[0].operation.num_states == 10
    
    def test_spacing_error_no_valid_combinations(self):
        """Test error when no valid combinations after filtering."""
        # Short sequence with impossible spacing constraints
        with pytest.raises(ValueError, match="No valid breakpoint combinations"):
            breakpoint_scan('ABCD', num_breakpoints=2, min_spacing=5)
    
    def test_single_breakpoint_ignores_spacing(self):
        """Test that spacing constraints have no effect with single breakpoint."""
        # Single breakpoint has no spacing to check
        with pp.Party() as party:
            left, right = breakpoint_scan(
                'ABCDEF', num_breakpoints=1,
                min_spacing=10, max_spacing=1, mode='sequential'
            )
            party.output(left, name='left')
        
        df = party.generate(num_complete_iterations=1)
        # Should still have all 5 possible positions
        assert len(df) == 5
    
    def test_spacing_with_random_mode(self):
        """Test that spacing constraints work in random mode."""
        with pp.Party() as party:
            seg0, seg1, seg2 = breakpoint_scan(
                'ABCDEFGHIJ', num_breakpoints=2,
                min_spacing=3, mode='random'
            )
            party.output(seg0, name='seg0')
            party.output(seg1, name='seg1')
            party.output(seg2, name='seg2')
        
        df = party.generate(num_seqs=50, seed=42)
        
        # All random samples should satisfy spacing constraints
        for _, row in df.iterrows():
            breakpoints = row['breakpoint_scan.breakpoints']
            spacing = breakpoints[1] - breakpoints[0]
            assert spacing >= 3
    
    def test_triple_breakpoint_spacing(self):
        """Test spacing constraints with 3 breakpoints."""
        # With 3 breakpoints, we have 2 spacing values to check
        with pp.Party() as party:
            pools = breakpoint_scan(
                'ABCDEFGHIJKLMNOP', num_breakpoints=3,
                min_spacing=2, max_spacing=4, mode='sequential'
            )
            for i, p in enumerate(pools):
                party.output(p, name=f'seg{i}')
        
        df = party.generate(num_complete_iterations=1)
        
        # Verify all spacings are within constraints
        for _, row in df.iterrows():
            breakpoints = row['breakpoint_scan.breakpoints']
            assert len(breakpoints) == 3
            for i in range(len(breakpoints) - 1):
                spacing = breakpoints[i + 1] - breakpoints[i]
                assert 2 <= spacing <= 4

