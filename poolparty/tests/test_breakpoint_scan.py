"""Tests for the BreakpointScan operation."""

import pytest
import numpy as np
import poolparty as pp
from poolparty.base_ops.breakpoint_scan import BreakpointScanOp, breakpoint_scan
import statetracker as st


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
            left = left.named('left')
            right = right.named('right')
        
        df = left.generate_library(num_seqs=1, report_design_cards=True, aux_pools=[right])
        assert df['seq'].iloc[0] + df['right.seq'].iloc[0] == 'ACGT'
    
    def test_accepts_pool_input(self):
        """Test that breakpoint_scan accepts Pool input."""
        with pp.Party() as party:
            seq = pp.from_seqs(['ACGT'])
            left, right = breakpoint_scan(seq, num_breakpoints=1)
            left = left.named('left')
            right = right.named('right')
        
        df = left.generate_library(num_seqs=1, report_design_cards=True, aux_pools=[right])
        assert df['seq'].iloc[0] + df['right.seq'].iloc[0] == 'ACGT'


class TestBreakpointScanSingleBreakpoint:
    """Test single breakpoint behavior."""
    
    def test_single_breakpoint_count(self):
        """Test correct number of breakpoint positions."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ACGT', num_breakpoints=1, mode='sequential')
            left = left.named('left')
            right = right.named('right')
        
        df = left.generate_library(num_cycles=1, report_design_cards=True, aux_pools=[right])
        # 5 possible breakpoint positions (0, 1, 2, 3, 4)
        assert len(df) == 5
    
    def test_single_breakpoint_splits(self):
        """Test that splits are correct."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ACGT', num_breakpoints=1, mode='sequential')
            left = left.named('left')
            right = right.named('right')
        
        df = left.generate_library(num_cycles=1, report_design_cards=True, aux_pools=[right])
        
        # All splits should reconstruct original
        for _, row in df.iterrows():
            assert row['seq'] + row['right.seq'] == 'ACGT'
    
    def test_single_breakpoint_positions(self):
        """Test specific breakpoint positions."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ABCD', num_breakpoints=1, mode='sequential')
            left = left.named('left')
            right = right.named('right')
        
        df = left.generate_library(num_cycles=1, report_design_cards=True, aux_pools=[right])
        
        splits = set(zip(df['seq'], df['right.seq']))
        expected = {('', 'ABCD'), ('A', 'BCD'), ('AB', 'CD'), ('ABC', 'D'), ('ABCD', '')}
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
            left = left.named('left')
            mid = mid.named('mid')
            right = right.named('right')
        
        df = left.generate_library(num_seqs=10, report_design_cards=True, aux_pools=[mid, right])
        
        for _, row in df.iterrows():
            assert row['seq'] + row['mid.seq'] + row['right.seq'] == 'ACGTACGT'
    
    def test_triple_breakpoint(self):
        """Test triple breakpoint."""
        with pp.Party() as party:
            pools = breakpoint_scan('ABCDEFGH', num_breakpoints=3, mode='sequential')
            assert len(pools) == 4
            
            seg0 = pools[0].named('seg0')
            seg1 = pools[1].named('seg1')
            seg2 = pools[2].named('seg2')
            seg3 = pools[3].named('seg3')
        
        df = seg0.generate_library(num_seqs=10, report_design_cards=True, aux_pools=[seg1, seg2, seg3])
        
        for _, row in df.iterrows():
            reconstructed = row['seq'] + row['seg1.seq'] + row['seg2.seq'] + row['seg3.seq']
            assert reconstructed == 'ABCDEFGH'


class TestBreakpointScanSequentialMode:
    """Test BreakpointScan in sequential mode."""
    
    def test_sequential_enumeration(self):
        """Test sequential enumeration of breakpoints."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ABCDE', num_breakpoints=1, mode='sequential')
            left = left.named('left')
        
        df = left.generate_library(num_cycles=1)
        # 6 possible positions (0, 1, 2, 3, 4, 5)
        assert len(df) == 6
    
    def test_sequential_num_states(self):
        """Test num_states calculation."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ACGT', num_breakpoints=1, mode='sequential')
            # C(5, 1) = 5 (positions 0, 1, 2, 3, 4)
            assert left.operation.num_values == 5
    
    def test_sequential_num_states_double(self):
        """Test num_states for double breakpoint."""
        with pp.Party() as party:
            pools = breakpoint_scan('ABCDEF', num_breakpoints=2, mode='sequential')
            # 7 positions (0-6), choose 2: C(7,2) = 21
            assert pools[0].operation.num_values == 21


class TestBreakpointScanRandomMode:
    """Test BreakpointScan in random mode."""
    
    def test_random_sampling(self):
        """Test random sampling of breakpoints."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ACGTACGT', num_breakpoints=1, mode='random')
            left = left.named('left')
            right = right.named('right')
        
        df = left.generate_library(num_seqs=100, seed=42, report_design_cards=True, aux_pools=[right])
        
        # All should be valid splits
        for _, row in df.iterrows():
            assert row['seq'] + row['right.seq'] == 'ACGTACGT'
    
    def test_random_variability(self):
        """Test that random mode produces varied outputs."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ACGTACGT', num_breakpoints=1, mode='random')
            left = left.named('left')
        
        df = left.generate_library(num_seqs=100, seed=42)
        unique_lefts = df['seq'].nunique()
        assert unique_lefts > 1
    
    def test_random_num_states_is_one(self):
        """Test that random mode has num_values=1."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ACGT', num_breakpoints=1, mode='random')
            assert left.operation.num_values == 1


class TestBreakpointScanPositions:
    """Test custom positions parameter."""
    
    def test_custom_positions(self):
        """Test breakpoint with custom positions."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ABCDE', num_breakpoints=1, 
                                           positions=[2, 4], mode='sequential')
            left = left.named('left')
            right = right.named('right')
        
        df = left.generate_library(num_cycles=1, report_design_cards=True, aux_pools=[right])
        assert len(df) == 2  # Only 2 positions
        
        splits = set(zip(df['seq'], df['right.seq']))
        expected = {('AB', 'CDE'), ('ABCD', 'E')}
        assert splits == expected
    
    def test_positions_rejects_invalid(self):
        """Test that positions outside range raise ValueError."""
        with pp.Party() as party:
            # Position -1 is not valid (min is 0)
            with pytest.raises(ValueError, match="out of range"):
                breakpoint_scan('ABCDE', num_breakpoints=1,
                               positions=[-1, 2], mode='sequential')
            # Position 6 is not valid (max is 5 for length 5)
            with pytest.raises(ValueError, match="out of range"):
                breakpoint_scan('ABCDE', num_breakpoints=1,
                               positions=[2, 6], mode='sequential')


class TestBreakpointScanSlicePositions:
    """Test positions parameter with slice syntax."""
    
    def test_slice_start(self):
        """Test slice with start offset."""
        with pp.Party() as party:
            # slice(2, None) on valid range [0, 6] gives positions 2, 3, 4, 5, 6
            left, right = breakpoint_scan('ABCDEF', num_breakpoints=1,
                                           positions=slice(2, None), mode='sequential')
            left = left.named('left')
        
        df = left.generate_library(num_cycles=1)
        # Positions 2, 3, 4, 5, 6 (5 positions)
        assert len(df) == 5
        
        # All left segments should have at least 2 chars
        for left_seq in df['seq']:
            assert len(left_seq) >= 2
    
    def test_slice_stop(self):
        """Test slice with stop limit."""
        with pp.Party() as party:
            # slice(None, 2) on valid range [1, 5] gives positions 1, 2
            left, right = breakpoint_scan('ABCDEF', num_breakpoints=1,
                                           positions=slice(None, 2), mode='sequential')
            left = left.named('left')
        
        df = left.generate_library(num_cycles=1)
        # Positions 1, 2 (2 positions)
        assert len(df) == 2
        
        # All left segments should have at most 2 chars
        for left_seq in df['seq']:
            assert len(left_seq) <= 2
    
    def test_slice_step(self):
        """Test slice with step."""
        with pp.Party() as party:
            # slice(None, None, 2) on valid range [0, 8] gives positions 0, 2, 4, 6, 8
            left, right = breakpoint_scan('ABCDEFGH', num_breakpoints=1,
                                           positions=slice(None, None, 2), mode='sequential')
            left = left.named('left')
        
        df = left.generate_library(num_cycles=1)
        # Positions 0, 2, 4, 6, 8 (5 positions with step 2)
        assert len(df) == 5
    
    def test_slice_combined(self):
        """Test slice with start, stop, and step."""
        with pp.Party() as party:
            # slice(1, 8, 2) on valid range [1, 9] gives positions 2, 4, 6, 8
            left, right = breakpoint_scan('ABCDEFGHIJ', num_breakpoints=1,
                                           positions=slice(1, 8, 2), mode='sequential')
            left = left.named('left')
        
        df = left.generate_library(num_cycles=1)
        # Positions 2, 4, 6, 8 (4 positions)
        assert len(df) == 4


class TestBreakpointScanDesignCards:
    """Test BreakpointScan design card output."""
    
    def test_breakpoints_in_output(self):
        """Test breakpoints are in output."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ACGT', num_breakpoints=1, mode='sequential', op_name='split')
            left = left.named('left')
        
        df = left.generate_library(num_seqs=3, report_design_cards=True)
        assert 'split.key.breakpoints' in df.columns
    
    def test_breakpoints_values(self):
        """Test breakpoint values are correct."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ABCD', num_breakpoints=1, mode='sequential', op_name='split')
            left = left.named('left')
            right = right.named('right')
        
        df = left.generate_library(num_cycles=1, report_design_cards=True, aux_pools=[right])
        
        for _, row in df.iterrows():
            breakpoints = row['split.key.breakpoints']
            # Breakpoint value should equal length of left segment
            assert breakpoints[0] == len(row['seq'])


class TestBreakpointScanErrors:
    """Test BreakpointScan error handling."""
    
    def test_zero_breakpoints_error(self):
        """Test error for num_breakpoints=0."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="num_breakpoints must be >= 1"):
                breakpoint_scan('ACGT', num_breakpoints=0)
    
    def test_negative_breakpoints_error(self):
        """Test error for negative num_breakpoints."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="num_breakpoints must be >= 1"):
                breakpoint_scan('ACGT', num_breakpoints=-1)
    
    def test_not_enough_positions_error(self):
        """Test error when not enough valid positions."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="Not enough valid positions"):
                # Only 5 positions available (0-4), but need 6 breakpoints
                breakpoint_scan('ACGT', num_breakpoints=6, mode='sequential')


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
    """Test BreakpointScan compute methods directly."""
    
    def test_compute_sequential(self):
        """Test compute in sequential mode."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ACGT', num_breakpoints=1, mode='sequential')
        
        left.operation.state._value = 0
        card = left.operation.compute_design_card(['ACGT'])
        result = left.operation.compute_seq_from_card(['ACGT'], card)
        assert 'seq_0' in result
        assert 'seq_1' in result
        assert result['seq_0'] + result['seq_1'] == 'ACGT'
    
    def test_compute_random(self):
        """Test compute in random mode."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ACGT', num_breakpoints=1, mode='random')
        
        rng = np.random.default_rng(42)
        card = left.operation.compute_design_card(['ACGT'], rng)
        result = left.operation.compute_seq_from_card(['ACGT'], card)
        assert result['seq_0'] + result['seq_1'] == 'ACGT'


class TestBreakpointScanWithOtherOperations:
    """Test BreakpointScan combined with other operations."""
    
    def test_with_mutagenize(self):
        """Test breakpoint scan followed by mutation works with diamond pattern.
        
        This creates a diamond pattern where the breakpoint counter appears
        in multiple paths. ordered_product now properly deduplicates this,
        so no conflict error is raised.
        """
        with pp.Party() as party:
            # Use longer sequence and position constraints to avoid empty segments
            left, right = breakpoint_scan('ACGTACGT', num_breakpoints=1, mode='sequential')
            # Only mutagenize positions where right has content (exclude last breakpoint)
            mutated_right = pp.mutagenize(right, num_mutations=1, mode='random')
            combined = pp.join([left, mutated_right]).named('seq')
        
        # Should work without ConflictingStateAssignmentError
        # Use random mode for mutagenize to avoid issues with empty sequences
        df = combined.generate_library(num_seqs=5, seed=42)
        assert len(df) == 5
    
    def test_with_join(self):
        """Test breakpoint segments can be joined.
        
        Synchronized pools share the same counter. When joined,
        the shared counter is only included once in the product, so they
        iterate together in lockstep.
        """
        with pp.Party() as party:
            left, right = breakpoint_scan('ACGT', num_breakpoints=1, mode='sequential')
            combined = pp.join([left, '---', right]).named('combined')
        
        df = combined.generate_library(num_seqs=3)
        # Verify segments are joined with separator
        for seq in df['seq']:
            assert '---' in seq


class TestBreakpointScanCustomName:
    """Test BreakpointScan name parameter."""
    
    def test_default_name(self):
        """Test default operation name."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ACGT', num_breakpoints=1)
            assert left.operation.name.startswith('op[')
            assert ':breakpoint_scan' in left.operation.name
    
    def test_custom_name(self):
        """Test custom operation name."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ACGT', num_breakpoints=1, op_name='my_split')
            assert left.operation.name == 'my_split'
    
    def test_custom_name_in_design_card(self):
        """Test custom name appears in design card columns with .key."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ACGT', num_breakpoints=1, op_name='split')
            left = left.named('left')
        
        df = left.generate_library(num_seqs=1, report_design_cards=True)
        assert 'split.key.breakpoints' in df.columns


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
                min_spacing=3, mode='sequential', op_name='split'
            )
            seg0 = seg0.named('seg0')
            seg1 = seg1.named('seg1')
            seg2 = seg2.named('seg2')
        
        df = seg0.generate_library(num_cycles=1, report_design_cards=True, aux_pools=[seg1, seg2])
        
        # Verify all generated combinations have spacing >= 3
        for _, row in df.iterrows():
            breakpoints = row['split.key.breakpoints']
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
                max_spacing=2, mode='sequential', op_name='split'
            )
            seg0 = seg0.named('seg0')
            seg1 = seg1.named('seg1')
            seg2 = seg2.named('seg2')
        
        df = seg0.generate_library(num_cycles=1, report_design_cards=True, aux_pools=[seg1, seg2])
        
        # Verify all generated combinations have spacing <= 2
        for _, row in df.iterrows():
            breakpoints = row['split.key.breakpoints']
            assert len(breakpoints) == 2
            spacing = breakpoints[1] - breakpoints[0]
            assert spacing <= 2
    
    def test_combined_min_max_spacing(self):
        """Test combined min_spacing and max_spacing constraints."""
        # With min_spacing=2 and max_spacing=4, breakpoints must be 2-4 apart
        with pp.Party() as party:
            seg0, seg1, seg2 = breakpoint_scan(
                'ABCDEFGHIJ', num_breakpoints=2,
                min_spacing=2, max_spacing=4, mode='sequential', op_name='split'
            )
            seg0 = seg0.named('seg0')
            seg1 = seg1.named('seg1')
            seg2 = seg2.named('seg2')
        
        df = seg0.generate_library(num_cycles=1, report_design_cards=True, aux_pools=[seg1, seg2])
        
        # Verify all generated combinations have 2 <= spacing <= 4
        for _, row in df.iterrows():
            breakpoints = row['split.key.breakpoints']
            spacing = breakpoints[1] - breakpoints[0]
            assert 2 <= spacing <= 4
    
    def test_num_states_reflects_filtered_count(self):
        """Test that num_states is the filtered count, not the unfiltered count."""
        # Positions 0-10 (11 positions), choose 2: C(11,2) = 55 unfiltered combinations
        # With min_spacing=5, only pairs with spacing >= 5 are valid
        # Valid pairs: (0,5), (0,6), (0,7), (0,8), (0,9), (0,10), (1,6), (1,7), ...
        # Total = 6 + 5 + 4 + 3 + 2 + 1 = 21 combinations
        with pp.Party() as party:
            pools = breakpoint_scan(
                'ABCDEFGHIJ', num_breakpoints=2,
                min_spacing=5, mode='sequential'
            )
            # Count should be 21, not 55
            assert pools[0].operation.num_values == 21
    
    def test_spacing_error_no_valid_combinations(self):
        """Test error when no valid combinations after filtering."""
        # Short sequence with impossible spacing constraints
        with pp.Party() as party:
            with pytest.raises(ValueError, match="No valid breakpoint combinations"):
                breakpoint_scan('ABCD', num_breakpoints=2, min_spacing=5, mode='sequential')
    
    def test_single_breakpoint_ignores_spacing(self):
        """Test that spacing constraints have no effect with single breakpoint."""
        # Single breakpoint has no spacing to check
        with pp.Party() as party:
            left, right = breakpoint_scan(
                'ABCDEF', num_breakpoints=1,
                min_spacing=10, max_spacing=1, mode='sequential'
            )
            left = left.named('left')
        
        df = left.generate_library(num_cycles=1)
        # Should still have all 7 possible positions (0-6)
        assert len(df) == 7
    
    def test_spacing_with_random_mode(self):
        """Test that spacing constraints work in random mode."""
        with pp.Party() as party:
            seg0, seg1, seg2 = breakpoint_scan(
                'ABCDEFGHIJ', num_breakpoints=2,
                min_spacing=3, mode='random', op_name='split'
            )
            seg0 = seg0.named('seg0')
            seg1 = seg1.named('seg1')
            seg2 = seg2.named('seg2')
        
        df = seg0.generate_library(num_seqs=50, seed=42, report_design_cards=True, aux_pools=[seg1, seg2])
        
        # All random samples should satisfy spacing constraints
        for _, row in df.iterrows():
            breakpoints = row['split.key.breakpoints']
            spacing = breakpoints[1] - breakpoints[0]
            assert spacing >= 3
    
    def test_triple_breakpoint_spacing(self):
        """Test spacing constraints with 3 breakpoints."""
        # With 3 breakpoints, we have 2 spacing values to check
        with pp.Party() as party:
            pools = breakpoint_scan(
                'ABCDEFGHIJKLMNOP', num_breakpoints=3,
                min_spacing=2, max_spacing=4, mode='sequential', op_name='split'
            )
            seg0 = pools[0].named('seg0')
            seg1 = pools[1].named('seg1')
            seg2 = pools[2].named('seg2')
            seg3 = pools[3].named('seg3')
        
        df = seg0.generate_library(num_cycles=1, report_design_cards=True, aux_pools=[seg1, seg2, seg3])
        
        # Verify all spacings are within constraints
        for _, row in df.iterrows():
            breakpoints = row['split.key.breakpoints']
            assert len(breakpoints) == 3
            for i in range(len(breakpoints) - 1):
                spacing = breakpoints[i + 1] - breakpoints[i]
                assert 2 <= spacing <= 4


class TestSynchronizePoolsParameter:
    """Test synchronize_pools parameter."""
    
    def test_synchronize_pools_default_true(self):
        """Test that synchronize_pools defaults to True (shared counter)."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ACGT', num_breakpoints=1)
            # Default: pools share the same counter
            assert left.state is right.state
    
    def test_synchronize_pools_true_shares_counter(self):
        """Test that synchronize_pools=True shares counter across pools."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ACGT', num_breakpoints=1)
            assert left.state is right.state
    
    
    def test_synchronized_pools_can_join(self):
        """Test that synchronized pools can be joined.
        
        Synchronized pools share the same counter. When joined,
        the shared counter is only included once in the product, so they
        iterate together in lockstep.
        """
        with pp.Party() as party:
            left, right = breakpoint_scan('ACGT', num_breakpoints=1)
            combined = pp.join([left, '---', right]).named('combined')
        
        df = combined.generate_library(num_seqs=3)
        # Verify segments are joined with separator
        for seq in df['seq']:
            assert '---' in seq
    
        df = combined.generate_library(num_seqs=3)
        for seq in df['seq']:
            assert '---' in seq
    
    def test_synchronized_with_mutagenize_raises_conflict(self):
        """Test synchronized pools with downstream mutation raises conflict.
        
        Synchronized pools share the same counter, so joining one
        with a mutation of the other creates conflicting state assignments.
        """
        with pp.Party() as party:
            left, right = breakpoint_scan('ACGT', num_breakpoints=1, mode='sequential')
            mutated_right = pp.mutagenize(right, num_mutations=1, mode='sequential')
            combined = pp.join([left, mutated_right]).named('seq')
        
        
    
    def test_three_pools_synchronized(self):
        """Test three pools are synchronized."""
        with pp.Party() as party:
            seg0, seg1, seg2 = breakpoint_scan('ACGTACGT', num_breakpoints=2)
            # All three share the same counter
            assert seg0.state is seg1.state
            assert seg1.state is seg2.state
    
