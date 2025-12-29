"""Tests for subseq_scan operation."""

import pytest
from poolparty import subseq_scan, from_seqs, Pool


class TestSubseqScanBasic:
    """Basic tests for subseq_scan factory function."""
    
    def test_basic_creation_from_string(self):
        """Test basic subseq_scan creation from string."""
        pool = subseq_scan('ACGTACGT', subseq_length=4)
        assert isinstance(pool, Pool)
        assert pool.seq_length == 4
    
    def test_basic_creation_from_pool(self):
        """Test subseq_scan creation from another Pool."""
        parent = from_seqs(['ACGTACGT'])
        pool = subseq_scan(parent, subseq_length=4)
        assert isinstance(pool, Pool)
        assert pool.seq_length == 4
    
    def test_seq_length_matches_subseq_length(self):
        """Test that output seq_length matches subseq_length parameter."""
        for length in [2, 4, 6]:
            pool = subseq_scan('ACGTACGTACGT', subseq_length=length)
            assert pool.seq_length == length


class TestSubseqScanRangeBased:
    """Tests for range-based interface (start, end, step_size)."""
    
    def test_default_range_all_positions(self):
        """Test default range covers all valid positions."""
        # Sequence length 8, subseq_length 4 → positions 0,1,2,3,4 (5 states)
        pool = subseq_scan('ACGTACGT', subseq_length=4)
        assert pool.operation.num_states == 5
    
    def test_num_states_formula(self):
        """Test num_states = (end - start - subseq_length + step_size) // step_size."""
        # Length 12, subseq_length 4, default range: positions 0-8 (9 states)
        pool = subseq_scan('ACGTACGTACGT', subseq_length=4)
        assert pool.operation.num_states == 9
    
    def test_custom_start(self):
        """Test custom start position."""
        pool = subseq_scan('ACGTACGTACGT', subseq_length=4, start=2)
        # Positions 2,3,4,5,6,7,8 = 7 states
        assert pool.operation.num_states == 7
    
    def test_custom_end(self):
        """Test custom end position."""
        pool = subseq_scan('ACGTACGTACGT', subseq_length=4, end=8)
        # Positions 0,1,2,3,4 (end=8 means last window is [4,8))
        assert pool.operation.num_states == 5
    
    def test_custom_step_size(self):
        """Test custom step_size."""
        pool = subseq_scan('ACGTACGTACGT', subseq_length=4, step_size=2)
        # Positions 0,2,4,6,8 = 5 states (with step=2)
        assert pool.operation.num_states == 5
    
    def test_step_size_larger_than_subseq_length(self):
        """Test step_size larger than subseq_length (non-overlapping windows)."""
        pool = subseq_scan('ACGTACGTACGT', subseq_length=3, step_size=4)
        # Positions 0,4,8 = 3 states (windows [0,3), [4,7), [8,11))
        assert pool.operation.num_states == 3
    
    def test_sequential_mode_range(self):
        """Test sequential mode iterates through all positions."""
        pool = subseq_scan('ACGTACGT', subseq_length=4, mode='sequential')
        result_df = pool.generate_library(num_complete_iterations=1)
        
        # Should have 5 subsequences at positions 0,1,2,3,4
        assert len(result_df) == 5
        expected_seqs = ['ACGT', 'CGTA', 'GTAC', 'TACG', 'ACGT']
        assert list(result_df['seq']) == expected_seqs
    
    def test_sequential_with_step_size(self):
        """Test sequential mode with step_size."""
        pool = subseq_scan('ACGTACGTACGT', subseq_length=4, step_size=4, mode='sequential')
        result_df = pool.generate_library(num_complete_iterations=1)
        
        # Positions 0,4,8 = 3 states
        assert len(result_df) == 3
        expected_seqs = ['ACGT', 'ACGT', 'ACGT']
        assert list(result_df['seq']) == expected_seqs


class TestSubseqScanPositionBased:
    """Tests for position-based interface (positions, position_probs)."""
    
    def test_explicit_positions(self):
        """Test explicit positions parameter."""
        pool = subseq_scan('ACGTACGTACGT', subseq_length=4, positions=[0, 4, 8])
        assert pool.operation.num_states == 3
    
    def test_positions_sequential_mode(self):
        """Test sequential mode with explicit positions."""
        pool = subseq_scan('ACGTACGTACGT', subseq_length=4, positions=[0, 4, 8], mode='sequential')
        result_df = pool.generate_library(num_complete_iterations=1)
        
        assert len(result_df) == 3
        expected_seqs = ['ACGT', 'ACGT', 'ACGT']
        assert list(result_df['seq']) == expected_seqs
    
    def test_positions_preserves_order(self):
        """Test that positions are iterated in given order."""
        pool = subseq_scan('0123456789AB', subseq_length=2, positions=[4, 0, 8], mode='sequential')
        result_df = pool.generate_library(num_complete_iterations=1)
        
        assert len(result_df) == 3
        # Positions [4, 0, 8] → subseqs at those positions
        expected_seqs = ['45', '01', '89']
        assert list(result_df['seq']) == expected_seqs
    
    def test_position_probs_weighted_sampling(self):
        """Test that position_probs affects sampling distribution."""
        # With heavily biased weights, most samples should come from position 0
        pool = subseq_scan('ABCDEFGH', subseq_length=2, 
                          positions=[0, 2, 4], position_probs=[0.99, 0.005, 0.005])
        
        result_df = pool.generate_library(num_seqs=100, seed=42)
        seq_counts = result_df['seq'].value_counts()
        
        # 'AB' (position 0) should appear most often
        assert seq_counts.get('AB', 0) > seq_counts.get('CD', 0)
        assert seq_counts.get('AB', 0) > seq_counts.get('EF', 0)
    
    def test_position_probs_normalized(self):
        """Test that position_probs are normalized automatically."""
        # Weights that don't sum to 1 should still work
        pool = subseq_scan('ACGTACGT', subseq_length=2, 
                          positions=[0, 2, 4], position_probs=[2.0, 2.0, 2.0])
        
        # Should work without error
        result_df = pool.generate_library(num_seqs=10, seed=42)
        assert len(result_df) == 10


class TestSubseqScanMutuallyExclusive:
    """Tests for mutually exclusive interface validation."""
    
    def test_range_and_positions_raises(self):
        """Test that providing both range and position params raises error."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            subseq_scan('ACGTACGT', subseq_length=4, start=0, positions=[0, 2])
    
    def test_end_and_positions_raises(self):
        """Test that providing end with positions raises error."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            subseq_scan('ACGTACGT', subseq_length=4, end=6, positions=[0, 2])
    
    def test_step_size_and_positions_raises(self):
        """Test that providing step_size with positions raises error."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            subseq_scan('ACGTACGT', subseq_length=4, step_size=2, positions=[0, 2])
    
    def test_position_probs_without_positions_raises(self):
        """Test that position_probs without positions raises error."""
        with pytest.raises(ValueError, match="position_probs requires positions"):
            subseq_scan('ACGTACGT', subseq_length=4, position_probs=[0.5, 0.5])
    
    def test_position_probs_with_sequential_mode_raises(self):
        """Test that position_probs with sequential mode raises error."""
        with pytest.raises(ValueError, match="Cannot specify position_probs with mode='sequential'"):
            subseq_scan('ACGTACGT', subseq_length=2, 
                       positions=[0, 2], position_probs=[0.7, 0.3], mode='sequential')


class TestSubseqScanValidation:
    """Tests for input validation."""
    
    def test_subseq_length_zero_raises(self):
        """Test that subseq_length=0 raises error."""
        with pytest.raises(ValueError, match="must be > 0"):
            subseq_scan('ACGTACGT', subseq_length=0)
    
    def test_subseq_length_negative_raises(self):
        """Test that negative subseq_length raises error."""
        with pytest.raises(ValueError, match="must be > 0"):
            subseq_scan('ACGTACGT', subseq_length=-1)
    
    def test_subseq_length_exceeds_parent_raises(self):
        """Test that subseq_length > parent length raises error."""
        with pytest.raises(ValueError, match="cannot be greater than"):
            subseq_scan('ACGT', subseq_length=10)
    
    def test_empty_positions_raises(self):
        """Test that empty positions list raises error."""
        with pytest.raises(ValueError, match="non-empty"):
            subseq_scan('ACGTACGT', subseq_length=2, positions=[])
    
    def test_position_out_of_bounds_raises(self):
        """Test that all positions out-of-bounds raises error."""
        with pytest.raises(ValueError, match="No valid positions"):
            # All positions are out of bounds for length 4 on a length-4 sequence
            subseq_scan('ACGT', subseq_length=4, positions=[2, 3])
    
    def test_negative_position_skipped(self):
        """Test that negative positions are filtered out (valid positions remain)."""
        # -1 is invalid but 0 and 2 are valid, so this should work
        pool = subseq_scan('ACGTACGT', subseq_length=2, positions=[-1, 0, 2])
        # Only positions 0 and 2 should be valid
        assert pool.operation.num_states == 2
    
    def test_duplicate_positions_raises(self):
        """Test that duplicate positions raise error."""
        with pytest.raises(ValueError, match="duplicates"):
            subseq_scan('ACGTACGT', subseq_length=2, positions=[0, 2, 0])
    
    def test_negative_start_raises(self):
        """Test that negative start raises error."""
        with pytest.raises(ValueError, match="start must be >= 0"):
            subseq_scan('ACGTACGT', subseq_length=2, start=-1)
    
    def test_zero_step_size_raises(self):
        """Test that step_size=0 raises error."""
        with pytest.raises(ValueError, match="step_size must be > 0"):
            subseq_scan('ACGTACGT', subseq_length=2, step_size=0)
    
    def test_negative_step_size_raises(self):
        """Test that negative step_size raises error."""
        with pytest.raises(ValueError, match="step_size must be > 0"):
            subseq_scan('ACGTACGT', subseq_length=2, step_size=-1)
    
    def test_position_probs_wrong_length_raises(self):
        """Test that position_probs with wrong length raises error."""
        with pytest.raises(ValueError, match="must match"):
            subseq_scan('ACGTACGT', subseq_length=2, 
                       positions=[0, 2, 4], position_probs=[0.5, 0.5])
    
    def test_position_probs_negative_raises(self):
        """Test that negative position_probs raise error."""
        with pytest.raises(ValueError, match="non-negative"):
            subseq_scan('ACGTACGT', subseq_length=2, 
                       positions=[0, 2], position_probs=[0.5, -0.5])
    
    def test_position_probs_all_zero_raises(self):
        """Test that all-zero position_probs raise error."""
        with pytest.raises(ValueError, match="must sum to > 0"):
            subseq_scan('ACGTACGT', subseq_length=2, 
                       positions=[0, 2], position_probs=[0.0, 0.0])
    
    def test_range_produces_no_positions_raises(self):
        """Test that range producing no valid positions raises error."""
        with pytest.raises(ValueError, match="no valid positions"):
            # Start at 10 in a length-8 sequence produces no positions
            subseq_scan('ACGTACGT', subseq_length=2, start=10)


class TestSubseqScanModes:
    """Tests for random and sequential modes."""
    
    def test_random_mode_is_default(self):
        """Test that random mode is the default."""
        pool = subseq_scan('ACGTACGT', subseq_length=4)
        assert pool.operation.mode == 'random'
    
    def test_deterministic_with_seed(self):
        """Test that same seed produces same sequences."""
        pool = subseq_scan('ACGTACGTACGT', subseq_length=4)
        
        result_df1 = pool.generate_library(num_seqs=10, seed=42)
        result_df2 = pool.generate_library(num_seqs=10, seed=42)
        
        assert list(result_df1['seq']) == list(result_df2['seq'])
    
    def test_different_seeds_different_sequences(self):
        """Test that different seeds produce different sequences."""
        pool = subseq_scan('ACGTACGTACGT', subseq_length=4)
        
        result_df1 = pool.generate_library(num_seqs=20, seed=42)
        result_df2 = pool.generate_library(num_seqs=20, seed=123)
        
        assert list(result_df1['seq']) != list(result_df2['seq'])
    
    def test_sequential_mode_state_progression(self):
        """Test that sequential mode progresses through states correctly."""
        pool = subseq_scan('0123456789', subseq_length=3, mode='sequential')
        
        # Generate one at a time and check state progression
        seqs = []
        for i in range(pool.operation.num_states):
            result_df = pool.generate_library(num_seqs=1, init_state=i, advance=False)
            seqs.append(result_df['seq'].iloc[0])
        
        expected = ['012', '123', '234', '345', '456', '567', '678', '789']
        assert seqs == expected


class TestSubseqScanDesignCards:
    """Tests for design card functionality."""
    
    def test_design_card_keys(self):
        """Test that design card keys are correct."""
        pool = subseq_scan('ACGTACGT', subseq_length=4)
        assert set(pool.operation.design_card_keys) == {'seq', 'position', 'subseq_length'}
    
    def test_design_card_output(self):
        """Test design card data in output DataFrame."""
        pool = subseq_scan('ACGTACGTACGT', subseq_length=4, mode='sequential')
        result_df = pool.generate_library(num_complete_iterations=1)
        
        # Check that a column with 'position' exists and has correct values
        position_cols = [c for c in result_df.columns if 'position' in c]
        assert len(position_cols) == 1
        positions = list(result_df[position_cols[0]])
        assert positions == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    
    def test_design_card_subseq_length(self):
        """Test that subseq_length is recorded in design cards."""
        pool = subseq_scan('ACGTACGT', subseq_length=3, mode='sequential')
        result_df = pool.generate_library(num_seqs=1)
        
        subseq_length_cols = [c for c in result_df.columns if 'subseq_length' in c]
        assert len(subseq_length_cols) == 1
        assert result_df[subseq_length_cols[0]].iloc[0] == 3
    
    def test_custom_design_card_keys(self):
        """Test filtering design card keys."""
        pool = subseq_scan('ACGTACGT', subseq_length=4, 
                          mode='sequential', design_card_keys=['position'])
        result_df = pool.generate_library(num_seqs=1)
        
        # Should have position but not subseq_length (beyond seq which is always included)
        position_cols = [c for c in result_df.columns if 'position' in c]
        subseq_length_cols = [c for c in result_df.columns if 'subseq_length' in c]
        assert len(position_cols) == 1
        assert len(subseq_length_cols) == 0


class TestSubseqScanAncestors:
    """Tests for ancestor/parent pool tracking."""
    
    def test_has_parent_pool_from_string(self):
        """Test that subseq_scan from string has parent pool (from_seqs wrapper)."""
        pool = subseq_scan('ACGTACGT', subseq_length=4)
        parents = pool.operation.parent_pools
        assert len(parents) == 1
    
    def test_has_parent_pool_from_pool(self):
        """Test that subseq_scan from Pool has the correct parent."""
        parent = from_seqs(['ACGTACGT'])
        pool = subseq_scan(parent, subseq_length=4)
        
        parents = pool.operation.parent_pools
        assert len(parents) == 1
        assert parents[0] is parent


class TestSubseqScanWithParentVariation:
    """Tests for subseq_scan with varying parent sequences."""
    
    def test_extract_from_different_parent_sequences(self):
        """Test that subseq_scan works with varying parent."""
        parent = from_seqs(['AAAA', 'TTTT'], mode='sequential')
        pool = subseq_scan(parent, subseq_length=2, mode='sequential')
        
        # Generate all combinations
        result_df = pool.generate_library(num_complete_iterations=1)
        seqs = list(result_df['seq'])
        
        # Parent has 2 states (AAAA, TTTT), subseq_scan has 3 states (pos 0,1,2)
        # Total: 2 × 3 = 6 sequences
        assert len(result_df) == 6
        
        # Should have AA from AAAA and TT from TTTT
        assert 'AA' in seqs
        assert 'TT' in seqs


class TestSubseqScanMultiLength:
    """Tests for multi-length subseq_length functionality."""
    
    def test_multi_length_num_states(self):
        """Test num_states is sum of positions for each length."""
        # Sequence 'ACGTACGT' (length 8)
        # Length 3: positions 0-5 = 6 positions
        # Length 5: positions 0-3 = 4 positions
        # Total: 10 states
        pool = subseq_scan('ACGTACGT', subseq_length=[3, 5], mode='sequential')
        assert pool.operation.num_states == 10
    
    def test_multi_length_seq_length_is_none(self):
        """Test that seq_length is None for multi-length."""
        pool = subseq_scan('ACGTACGT', subseq_length=[3, 5])
        assert pool.seq_length is None
    
    def test_multi_length_single_length_seq_length(self):
        """Test that seq_length is set for single-length list."""
        pool = subseq_scan('ACGTACGT', subseq_length=[4])
        assert pool.seq_length == 4
    
    def test_multi_length_sequential_order(self):
        """Test sequential mode iterates all pos for len[0], then all pos for len[1]."""
        # Sequence 'ACGTACGT' (length 8)
        # Length 2: positions 0-6 = 7 positions
        # Length 4: positions 0-4 = 5 positions
        pool = subseq_scan('ACGTACGT', subseq_length=[2, 4], mode='sequential')
        result_df = pool.generate_library(num_complete_iterations=1)
        
        assert len(result_df) == 12  # 7 + 5
        
        # First 7 should be length 2
        len_col = [c for c in result_df.columns if 'subseq_length' in c][0]
        lengths = list(result_df[len_col])
        assert lengths[:7] == [2] * 7
        assert lengths[7:] == [4] * 5
        
        # Check position progression within each length
        pos_col = [c for c in result_df.columns if 'position' in c][0]
        positions = list(result_df[pos_col])
        assert positions[:7] == [0, 1, 2, 3, 4, 5, 6]  # All positions for length 2
        assert positions[7:] == [0, 1, 2, 3, 4]  # All positions for length 4
    
    def test_multi_length_sequential_sequences(self):
        """Test actual subsequences are correct in sequential mode."""
        pool = subseq_scan('01234567', subseq_length=[2, 3], mode='sequential')
        result_df = pool.generate_library(num_complete_iterations=1)
        
        seqs = list(result_df['seq'])
        # Length 2: positions 0-6
        expected_len2 = ['01', '12', '23', '34', '45', '56', '67']
        # Length 3: positions 0-5
        expected_len3 = ['012', '123', '234', '345', '456', '567']
        
        assert seqs == expected_len2 + expected_len3
    
    def test_subseq_length_probs_weighted_sampling(self):
        """Test subseq_length_probs affects length distribution."""
        # Heavily bias toward length 2
        pool = subseq_scan('ACGTACGT', subseq_length=[2, 4], 
                          subseq_length_probs=[0.95, 0.05], mode='random')
        
        result_df = pool.generate_library(num_seqs=100, seed=42)
        
        len_col = [c for c in result_df.columns if 'subseq_length' in c][0]
        length_counts = result_df[len_col].value_counts()
        
        # Length 2 should appear much more often
        assert length_counts.get(2, 0) > length_counts.get(4, 0)
    
    def test_subseq_length_probs_normalized(self):
        """Test subseq_length_probs are normalized automatically."""
        pool = subseq_scan('ACGTACGT', subseq_length=[3, 5], 
                          subseq_length_probs=[2.0, 2.0], mode='random')
        
        # Should work without error
        result_df = pool.generate_library(num_seqs=10, seed=42)
        assert len(result_df) == 10
    
    def test_subseq_length_probs_sequential_raises(self):
        """Test subseq_length_probs with sequential mode raises error."""
        with pytest.raises(ValueError, match="Cannot specify subseq_length_probs with mode='sequential'"):
            subseq_scan('ACGTACGT', subseq_length=[3, 5], 
                       subseq_length_probs=[0.7, 0.3], mode='sequential')
    
    def test_subseq_length_probs_wrong_length_raises(self):
        """Test subseq_length_probs with wrong length raises error."""
        with pytest.raises(ValueError, match="must match"):
            subseq_scan('ACGTACGT', subseq_length=[3, 5], 
                       subseq_length_probs=[0.5, 0.3, 0.2])
    
    def test_subseq_length_probs_negative_raises(self):
        """Test negative subseq_length_probs raise error."""
        with pytest.raises(ValueError, match="non-negative"):
            subseq_scan('ACGTACGT', subseq_length=[3, 5], 
                       subseq_length_probs=[0.7, -0.3])
    
    def test_multi_length_position_based(self):
        """Test multi-length with explicit positions."""
        # Position 0 is valid for both lengths, position 5 only for length 2
        pool = subseq_scan('ACGTACGT', subseq_length=[2, 5], 
                          positions=[0, 5], mode='sequential')
        
        # Length 2: positions 0, 5 both valid = 2 states
        # Length 5: position 0 valid, 5 invalid (5+5=10 > 8) = 1 state
        # Total: 3 states
        assert pool.operation.num_states == 3
    
    def test_multi_length_position_filtering(self):
        """Test that invalid positions are filtered per length."""
        # Sequence length 8
        # Position 6 is valid for length 2 (6+2=8), invalid for length 4 (6+4=10)
        pool = subseq_scan('ACGTACGT', subseq_length=[2, 4], 
                          positions=[0, 6], mode='sequential')
        
        # Length 2: positions 0, 6 both valid = 2 states
        # Length 4: position 0 valid, 6 invalid = 1 state
        # Total: 3 states
        assert pool.operation.num_states == 3
        
        result_df = pool.generate_library(num_complete_iterations=1)
        seqs = list(result_df['seq'])
        
        # Should have: AC(pos0,len2), GT(pos6,len2), ACGT(pos0,len4)
        assert 'AC' in seqs
        assert 'GT' in seqs
        assert 'ACGT' in seqs
    
    def test_multi_length_all_positions_invalid_raises(self):
        """Test error when all positions invalid for a length."""
        with pytest.raises(ValueError, match="No valid positions"):
            # Position 6 is invalid for length 4 on length-8 sequence
            # This should fail since no valid positions for length 4
            subseq_scan('ACGTACGT', subseq_length=[4], positions=[6])
    
    def test_empty_subseq_length_list_raises(self):
        """Test that empty subseq_length list raises error."""
        with pytest.raises(ValueError, match="must not be empty"):
            subseq_scan('ACGTACGT', subseq_length=[])

