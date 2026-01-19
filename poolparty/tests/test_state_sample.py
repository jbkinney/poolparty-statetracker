"""Tests for state_sample operation - sample states from a pool."""

import pytest
import poolparty as pp
from poolparty.state_ops.state_sample import StateSampleOp, state_sample


class TestStateSampleFactory:
    """Test state_sample factory function."""
    
    def test_returns_pool(self):
        """Test that state_sample returns a Pool."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C', 'D', 'E'], mode='sequential')
            sampled = state_sample(pool, num_values=3, seed=42)
            assert sampled is not None
            assert hasattr(sampled, 'operation')
    
    def test_creates_state_sample_op(self):
        """Test that state_sample creates a StateSampleOp."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C', 'D', 'E'], mode='sequential')
            sampled = state_sample(pool, num_values=3, seed=42)
            assert isinstance(sampled.operation, StateSampleOp)


class TestStateSampleNumStates:
    """Test state sampling num_states."""
    
    def test_num_states_less_than_parent(self):
        """Test sampling fewer states than parent."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C', 'D', 'E'], mode='sequential')  # 5 states
            sampled = state_sample(pool, num_values=3, seed=42)
            assert sampled.num_states == 3
    
    def test_num_states_equal_to_parent(self):
        """Test sampling same number of states as parent."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C', 'D', 'E'], mode='sequential')  # 5 states
            sampled = state_sample(pool, num_values=5, seed=42, with_replacement=False)
            assert sampled.num_states == 5
    
    def test_num_states_greater_with_replacement(self):
        """Test sampling more states than parent with replacement."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C'], mode='sequential')  # 3 states
            sampled = state_sample(pool, num_values=10, seed=42, with_replacement=True)
            assert sampled.num_states == 10


class TestStateSampleOutput:
    """Test state sampling output."""
    
    def test_output_with_sampled_states(self):
        """Test that output matches sampled states."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C', 'D', 'E'], mode='sequential')
            sampled = state_sample(pool, sampled_states=[0, 2, 4]).named('samp')
        
        df = sampled.generate_library(num_cycles=1)
        output_seqs = df['seq'].tolist()
        assert output_seqs == ['A', 'C', 'E']
    
    def test_output_with_duplicates_in_sampled_states(self):
        """Test that sampled_states with duplicates works."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C'], mode='sequential')
            sampled = state_sample(pool, sampled_states=[0, 0, 1, 2, 2]).named('samp')
        
        df = sampled.generate_library(num_cycles=1)
        output_seqs = df['seq'].tolist()
        assert output_seqs == ['A', 'A', 'B', 'C', 'C']
    
    def test_all_outputs_from_parent(self):
        """Test that all sampled outputs come from parent sequences."""
        with pp.Party() as party:
            seqs = ['A', 'B', 'C', 'D', 'E']
            pool = pp.from_seqs(seqs, mode='sequential')
            sampled = state_sample(pool, num_values=3, seed=42).named('samp')
        
        df = sampled.generate_library(num_cycles=1)
        for seq in df['seq'].tolist():
            assert seq in seqs


class TestStateSampleDeterminism:
    """Test deterministic behavior with seeds."""
    
    def test_same_seed_same_result(self):
        """Test that same seed produces same result."""
        with pp.Party() as party:
            pool1 = pp.from_seqs(['A', 'B', 'C', 'D', 'E'], mode='sequential')
            sampled1 = state_sample(pool1, num_values=3, seed=42).named('samp1')
        df1 = sampled1.generate_library(num_cycles=1)
        
        with pp.Party() as party:
            pool2 = pp.from_seqs(['A', 'B', 'C', 'D', 'E'], mode='sequential')
            sampled2 = state_sample(pool2, num_values=3, seed=42).named('samp2')
        df2 = sampled2.generate_library(num_cycles=1)
        
        assert df1['seq'].tolist() == df2['seq'].tolist()
    
    def test_different_seed_different_result(self):
        """Test that different seeds produce different results."""
        with pp.Party() as party:
            pool1 = pp.from_seqs(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], mode='sequential')
            sampled1 = state_sample(pool1, num_values=4, seed=42).named('samp1')
        df1 = sampled1.generate_library(num_cycles=1)
        
        with pp.Party() as party:
            pool2 = pp.from_seqs(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], mode='sequential')
            sampled2 = state_sample(pool2, num_values=4, seed=123).named('samp2')
        df2 = sampled2.generate_library(num_cycles=1)
        
        # Very likely to be different with different seeds
        assert df1['seq'].tolist() != df2['seq'].tolist()


class TestStateSampleNoSeed:
    """Test behavior without explicit seed."""
    
    def test_no_seed_runs(self):
        """Test that state_sample works without explicit seed."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C', 'D', 'E'], mode='sequential')
            sampled = state_sample(pool, num_values=3).named('samp')  # No seed
        
        df = sampled.generate_library(num_cycles=1)
        # Should still output 3 valid sequences
        assert len(df) == 3
        for seq in df['seq'].tolist():
            assert seq in ['A', 'B', 'C', 'D', 'E']


class TestStateSampleCustomName:
    """Test StateSampleOp name parameter."""
    
    def test_default_name(self):
        """Test default operation name."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C'], mode='sequential')
            sampled = state_sample(pool, num_values=2, seed=42)
            assert sampled.operation.name.startswith('op[')
            assert ':state_sample' in sampled.operation.name
    
    def test_custom_name(self):
        """Test custom operation name."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C'], mode='sequential')
            sampled = state_sample(pool, num_values=2, seed=42, op_name='my_sample')
            assert sampled.operation.name == 'my_sample'


class TestStateSampleCompute:
    """Test StateSampleOp compute methods directly."""
    
    def test_compute_design_card_empty(self):
        """Test compute_design_card returns empty dict."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ACGT'], mode='sequential')
            sampled = state_sample(pool, sampled_states=[0])
        
        card = sampled.operation.compute_design_card(['ACGT'])
        assert card == {}
    
    def test_compute_seq_from_card(self):
        """Test compute_seq_from_card returns parent sequence."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ACGT'], mode='sequential')
            sampled = state_sample(pool, sampled_states=[0])
        
        card = sampled.operation.compute_design_card(['ACGT'])
        result = sampled.operation.compute_seq_from_card(['ACGT'], card)
        assert result == {'seq_0': 'ACGT'}


class TestStateSampleWithReplacement:
    """Test state_sample with/without replacement."""
    
    def test_with_replacement_allows_more_than_parent(self):
        """Test that with_replacement=True allows sampling more states."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C'], mode='sequential')
            sampled = state_sample(pool, num_values=10, seed=42, with_replacement=True).named('samp')
        
        assert sampled.num_states == 10
        df = sampled.generate_library(num_cycles=1)
        assert len(df) == 10
    
    def test_without_replacement_valid(self):
        """Test that with_replacement=False works when num_states <= parent."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C', 'D', 'E'], mode='sequential')
            sampled = state_sample(pool, num_values=3, seed=42, with_replacement=False).named('samp')
        
        assert sampled.num_states == 3
        df = sampled.generate_library(num_cycles=1)
        # All sequences should be unique
        assert len(df['seq'].unique()) == 3
    
    def test_without_replacement_exceeds_raises(self):
        """Test that with_replacement=False raises when num_states > parent."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C'], mode='sequential')
            with pytest.raises(ValueError, match="exceeds parent.num_values"):
                state_sample(pool, num_values=10, with_replacement=False)


class TestStateSampleValidation:
    """Test state_sample validation."""
    
    def test_must_specify_num_states_or_sampled_states(self):
        """Test that either num_states or sampled_states must be specified."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C'], mode='sequential')
            with pytest.raises(ValueError, match="Must specify either"):
                state_sample(pool)
    
    def test_cannot_specify_both_num_states_and_sampled_states(self):
        """Test that num_states and sampled_states are mutually exclusive."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C'], mode='sequential')
            with pytest.raises(ValueError, match="Cannot specify both"):
                state_sample(pool, num_values=2, sampled_states=[0, 1])
    
    def test_cannot_specify_seed_with_sampled_states(self):
        """Test that seed cannot be used with sampled_states."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C'], mode='sequential')
            with pytest.raises(ValueError, match="Cannot specify 'seed' with 'sampled_states'"):
                state_sample(pool, sampled_states=[0, 1], seed=42)
    
    def test_sampled_states_out_of_range_raises(self):
        """Test that sampled_states with out-of-range values raises."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C'], mode='sequential')
            with pytest.raises(ValueError, match="out of range"):
                state_sample(pool, sampled_states=[0, 1, 10])
    
    def test_sampled_states_negative_raises(self):
        """Test that sampled_states with negative values raises."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C'], mode='sequential')
            with pytest.raises(ValueError, match="out of range"):
                state_sample(pool, sampled_states=[0, -1, 2])


class TestStateSampleGetCopyParams:
    """Test StateSampleOp._get_copy_params method."""
    
    def test_get_copy_params_with_num_states(self):
        """Test _get_copy_params with num_values."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C'], mode='sequential')
            sampled = state_sample(pool, num_values=2, seed=42, with_replacement=False)
            params = sampled.operation._get_copy_params()
            
            assert params['parent_pool'] is pool
            assert params['num_values'] == 2
            assert params['sampled_states'] is None
            assert params['seed'] == 42
            assert params['with_replacement'] == False
    
    def test_get_copy_params_with_sampled_states(self):
        """Test _get_copy_params with sampled_states."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C'], mode='sequential')
            sampled = state_sample(pool, sampled_states=[0, 2])
            params = sampled.operation._get_copy_params()
            
            assert params['parent_pool'] is pool
            assert params['num_values'] is None
            assert params['sampled_states'] == [0, 2]
            assert params['seed'] is None
