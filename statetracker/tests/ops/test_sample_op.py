"""Tests for SampleOp and sample()."""
import pytest
from beartype.roar import BeartypeCallHintParamViolation
from statetracker import State, Manager, SampleOp, sample, product


class TestSampleOperation:
    """Test sample operation."""
    
    def test_sample_with_num_states(self):
        """Sample with num_states creates correct number of states."""
        with Manager():
            A = State(num_values=10, name='A')
            B = sample(A, num_values=5, seed=42)
            assert B.num_values == 5
    
    def test_sample_with_sampled_states(self):
        """Sample with explicit sampled_states works correctly."""
        with Manager():
            A = State(num_values=10, name='A')
            B = sample(A, sampled_states=[2, 5, 7])
            assert B.num_values == 3
            
            # Verify correct mapping
            for i, expected in enumerate([2, 5, 7]):
                B.value = i
                assert A.value == expected
    
    def test_sample_seed_reproducibility(self):
        """Same seed produces same sample."""
        with Manager():
            A1 = State(num_values=20, name='A1')
            B1 = sample(A1, num_values=5, seed=123)
            
            A2 = State(num_values=20, name='A2')
            B2 = sample(A2, num_values=5, seed=123)
            
            states1 = []
            for _ in B1:
                states1.append(A1.value)
            
            states2 = []
            for _ in B2:
                states2.append(A2.value)
            
            assert states1 == states2
    
    def test_sample_different_seeds_different_results(self):
        """Different seeds produce different samples."""
        with Manager():
            A1 = State(num_values=20, name='A1')
            B1 = sample(A1, num_values=5, seed=1)
            
            A2 = State(num_values=20, name='A2')
            B2 = sample(A2, num_values=5, seed=2)
            
            states1 = []
            for _ in B1:
                states1.append(A1.value)
            
            states2 = []
            for _ in B2:
                states2.append(A2.value)
            
            # Very unlikely to be the same
            assert states1 != states2
    
    def test_sample_with_replacement_true(self):
        """With with_replacement=True, can sample more states than parent."""
        with Manager():
            A = State(num_values=3, name='A')
            B = sample(A, num_values=10, seed=42, with_replacement=True)
            assert B.num_values == 10
            
            # All parent states should be in valid range
            for _ in B:
                assert 0 <= A.value < 3
    
    def test_sample_with_replacement_false_valid(self):
        """With with_replacement=False, can sample fewer states than parent."""
        with Manager():
            A = State(num_values=10, name='A')
            B = sample(A, num_values=5, seed=42, with_replacement=False)
            assert B.num_values == 5
            
            # All parent states should be unique
            visited = []
            for _ in B:
                visited.append(A.value)
            assert len(visited) == len(set(visited))
    
    def test_sample_with_replacement_false_exceeds_raises(self):
        """With with_replacement=False, exceeding parent.num_values raises."""
        with Manager():
            A = State(num_values=5, name='A')
            with pytest.raises(ValueError, match="exceeds parent.num_values"):
                sample(A, num_values=10, with_replacement=False)
    
    def test_sample_propagation(self):
        """Sample propagates state to parent correctly."""
        with Manager():
            A = State(num_values=10, name='A')
            sampled = [3, 7, 1, 9]
            B = sample(A, sampled_states=sampled)
            
            for i, expected in enumerate(sampled):
                B.value = i
                assert A.value == expected
    
    def test_sample_iteration(self):
        """Iterate sampled state and check states."""
        with Manager():
            A = State(num_values=10, name='A')
            sampled = [5, 2, 8]
            B = sample(A, sampled_states=sampled)
            
            results = []
            for b_state in B:
                results.append((b_state, A.value))
            
            # B states should be 0, 1, 2
            b_states = [r[0] for r in results]
            assert b_states == [0, 1, 2]
            
            # A states should match sampled
            a_states = [r[1] for r in results]
            assert a_states == sampled
    
    def test_sample_with_name(self):
        """Sample state can have a name."""
        with Manager():
            A = State(num_values=10, name='A')
            B = sample(A, num_values=3, seed=42, name='Sampled')
            assert B.name == 'Sampled'
    
    def test_sample_inactive_state(self):
        """Sample handles inactive state (None)."""
        with Manager():
            A = State(num_values=10, name='A')
            B = sample(A, sampled_states=[2, 5])
            B.value = None
            assert A.value == 0
            assert B.value is None
    
    def test_sample_single_state(self):
        """Sample single state works."""
        with Manager():
            A = State(num_values=10, name='A')
            B = sample(A, sampled_states=[7])
            
            assert B.num_values == 1
            B.value = 0
            assert A.value == 7
    
    def test_sample_with_duplicates_explicit(self):
        """Explicit sampled_states can have duplicates."""
        with Manager():
            A = State(num_values=5, name='A')
            B = sample(A, sampled_states=[2, 2, 3, 2])
            
            assert B.num_values == 4
            
            expected = [2, 2, 3, 2]
            for i, exp in enumerate(expected):
                B.value = i
                assert A.value == exp


class TestSampleOpValidation:
    """Test SampleOp validation."""
    
    def test_must_specify_num_states_or_sampled_states(self):
        """Must specify either num_states or sampled_states."""
        with Manager():
            A = State(num_values=10, name='A')
            with pytest.raises(ValueError, match="Must specify either"):
                sample(A)
    
    def test_cannot_specify_both_num_states_and_sampled_states(self):
        """Cannot specify both num_states and sampled_states."""
        with Manager():
            A = State(num_values=10, name='A')
            with pytest.raises(ValueError, match="Cannot specify both"):
                sample(A, num_values=5, sampled_states=[1, 2, 3])
    
    def test_cannot_specify_seed_with_sampled_states(self):
        """Cannot specify seed with sampled_states."""
        with Manager():
            A = State(num_values=10, name='A')
            with pytest.raises(ValueError, match="Cannot specify 'seed' with 'sampled_states'"):
                sample(A, sampled_states=[1, 2, 3], seed=42)
    
    def test_sampled_states_out_of_range_raises(self):
        """sampled_states with out-of-range values raises."""
        with Manager():
            A = State(num_values=5, name='A')
            with pytest.raises(ValueError, match="out of range"):
                sample(A, sampled_states=[1, 2, 10])
    
    def test_sampled_states_negative_raises(self):
        """sampled_states with negative values raises."""
        with Manager():
            A = State(num_values=5, name='A')
            with pytest.raises(ValueError, match="out of range"):
                sample(A, sampled_states=[1, -1, 2])


class TestSampleOp:
    """Test SampleOp class directly."""
    
    def test_sample_op_compute_num_states(self):
        """num_states equals length of sampled_states."""
        op = SampleOp(num_parent_values=10, sampled_states=[1, 3, 5])
        assert op.compute_num_states((10,)) == 3
    
    def test_sample_op_decompose(self):
        """decompose maps to sampled index."""
        op = SampleOp(num_parent_values=10, sampled_states=[7, 2, 9])
        
        assert op.decompose(0, (10,)) == (7,)
        assert op.decompose(1, (10,)) == (2,)
        assert op.decompose(2, (10,)) == (9,)
    
    def test_sample_op_inactive(self):
        """decompose handles inactive state (None)."""
        op = SampleOp(num_parent_values=10, sampled_states=[1, 2, 3])
        assert op.decompose(None, (10,)) == (None,)
    
    def test_sample_op_seed_stored(self):
        """Seed is stored for reference."""
        op = SampleOp(num_parent_values=10, num_values=5, seed=123)
        assert op.seed == 123
    
    def test_sample_op_seed_none_with_sampled_states(self):
        """Seed is None when using sampled_states."""
        op = SampleOp(num_parent_values=10, sampled_states=[1, 2, 3])
        assert op.seed is None
    
    def test_sample_op_no_seed_generates_random(self):
        """SampleOp with num_states but no seed generates random seed."""
        op = SampleOp(num_parent_values=10, num_values=5)
        
        assert op.seed is not None
        assert len(op.sampled_states) == 5


class TestSampleComposition:
    """Test sample with other operations."""
    
    def test_sample_of_product(self):
        """Composition: sample(product([A, B]))."""
        with Manager():
            A = State(num_values=3, name='A')
            B = State(num_values=4, name='B')
            C = product([A, B])  # 12 states
            D = sample(C, num_values=5, seed=42)
            
            assert D.num_values == 5
            
            for _ in D:
                assert 0 <= C.value < 12
    
    def test_sample_of_slice(self):
        """Composition: sample(A[2:8])."""
        with Manager():
            A = State(num_values=10, name='A')
            B = A[2:8]  # 6 states mapping to A states 2, 3, 4, 5, 6, 7
            C = sample(B, sampled_states=[0, 2, 4])
            
            assert C.num_values == 3
            
            expected_a = [2, 4, 6]  # B states 0, 2, 4 map to A states 2, 4, 6
            for i, exp in enumerate(expected_a):
                C.value = i
                assert A.value == exp
    
    def test_sample_then_slice(self):
        """Composition: sample(A)[1:3]."""
        with Manager():
            A = State(num_values=10, name='A')
            B = sample(A, sampled_states=[5, 2, 8, 1, 9])
            C = B[1:3]  # States 1, 2 of B which are A states 2, 8
            
            assert C.num_values == 2
            
            C.value = 0
            assert A.value == 2
            C.value = 1
            assert A.value == 8


class TestSampleFunction:
    """Test sample() helper function."""
    
    def test_sample_basic(self):
        """Basic sample usage."""
        with Manager():
            A = State(num_values=10, name='A')
            B = sample(A, num_values=5, seed=42)
            assert B.num_values == 5
    
    def test_sample_with_name(self):
        """sample with name parameter."""
        with Manager():
            A = State(num_values=10, name='A')
            B = sample(A, num_values=5, seed=42, name='Sampled')
            assert B.name == 'Sampled'
    
    def test_sample_not_state_raises(self):
        """sample with non-State raises BeartypeCallHintParamViolation."""
        with pytest.raises(BeartypeCallHintParamViolation):
            sample("not a state", num_values=5)
    
    def test_sample_no_seed_generates_random(self):
        """sample without seed uses random seed."""
        with Manager():
            A = State(num_values=10, name='A')
            B = sample(A, num_values=5)  # No seed provided
            
            assert B.num_values == 5
