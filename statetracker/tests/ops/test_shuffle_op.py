"""Tests for ShuffleOp and shuffle_state()."""
import pytest
from beartype.roar import BeartypeCallHintParamViolation
from statetracker import State, Manager, ShuffleOp, shuffle, product


class TestShuffleOperation:
    """Test shuffle operation."""
    
    def test_shuffle_basic(self):
        """Shuffled state has same num_states as parent."""
        with Manager():
            A = State(num_values=8, name='A')
            B = shuffle(A, seed=42)
            assert B.num_values == 8
    
    def test_shuffle_visits_all_states(self):
        """Shuffled state visits all parent states exactly once."""
        with Manager():
            A = State(num_values=5, name='A')
            B = shuffle(A, seed=42)
            
            visited = []
            for _ in B:
                visited.append(A.value)
            
            # All parent states visited exactly once
            assert sorted(visited) == [0, 1, 2, 3, 4]
    
    def test_shuffle_seed_reproducibility(self):
        """Same seed produces same shuffle order."""
        with Manager():
            A1 = State(num_values=10, name='A1')
            B1 = shuffle(A1, seed=123)
            
            A2 = State(num_values=10, name='A2')
            B2 = shuffle(A2, seed=123)
            
            states1 = []
            for _ in B1:
                states1.append(A1.value)
            
            states2 = []
            for _ in B2:
                states2.append(A2.value)
            
            assert states1 == states2
    
    def test_shuffle_different_seeds_different_order(self):
        """Different seeds produce different shuffle orders."""
        with Manager():
            A1 = State(num_values=10, name='A1')
            B1 = shuffle(A1, seed=1)
            
            A2 = State(num_values=10, name='A2')
            B2 = shuffle(A2, seed=2)
            
            states1 = []
            for _ in B1:
                states1.append(A1.value)
            
            states2 = []
            for _ in B2:
                states2.append(A2.value)
            
            # Very unlikely to be the same
            assert states1 != states2
    
    def test_shuffle_propagation(self):
        """Shuffle propagates state to parent correctly."""
        with Manager():
            A = State(num_values=5, name='A')
            B = shuffle(A, seed=42)
            
            # Get the expected permutation for seed=42
            import random
            indices = list(range(5))
            random.Random(42).shuffle(indices)
            
            for i in range(5):
                B.value = i
                assert A.value == indices[i]
    
    def test_shuffle_iteration(self):
        """Iterate shuffled state and check all states reached."""
        with Manager():
            A = State(num_values=4, name='A')
            B = shuffle(A, seed=99)
            
            results = []
            for b_state in B:
                results.append((b_state, A.value))
            
            # B states should be 0, 1, 2, 3 in order
            b_states = [r[0] for r in results]
            assert b_states == [0, 1, 2, 3]
            
            # A states should be some permutation
            a_states = [r[1] for r in results]
            assert sorted(a_states) == [0, 1, 2, 3]
    
    def test_shuffle_composition_with_product(self):
        """Composition: shuffle(product([A, B]))."""
        with Manager():
            A = State(num_values=2, name='A')
            B = State(num_values=3, name='B')
            C = product([A, B])  # 6 states
            D = shuffle(C, seed=42)
            
            assert D.num_values == 6
            
            visited_c = []
            for _ in D:
                visited_c.append(C.value)
            
            # All C states visited exactly once
            assert sorted(visited_c) == [0, 1, 2, 3, 4, 5]
    
    def test_shuffle_with_slice(self):
        """Composition: shuffle(A[1:5])."""
        with Manager():
            A = State(num_values=8, name='A')
            B = A[1:5]  # 4 states mapping to A states 1, 2, 3, 4
            C = shuffle(B, seed=42)
            
            assert C.num_values == 4
            
            visited_a = []
            for _ in C:
                visited_a.append(A.value)
            
            # Should visit A states 1, 2, 3, 4 in shuffled order
            assert sorted(visited_a) == [1, 2, 3, 4]
    
    def test_shuffle_of_shuffle(self):
        """Double shuffle: shuffle(shuffle(A))."""
        with Manager():
            A = State(num_values=5, name='A')
            B = shuffle(A, seed=42)
            C = shuffle(B, seed=99)
            
            assert C.num_values == 5
            
            visited_a = []
            for _ in C:
                visited_a.append(A.value)
            
            # All A states visited exactly once
            assert sorted(visited_a) == [0, 1, 2, 3, 4]
    
    def test_shuffle_with_name(self):
        """Shuffle state can have a name."""
        with Manager():
            A = State(num_values=5, name='A')
            B = shuffle(A, seed=42, name='Shuffled')
            assert B.name == 'Shuffled'
    
    def test_shuffle_inactive_state(self):
        """Shuffle handles inactive state (None)."""
        with Manager():
            A = State(num_values=5, name='A')
            B = shuffle(A, seed=42)
            B.value = None
            # Setting derived state to None doesn't propagate to parent
            # A remains at its default state (0 for leaf state)
            assert A.value == 0
            assert B.value is None
    
    def test_shuffle_single_state(self):
        """Shuffle of single-state state works."""
        with Manager():
            A = State(num_values=1, name='A')
            B = shuffle(A, seed=42)
            
            assert B.num_values == 1
            B.value = 0
            assert A.value == 0


class TestShuffleOp:
    """Test ShuffleOp class directly."""
    
    def test_shuffle_co_op_compute_num_states(self):
        """num_states equals parent num_states."""
        op = ShuffleOp(seed=42, num_parent_values=8)
        assert op.compute_num_states((8,)) == 8
    
    def test_shuffle_co_op_decompose(self):
        """decompose maps to permuted index."""
        op = ShuffleOp(seed=42, num_parent_values=5)
        
        # The permutation should be deterministic
        import random
        expected = list(range(5))
        random.Random(42).shuffle(expected)
        
        for i in range(5):
            assert op.decompose(i, (5,)) == (expected[i],)
    
    def test_shuffle_co_op_permutation_is_valid(self):
        """Permutation is a valid permutation of [0, n-1]."""
        op = ShuffleOp(seed=42, num_parent_values=10)
        assert sorted(op.permutation) == list(range(10))
    
    def test_shuffle_co_op_inactive(self):
        """decompose handles inactive state (None)."""
        op = ShuffleOp(seed=42, num_parent_values=8)
        assert op.decompose(None, (8,)) == (None,)
    
    def test_shuffle_co_op_seed_stored(self):
        """Seed is stored for reference."""
        op = ShuffleOp(seed=123, num_parent_values=5)
        assert op.seed == 123


class TestShuffleStateFunction:
    """Test shuffle_state() helper function."""
    
    def test_shuffle_state_basic(self):
        """Basic shuffle_state usage."""
        with Manager():
            A = State(num_values=8, name='A')
            B = shuffle(A, seed=42)
            assert B.num_values == 8
    
    def test_shuffle_state_with_name(self):
        """shuffle_state with name parameter."""
        with Manager():
            A = State(num_values=8, name='A')
            B = shuffle(A, seed=42, name='Shuffled')
            assert B.name == 'Shuffled'
    
    def test_shuffle_state_not_state_raises(self):
        """shuffle_state with non-State raises BeartypeCallHintParamViolation."""
        with pytest.raises(BeartypeCallHintParamViolation):
            shuffle("not a state", seed=42)
    
    def test_shuffle_state_no_seed(self):
        """shuffle_state without seed uses random seed."""
        with Manager():
            A = State(num_values=5, name='A')
            B = shuffle(A)  # No seed provided
            
            assert B.num_values == 5
            
            # Should still visit all states
            visited = []
            for _ in B:
                visited.append(A.value)
            assert sorted(visited) == [0, 1, 2, 3, 4]


class TestShuffleWithPermutation:
    """Test shuffle with custom permutation argument."""
    
    def test_shuffle_with_permutation(self):
        """Shuffle with custom permutation uses that permutation."""
        with Manager():
            A = State(num_values=5, name='A')
            perm = [4, 3, 2, 1, 0]  # Reverse order
            B = shuffle(A, permutation=perm)
            
            assert B.num_values == 5
            
            # Verify the permutation is applied
            for i in range(5):
                B.value = i
                assert A.value == perm[i]
    
    def test_shuffle_with_permutation_identity(self):
        """Shuffle with identity permutation preserves order."""
        with Manager():
            A = State(num_values=5, name='A')
            perm = [0, 1, 2, 3, 4]  # Identity
            B = shuffle(A, permutation=perm)
            
            for i in range(5):
                B.value = i
                assert A.value == i
    
    def test_shuffle_with_permutation_iteration(self):
        """Iterate shuffled state with custom permutation."""
        with Manager():
            A = State(num_values=4, name='A')
            perm = [3, 1, 0, 2]
            B = shuffle(A, permutation=perm)
            
            a_states = []
            for _ in B:
                a_states.append(A.value)
            
            assert a_states == perm
    
    def test_shuffle_seed_and_permutation_raises(self):
        """Specifying both seed and permutation raises ValueError."""
        with Manager():
            A = State(num_values=5, name='A')
            with pytest.raises(ValueError, match="mutually exclusive"):
                shuffle(A, seed=42, permutation=[0, 1, 2, 3, 4])
    
    def test_shuffle_permutation_wrong_length_raises(self):
        """Permutation with wrong length raises ValueError."""
        with Manager():
            A = State(num_values=5, name='A')
            with pytest.raises(ValueError, match="length"):
                shuffle(A, permutation=[0, 1, 2])  # Too short
    
    def test_shuffle_permutation_wrong_contents_raises(self):
        """Permutation with wrong contents raises ValueError."""
        with Manager():
            A = State(num_values=5, name='A')
            with pytest.raises(ValueError, match="must contain exactly"):
                shuffle(A, permutation=[0, 1, 2, 3, 5])  # 5 instead of 4
    
    def test_shuffle_permutation_duplicates_raises(self):
        """Permutation with duplicates raises ValueError."""
        with Manager():
            A = State(num_values=5, name='A')
            with pytest.raises(ValueError, match="must contain exactly"):
                shuffle(A, permutation=[0, 1, 2, 3, 3])  # Duplicate 3


class TestShuffleOpWithPermutation:
    """Test ShuffleOp class with permutation argument."""
    
    def test_shuffleop_with_permutation(self):
        """ShuffleOp with custom permutation uses that permutation."""
        perm = [4, 3, 2, 1, 0]
        op = ShuffleOp(num_parent_values=5, permutation=perm)
        
        assert op.permutation == tuple(perm)
        assert op.seed is None
    
    def test_shuffleop_permutation_decompose(self):
        """ShuffleOp with permutation decomposes correctly."""
        perm = [2, 0, 1]
        op = ShuffleOp(num_parent_values=3, permutation=perm)
        
        assert op.decompose(0, (3,)) == (2,)
        assert op.decompose(1, (3,)) == (0,)
        assert op.decompose(2, (3,)) == (1,)
    
    def test_shuffleop_seed_and_permutation_raises(self):
        """ShuffleOp with both seed and permutation raises ValueError."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            ShuffleOp(num_parent_values=5, seed=42, permutation=[0, 1, 2, 3, 4])
    
    def test_shuffleop_permutation_wrong_length_raises(self):
        """ShuffleOp with wrong length permutation raises ValueError."""
        with pytest.raises(ValueError, match="length"):
            ShuffleOp(num_parent_values=5, permutation=[0, 1, 2])
    
    def test_shuffleop_permutation_wrong_contents_raises(self):
        """ShuffleOp with wrong contents raises ValueError."""
        with pytest.raises(ValueError, match="must contain exactly"):
            ShuffleOp(num_parent_values=5, permutation=[0, 1, 2, 3, 5])
    
    def test_shuffleop_no_args_generates_random(self):
        """ShuffleOp with no seed or permutation generates random permutation."""
        op = ShuffleOp(num_parent_values=5)
        
        assert op.seed is not None
        assert len(op.permutation) == 5
        assert sorted(op.permutation) == [0, 1, 2, 3, 4]
