"""Tests for ShuffleCoOp and shuffle_counter()."""
import pytest
from poolparty.counter import Counter, CounterManager, ShuffleCoOp, shuffle_counter


class TestShuffleOperation:
    """Test shuffle operation."""
    
    def test_shuffle_basic(self):
        """Shuffled counter has same num_states as parent."""
        with CounterManager():
            A = Counter(num_states=8, name='A')
            B = shuffle_counter(A, seed=42)
            assert B.num_states == 8
    
    def test_shuffle_visits_all_states(self):
        """Shuffled counter visits all parent states exactly once."""
        with CounterManager():
            A = Counter(num_states=5, name='A')
            B = shuffle_counter(A, seed=42)
            
            visited = []
            for _ in B:
                visited.append(A.state)
            
            # All parent states visited exactly once
            assert sorted(visited) == [0, 1, 2, 3, 4]
    
    def test_shuffle_seed_reproducibility(self):
        """Same seed produces same shuffle order."""
        with CounterManager():
            A1 = Counter(num_states=10, name='A1')
            B1 = shuffle_counter(A1, seed=123)
            
            A2 = Counter(num_states=10, name='A2')
            B2 = shuffle_counter(A2, seed=123)
            
            states1 = []
            for _ in B1:
                states1.append(A1.state)
            
            states2 = []
            for _ in B2:
                states2.append(A2.state)
            
            assert states1 == states2
    
    def test_shuffle_different_seeds_different_order(self):
        """Different seeds produce different shuffle orders."""
        with CounterManager():
            A1 = Counter(num_states=10, name='A1')
            B1 = shuffle_counter(A1, seed=1)
            
            A2 = Counter(num_states=10, name='A2')
            B2 = shuffle_counter(A2, seed=2)
            
            states1 = []
            for _ in B1:
                states1.append(A1.state)
            
            states2 = []
            for _ in B2:
                states2.append(A2.state)
            
            # Very unlikely to be the same
            assert states1 != states2
    
    def test_shuffle_propagation(self):
        """Shuffle propagates state to parent correctly."""
        with CounterManager():
            A = Counter(num_states=5, name='A')
            B = shuffle_counter(A, seed=42)
            
            # Get the expected permutation for seed=42
            import random
            indices = list(range(5))
            random.Random(42).shuffle(indices)
            
            for i in range(5):
                B.state = i
                assert A.state == indices[i]
    
    def test_shuffle_iteration(self):
        """Iterate shuffled counter and check all states reached."""
        with CounterManager():
            A = Counter(num_states=4, name='A')
            B = shuffle_counter(A, seed=99)
            
            results = []
            for b_state in B:
                results.append((b_state, A.state))
            
            # B states should be 0, 1, 2, 3 in order
            b_states = [r[0] for r in results]
            assert b_states == [0, 1, 2, 3]
            
            # A states should be some permutation
            a_states = [r[1] for r in results]
            assert sorted(a_states) == [0, 1, 2, 3]
    
    def test_shuffle_composition_with_product(self):
        """Composition: shuffle(A * B)."""
        with CounterManager():
            A = Counter(num_states=2, name='A')
            B = Counter(num_states=3, name='B')
            C = A * B  # 6 states
            D = shuffle_counter(C, seed=42)
            
            assert D.num_states == 6
            
            visited_c = []
            for _ in D:
                visited_c.append(C.state)
            
            # All C states visited exactly once
            assert sorted(visited_c) == [0, 1, 2, 3, 4, 5]
    
    def test_shuffle_with_slice(self):
        """Composition: shuffle(A[1:5])."""
        with CounterManager():
            A = Counter(num_states=8, name='A')
            B = A[1:5]  # 4 states mapping to A states 1, 2, 3, 4
            C = shuffle_counter(B, seed=42)
            
            assert C.num_states == 4
            
            visited_a = []
            for _ in C:
                visited_a.append(A.state)
            
            # Should visit A states 1, 2, 3, 4 in shuffled order
            assert sorted(visited_a) == [1, 2, 3, 4]
    
    def test_shuffle_of_shuffle(self):
        """Double shuffle: shuffle(shuffle(A))."""
        with CounterManager():
            A = Counter(num_states=5, name='A')
            B = shuffle_counter(A, seed=42)
            C = shuffle_counter(B, seed=99)
            
            assert C.num_states == 5
            
            visited_a = []
            for _ in C:
                visited_a.append(A.state)
            
            # All A states visited exactly once
            assert sorted(visited_a) == [0, 1, 2, 3, 4]
    
    def test_shuffle_with_name(self):
        """Shuffle counter can have a name."""
        with CounterManager():
            A = Counter(num_states=5, name='A')
            B = shuffle_counter(A, seed=42, name='Shuffled')
            assert B.name == 'Shuffled'
    
    def test_shuffle_inactive_state(self):
        """Shuffle handles inactive state (None)."""
        with CounterManager():
            A = Counter(num_states=5, name='A')
            B = shuffle_counter(A, seed=42)
            B.state = None
            assert A.state is None
    
    def test_shuffle_single_state(self):
        """Shuffle of single-state counter works."""
        with CounterManager():
            A = Counter(num_states=1, name='A')
            B = shuffle_counter(A, seed=42)
            
            assert B.num_states == 1
            B.state = 0
            assert A.state == 0


class TestShuffleCoOp:
    """Test ShuffleCoOp class directly."""
    
    def test_shuffle_co_op_compute_num_states(self):
        """num_states equals parent num_states."""
        op = ShuffleCoOp(seed=42, num_parent_states=8)
        assert op.compute_num_states((8,)) == 8
    
    def test_shuffle_co_op_decompose(self):
        """decompose maps to permuted index."""
        op = ShuffleCoOp(seed=42, num_parent_states=5)
        
        # The permutation should be deterministic
        import random
        expected = list(range(5))
        random.Random(42).shuffle(expected)
        
        for i in range(5):
            assert op.decompose(i, (5,)) == (expected[i],)
    
    def test_shuffle_co_op_permutation_is_valid(self):
        """Permutation is a valid permutation of [0, n-1]."""
        op = ShuffleCoOp(seed=42, num_parent_states=10)
        assert sorted(op.permutation) == list(range(10))
    
    def test_shuffle_co_op_inactive(self):
        """decompose handles inactive state (None)."""
        op = ShuffleCoOp(seed=42, num_parent_states=8)
        assert op.decompose(None, (8,)) == (None,)
    
    def test_shuffle_co_op_seed_stored(self):
        """Seed is stored for reference."""
        op = ShuffleCoOp(seed=123, num_parent_states=5)
        assert op.seed == 123


class TestShuffleCounterFunction:
    """Test shuffle_counter() helper function."""
    
    def test_shuffle_counter_basic(self):
        """Basic shuffle_counter usage."""
        with CounterManager():
            A = Counter(num_states=8, name='A')
            B = shuffle_counter(A, seed=42)
            assert B.num_states == 8
    
    def test_shuffle_counter_with_name(self):
        """shuffle_counter with name parameter."""
        with CounterManager():
            A = Counter(num_states=8, name='A')
            B = shuffle_counter(A, seed=42, name='Shuffled')
            assert B.name == 'Shuffled'
    
    def test_shuffle_counter_not_counter_raises(self):
        """shuffle_counter with non-Counter raises TypeError."""
        with pytest.raises(TypeError, match="Expected Counter"):
            shuffle_counter("not a counter", seed=42)
    
    def test_shuffle_counter_no_seed(self):
        """shuffle_counter without seed uses random seed."""
        with CounterManager():
            A = Counter(num_states=5, name='A')
            B = shuffle_counter(A)  # No seed provided
            
            assert B.num_states == 5
            
            # Should still visit all states
            visited = []
            for _ in B:
                visited.append(A.state)
            assert sorted(visited) == [0, 1, 2, 3, 4]
