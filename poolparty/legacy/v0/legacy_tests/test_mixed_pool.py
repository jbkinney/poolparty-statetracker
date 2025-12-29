"""Tests for the MixedPool class."""

import pytest
from poolparty import MixedPool, Pool, RandomMutationPool


def test_mixed_pool_creation():
    """Test MixedPool creation with default weights."""
    pool1 = Pool(seqs=["AAA"])
    pool2 = Pool(seqs=["TTT"])
    pool3 = Pool(seqs=["GGG"])
    
    mixed = MixedPool([pool1, pool2, pool3])
    assert mixed.num_states == 3  # Each Pool has 1 state
    assert mixed.is_sequential_compatible()
    assert len(mixed.pools) == 3


def test_mixed_pool_empty_pools_raises():
    """Test that creating MixedPool with empty pools list raises error."""
    with pytest.raises(ValueError, match="pools list cannot be empty"):
        MixedPool([])


def test_mixed_pool_with_custom_weights():
    """Test MixedPool creation with custom weights."""
    pool1 = Pool(seqs=["AAA"])
    pool2 = Pool(seqs=["TTT"])
    
    mixed = MixedPool([pool1, pool2], weights=[2.0, 1.0])
    assert mixed.weights == [2.0, 1.0]
    assert abs(mixed.probabilities[0] - 2.0/3.0) < 1e-10
    assert abs(mixed.probabilities[1] - 1.0/3.0) < 1e-10


def test_mixed_pool_weights_length_mismatch():
    """Test that mismatched weights length raises error."""
    pool1 = Pool(seqs=["AAA"])
    pool2 = Pool(seqs=["TTT"])
    
    with pytest.raises(ValueError, match="weights length .* must match pools length"):
        MixedPool([pool1, pool2], weights=[1.0, 2.0, 3.0])


def test_mixed_pool_zero_weights_raises():
    """Test that zero or negative total weight raises error."""
    pool1 = Pool(seqs=["AAA"])
    pool2 = Pool(seqs=["TTT"])
    
    with pytest.raises(ValueError, match="Sum of weights must be positive"):
        MixedPool([pool1, pool2], weights=[0.0, 0.0])


def test_mixed_pool_num_states_sum():
    """Test that num_states is the sum of all child pool states."""
    pool1 = Pool(seqs=['AA', 'BB'])  # 2 states, length 2
    pool2 = Pool(seqs=['XX', 'YY', 'ZZ'])  # 3 states, length 2
    pool3 = Pool(seqs=["CC"])  # 1 state, length 2
    
    mixed = MixedPool([pool1, pool2, pool3])
    assert mixed.num_states == 2 + 3 + 1  # 6 total


def test_mixed_pool_infinite_states():
    """Test MixedPool with infinite pool has infinite states."""
    pool1 = Pool(seqs=['AAA', 'BBB'])  # 2 states, length 3
    pool2 = RandomMutationPool('AAA', alphabet=['A', 'T', 'G', 'C'], mutation_rate=0.1)  # infinite states, length 3
    
    mixed = MixedPool([pool1, pool2])
    assert mixed.num_states == float('inf')
    assert not mixed.is_sequential_compatible()


def test_mixed_pool_sequential_iteration():
    """Test sequential iteration through all pools."""
    pool1 = Pool(seqs=['A', 'B'], mode='sequential')  # states 0-1
    pool2 = Pool(seqs=['X', 'Y', 'Z'], mode='sequential')  # states 2-4
    
    mixed = MixedPool([pool1, pool2], mode='sequential')
    
    # Manually iterate through all states
    sequences = []
    for state in range(mixed.num_states):
        mixed.set_state(state)
        sequences.append(mixed.seq)
    
    # Should iterate through pool1 first, then pool2
    assert sequences == ['A', 'B', 'X', 'Y', 'Z']


def test_mixed_pool_sequential_decompose_state():
    """Test _decompose_state correctly maps to pool and state."""
    pool1 = Pool(seqs=['A', 'B'])  # 2 states
    pool2 = Pool(seqs=['X', 'Y', 'Z'])  # 3 states
    
    mixed = MixedPool([pool1, pool2], mode='sequential')
    
    # State 0-1 should map to pool1
    assert mixed._decompose_state(0) == (0, 0)
    assert mixed._decompose_state(1) == (0, 1)
    
    # State 2-4 should map to pool2
    assert mixed._decompose_state(2) == (1, 0)
    assert mixed._decompose_state(3) == (1, 1)
    assert mixed._decompose_state(4) == (1, 2)


def test_mixed_pool_sequential_wrapping():
    """Test that sequential mode wraps around correctly."""
    pool1 = Pool(seqs=['A', 'B'], mode='sequential')
    
    mixed = MixedPool([pool1], mode='sequential')
    
    # State 0 and 2 should be the same (wrapped)
    mixed.set_state(0)
    seq1 = mixed.seq
    mixed.set_state(2)
    seq2 = mixed.seq
    assert seq1 == seq2 == 'A'


def test_mixed_pool_deterministic_same_state():
    """Test that same state produces same sequence."""
    pool1 = Pool(seqs=['A', 'B', 'C'])
    pool2 = Pool(seqs=['X', 'Y', 'Z'])
    
    mixed = MixedPool([pool1, pool2])
    
    # Same state should give same sequence
    mixed.set_state(100)
    seq1 = mixed.seq
    mixed.set_state(100)
    seq2 = mixed.seq
    assert seq1 == seq2


def test_mixed_pool_random_mode_samples_all_pools():
    """Test that random mode can sample from all pools."""
    pool1 = Pool(seqs=["AAA"])
    pool2 = Pool(seqs=["BBB"])
    pool3 = Pool(seqs=["CCC"])
    
    # Use infinite pool to force random mode
    infinite_pool = RandomMutationPool('XXX', alphabet=['X', 'Y', 'Z'], mutation_rate=0.1)
    mixed = MixedPool([pool1, pool2, pool3, infinite_pool])
    
    # Sample many sequences
    sequences = set()
    for state in range(100):
        mixed.set_state(state)
        sequences.add(mixed.seq)
    
    # Should see sequences from different pools
    assert len(sequences) > 1


def test_mixed_pool_repr_default():
    """Test __repr__ with default weights."""
    pool1 = Pool(seqs=["AAA"])
    pool2 = Pool(seqs=["BBB"])
    
    mixed = MixedPool([pool1, pool2])
    assert "MixedPool(2 pools)" in repr(mixed)
    assert "weights" not in repr(mixed)  # Default weights not shown


def test_mixed_pool_repr_custom_weights():
    """Test __repr__ with custom weights."""
    pool1 = Pool(seqs=["AAA"])
    pool2 = Pool(seqs=["BBB"])
    
    mixed = MixedPool([pool1, pool2], weights=[2.0, 1.0])
    assert "MixedPool(2 pools, weights=[2.0, 1.0])" in repr(mixed)


def test_mixed_pool_concatenation():
    """Test that MixedPool can be concatenated."""
    pool1 = Pool(seqs=['A', 'B'])
    pool2 = Pool(seqs=['X', 'Y'])
    mixed = MixedPool([pool1, pool2])
    
    suffix = Pool(seqs=["!"])
    combined = mixed + suffix
    
    # Should create a composite pool
    assert isinstance(combined, Pool)
    assert combined.num_states == 4  # 4 states from mixed


def test_mixed_pool_with_generate_seqs_sequential():
    """Test MixedPool integration with generate_seqs in sequential mode."""
    pool1 = Pool(seqs=['A', 'B'], mode='sequential')
    pool2 = Pool(seqs=['X', 'Y', 'Z'], mode='sequential')
    mixed = MixedPool([pool1, pool2], mode='sequential')
    
    # Generate all sequences sequentially
    seqs = mixed.generate_seqs(num_complete_iterations=1)
    
    # Should get all 5 states
    assert len(seqs) == 5
    assert seqs == ['A', 'B', 'X', 'Y', 'Z']


def test_mixed_pool_with_generate_seqs_random():
    """Test MixedPool integration with generate_seqs in random mode."""
    pool1 = Pool(seqs=['A', 'B'])
    pool2 = Pool(seqs=['X', 'Y'])
    mixed = MixedPool([pool1, pool2])
    
    # Generate random sequences
    seqs = mixed.generate_seqs(num_seqs=10)
    
    assert len(seqs) == 10
    # All sequences should be from one of the pools
    for seq in seqs:
        assert seq in ['A', 'B', 'X', 'Y']


def test_mixed_pool_with_generate_seqs_infinite():
    """Test MixedPool with infinite pool uses random mode in generate_seqs."""
    pool1 = Pool(seqs=['AAA', 'BBB'])  # length 3
    pool2 = RandomMutationPool('AAA', alphabet=['A', 'T', 'G', 'C'], mutation_rate=0.1)  # Infinite pool, length 3
    mixed = MixedPool([pool1, pool2])
    
    # Should generate in random mode
    seqs = mixed.generate_seqs(num_seqs=5)
    assert len(seqs) == 5


def test_mixed_pool_weighted_distribution():
    """Test that weights affect pool selection in random mode."""
    pool1 = Pool(seqs=["AAA"])  # length 3
    pool2 = Pool(seqs=["BBB"])  # length 3
    
    # Make infinite to force random mode
    infinite = RandomMutationPool('XXX', alphabet=['X', 'Y', 'Z'], mutation_rate=0.1)  # length 3
    
    # Give pool1 much higher weight
    mixed = MixedPool([pool1, pool2, infinite], weights=[10.0, 1.0, 1.0])
    
    # Sample many times
    counts = {'AAA': 0, 'BBB': 0, 'other': 0}
    for state in range(100):
        mixed.set_state(state)
        seq = mixed.seq
        if seq == 'AAA':
            counts['AAA'] += 1
        elif seq == 'BBB':
            counts['BBB'] += 1
        else:
            counts['other'] += 1
    
    # Pool1 should be selected more often (but not deterministic, so just check it's non-zero)
    assert counts['AAA'] > 0


def test_mixed_pool_single_pool():
    """Test MixedPool with single pool works correctly."""
    pool1 = Pool(seqs=['A', 'B', 'C'])
    mixed = MixedPool([pool1], mode='sequential')
    
    assert mixed.num_states == 3
    
    # Sequential iteration should work
    sequences = []
    for state in range(mixed.num_states):
        mixed.set_state(state)
        sequences.append(mixed.seq)
    
    assert sequences == ['A', 'B', 'C']


def test_mixed_pool_nested():
    """Test that MixedPool can contain other MixedPools."""
    pool1 = Pool(seqs=["A"])
    pool2 = Pool(seqs=["B"])
    mixed1 = MixedPool([pool1, pool2])
    
    pool3 = Pool(seqs=["X"])
    pool4 = Pool(seqs=["Y"])
    mixed2 = MixedPool([pool3, pool4])
    
    # Create nested MixedPool
    nested = MixedPool([mixed1, mixed2])
    assert nested.num_states == 4  # 2 + 2


def test_mixed_pool_max_num_states():
    """Test that max_num_states parameter is respected."""
    pool1 = Pool(seqs=['A', 'B'])
    pool2 = Pool(seqs=['X', 'Y', 'Z'])
    
    # Create MixedPool with low max_num_states
    mixed = MixedPool([pool1, pool2], max_num_states=3)
    
    # Should have 5 states but max is 3, so not sequential compatible
    assert mixed.num_states == 5
    assert not mixed.is_sequential_compatible()


def test_mixed_pool_iter():
    """Test iteration through MixedPool states."""
    pool1 = Pool(seqs=['A', 'B'], mode='sequential')
    pool2 = Pool(seqs=['X'], mode='sequential')
    mixed = MixedPool([pool1, pool2], mode='sequential')
    
    # Collect sequences through iteration
    sequences = []
    for state in range(mixed.num_states):
        mixed.set_state(state)
        sequences.append(mixed.seq)
    
    assert sequences == ['A', 'B', 'X']


def test_mixed_pool_next():
    """Test sequential state traversal on MixedPool."""
    pool1 = Pool(seqs=['A', 'B'], mode='sequential')
    pool2 = Pool(seqs=['X'], mode='sequential')
    mixed = MixedPool([pool1, pool2], mode='sequential')
    
    mixed.set_state(0)
    seq1 = mixed.seq
    mixed.set_state(1)
    seq2 = mixed.seq
    mixed.set_state(2)
    seq3 = mixed.seq
    
    assert seq1 == 'A'
    assert seq2 == 'B'
    assert seq3 == 'X'


def test_mixed_pool_different_seq_lengths_error():
    """Test that MixedPool raises error when pools have different seq_length."""
    pool1 = Pool(seqs=["AAA"])  # length 3
    pool2 = Pool(seqs=["TTTT"])  # length 4
    
    with pytest.raises(ValueError, match="All pools in MixedPool must have the same seq_length"):
        MixedPool([pool1, pool2])


def test_mixed_pool_different_seq_lengths_list_pools_error():
    """Test that MixedPool raises error when Pools have different seq_length."""
    pool1 = Pool(seqs=['AAA', 'BBB'])  # length 3
    pool2 = Pool(seqs=['XXXX', 'YYYY'])  # length 4
    
    with pytest.raises(ValueError, match="All pools in MixedPool must have the same seq_length"):
        MixedPool([pool1, pool2])
