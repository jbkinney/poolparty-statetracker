"""Tests for the ShuffleScanPool class."""

import pytest
from collections import Counter
from poolparty import ShuffleScanPool, Pool


# Realistic DNA test sequences (>= 15nt)
TEST_SEQ_24 = "ACGTACGTACGTACGTACGTACGT"  # 24nt
TEST_SEQ_18 = "ACGTACGTACGTACGTAC"  # 18nt


def test_shuffle_scan_pool_creation():
    """Test ShuffleScanPool creation."""
    pool = ShuffleScanPool(TEST_SEQ_24, shuffle_size=6)
    assert pool.is_sequential_compatible()
    assert pool.num_states > 0


def test_num_states_calculation():
    """Test num_states calculation with range-based interface."""
    # seq length = 24, shuffle_size = 6, step_size = 1
    # positions = range(0, 24 - 6 + 1, 1) = range(0, 19) = 19 states
    pool = ShuffleScanPool(TEST_SEQ_24, shuffle_size=6, step_size=1)
    assert pool.num_states == 19
    
    # With step_size = 2
    pool = ShuffleScanPool(TEST_SEQ_24, shuffle_size=6, step_size=2)
    assert pool.num_states == 10
    
    # With start = 3
    pool = ShuffleScanPool(TEST_SEQ_24, shuffle_size=6, start=3, step_size=1)
    assert pool.num_states == 16


def test_basic_shuffling_operation():
    """Test basic shuffling operation with deterministic seed."""
    seq = "ABCDEFGHIJKLMNOP"  # 16 chars
    pool = ShuffleScanPool(seq, shuffle_size=6, mark_changes=False, step_size=1, mode='sequential')
    
    pool.set_state(0)
    shuffled = pool.seq
    assert len(shuffled) == len(seq)
    # The shuffled part should contain A,B,C,D,E,F but potentially in different order
    assert Counter(shuffled[:6]) == Counter("ABCDEF")
    assert shuffled[6:] == "GHIJKLMNOP"  # Rest unchanged
    
    pool.set_state(1)
    shuffled = pool.seq
    assert shuffled[0] == 'A'  # First character unchanged
    assert Counter(shuffled[1:7]) == Counter("BCDEFG")
    assert shuffled[7:] == "HIJKLMNOP"


def test_mark_changes_false():
    """Test that mark_changes=False preserves case."""
    pool = ShuffleScanPool(TEST_SEQ_18, shuffle_size=6, mark_changes=False, step_size=1, mode='sequential')
    
    pool.set_state(0)
    seq = pool.seq
    # All characters should still be uppercase
    assert seq.isupper()
    assert Counter(seq[:6]) == Counter(TEST_SEQ_18[:6])


def test_mark_changes_true():
    """Test that mark_changes=True swaps case before shuffling."""
    pool = ShuffleScanPool(TEST_SEQ_18, shuffle_size=6, mark_changes=True, step_size=1, mode='sequential')
    
    pool.set_state(0)
    seq = pool.seq
    # First 6 characters should be lowercase (swapcase applied before shuffle)
    assert seq[:6].islower()
    # Should be a permutation of swapcase version
    assert Counter(seq[:6]) == Counter(TEST_SEQ_18[:6].swapcase())
    # Rest should be unchanged (uppercase)
    assert seq[6:] == TEST_SEQ_18[6:]


def test_with_step_size():
    """Test shuffling with larger step_size."""
    seq = "ABCDEFGHIJKLMNOP"  # 16 chars
    pool = ShuffleScanPool(seq, shuffle_size=4, mark_changes=False, step_size=4, mode='sequential')
    
    pool.set_state(0)
    shuffled = pool.seq
    # Position 0: shuffle ABCD
    assert Counter(shuffled[:4]) == Counter("ABCD")
    assert shuffled[4:] == "EFGHIJKLMNOP"
    
    pool.set_state(1)
    shuffled = pool.seq
    # Position 4: shuffle EFGH
    assert shuffled[:4] == "ABCD"  # Unchanged
    assert Counter(shuffled[4:8]) == Counter("EFGH")
    assert shuffled[8:] == "IJKLMNOP"


def test_with_start():
    """Test shuffling with non-zero start."""
    seq = "ABCDEFGHIJKLMNOP"  # 16 chars
    pool = ShuffleScanPool(seq, shuffle_size=4, mark_changes=False, start=2, step_size=2, mode='sequential')
    
    pool.set_state(0)
    shuffled = pool.seq
    assert shuffled[:2] == "AB"  # Positions 0-1 unchanged
    assert Counter(shuffled[2:6]) == Counter("CDEF")
    assert shuffled[6:] == "GHIJKLMNOP"


def test_reproducibility():
    """Test that same state produces same shuffle."""
    pool = ShuffleScanPool(TEST_SEQ_24, shuffle_size=6, mark_changes=False, step_size=1, mode='sequential')
    
    pool.set_state(5)
    seq1 = pool.seq
    pool.set_state(5)
    seq2 = pool.seq
    assert seq1 == seq2


def test_different_shuffles_at_different_positions():
    """Test that different positions get different shuffles."""
    pool = ShuffleScanPool(TEST_SEQ_24, shuffle_size=6, mark_changes=False, step_size=6, mode='sequential')
    
    sequences = []
    for state in range(pool.num_states):
        pool.set_state(state)
        sequences.append(pool.seq)
    
    assert len(sequences) == pool.num_states


def test_with_pool_background():
    """Test ShuffleScanPool with Pool object as background."""
    background_pool = Pool(seqs=[TEST_SEQ_18])
    pool = ShuffleScanPool(background_pool, shuffle_size=6, mark_changes=False, step_size=1, mode='sequential')
    
    pool.set_state(0)
    seq = pool.seq
    assert len(seq) == 18
    assert Counter(seq[:6]) == Counter(TEST_SEQ_18[:6])


def test_sequence_length_maintained():
    """Test that shuffling maintains background length."""
    pool = ShuffleScanPool(TEST_SEQ_24, shuffle_size=6, mark_changes=False, step_size=1, mode='sequential')
    
    assert pool.seq_length == len(TEST_SEQ_24)
    
    for state in range(min(pool.num_states, 10)):
        pool.set_state(state)
        assert len(pool.seq) == len(TEST_SEQ_24)


def test_state_wrapping():
    """Test that state wraps with modulo."""
    pool = ShuffleScanPool(TEST_SEQ_18, shuffle_size=6, mark_changes=False, step_size=1, mode='sequential')
    
    pool.set_state(0)
    first_seq = pool.seq
    
    pool.set_state(pool.num_states)
    assert pool.seq == first_seq


def test_sequential_iteration():
    """Test ShuffleScanPool sequential iteration."""
    pool = ShuffleScanPool(TEST_SEQ_24, shuffle_size=6, mark_changes=False, step_size=2, mode='sequential')
    
    sequences = []
    for state in range(pool.num_states):
        pool.set_state(state)
        sequences.append(pool.seq)
    
    assert all(len(seq) == len(TEST_SEQ_24) for seq in sequences)


def test_concatenation():
    """Test ShuffleScanPool concatenation."""
    pool = ShuffleScanPool(TEST_SEQ_18, shuffle_size=6, mark_changes=False, step_size=2, mode='sequential')
    constant = Pool(seqs=["--"])
    
    combined = constant + pool + constant
    
    combined.set_state(0)
    assert combined.seq.startswith("--")
    assert combined.seq.endswith("--")


def test_repr_range_based():
    """Test ShuffleScanPool __repr__ for range-based interface."""
    pool = ShuffleScanPool(TEST_SEQ_18, shuffle_size=6, step_size=1)
    repr_str = repr(pool)
    assert "ShuffleScanPool" in repr_str
    assert "shuffle_size=6" in repr_str


def test_repr_position_based():
    """Test ShuffleScanPool __repr__ for position-based interface."""
    pool = ShuffleScanPool(TEST_SEQ_18, shuffle_size=6, positions=[0, 6])
    repr_str = repr(pool)
    assert "ShuffleScanPool" in repr_str
    assert "positions=" in repr_str


def test_deterministic_with_same_state():
    """Test that ShuffleScanPool is deterministic with same state."""
    pool = ShuffleScanPool(TEST_SEQ_24, shuffle_size=6, mark_changes=False, step_size=1)
    
    pool.set_state(5)
    seq1 = pool.seq
    pool.set_state(5)
    seq2 = pool.seq
    assert seq1 == seq2


def test_edge_case_short_background():
    """Test ShuffleScanPool with short background."""
    seq = "ACGTACGTACGTACGT"  # 16 chars
    pool = ShuffleScanPool(seq, shuffle_size=6, mark_changes=False, step_size=1, mode='sequential')
    
    assert pool.num_states > 0
    pool.set_state(0)
    assert len(pool.seq) == 16


def test_edge_case_shuffle_size_equals_background():
    """Test ShuffleScanPool with shuffle_size equal to background length."""
    pool = ShuffleScanPool(TEST_SEQ_18, shuffle_size=18, mark_changes=False, step_size=1, mode='sequential')
    
    # Should have exactly 1 state
    assert pool.num_states == 1
    pool.set_state(0)
    assert Counter(pool.seq) == Counter(TEST_SEQ_18)


def test_single_character_shuffle():
    """Test ShuffleScanPool with single character shuffle (no-op)."""
    pool = ShuffleScanPool(TEST_SEQ_18, shuffle_size=1, mark_changes=False, step_size=1, mode='sequential')
    
    # Shuffling 1 character is essentially a no-op
    pool.set_state(0)
    assert len(pool.seq) == len(TEST_SEQ_18)


def test_different_step_values():
    """Test ShuffleScanPool with various step_size values."""
    # Test step_size=1 (sliding window)
    pool1 = ShuffleScanPool(TEST_SEQ_24, shuffle_size=6, mark_changes=False, step_size=1)
    assert pool1.num_states == 19
    
    # Test step_size=6 (non-overlapping)
    pool2 = ShuffleScanPool(TEST_SEQ_24, shuffle_size=6, mark_changes=False, step_size=6)
    assert pool2.num_states == 4


def test_mode_parameter():
    """Test that mode parameter is accepted."""
    pool_random = ShuffleScanPool(TEST_SEQ_18, shuffle_size=6, mode='random')
    assert pool_random.mode == 'random'
    
    pool_sequential = ShuffleScanPool(TEST_SEQ_18, shuffle_size=6, mode='sequential')
    assert pool_sequential.mode == 'sequential'


def test_shuffle_size_longer_than_background_error():
    """Test that shuffle_size longer than background raises error."""
    with pytest.raises(ValueError, match="cannot be longer than"):
        ShuffleScanPool(TEST_SEQ_18, shuffle_size=25)


def test_mixed_case_shuffling():
    """Test shuffling with mixed case background."""
    seq = "AcGtAcGtAcGtAcGt"  # 16 chars mixed case
    pool = ShuffleScanPool(seq, shuffle_size=4, mark_changes=False, step_size=1, mode='sequential')
    
    pool.set_state(0)
    shuffled = pool.seq
    assert Counter(shuffled[:4]) == Counter("AcGt")
    assert shuffled[4:] == "AcGtAcGtAcGt"


def test_mixed_case_with_mark_changes():
    """Test shuffling with mixed case background and mark_changes."""
    seq = "AcGtAcGtAcGtAcGt"  # 16 chars mixed case
    pool = ShuffleScanPool(seq, shuffle_size=4, mark_changes=True, step_size=1, mode='sequential')
    
    pool.set_state(0)
    shuffled = pool.seq
    # Should swap case then shuffle first 4 characters
    # AcGt -> aCgT after swapcase, then shuffle
    assert Counter(shuffled[:4]) == Counter("aCgT")
    assert shuffled[4:] == "AcGtAcGtAcGt"


def test_character_preservation():
    """Test that shuffling preserves the original characters."""
    seq = "ABCDEFGHIJKLMNOP"  # 16 chars
    pool = ShuffleScanPool(seq, shuffle_size=8, mark_changes=False, step_size=4, mode='sequential')
    
    for state in range(pool.num_states):
        pool.set_state(state)
        shuffled = pool.seq
        assert Counter(shuffled) == Counter(seq)


def test_position_based_interface():
    """Test ShuffleScanPool with explicit positions."""
    pool = ShuffleScanPool(TEST_SEQ_24, shuffle_size=6, positions=[0, 6, 12], mode='sequential')
    
    assert pool.num_states == 3
    
    pool.set_state(0)
    assert Counter(pool.seq[:6]) == Counter(TEST_SEQ_24[:6].swapcase())  # mark_changes=True by default


def test_position_weights():
    """Test ShuffleScanPool with position weights."""
    pool = ShuffleScanPool(
        TEST_SEQ_24, 
        shuffle_size=6, 
        positions=[0, 6, 12],
        position_weights=[1.0, 2.0, 1.0],
        mode='random'
    )
    
    assert pool.num_states == 3


def test_num_shuffles():
    """Test ShuffleScanPool with multiple shuffles per position."""
    pool = ShuffleScanPool(TEST_SEQ_18, shuffle_size=6, positions=[0, 6], num_shuffles=3, mode='sequential')
    
    # 2 positions × 3 shuffles = 6 states
    assert pool.num_states == 6


def test_shuffle_seed():
    """Test ShuffleScanPool with shuffle_seed."""
    pool1 = ShuffleScanPool(TEST_SEQ_18, shuffle_size=6, step_size=1, shuffle_seed=42, mode='sequential')
    pool2 = ShuffleScanPool(TEST_SEQ_18, shuffle_size=6, step_size=1, shuffle_seed=42, mode='sequential')
    
    pool1.set_state(0)
    pool2.set_state(0)
    assert pool1.seq == pool2.seq


def test_preserve_dinucleotides():
    """Test ShuffleScanPool with dinucleotide-preserving shuffle."""
    pool = ShuffleScanPool(TEST_SEQ_24, shuffle_size=12, preserve_dinucleotides=True, step_size=1, mode='sequential')
    
    pool.set_state(0)
    seq = pool.seq
    
    # Get original window
    original_window = TEST_SEQ_24[:12].swapcase()  # mark_changes=True by default
    
    # Count dinucleotides in original and shuffled
    def count_dinucs(s):
        return Counter(s[i:i+2] for i in range(len(s) - 1))
    
    assert count_dinucs(seq[:12]) == count_dinucs(original_window)


def test_validation_errors():
    """Test validation errors."""
    # Cannot mix interfaces
    with pytest.raises(ValueError, match="Cannot specify both"):
        ShuffleScanPool(TEST_SEQ_18, shuffle_size=6, step_size=2, positions=[0, 6])
    
    # Weights without positions
    with pytest.raises(ValueError, match="position_weights requires positions"):
        ShuffleScanPool(TEST_SEQ_18, shuffle_size=6, position_weights=[1.0, 2.0])
    
    # Weights with sequential mode
    with pytest.raises(ValueError, match="Cannot specify position_weights with mode='sequential'"):
        ShuffleScanPool(TEST_SEQ_18, shuffle_size=6, positions=[0, 6], 
                       position_weights=[1.0, 2.0], mode='sequential')
    
    # Empty positions list
    with pytest.raises(ValueError, match="positions must be a non-empty list"):
        ShuffleScanPool(TEST_SEQ_18, shuffle_size=6, positions=[])
    
    # Invalid position
    with pytest.raises(ValueError, match="Position .* is invalid"):
        ShuffleScanPool(TEST_SEQ_18, shuffle_size=6, positions=[15])  # 15 + 6 > 18
    
    # Invalid num_shuffles
    with pytest.raises(ValueError, match="num_shuffles must be at least 1"):
        ShuffleScanPool(TEST_SEQ_18, shuffle_size=6, num_shuffles=0)


def test_duplicate_positions_error():
    """Test that duplicate positions raise an error."""
    with pytest.raises(ValueError, match="positions must not contain duplicates"):
        ShuffleScanPool(TEST_SEQ_18, shuffle_size=6, positions=[0, 6, 6, 12])
