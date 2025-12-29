"""Tests for the SubseqPool class."""

import pytest
from poolparty import SubseqPool, Pool


# Realistic DNA test sequences (>= 15nt)
TEST_SEQ_24 = "ACGTACGTACGTACGTACGTACGT"  # 24nt
TEST_SEQ_18 = "ACGTACGTACGTACGTAC"  # 18nt
TEST_SEQ_20 = "ACGTACGTACGTACGTACGT"  # 20nt


def test_subseq_pool_creation():
    """Test SubseqPool creation with range-based interface."""
    pool = SubseqPool(TEST_SEQ_24, width=6, step_size=2)
    assert pool.is_sequential_compatible()
    assert pool.num_states > 0


def test_subseq_pool_num_states_calculation_range():
    """Test SubseqPool num_states calculation with range-based interface."""
    # seq length = 24, width = 6, step_size = 1
    # positions = range(0, 24 - 6 + 1, 1) = range(0, 19) = 19 states
    pool = SubseqPool(TEST_SEQ_24, width=6, step_size=1)
    assert pool.num_states == 19
    
    # seq length = 24, width = 6, step_size = 2
    # positions = range(0, 19, 2) = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18] = 10 states
    pool = SubseqPool(TEST_SEQ_24, width=6, step_size=2)
    assert pool.num_states == 10
    
    # With start=2, end=20
    # positions = range(2, 20 - 6 + 1, 1) = range(2, 15) = 13 states
    pool = SubseqPool(TEST_SEQ_24, width=6, start=2, end=20, step_size=1)
    assert pool.num_states == 13


def test_subseq_pool_position_based():
    """Test SubseqPool with explicit positions."""
    pool = SubseqPool(TEST_SEQ_24, width=6, positions=[0, 6, 12], mode='sequential')
    assert pool.num_states == 3
    
    pool.set_state(0)
    assert pool.seq == TEST_SEQ_24[0:6]  # "ACGTAC"
    
    pool.set_state(1)
    assert pool.seq == TEST_SEQ_24[6:12]  # "GTACGT"
    
    pool.set_state(2)
    assert pool.seq == TEST_SEQ_24[12:18]  # "ACGTAC"


def test_subseq_pool_position_weights():
    """Test SubseqPool with position weights."""
    pool = SubseqPool(
        TEST_SEQ_24, 
        width=6, 
        positions=[0, 6, 12],
        position_weights=[1.0, 2.0, 1.0],
        mode='random'
    )
    assert pool.num_states == 3
    
    # Verify weighted sampling (statistical test)
    counts = {0: 0, 1: 0, 2: 0}
    for state in range(300):
        pool.set_state(state)
        seq = pool.seq
        if seq == TEST_SEQ_24[0:6]:
            counts[0] += 1
        elif seq == TEST_SEQ_24[6:12]:
            counts[1] += 1
        else:
            counts[2] += 1
    
    # Position 1 (weight 2.0) should appear more often
    assert counts[1] > counts[0]
    assert counts[1] > counts[2]


def test_subseq_pool_basic_extraction():
    """Test basic subsequence extraction."""
    pool = SubseqPool(TEST_SEQ_24, width=6, step_size=1, mode='sequential')
    
    pool.set_state(0)
    assert pool.seq == "ACGTAC"  # positions 0-5
    
    pool.set_state(1)
    assert pool.seq == "CGTACG"  # positions 1-6
    
    pool.set_state(2)
    assert pool.seq == "GTACGT"  # positions 2-7


def test_subseq_pool_with_step_size():
    """Test SubseqPool with larger step_size (skip positions)."""
    pool = SubseqPool(TEST_SEQ_24, width=6, step_size=3, mode='sequential')
    
    pool.set_state(0)
    assert pool.seq == "ACGTAC"  # Position 0
    
    pool.set_state(1)
    assert pool.seq == "TACGTA"  # Position 3
    
    pool.set_state(2)
    assert pool.seq == "GTACGT"  # Position 6


def test_subseq_pool_with_start_end():
    """Test SubseqPool with start and end parameters."""
    # Start at position 3, end before position 18
    pool = SubseqPool(TEST_SEQ_24, width=6, start=3, end=18, step_size=1, mode='sequential')
    
    pool.set_state(0)
    assert pool.seq == TEST_SEQ_24[3:9]  # "TACGTA"
    
    # Should not be able to get position 0 or positions near the end
    all_seqs = []
    for state in range(pool.num_states):
        pool.set_state(state)
        all_seqs.append(pool.seq)
    
    # First seq should not be from position 0
    assert all_seqs[0] == TEST_SEQ_24[3:9]


def test_subseq_pool_sequential_iteration():
    """Test SubseqPool sequential iteration through all states."""
    pool = SubseqPool(TEST_SEQ_18, width=6, step_size=2, mode='sequential')
    
    sequences = []
    for state in range(pool.num_states):
        pool.set_state(state)
        sequences.append(pool.seq)
    
    # Expected positions: 0, 2, 4, 6 (max is 18-6=12, so range(0, 13, 2))
    assert sequences[0] == TEST_SEQ_18[0:6]
    assert sequences[1] == TEST_SEQ_18[2:8]
    assert sequences[2] == TEST_SEQ_18[4:10]


def test_subseq_pool_state_wrapping():
    """Test SubseqPool wraps state with modulo."""
    pool = SubseqPool(TEST_SEQ_18, width=6, step_size=1, mode='sequential')
    
    pool.set_state(0)
    first_seq = pool.seq
    
    # Setting state beyond num_states should wrap
    pool.set_state(pool.num_states)
    assert pool.seq == first_seq


def test_subseq_pool_with_pool_input():
    """Test SubseqPool with Pool object as input."""
    base_pool = Pool(seqs=[TEST_SEQ_24])
    pool = SubseqPool(base_pool, width=6, step_size=2, mode='sequential')
    
    pool.set_state(0)
    assert pool.seq == TEST_SEQ_24[0:6]
    
    pool.set_state(1)
    assert pool.seq == TEST_SEQ_24[2:8]


def test_subseq_pool_concatenation():
    """Test SubseqPool concatenation."""
    pool1 = SubseqPool(TEST_SEQ_18, width=6, step_size=3)
    pool2 = SubseqPool(TEST_SEQ_18, width=6, step_size=3)
    
    combined = pool1 + pool2
    
    # Should have pool1.num_states * pool2.num_states states
    assert combined.num_states == pool1.num_states * pool2.num_states
    assert combined.is_sequential_compatible()


def test_subseq_pool_repr_range_based():
    """Test SubseqPool __repr__ for range-based interface."""
    pool = SubseqPool(TEST_SEQ_18, width=6, start=2, end=15, step_size=2)
    repr_str = repr(pool)
    assert "SubseqPool" in repr_str
    assert "W=6" in repr_str


def test_subseq_pool_repr_position_based():
    """Test SubseqPool __repr__ for position-based interface."""
    pool = SubseqPool(TEST_SEQ_18, width=6, positions=[0, 3, 6])
    repr_str = repr(pool)
    assert "SubseqPool" in repr_str
    assert "positions=" in repr_str


def test_subseq_pool_deterministic():
    """Test that SubseqPool is deterministic with same state."""
    pool = SubseqPool(TEST_SEQ_24, width=6, step_size=2)
    
    pool.set_state(3)
    seq1 = pool.seq
    pool.set_state(3)
    seq2 = pool.seq
    assert seq1 == seq2


def test_subseq_pool_all_subseqs_valid():
    """Test that all subsequences have correct width."""
    pool = SubseqPool(TEST_SEQ_24, width=8, step_size=2, mode='sequential')
    
    for state in range(pool.num_states):
        pool.set_state(state)
        assert len(pool.seq) == 8


def test_subseq_pool_sliding_window():
    """Test SubseqPool as a sliding window (step_size=1)."""
    seq = "ABCDEFGHIJKLMNOP"  # 16 characters
    pool = SubseqPool(seq, width=6, step_size=1, mode='sequential')
    
    # Should get all possible 6-mers with step=1
    # positions = range(0, 16-6+1, 1) = range(0, 11) = 11 states
    assert pool.num_states == 11
    
    expected = ["ABCDEF", "BCDEFG", "CDEFGH", "DEFGHI", "EFGHIJ", 
                "FGHIJK", "GHIJKL", "HIJKLM", "IJKLMN", "JKLMNO", "KLMNOP"]
    for state, expected_seq in enumerate(expected):
        pool.set_state(state)
        assert pool.seq == expected_seq


def test_subseq_pool_non_overlapping():
    """Test SubseqPool with non-overlapping tiles (step_size=width)."""
    seq = "ABCDEFGHIJKLMNOP"  # 16 characters
    pool = SubseqPool(seq, width=4, step_size=4, mode='sequential')
    
    # positions = range(0, 16-4+1, 4) = range(0, 13, 4) = [0, 4, 8, 12] = 4 states
    assert pool.num_states == 4
    
    pool.set_state(0)
    assert pool.seq == "ABCD"
    
    pool.set_state(1)
    assert pool.seq == "EFGH"
    
    pool.set_state(2)
    assert pool.seq == "IJKL"
    
    pool.set_state(3)
    assert pool.seq == "MNOP"


def test_subseq_pool_edge_case_width_equals_length():
    """Test SubseqPool when width equals sequence length."""
    pool = SubseqPool(TEST_SEQ_18, width=18, step_size=1)
    
    # Only 1 position possible
    assert pool.num_states == 1
    
    pool.set_state(0)
    assert pool.seq == TEST_SEQ_18


def test_subseq_pool_composition_with_other_pools():
    """Test SubseqPool composition with other pool types."""
    subseq_pool = SubseqPool(TEST_SEQ_24, width=8, step_size=2, mode='sequential')
    constant = Pool(seqs=["---"])
    
    combined = constant + subseq_pool + constant
    
    combined.set_state(0)
    assert combined.seq.startswith("---")
    assert combined.seq.endswith("---")
    assert len(combined.seq) == 14  # 3 + 8 + 3


def test_subseq_pool_validation_errors():
    """Test SubseqPool validation errors."""
    # Cannot mix range and position interfaces
    with pytest.raises(ValueError, match="Cannot specify both"):
        SubseqPool(TEST_SEQ_18, width=6, step_size=2, positions=[0, 3])
    
    # Weights without positions
    with pytest.raises(ValueError, match="position_weights requires positions"):
        SubseqPool(TEST_SEQ_18, width=6, step_size=2, position_weights=[1.0, 2.0])
    
    # Weights with sequential mode
    with pytest.raises(ValueError, match="Cannot specify position_weights with mode='sequential'"):
        SubseqPool(TEST_SEQ_18, width=6, positions=[0, 3], 
                   position_weights=[1.0, 2.0], mode='sequential')
    
    # Empty positions list
    with pytest.raises(ValueError, match="positions must be a non-empty list"):
        SubseqPool(TEST_SEQ_18, width=6, positions=[])
    
    # Invalid position
    with pytest.raises(ValueError, match="Position .* is invalid"):
        SubseqPool(TEST_SEQ_18, width=6, positions=[15])  # 15 + 6 > 18
    
    # Width longer than sequence
    with pytest.raises(ValueError, match="width .* cannot be longer than"):
        SubseqPool(TEST_SEQ_18, width=25)


def test_subseq_pool_seq_length_property():
    """Test that seq_length equals width."""
    pool = SubseqPool(TEST_SEQ_24, width=8, step_size=2)
    assert pool.seq_length == 8


def test_subseq_pool_generate_seqs():
    """Test SubseqPool with generate_seqs."""
    pool = SubseqPool(TEST_SEQ_18, width=6, step_size=3, mode='sequential')
    
    # Generate all sequences
    seqs = pool.generate_seqs(num_complete_iterations=1)
    
    # The generate_seqs returns pool.num_states sequences for 1 complete iteration
    assert len(seqs) == pool.num_states
    
    # All should be length 6
    assert all(len(s) == 6 for s in seqs)


def test_subseq_pool_duplicate_positions_error():
    """Test that duplicate positions raise an error."""
    with pytest.raises(ValueError, match="positions must not contain duplicates"):
        SubseqPool(TEST_SEQ_18, width=6, positions=[0, 3, 3, 6])
