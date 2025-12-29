"""Tests for the DeletionScanPool class."""

import pytest
from poolparty import DeletionScanPool, Pool


# Realistic DNA test sequences (>= 15nt)
TEST_SEQ_24 = "ACGTACGTACGTACGTACGTACGT"  # 24nt
TEST_SEQ_18 = "ACGTACGTACGTACGTAC"  # 18nt
TEST_SEQ_20 = "ACGTACGTACGTACGTACGT"  # 20nt


def test_deletion_scan_pool_creation():
    """Test DeletionScanPool creation."""
    pool = DeletionScanPool(TEST_SEQ_24, deletion_size=6)
    assert pool.is_sequential_compatible()
    assert pool.num_states > 0


def test_num_states_calculation():
    """Test num_states calculation with range-based interface."""
    # seq length = 24, deletion_size = 6, step_size = 1
    # positions = range(0, 24 - 6 + 1, 1) = range(0, 19) = 19 states
    pool = DeletionScanPool(TEST_SEQ_24, deletion_size=6, step_size=1)
    assert pool.num_states == 19
    
    # With step_size = 2
    # positions = range(0, 19, 2) = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18] = 10 states
    pool = DeletionScanPool(TEST_SEQ_24, deletion_size=6, step_size=2)
    assert pool.num_states == 10
    
    # With start = 3
    # positions = range(3, 19, 1) = 16 states
    pool = DeletionScanPool(TEST_SEQ_24, deletion_size=6, start=3, step_size=1)
    assert pool.num_states == 16


def test_marked_deletion_basic_operation():
    """Test basic marked deletion operation (mark_changes=True)."""
    # Use mode='sequential' for deterministic state-to-position mapping
    pool = DeletionScanPool(TEST_SEQ_24, deletion_size=6, mark_changes=True, step_size=1, mode='sequential')
    
    # TEST_SEQ_24 = "ACGTACGTACGTACGTACGTACGT"
    pool.set_state(0)
    assert pool.seq == "------GTACGTACGTACGTACGT"  # Deletes positions 0-5
    
    pool.set_state(1)
    assert pool.seq == "A------TACGTACGTACGTACGT"  # Deletes positions 1-6
    
    pool.set_state(2)
    assert pool.seq == "AC------ACGTACGTACGTACGT"  # Deletes positions 2-7


def test_unmarked_deletion_basic_operation():
    """Test basic unmarked deletion operation (mark_changes=False)."""
    pool = DeletionScanPool(TEST_SEQ_24, deletion_size=6, mark_changes=False, step_size=1, mode='sequential')
    
    pool.set_state(0)
    assert pool.seq == TEST_SEQ_24[6:]  # Removes first 6 chars
    assert len(pool.seq) == 18
    
    pool.set_state(1)
    expected = TEST_SEQ_24[0:1] + TEST_SEQ_24[7:]  # "A" + positions 7-23
    assert pool.seq == expected


def test_marked_deletion_with_step_size():
    """Test marked deletion with larger step_size."""
    pool = DeletionScanPool(TEST_SEQ_24, deletion_size=6, mark_changes=True, step_size=6, mode='sequential')
    
    # positions = [0, 6, 12, 18]
    pool.set_state(0)
    assert pool.seq == "------GTACGTACGTACGTACGT"  # Position 0
    
    pool.set_state(1)
    assert pool.seq == "ACGTAC------ACGTACGTACGT"  # Position 6
    
    pool.set_state(2)
    assert pool.seq == "ACGTACGTACGT------GTACGT"  # Position 12


def test_unmarked_deletion_with_step_size():
    """Test unmarked deletion with larger step_size."""
    pool = DeletionScanPool(TEST_SEQ_24, deletion_size=6, mark_changes=False, step_size=6, mode='sequential')
    
    pool.set_state(0)
    assert pool.seq == TEST_SEQ_24[6:]  # Remove first 6
    
    pool.set_state(1)
    assert pool.seq == TEST_SEQ_24[:6] + TEST_SEQ_24[12:]  # Remove 6-11


def test_marked_deletion_with_start():
    """Test marked deletion with non-zero start."""
    pool = DeletionScanPool(TEST_SEQ_24, deletion_size=6, mark_changes=True, start=3, step_size=2, mode='sequential')
    
    # positions = [3, 5, 7, 9, 11, 13, 15, 17]
    pool.set_state(0)
    assert pool.seq == "ACG------CGTACGTACGTACGT"  # Position 3
    
    pool.set_state(1)
    assert pool.seq == "ACGTA------TACGTACGTACGT"  # Position 5


def test_unmarked_deletion_with_start():
    """Test unmarked deletion with non-zero start."""
    pool = DeletionScanPool(TEST_SEQ_24, deletion_size=6, mark_changes=False, start=3, step_size=2, mode='sequential')
    
    pool.set_state(0)
    expected = TEST_SEQ_24[:3] + TEST_SEQ_24[9:]  # Remove positions 3-8
    assert pool.seq == expected


def test_custom_deletion_character():
    """Test marked deletion with custom deletion character."""
    pool = DeletionScanPool(TEST_SEQ_18, deletion_size=4, mark_changes=True, 
                           deletion_character='X', step_size=1, mode='sequential')
    
    pool.set_state(0)
    assert pool.seq == "XXXX" + TEST_SEQ_18[4:]
    
    pool.set_state(1)
    assert pool.seq == TEST_SEQ_18[0:1] + "XXXX" + TEST_SEQ_18[5:]


def test_with_pool_background():
    """Test DeletionScanPool with Pool object as background."""
    background_pool = Pool(seqs=[TEST_SEQ_18])
    pool = DeletionScanPool(background_pool, deletion_size=6, mark_changes=True, step_size=1, mode='sequential')
    
    # TEST_SEQ_18 = "ACGTACGTACGTACGTAC"
    pool.set_state(0)
    assert pool.seq == "------GTACGTACGTAC"
    
    pool.set_state(1)
    assert pool.seq == "A------TACGTACGTAC"


def test_sequence_length_marked_mode():
    """Test that marked mode maintains background length."""
    pool = DeletionScanPool(TEST_SEQ_24, deletion_size=6, mark_changes=True, step_size=1, mode='sequential')
    
    assert pool.seq_length == len(TEST_SEQ_24)
    
    for state in range(pool.num_states):
        pool.set_state(state)
        assert len(pool.seq) == len(TEST_SEQ_24)


def test_sequence_length_unmarked_mode():
    """Test that unmarked mode decreases length by deletion size."""
    pool = DeletionScanPool(TEST_SEQ_24, deletion_size=6, mark_changes=False, step_size=1, mode='sequential')
    
    expected_length = len(TEST_SEQ_24) - 6
    assert pool.seq_length == expected_length
    
    for state in range(min(pool.num_states, 10)):
        pool.set_state(state)
        assert len(pool.seq) == expected_length


def test_state_wrapping():
    """Test that state wraps with modulo."""
    pool = DeletionScanPool(TEST_SEQ_18, deletion_size=4, mark_changes=True, step_size=1, mode='sequential')
    
    pool.set_state(0)
    first_seq = pool.seq
    
    pool.set_state(pool.num_states)
    assert pool.seq == first_seq


def test_sequential_iteration():
    """Test DeletionScanPool sequential iteration."""
    pool = DeletionScanPool(TEST_SEQ_18, deletion_size=4, mark_changes=True, step_size=2, mode='sequential')
    
    sequences = []
    for state in range(pool.num_states):
        pool.set_state(state)
        sequences.append(pool.seq)
    
    # All sequences should be different
    assert len(sequences) == len(set(sequences))


def test_concatenation():
    """Test DeletionScanPool concatenation."""
    pool1 = DeletionScanPool(TEST_SEQ_18, deletion_size=4, mark_changes=True, step_size=2, mode='sequential')
    constant = Pool(seqs=["++"])
    
    combined = constant + pool1 + constant
    
    combined.set_state(0)
    assert combined.seq.startswith("++")
    assert combined.seq.endswith("++")


def test_repr_marked_mode():
    """Test DeletionScanPool __repr__ for marked mode."""
    pool = DeletionScanPool(TEST_SEQ_18, deletion_size=4, mark_changes=True, step_size=1)
    repr_str = repr(pool)
    assert "DeletionScanPool" in repr_str
    assert "del_size=4" in repr_str


def test_repr_position_based():
    """Test DeletionScanPool __repr__ for position-based interface."""
    pool = DeletionScanPool(TEST_SEQ_18, deletion_size=4, positions=[0, 4, 8])
    repr_str = repr(pool)
    assert "DeletionScanPool" in repr_str
    assert "positions=" in repr_str


def test_deterministic_with_same_state():
    """Test that DeletionScanPool is deterministic with same state."""
    pool = DeletionScanPool(TEST_SEQ_24, deletion_size=6, mark_changes=True, step_size=1)
    
    pool.set_state(5)
    seq1 = pool.seq
    pool.set_state(5)
    seq2 = pool.seq
    assert seq1 == seq2


def test_edge_case_short_background():
    """Test DeletionScanPool with short background."""
    seq = "ACGTACGTACGTACGT"  # 16 chars
    pool = DeletionScanPool(seq, deletion_size=4, mark_changes=True, step_size=1, mode='sequential')
    
    assert pool.num_states > 0
    pool.set_state(0)
    assert pool.seq == "----ACGTACGTACGT"


def test_edge_case_large_deletion():
    """Test DeletionScanPool with deletion almost as long as background."""
    pool = DeletionScanPool(TEST_SEQ_18, deletion_size=12, mark_changes=True, step_size=1, mode='sequential')
    
    assert pool.num_states > 0
    pool.set_state(0)
    assert len(pool.seq) == len(TEST_SEQ_18)


def test_single_position_deletion():
    """Test DeletionScanPool with single position deletion."""
    pool = DeletionScanPool(TEST_SEQ_18, deletion_size=1, mark_changes=True, step_size=1, mode='sequential')
    
    # TEST_SEQ_18 = "ACGTACGTACGTACGTAC"
    pool.set_state(0)
    assert pool.seq == "-CGTACGTACGTACGTAC"
    
    pool.set_state(1)
    assert pool.seq == "A-GTACGTACGTACGTAC"


def test_different_step_values():
    """Test DeletionScanPool with various step_size values."""
    # Test step_size=1 (sliding window)
    pool1 = DeletionScanPool(TEST_SEQ_24, deletion_size=6, mark_changes=True, step_size=1)
    assert pool1.num_states == 19
    
    # Test step_size=6 (non-overlapping)
    pool2 = DeletionScanPool(TEST_SEQ_24, deletion_size=6, mark_changes=True, step_size=6)
    assert pool2.num_states == 4


def test_mode_parameter():
    """Test that mode parameter is accepted."""
    pool_random = DeletionScanPool(TEST_SEQ_18, deletion_size=4, mode='random')
    assert pool_random.mode == 'random'
    
    pool_sequential = DeletionScanPool(TEST_SEQ_18, deletion_size=4, mode='sequential')
    assert pool_sequential.mode == 'sequential'


def test_deletion_longer_than_background_error():
    """Test that deletion_size longer than background raises error."""
    with pytest.raises(ValueError, match="cannot be longer than"):
        DeletionScanPool(TEST_SEQ_18, deletion_size=25)


def test_marked_vs_unmarked_length_consistency():
    """Test that marked and unmarked modes have consistent behavior."""
    pool_marked = DeletionScanPool(TEST_SEQ_24, deletion_size=6, mark_changes=True, step_size=1)
    pool_unmarked = DeletionScanPool(TEST_SEQ_24, deletion_size=6, mark_changes=False, step_size=1)
    
    # Should have same number of states
    assert pool_marked.num_states == pool_unmarked.num_states
    
    # Check length consistency
    assert pool_marked.seq_length == len(TEST_SEQ_24)
    assert pool_unmarked.seq_length == len(TEST_SEQ_24) - 6


def test_deletion_at_end_of_sequence():
    """Test deletion at the end of the sequence."""
    # Use step_size=1 to ensure we can delete at the very end
    pool = DeletionScanPool(TEST_SEQ_18, deletion_size=4, mark_changes=True, step_size=1, mode='sequential')
    
    # Last state should delete at the end (position 14 for 18-char seq with deletion_size=4)
    pool.set_state(pool.num_states - 1)
    assert pool.seq.endswith("----")


def test_multiple_deletions_do_not_overlap():
    """Test that different states produce distinct sequences."""
    pool = DeletionScanPool(TEST_SEQ_24, deletion_size=6, mark_changes=True, step_size=1, mode='sequential')
    
    sequences = set()
    for state in range(min(pool.num_states, 10)):
        pool.set_state(state)
        sequences.add(pool.seq)
    
    # All sequences should be unique
    assert len(sequences) == min(pool.num_states, 10)


def test_dynamic_pool_parent():
    """Test DeletionScanPool with dynamic Pool parent that updates."""
    background_pool = Pool(seqs=["GGGGGGGGGGGGGGGG"])  # 16 G's
    
    pool = DeletionScanPool(background_pool, deletion_size=4, mark_changes=True, step_size=1, mode='sequential')
    
    pool.set_state(0)
    assert pool.seq == "----GGGGGGGGGGGG"
    
    pool.set_state(1)
    assert pool.seq == "G----GGGGGGGGGGG"
    
    pool.set_state(2)
    assert pool.seq == "GG----GGGGGGGGGG"


def test_position_based_interface():
    """Test DeletionScanPool with explicit positions."""
    pool = DeletionScanPool(TEST_SEQ_24, deletion_size=6, positions=[0, 6, 12], mode='sequential')
    
    assert pool.num_states == 3
    
    # TEST_SEQ_24 = "ACGTACGTACGTACGTACGTACGT"
    pool.set_state(0)
    assert pool.seq == "------GTACGTACGTACGTACGT"  # Position 0
    
    pool.set_state(1)
    assert pool.seq == "ACGTAC------ACGTACGTACGT"  # Position 6
    
    pool.set_state(2)
    assert pool.seq == "ACGTACGTACGT------GTACGT"  # Position 12


def test_position_weights():
    """Test DeletionScanPool with position weights."""
    pool = DeletionScanPool(
        TEST_SEQ_24, 
        deletion_size=6, 
        positions=[0, 6, 12],
        position_weights=[1.0, 2.0, 1.0],
        mode='random'
    )
    
    # Verify weighted sampling works
    assert pool.num_states == 3


def test_validation_errors():
    """Test validation errors."""
    # Cannot mix interfaces
    with pytest.raises(ValueError, match="Cannot specify both"):
        DeletionScanPool(TEST_SEQ_18, deletion_size=4, step_size=2, positions=[0, 4])
    
    # Weights without positions
    with pytest.raises(ValueError, match="position_weights requires positions"):
        DeletionScanPool(TEST_SEQ_18, deletion_size=4, position_weights=[1.0, 2.0])
    
    # Weights with sequential mode
    with pytest.raises(ValueError, match="Cannot specify position_weights with mode='sequential'"):
        DeletionScanPool(TEST_SEQ_18, deletion_size=4, positions=[0, 4], 
                         position_weights=[1.0, 2.0], mode='sequential')
    
    # Empty positions list
    with pytest.raises(ValueError, match="positions must be a non-empty list"):
        DeletionScanPool(TEST_SEQ_18, deletion_size=4, positions=[])
    
    # Invalid position
    with pytest.raises(ValueError, match="Position .* is invalid"):
        DeletionScanPool(TEST_SEQ_18, deletion_size=6, positions=[15])  # 15 + 6 > 18


def test_sequential_mode_with_sequential_pool_ancestor():
    """Test sequential DeletionScanPool with sequential Pool ancestor."""
    background = Pool(seqs=['AAAAAAAAAAAAAAAA', 'BBBBBBBBBBBBBBBB', 'CCCCCCCCCCCCCCCC'], 
                      name='background', mode='sequential')
    
    pool = DeletionScanPool(
        background_seq=background,
        deletion_size=3,
        mode='sequential',
        start=0,
        step_size=6
    )
    
    # Verify pool properties
    assert pool.num_internal_states == 3  # 3 deletion positions (0, 6, 12)
    assert background.num_internal_states == 3
    assert pool.num_states == 9  # 3 × 3
    
    seqs = pool.generate_seqs(num_complete_iterations=1)
    
    assert len(seqs) == 9
    assert len(set(seqs)) == 9


def test_duplicate_positions_error():
    """Test that duplicate positions raise an error."""
    with pytest.raises(ValueError, match="positions must not contain duplicates"):
        DeletionScanPool(TEST_SEQ_18, deletion_size=4, positions=[0, 4, 4, 8])
