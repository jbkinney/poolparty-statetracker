"""Tests for the refactored InsertionScanORFPool class.

Note: The following original tests are no longer applicable due to API changes:
- test_orf_not_string_error: New API accepts Pool as seq input
- test_insertion_not_string_error: New API accepts Pool as insertion_seq input
- test_offset_modulo_overwrite_mode: offset parameter replaced by start
- test_repr_with_seed: Seed parameter not supported in new API
- TestSetSeed: Seed parameter not supported in new API
"""

import pytest
from poolparty import InsertionScanORFPool, Pool


def test_insertion_scan_orf_pool_creation():
    """Test InsertionScanORFPool creation."""
    orf = "ATGGCCAAA"  # 3 codons
    insertion = "TTT"  # 1 codon
    pool = InsertionScanORFPool(orf, insertion)
    assert pool.is_sequential_compatible()
    assert pool.num_states > 0


def test_stores_codons():
    """Test that initialization properly splits sequences into codons."""
    orf = "ATGGCCAAA"  # 3 codons
    insertion = "TTTCCC"  # 2 codons
    pool = InsertionScanORFPool(orf, insertion, insert_or_overwrite='overwrite')
    
    assert pool.codons == ['ATG', 'GCC', 'AAA']
    assert pool.insertion_codons == ['TTT', 'CCC']


class TestValidation:
    """Test input validation."""
    
    def test_orf_non_dna_characters_error(self):
        """Test that non-DNA characters in seq raise error."""
        with pytest.raises(ValueError, match="must contain only ACGT"):
            InsertionScanORFPool('ATGXXX', 'TTT')
    
    def test_orf_length_not_divisible_by_3_error(self):
        """Test that seq length not divisible by 3 raises error."""
        with pytest.raises(ValueError, match="(ORF|orf).*(length|Length).*divisible by 3"):
            InsertionScanORFPool('ATGG', 'TTT')
    
    def test_insertion_non_dna_characters_error(self):
        """Test that non-DNA characters in insert_seq raise error."""
        with pytest.raises(ValueError, match="insert_seq must contain only ACGT"):
            InsertionScanORFPool('ATGGCC', 'XXX')
    
    def test_insertion_length_not_divisible_by_3_error(self):
        """Test that insert_seq length not divisible by 3 raises error."""
        with pytest.raises(ValueError, match="insert_seq length must be divisible by 3"):
            InsertionScanORFPool('ATGGCC', 'TT')
    
    def test_invalid_insert_or_overwrite(self):
        """Test that invalid insert_or_overwrite value raises error."""
        with pytest.raises(ValueError, match="insert_or_overwrite must be"):
            InsertionScanORFPool('ATGGCC', 'TTT', insert_or_overwrite='invalid')
    
    def test_position_weights_without_positions(self):
        """Test that position_weights without positions raises error."""
        with pytest.raises(ValueError, match="position_weights requires positions"):
            InsertionScanORFPool('ATGGCC', 'TTT', position_weights=[1.0, 2.0])
    
    def test_position_weights_with_sequential_mode(self):
        """Test that position_weights with sequential mode raises error."""
        with pytest.raises(ValueError, match="position_weights.*sequential"):
            InsertionScanORFPool(
                'ATGGCC', 'TTT', 
                positions=[0, 1], 
                position_weights=[1.0, 2.0],
                mode='sequential'
            )
    
    def test_non_integer_position(self):
        """Test that non-integer position raises error."""
        with pytest.raises(ValueError, match="positions must be integers"):
            InsertionScanORFPool('ATGGCCAAA', 'TTT', positions=[0, 1.5])
    
    def test_position_weights_length_mismatch(self):
        """Test that position_weights length mismatch raises error."""
        with pytest.raises(ValueError, match="position_weights length"):
            InsertionScanORFPool(
                'ATGGCCAAA', 'TTT',
                positions=[0, 1],
                position_weights=[1.0],
                mode='random'
            )
    
    def test_position_weights_non_positive_sum(self):
        """Test that non-positive sum of weights raises error."""
        with pytest.raises(ValueError, match="Sum of position_weights must be positive"):
            InsertionScanORFPool(
                'ATGGCCAAA', 'TTT',
                positions=[0, 1],
                position_weights=[0.5, -0.5],
                mode='random'
            )
    
    def test_overwrite_window_exceeds_orf(self):
        """Test that overwrite position exceeding ORF raises error."""
        with pytest.raises(ValueError, match="(Position|window).*must fit within ORF"):
            InsertionScanORFPool(
                'ATGATGATGATG',  # 4 codons
                'AAACCCGGG',  # 3 codons
                positions=[2],
                insert_or_overwrite='overwrite'
            )
    
    def test_insert_position_beyond_orf(self):
        """Test that insert position beyond ORF boundary raises error."""
        with pytest.raises(ValueError, match="Position.*invalid.*insert mode"):
            InsertionScanORFPool(
                'ATGATGATGATG',  # 4 codons
                'CCC',
                positions=[5],
                insert_or_overwrite='insert'
            )


class TestOverwriteMode:
    """Test overwrite mode operations at codon level."""
    
    def test_num_states_calculation_basic(self):
        """Test num_states calculation for overwrite mode at codon level."""
        orf = "ATGGCCAAACCCTTTGGG"  # 6 codons
        insertion = "CTGCTG"  # 2 codons
        
        # L=6, W=2, start=0, step=1
        # range(0, 6-2+1, 1) = range(0, 5, 1) = [0, 1, 2, 3, 4] -> 5 states
        pool = InsertionScanORFPool(
            orf, insertion, 
            insert_or_overwrite='overwrite', 
            step_size=1, 
            start=0
        )
        assert pool.num_internal_states == 5
    
    def test_num_states_with_step_size(self):
        """Test num_states calculation with larger step_size."""
        orf = "ATGGCCAAACCCTTTGGG"  # 6 codons
        insertion = "TTT"  # 1 codon
        
        # L=6, W=1, start=0, step=2
        # range(0, 6-1+1, 2) = range(0, 6, 2) = [0, 2, 4] -> 3 states
        pool = InsertionScanORFPool(
            orf, insertion, 
            insert_or_overwrite='overwrite', 
            step_size=2, 
            start=0
        )
        assert pool.num_internal_states == 3
    
    def test_num_states_with_start(self):
        """Test num_states calculation with non-zero start."""
        orf = "ATGGCCAAACCCTTT"  # 5 codons
        insertion = "CTGCTG"  # 2 codons
        
        # L=5, W=2, start=1, step=1
        # range(1, 5-2+1, 1) = range(1, 4, 1) = [1, 2, 3] -> 3 states
        pool = InsertionScanORFPool(
            orf, insertion, 
            insert_or_overwrite='overwrite', 
            step_size=1, 
            start=1
        )
        assert pool.num_internal_states == 3
    
    def test_basic_operation(self):
        """Test basic overwrite operation at codon level."""
        orf = "ATGGCCAAACCCTTTGGG"  # 6 codons: ATG-GCC-AAA-CCC-TTT-GGG
        insertion = "CTGCTG"  # 2 codons: CTG-CTG
        pool = InsertionScanORFPool(
            orf, insertion, 
            insert_or_overwrite='overwrite', 
            step_size=1, 
            start=0,
            mode='sequential'
        )
        
        pool.set_state(0)
        # Overwrites codons 0-1 (ATG-GCC -> CTG-CTG)
        assert pool.seq == "CTGCTGAAACCCTTTGGG"
        
        pool.set_state(1)
        # Overwrites codons 1-2 (GCC-AAA -> CTG-CTG)
        assert pool.seq == "ATGCTGCTGCCCTTTGGG"
        
        pool.set_state(2)
        # Overwrites codons 2-3 (AAA-CCC -> CTG-CTG)
        assert pool.seq == "ATGGCCCTGCTGTTTGGG"
    
    def test_with_step_size(self):
        """Test overwrite mode with larger step_size."""
        orf = "ATGGCCAAACCCTTTGGG"  # 6 codons
        insertion = "CTGCTG"  # 2 codons
        pool = InsertionScanORFPool(
            orf, insertion, 
            insert_or_overwrite='overwrite', 
            step_size=2, 
            start=0,
            mode='sequential'
        )
        
        pool.set_state(0)
        # Position 0 (codons 0-1)
        assert pool.seq == "CTGCTGAAACCCTTTGGG"
        
        pool.set_state(1)
        # Position 2 (codons 2-3)
        assert pool.seq == "ATGGCCCTGCTGTTTGGG"
        
        pool.set_state(2)
        # Position 4 (codons 4-5)
        assert pool.seq == "ATGGCCAAACCCCTGCTG"
    
    def test_with_start(self):
        """Test overwrite mode with non-zero start."""
        orf = "ATGGCCAAACCCTTTGGG"  # 6 codons
        insertion = "CTGCTG"  # 2 codons
        pool = InsertionScanORFPool(
            orf, insertion, 
            insert_or_overwrite='overwrite', 
            step_size=2, 
            start=1,
            mode='sequential'
        )
        
        # start=1, step=2, positions = [1, 3]
        pool.set_state(0)
        assert pool.seq == "ATGCTGCTGCCCTTTGGG"  # Position 1 (codons 1-2)
        
        pool.set_state(1)
        assert pool.seq == "ATGGCCAAACTGCTGGGG"  # Position 3 (codons 3-4)
    
    def test_single_codon_insertion(self):
        """Test overwrite mode with single codon insertion."""
        orf = "ATGGCCAAACCC"  # 4 codons
        insertion = "TTT"  # 1 codon
        pool = InsertionScanORFPool(
            orf, insertion, 
            insert_or_overwrite='overwrite', 
            step_size=1, 
            start=0,
            mode='sequential'
        )
        
        pool.set_state(0)
        assert pool.seq == "TTTGCCAAACCC"  # Replaces codon 0 (ATG->TTT)
        
        pool.set_state(1)
        assert pool.seq == "ATGTTTAAACCC"  # Replaces codon 1 (GCC->TTT)
        
        pool.set_state(2)
        assert pool.seq == "ATGGCCTTTCCC"  # Replaces codon 2 (AAA->TTT)


class TestInsertMode:
    """Test insert mode operations at codon level."""
    
    def test_num_states_calculation_basic(self):
        """Test num_states calculation for insert mode at codon level."""
        orf = "ATGGCCAAA"  # 3 codons
        insertion = "TTT"  # 1 codon
        
        # L=3, start=0, step=1
        # range(0, 3+1, 1) = [0, 1, 2, 3] -> 4 states
        pool = InsertionScanORFPool(
            orf, insertion, 
            insert_or_overwrite='insert', 
            step_size=1, 
            start=0
        )
        assert pool.num_internal_states == 4
    
    def test_num_states_with_step_size(self):
        """Test num_states calculation with larger step_size in insert mode."""
        orf = "ATGGCCAAA"  # 3 codons
        insertion = "TTT"  # 1 codon
        
        # L=3, start=0, step=2
        # range(0, 3+1, 2) = [0, 2] -> 2 states
        pool = InsertionScanORFPool(
            orf, insertion, 
            insert_or_overwrite='insert', 
            step_size=2, 
            start=0
        )
        assert pool.num_internal_states == 2
    
    def test_num_states_with_start(self):
        """Test num_states calculation with start in insert mode."""
        orf = "ATGGCCAAA"  # 3 codons
        insertion = "TTT"  # 1 codon
        
        # L=3, start=1, step=1
        # range(1, 3+1, 1) = [1, 2, 3] -> 3 states
        pool = InsertionScanORFPool(
            orf, insertion, 
            insert_or_overwrite='insert', 
            step_size=1, 
            start=1
        )
        assert pool.num_internal_states == 3
    
    def test_basic_operation(self):
        """Test basic insert operation at codon level."""
        orf = "ATGGCCAAA"  # 3 codons
        insertion = "TTT"  # 1 codon
        pool = InsertionScanORFPool(
            orf, insertion, 
            insert_or_overwrite='insert', 
            step_size=1, 
            start=0,
            mode='sequential'
        )
        
        pool.set_state(0)
        assert pool.seq == "TTTATGGCCAAA"  # Inserts at codon position 0
        
        pool.set_state(1)
        assert pool.seq == "ATGTTTGCCAAA"  # Inserts at codon position 1
        
        pool.set_state(2)
        assert pool.seq == "ATGGCCTTTAAA"  # Inserts at codon position 2
        
        pool.set_state(3)
        assert pool.seq == "ATGGCCAAATTT"  # Inserts at codon position 3 (end)
    
    def test_with_step_size(self):
        """Test insert mode with larger step_size."""
        orf = "ATGGCCAAACCC"  # 4 codons
        insertion = "TTT"  # 1 codon
        pool = InsertionScanORFPool(
            orf, insertion, 
            insert_or_overwrite='insert', 
            step_size=2, 
            start=0,
            mode='sequential'
        )
        
        pool.set_state(0)
        assert pool.seq == "TTTATGGCCAAACCC"  # Position 0
        
        pool.set_state(1)
        assert pool.seq == "ATGGCCTTTAAACCC"  # Position 2
        
        pool.set_state(2)
        assert pool.seq == "ATGGCCAAACCCTTT"  # Position 4 (end)
    
    def test_with_start(self):
        """Test insert mode with non-zero start."""
        orf = "ATGGCCAAA"  # 3 codons
        insertion = "TTT"  # 1 codon
        pool = InsertionScanORFPool(
            orf, insertion, 
            insert_or_overwrite='insert', 
            step_size=1, 
            start=1,
            mode='sequential'
        )
        
        pool.set_state(0)
        assert pool.seq == "ATGTTTGCCAAA"  # Position 1
        
        pool.set_state(1)
        assert pool.seq == "ATGGCCTTTAAA"  # Position 2
        
        pool.set_state(2)
        assert pool.seq == "ATGGCCAAATTT"  # Position 3 (end)
    
    def test_multi_codon_insertion(self):
        """Test insert mode with multi-codon insertion."""
        orf = "ATGGCCAAA"  # 3 codons
        insertion = "TTTCCC"  # 2 codons
        pool = InsertionScanORFPool(
            orf, insertion, 
            insert_or_overwrite='insert', 
            step_size=1, 
            start=0,
            mode='sequential'
        )
        
        pool.set_state(0)
        assert pool.seq == "TTTCCCATGGCCAAA"  # Inserts at position 0
        
        pool.set_state(1)
        assert pool.seq == "ATGTTTCCCGCCAAA"  # Inserts at position 1
        
        pool.set_state(2)
        assert pool.seq == "ATGGCCTTTCCCAAA"  # Inserts at position 2


class TestMarkChanges:
    """Test mark_changes functionality (swapcase)."""
    
    def test_mark_changes_false(self):
        """Test mark_changes=False (default) functionality."""
        orf = "ATGGCCAAA"
        insertion = "ttt"  # lowercase
        pool = InsertionScanORFPool(
            orf, insertion, 
            insert_or_overwrite='overwrite',
            mark_changes=False, 
            step_size=1, 
            start=0,
            mode='sequential'
        )
        
        pool.set_state(0)
        assert pool.seq == "tttGCCAAA"  # 'ttt' stays lowercase
    
    def test_mark_changes_true_lowercase_insertion(self):
        """Test mark_changes=True with lowercase insertion."""
        orf = "ATGGCCAAA"
        insertion = "ttt"  # lowercase
        pool = InsertionScanORFPool(
            orf, insertion, 
            insert_or_overwrite='overwrite',
            mark_changes=True, 
            step_size=1, 
            start=0,
            mode='sequential'
        )
        
        pool.set_state(0)
        assert pool.seq == "TTTGCCAAA"  # 'ttt' becomes 'TTT' via swapcase
        
        pool.set_state(1)
        assert pool.seq == "ATGTTTAAA"  # Still 'TTT'
    
    def test_mark_changes_true_uppercase_insertion(self):
        """Test mark_changes=True with uppercase insertion."""
        orf = "ATGGCCAAA"
        insertion = "GGG"  # uppercase
        pool = InsertionScanORFPool(
            orf, insertion, 
            insert_or_overwrite='overwrite',
            mark_changes=True, 
            step_size=1, 
            start=0,
            mode='sequential'
        )
        
        pool.set_state(0)
        assert pool.seq == "gggGCCAAA"  # 'GGG' becomes 'ggg' via swapcase
    
    def test_mark_changes_multi_codon(self):
        """Test mark_changes with multi-codon insertion."""
        orf = "ATGGCCAAACCC"
        insertion = "TTTGGG"  # 2 codons
        pool = InsertionScanORFPool(
            orf, insertion, 
            insert_or_overwrite='overwrite',
            mark_changes=True, 
            step_size=1, 
            start=0,
            mode='sequential'
        )
        
        pool.set_state(0)
        assert pool.seq == "tttgggAAACCC"  # Both codons swapped
    
    def test_mark_changes_insert_mode(self):
        """Test mark_changes in insert mode."""
        orf = "ATGGCCAAA"
        insertion = "TTT"
        pool = InsertionScanORFPool(
            orf, insertion, 
            insert_or_overwrite='insert',
            mark_changes=True, 
            step_size=1, 
            start=0,
            mode='sequential'
        )
        
        pool.set_state(0)
        assert pool.seq == "tttATGGCCAAA"  # Inserted codon is lowercase


class TestSequenceLength:
    """Test sequence length calculations."""
    
    def test_length_overwrite_mode(self):
        """Test that overwrite mode maintains orf length."""
        orf = "ATGGCCAAACCCTTT"  # 5 codons = 15 nt
        insertion = "CTGCTG"  # 2 codons = 6 nt
        pool = InsertionScanORFPool(
            orf, insertion, 
            insert_or_overwrite='overwrite', 
            step_size=1, 
            start=0,
            mode='sequential'
        )
        
        assert pool.seq_length == 15
        
        for state in range(pool.num_internal_states):
            pool.set_state(state)
            assert len(pool.seq) == 15
    
    def test_length_insert_mode(self):
        """Test that insert mode increases length by insertion size."""
        orf = "ATGGCCAAA"  # 3 codons = 9 nt
        insertion = "TTT"  # 1 codon = 3 nt
        pool = InsertionScanORFPool(
            orf, insertion, 
            insert_or_overwrite='insert', 
            step_size=1, 
            start=0,
            mode='sequential'
        )
        
        expected_length = 12  # 9 + 3
        assert pool.seq_length == expected_length
        
        for state in range(min(pool.num_internal_states, 10)):
            pool.set_state(state)
            assert len(pool.seq) == expected_length


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_single_codon_orf(self):
        """Test with single codon ORF."""
        orf = "ATG"  # 1 codon
        insertion = "TTT"  # 1 codon
        pool = InsertionScanORFPool(
            orf, insertion, 
            insert_or_overwrite='overwrite', 
            step_size=1, 
            start=0,
            mode='sequential'
        )
        
        pool.set_state(0)
        assert pool.seq == "TTT"
        assert len(pool.seq) == 3
    
    def test_insertion_longer_than_orf_error_overwrite(self):
        """Test that insertion longer than orf raises error in overwrite mode."""
        orf = "ATGGCC"  # 2 codons
        insertion = "TTTCCCGGG"  # 3 codons
        
        with pytest.raises(ValueError) as excinfo:
            pool = InsertionScanORFPool(
                orf, insertion, 
                insert_or_overwrite='overwrite'
            )
        
        assert "insertion codon count" in str(excinfo.value).lower() or \
               "cannot exceed" in str(excinfo.value).lower()
    
    def test_insertion_longer_than_orf_allowed_insert_mode(self):
        """Test that insertion longer than orf is allowed in insert mode."""
        orf = "ATG"  # 1 codon
        insertion = "TTTCCCGGG"  # 3 codons
        
        # Should not raise an error
        pool = InsertionScanORFPool(
            orf, insertion, 
            insert_or_overwrite='insert', 
            step_size=1, 
            start=0,
            mode='sequential'
        )
        assert pool.num_internal_states > 0
        
        pool.set_state(0)
        assert pool.seq == "TTTCCCGGGATG"  # Insertion at position 0
    
    def test_state_wrapping(self):
        """Test that state wraps with modulo."""
        orf = "ATGGCCAAA"
        insertion = "TTT"
        pool = InsertionScanORFPool(
            orf, insertion, 
            insert_or_overwrite='overwrite', 
            step_size=1, 
            start=0,
            mode='sequential'
        )
        
        pool.set_state(0)
        first_seq = pool.seq
        
        # Setting state beyond num_internal_states should wrap
        pool.set_state(pool.num_internal_states)
        assert pool.seq == first_seq
    
    def test_full_orf_overwrite(self):
        """Test overwriting entire ORF."""
        orf = "ATGATGATG"  # 3 codons
        insertion = "CCCGGGAAA"  # 3 codons
        pool = InsertionScanORFPool(
            orf, insertion,
            insert_or_overwrite='overwrite',
            mode='sequential'
        )
        assert pool.num_internal_states == 1
        pool.set_state(0)
        assert pool.seq == "CCCGGGAAA"


class TestPositionBasedInterface:
    """Test position-based interface with explicit positions."""
    
    def test_explicit_positions(self):
        """Test with explicit positions list."""
        orf = "ATGGCCAAACCCTTT"  # 5 codons
        insertion = "GGG"  # 1 codon
        pool = InsertionScanORFPool(
            orf, insertion,
            insert_or_overwrite='overwrite',
            positions=[0, 2, 4],
            mode='sequential'
        )
        
        assert pool.num_internal_states == 3
        
        pool.set_state(0)
        assert pool.seq == "GGGGCCAAACCCTTT"  # Position 0
        
        pool.set_state(1)
        assert pool.seq == "ATGGCCGGGCCCTTT"  # Position 2
        
        pool.set_state(2)
        assert pool.seq == "ATGGCCAAACCCGGG"  # Position 4
    
    def test_weighted_positions(self):
        """Test position-based with weights in random mode."""
        orf = "ATGGCCAAACCCTTT"  # 5 codons
        insertion = "GGG"  # 1 codon
        pool = InsertionScanORFPool(
            orf, insertion,
            insert_or_overwrite='overwrite',
            positions=[0, 4],
            position_weights=[9.0, 1.0],
            mode='random'
        )
        
        # Verify weighted sampling produces expected distribution
        counts = {0: 0, 4: 0}
        for state in range(500):
            pool.set_state(state)
            seq = pool.seq
            if seq.startswith("GGG"):
                counts[0] += 1
            else:
                counts[4] += 1
        
        # Position 0 should be selected much more often
        assert counts[0] > counts[4] * 4
    
    def test_weighted_positions_full_verification(self):
        """Test that weighted positions produce correct content AND distribution."""
        orf_seq = "AAACCCGGGTTT"  # 4 codons: AAA CCC GGG TTT
        
        pool = InsertionScanORFPool(
            background_seq=orf_seq,
            insert_seq="ATG",  # Insert ATG
            insert_or_overwrite='insert',
            positions=[0, 4],  # Insert at start or end
            position_weights=[1.0, 9.0],  # 10% start, 90% end
            mode='random'
        )
        
        counts = {"start": 0, "end": 0}
        for state in range(500):
            pool.set_state(state)
            seq = pool.seq
            
            # Verify length is correct (4 + 1 = 5 codons = 15 nt)
            assert len(seq) == 15, f"Wrong length: {len(seq)}"
            
            codons = [seq[j:j+3] for j in range(0, len(seq), 3)]
            
            # Check where ATG was inserted
            if codons[0] == "ATG" and codons[1] == "AAA":
                counts["start"] += 1
                # Verify rest of sequence
                assert codons == ["ATG", "AAA", "CCC", "GGG", "TTT"], f"Start insert wrong: {codons}"
            elif codons[4] == "ATG" and codons[3] == "TTT":
                counts["end"] += 1
                # Verify rest of sequence
                assert codons == ["AAA", "CCC", "GGG", "TTT", "ATG"], f"End insert wrong: {codons}"
            else:
                assert False, f"Unexpected insertion result: {codons}"
        
        # Weight distribution check
        assert counts["end"] > counts["start"] * 4, "Weight distribution not respected"
    
    def test_cannot_mix_interfaces(self):
        """Test that mixing range and position interfaces raises error."""
        with pytest.raises(ValueError):
            InsertionScanORFPool(
                "ATGGCCAAA",
                "TTT",
                start=0,
                positions=[0, 1]
            )
    
    def test_duplicate_positions_error(self):
        """Test that duplicate positions raise error."""
        with pytest.raises(ValueError, match="duplicates"):
            InsertionScanORFPool(
                "ATGGCCAAA",
                "TTT",
                positions=[0, 1, 0]
            )


class TestDeterminism:
    """Test deterministic behavior."""
    
    def test_same_state_same_sequence(self):
        """Test that InsertionScanORFPool is deterministic with same state."""
        orf = "ATGGCCAAACCCTTT"
        insertion = "CTGCTG"
        pool = InsertionScanORFPool(
            orf, insertion, 
            insert_or_overwrite='overwrite', 
            step_size=1, 
            start=0,
            mode='sequential'
        )
        
        pool.set_state(2)
        seq1 = pool.seq
        pool.set_state(2)
        seq2 = pool.seq
        assert seq1 == seq2
    
    def test_different_states_different_sequences(self):
        """Test that different states produce different sequences."""
        orf = "ATGGCCAAACCCTTT"
        insertion = "CTGCTG"
        pool = InsertionScanORFPool(
            orf, insertion, 
            insert_or_overwrite='overwrite', 
            step_size=1, 
            start=0,
            mode='sequential'
        )
        
        sequences = []
        for state in range(pool.num_internal_states):
            pool.set_state(state)
            sequences.append(pool.seq)
        
        # All sequences should be different (for these parameters)
        assert len(sequences) == len(set(sequences))


class TestModes:
    """Test random and sequential modes."""
    
    def test_mode_parameter(self):
        """Test that mode parameter is accepted."""
        orf = "ATGGCCAAA"
        insertion = "TTT"
        
        pool_random = InsertionScanORFPool(orf, insertion, mode='random')
        assert pool_random.mode == 'random'
        
        pool_sequential = InsertionScanORFPool(orf, insertion, mode='sequential')
        assert pool_sequential.mode == 'sequential'
    
    def test_sequential_iteration(self):
        """Test InsertionScanORFPool sequential iteration."""
        orf = "ATGGCCAAACCC"
        insertion = "TTT"
        pool = InsertionScanORFPool(
            orf, insertion, 
            insert_or_overwrite='overwrite', 
            step_size=1, 
            start=0,
            mode='sequential'
        )
        
        sequences = []
        for state in range(pool.num_internal_states):
            pool.set_state(state)
            sequences.append(pool.seq)
        
        # All sequences should be different
        assert len(sequences) == len(set(sequences))
    
    def test_is_sequential_compatible(self):
        """Test that pool is sequential compatible."""
        orf = "ATGGCCAAA"
        insertion = "TTT"
        pool = InsertionScanORFPool(orf, insertion)
        
        assert pool.is_sequential_compatible()


class TestIntegration:
    """Test integration with Pool operations."""
    
    def test_concatenation(self):
        """Test InsertionScanORFPool concatenation."""
        orf = "ATGGCCAAA"
        insertion = "TTT"
        pool1 = InsertionScanORFPool(
            orf, insertion, 
            insert_or_overwrite='overwrite', 
            step_size=1, 
            start=0,
            mode='sequential'
        )
        constant = Pool(seqs=["---"])
        
        combined = constant + pool1 + constant
        
        combined.set_state(0)
        assert combined.seq.startswith("---")
        assert combined.seq.endswith("---")
    
    def test_generate_seqs(self):
        """Test that InsertionScanORFPool works with generate_seqs."""
        orf = "ATGGCCAAA"
        insertion = "TTT"
        pool = InsertionScanORFPool(
            orf, insertion, 
            insert_or_overwrite='overwrite', 
            step_size=1, 
            start=0,
            mode='sequential'
        )
        
        sequences = pool.generate_seqs(num_seqs=pool.num_internal_states)
        assert len(sequences) == pool.num_internal_states
        # All sequences should be different
        assert len(set(sequences)) == pool.num_internal_states


class TestGenerateSeqs:
    """Test generate_seqs functionality and quirks."""
    
    def test_generate_seqs_with_seed_reproducibility(self):
        """Test that seed produces reproducible results with random mode."""
        orf = "ATGGCCAAACCCTTT"  # 5 codons
        insertion = "GGG"
        
        pool = InsertionScanORFPool(
            orf, insertion,
            insert_or_overwrite='overwrite',
            mode='random'
        )
        
        # Same seed should produce same sequences
        seqs1 = pool.generate_seqs(num_seqs=10, seed=42)
        seqs2 = pool.generate_seqs(num_seqs=10, seed=42)
        assert seqs1 == seqs2, "Same seed should produce identical sequences"
        
        # Different seed should (likely) produce different sequences
        seqs3 = pool.generate_seqs(num_seqs=10, seed=123)
        assert seqs1 != seqs3, "Different seeds should produce different sequences"
    
    def test_generate_seqs_with_weighted_positions(self):
        """Test generate_seqs respects position_weights in random mode."""
        orf = "AAACCCGGGTTT"  # 4 codons
        
        pool = InsertionScanORFPool(
            orf,
            "ATG",
            insert_or_overwrite='overwrite',
            positions=[0, 3],  # First or last codon
            position_weights=[9.0, 1.0],  # 90% position 0, 10% position 3
            mode='random'
        )
        
        sequences = pool.generate_seqs(num_seqs=500, seed=42)
        
        # Count which position was used
        pos0_count = sum(1 for s in sequences if s.startswith("ATG"))
        pos3_count = sum(1 for s in sequences if s.endswith("ATG"))
        
        # Position 0 should be much more common
        assert pos0_count > pos3_count * 4, f"Weights not respected: pos0={pos0_count}, pos3={pos3_count}"
    
    def test_generate_seqs_num_complete_iterations(self):
        """Test generate_seqs with num_complete_iterations."""
        orf = "ATGGCCAAA"  # 3 codons
        insertion = "TTT"
        
        pool = InsertionScanORFPool(
            orf, insertion,
            insert_or_overwrite='overwrite',
            mode='sequential'
        )
        
        # 3 positions × 2 iterations = 6 sequences
        sequences = pool.generate_seqs(num_complete_iterations=2)
        assert len(sequences) == pool.num_internal_states * 2
        
        # First iteration
        first_iter = sequences[:pool.num_internal_states]
        # Second iteration
        second_iter = sequences[pool.num_internal_states:]
        
        # Both iterations should produce same sequences in same order
        assert first_iter == second_iter, "Iterations should repeat"
    
    def test_generate_seqs_return_computation_graph(self):
        """Test generate_seqs with return_computation_graph=True."""
        orf = "ATGGCCAAA"
        insertion = "TTT"
        
        pool = InsertionScanORFPool(
            orf, insertion,
            insert_or_overwrite='overwrite',
            mode='sequential'
        )
        
        result = pool.generate_seqs(num_seqs=3, return_computation_graph=True)
        
        assert isinstance(result, dict)
        assert "sequences" in result
        assert "graph" in result
        assert "node_sequences" in result
        
        assert len(result["sequences"]) == 3
        assert isinstance(result["graph"], dict)
        assert "nodes" in result["graph"]
    
    def test_generate_seqs_sequential_wraps(self):
        """Test that generate_seqs wraps around for sequential pools."""
        orf = "ATGGCC"  # 2 codons
        insertion = "TTT"  # 1 codon
        
        pool = InsertionScanORFPool(
            orf, insertion,
            insert_or_overwrite='overwrite',
            mode='sequential'
        )
        
        # Pool has 2 states, request 5 sequences
        sequences = pool.generate_seqs(num_seqs=5)
        assert len(sequences) == 5
        
        # Should wrap: [state0, state1, state0, state1, state0]
        assert sequences[0] == sequences[2] == sequences[4]
        assert sequences[1] == sequences[3]
    
    def test_generate_seqs_with_pool_parent_and_seed(self):
        """Test generate_seqs with Pool parent respects seed."""
        insertion_pool = Pool(seqs=["AAA", "GGG", "TTT"], mode='random')
        
        pool = InsertionScanORFPool(
            "ATGATGATG",
            insertion_pool,
            insert_or_overwrite='overwrite',
            mode='random'
        )
        
        # Same seed should produce same results
        seqs1 = pool.generate_seqs(num_seqs=20, seed=42)
        seqs2 = pool.generate_seqs(num_seqs=20, seed=42)
        assert seqs1 == seqs2
    
    def test_generate_seqs_mixed_modes(self):
        """Test generate_seqs with mixed sequential/random ancestry."""
        # Parent pool is sequential
        parent = Pool(seqs=["ATGATGATG", "CCCGGGAAA"], mode='sequential')
        
        # Child is random
        pool = InsertionScanORFPool(
            parent,
            "TTT",
            insert_or_overwrite='overwrite',
            positions=[0, 1, 2],  # 3 positions
            mode='random'
        )
        
        # With seed, should be reproducible
        seqs1 = pool.generate_seqs(num_seqs=10, seed=42)
        seqs2 = pool.generate_seqs(num_seqs=10, seed=42)
        assert seqs1 == seqs2
        
        # Should produce sequences from both parent backgrounds
        has_atg_background = any("ATG" in s[3:] for s in seqs1)
        has_ggg_background = any("GGG" in s[3:] for s in seqs1)
        # Note: With 10 samples, we should see both backgrounds
        assert has_atg_background or has_ggg_background


class TestRepr:
    """Test string representation."""
    
    def test_repr_overwrite_mode(self):
        """Test InsertionScanORFPool __repr__ for overwrite mode."""
        orf = "ATGGCC"
        insertion = "TTT"
        pool = InsertionScanORFPool(
            orf, insertion, 
            insert_or_overwrite='overwrite', 
            step_size=1, 
            start=0
        )
        repr_str = repr(pool)
        assert "InsertionScanORFPool" in repr_str
        assert "overwrite" in repr_str
    
    def test_repr_insert_mode(self):
        """Test InsertionScanORFPool __repr__ for insert mode."""
        orf = "ATGGCC"
        insertion = "TTT"
        pool = InsertionScanORFPool(
            orf, insertion, 
            insert_or_overwrite='insert', 
            step_size=2, 
            start=1
        )
        repr_str = repr(pool)
        assert "InsertionScanORFPool" in repr_str
        assert "insert" in repr_str
    
    def test_repr_with_long_sequences(self):
        """Test __repr__ with long sequences gets truncated."""
        orf = 'ATG' * 10  # 30 nucleotides
        insertion = 'CCC' * 5  # 15 nucleotides
        pool = InsertionScanORFPool(orf, insertion)
        repr_str = repr(pool)
        
        assert "InsertionScanORFPool" in repr_str
        assert "..." in repr_str  # Should be truncated
    
    def test_repr_with_positions(self):
        """Test __repr__ with explicit positions."""
        orf = "ATGGCCAAA"
        insertion = "TTT"
        pool = InsertionScanORFPool(orf, insertion, positions=[0, 2])
        repr_str = repr(pool)
        assert "InsertionScanORFPool" in repr_str
        assert "positions" in repr_str


class TestFlankingRegions:
    """Test flanking region (UTR) handling."""
    
    def test_with_flanking_regions(self):
        """Test InsertionScanORFPool with orf_start/orf_end."""
        # Full sequence: 5'UTR + ORF + 3'UTR
        full_seq = "AAAA" + "ATGATGATG" + "TTTT"  # 4 + 9 + 4 = 17 nt
        
        pool = InsertionScanORFPool(
            full_seq,
            "CCC",
            insert_or_overwrite='insert',
            orf_start=4,
            orf_end=13,
            mode='sequential'
        )
        
        assert pool.upstream_flank == "AAAA"
        assert pool.downstream_flank == "TTTT"
        assert pool.orf_seq == "ATGATGATG"
        
        # Check flanks are preserved in all outputs
        for state in range(pool.num_internal_states):
            pool.set_state(state)
            seq = pool.seq
            assert seq.startswith("AAAA"), f"5' UTR not preserved: {seq}"
            assert seq.endswith("TTTT"), f"3' UTR not preserved: {seq}"
    
    def test_flanks_with_positions_verification(self):
        """Test that flanks are preserved and content is correct with explicit positions."""
        # Full construct: 5'UTR + ORF + 3'UTR
        full_seq = "GGGGG" + "ATGCCCGGGTTT" + "AAAAA"  # 5 + 12 + 5 = 22 nt
        
        pool = InsertionScanORFPool(
            background_seq=full_seq,
            insert_seq="TAA",
            insert_or_overwrite='overwrite',
            positions=[1, 3],  # Overwrite CCC (pos 1) or TTT (pos 3)
            orf_start=5,
            orf_end=17,
            mode='sequential'
        )
        
        # State 0: overwrite at position 1 (CCC -> TAA)
        pool.set_state(0)
        seq = pool.seq
        assert seq.startswith("GGGGG"), "5'UTR corrupted"
        assert seq.endswith("AAAAA"), "3'UTR corrupted"
        orf_part = seq[5:17]
        orf_codons = [orf_part[j:j+3] for j in range(0, len(orf_part), 3)]
        assert orf_codons == ["ATG", "TAA", "GGG", "TTT"]
        
        # State 1: overwrite at position 3 (TTT -> TAA)
        pool.set_state(1)
        seq = pool.seq
        assert seq.startswith("GGGGG"), "5'UTR corrupted"
        assert seq.endswith("AAAAA"), "3'UTR corrupted"
        orf_part = seq[5:17]
        orf_codons = [orf_part[j:j+3] for j in range(0, len(orf_part), 3)]
        assert orf_codons == ["ATG", "CCC", "GGG", "TAA"]
    
    def test_flanks_with_insert_mode_length_change(self):
        """Test that flanks are preserved when insert mode changes ORF length."""
        # Full sequence: 5'UTR + ORF + 3'UTR
        full_seq = "CCCCC" + "ATGATGATG" + "GGGGG"  # 5 + 9 + 5 = 19 nt
        
        pool = InsertionScanORFPool(
            full_seq,
            "AAA",  # Insert 1 codon
            insert_or_overwrite='insert',
            orf_start=5,
            orf_end=14,
            mode='sequential'
        )
        
        # Original ORF is 3 codons = 9 nt
        # After insert, ORF is 4 codons = 12 nt
        # Total should be 5 + 12 + 5 = 22 nt
        expected_len = 22
        
        for state in range(pool.num_internal_states):
            pool.set_state(state)
            seq = pool.seq
            assert len(seq) == expected_len, f"State {state}: len={len(seq)}, expected {expected_len}"
            assert seq.startswith("CCCCC"), f"5' UTR corrupted at state {state}: {seq}"
            assert seq.endswith("GGGGG"), f"3' UTR corrupted at state {state}: {seq}"
    
    def test_flanks_with_mark_changes(self):
        """Test that flanks are preserved with mark_changes=True (swapcase)."""
        full_seq = "AAAA" + "ATGATGATG" + "TTTT"
        
        pool = InsertionScanORFPool(
            full_seq,
            "CCC",  # Will become 'ccc' via swapcase
            insert_or_overwrite='overwrite',
            mark_changes=True,
            orf_start=4,
            orf_end=13,
            mode='sequential'
        )
        
        pool.set_state(0)
        seq = pool.seq
        
        # Flanks should be unchanged
        assert seq.startswith("AAAA"), f"5' UTR corrupted: {seq}"
        assert seq.endswith("TTTT"), f"3' UTR corrupted: {seq}"
        
        # ORF should have lowercase insertion
        orf_part = seq[4:13]
        assert "ccc" in orf_part, f"mark_changes not applied in ORF: {orf_part}"
    
    def test_flanks_no_upstream(self):
        """Test with only downstream flank (orf_start=0)."""
        full_seq = "ATGATGATG" + "CCCCC"  # ORF starts at 0
        
        pool = InsertionScanORFPool(
            full_seq,
            "GGG",
            insert_or_overwrite='overwrite',
            orf_start=0,
            orf_end=9,
            mode='sequential'
        )
        
        assert pool.upstream_flank == ""
        assert pool.downstream_flank == "CCCCC"
        
        pool.set_state(0)
        seq = pool.seq
        assert seq.endswith("CCCCC"), f"3' UTR not preserved: {seq}"
    
    def test_flanks_no_downstream(self):
        """Test with only upstream flank (orf_end=len)."""
        full_seq = "CCCCC" + "ATGATGATG"  # ORF ends at end
        
        pool = InsertionScanORFPool(
            full_seq,
            "GGG",
            insert_or_overwrite='overwrite',
            orf_start=5,
            orf_end=14,
            mode='sequential'
        )
        
        assert pool.upstream_flank == "CCCCC"
        assert pool.downstream_flank == ""
        
        pool.set_state(0)
        seq = pool.seq
        assert seq.startswith("CCCCC"), f"5' UTR not preserved: {seq}"


class TestPoolChaining:
    """Test pool chaining / transformer pattern."""
    
    def test_pool_as_seq_input(self):
        """Test InsertionScanORFPool with Pool as seq input."""
        # Create a parent pool
        parent = Pool(seqs=["ATGATGATG", "CCCGGGAAA"], mode='sequential')
        
        pool = InsertionScanORFPool(
            parent,
            "TTT",
            insert_or_overwrite='overwrite',
            positions=[0],
            mode='sequential'
        )
        
        # Test with first parent sequence
        pool.set_state(0)
        assert pool.seq == "TTTATGATG"
        
        # Test with second parent sequence
        pool.set_state(1)
        assert pool.seq == "TTTGGGAAA"
    
    def test_pool_as_insertion_input(self):
        """Test InsertionScanORFPool with Pool as insertion_seq input."""
        insertion_pool = Pool(seqs=["AAA", "GGG", "TTT"], mode='sequential')
        
        pool = InsertionScanORFPool(
            "ATGATGATG",
            insertion_pool,
            insert_or_overwrite='overwrite',
            mode='sequential'
        )
        
        # Different combined states should produce different insertions
        results = []
        for combined_state in [0, 3, 6]:  # Same position (0), different insertions
            pool.set_state(combined_state)
            results.append(pool.seq)
        
        assert len(set(results)) == 3
        assert "AAA" in results[0]
        assert "GGG" in results[1]
        assert "TTT" in results[2]
    
    def test_chain_with_length_change(self):
        """Test chaining where length changes in parent pool."""
        from poolparty import DeletionScanPool
        
        base_seq = "AAACCCGGGTTT"  # 12 nt
        
        # Delete 3 nt at position 3 (CCC) - results in 9 nt
        deletion_pool = DeletionScanPool(
            background_seq=base_seq,
            deletion_size=3,
            positions=[3],
            mark_changes=False,  # Actually remove
            mode='sequential'
        )
        
        # Now insert at position 0 of the SHORTENED sequence
        insertion_pool = InsertionScanORFPool(
            deletion_pool,
            "ATG",
            insert_or_overwrite='insert',
            positions=[0],
            mode='sequential'
        )
        
        insertion_pool.set_state(0)
        result = insertion_pool.seq
        assert result == "ATGAAAGGGTTT", f"Chain result wrong: {result}"


class TestStateSpaceCalculation:
    """Test state space and positions calculation."""
    
    def test_positions_list_overwrite_mode(self):
        """Test that positions list is correctly computed for overwrite mode."""
        orf_seq = "ATGATGATGATGATGATGATGATG"  # 8 codons
        
        pool = InsertionScanORFPool(
            background_seq=orf_seq,
            insert_seq="AAACCC",  # 2 codons
            insert_or_overwrite='overwrite',
            start=1,
            end=6,
            step_size=1
        )
        # range(1, min(6,8)-2+1, 1) = range(1, 5, 1) = [1, 2, 3, 4]
        assert pool.positions == [1, 2, 3, 4], f"Expected [1,2,3,4], got {pool.positions}"
        assert pool.num_internal_states == 4
    
    def test_positions_list_insert_mode(self):
        """Test that positions list is correctly computed for insert mode."""
        orf_seq = "ATGATGATG"  # 3 codons
        
        pool = InsertionScanORFPool(
            background_seq=orf_seq,
            insert_seq="CCC",
            insert_or_overwrite='insert',
            start=0,
            step_size=1
        )
        # range(0, 3+1, 1) = [0, 1, 2, 3]
        assert pool.positions == [0, 1, 2, 3], f"Expected [0,1,2,3], got {pool.positions}"
        assert pool.num_internal_states == 4
    
    def test_positions_list_with_step_size(self):
        """Test positions list with non-trivial step_size."""
        orf_seq = "ATGATGATGATGATGATGATGATG"  # 8 codons
        
        pool = InsertionScanORFPool(
            background_seq=orf_seq,
            insert_seq="CCC",  # 1 codon
            insert_or_overwrite='overwrite',
            start=0,
            end=8,
            step_size=2
        )
        # range(0, 8-1+1, 2) = range(0, 8, 2) = [0, 2, 4, 6]
        assert pool.positions == [0, 2, 4, 6], f"Expected [0,2,4,6], got {pool.positions}"
        assert pool.num_internal_states == 4


class TestComplexChaining:
    """Test complex chaining with multiple ORF pools."""
    
    def test_deletion_orf_to_insertion_orf_chain(self):
        """Test chaining DeletionScanORFPool -> InsertionScanORFPool (Test 14 from notebook)."""
        from poolparty import DeletionScanORFPool
        
        base_orf = "ATGATGATGATGATG"  # 5 codons
        
        # First: delete 1 codon (position 2), actually removing it
        deletion_pool = DeletionScanORFPool(
            background_seq=base_orf,
            deletion_size=1,
            positions=[2],  # Always delete at position 2
            mark_changes=False,  # Actually remove (so output is valid DNA)
            mode='sequential'
        )
        
        # After deletion: 4 codons remain
        # Second: overwrite at position 0 with new codons
        insertion_pool = InsertionScanORFPool(
            background_seq=deletion_pool,
            insert_seq="GGGCCC",  # 2 codons
            insert_or_overwrite='overwrite',
            positions=[0],
            mode='sequential'
        )
        
        # Verify deletion result
        deletion_pool.set_state(0)
        del_result = deletion_pool.seq
        assert len(del_result) == 12, f"Should be 4 codons (12 nt) after deletion, got {len(del_result)}"
        
        # Verify chain result
        insertion_pool.set_state(0)
        final = insertion_pool.seq
        assert final.startswith("GGGCCC"), f"Insertion should be at start: {final}"
        assert len(final) == 12, f"Should still be 4 codons after overwrite, got {len(final)}"
    
    def test_chain_with_orf_length_change(self):
        """Test chain where deletion actually reduces ORF length."""
        from poolparty import DeletionScanORFPool
        
        base_orf = "AAACCCGGGTTT"  # 4 codons (12 nt)
        
        # Delete 1 codon at position 1 (CCC) - results in 3 codons
        deletion_pool = DeletionScanORFPool(
            background_seq=base_orf,
            deletion_size=1,
            positions=[1],
            mark_changes=False,  # Actually remove
            mode='sequential'
        )
        
        deletion_pool.set_state(0)
        del_result = deletion_pool.seq
        del_codons = [del_result[j:j+3] for j in range(0, len(del_result), 3)]
        assert del_codons == ["AAA", "GGG", "TTT"], f"Deletion result wrong: {del_codons}"
        
        # Now insert at position 1 of the SHORTENED sequence
        insertion_pool = InsertionScanORFPool(
            background_seq=deletion_pool,
            insert_seq="ATG",
            insert_or_overwrite='insert',
            positions=[1],
            mode='sequential'
        )
        
        insertion_pool.set_state(0)
        ins_result = insertion_pool.seq
        ins_codons = [ins_result[j:j+3] for j in range(0, len(ins_result), 3)]
        assert ins_codons == ["AAA", "ATG", "GGG", "TTT"], f"Chain result wrong: {ins_codons}"


class TestMultiCodonOperations:
    """Test operations with multiple codons."""
    
    def test_overwrite_multiple_codons(self):
        """Test overwriting 3 codons at a specific position."""
        orf_seq = "AAACCCGGGTTTATGATG"  # 6 codons
        insertion = "TAATAATAA"  # 3 codons
        
        pool = InsertionScanORFPool(
            orf_seq,
            insertion,
            insert_or_overwrite='overwrite',
            positions=[1],
            mode='sequential'
        )
        pool.set_state(0)
        seq = pool.seq
        codons = [seq[j:j+3] for j in range(0, len(seq), 3)]
        # Original: AAA CCC GGG TTT ATG ATG
        # After overwrite at 1 with 3 codons: AAA TAA TAA TAA ATG ATG
        assert codons == ["AAA", "TAA", "TAA", "TAA", "ATG", "ATG"], f"Overwrite wrong: {codons}"
    
    def test_overwrite_vs_insert_exact_position(self):
        """Test exact semantic difference between overwrite and insert."""
        orf_seq = "AAACCCGGGTTT"  # 4 codons: AAA CCC GGG TTT
        insertion = "ATG"
        
        # Overwrite at position 2 (should replace GGG)
        pool = InsertionScanORFPool(
            orf_seq,
            insertion,
            insert_or_overwrite='overwrite',
            positions=[2],
            mode='sequential'
        )
        pool.set_state(0)
        seq = pool.seq
        codons = [seq[j:j+3] for j in range(0, len(seq), 3)]
        assert codons == ["AAA", "CCC", "ATG", "TTT"], f"Overwrite wrong: {codons}"
        assert len(seq) == 12, "Overwrite should preserve length"
        
        # Insert at position 2 (should insert BEFORE GGG)
        pool = InsertionScanORFPool(
            orf_seq,
            insertion,
            insert_or_overwrite='insert',
            positions=[2],
            mode='sequential'
        )
        pool.set_state(0)
        seq = pool.seq
        codons = [seq[j:j+3] for j in range(0, len(seq), 3)]
        assert codons == ["AAA", "CCC", "ATG", "GGG", "TTT"], f"Insert wrong: {codons}"
        assert len(seq) == 15, "Insert should increase length"


class TestReadingFrameIntegrity:
    """Test that reading frame is maintained."""
    
    def test_overwrite_maintains_reading_frame(self):
        """Test that overwrite mode maintains reading frame."""
        orf = "ATGGCCAAACCCTTTGGG"  # 6 codons
        insertion = "CTGCTG"  # 2 codons
        pool = InsertionScanORFPool(
            orf, insertion, 
            insert_or_overwrite='overwrite', 
            step_size=1, 
            start=0,
            mode='sequential'
        )
        
        for state in range(pool.num_internal_states):
            pool.set_state(state)
            seq = pool.seq
            # Length should still be divisible by 3
            assert len(seq) % 3 == 0
            # Can split into codons
            codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
            assert len(codons) == 6
    
    def test_insert_maintains_reading_frame(self):
        """Test that insert mode maintains reading frame."""
        orf = "ATGGCCAAA"  # 3 codons
        insertion = "TTTCCC"  # 2 codons
        pool = InsertionScanORFPool(
            orf, insertion, 
            insert_or_overwrite='insert', 
            step_size=1, 
            start=0,
            mode='sequential'
        )
        
        for state in range(pool.num_internal_states):
            pool.set_state(state)
            seq = pool.seq
            # Length should be divisible by 3
            assert len(seq) % 3 == 0
            # Should be 5 codons total
            codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
            assert len(codons) == 5
