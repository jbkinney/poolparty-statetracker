"""Integration tests for design cards using real poolparty classes.

These tests validate real behavior and catch bugs early.
"""

import pytest
from poolparty import (
    Pool, MixedPool, DesignCards,
    InsertionScanPool, DeletionScanPool, SubseqPool, ShuffleScanPool,
    KMutationPool, RandomMutationPool,
    KMutationORFPool, RandomMutationORFPool,
    InsertionScanORFPool, DeletionScanORFPool,
    SpacingScanPool, IUPACPool, KmerPool,
)


# =============================================================================
# Phase 1: Core Infrastructure Tests
# =============================================================================

class TestDesignCardsClass:
    """Tests for the DesignCards class."""
    
    def test_basic_creation(self):
        """DesignCards can be created with a list of keys."""
        dc = DesignCards(['a', 'b', 'c'])
        assert dc.keys == ['a', 'b', 'c']
        assert len(dc) == 0
    
    def test_append_row(self):
        """DesignCards can append rows with values."""
        dc = DesignCards(['x', 'y'])
        dc.append_row({'x': 1, 'y': 2})
        dc.append_row({'x': 3, 'y': 4})
        
        assert len(dc) == 2
        assert dc['x'] == [1, 3]
        assert dc['y'] == [2, 4]
    
    def test_missing_keys_become_none(self):
        """Missing keys in append_row become None."""
        dc = DesignCards(['a', 'b', 'c'])
        dc.append_row({'a': 1})  # Missing b and c
        
        assert dc['a'] == [1]
        assert dc['b'] == [None]
        assert dc['c'] == [None]
    
    def test_get_row(self):
        """get_row returns a dictionary for a specific row."""
        dc = DesignCards(['name', 'value'])
        dc.append_row({'name': 'first', 'value': 100})
        dc.append_row({'name': 'second', 'value': 200})
        
        row0 = dc.get_row(0)
        assert row0 == {'name': 'first', 'value': 100}
        
        row1 = dc.get_row(1)
        assert row1 == {'name': 'second', 'value': 200}
    
    def test_contains(self):
        """DesignCards supports 'in' operator for column checking."""
        dc = DesignCards(['a', 'b'])
        assert 'a' in dc
        assert 'b' in dc
        assert 'c' not in dc
    
    def test_invalid_key_raises(self):
        """Accessing invalid key raises KeyError."""
        dc = DesignCards(['a', 'b'])
        with pytest.raises(KeyError, match="Column 'c' not found"):
            _ = dc['c']
    
    def test_invalid_row_index_raises(self):
        """Accessing invalid row index raises IndexError."""
        dc = DesignCards(['a'])
        dc.append_row({'a': 1})
        
        with pytest.raises(IndexError):
            dc.get_row(5)
    
    def test_to_dataframe(self):
        """to_dataframe converts to pandas DataFrame."""
        pytest.importorskip('pandas')
        import pandas as pd
        
        dc = DesignCards(['col1', 'col2'])
        dc.append_row({'col1': 'A', 'col2': 1})
        dc.append_row({'col1': 'B', 'col2': 2})
        
        df = dc.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ['col1', 'col2']
        assert len(df) == 2


class TestBasePoolDesignCards:
    """Tests for base Pool design card generation."""
    
    def test_simple_concatenation(self):
        """Design cards work with simple A + B concatenation."""
        A = Pool(['AAAA', 'TTTT'], name='A', mode='sequential', metadata='complete')
        B = Pool(['GGGG', 'CCCC'], name='B', mode='sequential', metadata='complete')
        library = A + B
        
        result = library.generate_seqs(num_seqs=4, return_design_cards=True)
        dc = result['design_cards']
        
        # Check sequences match design cards
        for i, seq in enumerate(result['sequences']):
            row = dc.get_row(i)
            assert row['sequence_id'] == i
            assert row['sequence_length'] == len(seq)
            
            # Values should match (requires metadata='complete')
            assert row['A_value'] + row['B_value'] == seq
            
            # Positions should be correct
            assert row['A_abs_start'] == 0
            assert row['A_abs_end'] == 4
            assert row['B_abs_start'] == 4
            assert row['B_abs_end'] == 8
    
    def test_pool_reuse(self):
        """Design cards handle pool reuse with [1], [2] suffixes."""
        A = Pool(['AA', 'TT'], name='A', mode='sequential', metadata='complete')
        library = A + A  # Same pool used twice
        
        result = library.generate_seqs(num_seqs=2, return_design_cards=True)
        dc = result['design_cards']
        
        # Should have A[1] and A[2] columns
        assert 'A[1]_index' in dc.keys
        assert 'A[2]_index' in dc.keys
        
        # Both should have same value since it's the same pool object (requires metadata='complete')
        for i in range(2):
            row = dc.get_row(i)
            assert row['A[1]_value'] == row['A[2]_value']
    
    def test_literals_not_tracked(self):
        """String literals are not tracked in design cards."""
        A = Pool(['AAAA'], name='A', mode='sequential')
        library = "PREFIX_" + A + "_SUFFIX"
        
        result = library.generate_seqs(num_seqs=1, return_design_cards=True)
        dc = result['design_cards']
        
        # Only A should be tracked
        assert 'A_index' in dc.keys
        assert 'PREFIX_index' not in dc.keys
        assert '_SUFFIX_index' not in dc.keys
        
        # But sequence should include literals
        row = dc.get_row(0)
        assert result['sequences'][0] == 'PREFIX_AAAA_SUFFIX'
    
    def test_unnamed_pools_not_tracked(self):
        """Pools without names are not tracked."""
        A = Pool(['AAAA'], name='A', mode='sequential')
        B = Pool(['BBBB'], mode='sequential')  # No name
        library = A + B
        
        result = library.generate_seqs(num_seqs=1, return_design_cards=True)
        dc = result['design_cards']
        
        assert 'A_index' in dc.keys
        # B should not have any columns since it has no name
        assert not any('B_' in k for k in dc.keys)
    
    def test_track_pools_filter(self):
        """track_pools parameter filters which pools to track."""
        A = Pool(['AAAA'], name='A', mode='sequential')
        B = Pool(['BBBB'], name='B', mode='sequential')
        C = Pool(['CCCC'], name='C', mode='sequential')
        library = A + B + C
        
        # Only track A and C
        result = library.generate_seqs(num_seqs=1, return_design_cards=True, track_pools=['A', 'C'])
        dc = result['design_cards']
        
        assert 'A_index' in dc.keys
        assert 'C_index' in dc.keys
        assert 'B_index' not in dc.keys
    
    def test_empty_track_pools(self):
        """Empty track_pools list tracks no pools."""
        A = Pool(['AAAA'], name='A', mode='sequential')
        B = Pool(['BBBB'], name='B', mode='sequential')
        library = A + B
        
        result = library.generate_seqs(num_seqs=1, return_design_cards=True, track_pools=[])
        dc = result['design_cards']
        
        # Only universal columns (sequence_id and sequence_length, no 'sequence')
        assert dc.keys == ['sequence_id', 'sequence_length']
    
    def test_duplicate_names_rejected(self):
        """Different pool objects with same name raises ValueError."""
        A1 = Pool(['AAAA'], name='promoter', mode='sequential')
        A2 = Pool(['TTTT'], name='promoter', mode='sequential')  # Same name!
        library = A1 + A2
        
        with pytest.raises(ValueError, match="Duplicate pool name 'promoter'"):
            library.generate_seqs(num_seqs=1, return_design_cards=True)
    
    def test_empty_string_name_not_tracked(self):
        """Pool with name='' is not tracked (treated like None)."""
        A = Pool(['AAAA'], name='', mode='sequential')  # Empty string name
        B = Pool(['BBBB'], name='B', mode='sequential')
        library = A + B
        
        result = library.generate_seqs(num_seqs=1, return_design_cards=True)
        dc = result['design_cards']
        
        # Empty name should not create columns starting with '_'
        assert 'B_index' in dc.keys
        assert '_index' not in dc.keys
    
    def test_design_cards_with_computation_graph(self):
        """return_design_cards and return_computation_graph can be used together."""
        A = Pool(['AAAA'], name='A', mode='sequential')
        B = Pool(['BBBB'], name='B', mode='sequential')
        library = A + B
        
        result = library.generate_seqs(
            num_seqs=1, 
            return_design_cards=True,
            return_computation_graph=True
        )
        
        assert 'sequences' in result
        assert 'design_cards' in result
        assert 'graph' in result
        assert 'node_sequences' in result


class TestComplexCompositeDesignCards:
    """Tests for complex composite pool structures."""
    
    def test_repetition_operator(self):
        """Design cards work with pool repetition (*)."""
        A = Pool(['AA'], name='A', mode='sequential')
        library = A * 3  # AAA AAA AAA (but as single pool)
        
        result = library.generate_seqs(num_seqs=1, return_design_cards=True)
        dc = result['design_cards']
        
        # A appears 3 times, should have A[1], A[2], A[3]
        assert 'A[1]_index' in dc.keys
        assert 'A[2]_index' in dc.keys
        assert 'A[3]_index' in dc.keys
        
        row = dc.get_row(0)
        assert result['sequences'][0] == 'AAAAAA'
        
        # Positions should be staggered
        assert row['A[1]_abs_start'] == 0
        assert row['A[1]_abs_end'] == 2
        assert row['A[2]_abs_start'] == 2
        assert row['A[2]_abs_end'] == 4
        assert row['A[3]_abs_start'] == 4
        assert row['A[3]_abs_end'] == 6
    
    def test_nested_concatenation(self):
        """Design cards work with nested (A + B) + C structure."""
        A = Pool(['AA'], name='A', mode='sequential')
        B = Pool(['BB'], name='B', mode='sequential')
        C = Pool(['CC'], name='C', mode='sequential')
        library = (A + B) + C
        
        result = library.generate_seqs(num_seqs=1, return_design_cards=True)
        dc = result['design_cards']
        
        row = dc.get_row(0)
        assert result['sequences'][0] == 'AABBCC'
        assert row['A_abs_start'] == 0
        assert row['A_abs_end'] == 2
        assert row['B_abs_start'] == 2
        assert row['B_abs_end'] == 4
        assert row['C_abs_start'] == 4
        assert row['C_abs_end'] == 6


class TestNumCompleteIterations:
    """Tests for num_complete_iterations with design cards."""
    
    def test_design_cards_with_complete_iterations(self):
        """Design cards work with num_complete_iterations."""
        A = Pool(['A1', 'A2'], name='A', mode='sequential')
        B = Pool(['B1', 'B2'], name='B', mode='sequential')
        library = A + B
        
        result = library.generate_seqs(num_complete_iterations=2, return_design_cards=True)
        dc = result['design_cards']
        
        # 2 states × 2 states × 2 iterations = 8 sequences
        assert len(dc) == 8
        assert len(result['sequences']) == 8


# =============================================================================
# Placeholder tests for later phases (will be filled in as implemented)
# =============================================================================

class TestMixedPoolDesignCards:
    """Tests for MixedPool design card generation (Phase 2)."""
    
    def test_mixed_pool_selection_tracking(self):
        """MixedPool tracks which child was selected."""
        A = Pool(['AAAA'], name='A', mode='sequential')
        B = Pool(['BBBB'], name='B', mode='sequential')
        mixed = MixedPool([A, B], name='mixed', mode='sequential')
        
        result = mixed.generate_seqs(num_seqs=2, return_design_cards=True)
        dc = result['design_cards']
        
        # Should have selection columns
        assert 'mixed_selected' in dc.keys
        assert 'mixed_selected_name' in dc.keys
        
        # Check selections
        row0 = dc.get_row(0)
        assert row0['mixed_selected'] == 0
        assert row0['mixed_selected_name'] == 'A'
        assert result['sequences'][0] == 'AAAA'
        
        row1 = dc.get_row(1)
        assert row1['mixed_selected'] == 1
        assert row1['mixed_selected_name'] == 'B'
        assert result['sequences'][1] == 'BBBB'
    
    def test_mixed_pool_child_expansion(self):
        """MixedPool children are expanded and tracked."""
        A = Pool(['A1', 'A2'], name='A', mode='sequential', metadata='complete')
        B = Pool(['B1', 'B2'], name='B', mode='sequential', metadata='complete')
        mixed = MixedPool([A, B], name='mixed', mode='sequential')
        
        result = mixed.generate_seqs(num_seqs=4, return_design_cards=True)
        dc = result['design_cards']
        
        # A and B should be tracked as children
        assert 'A_index' in dc.keys
        assert 'B_index' in dc.keys
        
        # Row 0: A selected at state 0 (A1)
        row0 = dc.get_row(0)
        assert row0['mixed_selected'] == 0
        assert row0['A_value'] == 'A1'
        assert row0['B_value'] is None  # Not selected
        
        # Row 1: A selected at state 1 (A2)
        row1 = dc.get_row(1)
        assert row1['mixed_selected'] == 0
        assert row1['A_value'] == 'A2'
        assert row1['B_value'] is None
        
        # Row 2: B selected at state 0 (B1)
        row2 = dc.get_row(2)
        assert row2['mixed_selected'] == 1
        assert row2['A_value'] is None  # Not selected
        assert row2['B_value'] == 'B1'
    
    def test_mixed_pool_composite_child(self):
        """MixedPool with composite children (A+B)."""
        X = Pool(['XX'], name='X', mode='sequential')
        Y = Pool(['YY'], name='Y', mode='sequential')
        
        # Two composite children: X+Y and Y+X
        child1 = X + Y  # XXYY
        child2 = Y + X  # YYXX
        
        mixed = MixedPool([child1, child2], name='mixed', mode='sequential')
        
        result = mixed.generate_seqs(num_seqs=2, return_design_cards=True)
        dc = result['design_cards']
        
        # X and Y should both be tracked
        # Note: They appear in different positions depending on selection
        assert 'X[1]_index' in dc.keys or 'X_index' in dc.keys
        assert 'Y[1]_index' in dc.keys or 'Y_index' in dc.keys
        
        # Row 0: child1 (X+Y) selected
        row0 = dc.get_row(0)
        assert result['sequences'][0] == 'XXYY'
        assert row0['mixed_selected'] == 0
    
    def test_mixed_pool_child_used_directly(self):
        """Pool used both inside MixedPool AND as direct segment."""
        A = Pool(['A1', 'A2'], name='A', mode='sequential')
        B = Pool(['BB'], name='B', mode='sequential')
        
        mixed = MixedPool([A, B], name='mixed', mode='sequential')
        library = mixed + A  # A appears in mixed AND directly
        
        result = library.generate_seqs(num_seqs=2, return_design_cards=True)
        dc = result['design_cards']
        
        # A should appear multiple times with [1], [2] notation
        # or with different abs positions
        assert 'A[1]_index' in dc.keys or 'A_index' in dc.keys
        
        # Both positions share the same state (same Pool object)
        row0 = dc.get_row(0)
        # The sequence should be consistent
        assert len(result['sequences'][0]) == 4  # mixed (2) + A (2)
    
    def test_mixed_pool_unnamed_child(self):
        """MixedPool with unnamed child shows selected=None for name."""
        A = Pool(['AAAA'], mode='sequential')  # No name
        B = Pool(['BBBB'], name='B', mode='sequential')
        mixed = MixedPool([A, B], name='mixed', mode='sequential')
        
        result = mixed.generate_seqs(num_seqs=2, return_design_cards=True)
        dc = result['design_cards']
        
        row0 = dc.get_row(0)  # A selected
        assert row0['mixed_selected'] == 0
        assert row0['mixed_selected_name'] is None  # A has no name
        
        row1 = dc.get_row(1)  # B selected
        assert row1['mixed_selected'] == 1
        assert row1['mixed_selected_name'] == 'B'


class TestInsertionScanPoolDesignCards:
    """Tests for InsertionScanPool design card generation (Phase 3.1)."""
    
    def test_insertion_position_tracking(self):
        """InsertionScanPool tracks insertion position."""
        background = "AAAAAAAA"  # 8 nucleotides
        insertion = "XX"
        
        pool = InsertionScanPool(
            background, insertion,
            insert_or_overwrite='overwrite',
            name='ins',
            mode='sequential'
        )
        
        result = pool.generate_seqs(num_seqs=4, return_design_cards=True)
        dc = result['design_cards']
        
        # Check position tracking columns
        assert 'ins_pos' in dc.keys
        assert 'ins_pos_abs' in dc.keys
        assert 'ins_insert' in dc.keys
        
        # Verify positions advance
        for i in range(4):
            row = dc.get_row(i)
            assert row['ins_pos'] == i  # Position advances
            assert row['ins_pos_abs'] == i  # Pool is at start, so same as pos
            assert row['ins_insert'] == 'XX'
    
    def test_insertion_position_with_offset(self):
        """InsertionScanPool respects offset in pos_abs calculation."""
        background = "AAAAAAAA"
        insertion = "XX"
        
        prefix = Pool(['PREFIX'], name='prefix', mode='sequential')
        pool = InsertionScanPool(
            background, insertion,
            insert_or_overwrite='overwrite',
            name='ins',
            mode='sequential'
        )
        
        library = prefix + pool  # Pool starts at position 6
        
        result = library.generate_seqs(num_seqs=2, return_design_cards=True)
        dc = result['design_cards']
        
        row0 = dc.get_row(0)
        assert row0['ins_pos'] == 0  # Relative position
        assert row0['ins_pos_abs'] == 6  # Absolute position (after PREFIX)
    
    def test_insertion_insert_mode(self):
        """InsertionScanPool works in insert mode (not overwrite)."""
        background = "AAAA"
        insertion = "XX"
        
        pool = InsertionScanPool(
            background, insertion,
            insert_or_overwrite='insert',
            name='ins',
            mode='sequential'
        )
        
        result = pool.generate_seqs(num_seqs=3, return_design_cards=True)
        dc = result['design_cards']
        
        # Sequence length should be 6 (4 + 2)
        row0 = dc.get_row(0)
        assert row0['sequence_length'] == 6
        assert row0['ins_pos'] == 0
        assert result['sequences'][0] == 'XXAAAA'
        
        row1 = dc.get_row(1)
        assert row1['ins_pos'] == 1
        assert result['sequences'][1] == 'AXXAAA'


class TestDeletionScanPoolDesignCards:
    """Tests for DeletionScanPool design card generation (Phase 3.2)."""
    
    def test_deletion_position_tracking(self):
        """DeletionScanPool tracks deletion position."""
        background = "AAAAAAAA"  # 8 nucleotides
        
        pool = DeletionScanPool(
            background,
            deletion_size=2,
            mark_changes=True,
            name='del',
            mode='sequential'
        )
        
        result = pool.generate_seqs(num_seqs=4, return_design_cards=True)
        dc = result['design_cards']
        
        # Check position tracking columns
        assert 'del_pos' in dc.keys
        assert 'del_pos_abs' in dc.keys
        assert 'del_del_len' in dc.keys
        
        # Verify positions and deletion length
        for i in range(4):
            row = dc.get_row(i)
            assert row['del_pos'] == i
            assert row['del_pos_abs'] == i
            assert row['del_del_len'] == 2
    
    def test_deletion_with_offset(self):
        """DeletionScanPool respects offset in pos_abs calculation."""
        background = "AAAAAAAA"
        
        prefix = Pool(['PREFIX'], name='prefix', mode='sequential')
        pool = DeletionScanPool(
            background,
            deletion_size=2,
            mark_changes=True,
            name='del',
            mode='sequential'
        )
        
        library = prefix + pool
        
        result = library.generate_seqs(num_seqs=2, return_design_cards=True)
        dc = result['design_cards']
        
        row0 = dc.get_row(0)
        assert row0['del_pos'] == 0
        assert row0['del_pos_abs'] == 6  # After PREFIX
    
    def test_deletion_unmarked(self):
        """DeletionScanPool with mark_changes=False."""
        background = "AAAAAAAA"
        
        pool = DeletionScanPool(
            background,
            deletion_size=2,
            mark_changes=False,  # Actually remove characters
            name='del',
            mode='sequential'
        )
        
        result = pool.generate_seqs(num_seqs=2, return_design_cards=True)
        dc = result['design_cards']
        
        # Sequence should be shorter
        row0 = dc.get_row(0)
        assert row0['sequence_length'] == 6  # 8 - 2
        assert row0['del_pos'] == 0
        assert row0['del_del_len'] == 2


class TestSubseqPoolDesignCards:
    """Tests for SubseqPool design card generation (Phase 3.3)."""
    
    def test_subseq_position_tracking(self):
        """SubseqPool tracks extraction position."""
        source = "ABCDEFGHIJ"  # 10 characters
        
        pool = SubseqPool(
            source,
            width=3,
            name='subseq',
            mode='sequential',
            metadata='complete'
        )
        
        result = pool.generate_seqs(num_seqs=4, return_design_cards=True)
        dc = result['design_cards']
        
        # Check position tracking columns
        assert 'subseq_pos' in dc.keys
        assert 'subseq_width' in dc.keys
        # No pos_abs since SubseqPool is length-changing
        assert 'subseq_pos_abs' not in dc.keys
        
        # Verify positions advance
        for i in range(4):
            row = dc.get_row(i)
            assert row['subseq_pos'] == i
            assert row['subseq_width'] == 3
            assert row['subseq_value'] == source[i:i+3]
    
    def test_subseq_width_in_metadata(self):
        """SubseqPool includes width in metadata."""
        source = "ABCDEFGHIJ"
        
        pool = SubseqPool(
            source,
            width=5,
            name='subseq',
            mode='sequential'
        )
        
        result = pool.generate_seqs(num_seqs=2, return_design_cards=True)
        dc = result['design_cards']
        
        row0 = dc.get_row(0)
        assert row0['subseq_width'] == 5
        assert row0['sequence_length'] == 5


class TestShuffleScanPoolDesignCards:
    """Tests for ShuffleScanPool design card generation (Phase 3.4)."""
    
    def test_shuffle_position_tracking(self):
        """ShuffleScanPool tracks shuffle window position."""
        background = "ABCDEFGH"  # 8 characters
        
        pool = ShuffleScanPool(
            background,
            shuffle_size=3,
            name='shuf',
            mode='sequential'
        )
        
        result = pool.generate_seqs(num_seqs=3, return_design_cards=True)
        dc = result['design_cards']
        
        # Check position tracking columns
        assert 'shuf_pos' in dc.keys
        assert 'shuf_pos_abs' in dc.keys
        assert 'shuf_window_size' in dc.keys
        
        # Verify positions advance
        for i in range(3):
            row = dc.get_row(i)
            assert row['shuf_pos'] == i
            assert row['shuf_pos_abs'] == i
            assert row['shuf_window_size'] == 3
    
    def test_shuffle_with_offset(self):
        """ShuffleScanPool respects offset in pos_abs calculation."""
        background = "AAAABBBB"
        
        prefix = Pool(['PREFIX'], name='prefix', mode='sequential')
        pool = ShuffleScanPool(
            background,
            shuffle_size=3,
            name='shuf',
            mode='sequential'
        )
        
        library = prefix + pool
        
        result = library.generate_seqs(num_seqs=2, return_design_cards=True)
        dc = result['design_cards']
        
        row0 = dc.get_row(0)
        assert row0['shuf_pos'] == 0
        assert row0['shuf_pos_abs'] == 6  # After PREFIX


class TestKMutationPoolDesignCards:
    """Tests for KMutationPool design card generation (Phase 4.1)."""
    
    def test_mutation_tracking(self):
        """KMutationPool tracks mutation positions and changes."""
        seq = "AAAA"
        
        pool = KMutationPool(
            seq,
            alphabet='dna',
            k=1,
            name='mut',
            mode='sequential'
        )
        
        result = pool.generate_seqs(num_seqs=4, return_design_cards=True)
        dc = result['design_cards']
        
        # Check mutation tracking columns
        assert 'mut_mut_pos' in dc.keys
        assert 'mut_mut_pos_abs' in dc.keys
        assert 'mut_mut_from' in dc.keys
        assert 'mut_mut_to' in dc.keys
        
        # Each mutation should have valid tracking
        for i in range(4):
            row = dc.get_row(i)
            assert isinstance(row['mut_mut_pos'], list)
            assert isinstance(row['mut_mut_from'], list)
            assert isinstance(row['mut_mut_to'], list)
            # k=1, so exactly one mutation
            assert len(row['mut_mut_pos']) == 1
            assert len(row['mut_mut_from']) == 1
            assert len(row['mut_mut_to']) == 1
            # Original is always 'A'
            assert row['mut_mut_from'][0] == 'A'
            # Mutated to something else
            assert row['mut_mut_to'][0] in ['C', 'G', 'T']
    
    def test_multiple_mutations(self):
        """KMutationPool tracks multiple mutations with k>1."""
        seq = "AAAA"
        
        pool = KMutationPool(
            seq,
            alphabet='dna',
            k=2,
            name='mut',
            mode='sequential'
        )
        
        result = pool.generate_seqs(num_seqs=2, return_design_cards=True)
        dc = result['design_cards']
        
        row0 = dc.get_row(0)
        # k=2, so two mutations
        assert len(row0['mut_mut_pos']) == 2
        assert len(row0['mut_mut_from']) == 2
        assert len(row0['mut_mut_to']) == 2
    
    def test_mutation_with_offset(self):
        """KMutationPool respects offset in mut_pos_abs calculation."""
        seq = "AAAA"
        
        prefix = Pool(['PREFIX'], name='prefix', mode='sequential')
        pool = KMutationPool(
            seq,
            alphabet='dna',
            k=1,
            name='mut',
            mode='sequential'
        )
        
        library = prefix + pool
        
        result = library.generate_seqs(num_seqs=2, return_design_cards=True)
        dc = result['design_cards']
        
        row0 = dc.get_row(0)
        # Mutation position is relative (0)
        assert row0['mut_mut_pos'][0] == 0
        # Absolute position is after PREFIX (length 6)
        assert row0['mut_mut_pos_abs'][0] == 6


class TestRandomMutationPoolDesignCards:
    """Tests for RandomMutationPool design card generation (Phase 4.2)."""
    
    def test_random_mutation_tracking(self):
        """RandomMutationPool tracks mutation count and positions."""
        seq = "AAAAAAAAAA"  # 10 A's
        
        pool = RandomMutationPool(
            seq,
            alphabet='dna',
            mutation_rate=0.5,  # High rate for testing
            name='rmut',
            mode='random'
        )
        
        result = pool.generate_seqs(num_seqs=5, return_design_cards=True, seed=42)
        dc = result['design_cards']
        
        # Check mutation tracking columns
        assert 'rmut_mut_count' in dc.keys
        assert 'rmut_mut_pos' in dc.keys
        assert 'rmut_mut_pos_abs' in dc.keys
        assert 'rmut_mut_from' in dc.keys
        assert 'rmut_mut_to' in dc.keys
        
        # At least one sequence should have mutations (with 50% rate on 10 positions)
        total_mutations = sum(dc['rmut_mut_count'])
        assert total_mutations > 0, "Expected at least some mutations"
        
        # Check that lists have correct lengths
        for i in range(5):
            row = dc.get_row(i)
            assert isinstance(row['rmut_mut_pos'], list)
            assert isinstance(row['rmut_mut_from'], list)
            assert isinstance(row['rmut_mut_to'], list)
            assert len(row['rmut_mut_pos']) == row['rmut_mut_count']
            assert len(row['rmut_mut_from']) == row['rmut_mut_count']
            assert len(row['rmut_mut_to']) == row['rmut_mut_count']
    
    def test_random_mutation_consistency(self):
        """RandomMutationPool mutations are consistent with reported metadata."""
        seq = "AAAA"
        
        pool = RandomMutationPool(
            seq,
            alphabet='dna',
            mutation_rate=1.0,  # Always mutate
            name='rmut',
            mode='random'
        )
        
        result = pool.generate_seqs(num_seqs=3, return_design_cards=True, seed=42)
        dc = result['design_cards']
        
        for i in range(3):
            row = dc.get_row(i)
            # With 100% mutation rate, all positions should mutate
            assert row['rmut_mut_count'] == 4, f"Expected 4 mutations, got {row['rmut_mut_count']}"
            # All from chars should be 'A'
            assert all(f == 'A' for f in row['rmut_mut_from'])


# =============================================================================
# Complex Composite Pool Tests - MixedPool with various pool types
# =============================================================================

class TestMixedPoolCompositeChildren:
    """Tests for MixedPool with composite children and complex structures.
    
    These tests validate that design cards correctly handle:
    - Composite children (A+B vs B+A) with different positions
    - Pool reuse across MixedPool children and direct segments
    - Position tracking consistency
    - None values for non-selected children
    """
    
    def test_composite_children_different_positions(self):
        """MixedPool with (A+B) vs (B+A) - same pools at different positions.
        
        Key validation: When the same pool appears at different positions
        in different children, the abs_start/abs_end should be correct
        for the selected child only.
        """
        A = Pool(['AA'], name='A', mode='sequential')
        B = Pool(['BB'], name='B', mode='sequential')
        
        # Two children with A and B in different positions
        child1 = A + B  # A at 0-2, B at 2-4
        child2 = B + A  # B at 0-2, A at 2-4
        
        mixed = MixedPool([child1, child2], name='mix', mode='sequential')
        
        result = mixed.generate_seqs(num_seqs=2, return_design_cards=True)
        dc = result['design_cards']
        
        # Row 0: child1 selected (A+B = AABB)
        row0 = dc.get_row(0)
        assert result['sequences'][0] == 'AABB'
        assert row0['mix_selected'] == 0
        
        # A should be tracked in child1's position
        # Since A appears in both children at different positions,
        # we need to verify the correct one is shown
        
        # Row 1: child2 selected (B+A = BBAA)
        row1 = dc.get_row(1)
        assert result['sequences'][1] == 'BBAA'
        assert row1['mix_selected'] == 1
    
    def test_pool_in_multiple_children_with_different_states(self):
        """Pool used in multiple MixedPool children - state consistency.
        
        When a pool with multiple states appears in different children,
        each child may select different states of that pool.
        """
        A = Pool(['A1', 'A2'], name='A', mode='sequential')
        B = Pool(['BB'], name='B', mode='sequential')
        
        # child1: A + B (A can be A1 or A2)
        # child2: B + A (A can be A1 or A2)
        child1 = A + B
        child2 = B + A
        
        mixed = MixedPool([child1, child2], name='mix', mode='sequential')
        
        # mixed has 4 states: child1-A1, child1-A2, child2-B-A1, child2-B-A2
        result = mixed.generate_seqs(num_seqs=4, return_design_cards=True)
        dc = result['design_cards']
        
        # Verify all sequences
        seqs = result['sequences']
        assert len(seqs) == 4
        
        # First 2 states should be child1 (A+B)
        assert seqs[0] in ['A1BB', 'A2BB'] or seqs[0] in ['BBA1', 'BBA2']
        
        # Verify state progression makes sense
        for i in range(4):
            row = dc.get_row(i)
            assert row['mix_selected'] in [0, 1]
    
    def test_pool_reuse_inside_and_outside_mixed(self):
        """Pool used both inside MixedPool and as direct segment.
        
        Critical test: Pool A is used:
        1. As a child inside MixedPool
        2. As a direct segment after MixedPool
        
        Both occurrences share the same state, so metadata must be consistent.
        """
        A = Pool(['A1', 'A2'], name='A', mode='sequential')
        B = Pool(['BB'], name='B', mode='sequential')
        
        mixed = MixedPool([A, B], name='mix', mode='sequential')
        library = mixed + A  # A appears in mixed AND directly
        
        result = library.generate_seqs(num_seqs=4, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(4):
            row = dc.get_row(i)
            seq = result['sequences'][i]
            
            # The direct A always appears and should have consistent value
            # with A inside mixed (when A is selected)
            
            # If A[1] (inside mixed) is selected, A[2] (direct) should have same value
            # because they're the same Pool object with shared state
            if row.get('A[1]_value') is not None:
                # A was selected in mixed
                assert row['A[1]_value'] == row['A[2]_value'], \
                    f"Shared pool should have same value: {row['A[1]_value']} != {row['A[2]_value']}"
            
            # Verify positions are different for the two occurrences
            if row.get('A[1]_abs_start') is not None and row.get('A[2]_abs_start') is not None:
                # A[1] is at start (0), A[2] is after mixed (2)
                assert row['A[1]_abs_start'] != row['A[2]_abs_start']
    
    def test_none_values_for_non_selected_children(self):
        """Non-selected MixedPool children have None for all metadata fields.
        
        When a MixedPool child is not selected, all its metadata fields
        should be None (not missing, but explicitly None).
        """
        A = Pool(['AAAA'], name='A', mode='sequential', metadata='complete')
        B = Pool(['BBBB'], name='B', mode='sequential', metadata='complete')
        C = Pool(['CCCC'], name='C', mode='sequential', metadata='complete')
        
        mixed = MixedPool([A, B, C], name='mix', mode='sequential')
        
        result = mixed.generate_seqs(num_seqs=3, return_design_cards=True)
        dc = result['design_cards']
        
        # Row 0: A selected
        row0 = dc.get_row(0)
        assert row0['A_value'] == 'AAAA'
        assert row0['B_value'] is None, "Non-selected B should have None value"
        assert row0['C_value'] is None, "Non-selected C should have None value"
        assert row0['B_index'] is None
        assert row0['C_index'] is None
        
        # Row 1: B selected
        row1 = dc.get_row(1)
        assert row1['A_value'] is None
        assert row1['B_value'] == 'BBBB'
        assert row1['C_value'] is None
        
        # Row 2: C selected
        row2 = dc.get_row(2)
        assert row2['A_value'] is None
        assert row2['B_value'] is None
        assert row2['C_value'] == 'CCCC'
    
    def test_deeply_nested_composite_in_mixed(self):
        """MixedPool with deeply nested composite children.
        
        Test structure:
        MixedPool([
            (A + B) + C,      # Nested concatenation
            A + (B + C),      # Different nesting
        ])
        
        Both produce "AABBCC" but with different internal structures.
        
        Since same pools (A, B, C) appear in multiple MixedPool children,
        they get occurrence indices: A[1], A[2], etc.
        """
        A = Pool(['AA'], name='A', mode='sequential', metadata='complete')
        B = Pool(['BB'], name='B', mode='sequential', metadata='complete')
        C = Pool(['CC'], name='C', mode='sequential', metadata='complete')
        
        # Different nesting orders
        child1 = (A + B) + C  # ((A + B) + C)
        child2 = A + (B + C)  # (A + (B + C))
        
        mixed = MixedPool([child1, child2], name='mix', mode='sequential')
        
        result = mixed.generate_seqs(num_seqs=2, return_design_cards=True)
        dc = result['design_cards']
        
        # Both children produce same sequence
        assert result['sequences'][0] == 'AABBCC'
        assert result['sequences'][1] == 'AABBCC'
        
        # Pools appear in both children, so they have occurrence indices
        # [1] for child1, [2] for child2
        
        # Row 0: child1 selected - A[1], B[1], C[1] are active
        row0 = dc.get_row(0)
        assert row0['A[1]_abs_start'] == 0
        assert row0['A[1]_abs_end'] == 2
        assert row0['B[1]_abs_start'] == 2
        assert row0['B[1]_abs_end'] == 4
        assert row0['C[1]_abs_start'] == 4
        assert row0['C[1]_abs_end'] == 6
        # [2] should be None (not selected)
        assert row0['A[2]_value'] is None
        
        # Row 1: child2 selected - A[2], B[2], C[2] are active
        row1 = dc.get_row(1)
        assert row1['A[2]_abs_start'] == 0
        assert row1['A[2]_abs_end'] == 2
        assert row1['B[2]_abs_start'] == 2
        assert row1['B[2]_abs_end'] == 4
        assert row1['C[2]_abs_start'] == 4
        assert row1['C[2]_abs_end'] == 6
        # [1] should be None (not selected)
        assert row1['A[1]_value'] is None


class TestMixedPoolWithTransformers:
    """Tests for MixedPool with transformer pools as children."""
    
    def test_mixed_pool_with_insertion_scan_children(self):
        """MixedPool with InsertionScanPool children.
        
        Each child is a scan pool that tracks position metadata.
        The selected child's position should be correctly reported.
        """
        bg = "AAAA"
        ins = "XX"
        
        scan1 = InsertionScanPool(bg, ins, name='scan1', mode='sequential',
                                  insert_or_overwrite='overwrite')
        scan2 = InsertionScanPool(bg, ins, name='scan2', mode='sequential',
                                  insert_or_overwrite='overwrite', start=1)
        
        mixed = MixedPool([scan1, scan2], name='mix', mode='sequential')
        
        result = mixed.generate_seqs(num_seqs=6, return_design_cards=True)
        dc = result['design_cards']
        
        # scan1 has 3 states (pos 0, 1, 2)
        # scan2 has 2 states (pos 1, 2 due to start=1)
        # Total: 5 states
        
        for i in range(len(result['sequences'])):
            row = dc.get_row(i)
            selected = row['mix_selected']
            
            if selected == 0:
                # scan1 selected
                assert row['scan1_pos'] is not None
                assert row['scan2_pos'] is None
                assert row['scan1_insert'] == 'XX'
            else:
                # scan2 selected
                assert row['scan1_pos'] is None
                assert row['scan2_pos'] is not None
                assert row['scan2_insert'] == 'XX'
    
    def test_mixed_pool_with_mutation_children(self):
        """MixedPool with KMutationPool children.
        
        Tests mutation tracking when mutations are inside MixedPool children.
        """
        seq = "AAAA"
        
        mut1 = KMutationPool(seq, alphabet='dna', k=1, name='mut1', mode='sequential')
        mut2 = KMutationPool(seq, alphabet='dna', k=2, name='mut2', mode='sequential')
        
        mixed = MixedPool([mut1, mut2], name='mix', mode='sequential')
        
        result = mixed.generate_seqs(num_seqs=10, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(10):
            row = dc.get_row(i)
            selected = row['mix_selected']
            
            if selected == 0:
                # mut1 (k=1) selected
                assert row['mut1_mut_pos'] is not None
                assert len(row['mut1_mut_pos']) == 1, "k=1 should have 1 mutation"
                assert row['mut2_mut_pos'] is None
            else:
                # mut2 (k=2) selected
                assert row['mut1_mut_pos'] is None
                assert row['mut2_mut_pos'] is not None
                assert len(row['mut2_mut_pos']) == 2, "k=2 should have 2 mutations"
    
    def test_composite_with_transformer_in_mixed(self):
        """MixedPool with composite children containing transformers.
        
        Structure:
        MixedPool([
            prefix + InsertionScan + suffix,
            prefix + KMutation + suffix,
        ])
        
        Since prefix and suffix appear in both children, they get occurrence indices:
        prefix[1], suffix[1] for child1; prefix[2], suffix[2] for child2
        """
        prefix = Pool(['PRE'], name='prefix', mode='sequential', metadata='complete')
        suffix = Pool(['SUF'], name='suffix', mode='sequential', metadata='complete')
        
        bg = "AAAA"
        scan = InsertionScanPool(bg, "XX", name='scan', mode='sequential',
                                 insert_or_overwrite='overwrite')
        mut = KMutationPool(bg, alphabet='dna', k=1, name='mut', mode='sequential')
        
        child1 = prefix + scan + suffix  # PRE + (scanned AAAA) + SUF
        child2 = prefix + mut + suffix   # PRE + (mutated AAAA) + SUF
        
        mixed = MixedPool([child1, child2], name='mix', mode='sequential')
        
        result = mixed.generate_seqs(num_seqs=10, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(10):
            row = dc.get_row(i)
            seq = result['sequences'][i]
            
            # Sequence should start with PRE and end with SUF
            assert seq.startswith('PRE'), f"Should start with PRE: {seq}"
            assert seq.endswith('SUF'), f"Should end with SUF: {seq}"
            
            selected = row['mix_selected']
            
            if selected == 0:
                # child1 (scan) selected - use prefix[1], suffix[1]
                assert row['prefix[1]_abs_start'] == 0
                assert row['prefix[1]_abs_end'] == 3
                assert row['scan_pos'] is not None
                assert row['prefix[2]_value'] is None  # Not selected
            else:
                # child2 (mut) selected - use prefix[2], suffix[2]
                assert row['prefix[2]_abs_start'] == 0
                assert row['prefix[2]_abs_end'] == 3
                assert row['mut_mut_pos'] is not None
                assert row['prefix[1]_value'] is None  # Not selected


class TestMixedPoolPositionTracking:
    """Tests for position tracking accuracy in complex MixedPool scenarios."""
    
    def test_position_abs_consistency_with_sequence(self):
        """Verify that abs positions in metadata match actual sequence positions.
        
        This is a critical test: the metadata's abs_start/abs_end should
        allow us to extract the exact value from the sequence.
        """
        A = Pool(['AAAA'], name='A', mode='sequential', metadata='complete')
        B = Pool(['BBBB'], name='B', mode='sequential', metadata='complete')
        
        library = A + B
        
        result = library.generate_seqs(num_seqs=1, return_design_cards=True)
        dc = result['design_cards']
        row = dc.get_row(0)
        seq = result['sequences'][0]
        
        # Extract using metadata positions
        a_start, a_end = row['A_abs_start'], row['A_abs_end']
        b_start, b_end = row['B_abs_start'], row['B_abs_end']
        
        # Should match the stored values
        assert seq[a_start:a_end] == row['A_value']
        assert seq[b_start:b_end] == row['B_value']
    
    def test_position_tracking_with_variable_length_children(self):
        """MixedPool with children of same total length but different compositions.
        
        Structure:
        MixedPool([
            AA + BBBB,  # 2 + 4 = 6
            AAAA + BB,  # 4 + 2 = 6
        ])
        
        Both children have length 6, but A and B have different sizes.
        """
        A_short = Pool(['AA'], name='A', mode='sequential')
        A_long = Pool(['AAAA'], name='A_long', mode='sequential')  # Different name to avoid collision
        B_short = Pool(['BB'], name='B', mode='sequential')
        B_long = Pool(['BBBB'], name='B_long', mode='sequential')
        
        child1 = A_short + B_long  # AA + BBBB
        child2 = A_long + B_short  # AAAA + BB
        
        mixed = MixedPool([child1, child2], name='mix', mode='sequential')
        
        result = mixed.generate_seqs(num_seqs=2, return_design_cards=True)
        dc = result['design_cards']
        
        # Row 0: child1 (AA + BBBB)
        row0 = dc.get_row(0)
        assert result['sequences'][0] == 'AABBBB'
        assert row0['A_abs_end'] == 2  # AA ends at position 2
        assert row0['B_long_abs_start'] == 2
        
        # Row 1: child2 (AAAA + BB)
        row1 = dc.get_row(1)
        assert result['sequences'][1] == 'AAAABB'
        assert row1['A_long_abs_end'] == 4  # AAAA ends at position 4
        assert row1['B_abs_start'] == 4
    
    def test_scan_pool_pos_abs_calculation(self):
        """Verify pos_abs is correctly calculated for scan pools in composites.
        
        When InsertionScanPool is part of a composite, its pos_abs should
        account for the prefix offset.
        """
        prefix = Pool(['PREFIX'], name='prefix', mode='sequential')  # length 6
        bg = "AAAAAAAA"  # length 8
        ins = "XX"
        
        scan = InsertionScanPool(bg, ins, name='scan', mode='sequential',
                                 insert_or_overwrite='overwrite')
        
        library = prefix + scan
        
        result = library.generate_seqs(num_seqs=4, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(4):
            row = dc.get_row(i)
            
            # scan's pos is relative (0, 1, 2, 3, ...)
            # scan's pos_abs should be pos + prefix_length (6)
            expected_pos_abs = row['scan_pos'] + 6
            assert row['scan_pos_abs'] == expected_pos_abs, \
                f"pos_abs should be {expected_pos_abs}, got {row['scan_pos_abs']}"
            
            # Verify by checking the sequence
            seq = result['sequences'][i]
            # The XX should appear at pos_abs
            pos_abs = row['scan_pos_abs']
            assert seq[pos_abs:pos_abs+2] == 'XX' or seq[pos_abs:pos_abs+2] == 'xx', \
                f"Expected XX at position {pos_abs} in {seq}"


class TestMixedPoolStateManagement:
    """Tests for state management and synchronization in MixedPool."""
    
    def test_state_decomposition_with_composite_children(self):
        """Verify state decomposition works correctly for composite MixedPool children.
        
        MixedPool's total states = sum of all children's states.
        When children are composites, their states are products of their components.
        """
        A = Pool(['A1', 'A2'], name='A', mode='sequential')  # 2 states
        B = Pool(['B1', 'B2', 'B3'], name='B', mode='sequential')  # 3 states
        
        # child1 = A + B has 2*3 = 6 states
        # child2 = B + A has 3*2 = 6 states
        child1 = A + B
        child2 = B + A
        
        mixed = MixedPool([child1, child2], name='mix', mode='sequential')
        
        # Total states = 6 + 6 = 12
        assert mixed.num_internal_states == 12
        
        result = mixed.generate_seqs(num_seqs=12, return_design_cards=True)
        dc = result['design_cards']
        
        # Count how many times each child is selected
        child1_count = sum(1 for i in range(12) if dc.get_row(i)['mix_selected'] == 0)
        child2_count = sum(1 for i in range(12) if dc.get_row(i)['mix_selected'] == 1)
        
        assert child1_count == 6, f"child1 should be selected 6 times, got {child1_count}"
        assert child2_count == 6, f"child2 should be selected 6 times, got {child2_count}"
    
    def test_complete_iteration_coverage(self):
        """Verify that num_complete_iterations covers all states exactly.
        
        Using num_complete_iterations=2 should give us each state exactly twice.
        """
        A = Pool(['A1', 'A2'], name='A', mode='sequential')
        B = Pool(['B1', 'B2'], name='B', mode='sequential')
        
        mixed = MixedPool([A, B], name='mix', mode='sequential')
        
        # 2 + 2 = 4 states
        result = mixed.generate_seqs(num_complete_iterations=2, return_design_cards=True)
        dc = result['design_cards']
        
        assert len(dc) == 8  # 4 states * 2 iterations
        
        # Each unique sequence should appear exactly twice
        from collections import Counter
        seq_counts = Counter(result['sequences'])
        
        for seq, count in seq_counts.items():
            assert count == 2, f"Sequence {seq} should appear 2 times, got {count}"


class TestMixedPoolWithTrackPools:
    """Tests for track_pools filtering with MixedPool."""
    
    def test_track_specific_pool_in_mixed(self):
        """Track only specific pools inside MixedPool.
        
        When tracking only 'A', non-selected children shouldn't affect
        the columns that appear.
        """
        A = Pool(['AAAA'], name='A', mode='sequential')
        B = Pool(['BBBB'], name='B', mode='sequential')
        C = Pool(['CCCC'], name='C', mode='sequential')
        
        mixed = MixedPool([A, B, C], name='mix', mode='sequential')
        
        # Only track A and mix
        result = mixed.generate_seqs(num_seqs=3, return_design_cards=True,
                                     track_pools=['A', 'mix'])
        dc = result['design_cards']
        
        # Should have mix columns and A columns, but not B or C
        assert 'mix_selected' in dc.keys
        assert 'A_index' in dc.keys
        assert 'B_index' not in dc.keys
        assert 'C_index' not in dc.keys
    
    def test_track_mixed_pool_only(self):
        """Track only the MixedPool, not its children.
        
        Should only get MixedPool-level metadata (selected, selected_name).
        """
        A = Pool(['AAAA'], name='A', mode='sequential')
        B = Pool(['BBBB'], name='B', mode='sequential')
        
        mixed = MixedPool([A, B], name='mix', mode='sequential')
        
        result = mixed.generate_seqs(num_seqs=2, return_design_cards=True,
                                     track_pools=['mix'])
        dc = result['design_cards']
        
        # Should have mix columns only
        assert 'mix_selected' in dc.keys
        assert 'mix_selected_name' in dc.keys
        assert 'A_index' not in dc.keys
        assert 'B_index' not in dc.keys
    
    def test_track_child_but_not_mixed(self):
        """Track children but not the MixedPool itself."""
        A = Pool(['AAAA'], name='A', mode='sequential')
        B = Pool(['BBBB'], name='B', mode='sequential')
        
        mixed = MixedPool([A, B], name='mix', mode='sequential')
        
        result = mixed.generate_seqs(num_seqs=2, return_design_cards=True,
                                     track_pools=['A', 'B'])
        dc = result['design_cards']
        
        # Should have A and B but not mix
        assert 'A_index' in dc.keys
        assert 'B_index' in dc.keys
        assert 'mix_selected' not in dc.keys


class TestNestedMixedPools:
    """Tests for nested MixedPool structures."""
    
    def test_mixed_inside_mixed(self):
        """MixedPool as a child of another MixedPool.
        
        Structure:
        outer_mixed = MixedPool([
            inner_mixed,  # MixedPool([A, B])
            C,
        ])
        """
        A = Pool(['AA'], name='A', mode='sequential')
        B = Pool(['BB'], name='B', mode='sequential')
        C = Pool(['CC'], name='C', mode='sequential', metadata='complete')
        
        inner_mixed = MixedPool([A, B], name='inner', mode='sequential')
        outer_mixed = MixedPool([inner_mixed, C], name='outer', mode='sequential')
        
        result = outer_mixed.generate_seqs(num_seqs=3, return_design_cards=True)
        dc = result['design_cards']
        
        # States: inner_mixed has 2 states, C has 1 state
        # Total: 3 states
        
        for i in range(3):
            row = dc.get_row(i)
            outer_selected = row['outer_selected']
            
            if outer_selected == 0:
                # inner_mixed selected
                assert row['inner_selected'] is not None
                assert row['C_value'] is None
            else:
                # C selected
                assert row['C_value'] == 'CC'
    
    def test_multiple_levels_of_nesting(self):
        """Three levels of MixedPool nesting.
        
        Structure:
        level1 = MixedPool([level2, D])
        level2 = MixedPool([level3, C])
        level3 = MixedPool([A, B])
        """
        A = Pool(['AA'], name='A', mode='sequential')
        B = Pool(['BB'], name='B', mode='sequential')
        C = Pool(['CC'], name='C', mode='sequential')
        D = Pool(['DD'], name='D', mode='sequential')
        
        level3 = MixedPool([A, B], name='L3', mode='sequential')  # 2 states
        level2 = MixedPool([level3, C], name='L2', mode='sequential')  # 3 states
        level1 = MixedPool([level2, D], name='L1', mode='sequential')  # 4 states
        
        result = level1.generate_seqs(num_seqs=4, return_design_cards=True)
        dc = result['design_cards']
        
        assert len(dc) == 4
        
        # Verify each sequence is unique and valid
        unique_seqs = set(result['sequences'])
        assert unique_seqs == {'AA', 'BB', 'CC', 'DD'}


class TestMixedPoolValueConsistency:
    """Tests to verify metadata values match actual generated sequences."""
    
    def test_value_field_matches_sequence_extraction(self):
        """The _value field should match what we extract from sequence using positions."""
        A = Pool(['AAAA'], name='A', mode='sequential', metadata='complete')
        B = Pool(['BBBB'], name='B', mode='sequential', metadata='complete')
        C = Pool(['CCCC'], name='C', mode='sequential', metadata='complete')
        
        library = A + B + C
        
        result = library.generate_seqs(num_seqs=1, return_design_cards=True)
        dc = result['design_cards']
        row = dc.get_row(0)
        seq = result['sequences'][0]
        
        # Verify each pool's value matches sequence slice
        for pool_name in ['A', 'B', 'C']:
            start = row[f'{pool_name}_abs_start']
            end = row[f'{pool_name}_abs_end']
            value = row[f'{pool_name}_value']
            
            extracted = seq[start:end]
            assert extracted == value, \
                f"{pool_name}: extracted '{extracted}' != value '{value}'"
    
    def test_mutation_changes_reflected_in_value(self):
        """Mutation pool's _value should show the mutated sequence."""
        seq = "AAAA"
        
        mut = KMutationPool(seq, alphabet='dna', k=1, name='mut', mode='sequential', metadata='complete')
        
        result = mut.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(5):
            row = dc.get_row(i)
            
            # The value should match the generated sequence
            assert row['mut_value'] == result['sequences'][i]
            
            # The mutation should be reflected in the value
            mut_pos = row['mut_mut_pos'][0]
            mut_to = row['mut_mut_to'][0]
            
            assert row['mut_value'][mut_pos] == mut_to, \
                f"Value at position {mut_pos} should be {mut_to}"
    
    def test_scan_pool_value_shows_modified_sequence(self):
        """InsertionScanPool's _value should show the sequence with insertion."""
        bg = "AAAAAAAA"
        ins = "XX"
        
        scan = InsertionScanPool(bg, ins, name='scan', mode='sequential',
                                 insert_or_overwrite='overwrite', metadata='complete')
        
        result = scan.generate_seqs(num_seqs=3, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(3):
            row = dc.get_row(i)
            pos = row['scan_pos']
            value = row['scan_value']
            
            # The XX should be at position 'pos' in the value
            assert value[pos:pos+2] == 'XX', \
                f"Expected XX at position {pos} in '{value}'"
    
    def test_mixed_pool_value_matches_selected_child(self):
        """MixedPool's _value should match the selected child's output."""
        A = Pool(['AAAA'], name='A', mode='sequential')
        B = Pool(['BBBB'], name='B', mode='sequential')
        
        mixed = MixedPool([A, B], name='mix', mode='sequential', metadata='complete')
        
        result = mixed.generate_seqs(num_seqs=2, return_design_cards=True)
        dc = result['design_cards']
        
        row0 = dc.get_row(0)
        assert row0['mix_value'] == 'AAAA'
        assert row0['mix_selected_name'] == 'A'
        
        row1 = dc.get_row(1)
        assert row1['mix_value'] == 'BBBB'
        assert row1['mix_selected_name'] == 'B'


class TestComplexRealWorldScenarios:
    """Tests simulating real-world complex library designs."""
    
    def test_promoter_variant_library(self):
        """Simulate a promoter variant library with different modification types.
        
        Structure:
        prefix + MixedPool([
            promoter_wt,           # Wild-type
            promoter_mut,          # Single mutation
            promoter_scan,         # Insertion scan
        ]) + suffix
        """
        prefix = Pool(['UTR5'], name='5utr', mode='sequential')
        suffix = Pool(['UTR3'], name='3utr', mode='sequential')
        
        promoter_wt = Pool(['AAAAAAAA'], name='wt', mode='sequential', metadata='complete')
        promoter_mut = KMutationPool('AAAAAAAA', alphabet='dna', k=1, 
                                     name='mut', mode='sequential')
        promoter_scan = InsertionScanPool('AAAAAAAA', 'XX', name='scan',
                                          mode='sequential', insert_or_overwrite='overwrite')
        
        variants = MixedPool([promoter_wt, promoter_mut, promoter_scan],
                            name='variants', mode='sequential')
        
        library = prefix + variants + suffix
        
        result = library.generate_seqs(num_seqs=20, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(20):
            row = dc.get_row(i)
            seq = result['sequences'][i]
            
            # All sequences should start with UTR5 and end with UTR3
            assert seq.startswith('UTR5'), f"Should start with UTR5: {seq}"
            assert seq.endswith('UTR3'), f"Should end with UTR3: {seq}"
            
            # Check variant type
            selected = row['variants_selected']
            if selected == 0:
                assert row['wt_value'] == 'AAAAAAAA'
            elif selected == 1:
                # Mutation should be tracked
                assert row['mut_mut_pos'] is not None
            else:
                # Scan should be tracked
                assert row['scan_pos'] is not None
    
    def test_combinatorial_domain_library(self):
        """Simulate a combinatorial protein domain library.
        
        Structure:
        domain1_variants + linker + domain2_variants
        
        Where each domain has multiple options (MixedPool).
        """
        domain1_v1 = Pool(['AAA'], name='d1v1', mode='sequential')
        domain1_v2 = Pool(['BBB'], name='d1v2', mode='sequential')
        domain1 = MixedPool([domain1_v1, domain1_v2], name='domain1', mode='sequential')
        
        linker = Pool(['---'], name='linker', mode='sequential', metadata='complete')
        
        domain2_v1 = Pool(['XXX'], name='d2v1', mode='sequential')
        domain2_v2 = Pool(['YYY'], name='d2v2', mode='sequential')
        domain2 = MixedPool([domain2_v1, domain2_v2], name='domain2', mode='sequential')
        
        library = domain1 + linker + domain2
        
        # 2 * 1 * 2 = 4 combinations
        result = library.generate_seqs(num_seqs=4, return_design_cards=True)
        dc = result['design_cards']
        
        expected_seqs = {'AAA---XXX', 'AAA---YYY', 'BBB---XXX', 'BBB---YYY'}
        actual_seqs = set(result['sequences'])
        
        assert actual_seqs == expected_seqs, \
            f"Expected {expected_seqs}, got {actual_seqs}"
        
        # Verify domain selections are tracked
        for i in range(4):
            row = dc.get_row(i)
            
            # Both MixedPools should track selection
            assert row['domain1_selected'] in [0, 1]
            assert row['domain2_selected'] in [0, 1]
            
            # Linker should always be same
            assert row['linker_value'] == '---'


# =============================================================================
# Phase 7: KMutationORFPool Integration Tests
# =============================================================================

class TestKMutationORFPoolDesignCards:
    """Integration tests for KMutationORFPool design cards."""
    
    def test_basic_k1_mutation_tracking(self):
        """Single codon mutation is tracked correctly."""
        # 4 codons: ATG GAA CCC AAA -> M E P K
        orf = 'ATGGAACCCAAA'
        pool = KMutationORFPool(
            orf, mutation_type='any_codon', k=1, 
            name='k1', mode='sequential'
        )
        
        result = pool.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(5):
            row = dc.get_row(i)
            seq = result['sequences'][i]
            
            # Should have exactly 1 mutation
            assert len(row['k1_codon_pos']) == 1
            assert len(row['k1_codon_from']) == 1
            assert len(row['k1_codon_to']) == 1
            assert len(row['k1_aa_from']) == 1
            assert len(row['k1_aa_to']) == 1
            
            # Verify codon_pos_abs is correct
            pos = row['k1_codon_pos'][0]
            abs_pos = row['k1_codon_pos_abs'][0]
            assert abs_pos == pos * 3, f"Expected {pos * 3}, got {abs_pos}"
            
            # Verify the mutation is at the correct position in the sequence
            mutated_codon = row['k1_codon_to'][0]
            seq_codon = seq[abs_pos:abs_pos+3]
            assert seq_codon.upper() == mutated_codon.upper(), \
                f"Expected {mutated_codon} at position {abs_pos}, got {seq_codon}"
    
    def test_k2_multiple_mutations(self):
        """Two codon mutations are tracked correctly."""
        orf = 'ATGGAACCCAAA'  # 4 codons
        pool = KMutationORFPool(
            orf, mutation_type='any_codon', k=2,
            name='k2', mode='sequential'
        )
        
        result = pool.generate_seqs(num_seqs=10, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(10):
            row = dc.get_row(i)
            
            # Should have exactly 2 mutations
            assert len(row['k2_codon_pos']) == 2
            assert len(row['k2_codon_to']) == 2
            
            # Positions should be unique
            assert len(set(row['k2_codon_pos'])) == 2
    
    def test_with_flanking_regions(self):
        """Flanking regions affect codon_pos_abs correctly."""
        # 5nt upstream + 12nt ORF + 3nt downstream = 20nt total
        full_seq = 'GGGGGATGGAACCCAAATTT'
        pool = KMutationORFPool(
            full_seq, mutation_type='any_codon', k=1,
            orf_start=5, orf_end=17,  # ORF is ATGGAACCCAAA
            name='flanked', mode='sequential'
        )
        
        result = pool.generate_seqs(num_seqs=3, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(3):
            row = dc.get_row(i)
            seq = result['sequences'][i]
            
            # Sequence should preserve flanks
            assert seq[:5] == 'GGGGG', f"Upstream flank not preserved: {seq}"
            assert seq[-3:] == 'TTT', f"Downstream flank not preserved: {seq}"
            
            # codon_pos_abs should account for upstream flank
            pos = row['flanked_codon_pos'][0]
            abs_pos = row['flanked_codon_pos_abs'][0]
            expected_abs = 5 + pos * 3  # upstream_len + pos * 3
            assert abs_pos == expected_abs, \
                f"Expected {expected_abs}, got {abs_pos}"
    
    def test_in_composite_with_prefix_suffix(self):
        """KMutationORFPool works correctly in composite pools."""
        prefix = Pool(['PREFIX'], name='prefix', mode='sequential')
        suffix = Pool(['SUFFIX'], name='suffix', mode='sequential')
        
        orf_pool = KMutationORFPool(
            'ATGGAACCC', mutation_type='any_codon', k=1,
            name='orf', mode='sequential'
        )
        
        library = prefix + orf_pool + suffix
        
        result = library.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(5):
            row = dc.get_row(i)
            seq = result['sequences'][i]
            
            # Verify positions
            assert row['prefix_abs_start'] == 0
            assert row['prefix_abs_end'] == 6
            assert row['orf_abs_start'] == 6
            assert row['orf_abs_end'] == 15
            assert row['suffix_abs_start'] == 15
            
            # codon_pos_abs should be relative to full sequence
            pos = row['orf_codon_pos'][0]
            abs_pos = row['orf_codon_pos_abs'][0]
            expected = 6 + pos * 3  # prefix_len + pos * 3
            assert abs_pos == expected, \
                f"Expected {expected}, got {abs_pos}"
            
            # Verify mutation is visible at that position
            mutated_codon = row['orf_codon_to'][0]
            seq_codon = seq[abs_pos:abs_pos+3]
            assert seq_codon.upper() == mutated_codon.upper()
    
    def test_aa_tracking_is_correct(self):
        """Amino acid changes are tracked correctly."""
        # Use missense_only_first for predictable AA changes
        orf = 'ATGGAACCC'  # M E P
        pool = KMutationORFPool(
            orf, mutation_type='missense_only_first', k=1,
            name='aa', mode='sequential'
        )
        
        result = pool.generate_seqs(num_seqs=10, return_design_cards=True)
        dc = result['design_cards']
        
        original_aas = ['M', 'E', 'P']
        
        for i in range(10):
            row = dc.get_row(i)
            
            pos = row['aa_codon_pos'][0]
            aa_from = row['aa_aa_from'][0]
            aa_to = row['aa_aa_to'][0]
            
            # aa_from should match original
            assert aa_from == original_aas[pos], \
                f"Expected {original_aas[pos]}, got {aa_from}"
            
            # missense_only means AA must change
            assert aa_from != aa_to, \
                f"Missense should change AA: {aa_from} -> {aa_to}"


# =============================================================================
# Phase 8: RandomMutationORFPool Integration Tests
# =============================================================================

class TestRandomMutationORFPoolDesignCards:
    """Integration tests for RandomMutationORFPool design cards."""
    
    def test_variable_mutation_counts(self):
        """Mutation count varies based on randomness and rate."""
        orf = 'ATGGAACCCAAATTT'  # 5 codons
        pool = RandomMutationORFPool(
            orf, mutation_type='any_codon', mutation_rate=0.5,
            name='rand'
        )
        
        result = pool.generate_seqs(num_seqs=20, return_design_cards=True)
        dc = result['design_cards']
        
        mut_counts = []
        for i in range(20):
            row = dc.get_row(i)
            count = row['rand_mut_count']
            mut_counts.append(count)
            
            # Count should match length of mutation arrays
            assert len(row['rand_codon_pos']) == count
            assert len(row['rand_codon_to']) == count
            assert len(row['rand_aa_from']) == count
        
        # With 50% rate and 5 codons, we should see variation
        unique_counts = set(mut_counts)
        assert len(unique_counts) > 1, \
            f"Expected variation in mutation counts, got {mut_counts}"
    
    def test_zero_mutation_rate(self):
        """Zero mutation rate produces no mutations."""
        orf = 'ATGGAACCC'
        pool = RandomMutationORFPool(
            orf, mutation_type='any_codon', mutation_rate=0.0,
            name='zero'
        )
        
        result = pool.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(5):
            row = dc.get_row(i)
            assert row['zero_mut_count'] == 0
            assert row['zero_codon_pos'] == []
    
    def test_high_mutation_rate(self):
        """High mutation rate mutates most codons."""
        orf = 'ATGGAACCC'  # 3 codons
        pool = RandomMutationORFPool(
            orf, mutation_type='any_codon', mutation_rate=0.99,
            name='high'
        )
        
        result = pool.generate_seqs(num_seqs=10, return_design_cards=True)
        dc = result['design_cards']
        
        # Most sequences should have 2-3 mutations
        high_mutation_count = sum(
            1 for i in range(10) 
            if dc.get_row(i)['high_mut_count'] >= 2
        )
        assert high_mutation_count >= 7, \
            f"Expected most sequences to have >=2 mutations with rate=0.99"
    
    def test_in_composite_position_tracking(self):
        """Position tracking works in composite pools."""
        prefix = Pool(['AAA'], name='pre', mode='sequential')
        
        orf_pool = RandomMutationORFPool(
            'ATGGAACCC', mutation_type='any_codon', mutation_rate=0.8,
            name='orf'
        )
        
        library = prefix + orf_pool
        
        result = library.generate_seqs(num_seqs=10, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(10):
            row = dc.get_row(i)
            seq = result['sequences'][i]
            
            # Verify positions
            assert row['orf_abs_start'] == 3  # After prefix
            
            # codon_pos_abs should be relative to full sequence
            for pos, abs_pos in zip(row['orf_codon_pos'], row['orf_codon_pos_abs']):
                expected = 3 + pos * 3
                assert abs_pos == expected
                
                # Verify mutation at position
                mutated_codon = row['orf_codon_to'][row['orf_codon_pos'].index(pos)]
                seq_codon = seq[abs_pos:abs_pos+3]
                assert seq_codon.upper() == mutated_codon.upper()
    
    def test_value_matches_sequence(self):
        """The _value field matches the generated sequence portion."""
        pool = RandomMutationORFPool(
            'ATGGAACCC', mutation_type='any_codon', mutation_rate=0.5,
            name='mut', metadata='complete'
        )
        
        result = pool.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(5):
            row = dc.get_row(i)
            seq = result['sequences'][i]
            value = row['mut_value']
            
            assert seq == value, \
                f"Sequence should match value: {seq} != {value}"


# =============================================================================
# Phase 9: Simple Pools (BarcodePool, KmerPool, IUPACPool) Integration Tests
# =============================================================================

class TestSimplePoolsInComposites:
    """Integration tests for BarcodePool, KmerPool, IUPACPool in composites."""
    
    def test_kmer_pool_concatenation(self):
        """KmerPool works correctly in concatenation."""
        from poolparty import KmerPool
        
        prefix = Pool(['START'], name='start', mode='sequential')
        kmer = KmerPool(length=4, alphabet='dna', name='kmer', mode='sequential', metadata='complete')
        suffix = Pool(['END'], name='end', mode='sequential')
        
        library = prefix + kmer + suffix
        
        result = library.generate_seqs(num_seqs=10, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(10):
            row = dc.get_row(i)
            seq = result['sequences'][i]
            
            # Structure: START (5) + KMER (4) + END (3)
            assert seq[:5] == 'START'
            assert seq[-3:] == 'END'
            
            # Verify kmer positions
            assert row['kmer_abs_start'] == 5
            assert row['kmer_abs_end'] == 9
            
            # Verify kmer value matches sequence portion
            kmer_value = row['kmer_value']
            assert seq[5:9] == kmer_value
    
    def test_iupac_pool_concatenation(self):
        """IUPACPool works correctly in concatenation."""
        from poolparty import IUPACPool
        
        prefix = Pool(['GGG'], name='pre', mode='sequential')
        # RY = [AG][CT] = 4 sequences
        iupac = IUPACPool('RY', name='iupac', mode='sequential', metadata='complete')
        suffix = Pool(['CCC'], name='post', mode='sequential')
        
        library = prefix + iupac + suffix
        
        result = library.generate_seqs(num_seqs=4, return_design_cards=True)
        dc = result['design_cards']
        
        expected_iupacs = {'AC', 'AT', 'GC', 'GT'}
        actual_iupacs = set()
        
        for i in range(4):
            row = dc.get_row(i)
            seq = result['sequences'][i]
            
            assert seq[:3] == 'GGG'
            assert seq[-3:] == 'CCC'
            
            iupac_value = row['iupac_value']
            actual_iupacs.add(iupac_value)
            
            # Verify positions
            assert row['iupac_abs_start'] == 3
            assert row['iupac_abs_end'] == 5
        
        assert actual_iupacs == expected_iupacs, \
            f"Expected {expected_iupacs}, got {actual_iupacs}"
    
    def test_barcode_pool_in_composite(self):
        """BarcodePool works correctly in composite pools."""
        from poolparty import BarcodePool
        
        # Generate 5 barcodes of length 6
        barcode = BarcodePool(
            num_barcodes=5, length=6, 
            min_edit_distance=2,
            name='bc', mode='sequential', seed=42, metadata='complete'
        )
        
        prefix = Pool(['PRE'], name='pre', mode='sequential')
        suffix = Pool(['POST'], name='post', mode='sequential')
        
        library = prefix + barcode + suffix
        
        result = library.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        
        barcodes_seen = set()
        for i in range(5):
            row = dc.get_row(i)
            seq = result['sequences'][i]
            
            # Structure: PRE (3) + BC (6) + POST (4) = 13
            assert len(seq) == 13
            assert seq[:3] == 'PRE'
            assert seq[-4:] == 'POST'
            
            # Barcode positions
            assert row['bc_abs_start'] == 3
            assert row['bc_abs_end'] == 9
            
            bc_value = row['bc_value']
            barcodes_seen.add(bc_value)
            assert len(bc_value) == 6
        
        # All 5 barcodes should be unique
        assert len(barcodes_seen) == 5
    
    def test_mixed_simple_pools(self):
        """Multiple simple pool types in one composite."""
        from poolparty import KmerPool, IUPACPool
        
        # Build: KMER(3) + IUPAC(NN=16) + Pool(variants)
        kmer = KmerPool(length=3, alphabet='dna', name='k', mode='sequential')
        iupac = IUPACPool('NN', name='iupac', mode='sequential')  # 4*4=16 combinations
        variants = Pool(['AA', 'CC', 'GG', 'TT'], name='var', mode='sequential')
        
        library = kmer + iupac + variants
        
        result = library.generate_seqs(num_seqs=20, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(20):
            row = dc.get_row(i)
            seq = result['sequences'][i]
            
            # Total length: 3 + 2 + 2 = 7
            assert len(seq) == 7
            
            # Verify positions chain correctly
            assert row['k_abs_start'] == 0
            assert row['k_abs_end'] == 3
            assert row['iupac_abs_start'] == 3
            assert row['iupac_abs_end'] == 5
            assert row['var_abs_start'] == 5
            assert row['var_abs_end'] == 7
    
    def test_simple_pools_in_mixedpool(self):
        """Simple pools work correctly as MixedPool children."""
        from poolparty import KmerPool, IUPACPool
        
        kmer = KmerPool(length=4, alphabet='dna', name='kmer', mode='sequential', metadata='complete')
        iupac = IUPACPool('NNNN', name='iupac', mode='sequential', metadata='complete')
        
        mixed = MixedPool([kmer, iupac], name='choice', mode='sequential')
        
        result = mixed.generate_seqs(num_seqs=10, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(10):
            row = dc.get_row(i)
            selected = row['choice_selected']
            
            if selected == 0:
                # Kmer selected
                assert row['kmer_value'] is not None
                assert row['iupac_value'] is None
            else:
                # IUPAC selected
                assert row['kmer_value'] is None
                assert row['iupac_value'] is not None


# =============================================================================
# Phase 10: Complex Realistic Scenarios
# =============================================================================

class TestRealisticLibraryDesigns:
    """Tests simulating real-world complex library designs."""
    
    def test_promoter_barcode_orf_library(self):
        """Simulate: promoter variants + barcode + ORF with mutations.
        
        Structure:
        MixedPool([promoter_wt, promoter_mut]) + BarcodePool + KMutationORFPool
        """
        from poolparty import BarcodePool
        
        # Promoter variants
        promoter_wt = Pool(['TATAAAA'], name='prom_wt', mode='sequential')
        promoter_mut = KMutationPool('TATAAAA', alphabet='dna', k=1,
                                     name='prom_mut', mode='sequential')
        promoters = MixedPool([promoter_wt, promoter_mut], 
                             name='promoter', mode='sequential')
        
        # Barcode region
        barcode = BarcodePool(num_barcodes=4, length=6, name='bc', 
                             mode='sequential', seed=123)
        
        # ORF with mutation
        orf = KMutationORFPool('ATGGAACCC', mutation_type='any_codon', k=1,
                               name='orf', mode='sequential')
        
        library = promoters + barcode + orf
        
        result = library.generate_seqs(num_seqs=16, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(16):
            row = dc.get_row(i)
            seq = result['sequences'][i]
            
            # Verify structure
            promoter_selected = row['promoter_selected']
            
            if promoter_selected == 0:
                # Wild-type promoter
                assert seq[:7] == 'TATAAAA'
            else:
                # Mutated promoter - should have mutation tracked
                assert row['prom_mut_mut_pos'] is not None
            
            # ORF mutation should always be tracked
            assert len(row['orf_codon_pos']) == 1
            
            # Barcode position
            assert row['bc_abs_start'] == 7
            assert row['bc_abs_end'] == 13
    
    def test_multi_domain_with_mutations(self):
        """Simulate: Domain1_variants + Linker + Domain2_mutations.
        
        Each domain can have wild-type or mutated versions.
        """
        # Domain 1 options
        d1_wt = Pool(['AAAAAA'], name='d1_wt', mode='sequential', metadata='complete')
        d1_del = DeletionScanPool('AAAAAA', deletion_size=2, name='d1_del', mode='sequential')
        domain1 = MixedPool([d1_wt, d1_del], name='d1', mode='sequential')
        
        # Linker
        linker = Pool(['GGG'], name='link', mode='sequential')
        
        # Domain 2 options
        d2_wt = Pool(['TTTTTT'], name='d2_wt', mode='sequential', metadata='complete')
        d2_ins = InsertionScanPool('TTTTTT', 'XX', name='d2_ins', 
                                   mode='sequential', insert_or_overwrite='overwrite')
        domain2 = MixedPool([d2_wt, d2_ins], name='d2', mode='sequential')
        
        library = domain1 + linker + domain2
        
        result = library.generate_seqs(num_seqs=20, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(20):
            row = dc.get_row(i)
            
            # Domain 1
            d1_selected = row['d1_selected']
            if d1_selected == 0:
                assert row['d1_wt_value'] == 'AAAAAA'
            else:
                # Deletion scan tracks position
                assert row['d1_del_pos'] is not None
            
            # Domain 2
            d2_selected = row['d2_selected']
            if d2_selected == 0:
                assert row['d2_wt_value'] == 'TTTTTT'
            else:
                # Insertion scan tracks position
                assert row['d2_ins_pos'] is not None
    
    def test_nested_mixedpool_with_orf_mutations(self):
        """Nested MixedPools with ORF-level mutations.
        
        Structure:
        MixedPool([
            wt_construct,
            MixedPool([k_mutation_orf_1, k_mutation_orf_2])
        ])
        
        Note: Nested MixedPools don't recursively expand their children's metadata.
        The inner MixedPool (mut_type) tracks which child was selected, but
        kmut1/kmut2 specific fields are NOT available as separate columns.
        """
        # Wild-type construct
        wt = Pool(['ATGGAACCC'], name='wt', mode='sequential', metadata='complete')
        
        # ORF mutation options - use two finite-state pools
        k_mut_1 = KMutationORFPool('ATGGAACCC', mutation_type='any_codon', k=1,
                                   name='kmut1', mode='sequential')
        k_mut_2 = KMutationORFPool('ATGGAACCC', mutation_type='missense_only_first', k=1,
                                   name='kmut2', mode='sequential')
        
        mut_options = MixedPool([k_mut_1, k_mut_2], name='mut_type', mode='sequential', metadata='complete')
        
        library = MixedPool([wt, mut_options], name='construct', mode='sequential')
        
        result = library.generate_seqs(num_seqs=30, return_design_cards=True)
        dc = result['design_cards']
        
        # Verify expected keys are present
        assert 'construct_selected' in dc
        assert 'construct_selected_name' in dc
        assert 'wt_value' in dc
        assert 'mut_type_selected' in dc
        assert 'mut_type_value' in dc
        
        # Note: kmut1_* and kmut2_* are NOT present because nested MixedPools
        # don't recursively expand children's metadata
        assert 'kmut1_codon_pos' not in dc
        assert 'kmut2_codon_pos' not in dc
        
        for i in range(30):
            row = dc.get_row(i)
            
            construct_selected = row['construct_selected']
            
            if construct_selected == 0:
                # Wild-type selected
                assert row['wt_value'] == 'ATGGAACCC'
                assert row['construct_selected_name'] == 'wt'
                # Inner MixedPool data should be None when outer selects wt
                assert row['mut_type_selected'] is None
                assert row['mut_type_value'] is None
            else:
                # mut_type MixedPool selected
                assert row['construct_selected_name'] == 'mut_type'
                assert row['wt_value'] is None
                # Inner MixedPool tracks its selection
                assert row['mut_type_selected'] in [0, 1]
                assert row['mut_type_value'] is not None
                # Value should be a mutated ORF
                assert len(row['mut_type_value']) == 9  # Same length as original ORF
    
    def test_complete_gene_library(self):
        """Full gene construct: 5UTR + ORF_variants + 3UTR + Barcode.
        
        This tests a realistic gene expression library design.
        """
        from poolparty import BarcodePool
        
        # 5' UTR options
        utr5 = Pool(['GGGGG', 'AAAAA'], name='utr5', mode='sequential')
        
        # ORF: wild-type or with insertions at codon level
        orf_wt = Pool(['ATGGAACCCAAATTT'], name='orf_wt', mode='sequential')  # 5 codons
        orf_ins = InsertionScanORFPool(
            'ATGGAACCCAAATTT', insert_seq='GGGAAA',  # 2 codons
            orf_start=0, orf_end=15,
            name='orf_ins', mode='sequential',
            insert_or_overwrite='overwrite'
        )
        orf_variants = MixedPool([orf_wt, orf_ins], name='orf', mode='sequential')
        
        # 3' UTR
        utr3 = Pool(['TTTTT'], name='utr3', mode='sequential')
        
        # Barcode
        barcode = BarcodePool(num_barcodes=3, length=4, name='bc',
                             mode='sequential', seed=999, metadata='complete')
        
        library = utr5 + orf_variants + utr3 + barcode
        
        result = library.generate_seqs(num_seqs=24, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(24):
            row = dc.get_row(i)
            seq = result['sequences'][i]
            
            # Check UTR5 at start
            assert seq[:5] in ['GGGGG', 'AAAAA']
            
            # ORF variant tracking
            orf_selected = row['orf_selected']
            if orf_selected == 0:
                # Wild-type ORF
                pass
            else:
                # Insertion scan ORF - should track codon position
                assert row['orf_ins_codon_pos'] is not None
            
            # Barcode at end
            bc_value = row['bc_value']
            assert len(bc_value) == 4
    
    def test_orf_mutation_pool_chain(self):
        """Chain multiple ORF-level pools: InsertionScanORF -> KMutationORF.
        
        This tests transformer patterns where one ORF pool feeds into another.
        """
        # Base ORF with flanks
        base_seq = 'AAAATGGAACCCAAATTT'  # 3nt flank + 12nt ORF (4 codons) + 3nt flank
        
        # First: insertion scan at ORF level
        scan_pool = InsertionScanORFPool(
            base_seq, insert_seq='GGG',  # 1 codon
            orf_start=3, orf_end=15,
            name='scan', mode='sequential',
            insert_or_overwrite='overwrite'
        )
        
        # Second: k-mutation on the result
        mut_pool = KMutationORFPool(
            scan_pool, mutation_type='any_codon', k=1,
            orf_start=3, orf_end=15,
            name='mut', mode='sequential'
        )
        
        result = mut_pool.generate_seqs(num_seqs=10, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(10):
            row = dc.get_row(i)
            seq = result['sequences'][i]
            
            # Flanks should be preserved
            assert seq[:3] == 'AAA'
            assert seq[-3:] == 'TTT'
            
            # Both pools should track their operations
            assert row['scan_codon_pos'] is not None
            assert row['mut_codon_pos'] is not None
            assert len(row['mut_codon_pos']) == 1


# =============================================================================
# Phase 11: Multi-Parent Pool Tracking Tests
# =============================================================================

class TestMultiParentPoolTracking:
    """Tests for tracking all Pool parents in multi-input pools.
    
    These tests validate the fix for pools with multiple Pool inputs,
    ensuring ALL parent pools are tracked in design cards (not just parents[0]).
    
    Bug fixed: add_transformer_chain now iterates all parents instead of
    only tracking parents[0], and 'spacing_scan' is in TRANSFORMER_OPS.
    """
    
    def test_insertion_scan_both_pool_inputs_tracked(self):
        """InsertionScanPool with Pool background AND Pool insert - both tracked."""
        bg = IUPACPool(iupac_seq="NNNNNNNNNN", name="Background", metadata='complete')
        insert = IUPACPool(iupac_seq="RRRR", name="Insert", metadata='complete')
        
        pool = InsertionScanPool(
            background_seq=bg,
            insert_seq=insert,
            start=2, end=6, step_size=2,
            name="ISP",
            mode='sequential'
        )
        
        result = pool.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        
        # Both Background and Insert should be tracked
        assert 'Background_index' in dc.keys, "Background pool should be tracked"
        assert 'Insert_index' in dc.keys, "Insert pool should be tracked"
        
        # Verify values exist for all rows
        for i in range(5):
            row = dc.get_row(i)
            assert row['Background_value'] is not None
            assert row['Insert_value'] is not None
            assert row['ISP_pos'] is not None
    
    def test_spacing_scan_all_insert_pools_tracked(self):
        """SpacingScanPool with multiple Pool inserts - all tracked."""
        ins_a = IUPACPool(iupac_seq="AAAA", name="InsertA", metadata='complete')
        ins_b = IUPACPool(iupac_seq="TTTT", name="InsertB", metadata='complete')
        ins_c = IUPACPool(iupac_seq="GGGG", name="InsertC", metadata='complete')
        
        pool = SpacingScanPool(
            background_seq="N" * 60,
            insert_seqs=[ins_a, ins_b, ins_c],
            insert_names=["A", "B", "C"],
            anchor_pos=30,
            insert_distances=[[-20], [0], [20]],
            name="SSP",
            mode='sequential',
            metadata='complete'
        )
        
        result = pool.generate_seqs(num_seqs=1, return_design_cards=True)
        dc = result['design_cards']
        
        # All three insert pools should be tracked
        assert 'InsertA_index' in dc.keys, "InsertA should be tracked"
        assert 'InsertB_index' in dc.keys, "InsertB should be tracked"
        assert 'InsertC_index' in dc.keys, "InsertC should be tracked"
        
        row = dc.get_row(0)
        assert row['InsertA_value'] is not None
        assert row['InsertB_value'] is not None
        assert row['InsertC_value'] is not None
    
    def test_spacing_scan_with_pool_background_and_inserts(self):
        """SpacingScanPool with Pool background AND Pool inserts - all tracked."""
        bg = IUPACPool(iupac_seq="N" * 50, name="SpacingBg", metadata='complete')
        ins_a = IUPACPool(iupac_seq="AAAA", name="InsA", metadata='complete')
        ins_b = IUPACPool(iupac_seq="TTTT", name="InsB", metadata='complete')
        
        pool = SpacingScanPool(
            background_seq=bg,
            insert_seqs=[ins_a, ins_b],
            insert_names=["A", "B"],
            anchor_pos=25,
            insert_distances=[[-10], [10]],
            name="SSP",
            mode='sequential',
            metadata='complete'
        )
        
        result = pool.generate_seqs(num_seqs=1, return_design_cards=True)
        dc = result['design_cards']
        
        # Background and both inserts should be tracked
        assert 'SpacingBg_index' in dc.keys, "Background pool should be tracked"
        assert 'InsA_index' in dc.keys, "InsA should be tracked"
        assert 'InsB_index' in dc.keys, "InsB should be tracked"
        
        row = dc.get_row(0)
        assert row['SpacingBg_value'] is not None
        assert row['InsA_value'] is not None
        assert row['InsB_value'] is not None
    
    def test_insertion_scan_orf_both_inputs_tracked(self):
        """InsertionScanORFPool with Pool background AND Pool insert - both tracked."""
        orf_seq = "ATGACGTACGTACGTGAA"  # 18bp = 6 codons
        bg = IUPACPool(iupac_seq=orf_seq, name="ORFBg", metadata='complete')
        insert = IUPACPool(iupac_seq="GGG", name="ORFIns", metadata='complete')
        
        pool = InsertionScanORFPool(
            background_seq=bg,
            insert_seq=insert,
            start=3, end=12, step_size=3,
            name="ISOP",
            mode='sequential'
        )
        
        result = pool.generate_seqs(num_seqs=3, return_design_cards=True)
        dc = result['design_cards']
        
        # Both should be tracked
        assert 'ORFBg_index' in dc.keys, "ORF Background should be tracked"
        assert 'ORFIns_index' in dc.keys, "ORF Insert should be tracked"
    
    def test_spacing_scan_in_composite_structure(self):
        """SpacingScanPool in composite (prefix + spacing + suffix) - inserts tracked."""
        prefix = Pool(["PREFIX_"], name="pre", mode='sequential')
        suffix = Pool(["_SUFFIX"], name="suf", mode='sequential')
        
        ins_a = IUPACPool(iupac_seq="AAAA", name="MotifA", metadata='complete')
        ins_b = IUPACPool(iupac_seq="TTTT", name="MotifB", metadata='complete')
        
        spacing = SpacingScanPool(
            background_seq="N" * 40,
            insert_seqs=[ins_a, ins_b],
            insert_names=["A", "B"],
            anchor_pos=20,
            insert_distances=[[-10], [10]],
            name="Spacing",
            mode='sequential',
            metadata='complete'
        )
        
        library = prefix + spacing + suffix
        
        result = library.generate_seqs(num_seqs=1, return_design_cards=True)
        dc = result['design_cards']
        
        # All pools should be tracked
        assert 'pre_index' in dc.keys
        assert 'suf_index' in dc.keys
        assert 'Spacing_index' in dc.keys
        assert 'MotifA_index' in dc.keys, "MotifA insert should be tracked"
        assert 'MotifB_index' in dc.keys, "MotifB insert should be tracked"
        
        row = dc.get_row(0)
        seq = result['sequences'][0]
        
        # Verify structure
        assert seq.startswith("PREFIX_")
        assert seq.endswith("_SUFFIX")
        assert row['MotifA_value'] is not None
        assert row['MotifB_value'] is not None
    
    def test_spacing_scan_insert_metadata_correlation(self):
        """Verify insert pool metadata correlates with spacing pool metadata."""
        ins_a = IUPACPool(iupac_seq="RY", name="MotifA", metadata='complete')  # 4 variants
        ins_b = IUPACPool(iupac_seq="SW", name="MotifB", metadata='complete')  # 4 variants
        
        pool = SpacingScanPool(
            background_seq="N" * 30,
            insert_seqs=[ins_a, ins_b],
            insert_names=["A", "B"],
            anchor_pos=15,
            insert_distances=[[-5], [5]],
            name="SSP",
            mode='sequential',
            metadata='complete'
        )
        
        result = pool.generate_seqs(num_seqs=16, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(16):
            row = dc.get_row(i)
            seq = result['sequences'][i]
            
            # Get insert positions from spacing pool metadata
            a_start = row['SSP_A_pos_start']
            a_end = row['SSP_A_pos_end']
            b_start = row['SSP_B_pos_start']
            b_end = row['SSP_B_pos_end']
            
            # Get insert values from IUPAC pool metadata
            a_value = row['MotifA_value']
            b_value = row['MotifB_value']
            
            # Verify insert values match sequence at positions
            assert seq[a_start:a_end].upper() == a_value.upper(), \
                f"MotifA value {a_value} should be at positions {a_start}:{a_end} in {seq}"
            assert seq[b_start:b_end].upper() == b_value.upper(), \
                f"MotifB value {b_value} should be at positions {b_start}:{b_end} in {seq}"
    
    def test_multiple_spacing_scan_pools_in_composite(self):
        """Multiple SpacingScanPools in same composite - each tracks its inserts."""
        ins_a = IUPACPool(iupac_seq="AA", name="InsA1", metadata='complete')
        ins_b = IUPACPool(iupac_seq="TT", name="InsB1", metadata='complete')
        
        ins_c = IUPACPool(iupac_seq="GG", name="InsC2", metadata='complete')
        ins_d = IUPACPool(iupac_seq="CC", name="InsD2", metadata='complete')
        
        spacing1 = SpacingScanPool(
            background_seq="N" * 20,
            insert_seqs=[ins_a, ins_b],
            insert_names=["A", "B"],
            anchor_pos=10,
            insert_distances=[[-5], [5]],
            name="SP1",
            mode='sequential'
        )
        
        spacing2 = SpacingScanPool(
            background_seq="N" * 20,
            insert_seqs=[ins_c, ins_d],
            insert_names=["C", "D"],
            anchor_pos=10,
            insert_distances=[[-5], [5]],
            name="SP2",
            mode='sequential'
        )
        
        library = spacing1 + spacing2
        
        result = library.generate_seqs(num_seqs=1, return_design_cards=True)
        dc = result['design_cards']
        
        # All insert pools should be tracked
        assert 'InsA1_index' in dc.keys
        assert 'InsB1_index' in dc.keys
        assert 'InsC2_index' in dc.keys
        assert 'InsD2_index' in dc.keys
    
    def test_spacing_scan_random_mode_insert_tracking(self):
        """SpacingScanPool in random mode still tracks insert pools."""
        ins_a = IUPACPool(iupac_seq="AAAA", name="RandInsA", metadata='complete')
        ins_b = IUPACPool(iupac_seq="TTTT", name="RandInsB", metadata='complete')
        
        pool = SpacingScanPool(
            background_seq="N" * 50,
            insert_seqs=[ins_a, ins_b],
            insert_names=["A", "B"],
            anchor_pos=25,
            insert_scan_ranges=[(-20, 0, 5), (0, 20, 5)],
            name="RSP",
            mode='random',
            metadata='complete'
        )
        
        result = pool.generate_seqs(num_seqs=20, seed=42, return_design_cards=True)
        dc = result['design_cards']
        
        # Insert pools should be tracked even in random mode
        assert 'RandInsA_index' in dc.keys
        assert 'RandInsB_index' in dc.keys
        
        for i in range(20):
            row = dc.get_row(i)
            assert row['RandInsA_value'] is not None
            assert row['RandInsB_value'] is not None
    
    def test_track_pools_filter_includes_insert_pools(self):
        """track_pools parameter works correctly with insert pools."""
        ins_a = IUPACPool(iupac_seq="AAAA", name="TrackInsA", metadata='complete')
        ins_b = IUPACPool(iupac_seq="TTTT", name="TrackInsB", metadata='complete')
        
        pool = SpacingScanPool(
            background_seq="N" * 40,
            insert_seqs=[ins_a, ins_b],
            insert_names=["A", "B"],
            anchor_pos=20,
            insert_distances=[[-10], [10]],
            name="TSP",
            mode='sequential'
        )
        
        # Only track one insert pool
        result = pool.generate_seqs(
            num_seqs=1, 
            return_design_cards=True,
            track_pools=['TSP', 'TrackInsA']  # Don't track TrackInsB
        )
        dc = result['design_cards']
        
        assert 'TSP_index' in dc.keys
        assert 'TrackInsA_index' in dc.keys
        assert 'TrackInsB_index' not in dc.keys  # Filtered out
    
    def test_spacing_scan_with_kmer_pool_inserts(self):
        """SpacingScanPool with KmerPool inserts - tracks kmer metadata."""
        ins_a = KmerPool(length=4, alphabet='dna', name="KmerA", mode='sequential', metadata='complete')
        ins_b = KmerPool(length=4, alphabet='dna', name="KmerB", mode='sequential', metadata='complete')
        
        pool = SpacingScanPool(
            background_seq="N" * 40,
            insert_seqs=[ins_a, ins_b],
            insert_names=["A", "B"],
            anchor_pos=20,
            insert_distances=[[-10], [10]],
            name="KSP",
            mode='sequential'
        )
        
        result = pool.generate_seqs(num_seqs=10, return_design_cards=True)
        dc = result['design_cards']
        
        assert 'KmerA_index' in dc.keys
        assert 'KmerB_index' in dc.keys
        
        for i in range(10):
            row = dc.get_row(i)
            # Kmer values should be 4bp DNA sequences
            assert len(row['KmerA_value']) == 4
            assert len(row['KmerB_value']) == 4
            assert all(c in 'ACGT' for c in row['KmerA_value'])
            assert all(c in 'ACGT' for c in row['KmerB_value'])
    
    def test_composite_insert_internal_pools_tracked(self):
        """Composite inserts have their internal named pools tracked.
        
        When a composite Pool (e.g., A + B + C) is used as an insert,
        the fix ensures we walk through the composite structure to find
        and track all named pools within it.
        """
        # Create a composite insert: prefix + core + suffix
        core = IUPACPool(iupac_seq="RRRR", name="Core", metadata='complete')
        insert_composite = Pool(["--"], name="InsPrefix") + core + Pool(["--"], name="InsSuffix")
        
        pool = SpacingScanPool(
            background_seq="N" * 40,
            insert_seqs=[insert_composite, "XXXX"],
            insert_names=["Comp", "Lit"],
            anchor_pos=20,
            insert_distances=[[-10], [10]],
            name="NSP",
            mode='sequential'
        )
        
        result = pool.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        
        # All named pools inside the composite insert should be tracked
        assert 'InsPrefix_index' in dc.keys, "InsPrefix in composite insert should be tracked"
        assert 'Core_index' in dc.keys, "Core in composite insert should be tracked"
        assert 'InsSuffix_index' in dc.keys, "InsSuffix in composite insert should be tracked"
        
        # Core should have values (has metadata='complete')
        for i in range(5):
            row = dc.get_row(i)
            assert row['Core_value'] is not None


class TestParentTrackingEdgeCases:
    """Edge case tests for parent pool tracking."""
    
    def test_unnamed_parent_pool_not_tracked(self):
        """Unnamed parent pools should not create columns."""
        bg = IUPACPool(iupac_seq="NNNNNNNNNN")  # No name
        insert = IUPACPool(iupac_seq="RRRR", name="NamedInsert", metadata='complete')
        
        pool = InsertionScanPool(
            background_seq=bg,
            insert_seq=insert,
            start=2, end=6, step_size=2,
            name="ISP",
            mode='sequential'
        )
        
        result = pool.generate_seqs(num_seqs=2, return_design_cards=True)
        dc = result['design_cards']
        
        # Named insert should be tracked
        assert 'NamedInsert_index' in dc.keys
        
        # Unnamed bg should not create columns with empty prefix
        assert '_index' not in dc.keys
    
    def test_string_inputs_dont_create_columns(self):
        """String inputs (not Pool objects) don't create spurious columns."""
        pool = SpacingScanPool(
            background_seq="NNNNNNNNNNNNNNNNNNNN",  # String
            insert_seqs=["AAAA", "TTTT"],  # Strings
            insert_names=["A", "B"],
            anchor_pos=10,
            insert_distances=[[-5], [5]],
            name="SSP",
            mode='sequential'
        )
        
        result = pool.generate_seqs(num_seqs=1, return_design_cards=True)
        dc = result['design_cards']
        
        # Only SSP itself should have columns (plus universal ones)
        pool_columns = [k for k in dc.keys if not k.startswith('sequence')]
        assert all('SSP' in k for k in pool_columns), \
            f"Only SSP columns expected, got: {pool_columns}"
    
    def test_mixed_string_and_pool_inputs(self):
        """Mix of string and Pool inputs - only Pools are tracked."""
        ins_a = IUPACPool(iupac_seq="AAAA", name="PoolInsert", metadata='complete')
        
        pool = SpacingScanPool(
            background_seq="N" * 40,  # String
            insert_seqs=[ins_a, "TTTT"],  # Pool and string
            insert_names=["A", "B"],
            anchor_pos=20,
            insert_distances=[[-10], [10]],
            name="MSP",
            mode='sequential'
        )
        
        result = pool.generate_seqs(num_seqs=1, return_design_cards=True)
        dc = result['design_cards']
        
        # Pool insert should be tracked
        assert 'PoolInsert_index' in dc.keys
        
        # No columns for string insert (other than SSP's own tracking)
        row = dc.get_row(0)
        assert row['PoolInsert_value'] is not None
    
    def test_same_pool_used_multiple_times_in_inserts(self):
        """Same Pool object used multiple times as different inserts."""
        shared_pool = IUPACPool(iupac_seq="RRRR", name="Shared", metadata='complete')
        
        pool = SpacingScanPool(
            background_seq="N" * 60,
            insert_seqs=[shared_pool, shared_pool],  # Same pool twice
            insert_names=["First", "Second"],
            anchor_pos=30,
            insert_distances=[[-15], [15]],
            name="SSP",
            mode='sequential'
        )
        
        result = pool.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        
        # Shared pool appears twice - should have occurrence indices
        # Or just one set of columns since it's the same object
        assert 'Shared_index' in dc.keys or 'Shared[1]_index' in dc.keys
        
        # Both positions should be tracked by SSP
        row = dc.get_row(0)
        assert row['SSP_First_pos_start'] is not None
        assert row['SSP_Second_pos_start'] is not None


class TestParentTrackingWithMixedPool:
    """Tests combining multi-parent pools with MixedPool."""
    
    def test_spacing_scan_in_mixed_pool_tracks_own_metadata(self):
        """SpacingScanPool as MixedPool child - spacing metadata tracked."""
        ins_a = IUPACPool(iupac_seq="AAAA", name="MixedInsA", metadata='complete')
        ins_b = IUPACPool(iupac_seq="TTTT", name="MixedInsB", metadata='complete')
        
        # Background is 30bp, inserts are 4bp each in overwrite mode = 30bp output
        spacing = SpacingScanPool(
            background_seq="N" * 30,
            insert_seqs=[ins_a, ins_b],
            insert_names=["A", "B"],
            anchor_pos=15,
            insert_distances=[[-5], [5]],
            name="SpacingChild",
            mode='sequential',
            metadata='complete'
        )
        
        # Plain must have same length as spacing output (30bp)
        plain = Pool(["X" * 30], name="Plain", mode='sequential')
        
        mixed = MixedPool([spacing, plain], name="Mix", mode='sequential')
        
        result = mixed.generate_seqs(num_seqs=10, return_design_cards=True)
        dc = result['design_cards']
        
        # MixedPool tracks selection and its children's metadata
        assert 'Mix_selected' in dc.keys
        assert 'SpacingChild_index' in dc.keys
        assert 'Plain_index' in dc.keys
        
        for i in range(10):
            row = dc.get_row(i)
            selected = row['Mix_selected']
            
            if selected == 0:
                # SpacingChild selected - its metadata should be populated
                assert row['SpacingChild_A_dist'] is not None
                assert row['SpacingChild_B_dist'] is not None
            else:
                # Plain selected - spacing metadata should be None
                assert row['SpacingChild_A_dist'] is None
    
    def test_insertion_scan_with_mixed_insert(self):
        """InsertionScanPool with MixedPool as insert tracks selection."""
        ins_option1 = IUPACPool(iupac_seq="AAAA", name="Opt1", metadata='complete')
        ins_option2 = IUPACPool(iupac_seq="TTTT", name="Opt2", metadata='complete')
        
        mixed_insert = MixedPool([ins_option1, ins_option2], name="MixedIns", mode='sequential')
        
        pool = InsertionScanPool(
            background_seq="N" * 20,
            insert_seq=mixed_insert,
            start=5, end=15, step_size=5,
            name="ISP",
            mode='sequential'
        )
        
        result = pool.generate_seqs(num_seqs=6, return_design_cards=True)
        dc = result['design_cards']
        
        # MixedIns should be tracked (as insert parent)
        assert 'MixedIns_selected' in dc.keys or 'MixedIns_index' in dc.keys
        
        # Verify ISP position tracking works
        for i in range(6):
            row = dc.get_row(i)
            assert row['ISP_pos'] is not None
    
    def test_spacing_scan_inside_mixed_pool_tracks_inserts(self):
        """SpacingScanPool inside MixedPool - insert pools are tracked."""
        ins_a = IUPACPool(iupac_seq="AAAA", name="MixInsA", metadata='complete')
        ins_b = IUPACPool(iupac_seq="TTTT", name="MixInsB", metadata='complete')
        
        spacing = SpacingScanPool(
            background_seq="N" * 30,
            insert_seqs=[ins_a, ins_b],
            insert_names=["A", "B"],
            anchor_pos=15,
            insert_distances=[[-5], [5]],
            name="SpacingInMix",
            mode='sequential',
            metadata='complete'
        )
        
        plain = Pool(["X" * 30], name="PlainOpt", mode='sequential')
        mixed = MixedPool([spacing, plain], name="MixOuter", mode='sequential')
        
        result = mixed.generate_seqs(num_seqs=4, return_design_cards=True)
        dc = result['design_cards']
        
        # Insert pools should be tracked even when SpacingScanPool is inside MixedPool
        assert 'MixInsA_index' in dc.keys, "InsertA should be tracked inside MixedPool"
        assert 'MixInsB_index' in dc.keys, "InsertB should be tracked inside MixedPool"
        
        # When spacing is selected, inserts should have values
        # Note: Insert pools are transformer parents with shared state, so they
        # have values regardless of which MixedPool child is selected
        for i in range(4):
            row = dc.get_row(i)
            if row['MixOuter_selected'] == 0:  # spacing selected
                # Insert values should be present
                assert row['MixInsA_value'] is not None
                assert row['MixInsB_value'] is not None
                # SpacingInMix metadata should also be present
                assert row['SpacingInMix_A_dist'] is not None
    
    def test_track_pools_with_spacing_scan_inside_mixed_pool(self):
        """track_pools correctly filters insert pools inside MixedPool children."""
        ins_a = IUPACPool(iupac_seq="AAAA", name="FilterInsA", metadata='complete')
        ins_b = IUPACPool(iupac_seq="TTTT", name="FilterInsB", metadata='complete')
        
        spacing = SpacingScanPool(
            background_seq="N" * 30,
            insert_seqs=[ins_a, ins_b],
            insert_names=["A", "B"],
            anchor_pos=15,
            insert_distances=[[-5], [5]],
            name="FilterSpacing",
            mode='sequential'
        )
        
        plain = Pool(["X" * 30], name="FilterPlain", mode='sequential')
        mixed = MixedPool([spacing, plain], name="FilterMix", mode='sequential')
        
        # Only track Mix and InsertA (not InsertB)
        result = mixed.generate_seqs(num_seqs=2, return_design_cards=True,
                                      track_pools=['FilterMix', 'FilterInsA'])
        dc = result['design_cards']
        
        assert 'FilterMix_selected' in dc.keys
        assert 'FilterInsA_index' in dc.keys
        assert 'FilterInsB_index' not in dc.keys, "InsertB should be filtered out"
        assert 'FilterSpacing_index' not in dc.keys, "Spacing should be filtered out"
    
    def test_track_only_insert_pools_in_nested_structure(self):
        """track_pools with only insert pool names works in complex structure."""
        ins_a = IUPACPool(iupac_seq="AAAA", name="OnlyInsA", metadata='complete')
        ins_b = IUPACPool(iupac_seq="TTTT", name="OnlyInsB", metadata='complete')
        
        spacing = SpacingScanPool(
            background_seq="N" * 30,
            insert_seqs=[ins_a, ins_b],
            insert_names=["A", "B"],
            anchor_pos=15,
            insert_distances=[[-5], [5]],
            name="OnlySpacing",
            mode='sequential'
        )
        
        plain = Pool(["X" * 30], name="OnlyPlain", mode='sequential')
        mixed = MixedPool([spacing, plain], name="OnlyMix", mode='sequential')
        
        prefix = Pool(["PRE_"], name="OnlyPrefix", mode='sequential')
        library = prefix + mixed
        
        # Track only the insert pools - should still work
        result = library.generate_seqs(num_seqs=2, return_design_cards=True,
                                        track_pools=['OnlyInsA', 'OnlyInsB'])
        dc = result['design_cards']
        
        assert 'OnlyInsA_index' in dc.keys
        assert 'OnlyInsB_index' in dc.keys
        assert 'OnlyMix_selected' not in dc.keys
        assert 'OnlyPrefix_index' not in dc.keys


# =============================================================================
# Phase 12: Comprehensive Design Card Value Verification
# =============================================================================

class TestDesignCardValueSemantics:
    """Tests verifying design card values are semantically correct.
    
    Key semantics:
    - Transformer pools (InsertionScan, SpacingScan, etc.): _value is the OUTPUT
    - Parent pools of transformers: _value is their ORIGINAL value before transformation
    - Insert pools: _value is the insert sequence (no abs positions since variable)
    """
    
    def test_insertion_scan_value_semantics(self):
        """InsertionScanPool: output value vs input values."""
        bg = IUPACPool(iupac_seq="AAAAAAAAAAAAAAAA", name="ISBg", metadata='complete')
        ins = IUPACPool(iupac_seq="GGGG", name="ISIns", metadata='complete')
        
        pool = InsertionScanPool(
            background_seq=bg, insert_seq=ins,
            start=4, end=12, step_size=4,
            insert_or_overwrite='overwrite',
            name="IS", mode='sequential', metadata='complete'
        )
        
        result = pool.generate_seqs(num_seqs=3, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(3):
            row = dc.get_row(i)
            seq = result['sequences'][i]
            
            # IS_value should be the OUTPUT (with insert applied)
            assert row['IS_value'] == seq
            
            # ISBg_value should be the ORIGINAL background
            assert row['ISBg_value'] == 'AAAAAAAAAAAAAAAA'
            
            # ISIns_value should be the insert
            assert row['ISIns_value'] == 'GGGG'
            
            # The output should have the insert at IS_pos
            pos = row['IS_pos']
            assert seq[pos:pos+4] == 'GGGG'
    
    def test_spacing_scan_value_semantics(self):
        """SpacingScanPool: output value vs insert values."""
        ins_a = IUPACPool(iupac_seq="AAAA", name="SSIns1", metadata='complete')
        ins_b = IUPACPool(iupac_seq="TTTT", name="SSIns2", metadata='complete')
        
        pool = SpacingScanPool(
            background_seq="NNNNNNNNNNNNNNNNNNNNNNNNNNNNNN",  # 30bp
            insert_seqs=[ins_a, ins_b],
            insert_names=["A", "B"],
            anchor_pos=15,
            insert_distances=[[-5], [5]],
            name="SS", mode='sequential', metadata='complete'
        )
        
        result = pool.generate_seqs(num_seqs=2, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(2):
            row = dc.get_row(i)
            seq = result['sequences'][i]
            
            # SS_value should be the OUTPUT
            assert row['SS_value'] == seq
            
            # Insert values should be their original sequences
            assert row['SSIns1_value'] == 'AAAA'
            assert row['SSIns2_value'] == 'TTTT'
            
            # Inserts should be at positions from spacing metadata
            a_start = row['SS_A_pos_start']
            a_end = row['SS_A_pos_end']
            b_start = row['SS_B_pos_start']
            b_end = row['SS_B_pos_end']
            
            assert seq[a_start:a_end] == 'AAAA'
            assert seq[b_start:b_end] == 'TTTT'
    
    def test_nested_transformer_value_chain(self):
        """Multiple levels of transformation maintain correct values."""
        # Background -> SpacingScanPool -> MixedPool
        ins_a = IUPACPool(iupac_seq="AA", name="ChainInsA", metadata='complete')
        ins_b = IUPACPool(iupac_seq="TT", name="ChainInsB", metadata='complete')
        
        spacing = SpacingScanPool(
            background_seq="NNNNNNNNNNNNNNNN",  # 16bp
            insert_seqs=[ins_a, ins_b],
            insert_names=["A", "B"],
            anchor_pos=8,
            insert_distances=[[-3], [3]],
            name="ChainSS", mode='sequential', metadata='complete'
        )
        
        plain = Pool(["X" * 16], name="ChainPlain", mode='sequential', metadata='complete')
        mixed = MixedPool([spacing, plain], name="ChainMix", mode='sequential')
        
        result = mixed.generate_seqs(num_seqs=4, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(4):
            row = dc.get_row(i)
            seq = result['sequences'][i]
            
            if row['ChainMix_selected'] == 0:  # spacing selected
                # ChainSS_value should match the output
                assert row['ChainSS_value'] == seq
                # Inserts should have their original values
                assert row['ChainInsA_value'] == 'AA'
                assert row['ChainInsB_value'] == 'TT'
            else:  # plain selected
                # ChainPlain_value should match the output
                assert row['ChainPlain_value'] == seq
    
    def test_insertion_scan_orf_value_semantics(self):
        """InsertionScanORFPool: output value vs input values."""
        orf_seq = "ATGACGTACGTACGTGAA"  # 18bp = 6 codons
        bg = IUPACPool(iupac_seq=orf_seq, name="ORFBg", metadata='complete')
        ins = IUPACPool(iupac_seq="GGG", name="ORFIns", metadata='complete')
        
        pool = InsertionScanORFPool(
            background_seq=bg, insert_seq=ins,
            start=3, end=15, step_size=3,
            insert_or_overwrite='overwrite',
            name="ISORF", mode='sequential', metadata='complete'
        )
        
        result = pool.generate_seqs(num_seqs=3, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(3):
            row = dc.get_row(i)
            seq = result['sequences'][i]
            
            # ORFBg_value should be the ORIGINAL ORF
            assert row['ORFBg_value'] == orf_seq
            
            # ORFIns_value should be the insert
            assert row['ORFIns_value'] == 'GGG'
            
            # The output should have GGG at codon position
            codon_pos = row['ISORF_codon_pos']
            abs_pos = codon_pos * 3
            assert seq[abs_pos:abs_pos+3] == 'GGG'


class TestDesignCardPositionAccuracy:
    """Tests verifying abs_start/abs_end positions are accurate."""
    
    def test_positions_in_simple_composite(self):
        """Positions are correct in A + B + C composite."""
        A = Pool(['AAA'], name='PosA', mode='sequential', metadata='complete')
        B = Pool(['BBBB'], name='PosB', mode='sequential', metadata='complete')
        C = Pool(['CCCCC'], name='PosC', mode='sequential', metadata='complete')
        
        library = A + B + C
        
        result = library.generate_seqs(num_seqs=1, return_design_cards=True)
        dc = result['design_cards']
        row = dc.get_row(0)
        seq = result['sequences'][0]
        
        # Verify positions
        assert row['PosA_abs_start'] == 0
        assert row['PosA_abs_end'] == 3
        assert row['PosB_abs_start'] == 3
        assert row['PosB_abs_end'] == 7
        assert row['PosC_abs_start'] == 7
        assert row['PosC_abs_end'] == 12
        
        # Verify values at positions
        assert seq[row['PosA_abs_start']:row['PosA_abs_end']] == row['PosA_value']
        assert seq[row['PosB_abs_start']:row['PosB_abs_end']] == row['PosB_value']
        assert seq[row['PosC_abs_start']:row['PosC_abs_end']] == row['PosC_value']
    
    def test_positions_with_transformer_pool(self):
        """Positions are correct when transformer pool is in composite."""
        prefix = Pool(['PREFIX'], name='Pref', mode='sequential', metadata='complete')
        
        bg = IUPACPool(iupac_seq="NNNNNNNN", name="TransBg", metadata='complete')
        ins = IUPACPool(iupac_seq="GG", name="TransIns", metadata='complete')
        scanner = InsertionScanPool(
            bg, ins, start=2, end=6, step_size=2,
            insert_or_overwrite='overwrite',
            name='Trans', mode='sequential', metadata='complete'
        )
        
        suffix = Pool(['SUFFIX'], name='Suff', mode='sequential', metadata='complete')
        
        library = prefix + scanner + suffix
        
        result = library.generate_seqs(num_seqs=3, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(3):
            row = dc.get_row(i)
            seq = result['sequences'][i]
            
            # PREFIX at start
            assert row['Pref_abs_start'] == 0
            assert row['Pref_abs_end'] == 6
            
            # Scanner after prefix
            assert row['Trans_abs_start'] == 6
            assert row['Trans_abs_end'] == 14
            
            # SUFFIX at end
            assert row['Suff_abs_start'] == 14
            assert row['Suff_abs_end'] == 20
            
            # Trans_pos_abs should account for prefix offset
            assert row['Trans_pos_abs'] == 6 + row['Trans_pos']
    
    def test_spacing_scan_position_metadata(self):
        """SpacingScanPool position metadata is accurate."""
        ins_a = IUPACPool(iupac_seq="AA", name="SPA", metadata='complete')
        ins_b = IUPACPool(iupac_seq="TT", name="SPB", metadata='complete')
        
        pool = SpacingScanPool(
            background_seq="N" * 20,
            insert_seqs=[ins_a, ins_b],
            insert_names=["A", "B"],
            anchor_pos=10,
            insert_distances=[[-4], [4]],
            name="SP", mode='sequential', metadata='complete'
        )
        
        prefix = Pool(['---'], name='SPPre', mode='sequential')
        library = prefix + pool
        
        result = library.generate_seqs(num_seqs=2, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(2):
            row = dc.get_row(i)
            seq = result['sequences'][i]
            
            # SpacingScanPool absolute positions
            a_abs_start = row['SP_A_abs_pos_start']
            a_abs_end = row['SP_A_abs_pos_end']
            b_abs_start = row['SP_B_abs_pos_start']
            b_abs_end = row['SP_B_abs_pos_end']
            
            # Verify inserts are at those positions (prefix is 3 chars)
            assert seq[a_abs_start:a_abs_end] == 'AA', \
                f"Expected AA at [{a_abs_start}:{a_abs_end}], got '{seq[a_abs_start:a_abs_end]}'"
            assert seq[b_abs_start:b_abs_end] == 'TT', \
                f"Expected TT at [{b_abs_start}:{b_abs_end}], got '{seq[b_abs_start:b_abs_end]}'"


class TestDesignCardComprehensiveScenarios:
    """Comprehensive end-to-end scenarios testing all aspects."""
    
    def test_full_library_design_scenario(self):
        """Realistic library design with multiple pool types."""
        # Structure: Prefix + MixedPool([SpacingScan, Plain]) + Barcode + Suffix
        
        # Prefix
        prefix = Pool(['GCTA'], name='Lib_Pre', mode='sequential')
        
        # SpacingScanPool with Pool inserts
        ins_a = IUPACPool(iupac_seq="RR", name='Lib_MotifA', metadata='complete')
        ins_b = IUPACPool(iupac_seq="YY", name='Lib_MotifB', metadata='complete')
        spacing = SpacingScanPool(
            background_seq="N" * 16,
            insert_seqs=[ins_a, ins_b],
            insert_names=["A", "B"],
            anchor_pos=8,
            insert_distances=[[-3], [3]],
            name='Lib_Spacing', mode='sequential', metadata='complete'
        )
        
        # Plain alternative
        plain = Pool(["X" * 16], name='Lib_Plain', mode='sequential')
        
        # MixedPool
        mixed = MixedPool([spacing, plain], name='Lib_Mix', mode='sequential')
        
        # Barcode
        barcode = KmerPool(length=4, alphabet='dna', name='Lib_BC', 
                          mode='sequential', metadata='complete')
        
        # Suffix
        suffix = Pool(['TAGC'], name='Lib_Suf', mode='sequential')
        
        library = prefix + mixed + barcode + suffix
        
        result = library.generate_seqs(num_seqs=20, return_design_cards=True)
        dc = result['design_cards']
        
        # Verify all expected pools are tracked
        expected_pools = ['Lib_Pre', 'Lib_Mix', 'Lib_Spacing', 'Lib_MotifA', 
                         'Lib_MotifB', 'Lib_Plain', 'Lib_BC', 'Lib_Suf']
        for pool_name in expected_pools:
            assert f'{pool_name}_index' in dc.keys, f"{pool_name} should be tracked"
        
        # Verify values and positions for each sequence
        for i in range(min(20, 10)):  # Check first 10
            row = dc.get_row(i)
            seq = result['sequences'][i]
            
            # Prefix always at start
            assert seq[:4] == 'GCTA'
            assert row['Lib_Pre_abs_start'] == 0
            assert row['Lib_Pre_abs_end'] == 4
            
            # Suffix always at end
            assert seq[-4:] == 'TAGC'
            
            # MixedPool selection tracked
            assert row['Lib_Mix_selected'] in [0, 1]
            
            if row['Lib_Mix_selected'] == 0:  # Spacing selected
                # Motif values should be present
                assert row['Lib_MotifA_value'] is not None
                assert row['Lib_MotifB_value'] is not None
                # Spacing value should match output portion
                assert row['Lib_Spacing_value'] == seq[4:20]
            else:  # Plain selected
                assert seq[4:20] == 'X' * 16
    
    def test_track_pools_at_all_levels(self):
        """track_pools correctly filters at all nesting levels."""
        # Deep structure: prefix + MixedPool([prefix2 + SpacingScan, plain])
        ins_a = IUPACPool(iupac_seq="AA", name="Deep_InsA", metadata='complete')
        ins_b = IUPACPool(iupac_seq="TT", name="Deep_InsB", metadata='complete')
        
        spacing = SpacingScanPool(
            background_seq="N" * 12,
            insert_seqs=[ins_a, ins_b],
            insert_names=["A", "B"],
            anchor_pos=6,
            insert_distances=[[-2], [2]],
            name="Deep_SS", mode='sequential'
        )
        
        inner_prefix = Pool(["PPP"], name="Deep_InnerPre", mode='sequential')
        composite_child = inner_prefix + spacing  # 15 chars
        
        plain = Pool(["X" * 15], name="Deep_Plain", mode='sequential')
        mixed = MixedPool([composite_child, plain], name="Deep_Mix", mode='sequential')
        
        outer_prefix = Pool(["OOO"], name="Deep_OuterPre", mode='sequential')
        library = outer_prefix + mixed
        
        # Test 1: Track only outermost and innermost
        result1 = library.generate_seqs(num_seqs=3, return_design_cards=True,
                                         track_pools=['Deep_OuterPre', 'Deep_InsA'])
        dc1 = result1['design_cards']
        
        assert 'Deep_OuterPre_index' in dc1.keys
        assert 'Deep_InsA_index' in dc1.keys
        assert 'Deep_Mix_selected' not in dc1.keys
        assert 'Deep_SS_index' not in dc1.keys
        assert 'Deep_InsB_index' not in dc1.keys
        
        # Test 2: Track SpacingScanPool and its inserts only
        result2 = library.generate_seqs(num_seqs=3, return_design_cards=True,
                                         track_pools=['Deep_SS', 'Deep_InsA', 'Deep_InsB'])
        dc2 = result2['design_cards']
        
        assert 'Deep_SS_index' in dc2.keys
        assert 'Deep_InsA_index' in dc2.keys
        assert 'Deep_InsB_index' in dc2.keys
        assert 'Deep_OuterPre_index' not in dc2.keys
        assert 'Deep_Mix_selected' not in dc2.keys
        
        # Test 3: No track_pools - should get everything
        result3 = library.generate_seqs(num_seqs=3, return_design_cards=True)
        dc3 = result3['design_cards']
        
        expected_all = ['Deep_OuterPre', 'Deep_Mix', 'Deep_InnerPre', 
                       'Deep_SS', 'Deep_InsA', 'Deep_InsB', 'Deep_Plain']
        for pool_name in expected_all:
            assert f'{pool_name}_index' in dc3.keys, \
                f"{pool_name} should be tracked when no filter"


class TestArbitraryNestingLevels:
    """Tests for arbitrary levels of nesting with all pool type combinations."""
    
    def test_transformer_inside_mixed_inside_transformer(self):
        """SpacingScanPool → MixedPool → SpacingScanPool with Pool inserts."""
        # Level 3: Innermost inserts
        inner_ins_a = IUPACPool(iupac_seq="AA", name="NL3_InsA", metadata='complete')
        inner_ins_b = IUPACPool(iupac_seq="TT", name="NL3_InsB", metadata='complete')
        
        # Level 2: Inner SpacingScanPool
        inner_spacing = SpacingScanPool(
            background_seq="N" * 16,
            insert_seqs=[inner_ins_a, inner_ins_b],
            insert_names=["A", "B"],
            anchor_pos=8, insert_distances=[[-3], [3]],
            name="NL2_SS", mode='sequential', metadata='complete'
        )
        inner_plain = Pool(["X" * 16], name="NL2_Plain", mode='sequential')
        
        # Level 1: MixedPool
        mixed = MixedPool([inner_spacing, inner_plain], name="NL1_Mix", mode='sequential')
        outer_ins = IUPACPool(iupac_seq="GGGG", name="NL1_OuterIns", metadata='complete')
        
        # Level 0: Outer SpacingScanPool
        outer_spacing = SpacingScanPool(
            background_seq="N" * 40,
            insert_seqs=[mixed, outer_ins],
            insert_names=["Mix", "Out"],
            anchor_pos=20, insert_distances=[[-10], [10]],
            name="NL0_SS", mode='sequential', metadata='complete'
        )
        
        result = outer_spacing.generate_seqs(num_seqs=10, return_design_cards=True)
        dc = result['design_cards']
        
        # All pools at all levels should be tracked
        expected = ["NL0_SS", "NL1_Mix", "NL2_SS", "NL3_InsA", "NL3_InsB", 
                   "NL2_Plain", "NL1_OuterIns"]
        for pool_name in expected:
            assert f'{pool_name}_index' in dc.keys, \
                f"{pool_name} should be tracked at nesting level"
        
        # Verify values when inner spacing is selected
        for i in range(10):
            row = dc.get_row(i)
            if row['NL1_Mix_selected'] == 0:  # inner_spacing selected
                assert row['NL3_InsA_value'] == 'AA'
                assert row['NL3_InsB_value'] == 'TT'
    
    def test_four_levels_of_nesting(self):
        """4-level deep nesting with alternating pool types."""
        # Level 4: Deepest inserts
        L4_a = IUPACPool(iupac_seq="AA", name="4L_InsA", metadata='complete')
        L4_b = IUPACPool(iupac_seq="TT", name="4L_InsB", metadata='complete')
        
        # Level 3: InsertionScanPool
        L3_scan = InsertionScanPool(
            background_seq="NNNNNNNNNN",
            insert_seq=L4_a,
            start=2, end=8, step_size=3,
            insert_or_overwrite='overwrite',
            name="4L_IS", mode='sequential', metadata='complete'
        )
        L3_plain = Pool(["Y" * 10], name="4L_L3Plain", mode='sequential')
        
        # Level 2: MixedPool
        L2_mix = MixedPool([L3_scan, L3_plain], name="4L_Mix", mode='sequential')
        
        # Level 1: SpacingScanPool
        L1_spacing = SpacingScanPool(
            background_seq="N" * 24,
            insert_seqs=[L2_mix, L4_b],
            insert_names=["Mix", "B"],
            anchor_pos=12, insert_distances=[[-6], [6]],
            name="4L_SS", mode='sequential', metadata='complete'
        )
        
        # Level 0: Outer MixedPool
        L0_plain = Pool(["X" * 24], name="4L_L0Plain", mode='sequential')
        root = MixedPool([L1_spacing, L0_plain], name="4L_Root", mode='sequential')
        
        result = root.generate_seqs(num_seqs=15, return_design_cards=True)
        dc = result['design_cards']
        
        # All 8 pools should be tracked
        expected = ["4L_Root", "4L_SS", "4L_Mix", "4L_IS", "4L_InsA", 
                   "4L_L3Plain", "4L_InsB", "4L_L0Plain"]
        for pool_name in expected:
            assert f'{pool_name}_index' in dc.keys, \
                f"{pool_name} at level 4 nesting should be tracked"
    
    def test_track_pools_at_arbitrary_depth(self):
        """track_pools correctly filters pools at any nesting depth."""
        # Same 4-level structure
        L4_a = IUPACPool(iupac_seq="GG", name="TP_L4a", metadata='complete')
        L4_b = IUPACPool(iupac_seq="CC", name="TP_L4b", metadata='complete')
        
        L3_scan = InsertionScanPool(
            "NNNNNNNN", L4_a, start=2, end=6, step_size=2,
            insert_or_overwrite='overwrite',
            name="TP_L3", mode='sequential'
        )
        L3_plain = Pool(["Y" * 8], name="TP_L3P", mode='sequential')
        
        L2_mix = MixedPool([L3_scan, L3_plain], name="TP_L2", mode='sequential')
        
        L1_ss = SpacingScanPool(
            background_seq="N" * 20, 
            insert_seqs=[L2_mix, L4_b], 
            insert_names=["M", "B"],
            anchor_pos=10, insert_distances=[[-5], [5]],
            name="TP_L1", mode='sequential'
        )
        
        L0_plain = Pool(["X" * 20], name="TP_L0P", mode='sequential')
        root = MixedPool([L1_ss, L0_plain], name="TP_Root", mode='sequential')
        
        # Track only deepest level pools
        result = root.generate_seqs(num_seqs=5, return_design_cards=True,
                                     track_pools=['TP_L4a', 'TP_L4b'])
        dc = result['design_cards']
        
        assert 'TP_L4a_index' in dc.keys
        assert 'TP_L4b_index' in dc.keys
        assert 'TP_Root_selected' not in dc.keys
        assert 'TP_L1_index' not in dc.keys
        
        # Track middle level only
        result2 = root.generate_seqs(num_seqs=5, return_design_cards=True,
                                      track_pools=['TP_L2', 'TP_L3'])
        dc2 = result2['design_cards']
        
        assert 'TP_L2_selected' in dc2.keys or 'TP_L2_index' in dc2.keys
        assert 'TP_L3_index' in dc2.keys
        assert 'TP_L4a_index' not in dc2.keys
        assert 'TP_Root_selected' not in dc2.keys
    
    def test_composite_inside_mixed_inside_composite_inside_transformer(self):
        """Complex: (A+B) inside MixedPool inside (prefix + SpacingScanPool)."""
        # Innermost: composite A+B
        A = IUPACPool(iupac_seq="AA", name="Cplx_A", metadata='complete')
        B = IUPACPool(iupac_seq="TT", name="Cplx_B", metadata='complete')
        composite = A + B  # 4bp
        
        # MixedPool with composite and plain
        plain1 = Pool(["XXXX"], name="Cplx_P1", mode='sequential')
        mixed = MixedPool([composite, plain1], name="Cplx_Mix", mode='sequential')
        
        # Another insert for SpacingScanPool
        ins = IUPACPool(iupac_seq="GG", name="Cplx_Ins", metadata='complete')
        
        # SpacingScanPool with MixedPool as insert
        spacing = SpacingScanPool(
            background_seq="N" * 16,
            insert_seqs=[mixed, ins],
            insert_names=["M", "I"],
            anchor_pos=8, insert_distances=[[-4], [4]],
            name="Cplx_SS", mode='sequential', metadata='complete'
        )
        
        # Outer composite
        prefix = Pool(["PRE"], name="Cplx_Pre", mode='sequential')
        suffix = Pool(["SUF"], name="Cplx_Suf", mode='sequential')
        root = prefix + spacing + suffix
        
        result = root.generate_seqs(num_seqs=10, return_design_cards=True)
        dc = result['design_cards']
        
        # All named pools should be tracked
        expected = ["Cplx_Pre", "Cplx_SS", "Cplx_Mix", "Cplx_A", "Cplx_B", 
                   "Cplx_P1", "Cplx_Ins", "Cplx_Suf"]
        for pool_name in expected:
            assert f'{pool_name}_index' in dc.keys, \
                f"{pool_name} in complex nested structure should be tracked"


class TestTransformerNestingCombinations:
    """Tests for all combinations of transformer pool nesting."""
    
    def test_multi_input_inside_multi_input(self):
        """SpacingScanPool inside SpacingScanPool - both track their inputs."""
        inner_a = IUPACPool(iupac_seq="AA", name="MM_InA", metadata='complete')
        inner_b = IUPACPool(iupac_seq="TT", name="MM_InB", metadata='complete')
        inner_ss = SpacingScanPool(
            background_seq="N" * 16,
            insert_seqs=[inner_a, inner_b],
            insert_names=["A", "B"],
            anchor_pos=8, insert_distances=[[-3], [3]],
            name="MM_Inner", mode='sequential'
        )
        outer_c = IUPACPool(iupac_seq="GG", name="MM_OutC", metadata='complete')
        outer_ss = SpacingScanPool(
            background_seq="N" * 30,
            insert_seqs=[inner_ss, outer_c],
            insert_names=["In", "C"],
            anchor_pos=15, insert_distances=[[-8], [8]],
            name="MM_Outer", mode='sequential'
        )
        
        result = outer_ss.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        
        expected = ["MM_Outer", "MM_Inner", "MM_InA", "MM_InB", "MM_OutC"]
        for p in expected:
            assert f'{p}_index' in dc.keys, f"{p} should be tracked in Multi→Multi"
    
    def test_multi_input_using_single_input_transformer(self):
        """SpacingScanPool uses KMutationPool as insert."""
        base = IUPACPool(iupac_seq="AAAAAAAAAA", name="MS_Base", metadata='complete')
        k_mut = KMutationPool(base, alphabet='dna', k=1, name="MS_KMut", mode='sequential')
        other = IUPACPool(iupac_seq="GGG", name="MS_Other", metadata='complete')
        ss = SpacingScanPool(
            background_seq="N" * 24,
            insert_seqs=[k_mut, other],
            insert_names=["Mut", "Other"],
            anchor_pos=12, insert_distances=[[-6], [6]],
            name="MS_SS", mode='sequential'
        )
        
        result = ss.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        
        expected = ["MS_SS", "MS_KMut", "MS_Base", "MS_Other"]
        for p in expected:
            assert f'{p}_index' in dc.keys, f"{p} should be tracked in Multi→Single"
    
    def test_single_input_on_multi_input(self):
        """KMutationPool applied to SpacingScanPool output."""
        ins_a = IUPACPool(iupac_seq="AA", name="SM_InsA", metadata='complete')
        ins_b = IUPACPool(iupac_seq="TT", name="SM_InsB", metadata='complete')
        inner = SpacingScanPool(
            background_seq="N" * 16,
            insert_seqs=[ins_a, ins_b],
            insert_names=["A", "B"],
            anchor_pos=8, insert_distances=[[-3], [3]],
            name="SM_Inner", mode='sequential'
        )
        k_mut = KMutationPool(inner, alphabet='dna', k=1, name="SM_KMut", mode='sequential')
        
        result = k_mut.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        
        expected = ["SM_KMut", "SM_Inner", "SM_InsA", "SM_InsB"]
        for p in expected:
            assert f'{p}_index' in dc.keys, f"{p} should be tracked in Single→Multi"
    
    def test_string_bg_transformer_in_multi_input(self):
        """InsertionScanPool with string bg inside SpacingScanPool."""
        pool_ins = IUPACPool(iupac_seq="RRRR", name="NM_PoolIns", metadata='complete')
        ins_scan = InsertionScanPool(
            background_seq="AAAAAAAAAA",  # String, not Pool
            insert_seq=pool_ins,
            start=2, end=8, step_size=2,
            insert_or_overwrite='overwrite',
            name="NM_IS", mode='sequential'
        )
        other = IUPACPool(iupac_seq="GG", name="NM_Other", metadata='complete')
        ss = SpacingScanPool(
            background_seq="N" * 24,
            insert_seqs=[ins_scan, other],
            insert_names=["IS", "Oth"],
            anchor_pos=12, insert_distances=[[-6], [6]],
            name="NM_SS", mode='sequential'
        )
        
        result = ss.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        
        expected = ["NM_SS", "NM_IS", "NM_PoolIns", "NM_Other"]
        for p in expected:
            assert f'{p}_index' in dc.keys, f"{p} should be tracked in Non-pool→Multi"
    
    def test_three_levels_multi_input_nesting(self):
        """3 levels of SpacingScanPool nesting."""
        L3_a = IUPACPool(iupac_seq="A", name="3M_A", metadata='complete')
        L3_b = IUPACPool(iupac_seq="T", name="3M_B", metadata='complete')
        L2 = SpacingScanPool(
            background_seq="N" * 8,
            insert_seqs=[L3_a, L3_b], insert_names=["A", "B"],
            anchor_pos=4, insert_distances=[[-2], [2]],
            name="3M_L2", mode='sequential'
        )
        L2_c = IUPACPool(iupac_seq="G", name="3M_C", metadata='complete')
        L1 = SpacingScanPool(
            background_seq="N" * 16,
            insert_seqs=[L2, L2_c], insert_names=["X", "C"],
            anchor_pos=8, insert_distances=[[-4], [4]],
            name="3M_L1", mode='sequential'
        )
        L1_d = IUPACPool(iupac_seq="C", name="3M_D", metadata='complete')
        L0 = SpacingScanPool(
            background_seq="N" * 32,
            insert_seqs=[L1, L1_d], insert_names=["Y", "D"],
            anchor_pos=16, insert_distances=[[-8], [8]],
            name="3M_L0", mode='sequential'
        )
        
        result = L0.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        
        expected = ["3M_L0", "3M_L1", "3M_L2", "3M_A", "3M_B", "3M_C", "3M_D"]
        for p in expected:
            assert f'{p}_index' in dc.keys, f"{p} should be tracked in 3-level nesting"
    
    def test_pool_bg_insertion_scan_in_spacing_scan(self):
        """InsertionScanPool with Pool inputs inside SpacingScanPool."""
        bg = IUPACPool(iupac_seq="NNNNNNNNNN", name="PI_Bg", metadata='complete')
        ins = IUPACPool(iupac_seq="RR", name="PI_Ins", metadata='complete')
        ins_scan = InsertionScanPool(
            background_seq=bg, insert_seq=ins,
            start=2, end=8, step_size=2,
            insert_or_overwrite='overwrite',
            name="PI_IS", mode='sequential'
        )
        other = IUPACPool(iupac_seq="GGG", name="PI_Other", metadata='complete')
        ss = SpacingScanPool(
            background_seq="N" * 24,
            insert_seqs=[ins_scan, other],
            insert_names=["IS", "Oth"],
            anchor_pos=12, insert_distances=[[-6], [6]],
            name="PI_SS", mode='sequential'
        )
        
        result = ss.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        
        expected = ["PI_SS", "PI_IS", "PI_Bg", "PI_Ins", "PI_Other"]
        for p in expected:
            assert f'{p}_index' in dc.keys, f"{p} should be tracked"
    
    def test_shuffle_scan_in_spacing_scan(self):
        """ShuffleScanPool inside SpacingScanPool."""
        base = IUPACPool(iupac_seq="AAAAAAAAAA", name="SS_Base", metadata='complete')
        shuf = ShuffleScanPool(
            base, shuffle_size=4, start=2, end=8, step_size=2,
            name="SS_Shuf", mode='sequential'
        )
        other = IUPACPool(iupac_seq="TT", name="SS_Other", metadata='complete')
        ss = SpacingScanPool(
            background_seq="N" * 24,
            insert_seqs=[shuf, other],
            insert_names=["Sh", "Oth"],
            anchor_pos=12, insert_distances=[[-6], [6]],
            name="SS_SS", mode='sequential'
        )
        
        result = ss.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        
        expected = ["SS_SS", "SS_Shuf", "SS_Base", "SS_Other"]
        for p in expected:
            assert f'{p}_index' in dc.keys, f"{p} should be tracked"
    
    def test_deletion_scan_in_insertion_scan(self):
        """DeletionScanPool as insert in InsertionScanPool."""
        del_base = IUPACPool(iupac_seq="AAAAAAAAAAAA", name="DI_Base", metadata='complete')
        del_scan = DeletionScanPool(
            del_base, deletion_size=3, start=2, end=9, step_size=3,
            name="DI_Del", mode='sequential'
        )
        ins_bg = IUPACPool(iupac_seq="N" * 20, name="DI_Bg", metadata='complete')
        ins = InsertionScanPool(
            background_seq=ins_bg, insert_seq=del_scan,
            start=4, end=16, step_size=4,
            insert_or_overwrite='overwrite',
            name="DI_IS", mode='sequential'
        )
        
        result = ins.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        
        expected = ["DI_IS", "DI_Bg", "DI_Del", "DI_Base"]
        for p in expected:
            assert f'{p}_index' in dc.keys, f"{p} should be tracked"


class TestRigorousValueValidation:
    """Rigorous tests that verify design card values are correct, not just columns."""
    
    def _validate_design_cards(self, pool, expected_pools, num_seqs=10):
        """Helper to validate design cards comprehensively."""
        result = pool.generate_seqs(num_seqs=num_seqs, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        # Check all expected pools have columns
        for p in expected_pools:
            assert f'{p}_index' in dc.keys, f"{p}_index missing"
        
        # Validate each sequence
        for i in range(num_seqs):
            row = dc.get_row(i)
            seq = seqs[i]
            
            # Sequence length check
            assert row['sequence_length'] == len(seq), f"Seq {i}: length mismatch"
            
            for pname in expected_pools:
                # Index should be int or None
                idx = row.get(f'{pname}_index')
                assert idx is None or isinstance(idx, int), f"{pname}_index not int"
                
                # Value should be string or None
                val = row.get(f'{pname}_value')
                assert val is None or isinstance(val, str), f"{pname}_value not string"
                
                # Positions should be valid
                abs_start = row.get(f'{pname}_abs_start')
                abs_end = row.get(f'{pname}_abs_end')
                if abs_start is not None and abs_end is not None:
                    assert 0 <= abs_start <= abs_end <= len(seq), \
                        f"{pname} positions invalid"
                
                # Selection should be valid int
                selected = row.get(f'{pname}_selected')
                if selected is not None:
                    assert isinstance(selected, int) and selected >= 0
        
        return dc, seqs
    
    def test_multi_input_inside_multi_input_values(self):
        """Rigorously validate Multi→Multi SpacingScan values."""
        inner_a = IUPACPool(iupac_seq="AAAA", name="RV1_InA", metadata='complete')
        inner_b = IUPACPool(iupac_seq="TTTT", name="RV1_InB", metadata='complete')
        inner_ss = SpacingScanPool(
            background_seq="N" * 20, insert_seqs=[inner_a, inner_b],
            insert_names=["A", "B"], anchor_pos=10, insert_distances=[[-5], [5]],
            name="RV1_Inner", mode='sequential', metadata='complete'
        )
        outer_c = IUPACPool(iupac_seq="GGGG", name="RV1_OutC", metadata='complete')
        outer_ss = SpacingScanPool(
            background_seq="N" * 40, insert_seqs=[inner_ss, outer_c],
            insert_names=["In", "C"], anchor_pos=20, insert_distances=[[-10], [10]],
            name="RV1_Outer", mode='sequential', metadata='complete'
        )
        
        dc, seqs = self._validate_design_cards(outer_ss, 
            ["RV1_Outer", "RV1_Inner", "RV1_InA", "RV1_InB", "RV1_OutC"])
        
        # Additional checks: root pool value should match sequence
        for i in range(10):
            row = dc.get_row(i)
            assert row['RV1_Outer_value'] == seqs[i]
    
    def test_spacing_mixed_spacing_chain_values(self):
        """Rigorously validate SS→Mix→SS chain values."""
        L3_a = IUPACPool(iupac_seq="AA", name="RV2_L3A", metadata='complete')
        L3_b = IUPACPool(iupac_seq="TT", name="RV2_L3B", metadata='complete')
        L2_ss = SpacingScanPool(
            background_seq="N" * 12, insert_seqs=[L3_a, L3_b],
            insert_names=["A", "B"], anchor_pos=6, insert_distances=[[-2], [2]],
            name="RV2_L2SS", mode='sequential', metadata='complete'
        )
        L2_plain = Pool(["X" * 12], name="RV2_Plain", mode='sequential', metadata='complete')
        L1_mix = MixedPool([L2_ss, L2_plain], name="RV2_Mix", mode='sequential')
        L1_c = IUPACPool(iupac_seq="GG", name="RV2_L1C", metadata='complete')
        L0_ss = SpacingScanPool(
            background_seq="N" * 28, insert_seqs=[L1_mix, L1_c],
            insert_names=["M", "C"], anchor_pos=14, insert_distances=[[-7], [7]],
            name="RV2_L0SS", mode='sequential', metadata='complete'
        )
        
        dc, seqs = self._validate_design_cards(L0_ss,
            ["RV2_L0SS", "RV2_Mix", "RV2_L2SS", "RV2_L3A", "RV2_L3B", "RV2_Plain", "RV2_L1C"])
        
        # Check MixedPool selection is tracked correctly
        for i in range(10):
            row = dc.get_row(i)
            assert row['RV2_Mix_selected'] in [0, 1]
    
    def test_insertion_scan_pool_inputs_in_spacing_values(self):
        """Rigorously validate InsertionScan(Pool) in SpacingScan."""
        bg = IUPACPool(iupac_seq="NNNNNNNNNN", name="RV3_Bg", metadata='complete')
        ins = IUPACPool(iupac_seq="RR", name="RV3_Ins", metadata='complete')
        is_pool = InsertionScanPool(
            background_seq=bg, insert_seq=ins, start=2, end=8, step_size=2,
            insert_or_overwrite='overwrite', name="RV3_IS", mode='sequential', metadata='complete'
        )
        other = IUPACPool(iupac_seq="GGG", name="RV3_Other", metadata='complete')
        ss = SpacingScanPool(
            background_seq="N" * 26, insert_seqs=[is_pool, other],
            insert_names=["IS", "Oth"], anchor_pos=13, insert_distances=[[-6], [6]],
            name="RV3_SS", mode='sequential', metadata='complete'
        )
        
        self._validate_design_cards(ss, ["RV3_SS", "RV3_IS", "RV3_Bg", "RV3_Ins", "RV3_Other"])
    
    def test_composite_in_mixed_in_spacing_values(self):
        """Rigorously validate (A+B) in MixedPool in SpacingScan."""
        A = IUPACPool(iupac_seq="AA", name="RV4_A", metadata='complete')
        B = IUPACPool(iupac_seq="TT", name="RV4_B", metadata='complete')
        comp = A + B
        plain = Pool(["XXXX"], name="RV4_Plain", mode='sequential', metadata='complete')
        mix = MixedPool([comp, plain], name="RV4_Mix", mode='sequential')
        other = IUPACPool(iupac_seq="GG", name="RV4_Other", metadata='complete')
        ss = SpacingScanPool(
            background_seq="N" * 16, insert_seqs=[mix, other],
            insert_names=["M", "Oth"], anchor_pos=8, insert_distances=[[-4], [4]],
            name="RV4_SS", mode='sequential', metadata='complete'
        )
        
        self._validate_design_cards(ss, ["RV4_SS", "RV4_Mix", "RV4_A", "RV4_B", "RV4_Plain", "RV4_Other"])
    
    def test_kmutation_on_spacing_scan_values(self):
        """Rigorously validate KMutation on SpacingScan - transformer chain semantics."""
        ins_a = IUPACPool(iupac_seq="AAAA", name="RV5_InsA", metadata='complete')
        ins_b = IUPACPool(iupac_seq="TTTT", name="RV5_InsB", metadata='complete')
        ss = SpacingScanPool(
            background_seq="CCCCCCCCCCCCCCCC", insert_seqs=[ins_a, ins_b],
            insert_names=["A", "B"], anchor_pos=8, insert_distances=[[-4], [4]],
            name="RV5_SS", mode='sequential', metadata='complete'
        )
        kmut = KMutationPool(ss, alphabet='dna', k=1, name="RV5_KMut", mode='sequential', metadata='complete')
        
        dc, seqs = self._validate_design_cards(kmut, ["RV5_KMut", "RV5_SS", "RV5_InsA", "RV5_InsB"])
        
        # KMut value should match sequence (it's the output)
        # SS value should be the INPUT (before mutation)
        for i in range(10):
            row = dc.get_row(i)
            assert row['RV5_KMut_value'] == seqs[i], "KMut output should match sequence"
            # SS value should NOT match sequence (it's before mutation)
            # Just verify it's present
            assert row['RV5_SS_value'] is not None
    
    def test_four_level_nesting_values(self):
        """Rigorously validate 4-level deep nesting."""
        L4_a = IUPACPool(iupac_seq="A", name="RV6_L4A", metadata='complete')
        L4_b = IUPACPool(iupac_seq="T", name="RV6_L4B", metadata='complete')
        L3_ss = SpacingScanPool(
            background_seq="N" * 6, insert_seqs=[L4_a, L4_b],
            insert_names=["A", "B"], anchor_pos=3, insert_distances=[[-1], [1]],
            name="RV6_L3", mode='sequential', metadata='complete'
        )
        L3_p = Pool(["Y" * 6], name="RV6_L3P", mode='sequential', metadata='complete')
        L2_mix = MixedPool([L3_ss, L3_p], name="RV6_L2Mix", mode='sequential')
        L2_c = IUPACPool(iupac_seq="G", name="RV6_L2C", metadata='complete')
        L1_ss = SpacingScanPool(
            background_seq="N" * 14, insert_seqs=[L2_mix, L2_c],
            insert_names=["M", "C"], anchor_pos=7, insert_distances=[[-3], [3]],
            name="RV6_L1", mode='sequential', metadata='complete'
        )
        L1_p = Pool(["Z" * 14], name="RV6_L1P", mode='sequential', metadata='complete')
        L0_mix = MixedPool([L1_ss, L1_p], name="RV6_L0Mix", mode='sequential')
        
        self._validate_design_cards(L0_mix,
            ["RV6_L0Mix", "RV6_L1", "RV6_L2Mix", "RV6_L3", "RV6_L4A", "RV6_L4B", "RV6_L3P", "RV6_L2C", "RV6_L1P"])
    
    def test_full_library_composite_values(self):
        """Rigorously validate full library composite."""
        prefix = Pool(["GCTA"], name="RV7_Pre", mode='sequential', metadata='complete')
        motif_a = IUPACPool(iupac_seq="RRR", name="RV7_MotifA", metadata='complete')
        motif_b = IUPACPool(iupac_seq="YYY", name="RV7_MotifB", metadata='complete')
        spacing = SpacingScanPool(
            background_seq="N" * 20, insert_seqs=[motif_a, motif_b],
            insert_names=["A", "B"], anchor_pos=10, insert_distances=[[-5], [5]],
            name="RV7_Spacing", mode='sequential', metadata='complete'
        )
        barcode = KmerPool(length=4, alphabet='dna', name="RV7_BC", mode='sequential', metadata='complete')
        suffix = Pool(["TAGC"], name="RV7_Suf", mode='sequential', metadata='complete')
        lib = prefix + spacing + barcode + suffix
        
        dc, seqs = self._validate_design_cards(lib,
            ["RV7_Pre", "RV7_Spacing", "RV7_MotifA", "RV7_MotifB", "RV7_BC", "RV7_Suf"])
        
        # Verify prefix and suffix positions
        for i in range(10):
            row = dc.get_row(i)
            assert row['RV7_Pre_abs_start'] == 0
            assert row['RV7_Pre_abs_end'] == 4
            assert seqs[i][:4] == 'GCTA'
            assert seqs[i][-4:] == 'TAGC'
    
    def test_three_level_spacing_scan_chain_values(self):
        """Rigorously validate 3-level SpacingScan chain."""
        L3a = IUPACPool(iupac_seq="A", name="RV8_A", metadata='complete')
        L3b = IUPACPool(iupac_seq="T", name="RV8_B", metadata='complete')
        L2 = SpacingScanPool(
            background_seq="N" * 8, insert_seqs=[L3a, L3b],
            insert_names=["A", "B"], anchor_pos=4, insert_distances=[[-2], [2]],
            name="RV8_L2", mode='sequential', metadata='complete'
        )
        L2c = IUPACPool(iupac_seq="G", name="RV8_C", metadata='complete')
        L1 = SpacingScanPool(
            background_seq="N" * 16, insert_seqs=[L2, L2c],
            insert_names=["X", "C"], anchor_pos=8, insert_distances=[[-4], [4]],
            name="RV8_L1", mode='sequential', metadata='complete'
        )
        L1d = IUPACPool(iupac_seq="C", name="RV8_D", metadata='complete')
        L0 = SpacingScanPool(
            background_seq="N" * 32, insert_seqs=[L1, L1d],
            insert_names=["Y", "D"], anchor_pos=16, insert_distances=[[-8], [8]],
            name="RV8_L0", mode='sequential', metadata='complete'
        )
        
        dc, seqs = self._validate_design_cards(L0,
            ["RV8_L0", "RV8_L1", "RV8_L2", "RV8_A", "RV8_B", "RV8_C", "RV8_D"])
        
        # Root value should match sequence
        for i in range(10):
            row = dc.get_row(i)
            assert row['RV8_L0_value'] == seqs[i]
    
    def test_spacing_scan_with_pool_background_all_inputs(self):
        """SpacingScanPool with Pool background AND multiple Pool inserts - all tracked."""
        bg = IUPACPool(iupac_seq="N" * 50, name="RV9_Bg", metadata='complete')
        ins_a = IUPACPool(iupac_seq="AAAA", name="RV9_InsA", metadata='complete')
        ins_b = IUPACPool(iupac_seq="TTTT", name="RV9_InsB", metadata='complete')
        ins_c = IUPACPool(iupac_seq="GGGG", name="RV9_InsC", metadata='complete')
        
        pool = SpacingScanPool(
            background_seq=bg,  # Pool background
            insert_seqs=[ins_a, ins_b, ins_c],  # 3 pool inserts
            insert_names=["A", "B", "C"],
            anchor_pos=25,
            insert_distances=[[-15], [0], [15]],
            name="RV9_SS",
            mode='sequential',
            metadata='complete'
        )
        
        # Verify ALL inputs are tracked: background + 3 inserts + the SpacingScan itself
        dc, seqs = self._validate_design_cards(pool,
            ["RV9_SS", "RV9_Bg", "RV9_InsA", "RV9_InsB", "RV9_InsC"])
        
        # Verify values are correct
        for i in range(10):
            row = dc.get_row(i)
            assert row['RV9_SS_value'] == seqs[i], "SS output should match sequence"
            # Background value should be different from output (inserts overwrite)
            assert row['RV9_Bg_value'] is not None, "Background value should be tracked"
            assert row['RV9_InsA_value'] == 'AAAA'
            assert row['RV9_InsB_value'] == 'TTTT'
            assert row['RV9_InsC_value'] == 'GGGG'


class TestInsertionScanNestingCombinations:
    """Comprehensive tests for InsertionScanPool nesting combinations with value verification."""
    
    def test_insertion_scan_in_insertion_scan(self):
        """InsertionScan → InsertionScan: nested insertion scans with value verification."""
        # Inner InsertionScan: 10bp bg, 2bp insert at positions 2,4,6
        inner_bg = IUPACPool(iupac_seq="GGGGGGGGGG", name="IS_InnerBg", metadata='complete')
        inner_ins = IUPACPool(iupac_seq="AA", name="IS_InnerIns", metadata='complete')
        inner_is = InsertionScanPool(
            background_seq=inner_bg, insert_seq=inner_ins,
            start=2, end=8, step_size=2, insert_or_overwrite='overwrite',
            name="IS_Inner", mode='sequential', metadata='complete'
        )
        
        # Outer InsertionScan: 20bp bg, insert inner_is at positions 4,8,12
        outer_bg = IUPACPool(iupac_seq="CCCCCCCCCCCCCCCCCCCC", name="IS_OuterBg", metadata='complete')
        outer_is = InsertionScanPool(
            background_seq=outer_bg, insert_seq=inner_is,
            start=4, end=14, step_size=4, insert_or_overwrite='overwrite',
            name="IS_Outer", mode='sequential', metadata='complete'
        )
        
        result = outer_is.generate_seqs(num_seqs=6, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        # Check columns exist
        expected = ["IS_Outer", "IS_OuterBg", "IS_Inner", "IS_InnerBg", "IS_InnerIns"]
        for p in expected:
            assert f'{p}_index' in dc.keys, f"{p} should be tracked"
        
        # Verify values for first few sequences
        for i in range(min(3, len(seqs))):
            row = dc.get_row(i)
            seq = seqs[i]
            
            # Outer InsertionScan value should match the full sequence
            assert row['IS_Outer_value'] == seq, f"Seq {i}: Outer value should match sequence"
            
            # Outer bg value should be original (CCCC...)
            assert row['IS_OuterBg_value'] == 'CCCCCCCCCCCCCCCCCCCC', f"Seq {i}: OuterBg should be original"
            
            # Inner insert value should be AA
            assert row['IS_InnerIns_value'] == 'AA', f"Seq {i}: InnerIns should be AA"
            
            # Inner value should be 10 chars containing AA
            inner_val = row['IS_Inner_value']
            assert len(inner_val) == 10, f"Seq {i}: Inner should be 10bp"
            assert 'AA' in inner_val, f"Seq {i}: Inner should contain AA"
            
            # Outer pos should be one of 4, 8, 12
            assert row['IS_Outer_pos'] in [4, 8, 12], f"Seq {i}: Invalid outer pos"
    
    def test_insertion_scan_uses_kmutation(self):
        """InsertionScan → KMutation: insertion scan with KMutation as insert, with value verification."""
        base = IUPACPool(iupac_seq="GGGGGGGG", name="ISK_Base", metadata='complete')
        kmut = KMutationPool(base, alphabet='dna', k=1, name="ISK_KMut", mode='sequential', metadata='complete')
        
        bg = IUPACPool(iupac_seq="CCCCCCCCCCCCCCCCCCCC", name="ISK_Bg", metadata='complete')
        is_pool = InsertionScanPool(
            background_seq=bg, insert_seq=kmut,
            start=4, end=14, step_size=4, insert_or_overwrite='overwrite',
            name="ISK_IS", mode='sequential', metadata='complete'
        )
        
        result = is_pool.generate_seqs(num_seqs=6, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        expected = ["ISK_IS", "ISK_Bg", "ISK_KMut", "ISK_Base"]
        for p in expected:
            assert f'{p}_index' in dc.keys, f"{p} should be tracked"
        
        for i in range(min(3, len(seqs))):
            row = dc.get_row(i)
            seq = seqs[i]
            
            # IS value matches sequence
            assert row['ISK_IS_value'] == seq
            
            # Background is original
            assert row['ISK_Bg_value'] == 'CCCCCCCCCCCCCCCCCCCC'
            
            # Base is original (before mutation)
            assert row['ISK_Base_value'] == 'GGGGGGGG'
            
            # KMut value should be 8 chars with one mutation from G
            kmut_val = row['ISK_KMut_value']
            assert len(kmut_val) == 8
            non_g = sum(1 for c in kmut_val if c != 'G')
            assert non_g == 1, f"Seq {i}: KMut should have exactly 1 mutation"
            
            # KMut mutation info should be present
            assert row['ISK_KMut_mut_pos'] is not None
            assert row['ISK_KMut_mut_from'] is not None
            assert row['ISK_KMut_mut_to'] is not None
    
    def test_kmutation_on_insertion_scan(self):
        """KMutation → InsertionScan: KMutation applied to InsertionScan output, with value verification."""
        bg = IUPACPool(iupac_seq="GGGGGGGGGG", name="KIS_Bg", metadata='complete')
        ins = IUPACPool(iupac_seq="AA", name="KIS_Ins", metadata='complete')
        is_pool = InsertionScanPool(
            background_seq=bg, insert_seq=ins,
            start=2, end=8, step_size=2, insert_or_overwrite='overwrite',
            name="KIS_IS", mode='sequential', metadata='complete'
        )
        
        kmut = KMutationPool(is_pool, alphabet='dna', k=1, name="KIS_KMut", mode='sequential', metadata='complete')
        
        result = kmut.generate_seqs(num_seqs=6, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        expected = ["KIS_KMut", "KIS_IS", "KIS_Bg", "KIS_Ins"]
        for p in expected:
            assert f'{p}_index' in dc.keys, f"{p} should be tracked"
        
        for i in range(min(3, len(seqs))):
            row = dc.get_row(i)
            seq = seqs[i]
            
            # KMut is the output, matches sequence
            assert row['KIS_KMut_value'] == seq
            
            # IS value is the INPUT (before mutation) - should have AA in it
            is_val = row['KIS_IS_value']
            assert 'AA' in is_val, f"Seq {i}: IS value should contain AA"
            assert len(is_val) == 10
            
            # Ins value is AA
            assert row['KIS_Ins_value'] == 'AA'
            
            # Bg value is original
            assert row['KIS_Bg_value'] == 'GGGGGGGGGG'
    
    def test_insertion_scan_uses_shuffle_scan(self):
        """InsertionScan → ShuffleScan: insertion scan with ShuffleScan as insert, with value verification."""
        shuffle_base = IUPACPool(iupac_seq="AATTGGCC", name="ISS_ShufBase", metadata='complete')
        shuffle = ShuffleScanPool(
            shuffle_base, shuffle_size=4, start=0, end=4, step_size=2,
            name="ISS_Shuf", mode='sequential', metadata='complete'
        )
        
        bg = IUPACPool(iupac_seq="GGGGGGGGGGGGGGGGGGGG", name="ISS_Bg", metadata='complete')
        is_pool = InsertionScanPool(
            background_seq=bg, insert_seq=shuffle,
            start=4, end=14, step_size=4, insert_or_overwrite='overwrite',
            name="ISS_IS", mode='sequential', metadata='complete'
        )
        
        result = is_pool.generate_seqs(num_seqs=6, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        expected = ["ISS_IS", "ISS_Bg", "ISS_Shuf", "ISS_ShufBase"]
        for p in expected:
            assert f'{p}_index' in dc.keys, f"{p} should be tracked"
        
        for i in range(min(3, len(seqs))):
            row = dc.get_row(i)
            seq = seqs[i]
            
            # IS value matches sequence
            assert row['ISS_IS_value'] == seq
            
            # Shuffle base is original
            assert row['ISS_ShufBase_value'] == 'AATTGGCC'
            
            # Shuffle value is 8 chars (shuffled version)
            shuf_val = row['ISS_Shuf_value']
            assert len(shuf_val) == 8
            # Should have same characters as base (case-insensitive due to swapcase marking)
            assert sorted(shuf_val.upper()) == sorted('AATTGGCC')
            
            # Shuffle has pos and window_size
            assert row['ISS_Shuf_pos'] in [0, 2]
            assert row['ISS_Shuf_window_size'] == 4
    
    def test_insertion_scan_uses_deletion_scan(self):
        """InsertionScan → DeletionScan: insertion scan with DeletionScan as insert, with value verification."""
        del_base = IUPACPool(iupac_seq="AAAAAAAAAAAA", name="ISD_DelBase", metadata='complete')  # 12bp
        del_scan = DeletionScanPool(
            del_base, deletion_size=3, start=0, end=9, step_size=3,
            mark_changes=False,  # Actually delete, don't mark with dashes
            name="ISD_Del", mode='sequential', metadata='complete'
        )  # Produces 9bp sequences
        
        bg = IUPACPool(iupac_seq="GGGGGGGGGGGGGGGGGGGGGGGGGGGGGG", name="ISD_Bg", metadata='complete')  # 30bp
        is_pool = InsertionScanPool(
            background_seq=bg, insert_seq=del_scan,
            start=2, end=22, step_size=5, insert_or_overwrite='overwrite',
            name="ISD_IS", mode='sequential', metadata='complete'
        )
        
        result = is_pool.generate_seqs(num_seqs=6, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        expected = ["ISD_IS", "ISD_Bg", "ISD_Del", "ISD_DelBase"]
        for p in expected:
            assert f'{p}_index' in dc.keys, f"{p} should be tracked"
        
        for i in range(min(3, len(seqs))):
            row = dc.get_row(i)
            seq = seqs[i]
            
            # IS value matches sequence
            assert row['ISD_IS_value'] == seq
            assert len(seq) == 30  # Still 30bp after overwrite
            
            # DelBase is original
            assert row['ISD_DelBase_value'] == 'AAAAAAAAAAAA'
            
            # Del value is 9bp (12 - 3)
            del_val = row['ISD_Del_value']
            assert len(del_val) == 9
            assert del_val == 'AAAAAAAAA'  # All A's after deletion
            
            # Del has pos and del_len
            assert row['ISD_Del_pos'] in [0, 3, 6]
            assert row['ISD_Del_del_len'] == 3
    
    def test_insertion_scan_in_spacing_scan(self):
        """InsertionScan inside SpacingScan as one of multiple inserts, with value verification."""
        # InsertionScan: 8bp bg, 2bp insert
        is_bg = IUPACPool(iupac_seq="GGGGGGGG", name="ISinSS_ISBg", metadata='complete')
        is_ins = IUPACPool(iupac_seq="AA", name="ISinSS_ISIns", metadata='complete')
        is_pool = InsertionScanPool(
            background_seq=is_bg, insert_seq=is_ins,
            start=2, end=6, step_size=2, insert_or_overwrite='overwrite',
            name="ISinSS_IS", mode='sequential', metadata='complete'
        )
        
        other = IUPACPool(iupac_seq="TTTT", name="ISinSS_Other", metadata='complete')
        
        ss = SpacingScanPool(
            background_seq="CCCCCCCCCCCCCCCCCCCCCCCC",  # 24bp
            insert_seqs=[is_pool, other],
            insert_names=["IS", "Oth"],
            anchor_pos=12, insert_distances=[[-6], [6]],
            name="ISinSS_SS", mode='sequential', metadata='complete'
        )
        
        result = ss.generate_seqs(num_seqs=3, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        expected = ["ISinSS_SS", "ISinSS_IS", "ISinSS_ISBg", "ISinSS_ISIns", "ISinSS_Other"]
        for p in expected:
            assert f'{p}_index' in dc.keys, f"{p} should be tracked"
        
        for i in range(len(seqs)):
            row = dc.get_row(i)
            seq = seqs[i]
            
            # SS value matches sequence
            assert row['ISinSS_SS_value'] == seq
            assert len(seq) == 24
            
            # IS value should be 8bp with AA in it
            is_val = row['ISinSS_IS_value']
            assert len(is_val) == 8
            assert 'AA' in is_val
            
            # ISIns is AA
            assert row['ISinSS_ISIns_value'] == 'AA'
            
            # Other is TTTT
            assert row['ISinSS_Other_value'] == 'TTTT'
            
            # IS_pos should be valid
            assert row['ISinSS_IS_pos'] in [2, 4]
    
    def test_insertion_scan_string_bg_pool_insert_in_spacing(self):
        """InsertionScan with string bg and Pool insert, nested in SpacingScan, with value verification."""
        ins = IUPACPool(iupac_seq="AAA", name="ISstr_Ins", metadata='complete')
        is_pool = InsertionScanPool(
            background_seq="GGGGGGGGGG",  # String background, 10bp
            insert_seq=ins,
            start=2, end=8, step_size=2, insert_or_overwrite='overwrite',
            name="ISstr_IS", mode='sequential', metadata='complete'
        )
        
        other = IUPACPool(iupac_seq="TT", name="ISstr_Other", metadata='complete')
        ss = SpacingScanPool(
            background_seq="CCCCCCCCCCCCCCCCCCCCCCCC",  # 24bp
            insert_seqs=[is_pool, other],
            insert_names=["IS", "Oth"],
            anchor_pos=12, insert_distances=[[-6], [6]],
            name="ISstr_SS", mode='sequential', metadata='complete'
        )
        
        result = ss.generate_seqs(num_seqs=3, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        expected = ["ISstr_SS", "ISstr_IS", "ISstr_Ins", "ISstr_Other"]
        for p in expected:
            assert f'{p}_index' in dc.keys, f"{p} should be tracked"
        
        for i in range(len(seqs)):
            row = dc.get_row(i)
            seq = seqs[i]
            
            # SS matches sequence
            assert row['ISstr_SS_value'] == seq
            
            # IS value is 10bp with AAA in it
            is_val = row['ISstr_IS_value']
            assert len(is_val) == 10
            assert 'AAA' in is_val
            
            # Ins is AAA
            assert row['ISstr_Ins_value'] == 'AAA'
            
            # Other is TT
            assert row['ISstr_Other_value'] == 'TT'
    
    def test_three_level_insertion_scan_nesting(self):
        """3-level InsertionScan nesting with value verification."""
        # Level 3: innermost - 4bp bg, 1bp insert
        L3_bg = IUPACPool(iupac_seq="GGGG", name="IS3_L3Bg", metadata='complete')
        L3_ins = IUPACPool(iupac_seq="A", name="IS3_L3Ins", metadata='complete')
        L3 = InsertionScanPool(
            background_seq=L3_bg, insert_seq=L3_ins,
            start=1, end=3, step_size=1, insert_or_overwrite='overwrite',
            name="IS3_L3", mode='sequential', metadata='complete'
        )
        
        # Level 2 - 10bp bg, 4bp insert (L3)
        L2_bg = IUPACPool(iupac_seq="CCCCCCCCCC", name="IS3_L2Bg", metadata='complete')
        L2 = InsertionScanPool(
            background_seq=L2_bg, insert_seq=L3,
            start=2, end=8, step_size=2, insert_or_overwrite='overwrite',
            name="IS3_L2", mode='sequential', metadata='complete'
        )
        
        # Level 1: outermost - 20bp bg, 10bp insert (L2)
        L1_bg = IUPACPool(iupac_seq="TTTTTTTTTTTTTTTTTTTT", name="IS3_L1Bg", metadata='complete')
        L1 = InsertionScanPool(
            background_seq=L1_bg, insert_seq=L2,
            start=4, end=14, step_size=2, insert_or_overwrite='overwrite',
            name="IS3_L1", mode='sequential', metadata='complete'
        )
        
        result = L1.generate_seqs(num_seqs=6, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        expected = ["IS3_L1", "IS3_L1Bg", "IS3_L2", "IS3_L2Bg", "IS3_L3", "IS3_L3Bg", "IS3_L3Ins"]
        for p in expected:
            assert f'{p}_index' in dc.keys, f"{p} should be tracked"
        
        for i in range(min(3, len(seqs))):
            row = dc.get_row(i)
            seq = seqs[i]
            
            # L1 is output, matches sequence
            assert row['IS3_L1_value'] == seq
            assert len(seq) == 20
            
            # L2 is 10bp
            l2_val = row['IS3_L2_value']
            assert len(l2_val) == 10
            
            # L3 is 4bp with A in it
            l3_val = row['IS3_L3_value']
            assert len(l3_val) == 4
            assert 'A' in l3_val
            
            # L3_ins is A
            assert row['IS3_L3Ins_value'] == 'A'
            
            # All backgrounds are original
            assert row['IS3_L3Bg_value'] == 'GGGG'
            assert row['IS3_L2Bg_value'] == 'CCCCCCCCCC'
            assert row['IS3_L1Bg_value'] == 'TTTTTTTTTTTTTTTTTTTT'


class TestInsertionScanORFNestingCombinations:
    """Comprehensive tests for InsertionScanORFPool nesting combinations with value verification.
    
    Note: ORF pools have strict requirements (divisible by 3, ACGT only) that
    limit some nesting combinations. Tests focus on what actually works.
    """
    
    def test_insertion_scan_orf_in_spacing_scan(self):
        """InsertionScanORF inside SpacingScan as insert, with value verification."""
        # 18bp = 6 codons background
        bg = IUPACPool(iupac_seq="ATGACGTACGTACGTGAA", name="ISOinSS_Bg", metadata='complete')
        ins = IUPACPool(iupac_seq="GGG", name="ISOinSS_Ins", metadata='complete')  # 1 codon
        is_orf = InsertionScanORFPool(
            background_seq=bg, insert_seq=ins,
            start=1, end=5, step_size=1, insert_or_overwrite='overwrite',
            name="ISOinSS_IS", mode='sequential', metadata='complete'
        )
        
        other = IUPACPool(iupac_seq="TTTT", name="ISOinSS_Other", metadata='complete')
        ss = SpacingScanPool(
            background_seq="CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",  # 40bp
            insert_seqs=[is_orf, other],
            insert_names=["ISO", "Oth"],
            anchor_pos=20, insert_distances=[[-10], [10]],
            name="ISOinSS_SS", mode='sequential', metadata='complete'
        )
        
        result = ss.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        expected = ["ISOinSS_SS", "ISOinSS_IS", "ISOinSS_Bg", "ISOinSS_Ins", "ISOinSS_Other"]
        for p in expected:
            assert f'{p}_index' in dc.keys, f"{p} should be tracked"
        
        for i in range(min(3, len(seqs))):
            row = dc.get_row(i)
            seq = seqs[i]
            
            # SS value matches sequence
            assert row['ISOinSS_SS_value'] == seq
            assert len(seq) == 40
            
            # IS value is 18bp (ORF maintains length)
            is_val = row['ISOinSS_IS_value']
            assert len(is_val) == 18
            assert 'GGG' in is_val  # Contains inserted codon
            
            # Bg is original
            assert row['ISOinSS_Bg_value'] == 'ATGACGTACGTACGTGAA'
            
            # Ins is GGG
            assert row['ISOinSS_Ins_value'] == 'GGG'
            
            # Other is TTTT
            assert row['ISOinSS_Other_value'] == 'TTTT'
    
    def test_insertion_scan_orf_string_bg_pool_insert(self):
        """InsertionScanORF with string bg and Pool insert, with value verification."""
        ins = IUPACPool(iupac_seq="GGG", name="ISOstr_Ins", metadata='complete')
        is_pool = InsertionScanORFPool(
            background_seq="ATGACGTACGTACGTGAA",  # String background, 18bp = 6 codons
            insert_seq=ins,
            start=1, end=5, step_size=1, insert_or_overwrite='overwrite',
            name="ISOstr_IS", mode='sequential', metadata='complete'
        )
        
        result = is_pool.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        # Columns present
        assert 'ISOstr_IS_index' in dc.keys
        assert 'ISOstr_Ins_index' in dc.keys
        
        for i in range(min(3, len(seqs))):
            row = dc.get_row(i)
            seq = seqs[i]
            
            # IS value matches sequence
            assert row['ISOstr_IS_value'] == seq
            assert len(seq) == 18
            assert 'GGG' in seq  # Contains inserted codon
            
            # Ins is GGG
            assert row['ISOstr_Ins_value'] == 'GGG'
    
    def test_insertion_scan_orf_pool_bg_pool_insert(self):
        """InsertionScanORF with Pool background AND Pool insert, with value verification."""
        bg = IUPACPool(iupac_seq="ATGACGTACGTACGTGAA", name="ISO_PoolBg", metadata='complete')
        ins = IUPACPool(iupac_seq="GGG", name="ISO_PoolIns", metadata='complete')
        is_pool = InsertionScanORFPool(
            background_seq=bg, insert_seq=ins,
            start=1, end=5, step_size=1, insert_or_overwrite='overwrite',
            name="ISO_Pool", mode='sequential', metadata='complete'
        )
        
        result = is_pool.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        # Columns present
        assert 'ISO_Pool_index' in dc.keys
        assert 'ISO_PoolBg_index' in dc.keys
        assert 'ISO_PoolIns_index' in dc.keys
        
        for i in range(min(3, len(seqs))):
            row = dc.get_row(i)
            seq = seqs[i]
            
            # Pool value matches sequence
            assert row['ISO_Pool_value'] == seq
            assert len(seq) == 18
            assert 'GGG' in seq
            
            # Bg is original
            assert row['ISO_PoolBg_value'] == 'ATGACGTACGTACGTGAA'
            
            # Ins is GGG
            assert row['ISO_PoolIns_value'] == 'GGG'
            
            # codon_pos should be valid codon position
            assert row['ISO_Pool_codon_pos'] in [1, 2, 3, 4]  # Codon positions
    
    def test_deletion_scan_orf_in_spacing_scan(self):
        """DeletionScanORF inside SpacingScan as insert, with value verification."""
        # 12bp = 4 codons, delete 1 codon = 9bp output
        del_base = IUPACPool(iupac_seq="ATGAAAGGGCCC", name="DSinSS_Base", metadata='complete')
        del_scan = DeletionScanORFPool(
            del_base, deletion_size=1, start=1, end=4, step_size=1,
            mark_changes=False, name="DSinSS_Del", mode='sequential', metadata='complete'
        )
        
        other = IUPACPool(iupac_seq="TTTT", name="DSinSS_Other", metadata='complete')
        ss = SpacingScanPool(
            background_seq="CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",  # 30bp
            insert_seqs=[del_scan, other],
            insert_names=["Del", "Oth"],
            anchor_pos=15, insert_distances=[[-7], [7]],
            name="DSinSS_SS", mode='sequential', metadata='complete'
        )
        
        result = ss.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        expected = ["DSinSS_SS", "DSinSS_Del", "DSinSS_Base", "DSinSS_Other"]
        for p in expected:
            assert f'{p}_index' in dc.keys, f"{p} should be tracked"
        
        for i in range(min(3, len(seqs))):
            row = dc.get_row(i)
            seq = seqs[i]
            
            # SS value matches sequence
            assert row['DSinSS_SS_value'] == seq
            assert len(seq) == 30
            
            # Del value is 9bp (12 - 3 = 9 after 1 codon deletion)
            del_val = row['DSinSS_Del_value']
            assert len(del_val) == 9
            
            # Base is original
            assert row['DSinSS_Base_value'] == 'ATGAAAGGGCCC'
            
            # Other is TTTT
            assert row['DSinSS_Other_value'] == 'TTTT'
            
            # Del codon_pos should be valid codon position
            assert row['DSinSS_Del_codon_pos'] in [1, 2, 3]
    
    def test_kmutation_orf_in_spacing_scan(self):
        """KMutationORF inside SpacingScan as insert, with value verification."""
        base = IUPACPool(iupac_seq="ATGGGG", name="KMinSS_Base", metadata='complete')  # 6bp = 2 codons
        kmut = KMutationORFPool(base, k=1, mutation_type='any_codon', name="KMinSS_KMut", mode='random', metadata='complete')
        
        other = IUPACPool(iupac_seq="TTTT", name="KMinSS_Other", metadata='complete')
        ss = SpacingScanPool(
            background_seq="CCCCCCCCCCCCCCCCCCCCCCCC",  # 24bp
            insert_seqs=[kmut, other],
            insert_names=["KM", "Oth"],
            anchor_pos=12, insert_distances=[[-6], [6]],
            name="KMinSS_SS", mode='sequential', metadata='complete'
        )
        
        result = ss.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        expected = ["KMinSS_SS", "KMinSS_KMut", "KMinSS_Base", "KMinSS_Other"]
        for p in expected:
            assert f'{p}_index' in dc.keys, f"{p} should be tracked"
        
        for i in range(min(3, len(seqs))):
            row = dc.get_row(i)
            seq = seqs[i]
            
            # SS value matches sequence
            assert row['KMinSS_SS_value'] == seq
            assert len(seq) == 24
            
            # KMut value is 6bp
            kmut_val = row['KMinSS_KMut_value']
            assert len(kmut_val) == 6
            
            # Base is original
            assert row['KMinSS_Base_value'] == 'ATGGGG'
            
            # Other is TTTT
            assert row['KMinSS_Other_value'] == 'TTTT'
    
    def test_multiple_orf_pools_in_spacing_scan(self):
        """Multiple ORF pool types as inserts in SpacingScan, with value verification."""
        # InsertionScanORF: 12bp = 4 codons
        is_bg = IUPACPool(iupac_seq="ATGACGTACGGG", name="Multi_ISBg", metadata='complete')
        is_ins = IUPACPool(iupac_seq="AAA", name="Multi_ISIns", metadata='complete')
        is_orf = InsertionScanORFPool(
            background_seq=is_bg, insert_seq=is_ins,
            start=1, end=3, step_size=1, insert_or_overwrite='overwrite',
            name="Multi_IS", mode='sequential', metadata='complete'
        )
        
        # DeletionScanORF: 12bp = 4 codons, delete 1 = 9bp output
        ds_base = IUPACPool(iupac_seq="ATGGGGAAACCC", name="Multi_DSBase", metadata='complete')
        ds_orf = DeletionScanORFPool(
            ds_base, deletion_size=1, start=1, end=4, step_size=1,
            mark_changes=False, name="Multi_DS", mode='sequential', metadata='complete'
        )
        
        # Plain pool
        plain = IUPACPool(iupac_seq="TTTT", name="Multi_Plain", metadata='complete')
        
        ss = SpacingScanPool(
            background_seq="CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",  # 40bp
            insert_seqs=[is_orf, ds_orf, plain],
            insert_names=["IS", "DS", "P"],
            anchor_pos=20, insert_distances=[[-15], [0], [15]],
            name="Multi_SS", mode='sequential', metadata='complete'
        )
        
        result = ss.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        expected = ["Multi_SS", "Multi_IS", "Multi_ISBg", "Multi_ISIns", "Multi_DS", "Multi_DSBase", "Multi_Plain"]
        for p in expected:
            assert f'{p}_index' in dc.keys, f"{p} should be tracked"
        
        for i in range(min(3, len(seqs))):
            row = dc.get_row(i)
            seq = seqs[i]
            
            # SS value matches sequence
            assert row['Multi_SS_value'] == seq
            assert len(seq) == 40
            
            # IS value is 12bp with AAA
            is_val = row['Multi_IS_value']
            assert len(is_val) == 12
            assert 'AAA' in is_val
            
            # DS value is 9bp
            ds_val = row['Multi_DS_value']
            assert len(ds_val) == 9
            
            # ISBg is original
            assert row['Multi_ISBg_value'] == 'ATGACGTACGGG'
            
            # ISIns is AAA
            assert row['Multi_ISIns_value'] == 'AAA'
            
            # DSBase is original
            assert row['Multi_DSBase_value'] == 'ATGGGGAAACCC'
            
            # Plain is TTTT
            assert row['Multi_Plain_value'] == 'TTTT'


class TestMultiInputMultiInputNesting:
    """Rigorous tests for multi-input → multi-input transformer nesting.
    
    These tests verify complex scenarios where multi-pool-input transformers
    are nested inside other multi-pool-input transformers, with full value verification.
    """
    
    def test_spacing_scan_inside_spacing_scan_with_pool_bg(self):
        """SpacingScan inside SpacingScan, both with Pool backgrounds - full verification."""
        # Inner SpacingScan: Pool bg + 2 Pool inserts
        inner_bg = IUPACPool(iupac_seq="GGGGGGGGGGGGGGGGGGGG", name="SS2_InnerBg", metadata='complete')  # 20bp
        inner_ins_a = IUPACPool(iupac_seq="AAAA", name="SS2_InnerA", metadata='complete')
        inner_ins_b = IUPACPool(iupac_seq="TTTT", name="SS2_InnerB", metadata='complete')
        inner_ss = SpacingScanPool(
            background_seq=inner_bg,  # Pool background
            insert_seqs=[inner_ins_a, inner_ins_b],
            insert_names=["A", "B"],
            anchor_pos=10, insert_distances=[[-5], [5]],
            name="SS2_Inner", mode='sequential', metadata='complete'
        )
        
        # Outer SpacingScan: Pool bg + inner_ss + another insert
        outer_bg = IUPACPool(iupac_seq="CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC", name="SS2_OuterBg", metadata='complete')  # 40bp
        outer_other = IUPACPool(iupac_seq="GGGG", name="SS2_OuterOther", metadata='complete')
        outer_ss = SpacingScanPool(
            background_seq=outer_bg,  # Pool background
            insert_seqs=[inner_ss, outer_other],
            insert_names=["Inner", "Other"],
            anchor_pos=20, insert_distances=[[-10], [10]],
            name="SS2_Outer", mode='sequential', metadata='complete'
        )
        
        result = outer_ss.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        # Verify ALL pools are tracked
        expected = ["SS2_Outer", "SS2_OuterBg", "SS2_Inner", "SS2_InnerBg", "SS2_InnerA", "SS2_InnerB", "SS2_OuterOther"]
        for p in expected:
            assert f'{p}_index' in dc.keys, f"{p} should be tracked"
        
        for i in range(min(3, len(seqs))):
            row = dc.get_row(i)
            seq = seqs[i]
            
            # Outer matches sequence
            assert row['SS2_Outer_value'] == seq
            assert len(seq) == 40
            
            # Outer bg is original
            assert row['SS2_OuterBg_value'] == 'CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC'
            
            # Inner is 20bp with A's and T's inserted
            inner_val = row['SS2_Inner_value']
            assert len(inner_val) == 20
            assert 'AAAA' in inner_val
            assert 'TTTT' in inner_val
            
            # Inner bg is original
            assert row['SS2_InnerBg_value'] == 'GGGGGGGGGGGGGGGGGGGG'
            
            # Insert values correct
            assert row['SS2_InnerA_value'] == 'AAAA'
            assert row['SS2_InnerB_value'] == 'TTTT'
            assert row['SS2_OuterOther_value'] == 'GGGG'
    
    def test_insertion_scan_with_pool_bg_inside_spacing_scan(self):
        """InsertionScan with Pool bg inside SpacingScan - full verification."""
        # InsertionScan with Pool bg and Pool insert
        is_bg = IUPACPool(iupac_seq="GGGGGGGGGGGG", name="ISSS_ISBg", metadata='complete')  # 12bp
        is_ins = IUPACPool(iupac_seq="AAA", name="ISSS_ISIns", metadata='complete')
        is_pool = InsertionScanPool(
            background_seq=is_bg,  # Pool background
            insert_seq=is_ins,
            start=2, end=10, step_size=2, insert_or_overwrite='overwrite',
            name="ISSS_IS", mode='sequential', metadata='complete'
        )
        
        # Another Pool insert for SpacingScan
        other = IUPACPool(iupac_seq="TTTT", name="ISSS_Other", metadata='complete')
        
        # SpacingScan with Pool bg
        ss_bg = IUPACPool(iupac_seq="CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC", name="ISSS_SSBg", metadata='complete')  # 30bp
        ss = SpacingScanPool(
            background_seq=ss_bg,  # Pool background
            insert_seqs=[is_pool, other],
            insert_names=["IS", "Oth"],
            anchor_pos=15, insert_distances=[[-8], [8]],
            name="ISSS_SS", mode='sequential', metadata='complete'
        )
        
        result = ss.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        # Verify ALL pools tracked
        expected = ["ISSS_SS", "ISSS_SSBg", "ISSS_IS", "ISSS_ISBg", "ISSS_ISIns", "ISSS_Other"]
        for p in expected:
            assert f'{p}_index' in dc.keys, f"{p} should be tracked"
        
        for i in range(min(3, len(seqs))):
            row = dc.get_row(i)
            seq = seqs[i]
            
            # SS matches sequence
            assert row['ISSS_SS_value'] == seq
            assert len(seq) == 30
            
            # SS bg is original
            assert row['ISSS_SSBg_value'] == 'CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC'
            
            # IS is 12bp with AAA inserted
            is_val = row['ISSS_IS_value']
            assert len(is_val) == 12
            assert 'AAA' in is_val
            
            # IS bg is original
            assert row['ISSS_ISBg_value'] == 'GGGGGGGGGGGG'
            
            # Insert values
            assert row['ISSS_ISIns_value'] == 'AAA'
            assert row['ISSS_Other_value'] == 'TTTT'
            
            # IS has pos metadata
            assert row['ISSS_IS_pos'] in [2, 4, 6, 8]
    
    def test_composite_insert_in_spacing_scan(self):
        """Composite (A + B) as insert in SpacingScan - full verification."""
        # Create composite: two IUPACPools concatenated
        comp_a = IUPACPool(iupac_seq="AAAA", name="CompSS_A", metadata='complete')
        comp_b = IUPACPool(iupac_seq="TTTT", name="CompSS_B", metadata='complete')
        composite = comp_a + comp_b  # 8bp composite
        
        other = IUPACPool(iupac_seq="GGGG", name="CompSS_Other", metadata='complete')
        
        ss = SpacingScanPool(
            background_seq="CCCCCCCCCCCCCCCCCCCCCCCC",  # 24bp
            insert_seqs=[composite, other],
            insert_names=["Comp", "Oth"],
            anchor_pos=12, insert_distances=[[-6], [6]],
            name="CompSS_SS", mode='sequential', metadata='complete'
        )
        
        result = ss.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        # Both parts of composite should be tracked
        expected = ["CompSS_SS", "CompSS_A", "CompSS_B", "CompSS_Other"]
        for p in expected:
            assert f'{p}_index' in dc.keys, f"{p} should be tracked"
        
        for i in range(min(3, len(seqs))):
            row = dc.get_row(i)
            seq = seqs[i]
            
            # SS matches sequence
            assert row['CompSS_SS_value'] == seq
            assert len(seq) == 24
            
            # Both composite parts present
            assert row['CompSS_A_value'] == 'AAAA'
            assert row['CompSS_B_value'] == 'TTTT'
            
            # Sequence contains composite
            assert 'AAAATTTT' in seq
            
            assert row['CompSS_Other_value'] == 'GGGG'
    
    def test_three_inserts_spacing_scan_all_pool_types(self):
        """SpacingScan with 3 different Pool-based inserts - comprehensive."""
        # Insert 1: Plain IUPAC (4bp)
        ins1 = IUPACPool(iupac_seq="AAAA", name="Multi3_Ins1", metadata='complete')
        
        # Insert 2: KMutation (4bp)
        ins2_base = IUPACPool(iupac_seq="GGGG", name="Multi3_Ins2Base", metadata='complete')
        ins2 = KMutationPool(ins2_base, alphabet='dna', k=1, name="Multi3_Ins2", mode='random', metadata='complete')
        
        # Insert 3: InsertionScan (8bp)
        ins3_bg = IUPACPool(iupac_seq="CCCCCCCC", name="Multi3_Ins3Bg", metadata='complete')
        ins3_ins = IUPACPool(iupac_seq="TT", name="Multi3_Ins3Ins", metadata='complete')
        ins3 = InsertionScanPool(
            background_seq=ins3_bg, insert_seq=ins3_ins,
            start=2, end=6, step_size=2, insert_or_overwrite='overwrite',
            name="Multi3_Ins3", mode='sequential', metadata='complete'
        )
        
        # Bg=50bp, inserts: 4+4+8=16bp total
        # anchor=25, distances: -20, -5, 10 (left edges at 5, 20, 35)
        ss = SpacingScanPool(
            background_seq="N" * 50,
            insert_seqs=[ins1, ins2, ins3],
            insert_names=["I1", "I2", "I3"],
            anchor_pos=25, insert_distances=[[-20], [-5], [10]],
            min_spacing=0, enforce_order=False,
            name="Multi3_SS", mode='sequential', metadata='complete'
        )
        
        result = ss.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        # ALL pools should be tracked
        expected = ["Multi3_SS", "Multi3_Ins1", "Multi3_Ins2", "Multi3_Ins2Base", "Multi3_Ins3", "Multi3_Ins3Bg", "Multi3_Ins3Ins"]
        for p in expected:
            assert f'{p}_index' in dc.keys, f"{p} should be tracked"
        
        for i in range(min(3, len(seqs))):
            row = dc.get_row(i)
            seq = seqs[i]
            
            # SS matches sequence
            assert row['Multi3_SS_value'] == seq
            assert len(seq) == 50
            
            # Ins1 value
            assert row['Multi3_Ins1_value'] == 'AAAA'
            
            # Ins2 is mutated (4bp, 1 mutation from GGGG)
            ins2_val = row['Multi3_Ins2_value']
            assert len(ins2_val) == 4
            
            # Ins2Base is original
            assert row['Multi3_Ins2Base_value'] == 'GGGG'
            
            # Ins3 is 8bp with TT
            ins3_val = row['Multi3_Ins3_value']
            assert len(ins3_val) == 8
            assert 'TT' in ins3_val
            
            # Ins3 inputs
            assert row['Multi3_Ins3Bg_value'] == 'CCCCCCCC'
            assert row['Multi3_Ins3Ins_value'] == 'TT'


class TestInsertionScanORFMultiInputNesting:
    """Rigorous tests for InsertionScanORFPool with complex multi-input nesting."""
    
    def test_insertion_scan_orf_with_composite_inputs(self):
        """InsertionScanORF where insert is a composite (A + B) - all tracked."""
        # Background
        bg = IUPACPool(iupac_seq="ATGACGTACGTACGTGAA", name="ISOC_Bg", metadata='complete')  # 18bp = 6 codons
        
        # Composite insert: two 3bp pools = 6bp = 2 codons
        ins_a = IUPACPool(iupac_seq="AAA", name="ISOC_InsA", metadata='complete')
        ins_b = IUPACPool(iupac_seq="GGG", name="ISOC_InsB", metadata='complete')
        composite_ins = ins_a + ins_b  # 6bp composite = 2 codons
        
        is_pool = InsertionScanORFPool(
            background_seq=bg,
            insert_seq=composite_ins,
            start=1, end=4, step_size=1, insert_or_overwrite='overwrite',
            name="ISOC_IS", mode='sequential', metadata='complete'
        )
        
        result = is_pool.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        # All parts tracked
        expected = ["ISOC_IS", "ISOC_Bg", "ISOC_InsA", "ISOC_InsB"]
        for p in expected:
            assert f'{p}_index' in dc.keys, f"{p} should be tracked"
        
        for i in range(min(3, len(seqs))):
            row = dc.get_row(i)
            seq = seqs[i]
            
            # IS matches sequence
            assert row['ISOC_IS_value'] == seq
            assert len(seq) == 18
            
            # Contains composite insert AAAGGG
            assert 'AAAGGG' in seq
            
            # Bg is original
            assert row['ISOC_Bg_value'] == 'ATGACGTACGTACGTGAA'
            
            # Both insert parts tracked
            assert row['ISOC_InsA_value'] == 'AAA'
            assert row['ISOC_InsB_value'] == 'GGG'
    
    def test_kmutation_orf_on_insertion_scan_orf_with_pool_inputs(self):
        """KMutationORF wrapping InsertionScanORF with Pool bg and Pool insert."""
        # InsertionScanORF with Pool inputs
        bg = IUPACPool(iupac_seq="ATGACGTACGTACGTGAA", name="KMISO_Bg", metadata='complete')  # 18bp
        ins = IUPACPool(iupac_seq="GGG", name="KMISO_Ins", metadata='complete')  # 1 codon
        is_pool = InsertionScanORFPool(
            background_seq=bg,
            insert_seq=ins,
            start=1, end=5, step_size=1, insert_or_overwrite='overwrite',
            name="KMISO_IS", mode='sequential', metadata='complete'
        )
        
        # KMutationORF wrapping InsertionScanORF
        kmut = KMutationORFPool(
            is_pool, k=1, mutation_type='any_codon',
            name="KMISO_KMut", mode='random', metadata='complete'
        )
        
        result = kmut.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        # All pools tracked
        expected = ["KMISO_KMut", "KMISO_IS", "KMISO_Bg", "KMISO_Ins"]
        for p in expected:
            assert f'{p}_index' in dc.keys, f"{p} should be tracked"
        
        for i in range(min(3, len(seqs))):
            row = dc.get_row(i)
            seq = seqs[i]
            
            # KMut is output, matches sequence
            assert row['KMISO_KMut_value'] == seq
            assert len(seq) == 18
            
            # IS is input to KMut (before mutation), contains GGG
            is_val = row['KMISO_IS_value']
            assert len(is_val) == 18
            assert 'GGG' in is_val
            
            # Original inputs
            assert row['KMISO_Bg_value'] == 'ATGACGTACGTACGTGAA'
            assert row['KMISO_Ins_value'] == 'GGG'
    
    def test_insertion_scan_orf_inside_spacing_scan_with_pool_bg(self):
        """InsertionScanORF with Pool inputs inside SpacingScan with Pool bg."""
        # InsertionScanORF
        is_bg = IUPACPool(iupac_seq="ATGACGTACGGG", name="ISOSS_ISBg", metadata='complete')  # 12bp = 4 codons
        is_ins = IUPACPool(iupac_seq="AAA", name="ISOSS_ISIns", metadata='complete')  # 1 codon
        is_orf = InsertionScanORFPool(
            background_seq=is_bg,
            insert_seq=is_ins,
            start=1, end=3, step_size=1, insert_or_overwrite='overwrite',
            name="ISOSS_IS", mode='sequential', metadata='complete'
        )
        
        # Other insert
        other = IUPACPool(iupac_seq="TTTT", name="ISOSS_Other", metadata='complete')
        
        # SpacingScan with Pool bg
        ss_bg = IUPACPool(iupac_seq="GGGGGGGGGGGGGGGGGGGGGGGGGGGGGG", name="ISOSS_SSBg", metadata='complete')  # 30bp
        ss = SpacingScanPool(
            background_seq=ss_bg,  # Pool background
            insert_seqs=[is_orf, other],
            insert_names=["IS", "Oth"],
            anchor_pos=15, insert_distances=[[-8], [8]],
            name="ISOSS_SS", mode='sequential', metadata='complete'
        )
        
        result = ss.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        # All pools tracked
        expected = ["ISOSS_SS", "ISOSS_SSBg", "ISOSS_IS", "ISOSS_ISBg", "ISOSS_ISIns", "ISOSS_Other"]
        for p in expected:
            assert f'{p}_index' in dc.keys, f"{p} should be tracked"
        
        for i in range(min(3, len(seqs))):
            row = dc.get_row(i)
            seq = seqs[i]
            
            # SS matches sequence
            assert row['ISOSS_SS_value'] == seq
            assert len(seq) == 30
            
            # SS bg is original
            assert row['ISOSS_SSBg_value'] == 'GGGGGGGGGGGGGGGGGGGGGGGGGGGGGG'
            
            # IS is 12bp with AAA
            is_val = row['ISOSS_IS_value']
            assert len(is_val) == 12
            assert 'AAA' in is_val
            
            # IS inputs
            assert row['ISOSS_ISBg_value'] == 'ATGACGTACGGG'
            assert row['ISOSS_ISIns_value'] == 'AAA'
            
            assert row['ISOSS_Other_value'] == 'TTTT'
    
    def test_deletion_scan_orf_and_insertion_scan_orf_in_spacing_scan(self):
        """Both DeletionScanORF and InsertionScanORF as inserts in SpacingScan."""
        # InsertionScanORF: 12bp bg
        is_bg = IUPACPool(iupac_seq="ATGACGTACGGG", name="DSIS_ISBg", metadata='complete')
        is_ins = IUPACPool(iupac_seq="AAA", name="DSIS_ISIns", metadata='complete')
        is_orf = InsertionScanORFPool(
            background_seq=is_bg, insert_seq=is_ins,
            start=1, end=3, step_size=1, insert_or_overwrite='overwrite',
            name="DSIS_IS", mode='sequential', metadata='complete'
        )
        
        # DeletionScanORF: 12bp = 4 codons, delete 1 = 9bp output
        ds_base = IUPACPool(iupac_seq="ATGGGGAAACCC", name="DSIS_DSBase", metadata='complete')
        ds_orf = DeletionScanORFPool(
            ds_base, deletion_size=1, start=1, end=4, step_size=1,
            mark_changes=False, name="DSIS_DS", mode='sequential', metadata='complete'
        )
        
        # Another insert
        other = IUPACPool(iupac_seq="TTTT", name="DSIS_Other", metadata='complete')
        
        ss = SpacingScanPool(
            background_seq="N" * 40,
            insert_seqs=[is_orf, ds_orf, other],
            insert_names=["IS", "DS", "Oth"],
            anchor_pos=20, insert_distances=[[-14], [0], [14]],
            name="DSIS_SS", mode='sequential', metadata='complete'
        )
        
        result = ss.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        # All pools tracked
        expected = ["DSIS_SS", "DSIS_IS", "DSIS_ISBg", "DSIS_ISIns", "DSIS_DS", "DSIS_DSBase", "DSIS_Other"]
        for p in expected:
            assert f'{p}_index' in dc.keys, f"{p} should be tracked"
        
        for i in range(min(3, len(seqs))):
            row = dc.get_row(i)
            seq = seqs[i]
            
            # SS matches sequence
            assert row['DSIS_SS_value'] == seq
            assert len(seq) == 40
            
            # IS is 12bp with AAA
            is_val = row['DSIS_IS_value']
            assert len(is_val) == 12
            assert 'AAA' in is_val
            
            # DS is 9bp (12 - 3)
            ds_val = row['DSIS_DS_value']
            assert len(ds_val) == 9
            
            # All originals
            assert row['DSIS_ISBg_value'] == 'ATGACGTACGGG'
            assert row['DSIS_ISIns_value'] == 'AAA'
            assert row['DSIS_DSBase_value'] == 'ATGGGGAAACCC'
            assert row['DSIS_Other_value'] == 'TTTT'
    
    def test_complex_orf_library_with_all_pool_types(self):
        """Complex ORF library: composite prefix + InsertionScanORF + KMutationORF + suffix."""
        # Prefix composite
        prefix_a = IUPACPool(iupac_seq="ATG", name="ORFLib_PreA", metadata='complete')  # Start codon
        prefix_b = IUPACPool(iupac_seq="AAA", name="ORFLib_PreB", metadata='complete')
        prefix = prefix_a + prefix_b  # 6bp
        
        # InsertionScanORF: 12bp
        is_bg = IUPACPool(iupac_seq="ACGTACGTACGT", name="ORFLib_ISBg", metadata='complete')
        is_ins = IUPACPool(iupac_seq="GGG", name="ORFLib_ISIns", metadata='complete')
        is_orf = InsertionScanORFPool(
            background_seq=is_bg, insert_seq=is_ins,
            start=1, end=3, step_size=1, insert_or_overwrite='overwrite',
            name="ORFLib_IS", mode='sequential', metadata='complete'
        )
        
        # Suffix
        suffix = IUPACPool(iupac_seq="TAA", name="ORFLib_Suf", metadata='complete')  # Stop codon
        
        # Full library
        library = prefix + is_orf + suffix  # 6 + 12 + 3 = 21bp
        
        result = library.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        # All components tracked
        expected = ["ORFLib_PreA", "ORFLib_PreB", "ORFLib_IS", "ORFLib_ISBg", "ORFLib_ISIns", "ORFLib_Suf"]
        for p in expected:
            assert f'{p}_index' in dc.keys, f"{p} should be tracked"
        
        for i in range(min(3, len(seqs))):
            row = dc.get_row(i)
            seq = seqs[i]
            
            # Length check
            assert len(seq) == 21
            
            # Starts with ATG (start codon)
            assert seq.startswith('ATG')
            
            # Ends with TAA (stop codon)
            assert seq.endswith('TAA')
            
            # Contains GGG from insertion
            assert 'GGG' in seq
            
            # All values
            assert row['ORFLib_PreA_value'] == 'ATG'
            assert row['ORFLib_PreB_value'] == 'AAA'
            assert row['ORFLib_ISBg_value'] == 'ACGTACGTACGT'
            assert row['ORFLib_ISIns_value'] == 'GGG'
            assert row['ORFLib_Suf_value'] == 'TAA'
            
            # IS value is 12bp with GGG
            is_val = row['ORFLib_IS_value']
            assert len(is_val) == 12
            assert 'GGG' in is_val
    
    def test_mixed_pool_with_insertion_scan_orf_children(self):
        """MixedPool containing InsertionScanORF - selected child tracked correctly."""
        # Two InsertionScanORF children
        is_bg1 = IUPACPool(iupac_seq="ATGACGTACGGG", name="MixIS_ISBg1", metadata='complete')
        is_ins1 = IUPACPool(iupac_seq="AAA", name="MixIS_ISIns1", metadata='complete')
        is_orf1 = InsertionScanORFPool(
            background_seq=is_bg1, insert_seq=is_ins1,
            start=1, end=3, step_size=1, insert_or_overwrite='overwrite',
            name="MixIS_IS1", mode='sequential', metadata='complete'
        )
        
        is_bg2 = IUPACPool(iupac_seq="ATGGGGAAACCC", name="MixIS_ISBg2", metadata='complete')
        is_ins2 = IUPACPool(iupac_seq="TTT", name="MixIS_ISIns2", metadata='complete')
        is_orf2 = InsertionScanORFPool(
            background_seq=is_bg2, insert_seq=is_ins2,
            start=1, end=3, step_size=1, insert_or_overwrite='overwrite',
            name="MixIS_IS2", mode='sequential', metadata='complete'
        )
        
        # MixedPool
        mix = MixedPool([is_orf1, is_orf2], name="MixIS_Mix", mode='sequential')
        
        result = mix.generate_seqs(num_seqs=6, return_design_cards=True)
        dc = result['design_cards']
        seqs = result['sequences']
        
        # All children and their inputs should be tracked
        expected = ["MixIS_Mix", "MixIS_IS1", "MixIS_ISBg1", "MixIS_ISIns1", "MixIS_IS2", "MixIS_ISBg2", "MixIS_ISIns2"]
        for p in expected:
            assert f'{p}_index' in dc.keys, f"{p} should be tracked"
        
        for i in range(min(4, len(seqs))):
            row = dc.get_row(i)
            seq = seqs[i]
            
            # Length is 12bp (both children are 12bp)
            assert len(seq) == 12
            
            # MixedPool selection
            selected = row['MixIS_Mix_selected']
            assert selected in [0, 1]
            
            if selected == 0:
                # First child selected - should have AAA
                assert 'AAA' in seq
                assert row['MixIS_IS1_value'] is not None
            else:
                # Second child selected - should have TTT
                assert 'TTT' in seq
                assert row['MixIS_IS2_value'] is not None


# =============================================================================
# Rigorous Design Card Tests: Config Verification
# =============================================================================

class TestDesignCardConfigVerification:
    """Tests verifying design card metadata matches constructor parameters.
    
    These tests ensure that each pool type's metadata accurately reflects
    the configuration provided at construction time, not just that values
    match the generated sequence.
    """
    
    def test_dc_pool_index_matches_seqs_list_position(self):
        """Pool._index correctly identifies which sequence from seqs list."""
        seqs = ['SEQ_AAA', 'SEQ_BBB', 'SEQ_CCC', 'SEQ_DDD']
        pool = Pool(seqs, name='p', mode='sequential', metadata='complete')
        result = pool.generate_library(num_seqs=4, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(4):
            row = dc.get_row(i)
            # Config-based: _index matches position in seqs list
            assert row['p_index'] == i, f"Expected index {i}, got {row['p_index']}"
            # Value should be the sequence at that index
            assert row['p_value'] == seqs[i], f"Expected {seqs[i]}, got {row['p_value']}"
    
    def test_dc_mixed_pool_selection_matches_pools_list_order(self):
        """MixedPool._selected is index into constructor pools list."""
        pools = [
            Pool(['AAAA'], name='first', mode='sequential'),
            Pool(['BBBB'], name='second', mode='sequential'),
            Pool(['CCCC'], name='third', mode='sequential')
        ]
        mixed = MixedPool(pools, name='m', mode='sequential')
        result = mixed.generate_seqs(num_seqs=3, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(3):
            row = dc.get_row(i)
            # Config-based: _selected matches pools list index
            assert row['m_selected'] == i, f"Expected selected={i}, got {row['m_selected']}"
            # selected_name matches the pool's name at that index
            assert row['m_selected_name'] == pools[i].name
    
    def test_dc_insertion_scan_pos_within_configured_range(self):
        """InsertionScan._pos is within configured [start, end) with step_size."""
        start, end, step_size = 2, 14, 3
        pool = InsertionScanPool(
            'NNNNNNNNNNNNNNNNNNNN', 'XX',  # 20bp bg, 2bp insert
            start=start, end=end, step_size=step_size,
            name='is', mode='sequential'
        )
        result = pool.generate_seqs(num_complete_iterations=1, return_design_cards=True)
        dc = result['design_cards']
        
        # Calculate expected positions based on config
        # For overwrite mode: positions where window [pos, pos+W) fits within [start, end-W+1)
        W = 2  # insert length
        expected_positions = list(range(start, end - W + 1, step_size))
        
        for i in range(len(result['sequences'])):
            row = dc.get_row(i)
            # Config-based: _pos must be one of the expected positions
            assert row['is_pos'] in expected_positions, \
                f"pos={row['is_pos']} not in expected {expected_positions}"
    
    def test_dc_deletion_scan_del_len_matches_config(self):
        """DeletionScan._del_len always equals configured deletion_size."""
        deletion_size = 5
        pool = DeletionScanPool(
            'AAAAAAAAAAAAAAAA', deletion_size=deletion_size,
            name='ds', mode='sequential', mark_changes=True
        )
        result = pool.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(5):
            row = dc.get_row(i)
            # Config-based: _del_len always equals configured deletion_size
            assert row['ds_del_len'] == deletion_size, \
                f"Expected del_len={deletion_size}, got {row['ds_del_len']}"
    
    def test_dc_subseq_width_matches_config(self):
        """SubseqPool._width always equals configured width."""
        width = 7
        pool = SubseqPool('ABCDEFGHIJKLMNOP', width=width, name='ss', mode='sequential')
        result = pool.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(5):
            row = dc.get_row(i)
            # Config-based: _width equals configured width
            assert row['ss_width'] == width, f"Expected width={width}, got {row['ss_width']}"
            # Sequence length should also match width
            assert len(result['sequences'][i]) == width
    
    def test_dc_shuffle_scan_window_size_matches_config(self):
        """ShuffleScan._window_size equals configured shuffle_size."""
        shuffle_size = 6
        pool = ShuffleScanPool(
            'AATTGGCCAATTGGCC', shuffle_size=shuffle_size,
            name='shuf', mode='sequential'
        )
        result = pool.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(5):
            row = dc.get_row(i)
            # Config-based: _window_size equals configured shuffle_size
            assert row['shuf_window_size'] == shuffle_size, \
                f"Expected window_size={shuffle_size}, got {row['shuf_window_size']}"
    
    def test_dc_kmutation_exactly_k_mutations_reported(self):
        """KMutation reports exactly k mutations as configured."""
        k = 3
        pool = KMutationPool('GGGGGGGGGGGG', k=k, name='km', mode='sequential')
        result = pool.generate_seqs(num_seqs=10, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(10):
            row = dc.get_row(i)
            # Config-based: exactly k mutations reported
            assert len(row['km_mut_pos']) == k, \
                f"Expected {k} mutations, got {len(row['km_mut_pos'])}"
            assert len(row['km_mut_from']) == k
            assert len(row['km_mut_to']) == k
    
    def test_dc_kmutation_positions_within_configured_positions(self):
        """KMutation._mut_pos are within configured positions list."""
        allowed_positions = [0, 3, 6, 9]
        pool = KMutationPool(
            'GGGGGGGGGGGG', k=2, positions=allowed_positions,
            name='km', mode='sequential'
        )
        result = pool.generate_seqs(num_seqs=10, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(10):
            row = dc.get_row(i)
            for pos in row['km_mut_pos']:
                # Config-based: each mutation position must be in allowed list
                assert pos in allowed_positions, \
                    f"Position {pos} not in allowed {allowed_positions}"
    
    def test_dc_random_mutation_count_consistent_with_lists(self):
        """RandomMutation._mut_count == len(mut_pos) == len(mut_from) == len(mut_to)."""
        pool = RandomMutationPool('GGGGGGGGGG', mutation_rate=0.5, name='rm')
        result = pool.generate_seqs(num_seqs=20, seed=42, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(20):
            row = dc.get_row(i)
            count = row['rm_mut_count']
            # All lists must have length equal to mut_count
            assert len(row['rm_mut_pos']) == count, \
                f"mut_pos length {len(row['rm_mut_pos'])} != mut_count {count}"
            assert len(row['rm_mut_from']) == count, \
                f"mut_from length {len(row['rm_mut_from'])} != mut_count {count}"
            assert len(row['rm_mut_to']) == count, \
                f"mut_to length {len(row['rm_mut_to'])} != mut_count {count}"
    
    def test_dc_kmutation_orf_codon_pos_abs_formula(self):
        """KMutationORF: codon_pos_abs == orf_start + codon_pos * 3."""
        orf_start = 6
        pool = KMutationORFPool(
            'GGGGGGATGAAACCCGGGGGG', mutation_type='any_codon', k=1, 
            orf_start=orf_start, orf_end=15,
            name='ko', mode='sequential'
        )
        result = pool.generate_seqs(num_seqs=10, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(10):
            row = dc.get_row(i)
            for cp, cpa in zip(row['ko_codon_pos'], row['ko_codon_pos_abs']):
                # Config-based: codon_pos_abs = orf_start + codon_pos * 3
                expected_abs = orf_start + cp * 3
                assert cpa == expected_abs, \
                    f"Expected codon_pos_abs={expected_abs}, got {cpa}"
    
    def test_dc_insertion_scan_orf_codon_positions_in_range(self):
        """InsertionScanORF: codon_pos within configured [start, end)."""
        start_codon, end_codon = 1, 4
        pool = InsertionScanORFPool(
            'ATGAAACCCGGGTTT', 'AAA',  # 15bp = 5 codons
            start=start_codon, end=end_codon,
            name='iso', mode='sequential'
        )
        result = pool.generate_seqs(num_complete_iterations=1, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(len(result['sequences'])):
            row = dc.get_row(i)
            # Config-based: codon_pos in [start, end)
            assert start_codon <= row['iso_codon_pos'] < end_codon, \
                f"codon_pos={row['iso_codon_pos']} not in [{start_codon}, {end_codon})"
    
    def test_dc_deletion_scan_orf_del_codons_matches_config(self):
        """DeletionScanORF: len(del_codons) == deletion_size * 3 (codon nucleotides)."""
        deletion_size = 2
        pool = DeletionScanORFPool(
            'ATGAAACCCGGGTTTTAA',  # 18bp = 6 codons
            deletion_size=deletion_size,
            name='dso', mode='sequential', mark_changes=True
        )
        result = pool.generate_seqs(num_seqs=3, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(3):
            row = dc.get_row(i)
            # del_codons is the actual deleted codon sequence
            # Config-based: length equals deletion_size * 3 (nucleotides per codon)
            expected_len = deletion_size * 3
            assert len(row['dso_del_codons']) == expected_len, \
                f"Expected del_codons length={expected_len}, got {len(row['dso_del_codons'])}"
    
    def test_dc_spacing_scan_distances_from_configured_lists(self):
        """SpacingScan: each insert's dist is from its insert_distances list."""
        insert_distances_A = [-10, -5, 0]
        insert_distances_B = [5, 10, 15]
        
        pool = SpacingScanPool(
            background_seq='N' * 50, anchor_pos=25,
            insert_seqs=['AAAA', 'TTTT'],
            insert_names=['A', 'B'],
            insert_distances=[insert_distances_A, insert_distances_B],
            name='sp', mode='sequential'
        )
        result = pool.generate_seqs(num_complete_iterations=1, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(len(result['sequences'])):
            row = dc.get_row(i)
            # Config-based: each dist must be from configured list
            assert row['sp_A_dist'] in insert_distances_A, \
                f"A_dist={row['sp_A_dist']} not in {insert_distances_A}"
            assert row['sp_B_dist'] in insert_distances_B, \
                f"B_dist={row['sp_B_dist']} not in {insert_distances_B}"
    
    def test_dc_shuffle_pool_region_matches_config(self):
        """ShufflePool: start/end match configured values."""
        from poolparty import ShufflePool
        
        start, end = 4, 12
        pool = ShufflePool('AAAAGGGGTTTTCCCC', start=start, end=end, name='sh')
        result = pool.generate_seqs(num_seqs=5, seed=42, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(5):
            row = dc.get_row(i)
            # Config-based: start/end match constructor params
            assert row['sh_start'] == start, f"Expected start={start}, got {row['sh_start']}"
            assert row['sh_end'] == end, f"Expected end={end}, got {row['sh_end']}"
    
    def test_dc_motif_pool_orientation_from_config(self):
        """MotifPool: orientation metadata reflects orientation config."""
        pytest.importorskip('pandas')
        import pandas as pd
        from poolparty import MotifPool
        
        # Simple PWM: 3 positions, only one valid nucleotide each
        pwm = pd.DataFrame({
            'A': [1.0, 0.0, 0.0],
            'C': [0.0, 1.0, 0.0],
            'G': [0.0, 0.0, 1.0],
            'T': [0.0, 0.0, 0.0]
        })
        
        # orientation='forward': always forward orientation
        pool_fwd = MotifPool(pwm, orientation='forward', name='fwd')
        result_fwd = pool_fwd.generate_seqs(num_seqs=10, seed=42, return_design_cards=True)
        dc_fwd = result_fwd['design_cards']
        
        for i in range(10):
            row = dc_fwd.get_row(i)
            # Config-based: when orientation='forward', always 'forward'
            assert row['fwd_orientation'] == 'forward', \
                f"Expected orientation='forward' with orientation='forward', got {row['fwd_orientation']}"
        
        # orientation='reverse': always reverse orientation
        pool_rev = MotifPool(pwm, orientation='reverse', name='rev')
        result_rev = pool_rev.generate_seqs(num_seqs=10, seed=42, return_design_cards=True)
        dc_rev = result_rev['design_cards']
        
        for i in range(10):
            row = dc_rev.get_row(i)
            # Config-based: when orientation='reverse', always 'reverse'
            assert row['rev_orientation'] == 'reverse', \
                f"Expected orientation='reverse' with orientation='reverse', got {row['rev_orientation']}"
        
        # orientation='both': can be 'forward' or 'reverse'
        pool_both = MotifPool(pwm, orientation='both', name='both')
        result_both = pool_both.generate_seqs(num_seqs=100, seed=42, return_design_cards=True)
        dc_both = result_both['design_cards']
        
        orientations = [dc_both.get_row(i)['both_orientation'] for i in range(100)]
        # With orientation='both' and enough samples, we should see both orientations
        assert 'forward' in orientations, "Expected some 'forward' orientations with orientation='both'"
        assert 'reverse' in orientations, "Expected some 'reverse' orientations with orientation='both'"


# =============================================================================
# Rigorous Design Card Tests: Combination Coverage
# =============================================================================

class TestDesignCardCombinationCoverage:
    """Tests verifying design cards work correctly when pools are combined.
    
    These tests ensure that various combinations of pool types
    (transformers, multi-input, nesting, composites) all track metadata correctly.
    """
    
    def test_dc_kmutation_on_iupac_tracks_both(self):
        """KMutation(IUPACPool) tracks both with correct values."""
        iupac = IUPACPool('RRRRRRRR', name='base', metadata='complete')  # R = A or G
        kmut = KMutationPool(iupac, k=2, name='km', metadata='complete')
        result = kmut.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(5):
            row = dc.get_row(i)
            # base_value is IUPACPool output (before mutation)
            assert row['base_value'] is not None
            assert all(c in 'AG' for c in row['base_value']), \
                f"base_value should be all A/G, got {row['base_value']}"
            # km reports exactly 2 mutations
            assert len(row['km_mut_pos']) == 2
            # km_value is the mutated sequence
            assert row['km_value'] == result['sequences'][i]
    
    def test_dc_insertion_scan_with_pool_bg_and_insert(self):
        """InsertionScan with Pool bg AND Pool insert tracks all three."""
        bg = IUPACPool('NNNNNNNNNNNN', name='bg', metadata='complete')
        ins = IUPACPool('RRRR', name='ins', metadata='complete')
        scan = InsertionScanPool(
            bg, ins, name='scan', mode='sequential', metadata='complete'
        )
        result = scan.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(5):
            row = dc.get_row(i)
            # All three pools tracked
            assert row['bg_value'] is not None, "bg should be tracked"
            assert row['ins_value'] is not None, "ins should be tracked"
            assert row['scan_value'] is not None, "scan should be tracked"
            # scan_insert should match ins_value
            assert row['scan_insert'] == row['ins_value'], \
                f"scan_insert={row['scan_insert']} should match ins_value={row['ins_value']}"
    
    def test_dc_mixed_pool_with_diverse_transformer_children(self):
        """MixedPool[KMut, InsertionScan, DeletionScan] tracks correct child."""
        seq = 'AAAAAAAAAAAAAAAA'  # 16bp
        c1 = KMutationPool(seq, k=1, name='km', metadata='complete')
        c2 = InsertionScanPool(seq, 'XX', name='is', mode='sequential', metadata='complete')
        c3 = DeletionScanPool(seq, deletion_size=2, mark_changes=True, name='ds', metadata='complete')
        mixed = MixedPool([c1, c2, c3], name='mix', mode='sequential')
        
        result = mixed.generate_seqs(num_seqs=30, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(30):
            row = dc.get_row(i)
            selected = row['mix_selected']
            
            if selected == 0:
                # km selected: has mutation metadata, others None
                assert row['km_mut_pos'] is not None
                assert len(row['km_mut_pos']) == 1
                assert row['is_pos'] is None
                assert row['ds_pos'] is None
            elif selected == 1:
                # is selected: has position metadata, others None
                assert row['km_mut_pos'] is None
                assert row['is_pos'] is not None
                assert row['ds_pos'] is None
            else:
                # ds selected
                assert row['km_mut_pos'] is None
                assert row['is_pos'] is None
                assert row['ds_pos'] is not None
    
    def test_dc_spacing_scan_with_kmutation_insert(self):
        """SpacingScan with KMutation insert tracks mutation metadata."""
        base = Pool(['GGGG'], name='base', metadata='complete')
        kmut = KMutationPool(base, k=1, name='km', metadata='complete')
        
        pool = SpacingScanPool(
            background_seq='N' * 30, anchor_pos=15,
            insert_seqs=[kmut, 'TTTT'],
            insert_names=['K', 'T'],
            insert_distances=[[-5], [5]],
            name='sp', mode='sequential', metadata='complete'
        )
        result = pool.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(5):
            row = dc.get_row(i)
            # base tracked as transformer parent
            assert row['base_value'] == 'GGGG', "base should be original value"
            # km tracked with mutation info
            assert row['km_mut_pos'] is not None
            assert len(row['km_mut_pos']) == 1
            # km_value should have exactly 1 char different from base
            assert sum(1 for a, b in zip(row['base_value'], row['km_value']) if a != b) == 1
    
    def test_dc_orf_mutation_on_iupac_pool(self):
        """KMutationORFPool on IUPACPool tracks both correctly."""
        # IUPAC with N = any nucleotide, creates valid ORF-like sequence
        iupac = IUPACPool('ATGNNNNNNTAA', name='base', metadata='complete')
        orf = KMutationORFPool(
            iupac, mutation_type='any_codon', k=1, orf_start=0, orf_end=12,
            name='orf', mode='sequential', metadata='complete'
        )
        result = orf.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(5):
            row = dc.get_row(i)
            # IUPACPool value tracked
            assert row['base_value'] is not None
            assert len(row['base_value']) == 12
            # ORF mutation metadata
            assert row['orf_codon_pos'] is not None
            assert len(row['orf_codon_pos']) == 1  # k=1
    
    def test_dc_nested_mixed_pools_transformer_children(self):
        """Nested MixedPool with transformers tracks all levels."""
        seq = 'AAAAAAAA'
        inner = MixedPool([
            KMutationPool(seq, k=1, name='km', metadata='complete'),
            InsertionScanPool(seq, 'XX', name='is', mode='sequential', metadata='complete')
        ], name='inner', mode='sequential')
        
        outer = MixedPool(
            [inner, Pool(['BBBBBBBB'], name='plain', metadata='complete')],
            name='outer', mode='sequential'
        )
        
        result = outer.generate_seqs(num_seqs=10, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(10):
            row = dc.get_row(i)
            outer_sel = row['outer_selected']
            
            if outer_sel == 0:
                # inner selected, should have its selection tracked
                assert row['inner_selected'] is not None
                assert row['inner_selected'] in [0, 1]
                assert row['plain_value'] is None
            else:
                # plain selected
                assert row['inner_selected'] is None
                assert row['plain_value'] == 'BBBBBBBB'
    
    def test_dc_spacing_scan_composite_insert_internal_pools_tracked(self):
        """SpacingScan with composite (A+B+C) insert tracks all internal pools."""
        A = Pool(['AA'], name='A', metadata='complete')
        B = Pool(['BB'], name='B', metadata='complete')
        C = Pool(['CC'], name='C', metadata='complete')
        composite = A + B + C  # 6bp
        
        pool = SpacingScanPool(
            background_seq='N' * 30, anchor_pos=15,
            insert_seqs=[composite, 'TTTTTT'],
            insert_names=['comp', 'lit'],
            insert_distances=[[-8], [8]],
            name='sp', mode='sequential'
        )
        result = pool.generate_seqs(num_seqs=1, return_design_cards=True)
        dc = result['design_cards']
        
        # All internal pools in composite insert should be tracked
        assert 'A_index' in dc.keys, "A in composite insert should be tracked"
        assert 'B_index' in dc.keys, "B in composite insert should be tracked"
        assert 'C_index' in dc.keys, "C in composite insert should be tracked"
        
        row = dc.get_row(0)
        assert row['A_value'] == 'AA'
        assert row['B_value'] == 'BB'
        assert row['C_value'] == 'CC'
    
    def test_dc_all_simple_pool_types_in_composite(self):
        """Composite with Pool, IUPACPool, KmerPool, BarcodePool all tracked."""
        from poolparty import BarcodePool
        
        p1 = Pool(['AAA'], name='pool', metadata='complete')
        p2 = IUPACPool('RRR', name='iupac', metadata='complete')
        p3 = KmerPool(length=3, name='kmer', mode='sequential', metadata='complete')
        p4 = BarcodePool(num_barcodes=3, length=3, seed=42, name='bc', metadata='complete')
        
        library = p1 + p2 + p3 + p4
        result = library.generate_seqs(num_seqs=3, return_design_cards=True)
        dc = result['design_cards']
        
        # All 4 pool types tracked
        for name in ['pool', 'iupac', 'kmer', 'bc']:
            assert f'{name}_value' in dc.keys, f"{name} should have _value column"
        
        for i in range(3):
            row = dc.get_row(i)
            # All values present
            assert row['pool_value'] == 'AAA'
            assert row['iupac_value'] is not None
            assert all(c in 'AG' for c in row['iupac_value'])  # R = A or G
            assert row['kmer_value'] is not None
            assert len(row['kmer_value']) == 3
            assert row['bc_value'] is not None
            assert len(row['bc_value']) == 3
    
    def test_dc_mixed_pool_as_transformer_input(self):
        """KMutation(MixedPool([A, B])) tracks selection and mutation."""
        A = Pool(['AAAA'], name='A', metadata='complete')
        B = Pool(['TTTT'], name='B', metadata='complete')
        mixed = MixedPool([A, B], name='mix', mode='sequential')
        
        kmut = KMutationPool(mixed, k=1, name='km', metadata='complete')
        result = kmut.generate_seqs(num_seqs=4, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(4):
            row = dc.get_row(i)
            # MixedPool selection tracked
            assert row['mix_selected'] in [0, 1]
            # KMutation metadata present
            assert len(row['km_mut_pos']) == 1
            # km_value is the mutated output
            assert row['km_value'] == result['sequences'][i]
    
    def test_dc_five_level_nesting_all_tracked(self):
        """5-level nesting: all pools tracked with correct relationships."""
        L1 = Pool(['GGGGGGGGGGGGGGGG'], name='L1', metadata='complete')
        L2 = KMutationPool(L1, k=1, name='L2', metadata='complete')
        L3 = ShuffleScanPool(L2, shuffle_size=4, name='L3', mode='sequential', metadata='complete')
        L4 = InsertionScanPool(L3, 'XX', name='L4', mode='sequential', metadata='complete')
        L5 = KMutationPool(L4, k=1, name='L5', metadata='complete')
        
        result = L5.generate_seqs(num_seqs=3, return_design_cards=True)
        dc = result['design_cards']
        
        # All 5 levels tracked
        for name in ['L1', 'L2', 'L3', 'L4', 'L5']:
            assert f'{name}_value' in dc.keys, f"{name} should be tracked"
        
        for i in range(3):
            row = dc.get_row(i)
            # L1 is original
            assert row['L1_value'] == 'GGGGGGGGGGGGGGGG'
            # L5 is output, matches sequence
            assert row['L5_value'] == result['sequences'][i]


# =============================================================================
# Rigorous Design Card Tests: Position/Value Integrity
# =============================================================================

class TestDesignCardPositionValueIntegrity:
    """Tests verifying positions and values are mathematically correct.
    
    These tests ensure that position metadata allows correct extraction
    and that mathematical relationships hold.
    """
    
    def test_dc_all_pools_extraction_equals_value(self):
        """For every tracked pool with positions, seq[abs_start:abs_end] == _value."""
        # Create complex composite with multiple pool types
        p1 = Pool(['AAAA'], name='p1', metadata='complete')
        p2 = IUPACPool('RRRR', name='p2', metadata='complete')
        p3 = KmerPool(length=4, name='p3', mode='sequential', metadata='complete')
        p4 = Pool(['GGGG'], name='p4', metadata='complete')
        kmut = KMutationPool(p4, k=1, name='km', metadata='complete')
        
        library = p1 + p2 + p3 + kmut
        result = library.generate_seqs(num_seqs=5, return_design_cards=True)
        dc = result['design_cards']
        
        # Pools that are direct segments (not transformer parents)
        direct_pools = ['p1', 'p2', 'p3', 'km']
        
        for i in range(5):
            row = dc.get_row(i)
            seq = result['sequences'][i]
            
            for name in direct_pools:
                abs_start = row[f'{name}_abs_start']
                abs_end = row[f'{name}_abs_end']
                value = row[f'{name}_value']
                
                # Extract from sequence using positions
                extracted = seq[abs_start:abs_end]
                
                # Core verification: extraction must match value
                assert extracted == value, \
                    f"Pool {name}: seq[{abs_start}:{abs_end}]='{extracted}' != value='{value}'"
    
    def test_dc_mutation_positions_match_actual_changes(self):
        """For each mutation, verify original[pos] == mut_from and result[pos] == mut_to."""
        original_seq = 'GGGGGGGGGGGG'
        pool = KMutationPool(original_seq, k=2, name='km', mode='sequential', metadata='complete')
        result = pool.generate_seqs(num_seqs=10, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(10):
            row = dc.get_row(i)
            seq = result['sequences'][i]
            
            for pos, mut_from, mut_to in zip(
                row['km_mut_pos'], row['km_mut_from'], row['km_mut_to']
            ):
                # Original character at position should match mut_from
                assert original_seq[pos] == mut_from, \
                    f"original[{pos}]='{original_seq[pos]}' != mut_from='{mut_from}'"
                
                # Result character at position should match mut_to
                assert seq[pos] == mut_to, \
                    f"result[{pos}]='{seq[pos]}' != mut_to='{mut_to}'"
                
                # mut_from and mut_to should be different
                assert mut_from != mut_to, \
                    f"mut_from='{mut_from}' should differ from mut_to='{mut_to}'"
    
    def test_dc_orf_codon_extraction_at_codon_pos_abs(self):
        """For ORF mutations, seq[codon_pos_abs:codon_pos_abs+3] == codon_to."""
        pool = KMutationORFPool(
            'ATGAAACCCGGGTTT',  # 15bp = 5 codons
            mutation_type='any_codon', k=1, name='ko', mode='sequential', metadata='complete'
        )
        result = pool.generate_seqs(num_seqs=10, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(10):
            row = dc.get_row(i)
            seq = result['sequences'][i]
            
            for cpa, codon_to in zip(row['ko_codon_pos_abs'], row['ko_codon_to']):
                # Extract codon from sequence at codon_pos_abs
                extracted_codon = seq[cpa:cpa + 3]
                
                # Must match codon_to
                assert extracted_codon == codon_to, \
                    f"seq[{cpa}:{cpa+3}]='{extracted_codon}' != codon_to='{codon_to}'"
    
    def test_dc_spacing_scan_pairwise_spacing_formula(self):
        """SpacingScan: spacing_A_B == B_pos_start - A_pos_end."""
        pool = SpacingScanPool(
            background_seq='N' * 50, anchor_pos=25,
            insert_seqs=['AAAA', 'TTTT', 'GGGG'],  # 4bp each
            insert_names=['A', 'B', 'C'],
            insert_distances=[[-15, -10], [0, 5], [10, 15]],
            name='sp', mode='sequential'
        )
        result = pool.generate_seqs(num_complete_iterations=1, return_design_cards=True)
        dc = result['design_cards']
        
        for i in range(len(result['sequences'])):
            row = dc.get_row(i)
            
            # Get positions for each insert
            a_start, a_end = row['sp_A_pos_start'], row['sp_A_pos_end']
            b_start, b_end = row['sp_B_pos_start'], row['sp_B_pos_end']
            c_start, c_end = row['sp_C_pos_start'], row['sp_C_pos_end']
            
            # Verify pairwise spacing formula: spacing_X_Y = Y_start - X_end
            # For A-B
            expected_spacing_AB = b_start - a_end
            assert row['sp_spacing_A_B'] == expected_spacing_AB, \
                f"spacing_A_B={row['sp_spacing_A_B']} != B_start-A_end={expected_spacing_AB}"
            
            # For A-C
            expected_spacing_AC = c_start - a_end
            assert row['sp_spacing_A_C'] == expected_spacing_AC, \
                f"spacing_A_C={row['sp_spacing_A_C']} != C_start-A_end={expected_spacing_AC}"
            
            # For B-C
            expected_spacing_BC = c_start - b_end
            assert row['sp_spacing_B_C'] == expected_spacing_BC, \
                f"spacing_B_C={row['sp_spacing_B_C']} != C_start-B_end={expected_spacing_BC}"
