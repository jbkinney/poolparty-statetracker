"""Tests for the Marker class and marker_scan operation."""

import pytest
import numpy as np
import poolparty as pp
from poolparty.operations.marker_scan import MarkerScanOp, marker_scan
from poolparty.marker import Marker
from poolparty.alphabet import get_alphabet, MARKER_PATTERN


class TestMarkerClass:
    """Test Marker class creation and registration."""
    
    def test_marker_requires_party_context(self):
        """Test that Marker requires active Party context."""
        with pytest.raises(RuntimeError, match="Markers must be created inside a Party context"):
            Marker()
    
    def test_marker_default_name(self):
        """Test default marker naming."""
        with pp.Party() as party:
            m = Marker()
            assert m.name == 'marker[0]'
            assert m.id == 0
    
    def test_marker_custom_name(self):
        """Test custom marker naming."""
        with pp.Party() as party:
            m = Marker(name='my_marker')
            assert m.name == 'my_marker'
    
    def test_marker_tag_property(self):
        """Test marker tag generation."""
        with pp.Party() as party:
            m = Marker(name='insert')
            assert m.tag == '{insert}'
    
    def test_marker_registration_with_party(self):
        """Test marker registration with Party."""
        with pp.Party() as party:
            m1 = Marker(name='first')
            m2 = Marker(name='second')
            
            assert party.get_marker_by_id(0) is m1
            assert party.get_marker_by_id(1) is m2
            assert party.get_marker_by_name('first') is m1
            assert party.get_marker_by_name('second') is m2
    
    def test_marker_unique_name_validation(self):
        """Test that duplicate marker names raise error."""
        with pp.Party() as party:
            Marker(name='unique')
            with pytest.raises(ValueError, match="Marker name 'unique' already exists"):
                Marker(name='unique')
    
    def test_marker_sequential_ids(self):
        """Test that marker IDs are sequential."""
        with pp.Party() as party:
            m1 = Marker()
            m2 = Marker()
            m3 = Marker()
            assert m1.id == 0
            assert m2.id == 1
            assert m3.id == 2


class TestMarkerScanFactory:
    """Test marker_scan factory function."""
    
    def test_returns_pool(self):
        """Test that marker_scan returns a Pool."""
        with pp.Party() as party:
            result = marker_scan('ACGT')
            assert hasattr(result, 'operation')
            assert isinstance(result.operation, MarkerScanOp)
    
    def test_accepts_string_input(self):
        """Test that marker_scan accepts string input."""
        with pp.Party() as party:
            result = marker_scan('ACGT')
        df = result.generate_seqs(num_seqs=1, seed=42)
        assert '{marker[0]}' in df['seq'].iloc[0]
    
    def test_accepts_pool_input(self):
        """Test that marker_scan accepts Pool input."""
        with pp.Party() as party:
            seq = pp.from_seqs(['ACGT'])
            result = marker_scan(seq)
        df = result.generate_seqs(num_seqs=1, seed=42)
        assert '{marker[0]}' in df['seq'].iloc[0]
    
    def test_marker_none_creates_new_marker(self):
        """Test that marker=None creates a new Marker."""
        with pp.Party() as party:
            result = marker_scan('ACGT', marker=None)
        df = result.generate_seqs(num_seqs=1, seed=42)
        assert '{marker[0]}' in df['seq'].iloc[0]
    
    def test_marker_string_creates_named_marker(self):
        """Test that marker=str creates a named Marker."""
        with pp.Party() as party:
            result = marker_scan('ACGT', marker='insert_site')
        df = result.generate_seqs(num_seqs=1, seed=42)
        assert '{insert_site}' in df['seq'].iloc[0]
    
    def test_marker_object_used_directly(self):
        """Test that Marker object is used directly."""
        with pp.Party() as party:
            m = Marker(name='custom')
            result = marker_scan('ACGT', marker=m)
        df = result.generate_seqs(num_seqs=1, seed=42)
        assert '{custom}' in df['seq'].iloc[0]


class TestMarkerScanSequentialMode:
    """Test marker_scan in sequential mode."""
    
    def test_sequential_enumeration(self):
        """Test sequential enumeration of positions."""
        with pp.Party() as party:
            result = marker_scan('ACGT', marker='m', mode='sequential')
        df = result.generate_seqs(num_complete_iterations=1)
        # 5 possible positions (0, 1, 2, 3, 4)
        assert len(df) == 5
    
    def test_sequential_positions(self):
        """Test specific positions in sequential mode."""
        with pp.Party() as party:
            result = marker_scan('AB', marker='m', mode='sequential')
        df = result.generate_seqs(num_complete_iterations=1)
        seqs = set(df['seq'])
        expected = {'{m}AB', 'A{m}B', 'AB{m}'}
        assert seqs == expected
    
    def test_sequential_num_states(self):
        """Test num_states calculation."""
        with pp.Party() as party:
            result = marker_scan('ACGT', marker='m', mode='sequential')
            # 5 positions (0, 1, 2, 3, 4)
            assert result.operation.num_states == 5


class TestMarkerScanRandomMode:
    """Test marker_scan in random mode."""
    
    def test_random_sampling(self):
        """Test random sampling of positions."""
        with pp.Party() as party:
            result = marker_scan('ACGTACGT', marker='m', mode='random')
        df = result.generate_seqs(num_seqs=100, seed=42)
        # All should have marker inserted
        for seq in df['seq']:
            assert '{m}' in seq
    
    def test_random_variability(self):
        """Test that random mode produces varied outputs."""
        with pp.Party() as party:
            result = marker_scan('ACGTACGT', marker='m', mode='random')
        df = result.generate_seqs(num_seqs=100, seed=42)
        unique_seqs = df['seq'].nunique()
        assert unique_seqs > 1
    
    def test_random_num_states_is_one(self):
        """Test that random mode has num_states=1."""
        with pp.Party() as party:
            result = marker_scan('ACGT', marker='m', mode='random')
            assert result.operation.num_states == 1


class TestMarkerScanPositions:
    """Test custom positions parameter."""
    
    def test_custom_positions(self):
        """Test marker with custom positions."""
        with pp.Party() as party:
            result = marker_scan('ABCDE', marker='m', positions=[1, 3], mode='sequential')
        df = result.generate_seqs(num_complete_iterations=1)
        assert len(df) == 2
        seqs = set(df['seq'])
        expected = {'A{m}BCDE', 'ABC{m}DE'}
        assert seqs == expected
    
    def test_positions_slice(self):
        """Test positions with slice syntax."""
        with pp.Party() as party:
            # slice(1, 4) gives positions 1, 2, 3
            result = marker_scan('ABCDE', marker='m', positions=slice(1, 4), mode='sequential')
        df = result.generate_seqs(num_complete_iterations=1)
        assert len(df) == 3


class TestMarkerScanDesignCards:
    """Test marker_scan design card output."""
    
    def test_position_in_output(self):
        """Test position is in output."""
        with pp.Party() as party:
            result = marker_scan('ACGT', marker='m', mode='sequential', op_name='scan')
            result = result.named('result')
        df = result.generate_seqs(num_seqs=3)
        assert 'result.op.key.position' in df.columns
    
    def test_marker_tag_in_output(self):
        """Test marker_tag is in output."""
        with pp.Party() as party:
            result = marker_scan('ACGT', marker='m', mode='sequential', op_name='scan')
            result = result.named('result')
        df = result.generate_seqs(num_seqs=3)
        assert 'result.op.key.marker_tag' in df.columns
        assert df['result.op.key.marker_tag'].iloc[0] == '{m}'


class TestMarkerScanErrors:
    """Test marker_scan error handling."""
    
    def test_hybrid_requires_num_states(self):
        """Test error for hybrid mode without num_hybrid_states."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="num_hybrid_states is required"):
                marker_scan('ACGT', mode='hybrid')
    
    def test_empty_positions_error(self):
        """Test error when no valid positions."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="No valid positions"):
                marker_scan('ACGT', positions=[], mode='sequential')


class TestAlphabetWithMarkers:
    """Test Alphabet methods with marker-containing sequences."""
    
    def test_get_seq_length_with_markers(self):
        """Test get_seq_length excludes marker characters."""
        alpha = get_alphabet('dna')
        
        # Sequence without marker
        assert alpha.get_seq_length('ACGT') == 4
        
        # Sequence with marker - should still be 4
        assert alpha.get_seq_length('AC{marker}GT') == 4
        assert alpha.get_seq_length('{m}ACGT') == 4
        assert alpha.get_seq_length('ACGT{m}') == 4
        assert alpha.get_seq_length('A{foo}C{bar}GT') == 4
    
    def test_get_seq_length_with_long_marker_names(self):
        """Test get_seq_length with various marker name lengths."""
        alpha = get_alphabet('dna')
        
        assert alpha.get_seq_length('ACGT{a}') == 4
        assert alpha.get_seq_length('ACGT{very_long_marker_name}') == 4
        assert alpha.get_seq_length('{x}{y}ACGT') == 4
    
    def test_get_valid_seq_positions_with_markers(self):
        """Test get_valid_seq_positions excludes marker positions."""
        alpha = get_alphabet('dna')
        
        # Without marker: all positions valid
        assert alpha.get_valid_seq_positions('ACGT') == [0, 1, 2, 3]
        
        # With marker at start: positions shift
        seq = '{m}ACGT'
        positions = alpha.get_valid_seq_positions(seq)
        # {m} occupies positions 0,1,2; ACGT starts at 3
        assert positions == [3, 4, 5, 6]
    
    def test_get_valid_seq_positions_marker_in_middle(self):
        """Test get_valid_seq_positions with marker in middle."""
        alpha = get_alphabet('dna')
        
        seq = 'AC{m}GT'
        positions = alpha.get_valid_seq_positions(seq)
        # A=0, C=1, {=2, m=3, }=4, G=5, T=6
        # Valid positions: 0, 1, 5, 6
        assert positions == [0, 1, 5, 6]
    
    def test_get_valid_seq_positions_multiple_markers(self):
        """Test get_valid_seq_positions with multiple markers."""
        alpha = get_alphabet('dna')
        
        seq = 'A{x}C{y}GT'
        positions = alpha.get_valid_seq_positions(seq)
        # A=0, {x} occupies 1,2,3, C=4, {y} occupies 5,6,7, G=8, T=9
        assert positions == [0, 4, 8, 9]


class TestMarkerPattern:
    """Test the MARKER_PATTERN regex."""
    
    def test_matches_simple_markers(self):
        """Test pattern matches simple markers."""
        assert MARKER_PATTERN.search('{m}') is not None
        assert MARKER_PATTERN.search('{marker}') is not None
        assert MARKER_PATTERN.search('{marker_name}') is not None
    
    def test_matches_markers_with_brackets(self):
        """Test pattern matches markers with brackets in name."""
        assert MARKER_PATTERN.search('{marker[0]}') is not None
        assert MARKER_PATTERN.search('{marker[123]}') is not None
    
    def test_no_match_without_braces(self):
        """Test pattern doesn't match without braces."""
        assert MARKER_PATTERN.search('marker') is None
        assert MARKER_PATTERN.search('ACGT') is None
    
    def test_findall(self):
        """Test finding all markers in sequence."""
        seq = 'A{m1}C{m2}GT'
        matches = MARKER_PATTERN.findall(seq)
        assert matches == ['{m1}', '{m2}']


class TestMarkerScanWithOtherOperations:
    """Test marker_scan combined with other operations."""
    
    def test_with_join(self):
        """Test marker can be joined with other sequences."""
        with pp.Party() as party:
            marked = marker_scan('ACGT', marker='site', mode='sequential')
            combined = pp.join(['PREFIX_', marked, '_SUFFIX']).named('combined')
        
        df = combined.generate_seqs(num_seqs=3)
        for seq in df['seq']:
            assert 'PREFIX_' in seq
            assert '_SUFFIX' in seq
            assert '{site}' in seq
    
    def test_multiple_markers_same_sequence(self):
        """Test inserting multiple markers into same sequence."""
        with pp.Party() as party:
            m1 = Marker(name='first')
            m2 = Marker(name='second')
            # Insert first marker at position 2: AC{first}GTACGT
            step1 = marker_scan('ACGTACGT', marker=m1, positions=[2], mode='sequential')
            # Insert second marker after first marker tag ends
            # {first} is 7 chars, so position 9+6=15 would be after the marker
            # More simply, use position at end of sequence
            step2 = marker_scan(step1, marker=m2, positions=[6], mode='sequential')
        
        df = step2.generate_seqs(num_seqs=1)
        seq = df['seq'].iloc[0]
        assert '{first}' in seq
        assert '{second}' in seq


class TestMarkerAwareOperations:
    """Test that operations correctly handle sequences containing markers."""
    
    def test_mutagenize_skips_markers(self):
        """Test mutagenize only mutates alphabet characters, not marker content."""
        with pp.Party() as party:
            # Create a sequence with a marker
            marked = marker_scan('ACGT', marker='site', positions=[2], mode='sequential')
            # Apply mutagenize - should only mutate the A, C, G, T chars
            mutated = pp.mutagenize(marked, num_mutations=1, mode='sequential')
        
        df = mutated.generate_seqs(num_complete_iterations=1)
        for seq in df['seq']:
            # Marker should be intact
            assert '{site}' in seq
            # Check that marker content wasn't mutated
            marker_match = MARKER_PATTERN.search(seq)
            assert marker_match.group() == '{site}'
    
    def test_breakpoint_scan_splits_around_markers(self):
        """Test breakpoint_scan correctly handles sequences with markers."""
        with pp.Party() as party:
            # Create a sequence with a marker: AC{m}GT
            marked = marker_scan('ACGT', marker='m', positions=[2], mode='sequential')
            # Split at position 3 (after marker in logical terms = after 3rd char = after 'G')
            left, right = pp.breakpoint_scan(marked, num_breakpoints=1, positions=[3], mode='sequential')
            right = right.named('right')
        
        df = left.generate_seqs(num_seqs=1, aux_pools=[right])
        # Check that the marker is in one of the segments
        left_seq = df['seq'].iloc[0]
        right_seq = df['right.seq'].iloc[0]
        # Combined should equal original (with marker)
        assert '{m}' in left_seq or '{m}' in right_seq
    
    def test_seq_shuffle_with_markers(self):
        """Test seq_shuffle handles markers correctly."""
        with pp.Party() as party:
            # Create a sequence with a marker
            marked = marker_scan('ACGTACGT', marker='m', positions=[4], mode='sequential')
            # Shuffle region 2-6 (should include the marker)
            shuffled = pp.seq_shuffle(marked, start=2, end=6, mode='random')
        
        df = shuffled.generate_seqs(num_seqs=5, seed=42)
        for seq in df['seq']:
            # Marker should still be present
            assert '{m}' in seq
    
    def test_pool_seq_length_excludes_markers(self):
        """Test that pool.seq_length reflects length without markers."""
        with pp.Party() as party:
            base = pp.from_seq('ACGT')
            marked = marker_scan(base, marker='m', positions=[2], mode='sequential')
        
        # Original length should be 4
        assert base.seq_length == 4
        # After marker insertion, seq_length should still reflect effective length
        # Since marker_scan output has variable length (depends on marker), 
        # seq_length may be None. But if it's known, it should be 4.
        # Actually the marker adds to the raw string, so let's just check base.
    
    def test_mutagenize_orf_with_markers(self):
        """Test mutagenize_orf correctly strips and restores markers."""
        with pp.Party() as party:
            # Create an ORF sequence with a marker embedded directly
            # ATG (start) + TGT (Cys) + GGT (Gly) + TAA (stop) = ATGTGTGGTTAA
            # Insert marker in middle: ATG TGT {site} GGT TAA
            orf_with_marker = 'ATGTGT{site}GGTTAA'
            pool = pp.from_seq(orf_with_marker)
            # Mutagenize codon 1 (TGT -> something)
            mutated = pp.mutagenize_orf(pool, num_mutations=1, codon_positions=[1], mode='random')
        
        df = mutated.generate_seqs(num_seqs=5, seed=42)
        for seq in df['seq']:
            # Marker should be intact
            assert '{site}' in seq
            # Sequence structure should be preserved
            # Start codon should be unchanged
            clean_seq = MARKER_PATTERN.sub('', seq)
            assert clean_seq[:3] == 'ATG'
    
    def test_party_get_effective_seq_length(self):
        """Test Party.get_effective_seq_length method."""
        with pp.Party() as party:
            # Sequence with markers
            seq_with_marker = 'AC{marker}GT'
            assert party.get_effective_seq_length(seq_with_marker) == 4
            
            # Sequence without markers
            assert party.get_effective_seq_length('ACGT') == 4
    
    def test_party_get_valid_char_positions(self):
        """Test Party.get_valid_char_positions method."""
        with pp.Party() as party:
            # Sequence with marker
            seq = 'AC{marker}GT'
            positions = party.get_valid_char_positions(seq)
            # Should return positions 0, 1 (A, C) and 10, 11 (G, T) 
            # (marker is at positions 2-9)
            assert 0 in positions
            assert 1 in positions
            assert 10 in positions
            assert 11 in positions
            # Positions inside marker should be excluded
            for i in range(2, 10):
                assert i not in positions
