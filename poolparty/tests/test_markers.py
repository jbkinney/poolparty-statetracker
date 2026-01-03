"""Tests for the new XML-style marker system."""

import pytest
import numpy as np
import poolparty as pp
from poolparty.markers import (
    MARKER_PATTERN,
    ParsedMarker,
    parse_marker,
    find_all_markers,
    has_marker,
    validate_single_marker,
    strip_all_markers,
    get_length_without_markers,
    get_nonmarker_positions,
    get_literal_positions,
    nonmarker_pos_to_literal_pos,
    build_marker_tag,
)


class TestMarkerParsing:
    """Test the XML marker parsing utilities."""
    
    def test_build_marker_tag_zero_length(self):
        """Test building zero-length marker tags."""
        assert build_marker_tag('ins', '') == '<ins/>'
        assert build_marker_tag('ins', '', '+') == '<ins/>'
        assert build_marker_tag('ins', '', '-') == "<ins strand='-'/>"
    
    def test_build_marker_tag_with_content(self):
        """Test building marker tags with content."""
        assert build_marker_tag('region', 'ACGT') == '<region>ACGT</region>'
        assert build_marker_tag('region', 'ACGT', '+') == '<region>ACGT</region>'
        assert build_marker_tag('region', 'ACGT', '-') == "<region strand='-'>ACGT</region>"
    
    def test_find_all_markers_zero_length(self):
        """Test finding zero-length markers."""
        seq = 'ACGT<ins/>TT'
        markers = find_all_markers(seq)
        assert len(markers) == 1
        assert markers[0].name == 'ins'
        assert markers[0].content == ''
        assert markers[0].strand == '+'
    
    def test_find_all_markers_with_content(self):
        """Test finding markers with content."""
        seq = 'AC<region>TG</region>AA'
        markers = find_all_markers(seq)
        assert len(markers) == 1
        assert markers[0].name == 'region'
        assert markers[0].content == 'TG'
        assert markers[0].strand == '+'
    
    def test_find_all_markers_with_strand(self):
        """Test finding markers with strand attribute."""
        seq = "AC<region strand='-'>TG</region>AA"
        markers = find_all_markers(seq)
        assert len(markers) == 1
        assert markers[0].strand == '-'
    
    def test_find_all_markers_multiple(self):
        """Test finding multiple markers."""
        seq = 'A<m1>B</m1>C<m2/>D'
        markers = find_all_markers(seq)
        assert len(markers) == 2
        assert markers[0].name == 'm1'
        assert markers[1].name == 'm2'
    
    def test_has_marker(self):
        """Test has_marker function."""
        seq = 'AC<region>TG</region>AA'
        assert has_marker(seq, 'region') is True
        assert has_marker(seq, 'other') is False
    
    def test_validate_single_marker_success(self):
        """Test validate_single_marker with valid input."""
        seq = 'AC<region>TG</region>AA'
        marker = validate_single_marker(seq, 'region')
        assert marker.name == 'region'
        assert marker.content == 'TG'
    
    def test_validate_single_marker_not_found(self):
        """Test validate_single_marker raises error when not found."""
        seq = 'ACGT'
        with pytest.raises(ValueError, match="not found"):
            validate_single_marker(seq, 'region')
    
    def test_validate_single_marker_multiple(self):
        """Test validate_single_marker raises error for multiple occurrences."""
        seq = 'A<m>B</m>C<m>D</m>E'
        with pytest.raises(ValueError, match="appears 2 times"):
            validate_single_marker(seq, 'm')
    
    def test_parse_marker(self):
        """Test parse_marker function."""
        seq = 'AC<region>TG</region>AA'
        prefix, content, suffix, strand = parse_marker(seq, 'region')
        assert prefix == 'AC'
        assert content == 'TG'
        assert suffix == 'AA'
        assert strand == '+'
    
    def test_strip_all_markers(self):
        """Test strip_all_markers function."""
        assert strip_all_markers('AC<region>TG</region>AA') == 'ACTGAA'
        assert strip_all_markers('AC<ins/>TG') == 'ACTG'
        assert strip_all_markers('A<m1>B</m1>C<m2/>D') == 'ABCD'
    
    def test_get_length_without_markers(self):
        """Test get_length_without_markers function."""
        assert get_length_without_markers('ACGT') == 4
        assert get_length_without_markers('AC<region>TG</region>AA') == 6
        assert get_length_without_markers('AC<ins/>GT') == 4
    
    def test_get_nonmarker_positions(self):
        """Test get_nonmarker_positions function."""
        # Without markers
        assert get_nonmarker_positions('ACGT') == [0, 1, 2, 3]
        
        # With marker - should skip tag positions
        seq = 'AC<ins/>GT'
        positions = get_nonmarker_positions(seq)
        # A=0, C=1, <ins/>=2-7, G=8, T=9
        assert positions == [0, 1, 8, 9]
    
    def test_get_literal_positions(self):
        """Test get_literal_positions function."""
        assert get_literal_positions('ACGT') == [0, 1, 2, 3]
        assert get_literal_positions('AC<ins/>GT') == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert get_literal_positions('') == []
    
    def test_nonmarker_pos_to_literal_pos(self):
        """Test nonmarker_pos_to_literal_pos function."""
        # Without markers - should be identity
        assert nonmarker_pos_to_literal_pos('ACGT', 0) == 0
        assert nonmarker_pos_to_literal_pos('ACGT', 2) == 2
        assert nonmarker_pos_to_literal_pos('ACGT', 4) == 4  # One past end
        
        # With marker - should skip marker tag positions
        seq = 'AC<ins/>GT'
        # nonmarker positions: 0->0, 1->1, 2->8, 3->9, 4->10 (one past end)
        assert nonmarker_pos_to_literal_pos(seq, 0) == 0
        assert nonmarker_pos_to_literal_pos(seq, 1) == 1
        assert nonmarker_pos_to_literal_pos(seq, 2) == 8  # G is at literal position 8
        assert nonmarker_pos_to_literal_pos(seq, 3) == 9  # T is at literal position 9
        assert nonmarker_pos_to_literal_pos(seq, 4) == 10  # One past end
    
    def test_nonmarker_pos_to_literal_pos_out_of_range(self):
        """Test nonmarker_pos_to_literal_pos raises on invalid positions."""
        with pytest.raises(ValueError):
            nonmarker_pos_to_literal_pos('ACGT', -1)
        with pytest.raises(ValueError):
            nonmarker_pos_to_literal_pos('ACGT', 5)


class TestMarkerPattern:
    """Test the MARKER_PATTERN regex."""
    
    def test_matches_self_closing(self):
        """Test pattern matches self-closing markers."""
        assert MARKER_PATTERN.search('<m/>') is not None
        assert MARKER_PATTERN.search('<marker_name/>') is not None
        assert MARKER_PATTERN.search("<m strand='-'/>") is not None
    
    def test_matches_opening_tag(self):
        """Test pattern matches opening tags."""
        assert MARKER_PATTERN.search('<region>') is not None
        assert MARKER_PATTERN.search("<region strand='-'>") is not None
    
    def test_matches_closing_tag(self):
        """Test pattern matches closing tags."""
        assert MARKER_PATTERN.search('</region>') is not None


class TestInsertMarker:
    """Test insert_marker operation."""
    
    def test_insert_zero_length_marker(self):
        """Test inserting zero-length marker."""
        with pp.Party():
            result = pp.insert_marker('ACGTACGT', 'ins', start=4)
        df = result.generate_seqs(num_seqs=1)
        assert df['seq'].iloc[0] == 'ACGT<ins/>ACGT'
    
    def test_insert_region_marker(self):
        """Test inserting region marker."""
        with pp.Party():
            result = pp.insert_marker('ACGTACGT', 'region', start=2, stop=5)
        df = result.generate_seqs(num_seqs=1)
        assert df['seq'].iloc[0] == 'AC<region>GTA</region>CGT'
    
    def test_insert_marker_with_strand(self):
        """Test inserting marker with strand attribute."""
        with pp.Party():
            result = pp.insert_marker('ACGT', 'region', start=1, stop=3, strand='-')
        df = result.generate_seqs(num_seqs=1)
        assert df['seq'].iloc[0] == "A<region strand='-'>CG</region>T"
    
    def test_insert_marker_into_sequence_with_existing_marker(self):
        """Test inserting marker into sequence that already has markers.
        
        This tests the fix for the bug where positions were interpreted
        as literal string positions instead of non-marker positions.
        """
        with pp.Party():
            # Start with a sequence that already has a marker
            bg = pp.from_seq('AC<a/>GT')
            # Insert marker 'b' at positions 2-4 (should be 'GT', not inside <a/>)
            marked = pp.insert_marker(bg, 'b', start=2, stop=4)
        df = marked.generate_seqs(num_complete_iterations=1)
        # Should get 'AC<a/><b>GT</b>', not 'AC<b><a</b>/>GT'
        assert df['seq'].iloc[0] == 'AC<a/><b>GT</b>'
    
    def test_insert_zero_length_marker_into_sequence_with_existing_marker(self):
        """Test inserting zero-length marker into sequence with existing markers."""
        with pp.Party():
            bg = pp.from_seq('AC<a/>GT')
            # Insert zero-length marker at position 2 (after <a/>)
            marked = pp.insert_marker(bg, 'b', start=2)
        df = marked.generate_seqs(num_complete_iterations=1)
        # Should get 'AC<a/><b/>GT'
        assert df['seq'].iloc[0] == 'AC<a/><b/>GT'


class TestMarkerScan:
    """Test marker_scan operation."""
    
    def test_sequential_zero_length_markers(self):
        """Test sequential enumeration of zero-length markers."""
        with pp.Party():
            result = pp.marker_scan('ACGT', marker='m', mode='sequential')
        df = result.generate_seqs(num_complete_iterations=1)
        # 5 positions: before A, after A, after C, after G, after T
        assert len(df) == 5
        seqs = set(df['seq'])
        assert '<m/>ACGT' in seqs
        assert 'A<m/>CGT' in seqs
        assert 'AC<m/>GT' in seqs
        assert 'ACG<m/>T' in seqs
        assert 'ACGT<m/>' in seqs
    
    def test_sequential_region_markers(self):
        """Test sequential enumeration of region markers."""
        with pp.Party():
            result = pp.marker_scan('ACGT', marker='m', mode='sequential', marker_length=2)
        df = result.generate_seqs(num_complete_iterations=1)
        # 3 positions for length-2 regions: positions 0, 1, 2
        assert len(df) == 3
        seqs = set(df['seq'])
        assert '<m>AC</m>GT' in seqs
        assert 'A<m>CG</m>T' in seqs
        assert 'AC<m>GT</m>' in seqs
    
    def test_random_mode(self):
        """Test random mode."""
        with pp.Party():
            result = pp.marker_scan('ACGTACGT', marker='m', mode='random')
        df = result.generate_seqs(num_seqs=10, seed=42)
        for seq in df['seq']:
            assert '<m/>' in seq
    
    def test_strand_both(self):
        """Test strand='both' doubles the states."""
        with pp.Party():
            result = pp.marker_scan('ACGT', marker='m', mode='sequential', strand='both')
        df = result.generate_seqs(num_complete_iterations=1)
        # 5 positions x 2 strands = 10 states
        assert len(df) == 10


class TestExtractMarkerContent:
    """Test extract_marker_content operation."""
    
    def test_extract_basic(self):
        """Test extracting content from marker."""
        with pp.Party():
            bg = pp.from_seq('ACGT<region>TTAA</region>GCGC')
            content = pp.extract_marker_content(bg, 'region')
        df = content.generate_seqs(num_seqs=1)
        assert df['seq'].iloc[0] == 'TTAA'
    
    def test_extract_with_minus_strand(self):
        """Test extracting content with strand='-' reverse complements."""
        with pp.Party():
            bg = pp.from_seq("ACGT<region strand='-'>TTAA</region>GCGC")
            content = pp.extract_marker_content(bg, 'region')
        df = content.generate_seqs(num_seqs=1)
        # TTAA reverse complement = TTAA (palindrome)
        # Let's use a non-palindrome
        
        with pp.Party():
            bg = pp.from_seq("ACGT<region strand='-'>AACG</region>GCGC")
            content = pp.extract_marker_content(bg, 'region')
        df = content.generate_seqs(num_seqs=1)
        # AACG reverse complement = CGTT
        assert df['seq'].iloc[0] == 'CGTT'


class TestReplaceMarkerContent:
    """Test replace_marker_content operation."""
    
    def test_replace_basic(self):
        """Test replacing marker with content from another pool."""
        with pp.Party():
            bg = pp.from_seq('ACGT<insert/>TTTT')
            inserts = pp.from_seqs(['AAA', 'GGG'], mode='sequential')
            result = pp.replace_marker_content(bg, inserts, 'insert')
        df = result.generate_seqs(num_complete_iterations=1)
        seqs = set(df['seq'])
        assert 'ACGTAAATTTT' in seqs
        assert 'ACGTGGGTTTT' in seqs
    
    def test_replace_region_marker(self):
        """Test replacing region marker (with existing content)."""
        with pp.Party():
            bg = pp.from_seq('PREFIX<var>OLDCONTENT</var>SUFFIX')
            variants = pp.from_seqs(['NEW1', 'NEW2'], mode='sequential')
            result = pp.replace_marker_content(bg, variants, 'var')
        df = result.generate_seqs(num_complete_iterations=1)
        seqs = set(df['seq'])
        assert 'PREFIXNEW1SUFFIX' in seqs
        assert 'PREFIXNEW2SUFFIX' in seqs
    
    def test_replace_with_minus_strand(self):
        """Test replacing with minus strand reverse complements content."""
        with pp.Party():
            bg = pp.from_seq("ACGT<region strand='-'>XX</region>TTTT")
            content = pp.from_seq('AAA')
            result = pp.replace_marker_content(bg, content, 'region')
        df = result.generate_seqs(num_seqs=1)
        # AAA reverse complement = TTT
        assert df['seq'].iloc[0] == 'ACGTTTTTTTT'


class TestApplyAtMarker:
    """Test apply_at_marker operation."""
    
    def test_apply_reverse_complement(self):
        """Test applying reverse_complement at marker."""
        with pp.Party():
            bg = pp.from_seq('ACGT<orf>ATGCCC</orf>TTTT')
            result = pp.apply_at_marker(bg, 'orf', pp.reverse_complement)
        df = result.generate_seqs(num_seqs=1)
        # ATGCCC reverse complement = GGGCAT
        assert df['seq'].iloc[0] == 'ACGTGGGCATTTTT'
    
    def test_apply_seq_shuffle(self):
        """Test applying seq_shuffle at marker."""
        with pp.Party():
            bg = pp.from_seq('AAA<region>ACGTACGT</region>TTT')
            result = pp.apply_at_marker(
                bg, 'region',
                lambda p: pp.seq_shuffle(p, mode='random')
            )
        df = result.generate_seqs(num_seqs=5, seed=42)
        # All should have 8 characters between AAA and TTT
        for seq in df['seq']:
            assert seq.startswith('AAA')
            assert seq.endswith('TTT')
            middle = seq[3:-3]
            assert len(middle) == 8


class TestRemoveMarker:
    """Test remove_marker operation."""
    
    def test_remove_keep_content(self):
        """Test removing marker but keeping content."""
        with pp.Party():
            bg = pp.from_seq('ACGT<region>TTAA</region>GCGC')
            result = pp.remove_marker(bg, 'region', keep_content=True)
        df = result.generate_seqs(num_seqs=1)
        assert df['seq'].iloc[0] == 'ACGTTTAAGCGC'
    
    def test_remove_discard_content(self):
        """Test removing marker and its content."""
        with pp.Party():
            bg = pp.from_seq('ACGT<region>TTAA</region>GCGC')
            result = pp.remove_marker(bg, 'region', keep_content=False)
        df = result.generate_seqs(num_seqs=1)
        assert df['seq'].iloc[0] == 'ACGTGCGC'


class TestAlphabetWithXMLMarkers:
    """Test Alphabet methods work with XML markers."""
    
    def test_get_seq_length_with_markers(self):
        """Test get_seq_length excludes marker tags but includes content."""
        from poolparty.alphabet import get_alphabet
        alpha = get_alphabet('dna')
        
        # Without marker
        assert alpha.get_seq_length('ACGT') == 4
        
        # With marker - counts content but not tags
        assert alpha.get_seq_length('AC<region>TG</region>GT') == 6
        assert alpha.get_seq_length('<m/>ACGT') == 4
        assert alpha.get_seq_length('ACGT<m/>') == 4
    
    def test_get_length_without_markers(self):
        """Test get_length_without_markers on Alphabet."""
        from poolparty.alphabet import get_alphabet
        alpha = get_alphabet('dna')
        
        assert alpha.get_length_without_markers('ACGT') == 4
        assert alpha.get_length_without_markers('AC<region>TG</region>GT') == 6
    
    def test_get_nonmarker_positions(self):
        """Test get_nonmarker_positions on Alphabet."""
        from poolparty.alphabet import get_alphabet
        alpha = get_alphabet('dna')
        
        # Without marker
        assert alpha.get_nonmarker_positions('ACGT') == [0, 1, 2, 3]
        
        # With marker - should skip tag positions
        positions = alpha.get_nonmarker_positions('A<m/>CG')
        # A=0, <m/>=1-4, C=5, G=6
        assert positions == [0, 5, 6]


class TestMutagenizeOrfWithXMLMarkers:
    """Test mutagenize_orf preserves XML markers."""
    
    def test_marker_preserved(self):
        """Test that markers are preserved through mutation."""
        with pp.Party():
            # ATG TGT GGT TAA with marker between codons
            orf_with_marker = 'ATGTGT<site/>GGTTAA'
            pool = pp.from_seq(orf_with_marker)
            mutated = pp.mutagenize_orf(pool, num_mutations=1, codon_positions=[1], mode='random')
        
        df = mutated.generate_seqs(num_seqs=5, seed=42)
        for seq in df['seq']:
            # Marker should be intact
            assert '<site/>' in seq
            # Start and stop codons should be intact
            clean_seq = strip_all_markers(seq)
            assert clean_seq[:3] == 'ATG'
            assert clean_seq[-3:] == 'TAA'


class TestPartyMethodsWithXMLMarkers:
    """Test Party helper methods with XML markers."""
    
    def test_get_effective_seq_length(self):
        """Test Party.get_effective_seq_length method."""
        with pp.Party() as party:
            seq_with_marker = 'AC<marker>TG</marker>GT'
            assert party.get_effective_seq_length(seq_with_marker) == 6
            assert party.get_effective_seq_length('ACGT') == 4
    
    def test_get_biological_positions(self):
        """Test Party.get_biological_positions method."""
        with pp.Party() as party:
            # With marker
            seq = 'AC<marker/>GT'
            positions = party.get_biological_positions(seq)
            # A=0, C=1, <marker/>=2-10, G=11, T=12
            assert 0 in positions
            assert 1 in positions
            assert 11 in positions
            assert 12 in positions
            # Tag positions should be excluded
            for i in range(2, 11):
                assert i not in positions


class TestMarkerClass:
    """Test the Marker class and Party registration."""
    
    def test_marker_creation(self):
        """Test creating Marker objects."""
        from poolparty.marker import Marker
        
        m1 = Marker(name='test', seq_length=10)
        assert m1.name == 'test'
        assert m1.seq_length == 10
        assert m1._id == -1  # Not registered yet
        
        m2 = Marker(name='zero', seq_length=0)
        assert m2.is_zero_length
        
        m3 = Marker(name='var', seq_length=None)
        assert m3.is_variable_length
    
    def test_marker_validation(self):
        """Test Marker validation."""
        from poolparty.marker import Marker
        
        # Empty name
        with pytest.raises(ValueError, match="cannot be empty"):
            Marker(name='', seq_length=5)
        
        # Invalid name (not identifier)
        with pytest.raises(ValueError, match="not a valid identifier"):
            Marker(name='123abc', seq_length=5)
        
        # Negative seq_length
        with pytest.raises(ValueError, match="must be None or >= 0"):
            Marker(name='test', seq_length=-1)
    
    def test_party_register_marker(self):
        """Test registering markers with Party."""
        with pp.Party() as party:
            # Register a marker
            m1 = party.register_marker('orf', 100)
            assert m1.name == 'orf'
            assert m1.seq_length == 100
            assert m1._id == 0
            
            # Registering same marker again returns existing
            m2 = party.register_marker('orf', 100)
            assert m2 is m1
            
            # Registering with different length raises error
            with pytest.raises(ValueError, match="already registered"):
                party.register_marker('orf', 50)
    
    def test_party_get_marker(self):
        """Test retrieving markers from Party."""
        with pp.Party() as party:
            party.register_marker('test', 10)
            
            # Get by name
            m = party.get_marker_by_name('test')
            assert m.name == 'test'
            
            # Get by id
            m2 = party.get_marker_by_id(0)
            assert m2.name == 'test'
            
            # Not found
            with pytest.raises(ValueError, match="not found"):
                party.get_marker_by_name('nonexistent')
    
    def test_party_has_marker(self):
        """Test checking if marker exists in Party."""
        with pp.Party() as party:
            assert not party.has_marker('test')
            party.register_marker('test', 10)
            assert party.has_marker('test')
    
    def test_from_seq_auto_registers_markers(self):
        """Test that from_seq automatically registers markers."""
        with pp.Party() as party:
            # Create pool with marked sequence
            pool = pp.from_seq('AAA<orf>ATGCCC</orf>TTT')
            
            # Marker should be registered
            assert party.has_marker('orf')
            m = party.get_marker_by_name('orf')
            assert m.seq_length == 6  # 'ATGCCC'
    
    def test_pool_markers_property(self):
        """Test pool.markers property."""
        with pp.Party():
            # Create pool with marker
            pool = pp.from_seq('AAA<orf>ATGCCC</orf>TTT')
            
            # Check markers set
            markers = pool.markers
            assert len(markers) == 1
            assert any(m.name == 'orf' for m in markers)
            
            # Check has_marker method
            assert pool.has_marker('orf')
            assert not pool.has_marker('nonexistent')
    
    def test_marker_inheritance(self):
        """Test that markers are inherited from parent pools."""
        with pp.Party():
            # Create pool with marker
            parent = pp.from_seq('AAA<orf>ATGCCC</orf>TTT')
            
            # Create child pool (e.g., through reverse_complement)
            child = pp.reverse_complement(parent)
            
            # Child should inherit markers
            assert child.has_marker('orf')
    
    def test_marker_removed_after_replace(self):
        """Test that marker is removed from pool's set after replacement."""
        with pp.Party():
            bg = pp.from_seq('AAA<target>XXXX</target>TTT')
            assert bg.has_marker('target')
            
            content = pp.from_seq('GGGG')
            result = pp.replace_marker_content(bg, content, 'target')
            
            # Marker should be removed from result
            assert not result.has_marker('target')
    
    def test_extract_marker_uses_registered_seq_length(self):
        """Test that extract_marker_content uses registered seq_length."""
        with pp.Party():
            # Create pool with 6-char marker content
            bg = pp.from_seq('AAA<orf>ATGCCC</orf>TTT')
            
            # Extract should have seq_length=6
            content = pp.extract_marker_content(bg, 'orf')
            assert content.seq_length == 6
    
    def test_mutagenize_at_marker_sequential(self):
        """Test that mutagenize works correctly via apply_at_marker in sequential mode."""
        with pp.Party():
            # Create template with marked region
            template = pp.from_seq('AAAA<target>ACGT</target>TTTT')
            
            # Apply mutagenize with 1 mutation per position (sequential)
            result = pp.apply_at_marker(
                template, 'target',
                lambda p: pp.mutagenize(p, num_mutations=1, mode='sequential')
            )
            
            # Should generate all single-mutation variants
            # 4 positions * 3 alternatives = 12 variants
            df = result.generate_seqs(num_complete_iterations=1)
            assert len(df) == 12
            
            # All variants should be 12 chars (4+4+4)
            for seq in df['seq']:
                assert len(seq) == 12


class TestSeqLengthAttribute:
    """Test the seq_length attribute in XML markers."""
    
    def test_parse_seq_length_integer(self):
        """Test parsing seq_length as integer."""
        markers = find_all_markers("<orf seq_length='6'>ATGCCC</orf>")
        assert len(markers) == 1
        assert markers[0].declared_seq_length_str == '6'
    
    def test_parse_seq_length_none(self):
        """Test parsing seq_length='None' for variable length."""
        markers = find_all_markers("<var seq_length='None'>ACGT</var>")
        assert len(markers) == 1
        assert markers[0].declared_seq_length_str == 'None'
        assert markers[0].is_variable_length
    
    def test_seq_length_validation(self):
        """Test that seq_length validates against content length."""
        # Correct length
        markers = find_all_markers("<orf seq_length='4'>ACGT</orf>")
        assert len(markers) == 1
        
        # Wrong length should raise error
        with pytest.raises(ValueError, match="seq_length='6'.*content has length 4"):
            find_all_markers("<orf seq_length='6'>ACGT</orf>")
    
    def test_self_closing_seq_length_zero(self):
        """Test self-closing markers with seq_length."""
        markers = find_all_markers("<ins seq_length='0'/>")
        assert len(markers) == 1
        assert markers[0].declared_seq_length_str == '0'
    
    def test_build_marker_with_seq_length(self):
        """Test building marker tags with seq_length."""
        tag = build_marker_tag('orf', 'ACGT', '+', seq_length=4)
        assert tag == "<orf seq_length='4'>ACGT</orf>"
        
        # Variable length
        tag = build_marker_tag('var', 'ACGT', '+', seq_length=-1)
        assert tag == "<var seq_length='None'>ACGT</var>"


class TestIntegration:
    """Integration tests combining multiple marker operations."""
    
    def test_scan_and_replace(self):
        """Test marker_scan followed by replace_marker_content."""
        with pp.Party():
            # Scan positions
            marked = pp.marker_scan('ACGTACGT', marker='ins', positions=[2, 6], mode='sequential')
            # Replace with insertions
            inserts = pp.from_seq('NNN')
            result = pp.replace_marker_content(marked, inserts, 'ins')
        
        df = result.generate_seqs(num_complete_iterations=1)
        for seq in df['seq']:
            assert 'NNN' in seq
            assert '<' not in seq  # No markers left
    
    def test_insert_transform_workflow(self):
        """Test typical workflow: insert marker -> transform at marker."""
        with pp.Party():
            # Mark a region for modification
            bg = pp.from_seq('AAAACGTACGTTTTT')  # 15 chars
            marked = pp.insert_marker(bg, 'target', start=4, stop=12)  # marks CGTACGTT
            # Apply transformation at marker
            result = pp.apply_at_marker(marked, 'target', pp.reverse_complement)
        
        df = result.generate_seqs(num_seqs=1)
        seq = df['seq'].iloc[0]
        # Check basic properties: no marker tags, correct length
        assert '<' not in seq
        assert len(seq) == 15
