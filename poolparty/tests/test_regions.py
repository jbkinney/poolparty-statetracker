"""Tests for the new XML-style region system."""

import pytest

import poolparty as pp
from poolparty.region_ops import (
    TAG_PATTERN,
    build_region_tags,
    find_all_regions,
    get_length_without_tags,
    get_literal_positions,
    get_nontag_positions,
    has_region,
    nontag_pos_to_literal_pos,
    parse_region,
    strip_all_tags,
    validate_single_region,
)


class TestRegionParsing:
    """Test the XML region parsing utilities."""

    def test_build_region_tags_zero_length(self):
        """Test building zero-length region tags."""
        assert build_region_tags("ins", "") == "<ins/>"

    def test_build_region_tags_with_content(self):
        """Test building region tags with content."""
        assert build_region_tags("region", "ACGT") == "<region>ACGT</region>"

    def test_find_all_regions_zero_length(self):
        """Test finding zero-length regions."""
        seq = "ACGT<ins/>TT"
        regions = find_all_regions(seq)
        assert len(regions) == 1
        assert regions[0].name == "ins"
        assert regions[0].content == ""

    def test_find_all_regions_with_content(self):
        """Test finding regions with content."""
        seq = "AC<region>TG</region>AA"
        regions = find_all_regions(seq)
        assert len(regions) == 1
        assert regions[0].name == "region"
        assert regions[0].content == "TG"

    def test_find_all_regions_multiple(self):
        """Test finding multiple regions."""
        seq = "A<m1>B</m1>C<m2/>D"
        regions = find_all_regions(seq)
        assert len(regions) == 2
        assert regions[0].name == "m1"
        assert regions[1].name == "m2"

    def test_has_region(self):
        """Test has_region function."""
        seq = "AC<region>TG</region>AA"
        assert has_region(seq, "region") is True
        assert has_region(seq, "other") is False

    def test_validate_single_region_success(self):
        """Test validate_single_region with valid input."""
        seq = "AC<region>TG</region>AA"
        region = validate_single_region(seq, "region")
        assert region.name == "region"
        assert region.content == "TG"

    def test_validate_single_region_not_found(self):
        """Test validate_single_region raises error when not found."""
        seq = "ACGT"
        with pytest.raises(ValueError, match="not found"):
            validate_single_region(seq, "region")

    def test_validate_single_region_multiple(self):
        """Test validate_single_region raises error for multiple occurrences."""
        seq = "A<m>B</m>C<m>D</m>E"
        with pytest.raises(ValueError, match="appears 2 times"):
            validate_single_region(seq, "m")

    def test_parse_region(self):
        """Test parse_region function."""
        seq = "AC<region>TG</region>AA"
        prefix, content, suffix = parse_region(seq, "region")
        assert prefix == "AC"
        assert content == "TG"
        assert suffix == "AA"

    def test_strip_all_tags(self):
        """Test strip_all_tags function."""
        assert strip_all_tags("AC<region>TG</region>AA") == "ACTGAA"
        assert strip_all_tags("AC<ins/>TG") == "ACTG"
        assert strip_all_tags("A<m1>B</m1>C<m2/>D") == "ABCD"

    def test_get_length_without_tags(self):
        """Test get_length_without_tags function."""
        assert get_length_without_tags("ACGT") == 4
        assert get_length_without_tags("AC<region>TG</region>AA") == 6
        assert get_length_without_tags("AC<ins/>GT") == 4

    def test_get_nontag_positions(self):
        """Test get_nontag_positions function."""
        # Without regions
        assert get_nontag_positions("ACGT") == [0, 1, 2, 3]

        # With region - should skip tag positions
        seq = "AC<ins/>GT"
        positions = get_nontag_positions(seq)
        # A=0, C=1, <ins/>=2-7, G=8, T=9
        assert positions == [0, 1, 8, 9]

    def test_get_literal_positions(self):
        """Test get_literal_positions function."""
        assert get_literal_positions("ACGT") == [0, 1, 2, 3]
        assert get_literal_positions("AC<ins/>GT") == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert get_literal_positions("") == []

    def test_nontag_pos_to_literal_pos(self):
        """Test nontag_pos_to_literal_pos function."""
        # Without regions - should be identity
        assert nontag_pos_to_literal_pos("ACGT", 0) == 0
        assert nontag_pos_to_literal_pos("ACGT", 2) == 2
        assert nontag_pos_to_literal_pos("ACGT", 4) == 4  # One past end

        # With region - should skip region tag positions
        seq = "AC<ins/>GT"
        # nonregion positions: 0->0, 1->1, 2->8, 3->9, 4->10 (one past end)
        assert nontag_pos_to_literal_pos(seq, 0) == 0
        assert nontag_pos_to_literal_pos(seq, 1) == 1
        assert nontag_pos_to_literal_pos(seq, 2) == 8  # G is at literal position 8
        assert nontag_pos_to_literal_pos(seq, 3) == 9  # T is at literal position 9
        assert nontag_pos_to_literal_pos(seq, 4) == 10  # One past end

    def test_nontag_pos_to_literal_pos_out_of_range(self):
        """Test nontag_pos_to_literal_pos raises on invalid positions."""
        with pytest.raises(ValueError):
            nontag_pos_to_literal_pos("ACGT", -1)
        with pytest.raises(ValueError):
            nontag_pos_to_literal_pos("ACGT", 5)


class TestMarkerPattern:
    """Test the TAG_PATTERN regex."""

    def test_matches_self_closing(self):
        """Test pattern matches self-closing regions."""
        assert TAG_PATTERN.search("<m/>") is not None
        assert TAG_PATTERN.search("<region_name/>") is not None

    def test_matches_opening_tag(self):
        """Test pattern matches opening tags."""
        assert TAG_PATTERN.search("<region>") is not None

    def test_matches_closing_tag(self):
        """Test pattern matches closing tags."""
        assert TAG_PATTERN.search("</region>") is not None


class TestInsertMarker:
    """Test insert_tags operation."""

    def test_insert_zero_length_region(self):
        """Test inserting zero-length region."""
        with pp.Party():
            result = pp.insert_tags("ACGTACGT", "ins", start=4)
        df = result.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "ACGT<ins/>ACGT"

    def test_insert_region_region(self):
        """Test inserting region region."""
        with pp.Party():
            result = pp.insert_tags("ACGTACGT", "region", start=2, stop=5)
        df = result.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "AC<region>GTA</region>CGT"

    def test_insert_tags_into_sequence_with_existing_region(self):
        """Test inserting region into sequence that already has regions.

        This tests the fix for the bug where positions were interpreted
        as literal string positions instead of non-region positions.
        """
        with pp.Party():
            # Start with a sequence that already has a region
            bg = pp.from_seq("AC<a/>GT")
            # Insert region 'b' at positions 2-4 (should be 'GT', not inside <a/>)
            marked = pp.insert_tags(bg, "b", start=2, stop=4)
        df = marked.generate_library(num_cycles=1)
        # Should get 'AC<a/><b>GT</b>', not 'AC<b><a</b>/>GT'
        assert df["seq"].iloc[0] == "AC<a/><b>GT</b>"

    def test_insert_zero_length_region_into_sequence_with_existing_region(self):
        """Test inserting zero-length region into sequence with existing regions."""
        with pp.Party():
            bg = pp.from_seq("AC<a/>GT")
            # Insert zero-length region at position 2 (after <a/>)
            marked = pp.insert_tags(bg, "b", start=2)
        df = marked.generate_library(num_cycles=1)
        # Should get 'AC<a/><b/>GT'
        assert df["seq"].iloc[0] == "AC<a/><b/>GT"


class TestMarkerScan:
    """Test region_scan operation."""

    def test_sequential_zero_length_regions(self):
        """Test sequential enumeration of zero-length regions."""
        with pp.Party():
            result = pp.region_scan("ACGT", tag_name="m", mode="sequential")
        df = result.generate_library(num_cycles=1)
        # 5 positions: before A, after A, after C, after G, after T
        assert len(df) == 5
        seqs = set(df["seq"])
        assert "<m/>ACGT" in seqs
        assert "A<m/>CGT" in seqs
        assert "AC<m/>GT" in seqs
        assert "ACG<m/>T" in seqs
        assert "ACGT<m/>" in seqs

    def test_sequential_region_regions(self):
        """Test sequential enumeration of region regions."""
        with pp.Party():
            result = pp.region_scan("ACGT", tag_name="m", mode="sequential", region_length=2)
        df = result.generate_library(num_cycles=1)
        # 3 positions for length-2 regions: positions 0, 1, 2
        assert len(df) == 3
        seqs = set(df["seq"])
        assert "<m>AC</m>GT" in seqs
        assert "A<m>CG</m>T" in seqs
        assert "AC<m>GT</m>" in seqs

    def test_random_mode(self):
        """Test random mode."""
        with pp.Party():
            result = pp.region_scan("ACGTACGT", tag_name="m", mode="random")
        df = result.generate_library(num_seqs=10, seed=42)
        for seq in df["seq"]:
            assert "<m/>" in seq


class TestExtractMarkerContent:
    """Test extract_region operation."""

    def test_extract_basic(self):
        """Test extracting content from region."""
        with pp.Party():
            bg = pp.from_seq("ACGT<region>TTAA</region>GCGC")
            content = pp.extract_region(bg, "region")
        df = content.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "TTAA"

    def test_extract_with_rc(self):
        """Test extracting content with rc=True reverse complements."""
        with pp.Party():
            bg = pp.from_seq("ACGT<region>AACG</region>GCGC")
            content = pp.extract_region(bg, "region", rc=True)
        df = content.generate_library(num_seqs=1)
        # AACG reverse complement = CGTT
        assert df["seq"].iloc[0] == "CGTT"


class TestReplaceMarkerContent:
    """Test replace_region operation."""

    def test_replace_basic(self):
        """Test replacing region with content from another pool."""
        with pp.Party():
            bg = pp.from_seq("ACGT<insert/>TTTT")
            inserts = pp.from_seqs(["AAA", "GGG"], mode="sequential")
            result = pp.replace_region(bg, inserts, "insert")
        df = result.generate_library(num_cycles=1)
        seqs = set(df["seq"])
        assert "ACGTAAATTTT" in seqs
        assert "ACGTGGGTTTT" in seqs

    def test_replace_region_region(self):
        """Test replacing region region (with existing content)."""
        with pp.Party():
            bg = pp.from_seq("PREFIX<var>OLDCONTENT</var>SUFFIX")
            variants = pp.from_seqs(["NEW1", "NEW2"], mode="sequential")
            result = pp.replace_region(bg, variants, "var")
        df = result.generate_library(num_cycles=1)
        seqs = set(df["seq"])
        assert "PREFIXNEW1SUFFIX" in seqs
        assert "PREFIXNEW2SUFFIX" in seqs

    def test_replace_with_rc(self):
        """Test replacing with rc=True reverse complements content."""
        with pp.Party():
            bg = pp.from_seq("ACGT<region>XX</region>TTTT")
            content = pp.from_seq("AAA")
            result = pp.replace_region(bg, content, "region", rc=True)
        df = result.generate_library(num_seqs=1)
        # AAA reverse complement = TTT
        assert df["seq"].iloc[0] == "ACGTTTTTTTT"


class TestApplyAtMarker:
    """Test apply_at_region operation."""

    def test_apply_rc(self):
        """Test applying rc at region."""
        with pp.Party():
            bg = pp.from_seq("ACGT<orf>ATGCCC</orf>TTTT")
            result = pp.apply_at_region(bg, "orf", pp.rc)
        df = result.generate_library(num_seqs=1)
        # ATGCCC reverse complement = GGGCAT
        assert df["seq"].iloc[0] == "ACGTGGGCATTTTT"

    def test_apply_shuffle_seq(self):
        """Test applying shuffle_seq at region."""
        with pp.Party():
            bg = pp.from_seq("AAA<region>ACGTACGT</region>TTT")
            result = pp.apply_at_region(bg, "region", lambda p: pp.shuffle_seq(p, mode="random"))
        df = result.generate_library(num_seqs=5, seed=42)
        # All should have 8 characters between AAA and TTT
        for seq in df["seq"]:
            assert seq.startswith("AAA")
            assert seq.endswith("TTT")
            middle = seq[3:-3]
            assert len(middle) == 8


class TestRemoveMarker:
    """Test remove_tags operation."""

    def test_remove_keep_content(self):
        """Test removing region but keeping content."""
        with pp.Party():
            bg = pp.from_seq("ACGT<region>TTAA</region>GCGC")
            result = pp.remove_tags(bg, "region", keep_content=True)
        df = result.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "ACGTTTAAGCGC"

    def test_remove_discard_content(self):
        """Test removing region and its content."""
        with pp.Party():
            bg = pp.from_seq("ACGT<region>TTAA</region>GCGC")
            result = pp.remove_tags(bg, "region", keep_content=False)
        df = result.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "ACGTGCGC"


class TestDNAWithXMLMarkers:
    """Test dna module functions work with XML regions."""

    def test_get_seq_length_with_regions(self):
        """Test get_seq_length excludes region tags but includes content."""
        from poolparty.utils import dna_utils

        # Without region
        assert dna_utils.get_seq_length("ACGT") == 4

        # With region - counts content but not tags
        assert dna_utils.get_seq_length("AC<region>TG</region>GT") == 6
        assert dna_utils.get_seq_length("<m/>ACGT") == 4
        assert dna_utils.get_seq_length("ACGT<m/>") == 4

    def test_get_length_without_tags(self):
        """Test get_length_without_tags from dna module."""
        from poolparty.utils import dna_utils

        assert dna_utils.get_length_without_tags("ACGT") == 4
        assert dna_utils.get_length_without_tags("AC<region>TG</region>GT") == 6

    def test_get_nontag_positions(self):
        """Test get_nontag_positions from dna module."""
        from poolparty.utils import dna_utils

        # Without region
        assert dna_utils.get_nontag_positions("ACGT") == [0, 1, 2, 3]

        # With region - should skip tag positions
        positions = dna_utils.get_nontag_positions("A<m/>CG")
        # A=0, <m/>=1-4, C=5, G=6
        assert positions == [0, 5, 6]


class TestMutagenizeOrfWithXMLMarkers:
    """Test mutagenize_orf preserves XML regions."""

    def test_region_preserved(self):
        """Test that regions are preserved through mutation."""
        with pp.Party():
            # ATG TGT GGT TAA with region between codons
            orf_with_region = "ATGTGT<site/>GGTTAA"
            pool = pp.from_seq(orf_with_region)
            mutated = pp.mutagenize_orf(
                pool, num_mutations=1, codon_positions=[1], frame=1, mode="random"
            )

        df = mutated.generate_library(num_seqs=5, seed=42)
        for seq in df["seq"]:
            # Marker should be intact
            assert "<site/>" in seq
            # Start and stop codons should be intact
            clean_seq = strip_all_tags(seq)
            assert clean_seq[:3] == "ATG"
            assert clean_seq[-3:] == "TAA"


class TestPartyMethodsWithXMLMarkers:
    """Test Party helper methods with XML regions."""

    def test_get_effective_seq_length(self):
        """Test Party.get_effective_seq_length method."""
        with pp.Party() as party:
            seq_with_region = "AC<region>TG</region>GT"
            assert party.get_effective_seq_length(seq_with_region) == 6
            assert party.get_effective_seq_length("ACGT") == 4

    def test_get_molecular_positions(self):
        """Test Party.get_molecular_positions method."""
        with pp.Party() as party:
            # With region
            seq = "AC<region/>GT"
            positions = party.get_molecular_positions(seq)
            # A=0, C=1, <region/>=2-10, G=11, T=12
            assert 0 in positions
            assert 1 in positions
            assert 11 in positions
            assert 12 in positions
            # Tag positions should be excluded
            for i in range(2, 11):
                assert i not in positions


class TestRegionClass:
    """Test the Region class and Party registration."""

    def test_region_creation(self):
        """Test creating Region objects."""
        from poolparty.region import Region

        m1 = Region(name="test", seq_length=10)
        assert m1.name == "test"
        assert m1.seq_length == 10
        assert m1._id == -1  # Not registered yet

        m2 = Region(name="zero", seq_length=0)
        assert m2.is_zero_length

        m3 = Region(name="var", seq_length=None)
        assert m3.is_variable_length

    def test_region_validation(self):
        """Test Region validation."""
        from poolparty.region import Region

        # Empty name
        with pytest.raises(ValueError, match="cannot be empty"):
            Region(name="", seq_length=5)

        # Invalid name (not identifier)
        with pytest.raises(ValueError, match="not a valid identifier"):
            Region(name="123abc", seq_length=5)

        # Negative seq_length
        with pytest.raises(ValueError, match="must be None or >= 0"):
            Region(name="test", seq_length=-1)

    def test_region_is_frozen(self):
        """Test that Region is an immutable frozen dataclass."""
        from poolparty.region import Region

        r = Region(name="test", seq_length=10)
        with pytest.raises(Exception):  # FrozenInstanceError
            r.name = "other"


class TestOrfRegionClass:
    """Test the OrfRegion class."""

    def test_orf_region_creation(self):
        """Test creating OrfRegion objects."""
        from poolparty.region import OrfRegion

        orf = OrfRegion(name="orf", seq_length=30, frame=1)
        assert orf.name == "orf"
        assert orf.seq_length == 30
        assert orf.frame == 1

    def test_orf_region_default_frame(self):
        """Test OrfRegion default frame is +1."""
        from poolparty.region import OrfRegion

        orf = OrfRegion(name="orf", seq_length=30)
        assert orf.frame == 1

    def test_orf_region_all_valid_frames(self):
        """Test all valid frame values."""
        from poolparty.region import OrfRegion

        for frame in [-3, -2, -1, 1, 2, 3]:
            orf = OrfRegion(name="orf", seq_length=30, frame=frame)
            assert orf.frame == frame

    def test_orf_region_invalid_frame(self):
        """Test invalid frame values raise error."""
        from poolparty.region import OrfRegion

        with pytest.raises(ValueError, match="frame must be one of"):
            OrfRegion(name="orf", seq_length=30, frame=0)

        with pytest.raises(ValueError, match="frame must be one of"):
            OrfRegion(name="orf", seq_length=30, frame=4)

        with pytest.raises(ValueError, match="frame must be one of"):
            OrfRegion(name="orf", seq_length=30, frame=-4)

    def test_orf_region_is_frozen(self):
        """Test that OrfRegion is an immutable frozen dataclass."""
        from poolparty.region import OrfRegion

        orf = OrfRegion(name="orf", seq_length=30, frame=1)
        with pytest.raises(Exception):  # FrozenInstanceError
            orf.frame = 2

    def test_orf_region_inherits_from_region(self):
        """Test that OrfRegion inherits Region properties."""
        from poolparty.region import OrfRegion, Region

        orf = OrfRegion(name="orf", seq_length=0, frame=1)
        assert isinstance(orf, Region)
        assert orf.is_zero_length

        orf2 = OrfRegion(name="orf2", seq_length=None, frame=-1)
        assert orf2.is_variable_length


class TestMarkerClass:
    """Test the Marker class and Party registration (legacy tests)."""

    def test_party_register_region(self):
        """Test registering regions with Party."""
        with pp.Party() as party:
            # Register a region
            m1 = party.register_region("orf", 100)
            assert m1.name == "orf"
            assert m1.seq_length == 100
            assert m1._id == 0

            # Registering same region again returns existing
            m2 = party.register_region("orf", 100)
            assert m2 is m1

            # Registering with different length raises error
            with pytest.raises(ValueError, match="already registered"):
                party.register_region("orf", 50)

    def test_party_get_region(self):
        """Test retrieving regions from Party."""
        with pp.Party() as party:
            party.register_region("test", 10)

            # Get by name
            m = party.get_region_by_name("test")
            assert m.name == "test"

            # Get by id
            m2 = party.get_region_by_id(0)
            assert m2.name == "test"

            # Not found
            with pytest.raises(ValueError, match="not found"):
                party.get_region_by_name("nonexistent")

    def test_party_has_region(self):
        """Test checking if region exists in Party."""
        with pp.Party() as party:
            assert not party.has_region("test")
            party.register_region("test", 10)
            assert party.has_region("test")

    def test_from_seq_auto_registers_regions(self):
        """Test that from_seq automatically registers regions."""
        with pp.Party() as party:
            # Create pool with marked sequence
            pool = pp.from_seq("AAA<orf>ATGCCC</orf>TTT")

            # Marker should be registered
            assert party.has_region("orf")
            m = party.get_region_by_name("orf")
            assert m.seq_length == 6  # 'ATGCCC'

    def test_pool_regions_property(self):
        """Test pool.regions property."""
        with pp.Party():
            # Create pool with region
            pool = pp.from_seq("AAA<orf>ATGCCC</orf>TTT")

            # Check regions set
            regions = pool.regions
            assert len(regions) == 1
            assert any(m.name == "orf" for m in regions)

            # Check has_region method
            assert pool.has_region("orf")
            assert not pool.has_region("nonexistent")

    def test_region_inheritance(self):
        """Test that regions are inherited from parent pools."""
        with pp.Party():
            # Create pool with region
            parent = pp.from_seq("AAA<orf>ATGCCC</orf>TTT")

            # Create child pool (e.g., through rc)
            child = pp.rc(parent)

            # Child should inherit regions
            assert child.has_region("orf")

    def test_region_removed_after_replace(self):
        """Test that region is removed from pool's set after replacement."""
        with pp.Party():
            bg = pp.from_seq("AAA<target>XXXX</target>TTT")
            assert bg.has_region("target")

            content = pp.from_seq("GGGG")
            result = pp.replace_region(bg, content, "target")

            # Marker should be removed from result
            assert not result.has_region("target")

    def test_extract_region_uses_registered_seq_length(self):
        """Test that extract_region uses registered seq_length."""
        with pp.Party():
            # Create pool with 6-char region content
            bg = pp.from_seq("AAA<orf>ATGCCC</orf>TTT")

            # Extract should have seq_length=6
            content = pp.extract_region(bg, "orf")
            assert content.seq_length == 6

    def test_mutagenize_at_region_sequential(self):
        """Test that mutagenize works correctly via apply_at_region in sequential mode."""
        with pp.Party():
            # Create template with marked region
            template = pp.from_seq("AAAA<target>ACGT</target>TTTT")

            # Apply mutagenize with 1 mutation per position (sequential)
            result = pp.apply_at_region(
                template, "target", lambda p: pp.mutagenize(p, num_mutations=1, mode="sequential")
            )

            # Should generate all single-mutation variants
            # 4 positions * 3 alternatives = 12 variants
            df = result.generate_library(num_cycles=1)
            assert len(df) == 12

            # All variants should be 12 chars (4+4+4)
            for seq in df["seq"]:
                assert len(seq) == 12


class TestIntegration:
    """Integration tests combining multiple region operations."""

    def test_scan_and_replace(self):
        """Test region_scan followed by replace_region."""
        with pp.Party():
            # Scan positions
            marked = pp.region_scan("ACGTACGT", tag_name="ins", positions=[2, 6], mode="sequential")
            # Replace with insertions
            inserts = pp.from_seq("NNN")
            result = pp.replace_region(marked, inserts, "ins")

        df = result.generate_library(num_cycles=1)
        for seq in df["seq"]:
            assert "NNN" in seq
            assert "<" not in seq  # No regions left

    def test_insert_transform_workflow(self):
        """Test typical workflow: insert region -> transform at region."""
        with pp.Party():
            # Mark a region for modification
            bg = pp.from_seq("AAAACGTACGTTTTT")  # 15 chars
            marked = pp.insert_tags(bg, "target", start=4, stop=12)  # marks CGTACGTT
            # Apply transformation at region
            result = pp.apply_at_region(marked, "target", pp.rc)

        df = result.generate_library(num_seqs=1)
        seq = df["seq"].iloc[0]
        # Check basic properties: no region tags, correct length
        assert "<" not in seq
        assert len(seq) == 15
