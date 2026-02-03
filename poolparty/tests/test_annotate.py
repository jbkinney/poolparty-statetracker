"""Tests for annotate_region and annotate_orf functions."""

import pytest

import poolparty as pp
from poolparty.region import OrfRegion


class TestAnnotateRegion:
    """Test annotate_region function."""

    def test_creates_region_with_extent(self):
        """Test annotate_region creates region with specified extent."""
        with pp.Party() as party:
            pool = pp.from_seq("ACGTACGTACGT")  # 12 bases
            result = pool.annotate_region("target", extent=(2, 8))

        df = result.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "AC<target>GTACGT</target>ACGT"

    def test_creates_region_full_sequence(self):
        """Test annotate_region with extent=None uses full sequence."""
        with pp.Party() as party:
            pool = pp.from_seq("ACGTACGT")
            result = pool.annotate_region("full")

        df = result.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "<full>ACGTACGT</full>"

    def test_registers_region_with_party(self):
        """Test that annotate_region registers the region with Party."""
        with pp.Party() as party:
            pool = pp.from_seq("ACGTACGT")
            result = pool.annotate_region("test", extent=(0, 4))

            assert party.has_region("test")
            region = party.get_region("test")
            assert region.seq_length == 4

    def test_existing_region_no_change(self):
        """Test calling annotate_region on existing region without style returns unchanged."""
        with pp.Party() as party:
            pool = pp.from_seq("<target>ACGT</target>TTTT")
            result = pool.annotate_region("target")  # No extent, no style

        df = result.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "<target>ACGT</target>TTTT"

    def test_existing_region_with_style(self):
        """Test calling annotate_region on existing region with style applies styling."""
        with pp.Party() as party:
            pool = pp.from_seq("<target>ACGT</target>TTTT")
            result = pool.annotate_region("target", style="red")

        # The pool should have styling applied (we can check it ran without error)
        df = result.generate_library(num_seqs=1)
        # Just verify the sequence structure is maintained
        assert "ACGT" in df["seq"].iloc[0]

    def test_existing_region_extent_raises_error(self):
        """Test specifying extent for existing region raises error."""
        with pp.Party():
            pool = pp.from_seq("<target>ACGT</target>TTTT")
            with pytest.raises(ValueError, match="already exists"):
                pool.annotate_region("target", extent=(0, 4))

    def test_new_region_with_style(self):
        """Test creating new region with style in one call."""
        with pp.Party():
            pool = pp.from_seq("ACGTTTTT")
            result = pool.annotate_region("target", extent=(0, 4), style="blue")

        df = result.generate_library(num_seqs=1)
        assert "<target>" in df["seq"].iloc[0]


class TestAnnotateOrf:
    """Test annotate_orf function."""

    def test_creates_orf_region_with_extent(self):
        """Test annotate_orf creates OrfRegion with specified extent."""
        with pp.Party() as party:
            pool = pp.from_seq("NNNATGCCCGGGTAANNN")  # 18 bases
            result = pool.annotate_orf("orf", extent=(3, 15), frame=1)

        df = result.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "NNN<orf>ATGCCCGGGTAA</orf>NNN"

        # Check it's registered as OrfRegion
        region = party.get_region("orf")
        assert isinstance(region, OrfRegion)
        assert region.frame == 1

    def test_creates_orf_region_full_sequence(self):
        """Test annotate_orf with extent=None uses full sequence."""
        with pp.Party() as party:
            pool = pp.from_seq("ATGCCCGGGTAA")
            result = pool.annotate_orf("orf", frame=2)

        df = result.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "<orf>ATGCCCGGGTAA</orf>"

        region = party.get_region("orf")
        assert isinstance(region, OrfRegion)
        assert region.frame == 2

    def test_default_frame_is_plus_one(self):
        """Test that default frame is +1."""
        with pp.Party() as party:
            pool = pp.from_seq("ATGCCC")
            result = pool.annotate_orf("orf")

        region = party.get_region("orf")
        assert region.frame == 1

    def test_upgrades_plain_region_to_orf_region(self):
        """Test annotate_orf upgrades existing plain Region to OrfRegion."""
        with pp.Party() as party:
            # First create a plain region
            pool = pp.from_seq("<target>ATGCCC</target>")
            assert party.has_region("target")
            assert not isinstance(party.get_region("target"), OrfRegion)

            # Now annotate as ORF
            result = pool.annotate_orf("target", frame=2)

            # Should now be an OrfRegion
            region = party.get_region("target")
            assert isinstance(region, OrfRegion)
            assert region.frame == 2

    def test_existing_orf_region_same_frame_ok(self):
        """Test calling annotate_orf on existing OrfRegion with same frame is ok."""
        with pp.Party() as party:
            pool = pp.from_seq("<orf>ATGCCC</orf>")
            # Register as OrfRegion
            party.upgrade_to_orf_region("orf", frame=1)

            # Should succeed with same frame
            result = pool.annotate_orf("orf", frame=1)
            df = result.generate_library(num_seqs=1)
            assert "ATGCCC" in df["seq"].iloc[0]

    def test_existing_orf_region_different_frame_raises_error(self):
        """Test annotate_orf on existing OrfRegion with different frame raises error."""
        with pp.Party() as party:
            pool = pp.from_seq("<orf>ATGCCC</orf>")
            # Register as OrfRegion with frame=1
            party.upgrade_to_orf_region("orf", frame=1)

            # Should fail with different frame
            with pytest.raises(ValueError, match="immutable"):
                pool.annotate_orf("orf", frame=2)

    def test_existing_region_extent_raises_error(self):
        """Test specifying extent for existing region raises error."""
        with pp.Party():
            pool = pp.from_seq("<orf>ATGCCC</orf>TTTT")
            with pytest.raises(ValueError, match="already exists"):
                pool.annotate_orf("orf", extent=(0, 6))

    def test_validates_frame_values(self):
        """Test invalid frame values raise error."""
        with pp.Party():
            pool = pp.from_seq("ATGCCC")

            with pytest.raises(ValueError, match="frame must be one of"):
                pool.annotate_orf("orf", frame=0)

            with pytest.raises(ValueError, match="frame must be one of"):
                pool.annotate_orf("orf", frame=4)

    def test_style_applies_via_stylize(self):
        """Test style parameter applies flat styling via stylize()."""
        with pp.Party():
            pool = pp.from_seq("NNNATGCCCNNN")
            result = pool.annotate_orf("orf", extent=(3, 9), style="red")

        df = result.generate_library(num_seqs=1)
        assert "<orf>" in df["seq"].iloc[0]

    def test_style_codons_applies_via_stylize_orf(self):
        """Test style_codons parameter applies via stylize_orf()."""
        with pp.Party():
            pool = pp.from_seq("ATGCCCGGG")  # 3 codons
            result = pool.annotate_orf("orf", style_codons=["red", "blue"])

        df = result.generate_library(num_seqs=1)
        assert "<orf>" in df["seq"].iloc[0]

    def test_style_frames_applies_via_stylize_orf(self):
        """Test style_frames parameter applies via stylize_orf()."""
        with pp.Party():
            pool = pp.from_seq("ATGCCCGGG")
            result = pool.annotate_orf("orf", style_frames=["red", "green", "blue"])

        df = result.generate_library(num_seqs=1)
        assert "<orf>" in df["seq"].iloc[0]

    def test_multiple_style_args_raises_error(self):
        """Test providing multiple style arguments raises error."""
        with pp.Party():
            pool = pp.from_seq("ATGCCC")

            with pytest.raises(ValueError, match="At most one"):
                pool.annotate_orf("orf", style="red", style_codons=["blue"])

            with pytest.raises(ValueError, match="At most one"):
                pool.annotate_orf("orf", style="red", style_frames=["blue", "green", "red"])

            with pytest.raises(ValueError, match="At most one"):
                pool.annotate_orf(
                    "orf", style_codons=["red"], style_frames=["blue", "green", "red"]
                )


class TestStylizeOrfFrameLookup:
    """Test stylize_orf frame lookup from OrfRegion."""

    def test_reads_frame_from_orf_region(self):
        """Test stylize_orf reads frame from OrfRegion when not specified."""
        with pp.Party() as party:
            pool = pp.from_seq("<orf>ATGCCCGGG</orf>")
            # Register as OrfRegion with frame=2
            party.upgrade_to_orf_region("orf", frame=2)

            # Should work without specifying frame
            result = pool.stylize_orf(region="orf", style_codons=["red", "blue"])

        df = result.generate_library(num_seqs=1)
        assert "ATGCCCGGG" in df["seq"].iloc[0]

    def test_raises_error_for_plain_region_no_frame(self):
        """Test stylize_orf raises error for plain Region without frame."""
        with pp.Party():
            pool = pp.from_seq("<target>ATGCCC</target>")

            with pytest.raises(ValueError, match="plain Region"):
                pool.stylize_orf(region="target", style_codons=["red", "blue"])

    def test_explicit_frame_overrides_orf_region(self):
        """Test explicit frame parameter works with OrfRegion."""
        with pp.Party() as party:
            pool = pp.from_seq("<orf>ATGCCCGGG</orf>")
            party.upgrade_to_orf_region("orf", frame=1)

            # Explicit frame=2 should work
            result = pool.stylize_orf(region="orf", style_codons=["red", "blue"], frame=2)

        df = result.generate_library(num_seqs=1)
        assert "ATGCCCGGG" in df["seq"].iloc[0]

    def test_defaults_to_frame_1_when_region_is_none(self):
        """Test stylize_orf defaults to frame=1 when region=None and frame not specified."""
        with pp.Party():
            pool = pp.from_seq("ATGCCCGGG")

            # Should succeed with default frame=1 (backward compatibility)
            result = pool.stylize_orf(style_codons=["red", "blue"])

        df = result.generate_library(num_seqs=1)
        assert "ATGCCCGGG" in df["seq"].iloc[0]


class TestMutagenizeOrfFrameLookup:
    """Test mutagenize_orf frame lookup from OrfRegion."""

    def test_reads_frame_from_orf_region(self):
        """Test mutagenize_orf reads frame from OrfRegion when not specified."""
        with pp.Party() as party:
            pool = pp.from_seq("<orf>ATGCCCGGGTAA</orf>")
            party.upgrade_to_orf_region("orf", frame=1)

            # Should work without specifying frame
            result = pool.mutagenize_orf(region="orf", num_mutations=1, mode="random")

        df = result.generate_library(num_seqs=5, seed=42)
        assert len(df) == 5

    def test_raises_error_for_plain_region_no_frame(self):
        """Test mutagenize_orf raises error for plain Region without frame."""
        with pp.Party():
            pool = pp.from_seq("<target>ATGCCCGGGTAA</target>")

            with pytest.raises(ValueError, match="plain Region"):
                pool.mutagenize_orf(region="target", num_mutations=1, mode="random")

    def test_explicit_frame_works(self):
        """Test explicit frame parameter works."""
        with pp.Party():
            pool = pp.from_seq("<target>ATGCCCGGGTAA</target>")

            # With explicit frame it should work
            result = pool.mutagenize_orf(region="target", num_mutations=1, frame=1, mode="random")

        df = result.generate_library(num_seqs=5, seed=42)
        assert len(df) == 5


class TestPartyOrfRegionMethods:
    """Test Party methods for OrfRegion handling."""

    def test_register_orf_region(self):
        """Test registering OrfRegion with Party."""
        with pp.Party() as party:
            orf = party.register_orf_region("orf", seq_length=30, frame=2)

            assert orf.name == "orf"
            assert orf.seq_length == 30
            assert orf.frame == 2
            assert isinstance(orf, OrfRegion)

    def test_register_orf_region_same_attributes_returns_existing(self):
        """Test registering same OrfRegion returns existing."""
        with pp.Party() as party:
            orf1 = party.register_orf_region("orf", seq_length=30, frame=2)
            orf2 = party.register_orf_region("orf", seq_length=30, frame=2)

            assert orf1 is orf2

    def test_register_orf_region_different_attributes_raises_error(self):
        """Test registering OrfRegion with different attributes raises error."""
        with pp.Party() as party:
            party.register_orf_region("orf", seq_length=30, frame=1)

            with pytest.raises(ValueError, match="different attributes"):
                party.register_orf_region("orf", seq_length=30, frame=2)

    def test_register_orf_region_when_plain_region_exists_raises_error(self):
        """Test registering OrfRegion when plain Region exists raises error."""
        with pp.Party() as party:
            party.register_region("test", seq_length=30)

            with pytest.raises(ValueError, match="plain Region"):
                party.register_orf_region("test", seq_length=30, frame=1)

    def test_upgrade_to_orf_region(self):
        """Test upgrading plain Region to OrfRegion."""
        with pp.Party() as party:
            # First register plain Region
            plain = party.register_region("target", seq_length=30)
            assert not isinstance(plain, OrfRegion)

            # Upgrade to OrfRegion
            orf = party.upgrade_to_orf_region("target", frame=2)

            assert isinstance(orf, OrfRegion)
            assert orf.name == "target"
            assert orf.seq_length == 30
            assert orf.frame == 2

            # Party should now return the OrfRegion
            assert party.get_region("target") is orf

    def test_upgrade_nonexistent_region_raises_error(self):
        """Test upgrading non-existent region raises error."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="not found"):
                party.upgrade_to_orf_region("nonexistent", frame=1)

    def test_upgrade_orf_region_raises_error(self):
        """Test upgrading existing OrfRegion raises error."""
        with pp.Party() as party:
            party.register_orf_region("orf", seq_length=30, frame=1)

            with pytest.raises(ValueError, match="already an OrfRegion"):
                party.upgrade_to_orf_region("orf", frame=2)


class TestAnnotateOrfIntegration:
    """Integration tests for annotate_orf with ORF operations."""

    def test_annotate_existing_region_then_mutagenize_without_frame(self):
        """Test annotate_orf on existing region followed by mutagenize_orf without specifying frame."""
        with pp.Party() as party:
            # Create pool with pre-existing tags (preserves seq_length)
            pool = pp.from_seq("<orf>ATGCCCGGGTAA</orf>")

            # Annotate as ORF with frame=2 (upgrades existing plain Region to OrfRegion)
            annotated = pool.annotate_orf("orf", frame=2)

            # Verify it's registered as OrfRegion
            region = party.get_region("orf")
            assert isinstance(region, OrfRegion)
            assert region.frame == 2

            # Mutagenize without specifying frame - should use frame from OrfRegion
            mutated = annotated.mutagenize_orf(region="orf", num_mutations=1, mode="random")

        df = mutated.generate_library(num_seqs=5, seed=42)
        assert len(df) == 5

    def test_annotate_then_stylize_without_frame(self):
        """Test annotate_orf followed by stylize_orf without specifying frame."""
        with pp.Party():
            pool = pp.from_seq("ATGCCCGGGTAA")
            annotated = pool.annotate_orf("orf", frame=3)

            # Stylize without specifying frame
            styled = annotated.stylize_orf(region="orf", style_codons=["red", "blue"])

        df = styled.generate_library(num_seqs=1)
        assert "<orf>" in df["seq"].iloc[0]

    def test_frame_lookup_with_stylize_orf(self):
        """Test that stylize_orf correctly looks up frame from OrfRegion."""
        with pp.Party() as party:
            # Create pool with tags
            pool = pp.from_seq("<orf>ATGCCCGGG</orf>")
            # Upgrade to OrfRegion with frame=2
            party.upgrade_to_orf_region("orf", frame=2)

            # stylize_orf should use the frame from OrfRegion
            styled = pool.stylize_orf(region="orf", style_codons=["red", "blue"])

        df = styled.generate_library(num_seqs=1)
        assert "ATGCCCGGG" in df["seq"].iloc[0]
