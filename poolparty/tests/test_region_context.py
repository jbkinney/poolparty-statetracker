"""Tests for RegionContext helper class."""

import numpy as np

from poolparty.utils.region_context import RegionContext
from poolparty.utils.style_utils import SeqStyle


class TestRegionContextFromSequence:
    """Test RegionContext.from_sequence() class method."""

    def test_from_named_region(self):
        """Create RegionContext from named region."""
        seq = "AAA<test>CCCC</test>GGG"
        ctx = RegionContext.from_sequence(seq, "test", remove_tags=True)

        assert ctx.region_content == "CCCC"
        assert ctx.region_name == "test"
        assert ctx.strand is None  # Strand no longer stored in tags
        assert ctx.remove_tags is True
        # Bounds should point to content positions
        assert ctx.region_start == 9  # After <test>
        assert ctx.region_end == 13  # Before </test>

    def test_from_interval_region(self):
        """Create RegionContext from interval region."""
        seq = "AAACCCCGGG"
        ctx = RegionContext.from_sequence(seq, [3, 7], remove_tags=True)

        assert ctx.region_content == "CCCC"
        assert ctx.prefix == "AAA"
        assert ctx.suffix == "GGG"
        assert ctx.region_name is None
        assert ctx.strand is None
        assert ctx.region_start == 3
        assert ctx.region_end == 7

    def test_prefix_suffix_clean_parts(self):
        """Prefix and suffix are clean parts (without tags) for named regions."""
        seq = "AAA<test>CCCC</test>GGG"
        ctx = RegionContext.from_sequence(seq, "test")

        # prefix/suffix are the parts before/after the entire region (including tags)
        assert ctx.prefix == "AAA"
        assert ctx.suffix == "GGG"


class TestRegionContextSplitStyles:
    """Test RegionContext.split_parent_styles()."""

    def test_split_styles_with_parent(self):
        """Split parent styles into prefix/region/suffix."""
        seq = "AAA<test>CCCC</test>GGG"
        ctx = RegionContext.from_sequence(seq, "test")

        # Create style on full sequence
        positions = np.array([0, 1, 2, 9, 10, 11, 12, 19, 20, 21])  # AAA, CCCC, GGG
        parent_style = SeqStyle.from_style_list([("red", positions)], len(seq))

        region_style = ctx.split_parent_styles([parent_style])

        # Region style should be 0-indexed for region content
        assert len(region_style) == len("CCCC")
        # Check that positions are adjusted (should be 0,1,2,3 for CCCC)
        spec, pos = region_style.style_list[0]
        assert spec == "red"
        assert list(pos) == [0, 1, 2, 3]

    def test_split_styles_no_parent(self):
        """Split with no parent styles returns empty."""
        seq = "AAA<test>CCCC</test>GGG"
        ctx = RegionContext.from_sequence(seq, "test")

        region_style = ctx.split_parent_styles(None)

        assert len(region_style) == len("CCCC")
        assert len(region_style.style_list) == 0

    def test_split_styles_empty_list(self):
        """Split with empty parent list returns empty."""
        seq = "AAA<test>CCCC</test>GGG"
        ctx = RegionContext.from_sequence(seq, "test")

        region_style = ctx.split_parent_styles([])

        assert len(region_style) == len("CCCC")
        assert len(region_style.style_list) == 0


class TestRegionContextReassembleSeq:
    """Test RegionContext.reassemble_seq()."""

    def test_reassemble_interval_region(self):
        """Reassemble sequence from interval region."""
        seq = "AAACCCCGGG"
        ctx = RegionContext.from_sequence(seq, [3, 7])

        result = ctx.reassemble_seq_string("TTTT")

        assert result == "AAATTTTGGG"

    def test_reassemble_named_region_remove_tags(self):
        """Reassemble named region with tags removed."""
        seq = "AAA<test>CCCC</test>GGG"
        ctx = RegionContext.from_sequence(seq, "test", remove_tags=True)

        result = ctx.reassemble_seq_string("TTTT")

        assert result == "AAATTTTGGG"
        assert "<test>" not in result

    def test_reassemble_named_region_keep_tags(self):
        """Reassemble named region keeping tags."""
        seq = "AAA<test>CCCC</test>GGG"
        ctx = RegionContext.from_sequence(seq, "test", remove_tags=False)

        result = ctx.reassemble_seq_string("TTTT")

        assert result == "AAA<test>TTTT</test>GGG"

    def test_reassemble_named_region_different_length(self):
        """Reassemble with different output length."""
        seq = "AAA<test>CCCC</test>GGG"
        ctx = RegionContext.from_sequence(seq, "test", remove_tags=True)

        result = ctx.reassemble_seq_string("TT")

        assert result == "AAATTGGG"


class TestRegionContextReassembleStyle:
    """Test RegionContext.reassemble_style()."""

    def test_reassemble_style_interval_region(self):
        """Reassemble style from interval region."""
        seq = "AAACCCCGGG"
        ctx = RegionContext.from_sequence(seq, [3, 7])

        # Create parent style
        parent_positions = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        parent_style = SeqStyle.from_style_list([("red", parent_positions)], len(seq))
        ctx.split_parent_styles([parent_style])

        # Create output style (0-indexed for region)
        output_positions = np.array([0, 1, 2, 3])
        output_style = SeqStyle.from_style_list([("blue", output_positions)], 4)

        result = ctx.reassemble_style(output_style, "TTTT")

        # Result should have blue in positions 3-6 (where region is in full seq)
        assert len(result) == len("AAATTTTGGG")
        spec, positions = result.style_list[1]  # Second item (first is red from prefix)
        assert spec == "blue"
        assert list(positions) == [3, 4, 5, 6]

    def test_reassemble_style_named_region_remove_tags(self):
        """Reassemble style from named region with tags removed."""
        seq = "AAA<test>CCCC</test>GGG"
        ctx = RegionContext.from_sequence(seq, "test", remove_tags=True)

        # Create parent style on full sequence (including tag positions)
        parent_style = SeqStyle.empty(len(seq))
        ctx.split_parent_styles([parent_style])

        # Create output style
        output_positions = np.array([0, 1])
        output_style = SeqStyle.from_style_list([("blue", output_positions)], 2)

        result = ctx.reassemble_style(output_style, "TT")

        # Result should be AAA + TT + GGG (no tags)
        assert len(result) == len("AAATTGGG")

    def test_reassemble_style_named_region_keep_tags(self):
        """Reassemble style from named region keeping tags."""
        seq = "AAA<test>CCCC</test>GGG"
        ctx = RegionContext.from_sequence(seq, "test", remove_tags=False)

        # Create parent style
        parent_style = SeqStyle.empty(len(seq))
        ctx.split_parent_styles([parent_style])

        # Create output style
        output_positions = np.array([0, 1])
        output_style = SeqStyle.from_style_list([("blue", output_positions)], 2)

        result = ctx.reassemble_style(output_style, "TT")

        # Result should be AAA + <test> + TT + </test> + GGG
        assert len(result) == len("AAA<test>TT</test>GGG")

    def test_reassemble_style_no_parent_styles(self):
        """Reassemble style with no parent styles."""
        seq = "AAA<test>CCCC</test>GGG"
        ctx = RegionContext.from_sequence(seq, "test", remove_tags=True)

        # No parent styles
        ctx.split_parent_styles(None)

        # Create output style
        output_positions = np.array([0, 1])
        output_style = SeqStyle.from_style_list([("blue", output_positions)], 4)

        result = ctx.reassemble_style(output_style, "TTTT")

        # Should have correct length
        assert len(result) == len("AAATTTTGGG")


class TestRegionContextIntegration:
    """Integration tests for complete region processing workflow."""

    def test_full_workflow_named_region(self):
        """Test complete workflow: split -> compute -> reassemble."""
        seq = "AAA<test>CCCC</test>GGG"
        ctx = RegionContext.from_sequence(seq, "test", remove_tags=True)

        # Create parent style
        parent_positions = np.array([0, 9, 10])  # Style on A and first two Cs
        parent_style = SeqStyle.from_style_list([("red", parent_positions)], len(seq))

        # Split
        region_style = ctx.split_parent_styles([parent_style])

        # Simulate compute (just uppercase the region)
        output_seq = ctx.region_content.upper()
        output_style = region_style  # Pass through

        # Reassemble
        final_seq = ctx.reassemble_seq_string(output_seq)
        final_style = ctx.reassemble_style(output_style, output_seq)

        assert final_seq == "AAACCCCGGG"
        assert len(final_style) == len(final_seq)

    def test_full_workflow_interval_region(self):
        """Test complete workflow with interval region."""
        seq = "AAACCCCGGG"
        ctx = RegionContext.from_sequence(seq, [3, 7], remove_tags=True)

        # Create parent style
        parent_positions = np.array([0, 3, 4])
        parent_style = SeqStyle.from_style_list([("red", parent_positions)], len(seq))

        # Split
        region_style = ctx.split_parent_styles([parent_style])

        # Simulate compute
        output_seq = "TT"
        output_positions = np.array([0])
        output_style = SeqStyle.from_style_list([("blue", output_positions)], len(output_seq))

        # Reassemble
        final_seq = ctx.reassemble_seq_string(output_seq)
        final_style = ctx.reassemble_style(output_style, output_seq)

        assert final_seq == "AAATTGGG"
        assert len(final_style) == len(final_seq)
