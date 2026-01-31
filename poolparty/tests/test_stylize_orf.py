"""Tests for stylize_orf functionality."""

import numpy as np
import pytest

import poolparty as pp
from poolparty.orf_ops.stylize_orf import stylize_orf


class TestStylizeOrfBasic:
    """Test basic stylize_orf functionality."""

    def test_style_codons_basic(self):
        """style_codons applies cycling styles to codons."""
        with pp.Party():
            pool = stylize_orf("ACGTACGTAC", style_codons=["red", "blue"]).named("test")

        df = pool.generate_library(
            num_seqs=1, report_design_cards=True, _include_inline_styles=True
        )
        seq_style = df["_inline_styles"].iloc[0]

        # Should have styles applied
        assert len(seq_style.style_list) > 0

        # Collect positions for each style
        red_positions = set()
        blue_positions = set()
        for spec, positions in seq_style.style_list:
            if spec == "red":
                red_positions.update(positions)
            elif spec == "blue":
                blue_positions.update(positions)

        # Codon 0 (ACG, positions 0-2) should be red
        # Codon 1 (TAC, positions 3-5) should be blue
        # Codon 2 (GTA, positions 6-8) should be red
        # Codon 3 (C, position 9) should be blue (partial codon)
        assert {0, 1, 2, 6, 7, 8} == red_positions
        assert {3, 4, 5, 9} == blue_positions

    def test_style_frames_basic(self):
        """style_frames applies styles based on frame position."""
        with pp.Party():
            pool = stylize_orf("ACGTAC", style_frames=["red", "green", "blue"]).named("test")

        df = pool.generate_library(
            num_seqs=1, report_design_cards=True, _include_inline_styles=True
        )
        seq_style = df["_inline_styles"].iloc[0]

        # Collect positions for each style
        style_positions = {"red": set(), "green": set(), "blue": set()}
        for spec, positions in seq_style.style_list:
            if spec in style_positions:
                style_positions[spec].update(positions)

        # Frame 0: positions 0, 3
        # Frame 1: positions 1, 4
        # Frame 2: positions 2, 5
        assert style_positions["red"] == {0, 3}
        assert style_positions["green"] == {1, 4}
        assert style_positions["blue"] == {2, 5}

    def test_mutual_exclusivity(self):
        """style_codons and style_frames are mutually exclusive."""
        with pp.Party():
            with pytest.raises(ValueError, match="mutually exclusive"):
                stylize_orf("ACGTAC", style_codons=["red"], style_frames=["a", "b", "c"])

    def test_one_required(self):
        """Either style_codons or style_frames must be provided."""
        with pp.Party():
            with pytest.raises(ValueError, match="must be provided"):
                stylize_orf("ACGTAC")

    def test_style_frames_requires_multiple_of_three(self):
        """style_frames length must be a multiple of 3."""
        with pp.Party():
            with pytest.raises(ValueError, match="multiple of 3"):
                stylize_orf("ACGTAC", style_frames=["red", "blue"])
            with pytest.raises(ValueError, match="multiple of 3"):
                stylize_orf("ACGTAC", style_frames=["a", "b", "c", "d"])

    def test_style_frames_empty(self):
        """style_frames must not be empty."""
        with pp.Party():
            with pytest.raises(ValueError, match="must not be empty"):
                stylize_orf("ACGTAC", style_frames=[])

    def test_style_codons_nonempty(self):
        """style_codons must not be empty."""
        with pp.Party():
            with pytest.raises(ValueError, match="must not be empty"):
                stylize_orf("ACGTAC", style_codons=[])

    def test_region_frame_validation(self):
        """region_frame must be 0, 1, or 2."""
        with pp.Party():
            with pytest.raises(ValueError, match="must be 0, 1, or 2"):
                stylize_orf("ACGTAC", style_frames=["a", "b", "c"], region_frame=3)


class TestStylizeOrfFramesCycling:
    """Test style_frames cycling through groups of 3."""

    def test_style_frames_six_styles(self):
        """style_frames with 6 styles cycles through 2 groups."""
        with pp.Party():
            # 9 nucleotides = 3 codons
            # Group 0 (codon 0): red, green, blue
            # Group 1 (codon 1): cyan, magenta, yellow
            # Group 0 (codon 2): red, green, blue (cycles back)
            pool = stylize_orf(
                "ACGTACGTA",
                style_frames=["red", "green", "blue", "cyan", "magenta", "yellow"]
            ).named("test")

        df = pool.generate_library(
            num_seqs=1, report_design_cards=True, _include_inline_styles=True
        )
        seq_style = df["_inline_styles"].iloc[0]

        style_positions = {}
        for spec, positions in seq_style.style_list:
            style_positions[spec] = set(positions)

        # Codon 0 (positions 0,1,2): red, green, blue
        # Codon 1 (positions 3,4,5): cyan, magenta, yellow
        # Codon 2 (positions 6,7,8): red, green, blue
        assert style_positions.get("red", set()) == {0, 6}
        assert style_positions.get("green", set()) == {1, 7}
        assert style_positions.get("blue", set()) == {2, 8}
        assert style_positions.get("cyan", set()) == {3}
        assert style_positions.get("magenta", set()) == {4}
        assert style_positions.get("yellow", set()) == {5}

    def test_style_frames_three_styles(self):
        """style_frames with 3 styles applies same group to all codons."""
        with pp.Party():
            pool = stylize_orf("ACGTACGTA", style_frames=["red", "green", "blue"]).named("test")

        df = pool.generate_library(
            num_seqs=1, report_design_cards=True, _include_inline_styles=True
        )
        seq_style = df["_inline_styles"].iloc[0]

        style_positions = {}
        for spec, positions in seq_style.style_list:
            style_positions[spec] = set(positions)

        # All codons use the same 3 styles
        assert style_positions.get("red", set()) == {0, 3, 6}
        assert style_positions.get("green", set()) == {1, 4, 7}
        assert style_positions.get("blue", set()) == {2, 5, 8}


class TestStylizeOrfRegionFrame:
    """Test region_frame parameter."""

    def test_region_frame_0(self):
        """region_frame=0 starts from frame 0."""
        with pp.Party():
            pool = stylize_orf("ACGTAC", style_frames=["red", "green", "blue"], region_frame=0).named("test")

        df = pool.generate_library(
            num_seqs=1, report_design_cards=True, _include_inline_styles=True
        )
        seq_style = df["_inline_styles"].iloc[0]

        style_positions = {"red": set(), "green": set(), "blue": set()}
        for spec, positions in seq_style.style_list:
            if spec in style_positions:
                style_positions[spec].update(positions)

        # Frame 0: positions 0, 3
        assert style_positions["red"] == {0, 3}

    def test_region_frame_1(self):
        """region_frame=1 shifts frame assignment."""
        with pp.Party():
            pool = stylize_orf("ACGTAC", style_frames=["red", "green", "blue"], region_frame=1).named("test")

        df = pool.generate_library(
            num_seqs=1, report_design_cards=True, _include_inline_styles=True
        )
        seq_style = df["_inline_styles"].iloc[0]

        style_positions = {"red": set(), "green": set(), "blue": set()}
        for spec, positions in seq_style.style_list:
            if spec in style_positions:
                style_positions[spec].update(positions)

        # With region_frame=1:
        # Position 0 -> frame (0 + 1) % 3 = 1 -> green
        # Position 1 -> frame (1 + 1) % 3 = 2 -> blue
        # Position 2 -> frame (2 + 1) % 3 = 0 -> red
        # Position 3 -> frame (3 + 1) % 3 = 1 -> green
        # Position 4 -> frame (4 + 1) % 3 = 2 -> blue
        # Position 5 -> frame (5 + 1) % 3 = 0 -> red
        assert style_positions["red"] == {2, 5}
        assert style_positions["green"] == {0, 3}
        assert style_positions["blue"] == {1, 4}

    def test_region_frame_2(self):
        """region_frame=2 shifts frame assignment."""
        with pp.Party():
            pool = stylize_orf("ACGTAC", style_frames=["red", "green", "blue"], region_frame=2).named("test")

        df = pool.generate_library(
            num_seqs=1, report_design_cards=True, _include_inline_styles=True
        )
        seq_style = df["_inline_styles"].iloc[0]

        style_positions = {"red": set(), "green": set(), "blue": set()}
        for spec, positions in seq_style.style_list:
            if spec in style_positions:
                style_positions[spec].update(positions)

        # With region_frame=2:
        # Position 0 -> frame (0 + 2) % 3 = 2 -> blue
        # Position 1 -> frame (1 + 2) % 3 = 0 -> red
        # Position 2 -> frame (2 + 2) % 3 = 1 -> green
        assert style_positions["red"] == {1, 4}
        assert style_positions["green"] == {2, 5}
        assert style_positions["blue"] == {0, 3}


class TestStylizeOrfReverse:
    """Test reverse parameter."""

    def test_reverse_style_codons(self):
        """reverse=True processes codons from end to start."""
        with pp.Party():
            # Without reverse
            pool_fwd = stylize_orf("ACGTAC", style_codons=["red", "blue"], reverse=False).named("fwd")
            # With reverse
            pool_rev = stylize_orf("ACGTAC", style_codons=["red", "blue"], reverse=True).named("rev")

        df_fwd = pool_fwd.generate_library(
            num_seqs=1, report_design_cards=True, _include_inline_styles=True
        )
        df_rev = pool_rev.generate_library(
            num_seqs=1, report_design_cards=True, _include_inline_styles=True
        )

        # Collect positions
        fwd_red = set()
        fwd_blue = set()
        for spec, positions in df_fwd["_inline_styles"].iloc[0].style_list:
            if spec == "red":
                fwd_red.update(positions)
            elif spec == "blue":
                fwd_blue.update(positions)

        rev_red = set()
        rev_blue = set()
        for spec, positions in df_rev["_inline_styles"].iloc[0].style_list:
            if spec == "red":
                rev_red.update(positions)
            elif spec == "blue":
                rev_blue.update(positions)

        # Forward: codon 0 (0-2)=red, codon 1 (3-5)=blue
        # Reverse: processes from end, so codon assignment is reversed
        assert fwd_red != rev_red or fwd_blue != rev_blue  # Styles should differ

    def test_reverse_style_frames(self):
        """reverse=True processes frames from end to start."""
        with pp.Party():
            pool_rev = stylize_orf("ACGTAC", style_frames=["red", "green", "blue"], reverse=True).named("test")

        df = pool_rev.generate_library(
            num_seqs=1, report_design_cards=True, _include_inline_styles=True
        )
        seq_style = df["_inline_styles"].iloc[0]

        style_positions = {"red": set(), "green": set(), "blue": set()}
        for spec, positions in seq_style.style_list:
            if spec in style_positions:
                style_positions[spec].update(positions)

        # With reverse=True, processing from position 5 to 0:
        # Position 5 is idx 0 -> frame 0 -> red
        # Position 4 is idx 1 -> frame 1 -> green
        # Position 3 is idx 2 -> frame 2 -> blue
        # Position 2 is idx 3 -> frame 0 -> red
        # Position 1 is idx 4 -> frame 1 -> green
        # Position 0 is idx 5 -> frame 2 -> blue
        assert style_positions["red"] == {5, 2}
        assert style_positions["green"] == {4, 1}
        assert style_positions["blue"] == {3, 0}


class TestStylizeOrfRegion:
    """Test region parameter."""

    def test_region_named(self):
        """Named region restricts styling."""
        with pp.Party():
            pool = stylize_orf(
                "AA<cre>ACGTAC</cre>TT",
                region="cre",
                style_frames=["red", "green", "blue"]
            ).named("test")

        df = pool.generate_library(
            num_seqs=1, report_design_cards=True, _include_inline_styles=True
        )
        seq = df["seq"].iloc[0]
        seq_style = df["_inline_styles"].iloc[0]

        # Collect all styled positions
        all_positions = set()
        for spec, positions in seq_style.style_list:
            all_positions.update(positions)

        # Only positions within the region content should be styled
        # AA<cre>ACGTAC</cre>TT
        # 0123456789...
        # Region content starts at position 6 (after <cre>)
        assert all(pos >= 6 for pos in all_positions)
        # Positions should not include the prefix AA or suffix TT
        assert 0 not in all_positions
        assert 1 not in all_positions

    def test_region_interval(self):
        """[start, stop] interval restricts styling."""
        with pp.Party():
            pool = stylize_orf(
                "AAACGTACTT",
                region=[2, 8],  # ACGTAC
                style_frames=["red", "green", "blue"]
            ).named("test")

        df = pool.generate_library(
            num_seqs=1, report_design_cards=True, _include_inline_styles=True
        )
        seq_style = df["_inline_styles"].iloc[0]

        # Collect all styled positions
        all_positions = set()
        for spec, positions in seq_style.style_list:
            all_positions.update(positions)

        # Only positions 2-7 should be styled
        assert all_positions == {2, 3, 4, 5, 6, 7}


class TestStylizeOrfSkipNonMolecular:
    """Test that non-molecular characters are skipped."""

    def test_skip_gaps(self):
        """Gap characters are skipped in frame calculation."""
        with pp.Party():
            # Sequence with gap: ACG-TAC
            pool = stylize_orf("ACG-TAC", style_frames=["red", "green", "blue"]).named("test")

        df = pool.generate_library(
            num_seqs=1, report_design_cards=True, _include_inline_styles=True
        )
        seq_style = df["_inline_styles"].iloc[0]

        # Collect positions
        style_positions = {"red": set(), "green": set(), "blue": set()}
        for spec, positions in seq_style.style_list:
            if spec in style_positions:
                style_positions[spec].update(positions)

        # Gap at position 3 should not be styled
        all_positions = style_positions["red"] | style_positions["green"] | style_positions["blue"]
        assert 3 not in all_positions

        # Molecular positions are 0,1,2,4,5,6 (ACG TAC)
        # Frame 0: 0, 4 (A, T)
        # Frame 1: 1, 5 (C, A)
        # Frame 2: 2, 6 (G, C)
        assert style_positions["red"] == {0, 4}
        assert style_positions["green"] == {1, 5}
        assert style_positions["blue"] == {2, 6}

    def test_skip_tags(self):
        """Tag characters are skipped in frame calculation."""
        with pp.Party():
            pool = stylize_orf("<x>ACGTAC</x>", style_frames=["red", "green", "blue"]).named("test")

        df = pool.generate_library(
            num_seqs=1, report_design_cards=True, _include_inline_styles=True
        )
        seq_style = df["_inline_styles"].iloc[0]

        # Only molecular positions should be styled
        all_positions = set()
        for spec, positions in seq_style.style_list:
            all_positions.update(positions)

        # Tags should not be in styled positions
        # <x> is positions 0-2, </x> is positions 9-12
        for pos in all_positions:
            assert pos >= 3 and pos <= 8  # Only ACGTAC content

    def test_codons_with_gaps(self):
        """Codon styling correctly handles gaps."""
        with pp.Party():
            # Sequence: ACG--TACGTA (gaps at positions 3,4)
            pool = stylize_orf("ACG--TACGTA", style_codons=["red", "blue"]).named("test")

        df = pool.generate_library(
            num_seqs=1, report_design_cards=True, _include_inline_styles=True
        )
        seq_style = df["_inline_styles"].iloc[0]

        # Collect positions
        red_positions = set()
        blue_positions = set()
        for spec, positions in seq_style.style_list:
            if spec == "red":
                red_positions.update(positions)
            elif spec == "blue":
                blue_positions.update(positions)

        # Molecular positions: 0,1,2,5,6,7,8,9,10 (ACG TACGTA)
        # Codon 0: positions 0,1,2 (ACG) -> red
        # Codon 1: positions 5,6,7 (TAC) -> blue
        # Codon 2: positions 8,9,10 (GTA) -> red
        assert red_positions == {0, 1, 2, 8, 9, 10}
        assert blue_positions == {5, 6, 7}


class TestStylizeOrfChain:
    """Test stylize_orf in operation chains."""

    def test_chain_with_other_ops(self):
        """stylize_orf works in operation chains."""
        with pp.Party():
            pool = (
                pp.from_seq("ACGTACGTAC")
                .stylize_orf(style_codons=["red", "blue"])
                .named("test")
            )

        df = pool.generate_library(
            num_seqs=1, report_design_cards=True, _include_inline_styles=True
        )
        seq_style = df["_inline_styles"].iloc[0]

        assert len(seq_style.style_list) > 0

    def test_styles_suppressed(self):
        """stylize_orf respects style suppression."""
        with pp.Party():
            pp.toggle_styles(on=False)
            pool = stylize_orf("ACGTAC", style_codons=["red"]).named("test")

        df = pool.generate_library(
            num_seqs=1, report_design_cards=True, _include_inline_styles=True
        )
        seq_style = df["_inline_styles"].iloc[0]

        # Styles should be None when suppressed
        assert seq_style is None
