"""Tests for inline styling functionality."""

import numpy as np
import pytest

import poolparty as pp
from poolparty.base_ops.mutagenize import mutagenize
from poolparty.utils.style_utils import (
    SeqStyle,
    apply_inline_styles,
    reset,
    validate_style_positions,
)


class TestInlineStylesBasic:
    """Test basic inline styling functionality."""

    def test_apply_inline_styles_empty(self):
        """Empty styles returns unchanged sequence."""
        result = apply_inline_styles("ACGT", [])
        assert result == "ACGT"

    def test_apply_inline_styles_single_position(self):
        """Single position styled correctly."""
        styles = [("red", np.array([1]))]
        result = apply_inline_styles("ACGT", styles)
        # Should have ANSI codes around position 1 (C)
        assert "\033[" in result
        assert "A" in result
        assert "C" in result
        assert "G" in result
        assert "T" in result

    def test_apply_inline_styles_multiple_positions(self):
        """Multiple positions styled correctly."""
        styles = [("blue", np.array([0, 2]))]
        result = apply_inline_styles("ACGT", styles)
        assert "\033[" in result

    def test_apply_inline_styles_overlapping(self):
        """Overlapping styles combine correctly."""
        styles = [
            ("bold", np.array([1, 2])),
            ("red", np.array([1])),  # Override with red at position 1
        ]
        result = apply_inline_styles("ACGT", styles)
        assert "\033[" in result


class TestMutagenizeWithStyleMutations:
    """Test mutagenize with style parameter."""

    def test_style_none_by_default(self):
        """Default style is None."""
        with pp.Party() as party:
            pool = mutagenize("ACGT", num_mutations=1)
            assert pool.operation._style is None

    def test_style_stored(self):
        """style parameter is stored on operation."""
        with pp.Party() as party:
            pool = mutagenize("ACGT", num_mutations=1, style="red bold")
            assert pool.operation._style == "red bold"

    def test_style_in_copy_params(self):
        """style is included in _get_copy_params."""
        with pp.Party() as party:
            pool = mutagenize("ACGT", num_mutations=1, style="blue")
            params = pool.operation._get_copy_params()
        assert params["style"] == "blue"

    def test_compute_returns_style(self):
        """compute() returns style key."""
        with pp.Party() as party:
            pool = mutagenize("ACGT", num_mutations=1, mode="sequential")

        pool.operation.state._value = 0
        output_seq, card = pool.operation.compute([pp.types.Seq.from_string("ACGT")])
        assert isinstance(output_seq.style, SeqStyle)

    def test_compute_with_style_includes_positions(self):
        """compute() with style adds mutation positions to style."""
        with pp.Party() as party:
            pool = mutagenize("ACGT", num_mutations=1, style="red", mode="sequential")

        pool.operation.state._value = 0
        output_seq, card = pool.operation.compute([pp.types.Seq.from_string("ACGT")])

        # Should have one style tuple with mutation positions
        seq_style = output_seq.style
        assert len(seq_style.style_list) == 1
        spec, positions = seq_style.style_list[0]
        assert spec == "red"
        assert len(positions) == 1  # 1 mutation

    def test_compute_without_style_empty(self):
        """compute() without style returns empty style."""
        with pp.Party() as party:
            pool = mutagenize("ACGT", num_mutations=1, mode="sequential")

        pool.operation.state._value = 0
        output_seq, card = pool.operation.compute([pp.types.Seq.from_string("ACGT")])

        seq_style = output_seq.style
        assert not seq_style  # Empty SeqStyle


class TestInlineStylesGeneration:
    """Test inline styles flow through library generation."""

    def test_generate_library_includes_inline_styles(self):
        """generate_library includes _inline_styles in rows."""
        with pp.Party() as party:
            pool = mutagenize("ACGT", num_mutations=1, style="red", mode="sequential").named(
                "mutant"
            )

        df = pool.generate_library(
            num_seqs=1, _include_inline_styles=True
        )
        assert "_inline_styles" in df.columns

        # Should have the mutation style
        seq_style = df["_inline_styles"].iloc[0]
        assert len(seq_style.style_list) == 1
        assert seq_style.style_list[0][0] == "red"

    def test_generate_library_without_style(self):
        """generate_library without style has empty _inline_styles."""
        with pp.Party() as party:
            pool = mutagenize("ACGT", num_mutations=1, mode="sequential").named("mutant")

        df = pool.generate_library(
            num_seqs=1, _include_inline_styles=True
        )
        assert "_inline_styles" in df.columns

        seq_style = df["_inline_styles"].iloc[0]
        assert not seq_style  # Empty SeqStyle


class TestInlineStylesChain:
    """Test inline styles propagate through operation chains."""

    def test_styles_pass_through_state_ops(self):
        """Styles pass through state operations unchanged."""
        with pp.Party() as party:
            pool = mutagenize("ACGT", num_mutations=1, style="red", mode="sequential")
            repeated = pool.repeat(2).named("repeated")

        df = repeated.generate_library(
            num_seqs=1, _include_inline_styles=True
        )

        # Styles should have passed through repeat
        seq_style = df["_inline_styles"].iloc[0]
        assert len(seq_style.style_list) == 1
        assert seq_style.style_list[0][0] == "red"

    def test_styles_from_stacked_pools(self):
        """Stack operation passes through styles from active parent."""
        with pp.Party() as party:
            pool1 = mutagenize("ACGT", num_mutations=1, style="red", mode="sequential")
            pool2 = mutagenize("TTTT", num_mutations=1, style="blue", mode="sequential")
            stacked = pp.stack([pool1, pool2]).named("stacked")

        df = stacked.generate_library(
            num_seqs=2, _include_inline_styles=True
        )

        # First row from pool1 should have red style
        seq_style0 = df["_inline_styles"].iloc[0]
        if seq_style0:  # May be empty if mutation didn't occur
            assert seq_style0.style_list[0][0] == "red"


class TestInlineStylesPositionAdjustment:
    """Test position adjustment in compute() for regions."""

    def test_region_adjusts_positions(self):
        """Region-based operations adjust style positions."""
        with pp.Party() as party:
            # Create a sequence with a marker
            bg = pp.from_seq("AA<test>CCCC</test>GG").named("bg")
            # Mutagenize within the region
            mutated = bg.mutagenize(
                region="test", num_mutations=1, style="red", mode="sequential"
            ).named("mutated")

        df = mutated.generate_library(
            num_seqs=1, _include_inline_styles=True
        )

        # Check that style positions are adjusted to full sequence positions
        seq_style = df["_inline_styles"].iloc[0]
        if seq_style:
            spec, positions = seq_style.style_list[0]
            # Positions should be relative to full sequence, not just region
            # The region starts at position 2 (after 'AA')
            assert all(pos >= 2 for pos in positions)


class TestPositionValidation:
    """Test position validation for inline styles."""

    def test_valid_positions_pass(self):
        """Valid positions don't raise any errors."""
        styles = [("red", np.array([0, 1, 2, 3]))]
        # Should not raise
        validate_style_positions(4, styles)

    def test_valid_positions_empty_array(self):
        """Empty position arrays are valid."""
        styles = [("red", np.array([], dtype=np.int64))]
        # Should not raise
        validate_style_positions(4, styles)

    def test_negative_position_raises(self):
        """Negative positions raise ValueError."""
        styles = [("red", np.array([-1, 0, 1]))]
        with pytest.raises(ValueError) as excinfo:
            validate_style_positions(4, styles)
        assert "negative" in str(excinfo.value).lower()
        assert "red" in str(excinfo.value)

    def test_out_of_bounds_position_raises(self):
        """Position >= seq_len raises ValueError."""
        styles = [("blue", np.array([0, 4]))]  # 4 is out of bounds for len=4
        with pytest.raises(ValueError) as excinfo:
            validate_style_positions(4, styles)
        assert ">= seq_len" in str(excinfo.value)
        assert "blue" in str(excinfo.value)

    def test_validation_error_message_includes_context(self):
        """Error message includes spec name and position values."""
        styles = [("cyan bold", np.array([0, 10, 20]))]
        with pytest.raises(ValueError) as excinfo:
            validate_style_positions(5, styles)
        error_msg = str(excinfo.value)
        assert "cyan bold" in error_msg
        assert "20" in error_msg or "max=" in error_msg

    def test_apply_inline_styles_validates_by_default(self):
        """apply_inline_styles validates positions by default."""
        styles = [("red", np.array([10]))]  # Out of bounds
        with pytest.raises(ValueError):
            apply_inline_styles("ACGT", styles)

    def test_apply_inline_styles_can_skip_validation(self):
        """apply_inline_styles can skip validation with validate=False."""
        styles = [("red", np.array([10]))]  # Out of bounds
        # Should not raise when validation disabled (positions just won't be styled)
        result = apply_inline_styles("ACGT", styles, validate=False)
        assert "ACGT" in result or result == "ACGT"


class TestPositionAdjustmentWithMarkers:
    """Test position adjustment edge cases with markers and regions."""

    def test_mutagenize_region_preserves_tags(self):
        """Positions correct when marker tags are preserved (default behavior)."""
        with pp.Party() as party:
            # 'AA' prefix (2 chars), marker content 'CCCC' (4 chars), 'GG' suffix
            bg = pp.from_seq("AA<test>CCCC</test>GG").named("bg")
            # Mutagenize first position of region, tags preserved by default
            mutated = bg.mutagenize(
                region="test", num_mutations=1, style="red", mode="sequential"
            ).named("mutated")

        df = mutated.generate_library(
            num_seqs=1, _include_inline_styles=True
        )
        seq = df["seq"].iloc[0]
        seq_style = df["_inline_styles"].iloc[0]

        # Sequence should preserve marker tags (default behavior)
        assert "<test>" in seq
        assert "</test>" in seq
        assert seq.startswith("AA")
        assert seq.endswith("GG")

        # Style positions should account for tag characters
        if seq_style:
            spec, positions = seq_style.style_list[0]
            # The mutation is within the region content
            # With tags preserved: 'AA<test>' = 8 chars, then content starts
            assert all(8 <= pos < 12 for pos in positions)

    def test_mutagenize_region_default_keeps_tags(self):
        """Default behavior preserves region tags."""
        with pp.Party() as party:
            # 'AA' prefix, marker with content 'CCCC', 'GG' suffix
            bg = pp.from_seq("AA<test>CCCC</test>GG").named("bg")
            mutated = bg.mutagenize(
                region="test", num_mutations=1, style="red", mode="sequential"
            ).named("mutated")

        df = mutated.generate_library(
            num_seqs=1, _include_inline_styles=True
        )
        seq = df["seq"].iloc[0]
        seq_style = df["_inline_styles"].iloc[0]

        # Sequence should preserve marker tags (default behavior)
        assert "<test>" in seq
        assert "</test>" in seq
        assert seq.startswith("AA")
        assert seq.endswith("GG")

        # Style positions should account for tag characters
        if seq_style:
            spec, positions = seq_style.style_list[0]
            # The mutation is within the region content
            # With tags preserved: 'AA<test>' = 8 chars, then content starts
            assert all(8 <= pos < 12 for pos in positions)

    def test_mutagenize_interval_region(self):
        """Positions correct for [start, stop] interval region."""
        with pp.Party() as party:
            bg = pp.from_seq("AACCCCGG").named("bg")
            # Use the mutagenize function directly (not method) for interval region
            mutated = mutagenize(
                bg, region=[2, 6], num_mutations=1, style="red", mode="sequential"
            ).named("mutated")

        df = mutated.generate_library(
            num_seqs=1, _include_inline_styles=True
        )
        seq_style = df["_inline_styles"].iloc[0]

        if seq_style:
            spec, positions = seq_style.style_list[0]
            # Positions should be in range [2, 6)
            assert all(2 <= pos < 6 for pos in positions)


class TestCaseTransformInlineStyles:
    """Test 'upper' and 'lower' case transformation in inline styles."""

    def test_lower_transforms_to_lowercase(self):
        """'lower' in style spec converts characters to lowercase."""
        styles = [("lower red", np.array([0, 2]))]
        result = apply_inline_styles("ACGT", styles)
        # Positions 0 and 2 should be lowercase: 'a' and 'g'
        # Strip ANSI codes to check character case
        clean = reset(result)
        assert clean == "aCgT"

    def test_upper_transforms_to_uppercase(self):
        """'upper' in style spec converts characters to uppercase."""
        styles = [("upper blue", np.array([1, 3]))]
        result = apply_inline_styles("acgt", styles)
        # Positions 1 and 3 should be uppercase: 'C' and 'T'
        clean = reset(result)
        assert clean == "aCgT"

    def test_lower_with_multiple_styles(self):
        """'lower' works with combined styles like 'lower cyan bold'."""
        styles = [("lower cyan bold", np.array([0, 1, 2, 3]))]
        result = apply_inline_styles("ACGT", styles)
        # All positions should be lowercase
        clean = reset(result)
        assert clean == "acgt"
        # Should have ANSI codes for styling
        assert "\033[" in result

    def test_upper_with_multiple_styles(self):
        """'upper' works with combined styles like 'upper red underline'."""
        styles = [("upper red underline", np.array([0, 1, 2, 3]))]
        result = apply_inline_styles("acgt", styles)
        # All positions should be uppercase
        clean = reset(result)
        assert clean == "ACGT"
        # Should have ANSI codes for styling
        assert "\033[" in result

    def test_only_specified_positions_transformed(self):
        """Only positions in the style array are case-transformed."""
        styles = [("lower", np.array([1]))]
        result = apply_inline_styles("ACGT", styles)
        clean = reset(result)
        # Only position 1 should be lowercase
        assert clean == "AcGT"

    def test_case_transform_only_no_color(self):
        """Case transform can be used without additional styling."""
        styles = [("lower", np.array([0, 1, 2, 3]))]
        result = apply_inline_styles("ACGT", styles)
        clean = reset(result)
        assert clean == "acgt"

    def test_later_transform_overrides_earlier(self):
        """Later case transforms override earlier ones at same position."""
        styles = [
            ("lower", np.array([0, 1])),  # First: make lowercase
            ("upper", np.array([0])),  # Second: make uppercase (overrides at pos 0)
        ]
        result = apply_inline_styles("acgt", styles)
        clean = reset(result)
        # Position 0: upper (later) wins, position 1: lower wins
        assert clean == "Acgt"

    def test_case_transform_preserves_non_alpha(self):
        """Case transforms preserve non-alphabetic characters."""
        styles = [("lower", np.array([0, 1, 2, 3]))]
        result = apply_inline_styles("A-G.", styles)
        clean = reset(result)
        # Non-alpha chars unchanged, alpha chars lowercased
        assert clean == "a-g."

    def test_empty_positions_no_transform(self):
        """Empty position array means no transforms."""
        styles = [("lower red", np.array([], dtype=np.int64))]
        result = apply_inline_styles("ACGT", styles)
        clean = reset(result)
        assert clean == "ACGT"


class TestBlinkInlineStyles:
    """Test 'blink' and 'blinking' style modifiers."""

    def test_blink_applies_ansi_code(self):
        """'blink' applies ANSI blink code (5) to characters."""
        styles = [("blink", np.array([0, 1]))]
        result = apply_inline_styles("ACGT", styles)
        # Should have ANSI codes
        assert "\033[" in result
        # ANSI code 5 is for blink
        assert "\033[5m" in result

    def test_blinking_alias_applies_ansi_code(self):
        """'blinking' is an alias for 'blink' and applies same ANSI code."""
        styles = [("blinking", np.array([0, 1]))]
        result = apply_inline_styles("ACGT", styles)
        assert "\033[5m" in result

    def test_blink_with_color(self):
        """'blink' combines with color styles like 'blink red'."""
        styles = [("blink red", np.array([0, 1, 2, 3]))]
        result = apply_inline_styles("ACGT", styles)
        # Should have ANSI codes for both blink and color
        assert "\033[" in result
        # Should contain all characters
        clean = reset(result)
        assert clean == "ACGT"

    def test_blink_with_multiple_modifiers(self):
        """'blink' works with combined styles like 'blink bold cyan'."""
        styles = [("blink bold cyan", np.array([0, 1, 2, 3]))]
        result = apply_inline_styles("ACGT", styles)
        assert "\033[" in result
        clean = reset(result)
        assert clean == "ACGT"

    def test_blink_only_specified_positions(self):
        """Only positions in the style array get blink effect."""
        styles = [("blink", np.array([1]))]
        result = apply_inline_styles("ACGT", styles)
        # Position 1 (C) should be styled, others should not
        # The result should have reset codes around position 1
        assert "\033[5m" in result
        clean = reset(result)
        assert clean == "ACGT"


class TestInvertInlineStyles:
    """Test 'invert' and 'reverse' style modifiers (reverse video)."""

    def test_invert_applies_ansi_code(self):
        """'invert' applies ANSI reverse video code (7) to characters."""
        styles = [("invert", np.array([0, 1]))]
        result = apply_inline_styles("ACGT", styles)
        # Should have ANSI codes
        assert "\033[" in result
        # ANSI code 7 is for invert/reverse video
        assert "\033[7m" in result

    def test_reverse_alias_applies_ansi_code(self):
        """'reverse' is an alias for 'invert' and applies same ANSI code."""
        styles = [("reverse", np.array([0, 1]))]
        result = apply_inline_styles("ACGT", styles)
        assert "\033[7m" in result

    def test_invert_with_color(self):
        """'invert' combines with color styles like 'invert red'."""
        styles = [("invert red", np.array([0, 1, 2, 3]))]
        result = apply_inline_styles("ACGT", styles)
        # Should have ANSI codes for both invert and color
        assert "\033[" in result
        # Should contain all characters
        clean = reset(result)
        assert clean == "ACGT"

    def test_invert_with_multiple_modifiers(self):
        """'invert' works with combined styles like 'invert bold cyan'."""
        styles = [("invert bold cyan", np.array([0, 1, 2, 3]))]
        result = apply_inline_styles("ACGT", styles)
        assert "\033[" in result
        clean = reset(result)
        assert clean == "ACGT"

    def test_invert_only_specified_positions(self):
        """Only positions in the style array get invert effect."""
        styles = [("invert", np.array([1]))]
        result = apply_inline_styles("ACGT", styles)
        # Position 1 (C) should be styled, others should not
        assert "\033[7m" in result
        clean = reset(result)
        assert clean == "ACGT"


class TestBackgroundInlineStyles:
    """Test background color styles using 'on_xxx' syntax."""

    def test_on_red_applies_ansi_code(self):
        """'on_red' applies ANSI background red code (101)."""
        styles = [("on_red", np.array([0, 1]))]
        result = apply_inline_styles("ACGT", styles)
        assert "\033[101m" in result

    def test_on_blue_applies_ansi_code(self):
        """'on_blue' applies ANSI background blue code (104)."""
        styles = [("on_blue", np.array([0, 1]))]
        result = apply_inline_styles("ACGT", styles)
        assert "\033[104m" in result

    def test_on_white_applies_ansi_code(self):
        """'on_white' applies ANSI background white code (107)."""
        styles = [("on_white", np.array([0, 1]))]
        result = apply_inline_styles("ACGT", styles)
        assert "\033[107m" in result

    def test_on_black_applies_ansi_code(self):
        """'on_black' applies ANSI background black code (40)."""
        styles = [("on_black", np.array([0, 1]))]
        result = apply_inline_styles("ACGT", styles)
        assert "\033[40m" in result

    def test_on_css_color(self):
        """'on_coral' applies CSS color as background."""
        styles = [("on_coral", np.array([0, 1, 2, 3]))]
        result = apply_inline_styles("ACGT", styles)
        # Should have true-color background code (48;2;R;G;B)
        assert "\033[48;2;" in result
        clean = reset(result)
        assert clean == "ACGT"

    def test_on_hex_color(self):
        """'on_#ff7f50' applies hex color as background."""
        styles = [("on_#ff7f50", np.array([0, 1, 2, 3]))]
        result = apply_inline_styles("ACGT", styles)
        # Coral hex is #ff7f50 -> 48;2;255;127;80
        assert "\033[48;2;255;127;80m" in result
        clean = reset(result)
        assert clean == "ACGT"

    def test_foreground_and_background_combined(self):
        """'red on_blue' applies both foreground and background."""
        styles = [("red on_blue", np.array([0, 1, 2, 3]))]
        result = apply_inline_styles("ACGT", styles)
        # Should have both foreground (91) and background (104) codes
        assert "\033[" in result
        clean = reset(result)
        assert clean == "ACGT"

    def test_white_foreground(self):
        """'white' applies ANSI foreground white code (97)."""
        styles = [("white", np.array([0, 1]))]
        result = apply_inline_styles("ACGT", styles)
        assert "\033[97m" in result

    def test_black_foreground(self):
        """'black' applies ANSI foreground black code (30)."""
        styles = [("black", np.array([0, 1]))]
        result = apply_inline_styles("ACGT", styles)
        assert "\033[30m" in result

    def test_white_on_blue(self):
        """'white on_blue' applies white text on blue background."""
        styles = [("white on_blue", np.array([0, 1, 2, 3]))]
        result = apply_inline_styles("ACGT", styles)
        assert "\033[" in result
        clean = reset(result)
        assert clean == "ACGT"

    def test_background_conflict_resolution(self):
        """Later background color wins when multiple are applied."""
        styles = [
            ("on_red", np.array([0, 1])),
            ("on_blue", np.array([0])),  # Later, should win at position 0
        ]
        result = apply_inline_styles("ACGT", styles)
        # Position 0 should have blue (104), position 1 should have red (101)
        assert "\033[" in result
        clean = reset(result)
        assert clean == "ACGT"

    def test_background_with_modifiers(self):
        """Background colors work with modifiers like bold."""
        styles = [("bold on_yellow", np.array([0, 1, 2, 3]))]
        result = apply_inline_styles("ACGT", styles)
        assert "\033[" in result
        clean = reset(result)
        assert clean == "ACGT"


class TestDeletionScanStylePropagation:
    """Test inline styles propagate through deletion_scan operations."""

    def test_styles_propagate_through_deletion_scan(self):
        """Styles from parent pool propagate through deletion_scan."""
        with pp.Party() as party:
            # Create a styled sequence - style the flanking regions
            bg = (
                pp.from_seq("AAAA<test>CCCCGGGG</test>TTTT")
                .stylize(region=[0, 4], style="red")
                .named("bg")
            )
            # Apply deletion scan
            deleted = bg.deletion_scan(region="test", deletion_length=2, mode="sequential").named(
                "deleted"
            )

        df = deleted.generate_library(
            num_seqs=1, _include_inline_styles=True
        )
        seq_style = df["_inline_styles"].iloc[0]

        # Styles should have propagated through
        # The 'red' style should be present (positions may have shifted)
        assert len(seq_style.style_list) > 0
        assert any(spec == "red" for spec, _ in seq_style.style_list)

    def test_deletion_scan_styles_adjust_positions(self):
        """Style positions are adjusted correctly through deletion_scan chain."""
        with pp.Party() as party:
            # Create background with styled prefix (not the deletion region itself)
            bg = (
                pp.from_seq("AAAA<test>CCCCGGGG</test>TTTT")
                .stylize(region=[0, 4], style="blue")
                .named("bg")
            )
            deleted = bg.deletion_scan(region="test", deletion_length=2, mode="sequential").named(
                "deleted"
            )

        df = deleted.generate_library(
            num_seqs=1, _include_inline_styles=True
        )
        seq = df["seq"].iloc[0]
        seq_style = df["_inline_styles"].iloc[0]

        # Verify styles exist and have valid positions within seq bounds
        assert len(seq_style.style_list) > 0
        for spec, positions in seq_style.style_list:
            assert all(0 <= pos < len(seq) for pos in positions), (
                f"Invalid positions for {spec}: {positions}, seq_len={len(seq)}"
            )

    def test_gap_chars_do_not_inherit_styles(self):
        """Gap characters should NOT inherit styles from deleted characters."""
        with pp.Party() as party:
            # Style the ENTIRE region (including what will be deleted)
            bg = pp.from_seq("AAA<test>CCCC</test>TTT").stylize("test", style="red").named("bg")
            deleted = bg.deletion_scan(region="test", deletion_length=2, mode="sequential").named(
                "deleted"
            )

        df = deleted.generate_library(
            num_seqs=1, _include_inline_styles=True
        )
        seq = df["seq"].iloc[0]
        seq_style = df["_inline_styles"].iloc[0]

        # Find gap positions
        gap_positions = set(i for i, c in enumerate(seq) if c == "-")
        assert len(gap_positions) > 0, "Should have gap characters"

        # Verify no styles apply to gap positions
        for spec, positions in seq_style.style_list:
            styled_gaps = [p for p in positions if p in gap_positions]
            assert len(styled_gaps) == 0, (
                f"Gap chars should not have {spec} style, but found at positions {styled_gaps}"
            )

    def test_style_parameter(self):
        """style parameter applies style to gap characters."""
        with pp.Party() as party:
            bg = pp.from_seq("AAA<test>CCCC</test>TTT").named("bg")
            deleted = bg.deletion_scan(
                region="test", deletion_length=2, mode="sequential", style="cyan"
            ).named("deleted")

        df = deleted.generate_library(
            num_seqs=1, _include_inline_styles=True
        )
        seq = df["seq"].iloc[0]
        seq_style = df["_inline_styles"].iloc[0]

        # Find gap positions
        gap_positions = set(i for i, c in enumerate(seq) if c == "-")
        assert len(gap_positions) > 0, "Should have gap characters"

        # Verify cyan style applies to gap positions
        cyan_positions = set()
        for spec, positions in seq_style.style_list:
            if spec == "cyan":
                cyan_positions.update(positions)

        assert gap_positions == cyan_positions, (
            f"Gap positions {gap_positions} should match cyan positions {cyan_positions}"
        )

    def test_style_with_region_styles(self):
        """style works correctly with pre-existing region styles."""
        with pp.Party() as party:
            # Style the region, then delete with style
            bg = pp.from_seq("AAA<test>CCCC</test>TTT").stylize("test", style="red").named("bg")
            deleted = bg.deletion_scan(
                region="test", deletion_length=2, mode="sequential", style="cyan"
            ).named("deleted")

        df = deleted.generate_library(
            num_seqs=1, _include_inline_styles=True
        )
        seq = df["seq"].iloc[0]
        seq_style = df["_inline_styles"].iloc[0]

        # Find gap positions
        gap_positions = set(i for i, c in enumerate(seq) if c == "-")

        # Verify red style does NOT apply to gaps
        # and cyan style DOES apply to gaps
        for spec, positions in seq_style.style_list:
            if spec == "red":
                styled_gaps = [p for p in positions if p in gap_positions]
                assert len(styled_gaps) == 0, "Red style should not apply to gaps"
            elif spec == "cyan":
                styled_gaps = [p for p in positions if p in gap_positions]
                assert len(styled_gaps) == len(gap_positions), "Cyan style should apply to all gaps"

    def test_style_ignored_when_no_marker(self):
        """style is ignored when deletion_marker=None."""
        with pp.Party() as party:
            bg = pp.from_seq("AAA<test>CCCC</test>TTT").named("bg")
            deleted = bg.deletion_scan(
                region="test",
                deletion_length=2,
                deletion_marker=None,
                style="cyan",
                mode="sequential",
            ).named("deleted")

        df = deleted.generate_library(
            num_seqs=1, _include_inline_styles=True
        )
        seq = df["seq"].iloc[0]
        seq_style = df["_inline_styles"].iloc[0]

        # No gap characters (deletion_marker=None removes them), no cyan style
        assert "-" not in seq
        assert not any(spec == "cyan" for spec, _ in seq_style.style_list)


class TestInsertionScanStylePropagation:
    """Test inline styles propagate through insertion_scan operations."""

    def test_styles_propagate_through_insertion_scan(self):
        """Styles from parent pool propagate through insertion_scan."""
        with pp.Party() as party:
            # Create a styled background - style the flanking regions
            bg = (
                pp.from_seq("AAAA<test>CCCCGGGG</test>TTTT")
                .stylize(region=[0, 4], style="green")
                .named("bg")
            )
            # Create insert pool (same length as region for replace)
            inserts = pp.from_seqs(["XXXXXXXX"], mode="sequential").named("inserts")
            # Apply insertion scan with replace
            inserted = bg.insertion_scan(
                region="test", ins_pool=inserts, positions=[0], replace=True, mode="sequential"
            ).named("inserted")

        df = inserted.generate_library(
            num_seqs=1, _include_inline_styles=True
        )
        seq_style = df["_inline_styles"].iloc[0]

        # Styles should have propagated (green style from bg prefix)
        assert len(seq_style.style_list) > 0
        assert any(spec == "green" for spec, _ in seq_style.style_list)

    def test_insertion_scan_preserves_non_region_styles(self):
        """Styles outside the insertion region are preserved."""
        with pp.Party() as party:
            # Style the prefix, then insert within a region
            bg = (
                pp.from_seq("AAAA<ins>XXXX</ins>TTTT")
                .stylize(region=[0, 4], style="cyan")
                .named("bg")
            )
            # Insert replaces with same-length content
            inserts = pp.from_seqs(["GGGG"], mode="sequential").named("inserts")
            inserted = bg.insertion_scan(
                region="ins", ins_pool=inserts, positions=[0], replace=True, mode="sequential"
            ).named("inserted")

        df = inserted.generate_library(
            num_seqs=1, _include_inline_styles=True
        )
        seq = df["seq"].iloc[0]
        seq_style = df["_inline_styles"].iloc[0]

        # Should have styles for the prefix (AAAA)
        assert len(seq_style.style_list) > 0
        for spec, positions in seq_style.style_list:
            assert all(0 <= pos < len(seq) for pos in positions), (
                f"Invalid positions for {spec}: {positions}, seq_len={len(seq)}"
            )

    def test_insert_pool_styles_propagate_through_insertion_scan(self):
        """Styles from the insert pool (content_pool) propagate through insertion_scan."""
        with pp.Party() as party:
            # Create unstyled background with marker
            bg = pp.from_seq("AAAA<ins>XXXX</ins>TTTT").named("bg")
            # Create styled insert pool - these styles should propagate
            inserts = (
                pp.from_seqs(["GGGG"], mode="sequential").stylize(style="magenta").named("inserts")
            )
            inserted = bg.insertion_scan(
                region="ins", ins_pool=inserts, positions=[0], replace=True, mode="sequential"
            ).named("inserted")

        df = inserted.generate_library(
            num_seqs=1, _include_inline_styles=True
        )
        seq = df["seq"].iloc[0]
        seq_style = df["_inline_styles"].iloc[0]

        # Insert pool styles should have propagated
        assert len(seq_style.style_list) > 0
        assert any(spec == "magenta" for spec, _ in seq_style.style_list)
        # Positions should be valid and point to where insert was placed
        for spec, positions in seq_style.style_list:
            assert all(0 <= pos < len(seq) for pos in positions), (
                f"Invalid positions for {spec}: {positions}, seq_len={len(seq)}"
            )

    def test_insert_pool_styles_with_multiple_sites(self):
        """Styles propagate correctly when insert pool has multiple styled sequences."""
        with pp.Party() as party:
            bg = pp.from_seq("AA<ins>XX</ins>TT").named("bg")
            # Two different inserts, both styled
            inserts = (
                pp.from_seqs(["GG", "CC"], mode="sequential").stylize(style="cyan").named("inserts")
            )
            inserted = bg.insertion_scan(
                region="ins", ins_pool=inserts, positions=[0], replace=True, mode="sequential"
            ).named("inserted")

        df = inserted.generate_library(
            num_seqs=2, _include_inline_styles=True
        )

        # Both sequences should have cyan styles from the insert pool
        for i in range(2):
            seq_style = df["_inline_styles"].iloc[i]
            seq = df["seq"].iloc[i]
            assert len(seq_style.style_list) > 0, f"Row {i} should have styles"
            assert any(spec == "cyan" for spec, _ in seq_style.style_list), (
                f"Row {i} should have cyan style"
            )
            for spec, positions in seq_style.style_list:
                assert all(0 <= pos < len(seq) for pos in positions), (
                    f"Invalid positions in row {i}"
                )


class TestInsertionScanStyleInsertion:
    """Test insertion_scan with style parameter."""

    def test_style_none_by_default(self):
        """Default style is None."""
        with pp.Party() as party:
            bg = pp.from_seq("AAAAAAAAAA")
            ins = pp.from_seq("TTT")
            pool = pp.insertion_scan(bg, ins, positions=[0], mode="sequential")
            # The final operation is replace_marker_content
            assert pool.operation._style is None

    def test_style_stored(self):
        """style parameter is stored on operation."""
        with pp.Party() as party:
            bg = pp.from_seq("AAAAAAAAAA")
            ins = pp.from_seq("TTT")
            pool = pp.insertion_scan(bg, ins, positions=[0], mode="sequential", style="red bold")
            assert pool.operation._style == "red bold"

    def test_style_in_copy_params(self):
        """style is included in _get_copy_params."""
        with pp.Party() as party:
            bg = pp.from_seq("AAAAAAAAAA")
            ins = pp.from_seq("TTT")
            pool = pp.insertion_scan(bg, ins, positions=[0], mode="sequential", style="blue")
            params = pool.operation._get_copy_params()
        assert params["_style"] == "blue"

    def test_style_applies_to_inserted_positions(self):
        """style applies style to all inserted positions."""
        with pp.Party() as party:
            bg = pp.from_seq("AAAAAAAAAA")
            ins = pp.from_seq("TTT")
            pool = pp.insertion_scan(bg, ins, positions=[5], mode="sequential", style="red").named(
                "result"
            )

        df = pool.generate_library(
            num_seqs=1, _include_inline_styles=True
        )
        seq_style = df["_inline_styles"].iloc[0]
        seq = df["seq"].iloc[0]

        # Should have style for 3 positions (length of insert)
        assert len(seq_style.style_list) >= 1

        # Find the red style entry
        red_styles = [(spec, pos) for spec, pos in seq_style.style_list if spec == "red"]
        assert len(red_styles) == 1
        spec, positions = red_styles[0]
        assert len(positions) == 3  # TTT is 3 chars
        # Insert at position 5 should style positions 5, 6, 7
        assert list(positions) == [5, 6, 7]

    def test_style_at_start(self):
        """style works when inserting at position 0."""
        with pp.Party() as party:
            bg = pp.from_seq("AAAAAAAAAA")
            ins = pp.from_seq("GGG")
            pool = pp.insertion_scan(bg, ins, positions=[0], mode="sequential", style="cyan").named(
                "result"
            )

        df = pool.generate_library(
            num_seqs=1, _include_inline_styles=True
        )
        seq_style = df["_inline_styles"].iloc[0]

        # Should style positions 0, 1, 2
        cyan_styles = [(spec, pos) for spec, pos in seq_style.style_list if spec == "cyan"]
        assert len(cyan_styles) == 1
        _, positions = cyan_styles[0]
        assert list(positions) == [0, 1, 2]

    def test_style_at_end(self):
        """style works when inserting at the end."""
        with pp.Party() as party:
            bg = pp.from_seq("AAAAAAAAAA")  # 10 chars
            ins = pp.from_seq("GGG")
            pool = pp.insertion_scan(
                bg, ins, positions=[10], mode="sequential", style="magenta"
            ).named("result")

        df = pool.generate_library(
            num_seqs=1, _include_inline_styles=True
        )
        seq_style = df["_inline_styles"].iloc[0]

        # Insert at end: positions 10, 11, 12
        magenta_styles = [(spec, pos) for spec, pos in seq_style.style_list if spec == "magenta"]
        assert len(magenta_styles) == 1
        _, positions = magenta_styles[0]
        assert list(positions) == [10, 11, 12]

    def test_style_with_replacement_scan(self):
        """style works with replacement_scan."""
        with pp.Party() as party:
            bg = pp.from_seq("AAAAAAAAAA")  # 10 chars
            ins = pp.from_seq("GGG")
            pool = pp.replacement_scan(
                bg, ins, positions=[5], mode="sequential", style="yellow"
            ).named("result")

        df = pool.generate_library(
            num_seqs=1, _include_inline_styles=True
        )
        seq_style = df["_inline_styles"].iloc[0]

        # Should style the 3 replacement positions starting at 5
        yellow_styles = [(spec, pos) for spec, pos in seq_style.style_list if spec == "yellow"]
        assert len(yellow_styles) == 1
        _, positions = yellow_styles[0]
        assert list(positions) == [5, 6, 7]

        seq = df["seq"].iloc[0]

        # Sequence should be AAAAAGGGAA (GGG replaced at position 5)
        assert "GGG" in seq

        # Style should cover the GGG (3 positions)
        yellow_styles = [(spec, pos) for spec, pos in seq_style.style_list if spec == "yellow"]
        assert len(yellow_styles) == 1
        _, positions = yellow_styles[0]
        assert len(positions) == 3  # GGG positions

    def test_style_without_style_returns_empty(self):
        """insertion_scan without style has no insertion styles."""
        with pp.Party() as party:
            bg = pp.from_seq("AAAAAAAAAA")
            ins = pp.from_seq("TTT")
            pool = pp.insertion_scan(bg, ins, positions=[5], mode="sequential").named("result")

        df = pool.generate_library(
            num_seqs=1, _include_inline_styles=True
        )
        seq_style = df["_inline_styles"].iloc[0]

        # Should be empty (no style specified)
        assert not seq_style

    def test_style_combines_with_insert_pool_styles(self):
        """style combines with styles already on insert pool."""
        with pp.Party() as party:
            bg = pp.from_seq("AAAAAAAAAA")
            # Insert pool already has styling
            ins = pp.from_seq("TTT").stylize(style="blue")
            pool = pp.insertion_scan(bg, ins, positions=[5], mode="sequential", style="red").named(
                "result"
            )

        df = pool.generate_library(
            num_seqs=1, _include_inline_styles=True
        )
        seq_style = df["_inline_styles"].iloc[0]

        # Should have both blue (from insert pool) and red (from style)
        style_specs = [spec for spec, _ in seq_style.style_list]
        assert "blue" in style_specs
        assert "red" in style_specs


class TestInsertKmersStylePropagation:
    """Test inline styles propagate through insert_kmers operations."""

    def test_styles_propagate_through_insert_kmers(self):
        """Styles from parent pool propagate through insert_kmers."""
        with pp.Party() as party:
            # Create a styled background with a zero-length marker
            # Style the prefix region specifically
            bg = pp.from_seq("AAAA<bc/>TTTT").stylize(region=[0, 4], style="red").named("bg")
            # Insert kmers at the marker
            result = bg.insert_kmers(region="bc", length=3, mode="sequential").named("result")

        df = result.generate_library(
            num_seqs=1, _include_inline_styles=True
        )
        seq = df["seq"].iloc[0]
        seq_style = df["_inline_styles"].iloc[0]

        # Styles should propagate (red style for AAAA prefix)
        assert len(seq_style.style_list) > 0
        for spec, positions in seq_style.style_list:
            assert all(0 <= pos < len(seq) for pos in positions), (
                f"Invalid positions for {spec}: {positions}, seq_len={len(seq)}"
            )

    def test_insert_kmers_preserves_surrounding_styles(self):
        """Styles around the kmer insertion point are preserved."""
        with pp.Party() as party:
            # Style the prefix
            bg = pp.from_seq("AAAA<bc/>TTTT").stylize(region=[0, 4], style="blue").named("bg")
            result = bg.insert_kmers(region="bc", length=3, mode="sequential").named("result")

        df = result.generate_library(
            num_seqs=1, _include_inline_styles=True
        )
        seq = df["seq"].iloc[0]
        seq_style = df["_inline_styles"].iloc[0]

        # Blue style for AAAA should be preserved
        assert len(seq_style.style_list) > 0
        # Check that the 'blue' style is present
        assert any(spec == "blue" for spec, _ in seq_style.style_list)

    def test_insert_kmers_suffix_styles_shifted_correctly(self):
        """Suffix styles are correctly shifted when self-closing marker is replaced.

        This tests the specific bug where <bc/> becoming <bc>content</bc> caused
        suffix styles to incorrectly land inside the </bc> closing tag because
        the length delta wasn't accounting for marker tag format changes.
        """
        with pp.Party() as party:

            # Style the ENTIRE sequence (including suffix after marker)
            bg = pp.from_seq("AAAA<bc/>TTTT").stylize(style="red").named("bg")
            result = bg.insert_kmers(region="bc", length=3, mode="sequential").named("result")

        df = result.generate_library(
            num_seqs=1, _include_inline_styles=True
        )
        seq = df["seq"].iloc[0]
        seq_style = df["_inline_styles"].iloc[0]

        # Sequence should be: AAAA<bc>XXX</bc>TTTT (where XXX is 3-char kmer)
        # Structure: AAAA (4) + <bc> (4) + XXX (3) + </bc> (5) + TTTT (4) = 20 chars
        assert "<bc>" in seq
        assert "</bc>" in seq
        assert len(seq) == 20

        # Find the positions of the suffix 'TTTT' (should be at positions 16-19)
        suffix_start = seq.index("</bc>") + 5  # Position after </bc>
        assert seq[suffix_start : suffix_start + 4] == "TTTT"

        # Red style should apply to AAAA (0-3) and TTTT (16-19)
        # NOT to positions inside the <bc>...</bc> tags
        red_styles = [(spec, pos) for spec, pos in seq_style.style_list if spec == "red"]
        assert len(red_styles) > 0

        # Collect all red-styled positions
        red_positions = set()
        for _, positions in red_styles:
            red_positions.update(positions.tolist())

        # Suffix positions (TTTT at 16-19) should be styled
        for i in range(suffix_start, suffix_start + 4):
            assert i in red_positions, f"Position {i} (in suffix TTTT) should have red style"

        # Positions inside marker tags should NOT be styled
        # <bc> is at positions 4-7, </bc> is at positions 11-15
        for i in range(4, 16):  # All positions in <bc>XXX</bc>
            assert i not in red_positions, f"Position {i} (inside marker) should not have red style"


class TestStackRepeatStylePropagation:
    """Test that stack and repeat operations correctly pass through styles."""

    def test_stack_passes_styles_from_active_parent(self):
        """stack() passes through styles from the active parent pool."""
        with pp.Party() as party:
            pool1 = pp.from_seq("AAAA").stylize(style="red").named("pool1")
            pool2 = pp.from_seq("GGGG").stylize(style="blue").named("pool2")
            stacked = pp.stack([pool1, pool2]).named("stacked")

        df = stacked.generate_library(
            num_seqs=2, _include_inline_styles=True
        )

        # First row from pool1 should have red style
        seq_style0 = df["_inline_styles"].iloc[0]
        assert len(seq_style0.style_list) > 0
        assert seq_style0.style_list[0][0] == "red"

        # Second row from pool2 should have blue style
        seq_style1 = df["_inline_styles"].iloc[1]
        assert len(seq_style1.style_list) > 0
        assert seq_style1.style_list[0][0] == "blue"

    def test_repeat_passes_styles_unchanged(self):
        """repeat() passes through parent styles unchanged."""
        with pp.Party() as party:
            pool = pp.from_seq("ACGT").stylize(style="green").named("pool")
            repeated = pool.repeat(3).named("repeated")

        df = repeated.generate_library(
            num_seqs=3, _include_inline_styles=True
        )

        # All rows should have the green style
        for i in range(3):
            seq_style = df["_inline_styles"].iloc[i]
            assert len(seq_style.style_list) > 0
            assert seq_style.style_list[0][0] == "green"


class TestCompositeOperationsStyleChain:
    """Test styles flow through complex chains of operations."""

    def test_stylize_through_mutagenize_and_stack(self):
        """Styles propagate through mutagenize and stack operations."""
        with pp.Party() as party:
            bg = pp.from_seq("AA<cre>CCCCGGGG</cre>TT").stylize("cre", style="red").named("bg")
            mutated = bg.mutagenize(
                region="cre", num_mutations=1, style="yellow", mode="sequential"
            ).named("mutated")
            stacked = pp.stack([mutated]).named("stacked")

        df = stacked.generate_library(
            num_seqs=1, _include_inline_styles=True
        )
        seq_style = df["_inline_styles"].iloc[0]

        # Should have both red (from stylize) and yellow (from mutagenize changes)
        style_specs = [spec for spec, _ in seq_style.style_list]
        assert "red" in style_specs
        assert "yellow" in style_specs

    def test_styles_through_full_chain(self):
        """Styles flow through a full chain: stylize -> mutagenize -> repeat -> stack."""
        with pp.Party() as party:
            bg = pp.from_seq("ACGTACGT").stylize(style="cyan").named("bg")
            mutated = bg.mutagenize(num_mutations=1, style="bold", mode="sequential").named(
                "mutated"
            )
            repeated = mutated.repeat(2).named("repeated")
            pool2 = pp.from_seq("TTTTTTTT").stylize(style="red").named("pool2")
            stacked = pp.stack([repeated, pool2]).named("stacked")

        df = stacked.generate_library(
            num_seqs=3, _include_inline_styles=True
        )

        # All rows should have styles
        for i in range(3):
            seq_style = df["_inline_styles"].iloc[i]
            assert len(seq_style.style_list) > 0


class TestFromSeqStyle:
    """Test style parameter on from_seq."""

    def test_from_seq_style_applies_to_all_positions(self):
        """style parameter applies to all sequence positions."""
        with pp.Party() as party:
            pool = pp.from_seq("ACGT", style="red").named("result")

        df = pool.generate_library(
            num_seqs=1, _include_inline_styles=True
        )
        seq_style = df["_inline_styles"].iloc[0]

        assert len(seq_style.style_list) == 1
        spec, positions = seq_style.style_list[0]
        assert spec == "red"
        assert list(positions) == [0, 1, 2, 3]

    def test_from_seq_no_style_by_default(self):
        """from_seq without style has empty styles."""
        with pp.Party() as party:
            pool = pp.from_seq("ACGT").named("result")

        df = pool.generate_library(
            num_seqs=1, _include_inline_styles=True
        )
        seq_style = df["_inline_styles"].iloc[0]
        assert not seq_style


class TestFromSeqsStyle:
    """Test style parameter on from_seqs."""

    def test_from_seqs_style_applies_to_all_positions(self):
        """style parameter applies to all sequence positions."""
        with pp.Party() as party:
            pool = pp.from_seqs(["ACGT", "TTTT"], mode="sequential", style="blue").named("result")

        df = pool.generate_library(
            num_seqs=2, _include_inline_styles=True
        )

        for i in range(2):
            seq_style = df["_inline_styles"].iloc[i]
            assert len(seq_style.style_list) == 1
            spec, positions = seq_style.style_list[0]
            assert spec == "blue"
            assert list(positions) == [0, 1, 2, 3]

    def test_from_seqs_style_in_copy_params(self):
        """style is included in _get_copy_params."""
        with pp.Party() as party:
            pool = pp.from_seqs(["ACGT"], style="cyan")
            params = pool.operation._get_copy_params()
        assert params["style"] == "cyan"


class TestGetKmersStyle:
    """Test style parameter on get_kmers."""

    def test_get_kmers_style_applies_to_kmer(self):
        """style parameter applies to kmer positions."""
        with pp.Party() as party:
            pool = pp.get_kmers(length=3, mode="sequential", style="green").named("result")

        df = pool.generate_library(
            num_seqs=1, _include_inline_styles=True
        )
        seq_style = df["_inline_styles"].iloc[0]

        assert len(seq_style.style_list) == 1
        spec, positions = seq_style.style_list[0]
        assert spec == "green"
        assert list(positions) == [0, 1, 2]

    def test_get_kmers_style_in_copy_params(self):
        """style is included in _get_copy_params."""
        with pp.Party() as party:
            pool = pp.get_kmers(length=3, style="magenta")
            params = pool.operation._get_copy_params()
        assert params["style"] == "magenta"


class TestInsertKmersStyleParams:
    """Test style parameter on insert_kmers and chained stylize."""

    def test_style_kmers_applies_to_kmer(self):
        """style parameter applies style to inserted kmer."""
        with pp.Party() as party:
            bg = pp.from_seq("AA<kmer/>TT")
            pool = bg.insert_kmers(region="kmer", length=2, mode="sequential", style="red").named(
                "result"
            )

        df = pool.generate_library(
            num_seqs=1, _include_inline_styles=True
        )
        seq_style = df["_inline_styles"].iloc[0]

        red_styles = [(spec, pos) for spec, pos in seq_style.style_list if spec == "red"]
        assert len(red_styles) == 1
        _, positions = red_styles[0]
        assert len(positions) == 2  # kmer length

    def test_style_background_applies_to_non_kmer(self):
        """Chained stylize applies to non-kmer positions."""
        with pp.Party() as party:
            bg = pp.from_seq("AA<kmer/>TT")
            pool = (
                bg.stylize(style="blue")
                .insert_kmers(region="kmer", length=2, mode="sequential")
                .named("result")
            )

        df = pool.generate_library(
            num_seqs=1, _include_inline_styles=True
        )
        seq_style = df["_inline_styles"].iloc[0]

        # Collect all blue-styled positions (may be split across entries)
        blue_positions = []
        for spec, pos in seq_style.style_list:
            if spec == "blue":
                blue_positions.extend(pos.tolist())

        # Background: AA (positions 0,1) + TT (positions 4,5) = 4 positions
        # Kmer is at positions 2,3
        assert len(blue_positions) == 4
        assert 2 not in blue_positions  # kmer position
        assert 3 not in blue_positions  # kmer position

    def test_style_kmers_and_style_background_combined(self):
        """style parameter and chained stylize can be combined."""
        with pp.Party() as party:
            bg = pp.from_seq("AA<kmer/>TT")
            pool = (
                bg.stylize(style="blue")
                .insert_kmers(region="kmer", length=2, mode="sequential", style="red")
                .named("result")
            )

        df = pool.generate_library(
            num_seqs=1, _include_inline_styles=True
        )
        seq_style = df["_inline_styles"].iloc[0]

        style_specs = [spec for spec, _ in seq_style.style_list]
        assert "red" in style_specs
        assert "blue" in style_specs


class TestSeqStyle:
    """Test SeqStyle class operations."""

    # Construction
    def test_empty_creates_spacer(self):
        """SeqStyle.empty(n) creates SeqStyle with no styles and length n."""
        s = SeqStyle.empty(10)
        assert s.length == 10
        assert len(s) == 10
        assert s.style_list == []
        assert not s  # bool(s) should be False

    def test_from_style_list(self):
        """SeqStyle.from_style_list wraps existing StyleList."""
        style_list = [("red", np.array([0, 1, 2])), ("blue", np.array([5, 6]))]
        s = SeqStyle.from_style_list(style_list, length=10)
        assert s.length == 10
        assert len(s.style_list) == 2
        assert s  # bool(s) should be True

    def test_add_style_returns_new(self):
        """add_style() returns new SeqStyle with appended style."""
        s = SeqStyle.empty(10)
        s2 = s.add_style("red", np.array([0, 1]))
        # Original unchanged
        assert s.style_list == []
        # New has added style
        assert len(s2.style_list) == 1
        assert s2.style_list[0][0] == "red"
        assert list(s2.style_list[0][1]) == [0, 1]

    # Slicing
    def test_slice_extracts_region(self):
        """seq_style[10:50] extracts and 0-indexes positions."""
        style_list = [("red", np.array([15, 20, 25]))]
        s = SeqStyle.from_style_list(style_list, length=100)
        region = s[10:50]

        assert region.length == 40
        assert len(region.style_list) == 1
        # Positions should be shifted to 0-indexed: 15->5, 20->10, 25->15
        assert list(region.style_list[0][1]) == [5, 10, 15]

    def test_slice_start_only(self):
        """seq_style[50:] extracts from position to end."""
        style_list = [("blue", np.array([60, 80]))]
        s = SeqStyle.from_style_list(style_list, length=100)
        region = s[50:]

        assert region.length == 50
        # Positions 60->10, 80->30
        assert list(region.style_list[0][1]) == [10, 30]

    def test_slice_end_only(self):
        """seq_style[:50] extracts first n positions."""
        style_list = [("green", np.array([10, 20, 30]))]
        s = SeqStyle.from_style_list(style_list, length=100)
        region = s[:50]

        assert region.length == 50
        # Positions unchanged: 10, 20, 30
        assert list(region.style_list[0][1]) == [10, 20, 30]

    def test_slice_filters_positions(self):
        """Positions outside slice range are excluded."""
        style_list = [("red", np.array([5, 15, 25, 35]))]
        s = SeqStyle.from_style_list(style_list, length=100)
        region = s[10:30]

        # Only positions 15 and 25 are in [10, 30)
        # They become 5 and 15 in the new 0-indexed region
        assert list(region.style_list[0][1]) == [5, 15]

    def test_slice_empty_when_no_positions(self):
        """Slice with no matching positions returns empty style_list."""
        style_list = [("red", np.array([5, 6, 7]))]
        s = SeqStyle.from_style_list(style_list, length=100)
        region = s[50:60]

        # No positions in [50, 60), so style_list is empty
        assert region.style_list == []
        assert not region  # bool(region) is False

    # Reversal
    def test_reversed_mirrors_positions(self):
        """reversed() mirrors positions within length."""
        style_list = [("red", np.array([0, 1, 8, 9]))]
        s = SeqStyle.from_style_list(style_list, length=10)
        rev = s.reversed()

        # Positions: 0->9, 1->8, 8->1, 9->0
        assert set(rev.style_list[0][1]) == {0, 1, 8, 9}

    def test_reversed_false_returns_self(self):
        """reversed(False) returns unchanged SeqStyle."""
        style_list = [("blue", np.array([0, 1, 2]))]
        s = SeqStyle.from_style_list(style_list, length=10)
        rev = s.reversed(do_reverse=False)

        # Should be identical
        assert rev is s

    def test_reversed_with_multiple_styles(self):
        """reversed() works with multiple styles."""
        style_list = [("red", np.array([0, 1])), ("blue", np.array([8, 9]))]
        s = SeqStyle.from_style_list(style_list, length=10)
        rev = s.reversed()

        # red: 0->9, 1->8
        # blue: 8->1, 9->0
        assert set(rev.style_list[0][1]) == {8, 9}
        assert set(rev.style_list[1][1]) == {0, 1}

    # Concatenation
    def test_join_concatenates(self):
        """SeqStyle.join shifts positions by cumulative lengths."""
        s1 = SeqStyle.from_style_list([("red", np.array([0, 1]))], length=5)
        s2 = SeqStyle.from_style_list([("blue", np.array([0, 1]))], length=5)
        s3 = SeqStyle.from_style_list([("green", np.array([0, 1]))], length=5)

        combined = SeqStyle.join([s1, s2, s3])

        assert combined.length == 15
        assert len(combined.style_list) == 3
        # s1: 0, 1 (unchanged)
        # s2: 0->5, 1->6
        # s3: 0->10, 1->11
        assert list(combined.style_list[0][1]) == [0, 1]
        assert list(combined.style_list[1][1]) == [5, 6]
        assert list(combined.style_list[2][1]) == [10, 11]

    def test_join_empty_list(self):
        """SeqStyle.join([]) returns empty SeqStyle."""
        combined = SeqStyle.join([])
        assert combined.length == 0
        assert combined.style_list == []

    def test_add_operator(self):
        """seq_style1 + seq_style2 concatenates."""
        s1 = SeqStyle.from_style_list([("red", np.array([0, 1]))], length=5)
        s2 = SeqStyle.from_style_list([("blue", np.array([0, 1]))], length=5)

        combined = s1 + s2

        assert combined.length == 10
        assert list(combined.style_list[0][1]) == [0, 1]
        assert list(combined.style_list[1][1]) == [5, 6]

    def test_join_with_empty_styles(self):
        """SeqStyle.join works with empty SeqStyles (spacers)."""
        s1 = SeqStyle.from_style_list([("red", np.array([0, 1]))], length=5)
        spacer = SeqStyle.empty(3)
        s2 = SeqStyle.from_style_list([("blue", np.array([0, 1]))], length=5)

        combined = SeqStyle.join([s1, spacer, s2])

        assert combined.length == 13
        # s1: 0, 1
        # spacer: no styles
        # s2: 0->8, 1->9
        assert list(combined.style_list[0][1]) == [0, 1]
        assert list(combined.style_list[1][1]) == [8, 9]

    # Split
    def test_split_single_breakpoint(self):
        """split([n]) returns two parts."""
        style_list = [("red", np.array([0, 1, 2])), ("blue", np.array([5, 6, 7]))]
        s = SeqStyle.from_style_list(style_list, length=10)
        parts = s.split([5])

        assert len(parts) == 2
        # Part 0: [0:5], Part 1: [5:10]
        assert parts[0].length == 5
        assert parts[1].length == 5

        # Part 0 should have red positions 0,1,2
        assert len(parts[0].style_list) == 1
        assert list(parts[0].style_list[0][1]) == [0, 1, 2]

        # Part 1 should have blue positions 0,1,2 (shifted from 5,6,7)
        assert len(parts[1].style_list) == 1
        assert list(parts[1].style_list[0][1]) == [0, 1, 2]

    def test_split_two_breakpoints(self):
        """split([a, b]) returns three parts."""
        style_list = [("red", np.array([2, 5, 8]))]
        s = SeqStyle.from_style_list(style_list, length=10)
        parts = s.split([3, 7])

        assert len(parts) == 3
        # Part 0: [0:3], Part 1: [3:7], Part 2: [7:10]
        assert parts[0].length == 3
        assert parts[1].length == 4
        assert parts[2].length == 3

        # Part 0: position 2
        assert list(parts[0].style_list[0][1]) == [2]
        # Part 1: position 5 -> 2
        assert list(parts[1].style_list[0][1]) == [2]
        # Part 2: position 8 -> 1
        assert list(parts[2].style_list[0][1]) == [1]

    def test_split_multiple_breakpoints(self):
        """split([a, b, c]) returns four parts."""
        s = SeqStyle.from_style_list([("red", np.array([1, 4, 7, 10]))], length=12)
        parts = s.split([3, 6, 9])

        assert len(parts) == 4
        assert [p.length for p in parts] == [3, 3, 3, 3]

    # Bool and repr
    def test_bool_true_when_has_styles(self):
        """bool(seq_style) is True when has styles."""
        s = SeqStyle.from_style_list([("red", np.array([0]))], length=10)
        assert bool(s) is True

    def test_bool_false_when_empty(self):
        """bool(SeqStyle.empty(n)) is False."""
        s = SeqStyle.empty(10)
        assert bool(s) is False

    def test_repr_single_style(self):
        """repr shows style count and length."""
        s = SeqStyle.from_style_list([("red", np.array([0]))], length=10)
        r = repr(s)
        assert "1 style" in r
        assert "length=10" in r

    def test_repr_multiple_styles(self):
        """repr pluralizes 'styles' correctly."""
        s = SeqStyle.from_style_list([("red", np.array([0])), ("blue", np.array([1]))], length=10)
        r = repr(s)
        assert "2 styles" in r
        assert "length=10" in r

    # Validation
    def test_validate_passes_on_valid(self):
        """validate() passes when positions are within bounds."""
        s = SeqStyle.from_style_list([("red", np.array([0, 5, 9]))], length=10)
        # Should not raise
        s.validate()

    def test_validate_raises_on_invalid(self):
        """validate() raises if positions out of bounds."""
        s = SeqStyle.from_style_list([("red", np.array([0, 10]))], length=10)
        with pytest.raises(ValueError):
            s.validate()

    # Application
    def test_apply_styles_sequence(self):
        """apply() returns ANSI-styled string."""
        s = SeqStyle.from_style_list([("red", np.array([0, 1]))], length=4)
        result = s.apply("ACGT")
        # Should contain ANSI codes
        assert "\033[" in result
        # Should contain the sequence
        assert "A" in result and "C" in result

    # Edge cases
    def test_slice_zero_length(self):
        """Slicing with start==end returns empty SeqStyle."""
        s = SeqStyle.from_style_list([("red", np.array([5]))], length=10)
        region = s[5:5]
        assert region.length == 0
        assert region.style_list == []

    def test_slice_invalid_step_raises(self):
        """Slicing with step raises ValueError."""
        s = SeqStyle.empty(10)
        with pytest.raises(ValueError):
            _ = s[::2]

    def test_slice_non_slice_raises(self):
        """Integer indexing raises TypeError."""
        s = SeqStyle.empty(10)
        with pytest.raises(TypeError):
            _ = s[5]


class TestSeqStyleFromParent:
    """Test SeqStyle.from_parent() class method."""

    def test_from_parent_returns_parent_style(self):
        """from_parent returns parent style when available."""
        parent_style = SeqStyle.from_style_list([("red", np.array([0, 1]))], length=5)
        parent_styles = [parent_style]

        result = SeqStyle.from_parent(parent_styles, 0, 10)

        # Should return the parent style (same object)
        assert result is parent_style

    def test_from_parent_returns_empty_when_none(self):
        """from_parent returns empty SeqStyle when parent_styles is None."""
        result = SeqStyle.from_parent(None, 0, 10)

        assert result.length == 10
        assert result.style_list == []
        assert not result

    def test_from_parent_returns_empty_when_empty_list(self):
        """from_parent returns empty SeqStyle when parent_styles is empty list."""
        result = SeqStyle.from_parent([], 0, 10)

        assert result.length == 10
        assert result.style_list == []

    def test_from_parent_returns_empty_when_index_out_of_range(self):
        """from_parent returns empty SeqStyle when index >= len(parent_styles)."""
        parent_style = SeqStyle.from_style_list([("red", np.array([0]))], length=5)
        parent_styles = [parent_style]

        result = SeqStyle.from_parent(parent_styles, 1, 10)

        assert result.length == 10
        assert result.style_list == []

    def test_from_parent_with_multiple_parents(self):
        """from_parent can retrieve from different indices."""
        style0 = SeqStyle.from_style_list([("red", np.array([0]))], length=5)
        style1 = SeqStyle.from_style_list([("blue", np.array([0]))], length=5)
        parent_styles = [style0, style1]

        result0 = SeqStyle.from_parent(parent_styles, 0, 10)
        result1 = SeqStyle.from_parent(parent_styles, 1, 10)

        assert result0 is style0
        assert result1 is style1


class TestSeqStyleFull:
    """Test SeqStyle.full() class method."""

    def test_full_with_style(self):
        """full() creates SeqStyle with style on all positions."""
        result = SeqStyle.full(5, "red")

        assert result.length == 5
        assert len(result.style_list) == 1
        spec, positions = result.style_list[0]
        assert spec == "red"
        assert list(positions) == [0, 1, 2, 3, 4]

    def test_full_without_style_returns_empty(self):
        """full() with style=None returns empty SeqStyle."""
        result = SeqStyle.full(5, None)

        assert result.length == 5
        assert result.style_list == []
        assert not result

    def test_full_with_zero_length(self):
        """full() with length=0 returns empty SeqStyle."""
        result = SeqStyle.full(0, "red")

        assert result.length == 0
        assert result.style_list == []

    def test_full_with_complex_style(self):
        """full() works with complex style specs."""
        result = SeqStyle.full(3, "bold cyan underline")

        assert result.length == 3
        assert len(result.style_list) == 1
        spec, positions = result.style_list[0]
        assert spec == "bold cyan underline"
        assert list(positions) == [0, 1, 2]

    def test_full_positions_are_int64(self):
        """full() creates positions as int64 dtype."""
        result = SeqStyle.full(3, "red")

        spec, positions = result.style_list[0]
        assert positions.dtype == np.int64


class TestShuffleScanStylePropagation:
    """Test that parent styles propagate correctly through shuffle_scan."""

    def test_shuffle_scan_preserves_parent_styles(self):
        """Parent styles should propagate through shuffle_scan."""
        with pp.Party():
            pool = pp.from_seq("AA<cre>TTTTTTTTTT</cre>GG")
            pool = pool.stylize(region="cre", style="purple")
            pool = pool.shuffle_scan(region="cre", shuffle_length=4, mode="sequential").named(
                "test"
            )

        df = pool.generate_library(
            num_cycles=1, seed=42, _include_inline_styles=True
        )
        # Verify purple style is present on CRE positions
        assert "_inline_styles" in df.columns
        for _, row in df.iterrows():
            style = row["_inline_styles"]
            assert style is not None
            assert len(style.style_list) > 0
            # Check that purple style exists
            has_purple = any("purple" in spec for spec, _ in style.style_list)
            assert has_purple, "Purple style from parent should be present"

    def test_shuffle_scan_combines_parent_and_operation_styles(self):
        """shuffle_scan should preserve parent styles and add its own."""
        with pp.Party():
            pool = pp.from_seq("AA<cre>TTTTTTTTTT</cre>GG")
            pool = pool.stylize(region="cre", style="purple")
            pool = pool.shuffle_scan(
                region="cre", shuffle_length=4, style="magenta bold", mode="sequential"
            ).named("test")

        df = pool.generate_library(
            num_cycles=1, seed=42, _include_inline_styles=True
        )
        assert "_inline_styles" in df.columns
        for _, row in df.iterrows():
            style = row["_inline_styles"]
            assert style is not None
            # Should have both purple (parent) and magenta (shuffle_scan)
            specs = [spec for spec, _ in style.style_list]
            has_purple = any("purple" in spec for spec in specs)
            has_magenta = any("magenta" in spec for spec in specs)
            assert has_purple, "Purple style from parent should be present"
            assert has_magenta, "Magenta style from shuffle_scan should be present"


class TestStyleSuppression:
    """Test style suppression via toggle_styles()."""

    def test_toggle_styles_off_suppresses_styles(self):
        """toggle_styles(on=False) sets Seq.style to None."""
        with pp.Party():
            pp.toggle_styles(on=False)
            pool = pp.from_seqs(["ACGT", "TTTT"], mode="sequential").named("test")
            df = pool.generate_library(
                num_seqs=2, _include_inline_styles=True
            )

            # Check that styles are None
            for i in range(2):
                seq_style = df["_inline_styles"].iloc[i]
                assert seq_style is None, f"Row {i} should have None style when suppressed"

    def test_toggle_styles_on_enables_styles(self):
        """toggle_styles(on=True) restores normal style behavior."""
        with pp.Party():
            pp.toggle_styles(on=True)
            pool = pp.from_seqs(["ACGT", "TTTT"], mode="sequential", style="red").named("test")

        df = pool.generate_library(
            num_seqs=2, _include_inline_styles=True
        )

        # Check that styles are present
        for i in range(2):
            seq_style = df["_inline_styles"].iloc[i]
            assert seq_style is not None, f"Row {i} should have style when enabled"
            assert len(seq_style.style_list) == 1
            assert seq_style.style_list[0][0] == "red"

    def test_stylize_is_noop_when_suppressed(self):
        """stylize() is a no-op when styles are suppressed."""
        with pp.Party():
            pp.toggle_styles(on=False)
            pool = pp.from_seq("ACGT").stylize(style="red").named("test")
            df = pool.generate_library(
                num_seqs=1, _include_inline_styles=True
            )
            seq_style = df["_inline_styles"].iloc[0]
            assert seq_style is None, "stylize should be no-op when suppressed"

    def test_mutagenize_no_styles_when_suppressed(self):
        """mutagenize with style parameter doesn't create styles when suppressed."""
        with pp.Party():
            pp.toggle_styles(on=False)
            pool = pp.mutagenize("ACGT", num_mutations=1, style="red", mode="sequential").named(
                "test"
            )
            df = pool.generate_library(
                num_seqs=1, _include_inline_styles=True
            )
            seq_style = df["_inline_styles"].iloc[0]
            assert seq_style is None

    def test_styles_pass_through_none_in_operations(self):
        """Operations correctly pass through None styles."""
        with pp.Party():
            pp.toggle_styles(on=False)
            pool = pp.from_seq("ACGT").mutagenize(num_mutations=1, mode="sequential").named("test")
            df = pool.generate_library(
                num_seqs=1, _include_inline_styles=True
            )
            seq_style = df["_inline_styles"].iloc[0]
            assert seq_style is None

    def test_toggle_persists_through_operations(self):
        """Style suppression persists through operation chains."""
        with pp.Party():
            pp.toggle_styles(on=False)
            pool = (
                pp.from_seq("ACGTACGT")
                .mutagenize(num_mutations=1, style="red", mode="sequential")
                .stylize(style="blue")
                .repeat(2)
                .named("test")
            )
            df = pool.generate_library(
                num_seqs=2, _include_inline_styles=True
            )
            for i in range(2):
                seq_style = df["_inline_styles"].iloc[i]
                assert seq_style is None

    def test_region_ops_work_with_suppressed_styles(self):
        """Region operations work correctly with suppressed styles."""
        with pp.Party():
            pp.toggle_styles(on=False)
            bg = pp.from_seq("AA<test>CCCC</test>GG").named("bg")
            mutated = bg.mutagenize(region="test", num_mutations=1, mode="sequential").named(
                "mutated"
            )
            df = mutated.generate_library(
                num_seqs=1, _include_inline_styles=True
            )
            seq_style = df["_inline_styles"].iloc[0]
            assert seq_style is None

    def test_recombine_with_suppressed_styles(self):
        """recombine works correctly with suppressed styles."""
        with pp.Party():
            pp.toggle_styles(on=False)
            pool1 = pp.from_seq("AAAAAAAAAA").named("pool1")
            pool2 = pp.from_seq("TTTTTTTTTT").named("pool2")
            recombined = pp.recombine(
                sources=[pool1, pool2], num_breakpoints=2, mode="sequential"
            ).named("recombined")
            df = recombined.generate_library(
                num_seqs=1, _include_inline_styles=True
            )
            seq_style = df["_inline_styles"].iloc[0]
            assert seq_style is None

    def test_insert_kmers_with_suppressed_styles(self):
        """insert_kmers (which uses region_scan) works with suppressed styles."""
        with pp.Party():
            pp.toggle_styles(on=False)
            pool = pp.from_seq("AA<bc/>TT").named("bg")
            pool_with_kmers = pool.insert_kmers(region="bc", length=3, mode="sequential").named(
                "with_kmers"
            )
            df = pool_with_kmers.generate_library(
                num_seqs=1, _include_inline_styles=True
            )
            seq_style = df["_inline_styles"].iloc[0]
            assert seq_style is None
            # Verify sequence was generated correctly (tags preserved by default)
            seq = df["seq"].iloc[0]
            # AA (2) + <bc> (4) + 3-char kmer (3) + </bc> (5) + TT (2) = 16 chars
            assert len(seq) == 16
            assert seq.startswith("AA")
            assert "<bc>" in seq
            assert seq.endswith("TT")


class TestDesignCardsOptIn:
    """Test that design cards are opt-in via cards parameter (new API)."""

    def test_default_output_has_no_design_cards(self):
        """Without cards param, output only has name and seq columns."""
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TTTT"], mode="sequential").named("test")
            df = pool.generate_library(num_seqs=2)

            # Check that design card keys are not in columns by default
            seq_name_cols = [c for c in df.columns if "seq_name" in c]
            seq_index_cols = [c for c in df.columns if "seq_index" in c]
            assert len(seq_name_cols) == 0
            assert len(seq_index_cols) == 0

            # name and seq should be present
            assert "name" in df.columns
            assert "seq" in df.columns

    def test_cards_list_adds_columns(self):
        """cards=["key"] adds the specified design card column."""
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TTTT"], mode="sequential", cards=["seq_name"]).named("test")
            df = pool.generate_library(num_seqs=2)

            # Find seq_name column (operation name is auto-generated)
            seq_name_cols = [c for c in df.columns if "seq_name" in c]
            assert len(seq_name_cols) > 0

    def test_cards_dict_custom_naming(self):
        """cards={"key": "custom"} uses custom column name."""
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TTTT"], mode="sequential", cards={"seq_index": "my_idx"}).named("test")
            df = pool.generate_library(num_seqs=2)

            # Custom column name should be present
            assert "my_idx" in df.columns
            # Default naming should not be used
            default_cols = [c for c in df.columns if "from_seqs.seq_index" in c]
            assert len(default_cols) == 0

    def test_styles_preserved_without_cards(self):
        """Inline styles work independently of cards parameter."""
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TTTT"], mode="sequential", style="red").named("test")
            df = pool.generate_library(
                num_seqs=2, _include_inline_styles=True
            )

            # Check that styles are present
            for i in range(2):
                seq_style = df["_inline_styles"].iloc[i]
                assert seq_style is not None, f"Row {i} should have style"
                assert len(seq_style.style_list) == 1
                assert seq_style.style_list[0][0] == "red"

            # But no design card columns
            seq_name_cols = [c for c in df.columns if "seq_name" in c]
            assert len(seq_name_cols) == 0

    def test_mutagenize_cards_opt_in(self):
        """mutagenize design cards require cards parameter."""
        with pp.Party():
            # Without cards
            pool = pp.mutagenize("ACGT", num_mutations=1, mode="sequential").named("test")
            df = pool.generate_library(num_seqs=1)

            # No design card keys
            positions_cols = [c for c in df.columns if "positions" in c]
            assert len(positions_cols) == 0

            # With cards
            pool2 = pp.mutagenize("ACGT", num_mutations=1, mode="sequential", cards=["positions"]).named("test2")
            df2 = pool2.generate_library(num_seqs=1)

            positions_cols2 = [c for c in df2.columns if "positions" in c]
            assert len(positions_cols2) > 0

    def test_toggle_styles_still_works(self):
        """toggle_styles(on=False) still suppresses styles."""
        with pp.Party():
            pp.toggle_styles(on=False)
            pool = pp.from_seqs(["ACGT"], style="red").named("test")
            df = pool.generate_library(
                num_seqs=1, _include_inline_styles=True
            )

            # Styles should be None
            assert df["_inline_styles"].iloc[0] is None

    def test_complex_dag_minimal_output(self):
        """Complex DAG has minimal output without cards param."""
        with pp.Party():
            pool1 = pp.from_seq("AAAA").named("pool1")
            pool2 = pp.from_seq("TTTT").named("pool2")
            recomb = pp.recombine(
                sources=[pool1, pool2], num_breakpoints=1, mode="sequential"
            ).named("recomb")
            mutated = recomb.mutagenize(num_mutations=1, mode="sequential").named("mutated")
            final = mutated.repeat(2).named("final")

            df = final.generate_library(num_seqs=2)

            # Should only have name and seq columns
            expected_cols = {"name", "seq"}
            actual_cols = set(df.columns)
            assert actual_cols == expected_cols, f"Expected {expected_cols}, got {actual_cols}"
