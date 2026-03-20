"""Tests for stylize operation."""

import poolparty as pp
from poolparty.fixed_ops.stylize import stylize


class TestStylizeBasic:
    """Test basic stylize functionality."""

    def test_stylize_returns_pool(self):
        """stylize() returns a Pool."""
        with pp.Party():
            pool = stylize("ACGT", style="red")
            assert isinstance(pool, pp.Pool)

    def test_stylize_preserves_sequence(self):
        """stylize() doesn't modify the sequence."""
        with pp.Party():
            pool = stylize("ACGT", style="red").named("styled")
            df = pool.generate_library(num_seqs=1)
            assert df["seq"].iloc[0] == "ACGT"

    def test_stylize_adds_inline_style(self):
        """stylize() adds inline styles to _inline_styles."""
        with pp.Party():
            pool = stylize("ACGT", style="red").named("styled")
            df = pool.generate_library(
                num_seqs=1, _include_inline_styles=True
            )
            seq_style = df["_inline_styles"].iloc[0]
            assert len(seq_style.style_list) == 1
            assert seq_style.style_list[0][0] == "red"

    def test_stylize_default_which_is_contents(self):
        """Default which='contents' styles all non-tag characters."""
        with pp.Party():
            pool = stylize("ACGT", style="blue").named("styled")
            df = pool.generate_library(
                num_seqs=1, _include_inline_styles=True
            )
            seq_style = df["_inline_styles"].iloc[0]
            spec, positions = seq_style.style_list[0]
            # All 4 positions should be styled
            assert len(positions) == 4
            assert list(positions) == [0, 1, 2, 3]


class TestStylizeWhichParameter:
    """Test stylize with different 'which' patterns."""

    def test_which_upper(self):
        """which='upper' styles only uppercase characters."""
        with pp.Party():
            pool = stylize("AcGt", style="red", which="upper").named("styled")
            df = pool.generate_library(
                num_seqs=1, _include_inline_styles=True
            )
            seq_style = df["_inline_styles"].iloc[0]
            spec, positions = seq_style.style_list[0]
            # Only positions 0 and 2 are uppercase
            assert list(positions) == [0, 2]

    def test_which_lower(self):
        """which='lower' styles only lowercase characters."""
        with pp.Party():
            pool = stylize("AcGt", style="blue", which="lower").named("styled")
            df = pool.generate_library(
                num_seqs=1, _include_inline_styles=True
            )
            seq_style = df["_inline_styles"].iloc[0]
            spec, positions = seq_style.style_list[0]
            # Only positions 1 and 3 are lowercase
            assert list(positions) == [1, 3]

    def test_which_gap(self):
        """which='gap' styles gap characters (-, ., space)."""
        with pp.Party():
            pool = stylize("A-C.G T", style="gray", which="gap").named("styled")
            df = pool.generate_library(
                num_seqs=1, _include_inline_styles=True
            )
            seq_style = df["_inline_styles"].iloc[0]
            spec, positions = seq_style.style_list[0]
            # Positions 1, 3, 5 are gap chars
            assert list(positions) == [1, 3, 5]

    def test_which_all(self):
        """which='all' styles all characters including tags."""
        with pp.Party():
            pool = stylize("<m>AC</m>", style="cyan", which="all").named("styled")
            df = pool.generate_library(
                num_seqs=1, _include_inline_styles=True
            )
            seq_style = df["_inline_styles"].iloc[0]
            spec, positions = seq_style.style_list[0]
            # All 9 positions
            assert len(positions) == 9

    def test_which_tags(self):
        """which='tags' styles only XML tag characters."""
        with pp.Party():
            pool = stylize("<m>AC</m>", style="gray", which="tags").named("styled")
            df = pool.generate_library(
                num_seqs=1, _include_inline_styles=True
            )
            seq_style = df["_inline_styles"].iloc[0]
            spec, positions = seq_style.style_list[0]
            # Tag positions: <m> is 0,1,2 and </m> is 5,6,7,8
            assert list(positions) == [0, 1, 2, 5, 6, 7, 8]


class TestStylizeWithRegion:
    """Test stylize with region parameter."""

    def test_marker_region(self):
        """Styling restricted to marker region."""
        with pp.Party():
            pool = stylize("AA<test>CCCC</test>GG", "test", style="red").named("styled")
            df = pool.generate_library(
                num_seqs=1, _include_inline_styles=True
            )
            seq_style = df["_inline_styles"].iloc[0]
            spec, positions = seq_style.style_list[0]
            # Region content is at positions 8-11 (after 'AA<test>')
            assert all(8 <= p < 12 for p in positions)
            assert len(positions) == 4

    def test_interval_region(self):
        """Styling restricted to [start, stop] interval."""
        with pp.Party():
            pool = stylize("AACCCCGG", [2, 6], style="blue").named("styled")
            df = pool.generate_library(
                num_seqs=1, _include_inline_styles=True
            )
            seq_style = df["_inline_styles"].iloc[0]
            spec, positions = seq_style.style_list[0]
            # Positions 2-5 (CCCC)
            assert list(positions) == [2, 3, 4, 5]

    def test_region_not_found(self):
        """No styles when region not found in sequence."""
        with pp.Party():
            pool = stylize("ACGT", "nonexistent", style="red").named("styled")
            df = pool.generate_library(
                num_seqs=1, _include_inline_styles=True
            )
            seq_style = df["_inline_styles"].iloc[0]
            assert not seq_style


class TestStylizeWithRegex:
    """Test stylize with custom regex pattern."""

    def test_regex_pattern(self):
        """Custom regex pattern for styling."""
        with pp.Party():
            pool = stylize("ACGTACGT", style="red", regex=r"ACG").named("styled")
            df = pool.generate_library(
                num_seqs=1, _include_inline_styles=True
            )
            seq_style = df["_inline_styles"].iloc[0]
            spec, positions = seq_style.style_list[0]
            # ACG at positions 0-2 and 4-6
            assert list(positions) == [0, 1, 2, 4, 5, 6]

    def test_regex_overrides_which(self):
        """regex parameter overrides which parameter."""
        with pp.Party():
            pool = stylize("AcGt", style="blue", which="upper", regex=r"[a-z]").named("styled")
            df = pool.generate_library(
                num_seqs=1, _include_inline_styles=True
            )
            seq_style = df["_inline_styles"].iloc[0]
            spec, positions = seq_style.style_list[0]
            # regex matches lowercase, not uppercase
            assert list(positions) == [1, 3]


class TestStylizeCaseTransforms:
    """Test stylize with case transform style specs."""

    def test_lower_case_transform(self):
        """Style spec with 'lower' transforms to lowercase."""
        with pp.Party():
            pool = stylize("ACGT", style="lower cyan").named("styled")
            pool.print_library(show_state=False)
            # The actual case transform happens at render time
            # Just verify the style is stored correctly
            df = pool.generate_library(
                num_seqs=1, _include_inline_styles=True
            )
            seq_style = df["_inline_styles"].iloc[0]
            assert seq_style.style_list[0][0] == "lower cyan"

    def test_upper_case_transform(self):
        """Style spec with 'upper' transforms to uppercase."""
        with pp.Party():
            pool = stylize("acgt", style="upper red bold").named("styled")
            df = pool.generate_library(
                num_seqs=1, _include_inline_styles=True
            )
            seq_style = df["_inline_styles"].iloc[0]
            assert seq_style.style_list[0][0] == "upper red bold"


class TestStylizeChaining:
    """Test chaining multiple stylize operations."""

    def test_multiple_stylize_operations(self):
        """Multiple stylize ops accumulate styles."""
        with pp.Party():
            pool = pp.from_seq("AcGt").named("base")
            styled1 = pool.stylize(style="red", which="upper").named("upper_styled")
            styled2 = styled1.stylize(style="blue", which="lower").named("both_styled")

            df = styled2.generate_library(
                num_seqs=1, _include_inline_styles=True
            )
            seq_style = df["_inline_styles"].iloc[0]

            # Should have two style entries
            assert len(seq_style.style_list) == 2
            assert seq_style.style_list[0][0] == "red"  # First: uppercase
            assert seq_style.style_list[1][0] == "blue"  # Second: lowercase

    def test_stylize_after_mutagenize(self):
        """stylize works after mutagenize in chain."""
        with pp.Party():
            pool = pp.from_seq("ACGT").named("base")
            mutated = pool.mutagenize(num_mutations=1, mode="sequential").named("mutated")
            styled = mutated.stylize(style="cyan", which="lower").named("styled")

            df = styled.generate_library(
                num_seqs=1, _include_inline_styles=True
            )
            # Should have style for lowercase (mutations)
            assert "_inline_styles" in df.columns


class TestStylizeIntegration:
    """Test stylize integration with print_library."""

    def test_print_library_applies_styles(self):
        """print_library() renders stylize styles."""
        with pp.Party():
            pool = stylize("ACGT", style="red").named("styled")
            # Just verify it doesn't crash
            pool.print_library()


class TestStylizeOp:
    """Test StylizeOp class directly."""

    def test_factory_name(self):
        """StylizeOp has correct factory_name."""
        with pp.Party():
            pool = stylize("ACGT", style="red")
            assert pool.operation.factory_name == "stylize"

    def test_get_copy_params(self):
        """_get_copy_params returns correct parameters."""
        with pp.Party():
            pool = stylize("ACGT", "test", style="red", which="upper", regex=r"A")
            params = pool.operation._get_copy_params()
            assert params["style"] == "red"
            assert params["region"] == "test"
            assert params["which"] is None  # regex overrides which
            assert params["regex"] == r"A"

    def test_seq_length_preserved(self):
        """StylizeOp preserves parent seq_length."""
        with pp.Party():
            base = pp.from_seq("ACGTACGT").named("base")
            styled = base.stylize(style="red").named("styled")
            assert styled.seq_length == 8


class TestStylizeEdgeCases:
    """Test edge cases for stylize."""

    def test_empty_sequence(self):
        """stylize handles empty sequence."""
        with pp.Party():
            pool = stylize("", style="red").named("styled")
            df = pool.generate_library(
                num_seqs=1, _include_inline_styles=True
            )
            styles = df["_inline_styles"].iloc[0]
            assert len(styles) == 0

    def test_no_matching_positions(self):
        """No styles when pattern doesn't match."""
        with pp.Party():
            pool = stylize("ACGT", style="red", which="lower").named("styled")
            df = pool.generate_library(
                num_seqs=1, _include_inline_styles=True
            )
            seq_style = df["_inline_styles"].iloc[0]
            assert not seq_style

    def test_stylize_excludes_tags_by_default(self):
        """Default 'contents' excludes tag characters."""
        with pp.Party():
            pool = stylize("<m>AC</m>", style="red").named("styled")
            df = pool.generate_library(
                num_seqs=1, _include_inline_styles=True
            )
            seq_style = df["_inline_styles"].iloc[0]
            spec, positions = seq_style.style_list[0]
            # Only positions 3 and 4 (AC between tags)
            assert list(positions) == [3, 4]
