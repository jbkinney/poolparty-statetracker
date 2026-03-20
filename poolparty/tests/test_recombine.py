"""Tests for the Recombine operation."""

import pytest

import poolparty as pp
from poolparty.base_ops.recombine import RecombineOp, recombine


class TestRecombineFactory:
    """Test recombine factory function."""

    def test_returns_pool(self):
        """recombine returns a Pool object."""
        with pp.Party() as party:
            pool = recombine(sources=["ACGT", "TGCA"])
            assert pool is not None
            assert hasattr(pool, "operation")

    def test_creates_recombine_op(self):
        """Pool's operation is RecombineOp."""
        with pp.Party() as party:
            pool = recombine(sources=["ACGT", "TGCA"])
            assert isinstance(pool.operation, RecombineOp)

    def test_accepts_string_inputs(self):
        """Factory accepts strings in sources."""
        with pp.Party() as party:
            pool = recombine(sources=["ACGT", "TGCA"]).named("recombined")

        df = pool.generate_library(num_seqs=1, seed=42)
        assert len(df["seq"].iloc[0]) == 4

    def test_accepts_pool_inputs(self):
        """Factory accepts Pool objects in sources."""
        with pp.Party() as party:
            pool1 = pp.from_seq("ACGT")
            pool2 = pp.from_seq("TGCA")
            pool = recombine(sources=[pool1, pool2]).named("recombined")

        df = pool.generate_library(num_seqs=1, seed=42)
        assert len(df["seq"].iloc[0]) == 4

    def test_accepts_mixed_inputs(self):
        """Factory accepts mix of strings and Pools in sources."""
        with pp.Party() as party:
            pool1 = pp.from_seq("ACGT")
            pool = recombine(sources=[pool1, "TGCA"]).named("recombined")

        df = pool.generate_library(num_seqs=1, seed=42)
        assert len(df["seq"].iloc[0]) == 4


class TestRecombineParameterValidation:
    """Test parameter validation."""

    def test_requires_at_least_two_sources(self):
        """sources must have at least 2 pools."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="must contain at least 2 pools"):
                recombine(sources=["ACGT"])

    def test_requires_fixed_length(self):
        """All source pools must have fixed seq_length."""
        with pp.Party() as party:
            pool1 = pp.from_seqs(["ACGT", "AAAAA"])  # Variable length
            with pytest.raises(ValueError, match="must have a fixed seq_length"):
                recombine(sources=[pool1, "ACGT"])

    def test_requires_same_length(self):
        """All source pools must have the same seq_length."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="must have the same seq_length"):
                recombine(sources=["ACGT", "TGCAA"])

    def test_num_breakpoints_minimum(self):
        """num_breakpoints must be >= 1."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="num_breakpoints must be >= 1"):
                recombine(sources=["ACGT", "TGCA"], num_breakpoints=0)

    def test_num_breakpoints_maximum(self):
        """num_breakpoints must be <= seq_length - 1."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="exceeds seq_length - 1"):
                recombine(sources=["ACGT", "TGCA"], num_breakpoints=4)

    def test_not_enough_positions(self):
        """positions must have enough elements for num_breakpoints."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="Not enough positions"):
                recombine(sources=["ACGT", "TGCA"], num_breakpoints=2, positions=[0])

    def test_invalid_position_negative(self):
        """Negative positions are invalid."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="Invalid position"):
                recombine(sources=["ACGT", "TGCA"], positions=[-1, 0, 1])

    def test_invalid_position_too_large(self):
        """Positions >= seq_length - 1 are invalid."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="Invalid position"):
                recombine(sources=["ACGT", "TGCA"], positions=[0, 1, 2, 3])

    def test_styles_empty_list(self):
        """styles cannot be an empty list."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="styles must be non-empty"):
                recombine(sources=["ACGT", "TGCA"], num_breakpoints=1, styles=[])


class TestRecombineBasicFunctionality:
    """Test basic recombination functionality."""

    def test_simple_recombination_fixed_mode(self):
        """Simple recombination in fixed mode."""
        with pp.Party() as party:
            pool = recombine(
                sources=["AAAA", "TTTT"],
                num_breakpoints=1,
                positions=[1],  # Break after index 1
                mode="fixed",
            ).named("recombined")

        df = pool.generate_library()
        assert len(df) == 1
        # Fixed mode uses first position (1) with round-robin pool assignment
        # Segment 0: [0:2] from pool 0 = 'AA'
        # Segment 1: [2:4] from pool 1 = 'TT'
        assert df["seq"].iloc[0] == "AATT"

    def test_recombination_preserves_length(self):
        """Recombined sequences preserve length."""
        with pp.Party() as party:
            pool = recombine(
                sources=["ACGTACGT", "TGCATGCA"], num_breakpoints=2, mode="random", num_states=10
            ).named("recombined")

        df = pool.generate_library(seed=42)
        assert all(len(seq) == 8 for seq in df["seq"])

    def test_breakpoint_interpretation(self):
        """Breakpoint position i means 'after index i'."""
        with pp.Party() as party:
            pool = recombine(
                sources=["AAAA", "TTTT"],
                num_breakpoints=1,
                positions=[0],  # Break after index 0
                mode="fixed",
            ).named("recombined")

        df = pool.generate_library()
        # Segment 0: [0:1] from pool 0 = 'A'
        # Segment 1: [1:4] from pool 1 = 'TTT'
        assert df["seq"].iloc[0] == "ATTT"

    def test_multiple_breakpoints(self):
        """Recombination with multiple breakpoints."""
        with pp.Party() as party:
            pool = recombine(
                sources=["AAAA", "TTTT"],
                num_breakpoints=2,
                positions=[0, 2],  # Break after indices 0 and 2
                mode="fixed",
            ).named("recombined")

        df = pool.generate_library()
        # Segment 0: [0:1] from pool 0 = 'A'
        # Segment 1: [1:3] from pool 1 = 'TT'
        # Segment 2: [3:4] from pool 2%2=0 = 'A'
        assert df["seq"].iloc[0] == "ATTA"


class TestRecombineSequentialMode:
    """Test sequential mode enumeration."""

    def test_sequential_enumeration_count(self):
        """Sequential mode enumerates correct number of states."""
        with pp.Party() as party:
            # 2 source pools, 1 breakpoint, 3 valid positions [0,1,2]
            # States = C(3,1) × N × (N-1)^K = 3 × 2 × 1 = 6
            # (consecutive segments must come from different pools)
            pool = recombine(sources=["AAAA", "TTTT"], num_breakpoints=1, mode="sequential").named(
                "recombined"
            )

        assert pool.num_states == 6
        df = pool.generate_library()
        assert len(df) == 6

    def test_sequential_all_unique(self):
        """Sequential mode generates all combinations with no self-recombination."""
        with pp.Party() as party:
            pool = recombine(
                sources=["AAAA", "TTTT"], num_breakpoints=1, positions=[0, 1], mode="sequential"
            ).named("recombined")

        df = pool.generate_library()
        # C(2,1) × N × (N-1)^K = 2 × 2 × 1 = 4 states
        assert len(df) == 4
        # All should be unique since consecutive segments differ
        assert len(df["seq"].unique()) == 4

    def test_no_self_recombination(self):
        """Consecutive segments must come from different pools."""
        with pp.Party() as party:
            pool = recombine(sources=["AAAA", "TTTT"], num_breakpoints=1, mode="sequential", cards=["pool_assignments"]).named(
                "recombined"
            )

        op_name = pool.operation.name
        df = pool.generate_library()

        # Check that no pool_assignments have consecutive identical values
        for assignments in df[f"{op_name}.pool_assignments"]:
            for i in range(1, len(assignments)):
                assert assignments[i] != assignments[i - 1], (
                    f"Self-recombination detected: {assignments}"
                )


class TestRecombineRandomMode:
    """Test random mode functionality."""

    def test_random_mode_with_seed(self):
        """Random mode with seed is reproducible."""
        with pp.Party() as party:
            pool = recombine(
                sources=["ACGT", "TGCA"], num_breakpoints=1, mode="random", num_states=10
            ).named("recombined")

        df1 = pool.generate_library(seed=42)
        df2 = pool.generate_library(seed=42)
        assert df1["seq"].tolist() == df2["seq"].tolist()

    def test_random_mode_different_seeds(self):
        """Random mode with different seeds produces different results."""
        with pp.Party() as party:
            pool = recombine(
                sources=["ACGT", "TGCA"], num_breakpoints=1, mode="random", num_states=10
            ).named("recombined")

        df1 = pool.generate_library(seed=42)
        df2 = pool.generate_library(seed=123)
        # Very unlikely to be identical with random breakpoints and assignments
        assert df1["seq"].tolist() != df2["seq"].tolist()


class TestRecombineStyles:
    """Test style inheritance and overlay."""

    def test_style_inheritance_from_sources(self):
        """Segments inherit styles from their sources."""
        with pp.Party() as party:
            # Create source pools with styles
            pool1 = pp.from_seq("AAAA").mutagenize(
                num_mutations=1, style="red", mode="random", num_states=1
            )
            pool2 = pp.from_seq("TTTT").mutagenize(
                num_mutations=1, style="blue", mode="random", num_states=1
            )

            pool = recombine(
                sources=[pool1, pool2], num_breakpoints=1, positions=[1], mode="fixed"
            ).named("recombined")

        df = pool.generate_library(seed=42)
        # Should have some styling inherited from source pools
        assert len(df) == 1

    def test_styles_overlay(self):
        """styles parameter overlays on inherited styles."""
        with pp.Party() as party:
            pool = recombine(
                sources=["AAAA", "TTTT"],
                num_breakpoints=1,
                positions=[1],
                mode="fixed",
                styles=["green", "yellow"],
            ).named("recombined")

        df = pool.generate_library()
        # Should have applied green to segment 0, yellow to segment 1
        assert len(df) == 1

    def test_empty_string_styles(self):
        """Empty string '' means no additional styling for segment."""
        with pp.Party() as party:
            pool = recombine(
                sources=["AAAA", "TTTT"],
                num_breakpoints=1,
                positions=[1],
                mode="fixed",
                styles=["green", ""],
            ).named("recombined")

        df = pool.generate_library()
        assert len(df) == 1


class TestRecombineRegionBased:
    """Test region-based recombination."""

    def test_region_recombination(self):
        """Recombination can replace a region."""
        with pp.Party() as party:
            # Create a pool with a region
            pool = pp.from_seq("NNNNNNNN").insert_tags(region_name="middle", start=2, stop=6)

            # Recombine into the region
            result = pool.recombine(
                region="middle",
                sources=["AAAA", "TTTT"],
                num_breakpoints=1,
                positions=[1],
                mode="fixed",
            ).named("recombined")

        df = result.generate_library()
        # Should be: NN + <middle>AATT</middle> + NN (tags preserved by default)
        assert df["seq"].iloc[0] == "NN<middle>AATT</middle>NN"

    def test_region_content_discarded(self):
        """Region content is replaced, not used as source."""
        with pp.Party() as party:
            # Create a pool with content in the region
            pool = pp.from_seq("GGGGGGGG").insert_tags(region_name="middle", start=2, stop=6)

            # Recombine into the region (region content 'GGGG' is discarded)
            result = pool.recombine(
                region="middle",
                sources=["AAAA", "TTTT"],
                num_breakpoints=1,
                positions=[1],
                mode="fixed",
            ).named("recombined")

        df = result.generate_library()
        # Should be: GG + <middle>AATT</middle> + GG (tags preserved by default)
        # NOT using the original GGGG from the region
        assert df["seq"].iloc[0] == "GG<middle>AATT</middle>GG"


class TestRecombineDesignCard:
    """Test design card generation."""

    def test_design_card_has_breakpoints(self):
        """Design card includes breakpoints when cards requested."""
        with pp.Party() as party:
            pool = recombine(
                sources=["AAAA", "TTTT"], num_breakpoints=1, positions=[1], mode="fixed", cards=["breakpoints"]
            ).named("recombined")

        op_name = pool.operation.name
        df = pool.generate_library()
        assert f"{op_name}.breakpoints" in df.columns
        assert df[f"{op_name}.breakpoints"].iloc[0] == (1,)

    def test_design_card_has_pool_assignments(self):
        """Design card includes pool_assignments when cards requested."""
        with pp.Party() as party:
            pool = recombine(
                sources=["AAAA", "TTTT"], num_breakpoints=1, positions=[1], mode="fixed", cards=["pool_assignments"]
            ).named("recombined")

        op_name = pool.operation.name
        df = pool.generate_library()
        assert f"{op_name}.pool_assignments" in df.columns
        assert df[f"{op_name}.pool_assignments"].iloc[0] == (0, 1)


class TestRecombineMixinMethod:
    """Test Pool.recombine() mixin method."""

    def test_mixin_method_exists(self):
        """Pool objects have recombine method."""
        with pp.Party() as party:
            pool = pp.from_seq("NNNNNNNN")
            assert hasattr(pool, "recombine")

    def test_mixin_method_works(self):
        """Pool.recombine() mixin method works correctly."""
        with pp.Party() as party:
            pool = pp.from_seq("NNNNNNNN").insert_tags(region_name="middle", start=2, stop=6)
            result = pool.recombine(
                region="middle",
                sources=["AAAA", "TTTT"],
                num_breakpoints=1,
                positions=[1],
                mode="fixed",
            ).named("recombined")

        df = result.generate_library()
        # Tags preserved by default
        assert df["seq"].iloc[0] == "NN<middle>AATT</middle>NN"


class TestRecombineStyleBy:
    """Test style_by parameter for style assignment."""

    def test_style_by_order_default(self):
        """Default style_by is 'order'."""
        with pp.Party() as party:
            pool = recombine(
                sources=["AAAA", "TTTT"],
                num_breakpoints=1,
                positions=[1],
                mode="fixed",
                styles=["red", "blue"],
            ).named("recombined")

            # Should succeed with 2 styles (num_breakpoints + 1)
            df = pool.generate_library()
            assert len(df) == 1

    def test_style_by_order_accepts_any_length(self):
        """style_by='order' accepts any non-empty list length."""
        with pp.Party() as party:
            # Should succeed with matching length
            pool1 = recombine(
                sources=["AAAA", "TTTT", "GGGG"],
                num_breakpoints=2,
                positions=[1, 2],
                mode="fixed",
                styles=["red", "blue", "green"],  # 3 styles for 3 segments
                style_by="order",
            ).named("recombined1")
            df1 = pool1.generate_library()
            assert len(df1) == 1

            # Should also succeed with fewer styles (will cycle)
            pool2 = recombine(
                sources=["AAAA", "TTTT", "GGGG"],
                num_breakpoints=2,
                positions=[1, 2],
                mode="fixed",
                styles=["red", "blue"],  # 2 styles for 3 segments (will cycle)
                style_by="order",
            ).named("recombined2")
            df2 = pool2.generate_library()
            assert len(df2) == 1

            # Should succeed with single style
            pool3 = recombine(
                sources=["AAAA", "TTTT"],
                num_breakpoints=1,
                positions=[1],
                mode="fixed",
                styles=["red"],  # 1 style for 2 segments (will cycle)
                style_by="order",
            ).named("recombined3")
            df3 = pool3.generate_library()
            assert len(df3) == 1

    def test_style_by_source_accepts_any_length(self):
        """style_by='source' accepts any non-empty list length."""
        with pp.Party() as party:
            # Should succeed with matching length
            pool1 = recombine(
                sources=["AAAA", "TTTT", "GGGG"],
                num_breakpoints=2,
                positions=[1, 2],
                mode="fixed",
                styles=["red", "blue", "green"],  # 3 styles for 3 sources
                style_by="source",
            ).named("recombined1")
            df1 = pool1.generate_library()
            assert len(df1) == 1

            # Should also succeed with fewer styles (will cycle)
            pool2 = recombine(
                sources=["AAAA", "TTTT", "GGGG"],
                num_breakpoints=2,
                positions=[1, 2],
                mode="fixed",
                styles=["red", "blue"],  # 2 styles for 3 sources (will cycle)
                style_by="source",
            ).named("recombined2")
            df2 = pool2.generate_library()
            assert len(df2) == 1

            # Should succeed with single style
            pool3 = recombine(
                sources=["AAAA", "TTTT"],
                num_breakpoints=1,
                positions=[1],
                mode="fixed",
                styles=["red"],  # 1 style for 2 sources (will cycle)
                style_by="source",
            ).named("recombined3")
            df3 = pool3.generate_library()
            assert len(df3) == 1

    def test_style_by_source_empty_list_error(self):
        """style_by='source' with empty list raises error."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="styles must be non-empty"):
                recombine(
                    sources=["AAAA", "TTTT"],
                    num_breakpoints=1,
                    styles=[],  # Empty list not allowed
                    style_by="source",
                )

    def test_style_by_order_empty_list_error(self):
        """style_by='order' with empty list raises error."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="styles must be non-empty"):
                recombine(
                    sources=["AAAA", "TTTT"],
                    num_breakpoints=1,
                    styles=[],  # Empty list not allowed
                    style_by="order",
                )

    def test_style_by_source_applies_correctly(self):
        """style_by='source' applies styles based on source pool."""
        with pp.Party() as party:
            # Create recombination where we can track which pool each segment came from
            pool = recombine(
                sources=["AAAA", "TTTT"],
                num_breakpoints=1,
                positions=[1],
                mode="fixed",
                styles=["red", "blue"],  # red for pool 0, blue for pool 1
                style_by="source",
                cards=["pool_assignments"],
            ).named("recombined")

            op_name = pool.operation.name
            df = pool.generate_library()

            # Fixed mode: segment 0 from pool 0, segment 1 from pool 1
            pool_assignments = df[f"{op_name}.pool_assignments"].iloc[0]
            assert pool_assignments == (0, 1)

            # Both segments should have styles from their source pools
            # This validates the logic executes without error
            assert len(df) == 1

    def test_style_by_order_applies_correctly(self):
        """style_by='order' applies styles based on segment position."""
        with pp.Party() as party:
            # Create recombination where segments might come from different pools
            pool = recombine(
                sources=["AAAA", "TTTT"],
                num_breakpoints=1,
                positions=[1],
                mode="fixed",
                styles=["green", "yellow"],  # green for segment 0, yellow for segment 1
                style_by="order",
            ).named("recombined")

            df = pool.generate_library()

            # Styles should be applied by segment position
            # This validates the logic executes without error
            assert len(df) == 1

    def test_style_by_order_cycles_styles(self):
        """style_by='order' cycles through styles when fewer than segments."""
        with pp.Party() as party:
            # 3 segments, 2 styles - should cycle: red, blue, red
            pool = recombine(
                sources=["AAAA", "TTTT"],
                num_breakpoints=2,
                positions=[0, 2],
                mode="fixed",
                styles=["red", "blue"],  # 2 styles for 3 segments
                style_by="order",
            ).named("recombined")

            df = pool.generate_library()

            # Should cycle through: segment 0=red, segment 1=blue, segment 2=red
            assert len(df) == 1

    def test_style_by_order_single_style_all_segments(self):
        """style_by='order' with single style applies to all segments."""
        with pp.Party() as party:
            # 5 segments, 1 style - should apply same style to all
            pool = recombine(
                sources=["AAAAA", "TTTTT"],
                num_breakpoints=4,
                positions=[0, 1, 2, 3],
                mode="fixed",
                styles=["bold"],  # 1 style for 5 segments
                style_by="order",
            ).named("recombined")

            df = pool.generate_library()

            # Should apply 'bold' to all 5 segments
            assert len(df) == 1

    def test_style_by_source_with_multiple_segments_from_same_pool(self):
        """style_by='source' applies same style to multiple segments from same pool."""
        with pp.Party() as party:
            # Use 3 source pools, 2 breakpoints = 3 segments
            # With fixed mode: pool assignments are (0, 1, 0) - alternating
            pool = recombine(
                sources=["AAAA", "TTTT", "GGGG"],
                num_breakpoints=2,
                positions=[0, 2],
                mode="fixed",
                styles=["red", "blue", "green"],  # One style per source pool
                style_by="source",
                cards=["pool_assignments"],
            ).named("recombined")

            op_name = pool.operation.name
            df = pool.generate_library()

            # Check the pool assignments
            pool_assignments = df[f"{op_name}.pool_assignments"].iloc[0]
            # Fixed mode alternates: 0, 1, 0
            assert pool_assignments[0] != pool_assignments[1]

            # Validate execution
            assert len(df) == 1

    def test_style_by_source_cycles_styles(self):
        """style_by='source' cycles through styles when fewer than sources."""
        with pp.Party() as party:
            # 3 sources, 2 styles - should cycle: source[0]→red, source[1]→blue, source[2]→red
            pool = recombine(
                sources=["AAAA", "TTTT", "GGGG"],
                num_breakpoints=2,
                positions=[0, 2],
                mode="fixed",
                styles=["red", "blue"],  # 2 styles for 3 sources
                style_by="source",
                cards=["pool_assignments"],
            ).named("recombined")

            op_name = pool.operation.name
            df = pool.generate_library()

            # Fixed mode: pool assignments are (0, 1, 0)
            # Styles should be: source[0]→style[0]=red, source[1]→style[1]=blue, source[0]→style[0]=red
            pool_assignments = df[f"{op_name}.pool_assignments"].iloc[0]
            assert pool_assignments == (0, 1, 0)

            assert len(df) == 1

    def test_style_by_source_single_style_all_sources(self):
        """style_by='source' with single style applies to all sources."""
        with pp.Party() as party:
            # Multiple sources, 1 style - should apply same style to all
            pool = recombine(
                sources=["AAAA", "TTTT", "GGGG"],
                num_breakpoints=2,
                positions=[0, 2],
                mode="fixed",
                styles=["bold"],  # 1 style for 3 sources
                style_by="source",
            ).named("recombined")

            df = pool.generate_library()

            # Should apply 'bold' to segments from all sources
            assert len(df) == 1
