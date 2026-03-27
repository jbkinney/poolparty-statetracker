"""Tests for the mutagenize_scan operation."""

import warnings

import pytest

import poolparty as pp
from poolparty.scan_ops import mutagenize_scan
from poolparty.utils.parsing_utils import strip_all_tags


class TestMutagenizeScanBasics:
    """Test basic mutagenize_scan functionality."""

    def test_returns_pool(self):
        with pp.Party():
            bg = pp.from_seqs(["AAAAAAAAAA"])
            result = mutagenize_scan(bg, mutagenize_length=3, num_mutations=1)
            assert hasattr(result, "operation")

    def test_sequential_mode(self):
        with pp.Party():
            bg = pp.from_seq("AAAAAAAAAA")
            result = mutagenize_scan(
                bg, mutagenize_length=3, num_mutations=1,
                mode=("sequential", "sequential"),
            )
            df = result.generate_library(num_cycles=1)
            assert len(df) > 0

    def test_preserves_length(self):
        with pp.Party():
            bg = pp.from_seq("AAAAAAAAAA")
            result = mutagenize_scan(
                bg, mutagenize_length=3, num_mutations=1,
                mode=("sequential", "sequential"),
            )
            df = result.generate_library(num_cycles=1)
            for seq in df["seq"]:
                assert len(strip_all_tags(seq)) == 10

    def test_string_input(self):
        with pp.Party():
            result = mutagenize_scan(
                "AAAAAAAAAA", mutagenize_length=3, num_mutations=1,
            )
            df = result.generate_library(num_seqs=3)
            for seq in df["seq"]:
                assert len(strip_all_tags(seq)) == 10


class TestMutagenizeScanMutationParams:
    """Test num_mutations vs mutation_rate exclusivity."""

    def test_num_mutations_only(self):
        with pp.Party():
            result = mutagenize_scan(
                "AAAA", mutagenize_length=2, num_mutations=1,
                mode=("sequential", "sequential"),
            )
            df = result.generate_library()
            assert len(df) == result.num_states

    def test_mutation_rate_only(self):
        with pp.Party():
            result = mutagenize_scan(
                "AAAA", mutagenize_length=2, mutation_rate=0.5,
                mode=("sequential", "random"), num_states=(None, 3),
            )
            df = result.generate_library()
            assert len(df) == result.num_states

    def test_neither_raises(self):
        with pp.Party():
            with pytest.raises(ValueError, match="Either num_mutations or mutation_rate"):
                mutagenize_scan("AAAA", mutagenize_length=2)

    def test_both_raises(self):
        with pp.Party():
            with pytest.raises(ValueError, match="Only one of"):
                mutagenize_scan(
                    "AAAA", mutagenize_length=2,
                    num_mutations=1, mutation_rate=0.5,
                )


class TestMutagenizeScanTupleParams:
    """Test tuple parameter broadcasting for mode, num_states, prefix, iter_order."""

    def test_tuple_mode(self):
        with pp.Party():
            result = mutagenize_scan(
                "ACGTACGT", mutagenize_length=2, num_mutations=1,
                mode=("sequential", "random"), num_states=(None, 3),
            )
            df = result.generate_library()
            assert len(df) == result.num_states

    def test_scalar_mode_broadcast_warns(self):
        with pp.Party():
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                mutagenize_scan(
                    "ACGT", mutagenize_length=2, num_mutations=1,
                    mode="sequential",
                )
            broadcast_warns = [x for x in w if "broadcast" in str(x.message).lower()]
            assert len(broadcast_warns) >= 1

    def test_scalar_num_states_broadcast_warns(self):
        with pp.Party():
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                mutagenize_scan(
                    "ACGT", mutagenize_length=2, num_mutations=1,
                    mode=("random", "random"), num_states=3,
                )
            broadcast_warns = [x for x in w if "broadcast" in str(x.message).lower()]
            assert len(broadcast_warns) >= 1

    def test_tuple_num_states_no_warning(self):
        with pp.Party():
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                mutagenize_scan(
                    "ACGT", mutagenize_length=2, num_mutations=1,
                    mode=("random", "random"), num_states=(3, 2),
                )
            broadcast_warns = [x for x in w if "num_states" in str(x.message)]
            assert len(broadcast_warns) == 0

    def test_tuple_prefix(self):
        with pp.Party():
            result = mutagenize_scan(
                "ACGTACGT", mutagenize_length=2, num_mutations=1,
                mode=("sequential", "sequential"),
                prefix=("scan", "mut"),
            )
            df = result.generate_library()
            names = df["name"].tolist()
            assert any("scan" in n for n in names)
            assert any("mut" in n for n in names)

    def test_tuple_iter_order(self):
        with pp.Party():
            result = mutagenize_scan(
                "ACGTACGT", mutagenize_length=2, num_mutations=1,
                mode=("sequential", "sequential"),
                iter_order=(1.0, 2.0),
            )
            df = result.generate_library()
            assert len(df) == result.num_states

    def test_invalid_mode_length_raises(self):
        with pp.Party():
            with pytest.raises((ValueError, Exception)):
                mutagenize_scan(
                    "ACGT", mutagenize_length=2, num_mutations=1,
                    mode=("sequential", "random", "fixed"),
                )

    def test_invalid_num_states_length_raises(self):
        with pp.Party():
            with pytest.raises(ValueError, match="num_states must be a sequence of length 2"):
                mutagenize_scan(
                    "ACGT", mutagenize_length=2, num_mutations=1,
                    mode=("random", "random"), num_states=(1, 2, 3),
                )

    def test_num_states_with_none_element(self):
        """num_states=(3, None) — 3 random scan positions, auto-compute mutagenize."""
        with pp.Party():
            result = mutagenize_scan(
                "ACGTACGT", mutagenize_length=2, num_mutations=1,
                mode=("random", "sequential"), num_states=(3, None),
            )
            df = result.generate_library()
            assert len(df) == result.num_states


class TestMutagenizeScanRegion:
    """Test region constraint support."""

    def test_named_region(self):
        with pp.Party():
            pool = pp.from_seq("AAAA<core>CCCCCC</core>AAAA")
            result = mutagenize_scan(
                pool, mutagenize_length=2, num_mutations=1,
                region="core", mode=("sequential", "sequential"),
            )
            df = result.generate_library()
            for seq in df["seq"]:
                clean = strip_all_tags(seq)
                assert clean.startswith("AAAA")
                assert clean.endswith("AAAA")

    def test_interval_region(self):
        with pp.Party():
            pool = pp.from_seq("AAACCCAAA")
            result = mutagenize_scan(
                pool, mutagenize_length=2, num_mutations=1,
                region=[3, 6], mode=("sequential", "sequential"),
            )
            df = result.generate_library()
            for seq in df["seq"]:
                clean = strip_all_tags(seq)
                assert clean[:3] == "AAA"
                assert clean[-3:] == "AAA"


class TestMutagenizeScanPositions:
    """Test position constraint support."""

    def test_explicit_positions(self):
        with pp.Party():
            result = mutagenize_scan(
                "ACGTACGT", mutagenize_length=2, num_mutations=1,
                positions=[0, 2, 4], mode=("sequential", "sequential"),
                cards=(["start"], None),
            )
            df = result.generate_library()
            start_col = [c for c in df.columns if c.endswith(".start")][0]
            positions_seen = sorted(set(df[start_col].tolist()))
            assert positions_seen == [0, 2, 4]

    def test_slice_positions(self):
        with pp.Party():
            result = mutagenize_scan(
                "ACGTACGT", mutagenize_length=2, num_mutations=1,
                positions=slice(0, 3), mode=("sequential", "sequential"),
            )
            df = result.generate_library()
            assert len(df) == result.num_states


class TestMutagenizeScanCards:
    """Test design card output."""

    def test_scan_cards(self):
        with pp.Party():
            result = mutagenize_scan(
                "ACGT", mutagenize_length=2, num_mutations=1,
                mode=("sequential", "sequential"),
                cards=(["start", "end"], None),
            )
            df = result.generate_library()
            start_cols = [c for c in df.columns if c.endswith(".start")]
            end_cols = [c for c in df.columns if c.endswith(".end")]
            assert len(start_cols) == 1
            assert len(end_cols) == 1

    def test_mutagenize_cards(self):
        with pp.Party():
            result = mutagenize_scan(
                "ACGT", mutagenize_length=2, num_mutations=1,
                mode=("sequential", "sequential"),
                cards=(None, ["positions"]),
            )
            df = result.generate_library()
            pos_cols = [c for c in df.columns if c.endswith(".positions")]
            assert len(pos_cols) == 1

    def test_both_cards(self):
        with pp.Party():
            result = mutagenize_scan(
                "ACGT", mutagenize_length=2, num_mutations=1,
                mode=("sequential", "sequential"),
                cards=(["start"], ["positions"]),
            )
            df = result.generate_library()
            start_cols = [c for c in df.columns if c.endswith(".start")]
            pos_cols = [c for c in df.columns if c.endswith(".positions")]
            assert len(start_cols) == 1
            assert len(pos_cols) == 1


class TestMutagenizeScanDeterminism:
    """Test determinism with seeded RNG."""

    def test_same_seed_same_output(self):
        with pp.Party():
            result = mutagenize_scan(
                "ACGTACGT", mutagenize_length=3, num_mutations=1,
                mode=("random", "random"), num_states=(5, 3),
            )
        df1 = result.generate_library(seed=42, init_state=0)
        df2 = result.generate_library(seed=42, init_state=0)
        assert df1["seq"].tolist() == df2["seq"].tolist()

    def test_different_seed_different_output(self):
        with pp.Party():
            result = mutagenize_scan(
                "ACGTACGT", mutagenize_length=3, num_mutations=1,
                mode=("random", "random"), num_states=(5, 3),
            )
        df1 = result.generate_library(seed=42, init_state=0)
        df2 = result.generate_library(seed=99, init_state=0)
        assert df1["seq"].tolist() != df2["seq"].tolist()


class TestMutagenizeScanStyle:
    """Test style parameter."""

    def test_style_accepted(self):
        with pp.Party():
            result = mutagenize_scan(
                "AAAAAAAAAA", mutagenize_length=3, num_mutations=1,
                style="red", mode=("sequential", "sequential"),
            )
            df = result.generate_library()
            assert len(df) > 0

    def test_style_none_default(self):
        with pp.Party():
            result = mutagenize_scan(
                "AAAAAAAAAA", mutagenize_length=3, num_mutations=1,
                style=None, mode=("sequential", "sequential"),
            )
            df = result.generate_library()
            assert len(df) > 0


class TestMutagenizeScanMarkerRemoval:
    """Test that internal _mut marker is cleaned from output."""

    def test_no_mut_marker_in_output(self):
        with pp.Party():
            result = mutagenize_scan(
                "AAAAAAAAAA", mutagenize_length=3, num_mutations=1,
                mode=("sequential", "sequential"),
            )
            df = result.generate_library()
            for seq in df["seq"]:
                assert "<_mut>" not in seq
                assert "</_mut>" not in seq


class TestMutagenizeScanEdgeCases:
    """Test boundary conditions and edge cases."""

    def test_mutagenize_length_1(self):
        """Minimum window size: single-base mutagenesis at each position."""
        with pp.Party():
            result = mutagenize_scan(
                "AAAA", mutagenize_length=1, num_mutations=1,
                mode=("sequential", "sequential"),
            )
            df = result.generate_library()
            assert len(df) == result.num_states
            for seq in df["seq"]:
                clean = strip_all_tags(seq)
                assert len(clean) == 4
                diffs = sum(1 for a, b in zip(clean, "AAAA") if a != b)
                assert diffs <= 1

    def test_mutagenize_length_equals_seq_length(self):
        """Window covers entire sequence — only 1 scan position."""
        with pp.Party():
            result = mutagenize_scan(
                "ACGT", mutagenize_length=4, num_mutations=1,
                mode=("sequential", "sequential"),
            )
            df = result.generate_library()
            assert len(df) == result.num_states
            for seq in df["seq"]:
                assert len(strip_all_tags(seq)) == 4

    def test_num_mutations_equals_mutagenize_length(self):
        """Mutate every position in the window."""
        with pp.Party():
            result = mutagenize_scan(
                "AAAA", mutagenize_length=2, num_mutations=2,
                mode=("sequential", "random"), num_states=(None, 5),
            )
            df = result.generate_library()
            assert len(df) == result.num_states

    def test_chained_with_from_seqs(self):
        """mutagenize_scan on multi-state parent produces Cartesian product."""
        with pp.Party():
            bg = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = mutagenize_scan(
                    bg, mutagenize_length=2, num_mutations=1,
                    mode=("sequential", "sequential"),
                )
            df = result.generate_library()
            assert len(df) == result.num_states
            assert result.num_states % 2 == 0

    def test_seq_length_preserved(self):
        """mutagenize_scan should preserve seq_length metadata."""
        with pp.Party():
            result = mutagenize_scan(
                "ACGTACGT", mutagenize_length=3, num_mutations=1,
                mode=("sequential", "sequential"),
            )
            assert result.seq_length == 8

    def test_mixin_forwarding(self):
        """pool.mutagenize_scan(...) matches factory call."""
        with pp.Party():
            bg = pp.from_seq("ACGTACGT")
            result = bg.mutagenize_scan(
                mutagenize_length=2, num_mutations=1,
                mode=("sequential", "sequential"),
            )
            df = result.generate_library()
            assert len(df) == result.num_states

    def test_factory_name_threaded(self):
        """_factory_name threads through sub-ops."""
        with pp.Party():
            result = mutagenize_scan(
                "ACGTACGT", mutagenize_length=2, num_mutations=1,
                mode=("sequential", "sequential"),
                _factory_name="my_mut_scan",
            )
            df = result.generate_library()
            assert len(df) == result.num_states

    def test_sequential_variable_length_raises(self):
        """Sequential mode with variable-length parent raises ValueError."""
        with pp.Party():
            bg = pp.from_seqs(["ACGT", "ACGTAC"])
            with pytest.raises(ValueError, match="sequential"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mutagenize_scan(
                        bg, mutagenize_length=2, num_mutations=1,
                        mode=("sequential", "sequential"),
                    )
