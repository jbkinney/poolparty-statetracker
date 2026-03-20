"""Tests for the FromIupac operation."""

import pytest

import poolparty as pp
from poolparty.base_ops.from_iupac import FromIupacOp, from_iupac


class TestFromIupacFactory:
    """Test from_iupac factory function."""

    def test_returns_pool(self):
        """from_iupac returns a Pool object."""
        with pp.Party() as party:
            pool = from_iupac("ACGT")
            assert pool is not None
            assert hasattr(pool, "operation")

    def test_creates_from_iupac_op(self):
        """Pool's operation is FromIupacOp."""
        with pp.Party() as party:
            pool = from_iupac("ACGT")
            assert isinstance(pool.operation, FromIupacOp)

    def test_works_with_default_party(self):
        """from_iupac works with default party context (no explicit context needed)."""
        pool = from_iupac("ACGT")
        assert pool is not None
        assert hasattr(pool, "operation")


class TestFromIupacValidation:
    """Test parameter validation."""

    def test_empty_string_error(self):
        """Empty iupac_seq raises ValueError."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="non-empty string"):
                from_iupac("")

    def test_invalid_char_error(self):
        """Invalid IUPAC character raises ValueError."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="invalid IUPAC character"):
                from_iupac("ACGTX")


class TestFromIupacSequentialMode:
    """Test sequential mode."""

    def test_sequential_enumeration(self):
        """Sequential mode enumerates all possibilities."""
        with pp.Party() as party:
            pool = from_iupac("RY", mode="sequential").named("iupac")

        df = pool.generate_library(num_cycles=1)
        # R = A|G, Y = C|T -> 2*2 = 4 sequences
        assert len(df) == 4
        seqs = set(df["seq"])
        assert seqs == {"AC", "AT", "GC", "GT"}

    def test_num_states_computation(self):
        """num_states equals product of possibilities."""
        with pp.Party() as party:
            # N = 4 options, so NN = 16 states
            pool = from_iupac("NN", mode="sequential")
            assert pool.operation.num_states == 16


class TestFromIupacRandomMode:
    """Test random mode."""

    def test_random_num_states_is_one(self):
        """Random mode has num_states=1."""
        with pp.Party() as party:
            pool = from_iupac("ACGT", mode="random")
            assert pool.operation.num_states == 1

    def test_random_sampling(self):
        """Random mode produces valid DNA sequences."""
        with pp.Party() as party:
            pool = from_iupac("NNNN", mode="random").named("iupac")

        df = pool.generate_library(num_seqs=100, seed=42)
        assert len(df) == 100
        for seq in df["seq"]:
            assert len(seq) == 4
            assert all(c in "ACGT" for c in seq)

    def test_deterministic_with_seed(self):
        """Same seed produces same results."""
        with pp.Party() as party:
            pool1 = from_iupac("NNNN", mode="random").named("iupac")
        df1 = pool1.generate_library(num_seqs=10, seed=42)

        with pp.Party() as party:
            pool2 = from_iupac("NNNN", mode="random").named("iupac")
        df2 = pool2.generate_library(num_seqs=10, seed=42)

        assert list(df1["seq"]) == list(df2["seq"])


class TestFromIupacCustomName:
    """Test name parameters."""

    def test_default_operation_name(self):
        """Default operation name contains from_iupac."""
        with pp.Party() as party:
            pool = from_iupac("ACGT")
            assert pool.operation.name.startswith("op[")
            assert ":from_iupac" in pool.operation.name

    def test_custom_operation_name(self):
        """Custom operation name."""
        with pp.Party() as party:
            pool = from_iupac("ACGT").named("my_motif")
            assert pool.name == "my_motif"

    def test_custom_pool_name(self):
        """Custom pool name."""
        with pp.Party() as party:
            pool = from_iupac("ACGT").named("my_pool")
            assert pool.name == "my_pool"


class TestFromIupacDesignCards:
    """Test design card output."""

    def test_iupac_state_in_output(self):
        """Design card contains iupac_state when requested via cards parameter."""
        with pp.Party() as party:
            pool = from_iupac("ACGT", cards=["iupac_state"]).named("mypool")

        df = pool.generate_library(num_seqs=1, seed=42)
        # Check for iupac_state in design card columns (operation name is auto-generated)
        iupac_cols = [c for c in df.columns if "iupac_state" in c]
        assert len(iupac_cols) > 0

    def test_design_card_keys_defined(self):
        """design_card_keys is defined correctly."""
        with pp.Party() as party:
            pool = from_iupac("ACGT")
            assert "iupac_state" in pool.operation.design_card_keys


class TestFromIupacSeqLength:
    """Test sequence length computation."""

    def test_seq_length_simple(self):
        """seq_length equals IUPAC sequence length."""
        with pp.Party() as party:
            pool = from_iupac("ACGT")
            assert pool.operation.seq_length == 4

    def test_seq_length_with_degenerate(self):
        """seq_length works with degenerate positions."""
        with pp.Party() as party:
            pool = from_iupac("ANNN")
            assert pool.operation.seq_length == 4


class TestFromIupacIgnoreChars:
    """Test handling of ignore characters."""

    def test_dot_separator_preserved(self):
        """Dot separator is preserved in output."""
        with pp.Party() as party:
            pool = from_iupac("ACG.TNN", mode="random").named("iupac")

        df = pool.generate_library(num_seqs=5, seed=42)
        for seq in df["seq"]:
            # Dot should be at position 3
            assert seq[3] == "."
            assert len(seq) == 7

    def test_multiple_separators(self):
        """Multiple dot separators are preserved."""
        with pp.Party() as party:
            pool = from_iupac("A.C.G.T", mode="random").named("iupac")

        df = pool.generate_library(num_seqs=5, seed=42)
        for seq in df["seq"]:
            assert seq == "A.C.G.T"

    def test_dash_separator_preserved(self):
        """Dash separator is preserved in output."""
        with pp.Party() as party:
            pool = from_iupac("ACG-TNN", mode="random").named("iupac")

        df = pool.generate_library(num_seqs=5, seed=42)
        for seq in df["seq"]:
            assert seq[3] == "-"

    def test_ignore_chars_not_degenerate(self):
        """Ignore characters are not treated as degenerate."""
        with pp.Party() as party:
            pool = from_iupac("A.N", mode="random").named("iupac")

        df = pool.generate_library(num_seqs=5, seed=42)
        for seq in df["seq"]:
            assert seq[0] == "A"
            assert seq[1] == "."
            assert seq[2].isupper()

    def test_num_states_ignores_separators(self):
        """Separators don't affect num_states calculation."""
        with pp.Party() as party:
            # N = 4 options, separators = 1 option each
            pool = from_iupac("N.N", mode="sequential")
            # 4 * 1 * 4 = 16 states
            assert pool.operation.num_states == 16


class TestFromIupacIgnoreCharsExtended:
    """Test handling of ignore characters."""

    def test_dot_separator_allowed(self):
        """Dot separator is allowed in IUPAC sequences."""
        with pp.Party() as party:
            pool = from_iupac("AC.GT", mode="sequential").named("iupac")

        df = pool.generate_library(num_cycles=1)
        assert len(df) == 1  # Only one state since all positions are fixed
        assert df["seq"].iloc[0] == "AC.GT"

    def test_hyphen_separator_allowed(self):
        """Hyphen separator is allowed in IUPAC sequences."""
        with pp.Party() as party:
            pool = from_iupac("AC-GT", mode="sequential").named("iupac")

        df = pool.generate_library(num_cycles=1)
        assert df["seq"].iloc[0] == "AC-GT"

    def test_space_separator_allowed(self):
        """Space separator is allowed in IUPAC sequences."""
        with pp.Party() as party:
            pool = from_iupac("AC GT", mode="sequential").named("iupac")

        df = pool.generate_library(num_cycles=1)
        assert df["seq"].iloc[0] == "AC GT"

    def test_ignore_chars_with_degenerate(self):
        """Ignore characters work alongside degenerate positions."""
        with pp.Party() as party:
            pool = from_iupac("A.N.T", mode="sequential").named("iupac")

        df = pool.generate_library(num_cycles=1)
        # N has 4 options, so 4 states total
        assert len(df) == 4
        # Check that dots are preserved
        for seq in df["seq"]:
            assert seq[1] == "."
            assert seq[3] == "."

    def test_separators_preserved(self):
        """Separators preserved in output."""
        with pp.Party() as party:
            pool = from_iupac("A.N.T", mode="random").named("iupac")

        df = pool.generate_library(num_seqs=10, seed=42)
        for seq in df["seq"]:
            # Separators should remain unchanged
            assert seq[1] == "."
            assert seq[3] == "."
            assert seq[0] == "A"
            assert seq[4] == "T"
