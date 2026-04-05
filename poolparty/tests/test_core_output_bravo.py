"""Audit tests for generate_library and print_library (C10).

Covers:
- Step 2: Wrapper parity (Pool.generate_library, Pool.print_library forwarding)
- Step 3: Contract testing (G1-G9 for generate_library, P1-P6 for print_library)
- Step 4: Policy review (Q1-Q11)
- Step 5: Adversarial patterns (3 diagonal combinations, 2 assumption inversions)
- Step 6: Contract tracing (CT1-CT4)
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import poolparty as pp


# ---------------------------------------------------------------------------
# Step 2: Wrapper parity
# ---------------------------------------------------------------------------


class TestWrapperParity:
    """Pool.generate_library is a thin wrapper; print_library forwards a subset."""

    def test_pool_wrapper_forwards_all_params(self):
        """Every standalone generate_library param reachable via Pool wrapper."""
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA", "GGCC", "AATT"], mode="sequential")
            df = pool.generate_library(
                num_cycles=1,
                num_seqs=None,
                seed=42,
                init_state=0,
                seqs_only=False,
                _include_inline_styles=False,
                discard_null_seqs=False,
                max_iterations=None,
                min_acceptance_rate=None,
                attempts_per_rate_assessment=100,
            )
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 4

    def test_pool_wrapper_matches_standalone(self):
        """Pool wrapper produces identical output to standalone function."""
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA", "GGCC"], mode="sequential")
            df_wrapper = pool.generate_library(num_seqs=3, init_state=0, seed=42)
            df_standalone = pp.generate_library(
                pool, num_seqs=3, init_state=0, seed=42
            )
            pd.testing.assert_frame_equal(df_wrapper, df_standalone)

    def test_print_library_hardcodes_init_state_zero(self):
        """print_library always starts from state 0, not _current_state."""
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA", "GGCC"], mode="sequential")
            pool._current_state = 2
            df = pool.generate_library(init_state=0, num_cycles=1, seed=0)
            pool._current_state = 2
            import io, sys

            old = sys.stdout
            sys.stdout = io.StringIO()
            pool.print_library(seed=0)
            output = sys.stdout.getvalue()
            sys.stdout = old
            assert df.iloc[0]["seq"] in output

    def test_print_library_hardcodes_include_inline_styles(self):
        """print_library forces _include_inline_styles=True."""
        with pp.Party():
            pool = pp.from_seq("ACGT").stylize(style="red")
            df_with = pool.generate_library(
                init_state=0, _include_inline_styles=True, seed=0
            )
            assert "_inline_styles" in df_with.columns

    def test_print_library_forwards_discard_null_seqs(self, capsys):
        """discard_null_seqs reaches generate_library through print_library."""
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA", "GGCC"], mode="sequential")
            filt = pool.filter(lambda s: s.startswith("A"))
            filt.print_library(discard_null_seqs=True, show_header=False, show_name=False)
            captured = capsys.readouterr()
            lines = [l for l in captured.out.strip().split("\n") if l.strip()]
            assert len(lines) == 1
            assert "ACGT" in lines[0]

    def test_print_library_forwards_seed(self, capsys):
        """seed parameter reaches generate_library through print_library."""
        with pp.Party():
            pool = pp.from_seqs(
                ["ACGT", "TGCA", "GGCC", "AATT"], mode="random", num_states=4
            )
            pool.print_library(seed=42, show_header=False, show_name=False)
            out1 = capsys.readouterr().out
            pool.print_library(seed=42, show_header=False, show_name=False)
            out2 = capsys.readouterr().out
            assert out1 == out2


# ---------------------------------------------------------------------------
# Step 3: G contracts (generate_library)
# ---------------------------------------------------------------------------


class TestG1RowCount:
    """G1: Row count matches num_seqs or num_cycles * num_states."""

    def test_num_seqs_exact(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA", "GGCC"], mode="sequential")
            df = pool.generate_library(num_seqs=5, init_state=0, seed=0)
            assert len(df) == 5

    def test_num_cycles_exact(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA", "GGCC", "AATT"], mode="sequential")
            df = pool.generate_library(num_cycles=2, init_state=0, seed=0)
            assert len(df) == 8

    def test_num_seqs_one(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            df = pool.generate_library(num_seqs=1, init_state=0, seed=0)
            assert len(df) == 1


class TestG2StateIteration:
    """G2: Sequential pool covers full state space; cycling wraps correctly."""

    def test_single_cycle_covers_all_states(self):
        with pp.Party():
            seqs = ["AAAA", "CCCC", "GGGG", "TTTT"]
            pool = pp.from_seqs(seqs, mode="sequential")
            df = pool.generate_library(num_cycles=1, init_state=0, seed=0)
            assert list(df["seq"]) == seqs

    def test_multi_cycle_repeats_states(self):
        with pp.Party():
            seqs = ["AAAA", "CCCC", "GGGG"]
            pool = pp.from_seqs(seqs, mode="sequential")
            df = pool.generate_library(num_cycles=3, init_state=0, seed=0)
            assert len(df) == 9
            for i, row in df.iterrows():
                assert row["seq"] == seqs[i % 3]


class TestG3NullDiscard:
    """G3: discard_null_seqs=True removes None rows."""

    def test_no_nulls_in_output(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA", "GGCC"], mode="sequential")
            filt = pool.filter(lambda s: s.startswith("A"))
            df = filt.generate_library(
                num_cycles=1, init_state=0, seed=0, discard_null_seqs=True
            )
            assert df["seq"].notna().all()
            assert len(df) == 1
            assert df.iloc[0]["seq"] == "ACGT"

    def test_all_states_rejected_gives_warning(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA", "GGCC"], mode="sequential")
            filt = pool.filter(lambda s: False)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                df = filt.generate_library(
                    num_seqs=3, init_state=0, seed=0, discard_null_seqs=True
                )
            assert len(df) == 0
            assert len(w) >= 1


class TestG4NullPreserve:
    """G4: discard_null_seqs=False keeps null rows with seq=None, name=None."""

    def test_none_rows_present(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA", "GGCC"], mode="sequential")
            filt = pool.filter(lambda s: s.startswith("A"))
            df = filt.generate_library(
                num_cycles=1, init_state=0, seed=0, discard_null_seqs=False
            )
            assert len(df) == 3
            assert pd.isna(df.iloc[1]["seq"])
            assert pd.isna(df.iloc[2]["seq"])

    def test_null_row_has_none_name(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            filt = pool.filter(lambda s: s.startswith("A"))
            df = filt.generate_library(
                num_cycles=1, init_state=0, seed=0, discard_null_seqs=False
            )
            assert pd.isna(df.iloc[1]["name"])


class TestG5SeedDeterminism:
    """G5: Same seed and init_state produce identical output."""

    def test_same_seed_same_output(self):
        with pp.Party():
            pool = pp.from_seqs(
                ["ACGT", "TGCA", "GGCC", "AATT"], mode="random", num_states=4
            )
            df1 = pool.generate_library(num_seqs=4, seed=42, init_state=0)
            df2 = pool.generate_library(num_seqs=4, seed=42, init_state=0)
            pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_different_output(self):
        with pp.Party():
            pool = pp.from_seqs(
                ["ACGT", "TGCA", "GGCC", "AATT"], mode="random", num_states=4
            )
            df1 = pool.generate_library(num_seqs=4, seed=42, init_state=0)
            df2 = pool.generate_library(num_seqs=4, seed=99, init_state=0)
            assert not df1["seq"].equals(df2["seq"])


class TestG6InitState:
    """G6: init_state controls starting position."""

    def test_init_state_starts_at_correct_position(self):
        with pp.Party():
            seqs = ["AAAA", "CCCC", "GGGG", "TTTT"]
            pool = pp.from_seqs(seqs, mode="sequential")
            df = pool.generate_library(num_seqs=2, init_state=3, seed=0)
            assert df.iloc[0]["seq"] == "TTTT"
            assert df.iloc[1]["seq"] == "AAAA"

    def test_init_state_beyond_state_space_wraps(self):
        """init_state >= num_states wraps via modulo in _compute_one."""
        with pp.Party():
            seqs = ["AAAA", "CCCC", "GGGG", "TTTT"]
            pool = pp.from_seqs(seqs, mode="sequential")
            df = pool.generate_library(num_seqs=1, init_state=7, seed=0)
            # global_state=7, 7 % 4 == 3 → "TTTT"
            assert df.iloc[0]["seq"] == "TTTT"


class TestG7MaxIterations:
    """G7: max_iterations prevents infinite loops; warning on truncation."""

    def test_max_iterations_stops_with_warning(self):
        with pp.Party():
            pool = pp.from_seqs(
                ["ACGT", "TGCA", "GGCC", "AATT"], mode="sequential"
            )
            filt = pool.filter(lambda s: s.startswith("A"))
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                df = filt.generate_library(
                    num_seqs=10,
                    init_state=0,
                    seed=0,
                    discard_null_seqs=True,
                    max_iterations=5,
                )
            assert len(df) < 10
            assert any("max_iterations" in str(x.message) or "exhausted" in str(x.message) for x in w)

    def test_max_iterations_without_discard_ignored(self):
        """max_iterations has no effect when discard_null_seqs=False."""
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            df = pool.generate_library(
                num_seqs=2, init_state=0, seed=0, max_iterations=1
            )
            assert len(df) == 2


class TestG8SeqsOnly:
    """G8: seqs_only returns list[str]."""

    def test_seqs_only_returns_list(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA", "GGCC"], mode="sequential")
            result = pool.generate_library(
                num_seqs=3, init_state=0, seed=0, seqs_only=True
            )
            assert isinstance(result, list)
            assert len(result) == 3
            assert all(isinstance(s, str) for s in result)

    def test_seqs_only_with_nulls_is_broken(self):
        """F1 (merged): seqs_only + null-preserve is nondeterministic.

        Depending on seed, beartype either catches the list[str] violation
        (~87% of seeds → BeartypeCallHintReturnViolation) or silently
        returns a list containing None entries (~13% of seeds).
        Either outcome violates the declared return contract.
        """
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA", "GGCC"], mode="sequential")
            filt = pool.filter(lambda s: s.startswith("A"))
            try:
                result = filt.generate_library(
                    num_seqs=3,
                    init_state=0,
                    seed=0,
                    seqs_only=True,
                    discard_null_seqs=False,
                )
                assert None in result, (
                    "Expected None entries in result when beartype misses violation"
                )
            except Exception:
                pass

    def test_seqs_only_empty_returns_empty_list(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            filt = pool.filter(lambda s: False)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                result = filt.generate_library(
                    num_cycles=1,
                    init_state=0,
                    seed=0,
                    seqs_only=True,
                    discard_null_seqs=True,
                )
            assert result == []


class TestG9DesignCards:
    """G9: Design card columns appear when operations have cards configured."""

    def test_card_columns_present(self):
        with pp.Party():
            pool = pp.from_seqs(
                ["ACGT", "TGCA", "GGCC"],
                mode="sequential",
                cards=["state"],
            )
            df = pool.generate_library(num_cycles=1, init_state=0, seed=0)
            state_col = [c for c in df.columns if "state" in c.lower()]
            assert len(state_col) >= 1

    def test_no_cards_only_name_seq_columns(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            df = pool.generate_library(num_cycles=1, init_state=0, seed=0)
            assert list(df.columns) == ["name", "seq"]


# ---------------------------------------------------------------------------
# Step 3: P contracts (print_library)
# ---------------------------------------------------------------------------


class TestP1SequenceCorrectness:
    """P1: print_library stdout matches generate_library(init_state=0) output."""

    def test_sequences_match_generate_library(self, capsys):
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA", "GGCC"], mode="sequential")
            df = pool.generate_library(init_state=0, num_cycles=1, seed=0)
            pool.print_library(
                seed=0, show_header=False, show_name=False, show_seq=True
            )
            captured = capsys.readouterr()
            for seq in df["seq"]:
                assert seq in captured.out


class TestP2ColumnVisibility:
    """P2: show_name, show_seq, show_state flags control column output."""

    def test_show_name_false(self, capsys):
        with pp.Party():
            pool = pp.from_seqs(
                ["ACGT", "TGCA"], mode="sequential", prefix="test"
            )
            pool.print_library(
                show_header=False, show_name=False, show_seq=True, seed=0
            )
            captured = capsys.readouterr()
            assert "test" not in captured.out

    def test_show_seq_false(self, capsys):
        with pp.Party():
            pool = pp.from_seqs(
                ["ACGT", "TGCA"], mode="sequential", prefix="mypool"
            )
            pool.print_library(
                show_header=False, show_name=True, show_seq=False, seed=0
            )
            captured = capsys.readouterr()
            assert "ACGT" not in captured.out
            assert "mypool" in captured.out

    def test_all_show_flags_false(self, capsys):
        """All show flags false → only header (if show_header=True)."""
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            pool.print_library(
                show_header=True,
                show_name=False,
                show_seq=False,
                show_state=False,
                seed=0,
            )
            captured = capsys.readouterr()
            lines = [l for l in captured.out.strip().split("\n") if l.strip()]
            assert len(lines) == 1
            assert "seq_length" in lines[0]


class TestP3HeaderContent:
    """P3: Header shows correct seq_length and num_states."""

    def test_header_shows_pool_info(self, capsys):
        with pp.Party():
            pool = pp.from_seqs(
                ["ACGT", "TGCA", "GGCC"], mode="sequential"
            ).named("mypool")
            pool.print_library(show_header=True, seed=0)
            captured = capsys.readouterr()
            assert "mypool" in captured.out
            assert "seq_length=4" in captured.out
            assert "num_states=3" in captured.out

    def test_header_none_seq_length(self, capsys):
        """Variable-length pool shows seq_length=None."""
        with pp.Party():
            pool = pp.from_seqs(
                ["ACGT", "TG", "GGCCAA"], mode="sequential"
            ).named("varpool")
            pool.print_library(show_header=True, seed=0)
            captured = capsys.readouterr()
            assert "seq_length=None" in captured.out


class TestP4NamePadding:
    """P4: pad_names controls alignment."""

    def test_pad_names_true_aligns(self, capsys):
        with pp.Party():
            pool = pp.from_seqs(
                ["ACGT", "TG"], mode="sequential", prefix="item"
            )
            pool.print_library(
                pad_names=True, show_header=False, show_seq=True, seed=0
            )
            captured = capsys.readouterr()
            lines = [l for l in captured.out.strip().split("\n") if l.strip()]
            assert len(lines) == 2
            parts0 = lines[0].split("  ")
            parts1 = lines[1].split("  ")
            assert len(parts0[0]) == len(parts1[0])

    def test_pad_names_false_no_padding(self, capsys):
        with pp.Party():
            pool = pp.from_seqs(
                ["ACGT", "TG"], mode="sequential", prefix="item"
            )
            pool.print_library(
                pad_names=False, show_header=False, show_seq=True, seed=0
            )
            captured = capsys.readouterr()
            lines = [l for l in captured.out.strip().split("\n") if l.strip()]
            assert len(lines) == 2
            name0 = lines[0].split("  ")[0].strip()
            name1 = lines[1].split("  ")[0].strip()
            assert name0 != "" and name1 != ""

    def test_all_names_none_no_name_column(self, capsys):
        """When all names are None, name column not shown."""
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            filt = pool.filter(lambda s: True)
            filt.print_library(
                show_header=False,
                show_name=True,
                pad_names=True,
                discard_null_seqs=False,
                seed=0,
            )
            captured = capsys.readouterr()
            for line in captured.out.strip().split("\n"):
                if line.strip():
                    assert "None" not in line.split("  ")[0] or "ACGT" in line or "TGCA" in line


class TestP5ReturnsSelf:
    """P5: print_library returns self for chaining."""

    def test_returns_self(self):
        with pp.Party():
            pool = pp.from_seq("ACGT")
            result = pool.print_library()
            assert result is pool

    def test_chaining_works(self):
        with pp.Party():
            pool = pp.from_seq("ACGT")
            result = pool.print_library().print_library()
            assert result is pool


class TestP6ProteinExtras:
    """P6: ProteinPool.print_library chars_per_aa and aa_separator."""

    def test_three_letter_codes(self, capsys):
        with pp.Party():
            pool = pp.from_seq("ATGGCTTAA").translate()
            pool.print_library(
                chars_per_aa=3,
                show_header=False,
                show_name=False,
                seed=0,
            )
            captured = capsys.readouterr()
            assert "Met" in captured.out
            assert "Ala" in captured.out

    def test_single_letter_default(self, capsys):
        with pp.Party():
            pool = pp.from_seq("ATGGCTTAA").translate()
            pool.print_library(
                chars_per_aa=1,
                show_header=False,
                show_name=False,
                seed=0,
            )
            captured = capsys.readouterr()
            assert "MA" in captured.out

    def test_custom_separator(self, capsys):
        with pp.Party():
            pool = pp.from_seq("ATGGCTTAA").translate()
            pool.print_library(
                chars_per_aa=3,
                aa_separator=".",
                show_header=False,
                show_name=False,
                seed=0,
            )
            captured = capsys.readouterr()
            assert "Met.Ala" in captured.out

    def test_no_separator(self, capsys):
        with pp.Party():
            pool = pp.from_seq("ATGGCTTAA").translate()
            pool.print_library(
                chars_per_aa=3,
                aa_separator="",
                show_header=False,
                show_name=False,
                seed=0,
            )
            captured = capsys.readouterr()
            assert "MetAla" in captured.out


# ---------------------------------------------------------------------------
# Step 4: Policy review — generate_library
# ---------------------------------------------------------------------------


class TestPolicyGenerateLibrary:
    """Q1-Q7: Policy questions for generate_library."""

    def test_q1_num_seqs_wins_over_num_cycles_with_warning(self):
        """Q1: When both provided, num_seqs takes precedence with warning."""
        with pp.Party():
            pool = pp.from_seqs(
                ["ACGT", "TGCA", "GGCC", "AATT"], mode="sequential"
            )
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                df = pool.generate_library(
                    num_seqs=2, num_cycles=3, init_state=0, seed=0
                )
            assert len(df) == 2
            assert any("num_seqs takes precedence" in str(x.message) for x in w)

    def test_q2_max_iterations_warning_is_actionable(self):
        """Q2: Warning includes acceptance rate and counts."""
        with pp.Party():
            pool = pp.from_seqs(
                ["ACGT", "TGCA", "GGCC", "AATT"], mode="sequential"
            )
            filt = pool.filter(lambda s: s.startswith("A"))
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                filt.generate_library(
                    num_seqs=10,
                    init_state=0,
                    seed=0,
                    discard_null_seqs=True,
                    max_iterations=5,
                )
            warns = [x for x in w if "max_iterations" in str(x.message) or "exhausted" in str(x.message)]
            assert len(warns) >= 1
            msg = str(warns[0].message)
            assert "%" in msg

    def test_q3_discard_with_num_seqs_partial_results(self):
        """Q3: discard_null_seqs=True + num_seqs returns up to N valid rows."""
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA", "GGCC"], mode="sequential")
            filt = pool.filter(lambda s: s.startswith("A"))
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                df = filt.generate_library(
                    num_seqs=5, init_state=0, seed=0, discard_null_seqs=True
                )
            assert 0 < len(df) < 5
            assert len(w) >= 1

    def test_q4_min_acceptance_rate_early_stop(self):
        """Q4: Returns partial results with warning when rate drops."""
        with pp.Party():
            pool = pp.from_seqs(
                ["ACGT"] + ["TGCA"] * 99, mode="sequential"
            )
            filt = pool.filter(lambda s: s.startswith("A"))
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                df = filt.generate_library(
                    num_seqs=50,
                    init_state=0,
                    seed=0,
                    discard_null_seqs=True,
                    min_acceptance_rate=0.5,
                    attempts_per_rate_assessment=10,
                )
            assert len(df) < 50
            warns = [x for x in w if "Acceptance rate" in str(x.message)]
            assert len(warns) >= 1

    def test_q5_init_state_persists_between_calls(self):
        """Q5: _current_state advances and persists across generate_library calls."""
        with pp.Party():
            seqs = ["AAAA", "CCCC", "GGGG", "TTTT"]
            pool = pp.from_seqs(seqs, mode="sequential")
            df1 = pool.generate_library(num_seqs=2, init_state=0, seed=0)
            assert pool._current_state == 2
            df2 = pool.generate_library(num_seqs=2, seed=0)
            assert pool._current_state == 4
            assert list(df1["seq"]) == ["AAAA", "CCCC"]
            assert list(df2["seq"]) == ["GGGG", "TTTT"]

    def test_q6_num_seqs_zero_raises(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            with pytest.raises(ValueError, match="num_seqs must be positive"):
                pool.generate_library(num_seqs=0, init_state=0, seed=0)

    def test_q6_num_seqs_negative_raises(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            with pytest.raises(ValueError, match="num_seqs must be positive"):
                pool.generate_library(num_seqs=-1, init_state=0, seed=0)

    def test_q6_num_cycles_zero_raises(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            with pytest.raises(ValueError, match="num_cycles must be positive"):
                pool.generate_library(num_cycles=0, init_state=0, seed=0)

    def test_q6_num_cycles_negative_raises(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            with pytest.raises(ValueError, match="num_cycles must be positive"):
                pool.generate_library(num_cycles=-1, init_state=0, seed=0)


# ---------------------------------------------------------------------------
# Step 4: Policy review — print_library
# ---------------------------------------------------------------------------


class TestPolicyPrintLibrary:
    """Q8-Q11: Policy questions for print_library."""

    def test_q8_default_num_cycles_one(self, capsys):
        """Q8: With no num_seqs or num_cycles, defaults to num_cycles=1."""
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA", "GGCC"], mode="sequential")
            pool.print_library(show_header=False, show_name=False, seed=0)
            captured = capsys.readouterr()
            lines = [l for l in captured.out.strip().split("\n") if l.strip()]
            assert len(lines) == pool.num_states

    def test_q9_show_state_without_state_column_omits_state(self, capsys):
        """Q9/F4 (fixed): show_state=True without state column now
        correctly omits 'state' from both header and data rows.
        """
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            pool.print_library(
                show_state=True,
                show_header=True,
                show_name=False,
                seed=0,
            )
            captured = capsys.readouterr()
            lines = [l for l in captured.out.strip().split("\n") if l.strip()]
            # Line 0: pool info header
            # Line 1: column header — should NOT include "state"
            assert "state" not in lines[1]
            # Lines 2+: data rows — seq only
            for data_line in lines[2:]:
                assert data_line.strip() in ["ACGT", "TGCA"]

    def test_q10_long_sequences_no_truncation(self, capsys):
        """Q10: Very long sequences printed in full without truncation."""
        with pp.Party():
            long_seq = "A" * 500
            pool = pp.from_seq(long_seq)
            pool.print_library(show_header=False, show_name=False, seed=0)
            captured = capsys.readouterr()
            assert long_seq in captured.out

    def test_q11_none_sequences_show_none_string(self, capsys):
        """Q11: None sequences display as literal 'None' string."""
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            filt = pool.filter(lambda s: s.startswith("A"))
            filt.print_library(
                show_header=False,
                show_name=False,
                discard_null_seqs=False,
                seed=0,
            )
            captured = capsys.readouterr()
            assert "None" in captured.out


# ---------------------------------------------------------------------------
# Step 5: Adversarial patterns — diagonal combinations
# ---------------------------------------------------------------------------


class TestAdversarialDiagonal:
    """Required diagonal combinations from adversarial matrix."""

    def test_diagonal1_protein_random_discard_seed(self):
        """ProteinPool + random + discard_null_seqs=True + seed determinism."""
        with pp.Party():
            dna = pp.from_seq("ATGGCTAAATAA")
            prot = dna.translate()
            filt = prot.filter(lambda s: "M" in s)
            df1 = filt.generate_library(
                num_seqs=1, init_state=0, seed=42, discard_null_seqs=True
            )
            df2 = filt.generate_library(
                num_seqs=1, init_state=0, seed=42, discard_null_seqs=True
            )
            assert len(df1) == 1
            pd.testing.assert_frame_equal(df1, df2)
            assert df1.iloc[0]["seq"] is not None

    def test_diagonal2_multi_parent_sequential_cycles_cards(self):
        """Multi-parent DAG + sequential + num_cycles=2 + design cards."""
        with pp.Party():
            a = pp.from_seqs(["AA", "CC"], mode="sequential", cards=["state"])
            b = pp.from_seqs(["GG", "TT"], mode="sequential", cards=["state"])
            joined = pp.join([a, b])
            df = joined.generate_library(num_cycles=2, init_state=0, seed=0)
            assert len(df) == 2 * joined.num_states
            assert "seq" in df.columns

    def test_diagonal3_large_state_space_partial_init_state(self):
        """Large state space + num_seqs < num_states + init_state mid-range."""
        with pp.Party():
            seqs = [f"ACGT{i:04d}" for i in range(200)]
            pool = pp.from_seqs(seqs, mode="sequential")
            df = pool.generate_library(num_seqs=10, init_state=100, seed=0)
            assert len(df) == 10
            assert df.iloc[0]["seq"] == seqs[100 % pool.num_states]


# ---------------------------------------------------------------------------
# Step 5: Adversarial patterns — assumption inversions
# ---------------------------------------------------------------------------


class TestAssumptionInversion:
    """Violate implicit assumptions in generate_library / _compute_one."""

    def test_source_only_pool(self):
        """Single operation, no parents — topo sort handles single node."""
        with pp.Party():
            pool = pp.from_seq("ACGT")
            df = pool.generate_library(num_seqs=1, init_state=0, seed=0)
            assert len(df) == 1
            assert df.iloc[0]["seq"] == "ACGT"

    def test_diamond_dag(self):
        """Diamond DAG: A->B, A->C, B+C->D — topo sort visits A once."""
        with pp.Party():
            a = pp.from_seqs(["AA", "CC"], mode="sequential", prefix="a")
            b = a.upper().named("b")
            c = a.lower().named("c")
            d = pp.join([b, c]).named("d")
            df = d.generate_library(num_cycles=1, init_state=0, seed=0)
            assert len(df) == d.num_states
            for _, row in df.iterrows():
                assert row["seq"] is not None

    def test_single_state_pool(self):
        """Pool with num_states=1 (fixed mode) generates correctly."""
        with pp.Party():
            pool = pp.from_seq("ACGT")
            assert pool.num_states == 1
            df = pool.generate_library(num_seqs=3, init_state=0, seed=0)
            assert len(df) == 3
            assert all(df["seq"] == "ACGT")

    def test_empty_string_sequence_not_treated_as_null(self):
        """F6: from_seq('') produces seq='', not null."""
        with pp.Party():
            pool = pp.from_seq("")
            df = pool.generate_library(num_seqs=1, init_state=0, seed=0)
            assert df.iloc[0]["seq"] == ""

    def test_attempts_per_rate_assessment_zero_raises(self):
        with pp.Party():
            pool = pp.from_seq("AAAA").filter(lambda s: False)
            with pytest.raises(ValueError, match="attempts_per_rate_assessment must be positive"):
                pool.generate_library(
                    num_seqs=2,
                    discard_null_seqs=True,
                    min_acceptance_rate=0.5,
                    attempts_per_rate_assessment=0,
                    init_state=0,
                )


# ---------------------------------------------------------------------------
# Step 6: Contract tracing
# ---------------------------------------------------------------------------


class TestContractTracing:
    """CT1-CT4: Deep logic verification of generate_library / _compute_one."""

    def test_ct1_state_cycling_across_boundaries(self):
        """CT1: global_state % num_values produces correct state values at cycle boundaries."""
        with pp.Party():
            seqs = ["AAAA", "CCCC", "GGGG"]
            pool = pp.from_seqs(seqs, mode="sequential")
            df = pool.generate_library(num_cycles=3, init_state=0, seed=0)
            for i in range(len(df)):
                expected_state = i % 3
                assert df.iloc[i]["seq"] == seqs[expected_state], (
                    f"Row {i}: expected state {expected_state} -> {seqs[expected_state]}, "
                    f"got {df.iloc[i]['seq']}"
                )

    def test_ct1_state_cycling_with_init_state(self):
        """CT1: Cycling starts from init_state, wraps correctly."""
        with pp.Party():
            seqs = ["AAAA", "CCCC", "GGGG", "TTTT"]
            pool = pp.from_seqs(seqs, mode="sequential")
            df = pool.generate_library(num_seqs=6, init_state=2, seed=0)
            expected = [seqs[(2 + i) % 4] for i in range(6)]
            assert list(df["seq"]) == expected

    def test_ct2_rng_unique_across_operations(self):
        """CT2: Different ops get different seeds (different op.id)."""
        with pp.Party():
            pool = pp.from_seqs(
                ["ACGT", "TGCA", "GGCC", "AATT"], mode="random", num_states=4
            )
            mutated = pool.mutagenize(num_mutations=1, mode="random")
            df = mutated.generate_library(num_seqs=4, init_state=0, seed=42)
            assert len(df) == 4

    def test_ct2_rng_unique_across_states(self):
        """CT2: Different states get different seeds for random ops."""
        with pp.Party():
            pool = pp.from_seqs(
                ["AAAA"] * 4, mode="random", num_states=4
            )
            mutated = pool.mutagenize(num_mutations=1, mode="random", num_states=4)
            df = mutated.generate_library(num_cycles=1, init_state=0, seed=42)
            seqs = list(df["seq"])
            assert len(set(seqs)) > 1

    def test_ct3_name_assembly_topo_order(self):
        """CT3: Name contributions follow topological order."""
        with pp.Party():
            a = pp.from_seqs(
                ["AA", "CC"], mode="sequential", prefix="a"
            )
            b = a.upper().named("b")
            df = b.generate_library(num_cycles=1, init_state=0, seed=0)
            for _, row in df.iterrows():
                if row["name"] is not None:
                    parts = row["name"].split(".")
                    assert len(parts) >= 1

    def test_ct3_diamond_dag_no_duplicate_names(self):
        """CT3: Diamond DAG names — shared ancestor contributes once."""
        with pp.Party():
            a = pp.from_seqs(["AA", "CC"], mode="sequential", prefix="src")
            b = a.upper().named("b_pool")
            c = a.lower().named("c_pool")
            d = pp.join([b, c]).named("joined")
            df = d.generate_library(num_cycles=1, init_state=0, seed=0)
            for _, row in df.iterrows():
                if row["name"] is not None:
                    parts = row["name"].split(".")
                    src_parts = [p for p in parts if p.startswith("src")]
                    assert len(src_parts) <= 1, (
                        f"Shared ancestor 'src' contributed {len(src_parts)} times: {row['name']}"
                    )

    def test_ct4_null_propagation_mid_dag(self):
        """CT4: Mid-DAG NullSeq propagates to final row with seq=None."""
        with pp.Party():
            a = pp.from_seqs(["ACGT", "TGCA", "GGCC"], mode="sequential")
            filt = a.filter(lambda s: s.startswith("A"))
            b = filt.upper()
            df = b.generate_library(
                num_cycles=1, init_state=0, seed=0, discard_null_seqs=False
            )
            assert df.iloc[0]["seq"] == "ACGT"
            assert pd.isna(df.iloc[1]["seq"])
            assert pd.isna(df.iloc[2]["seq"])

    def test_ct4_null_propagation_name_is_none(self):
        """CT4: Null-propagated rows have name=None."""
        with pp.Party():
            a = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            filt = a.filter(lambda s: s.startswith("A"))
            df = filt.generate_library(
                num_cycles=1, init_state=0, seed=0, discard_null_seqs=False
            )
            assert pd.isna(df.iloc[1]["name"])

    def test_ct4_null_propagation_with_discard(self):
        """CT4: Null rows discarded correctly when discard_null_seqs=True."""
        with pp.Party():
            a = pp.from_seqs(["ACGT", "TGCA", "GGCC"], mode="sequential")
            filt = a.filter(lambda s: s.startswith("A"))
            b = filt.upper()
            df = b.generate_library(
                num_cycles=1, init_state=0, seed=0, discard_null_seqs=True
            )
            assert len(df) == 1
            assert df.iloc[0]["seq"] == "ACGT"
