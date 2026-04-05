"""Audit tests for C10 core output methods.

Scope:
- generate_library (standalone + Pool wrapper)
- Pool.print_library
- ProteinPool.print_library

Methodology follows .cursor/rules/core_output_audit.mdc Steps 2-6.
Step 7 findings and policy classifications are documented in:
- dev/audit/core_output_alpha.md
"""

from __future__ import annotations

import importlib
import inspect
import io
import warnings
from contextlib import redirect_stdout

import pandas as pd
import pytest
import poolparty as pp
from poolparty.generate_library import generate_library


def _capture_print(func, *args, **kwargs) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        func(*args, **kwargs)
    return buf.getvalue()


def _nonempty_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()]


def _state_col(df: pd.DataFrame) -> str:
    cols = [c for c in df.columns if c.endswith(".state")]
    assert cols, f"missing .state column in {list(df.columns)}"
    return cols[0]


class TestStep2WrapperParity:
    """Step 2: wrapper parity and runtime forwarding checks."""

    def test_generate_library_wrapper_signature_parity(self):
        sig_fn = inspect.signature(generate_library)
        sig_wrapper = inspect.signature(pp.Pool.generate_library)

        fn_params = [p for name, p in sig_fn.parameters.items() if name != "pool"]
        wrapper_params = [p for name, p in sig_wrapper.parameters.items() if name != "self"]

        assert [p.name for p in wrapper_params] == [p.name for p in fn_params]
        for wp, fp in zip(wrapper_params, fn_params):
            assert wp.default == fp.default
            assert wp.annotation == fp.annotation

    def test_generate_library_wrapper_forwards_all_params_runtime(self):
        with pp.Party():
            pool = pp.from_seqs(["AA", "TT", "GG"], mode="sequential")
            df = pool.generate_library(
                num_cycles=2,
                num_seqs=3,
                seed=11,
                init_state=0,
                seqs_only=False,
                _include_inline_styles=True,
                discard_null_seqs=False,
                max_iterations=9,
                min_acceptance_rate=0.1,
                attempts_per_rate_assessment=1,
            )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "_inline_styles" in df.columns

    def test_print_library_forwards_subset_to_generate_library_num_seqs(self, monkeypatch):
        with pp.Party():
            pool = pp.from_seq("AA")
            captured = {}

            def fake_generate_library(**kwargs):
                captured.update(kwargs)
                return pd.DataFrame({"name": ["n0"], "seq": ["AA"]})

            monkeypatch.setattr(pool, "generate_library", fake_generate_library)
            pool.print_library(
                num_seqs=7,
                show_header=False,
                show_state=False,
                show_name=False,
                seed=5,
                discard_null_seqs=True,
                max_iterations=13,
                min_acceptance_rate=0.25,
                attempts_per_rate_assessment=4,
            )

        assert captured["seqs_only"] is False
        assert captured["init_state"] == 0
        assert captured["_include_inline_styles"] is True
        assert captured["seed"] == 5
        assert captured["discard_null_seqs"] is True
        assert captured["max_iterations"] == 13
        assert captured["min_acceptance_rate"] == 0.25
        assert captured["attempts_per_rate_assessment"] == 4
        assert captured["num_seqs"] == 7
        assert "num_cycles" not in captured

    def test_print_library_forwards_num_cycles_when_num_seqs_missing(self, monkeypatch):
        with pp.Party():
            pool = pp.from_seq("AA")
            captured = {}

            def fake_generate_library(**kwargs):
                captured.update(kwargs)
                return pd.DataFrame({"name": ["n0"], "seq": ["AA"]})

            monkeypatch.setattr(pool, "generate_library", fake_generate_library)
            pool.print_library(
                num_cycles=4,
                show_header=False,
                show_state=False,
                show_name=False,
            )

        assert captured["num_cycles"] == 4
        assert "num_seqs" not in captured


class TestStep3GenerateLibraryContracts:
    """Step 3 G contracts (G1-G9)."""

    def test_g1_row_count_num_seqs_and_num_cycles(self):
        with pp.Party():
            pool = pp.from_seqs(["A", "C", "G", "T"], mode="sequential")
            df_seqs = pool.generate_library(num_seqs=5, init_state=0)
            df_cycles = pool.generate_library(num_cycles=2, init_state=0)

        assert len(df_seqs) == 5
        assert len(df_cycles) == 8

    def test_g1_boundary_num_seqs_zero_raises_and_one_works(self):
        with pp.Party():
            pool = pp.from_seqs(["A", "C"], mode="sequential")
            with pytest.raises(ValueError, match="num_seqs must be positive"):
                pool.generate_library(num_seqs=0, init_state=0)
            df1 = pool.generate_library(num_seqs=1, init_state=0)

        assert len(df1) == 1

    def test_g2_sequential_covers_full_state_space(self):
        with pp.Party():
            pool = pp.from_seqs(["AA", "TT", "GG", "CC"], mode="sequential")
            df = pool.generate_library(num_cycles=1, init_state=0)
        assert df["seq"].tolist() == ["AA", "TT", "GG", "CC"]

    def test_g2_boundary_state_cycling_across_cycles(self):
        with pp.Party():
            pool = pp.from_seqs(["AA", "TT", "GG", "CC"], mode="sequential", cards=["state"])
            df = pool.generate_library(num_cycles=3, init_state=0)
            state_col = _state_col(df)
        assert df[state_col].tolist() == [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]

    def test_g3_discard_null_true_has_no_null_rows(self):
        with pp.Party():
            pool = pp.from_seqs(["AA", "TT", "GG", "CC"], mode="sequential").filter(
                lambda s: s in {"AA", "TT"}
            )
            df = pool.generate_library(num_seqs=2, discard_null_seqs=True, init_state=0)
        assert len(df) == 2
        assert df["seq"].notna().all()

    def test_g3_boundary_all_rejected_warns_and_returns_empty(self):
        with pp.Party():
            pool = pp.from_seq("AAAA").filter(lambda s: False)
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always")
                df = pool.generate_library(
                    num_seqs=2,
                    discard_null_seqs=True,
                    max_iterations=5,
                    init_state=0,
                )
        assert len(df) == 0
        assert any("max_iterations" in str(w.message) for w in ws)

    def test_g4_discard_null_false_preserves_null_rows(self):
        with pp.Party():
            pool = pp.from_seqs(["AA", "TT", "GG", "CC"], mode="sequential").filter(
                lambda s: s in {"AA", "TT"}
            )
            df = pool.generate_library(num_seqs=4, discard_null_seqs=False, init_state=0)
        assert len(df) == 4
        assert df["seq"].isna().sum() == 2

    def test_g4_boundary_name_is_none_for_null_rows(self):
        with pp.Party():
            pool = pp.from_seq("AAAA").filter(lambda s: False)
            df = pool.generate_library(num_seqs=2, discard_null_seqs=False, init_state=0)
        assert df["seq"].isna().all()
        assert df["name"].isna().all()

    def test_g5_seed_determinism_same_seed_same_output(self):
        with pp.Party():
            pool = pp.from_seqs(["A", "C", "G", "T"], mode="random", num_states=4)
            d1 = pool.generate_library(num_seqs=12, seed=42, init_state=0)
            d2 = pool.generate_library(num_seqs=12, seed=42, init_state=0)
        assert d1.equals(d2)

    def test_g5_boundary_different_seed_changes_random_output(self):
        with pp.Party():
            pool = pp.from_seqs(["A", "C", "G", "T"], mode="random", num_states=4)
            d1 = pool.generate_library(num_seqs=20, seed=41, init_state=0)
            d2 = pool.generate_library(num_seqs=20, seed=42, init_state=0)
        assert d1["seq"].tolist() != d2["seq"].tolist()

    def test_g6_init_state_starts_at_requested_state(self):
        with pp.Party():
            pool = pp.from_seqs(["A", "B", "C", "D"], mode="sequential")
            df = pool.generate_library(num_seqs=3, init_state=3)
        assert df["seq"].tolist() == ["D", "A", "B"]

    def test_g6_boundary_init_state_wraps_by_modulo(self):
        with pp.Party():
            pool = pp.from_seqs(["A", "B", "C", "D"], mode="sequential")
            df = pool.generate_library(num_seqs=2, init_state=7)
        assert df["seq"].tolist() == ["D", "A"]

    def test_g7_max_iterations_stops_high_rejection_sampling(self):
        with pp.Party():
            pool = pp.from_seq("AAAA").filter(lambda s: False)
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always")
                df = pool.generate_library(
                    num_seqs=4,
                    discard_null_seqs=True,
                    max_iterations=5,
                    init_state=0,
                )
        assert len(df) < 4
        assert any("Reached max_iterations" in str(w.message) for w in ws)

    def test_g7_boundary_max_iterations_ignored_without_discard(self):
        with pp.Party():
            pool = pp.from_seqs(["A", "B", "C", "D"], mode="sequential")
            d1 = pool.generate_library(num_seqs=4, max_iterations=1, init_state=0)
            d2 = pool.generate_library(num_seqs=4, max_iterations=999, init_state=0)
        assert d1["seq"].tolist() == d2["seq"].tolist()

    def test_g8_seqs_only_returns_list_of_strings(self):
        with pp.Party():
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential")
            seqs = pool.generate_library(num_seqs=3, seqs_only=True, init_state=0)
        assert isinstance(seqs, list)
        assert len(seqs) == 3
        assert all(isinstance(s, str) for s in seqs)

    def test_g8_boundary_seqs_only_with_null_preserve_returns_none_entries(self):
        with pp.Party():
            pool = pp.from_seq("AAAA").filter(lambda s: False)
            result = pool.generate_library(
                num_seqs=2,
                seqs_only=True,
                discard_null_seqs=False,
                init_state=0,
            )
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(v is None for v in result)

    def test_g9_design_card_columns_present_when_configured(self):
        with pp.Party():
            pool = pp.from_seqs(["A", "B"], mode="sequential", cards=["state"])
            df = pool.generate_library(num_cycles=1, init_state=0)
        assert any(c.endswith(".state") for c in df.columns)

    def test_g9_boundary_no_cards_only_name_and_seq_columns(self):
        with pp.Party():
            pool = pp.from_seqs(["A", "B"], mode="sequential")
            df = pool.generate_library(num_cycles=1, init_state=0)
        assert list(df.columns) == ["name", "seq"]


class TestStep3PrintLibraryContracts:
    """Step 3 P contracts (P1-P6)."""

    def test_p1_printed_sequences_match_generate_library_init_state_zero(self, capsys):
        with pp.Party():
            pool = pp.from_seqs(["AA", "TT", "GG"], mode="sequential")
            df = pool.generate_library(
                num_seqs=3,
                seed=17,
                init_state=0,
                _include_inline_styles=True,
            )
            pool.print_library(
                num_seqs=3,
                seed=17,
                show_header=False,
                show_state=False,
                show_name=False,
            )
            out = capsys.readouterr().out

        assert _nonempty_lines(out) == df["seq"].tolist()

    def test_p2_column_visibility_show_flags(self, capsys):
        with pp.Party():
            pool = pp.from_seqs(["AA", "TT"], mode="sequential")
            pool.print_library(
                num_seqs=2,
                show_header=True,
                show_state=False,
                show_name=False,
                show_seq=False,
            )
            out = capsys.readouterr().out
        lines = _nonempty_lines(out)
        assert len(lines) == 1
        assert "seq_length=" in lines[0]

    def test_p3_header_content_reports_seq_length_and_num_states(self, capsys):
        with pp.Party():
            pool = pp.from_seqs(["AA", "TT", "GG"], mode="sequential")
            pool.print_library(num_seqs=1, show_state=False)
            out = capsys.readouterr().out
        assert "seq_length=2" in out
        assert "num_states=3" in out

    def test_p3_boundary_header_shows_seq_length_none(self, capsys):
        with pp.Party():
            pool = pp.from_seqs(["A", "AA"], mode="sequential")
            pool.print_library(num_seqs=1, show_state=False)
            out = capsys.readouterr().out
        assert "seq_length=None" in out

    def test_p4_name_padding_true_aligns_widths_false_does_not(self, monkeypatch):
        with pp.Party():
            pool = pp.from_seq("AA")

            def fake_generate_library(**kwargs):
                return pd.DataFrame({"name": ["a", "long_name"], "seq": ["AA", "TT"]})

            monkeypatch.setattr(pool, "generate_library", fake_generate_library)
            out_pad = _capture_print(
                pool.print_library,
                num_seqs=2,
                show_header=False,
                show_state=False,
                show_seq=False,
                pad_names=True,
            )
            out_no_pad = _capture_print(
                pool.print_library,
                num_seqs=2,
                show_header=False,
                show_state=False,
                show_seq=False,
                pad_names=False,
            )

        pad_lines = [ln for ln in out_pad.splitlines() if ln]
        no_pad_lines = [ln for ln in out_no_pad.splitlines() if ln]
        assert len(set(len(ln) for ln in pad_lines)) == 1
        assert len(set(len(ln) for ln in no_pad_lines)) > 1

    def test_p4_boundary_all_names_none_hides_name_column(self, capsys):
        with pp.Party():
            pool = pp.from_seq("AAAA").filter(lambda s: False)
            pool.print_library(
                num_seqs=1,
                discard_null_seqs=False,
                show_header=True,
                show_state=False,
                show_seq=False,
                show_name=True,
            )
            out = capsys.readouterr().out
        assert "name" not in out

    def test_p5_print_library_returns_self_and_chaining_works(self):
        with pp.Party():
            pool = pp.from_seq("ATG")
            assert pool.print_library(show_header=False, show_state=False) is pool
            assert (
                pool.print_library(show_header=False, show_state=False)
                .print_library(show_header=False, show_state=False)
                is pool
            )

    def test_p6_protein_three_letter_and_separator_behavior(self, capsys):
        with pp.Party():
            pool = pp.from_seq("ATGGCTTAA").translate()
            pool.print_library(
                chars_per_aa=3,
                aa_separator=" ",
                show_header=False,
                show_state=False,
                show_name=False,
            )
            out_space = capsys.readouterr().out

            pool.print_library(
                chars_per_aa=3,
                aa_separator="",
                show_header=False,
                show_state=False,
                show_name=False,
            )
            out_empty = capsys.readouterr().out

            pool.print_library(show_header=False, show_state=False, show_name=False)
            out_one = capsys.readouterr().out

        assert "Met Ala ***" in out_space
        assert "MetAla***" in out_empty
        assert "MA*" in out_one


class TestStep4PolicyReview:
    """Step 4 policy behavior checks (Q1-Q11)."""

    def test_q1_num_seqs_wins_when_num_seqs_and_num_cycles_both_provided(self):
        with pp.Party():
            pool = pp.from_seqs(["A", "B", "C", "D"], mode="sequential")
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                df = pool.generate_library(num_seqs=3, num_cycles=99, init_state=0)
        assert len(df) == 3
        assert df["seq"].tolist() == ["A", "B", "C"]
        assert any("num_seqs takes precedence" in str(x.message) for x in w)

    def test_q2_max_iterations_hit_warns_with_acceptance_rate(self):
        with pp.Party():
            pool = pp.from_seq("AAAA").filter(lambda s: False)
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always")
                _ = pool.generate_library(
                    num_seqs=2,
                    discard_null_seqs=True,
                    max_iterations=5,
                    init_state=0,
                )
        assert any("Acceptance rate" in str(w.message) for w in ws)

    def test_q3_discard_null_true_with_num_seqs_may_return_partial_with_warning(self):
        with pp.Party():
            pool = pp.from_seqs(["AA", "TT", "GG", "CC"], mode="sequential").filter(
                lambda s: s in {"AA", "TT"}
            )
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always")
                df = pool.generate_library(num_seqs=3, discard_null_seqs=True, init_state=0)
        assert len(df) == 2
        assert any("requested 3" in str(w.message) for w in ws)

    def test_q4_min_acceptance_rate_stops_early_with_warning(self):
        with pp.Party():
            pool = pp.from_seq("AAAA").filter(lambda s: False)
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always")
                df = pool.generate_library(
                    num_seqs=10,
                    discard_null_seqs=True,
                    min_acceptance_rate=0.1,
                    attempts_per_rate_assessment=2,
                    init_state=0,
                )
        assert len(df) == 0
        assert any("below minimum" in str(w.message) for w in ws)

    def test_q5_init_state_persists_across_calls_when_not_overridden(self):
        with pp.Party():
            pool = pp.from_seqs(["A", "B", "C", "D"], mode="sequential")
            d1 = pool.generate_library(num_seqs=2, init_state=1)
            d2 = pool.generate_library(num_seqs=2)
        assert d1["seq"].tolist() == ["B", "C"]
        assert d2["seq"].tolist() == ["D", "A"]

    def test_q6_nonpositive_num_seqs_or_num_cycles_raises(self):
        with pp.Party():
            pool = pp.from_seqs(["A", "B"], mode="sequential")
            with pytest.raises(ValueError, match="num_seqs must be positive"):
                pool.generate_library(num_seqs=0)
            with pytest.raises(ValueError, match="num_seqs must be positive"):
                pool.generate_library(num_seqs=-1)
            with pytest.raises(ValueError, match="num_cycles must be positive"):
                pool.generate_library(num_cycles=0)
            with pytest.raises(ValueError, match="num_cycles must be positive"):
                pool.generate_library(num_cycles=-2)

    def test_q7_seqs_only_with_null_preserve_returns_none_entries(self):
        with pp.Party():
            pool = pp.from_seq("AAAA").filter(lambda s: False)
            result = pool.generate_library(
                num_seqs=2,
                seqs_only=True,
                discard_null_seqs=False,
                init_state=0,
            )
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(v is None for v in result)

    def test_q8_print_library_defaults_to_num_cycles_one(self, monkeypatch):
        with pp.Party():
            pool = pp.from_seq("AA")
            captured = {}

            def fake_generate_library(**kwargs):
                captured.update(kwargs)
                return pd.DataFrame({"name": ["n0"], "seq": ["AA"]})

            monkeypatch.setattr(pool, "generate_library", fake_generate_library)
            pool.print_library(show_header=False, show_state=False, show_name=False)

        assert captured["num_cycles"] == 1
        assert "num_seqs" not in captured

    def test_q9_show_state_true_missing_state_column_omits_state_everywhere(self):
        with pp.Party():
            pool = pp.from_seqs(["A", "B"], mode="sequential")
            out = _capture_print(pool.print_library, num_seqs=2, show_state=True, show_header=True)
        lines = _nonempty_lines(out)
        assert "state" not in lines[1]
        assert lines[2] in {"A", "B"}

    def test_q10_very_long_sequences_not_truncated(self):
        with pp.Party():
            pool = pp.from_seq("A" * 200)
            out = _capture_print(
                pool.print_library,
                show_header=False,
                show_state=False,
                show_name=False,
            )
        lines = _nonempty_lines(out)
        assert len(lines[0]) == 200
        assert set(lines[0]) == {"A"}

    def test_q11_none_sequences_print_as_literal_none(self):
        with pp.Party():
            pool = pp.from_seq("AAAA").filter(lambda s: False)
            out = _capture_print(
                pool.print_library,
                num_seqs=2,
                discard_null_seqs=False,
                show_header=False,
                show_state=False,
                show_name=False,
            )
        assert _nonempty_lines(out) == ["None", "None"]


class TestStep5AdversarialPatterns:
    """Step 5 required diagonals plus assumption inversions."""

    def test_diagonal1_proteinpool_random_discard_null_seed(self):
        with pp.Party():
            dna = pp.from_seqs(["ATG", "GCT", "TAA", "AAA"], mode="random", num_states=4)
            pool = dna.translate().filter(lambda s: s != "*")
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                d1 = pool.generate_library(
                    num_seqs=4,
                    seed=7,
                    init_state=0,
                    discard_null_seqs=True,
                    max_iterations=100,
                )
                d2 = pool.generate_library(
                    num_seqs=4,
                    seed=7,
                    init_state=0,
                    discard_null_seqs=True,
                    max_iterations=100,
                )

        assert isinstance(pool, pp.ProteinPool)
        assert d1["seq"].tolist() == d2["seq"].tolist()
        assert d1["seq"].notna().all()

    def test_diagonal2_multi_parent_dag_sequential_two_cycles_with_cards(self):
        with pp.Party():
            base = pp.from_seqs(["AA", "TT"], mode="sequential", prefix="base", cards=["state"])
            left = base.upper(prefix="L")
            right = base.lower(prefix="R")
            pool = pp.join([left, right], prefix="J")
            df = pool.generate_library(num_cycles=2, init_state=0)

        assert len(df) == 4
        assert any(c.endswith(".state") for c in df.columns)
        assert all(name.endswith(".L.R.J") for name in df["name"])

    def test_diagonal3_large_state_space_mid_init_state(self):
        with pp.Party():
            pool = pp.get_kmers(length=6, mode="sequential")
            df = pool.generate_library(num_seqs=5, init_state=2000)
            df_shift = pool.generate_library(num_seqs=5, init_state=2001)

        assert pool.num_states >= 1000
        assert len(df) == 5
        assert df["seq"].iloc[0] != df_shift["seq"].iloc[0]

    def test_assumption_inversion_attempts_per_rate_assessment_zero_raises(self):
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

    def test_assumption_inversion_negative_num_cycles_raises(self):
        with pp.Party():
            pool = pp.from_seqs(["A", "B"], mode="sequential")
            with pytest.raises(ValueError, match="num_cycles must be positive"):
                pool.generate_library(num_cycles=-2)


class TestStep6ContractTracing:
    """Step 6 CT1-CT4 deep tracing."""

    def test_ct1_state_cycling_modulo_path(self):
        with pp.Party():
            pool = pp.from_seqs(["A", "B", "C", "D"], mode="sequential", cards=["state"])
            df = pool.generate_library(num_cycles=2, init_state=2)
            state_col = _state_col(df)

        assert df[state_col].tolist() == [2, 3, 0, 1, 2, 3, 0, 1]
        assert df["seq"].tolist() == ["C", "D", "A", "B", "C", "D", "A", "B"]

    def test_ct2_rng_seed_sequence_components_vary_by_op_and_state(self, monkeypatch):
        glm = importlib.import_module("poolparty.generate_library")
        original_seed_sequence = glm.np.random.SeedSequence
        calls: list[tuple[int, int, int]] = []

        def record_seed(vals):
            calls.append(tuple(vals))
            return original_seed_sequence(vals)

        monkeypatch.setattr(glm.np.random, "SeedSequence", record_seed)

        with pp.Party():
            base = pp.from_seqs(["A", "C", "G"], mode="random")  # action_unique=False
            pool = base.mutagenize(num_mutations=1, mode="random", num_states=3)
            pool.generate_library(num_seqs=4, seed=9, init_state=0)

        assert len(calls) == 8  # 2 random ops * 4 rows
        # Different operations are seeded independently.
        op_ids = {c[1] for c in calls}
        assert len(op_ids) == 2

        base_id = base.operation.id
        mut_id = pool.operation.id
        base_state_vals = [state for _, op_id, state in calls if op_id == base_id]
        mut_state_vals = [state for _, op_id, state in calls if op_id == mut_id]

        # action_uniquely_determined_by_state=False uses global_state (row index).
        assert base_state_vals == [0, 1, 2, 3]
        # action_uniquely_determined_by_state=True uses op.state.value cycling by num_states.
        assert mut_state_vals == [0, 1, 2, 0]

    def test_ct3_name_assembly_in_multi_parent_dag_has_no_duplicate_contributions(self):
        with pp.Party():
            base = pp.from_seqs(["AA", "TT"], mode="sequential", prefix="base")
            left = base.upper(prefix="L")
            right = base.lower(prefix="R")
            pool = pp.join([left, right], prefix="J")
            df = pool.generate_library(num_cycles=1, init_state=0)

        assert df["name"].tolist() == ["base_0.L.R.J", "base_1.L.R.J"]
        assert all(name.count("base_") == 1 for name in df["name"])

    def test_ct4_null_propagation_compute_to_dataframe_to_print(self):
        glm = importlib.import_module("poolparty.generate_library")

        with pp.Party():
            pool = pp.from_seq("AAAA").filter(lambda s: False)
            sorted_ops = glm._topo_sort_operations(pool)
            raw_row = glm._compute_one(
                pool=pool,
                sorted_ops=sorted_ops,
                global_state=0,
                max_global_state=0,
                include_inline_styles=False,
            )
            df = pool.generate_library(num_seqs=1, discard_null_seqs=False, init_state=0)
            out = _capture_print(
                pool.print_library,
                num_seqs=1,
                discard_null_seqs=False,
                show_header=False,
                show_state=False,
                show_name=False,
            )

        assert raw_row["seq"] is None
        assert raw_row["name"] is None
        assert df["seq"].iloc[0] is None
        assert df["name"].iloc[0] is None
        assert _nonempty_lines(out) == ["None"]
