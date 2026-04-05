"""Audit tests for C11 infrastructure convenience methods.

Scope:
- C11a (medium depth): Party contracts Y1-Y10, FilterMixin contracts F1-F12.
- C11b (light depth): operators, copy/deepcopy, runtime toggles, text viz.

This file intentionally captures current behavior (including surprising cases)
to support the paired audit report.
"""

from __future__ import annotations

import inspect
import io
from contextlib import redirect_stdout

import pytest

import poolparty as pp
from poolparty.codon_table import STANDARD_GENETIC_CODE


def _capture_print(func, *args, **kwargs) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        func(*args, **kwargs)
    return buf.getvalue()


def _seq_rows(pool) -> list[object]:
    df = pool.generate_library(num_cycles=1, discard_null_seqs=False, init_state=0)
    return df["seq"].tolist()


class TestC11aPartyContracts:
    def test_y1_context_nesting_inner_active_outer_restored(self):
        with pp.Party() as outer:
            assert pp.get_active_party() is outer
            with pp.Party() as inner:
                assert pp.get_active_party() is inner
            assert pp.get_active_party() is outer

    def test_y2_context_cleanup_and_previous_party_restore(self):
        previous = pp.get_active_party()
        with pp.Party() as party:
            assert party._is_active is True
            assert pp.get_active_party() is party
        assert party._is_active is False
        assert pp.get_active_party() is previous

    def test_y3_pool_isolation_across_nested_parties(self):
        with pp.Party() as outer:
            outer_pool = pp.from_seq("AAAA").named("outer_pool")
            assert outer.get_pool_by_name("outer_pool") is outer_pool
            with pp.Party() as inner:
                inner_pool = pp.from_seq("TTTT").named("inner_pool")
                assert inner.get_pool_by_name("inner_pool") is inner_pool
            with pytest.raises(KeyError):
                _ = outer.get_pool_by_name("inner_pool")
            assert outer.get_pool_by_name("outer_pool") is outer_pool

    def test_y4_clear_pools_resets_registries_ids_and_manager(self):
        with pp.Party() as party:
            base = pp.from_seqs(["AA", "TT"], mode="sequential").named("base")
            out = base.upper().named("out")
            party.register_region("r1", seq_length=2)
            party.output(out, name="out1")
            old_manager = party.state_manager

            party.clear_pools()

            assert len(party._pools_by_id) == 0
            assert len(party._ops_by_id) == 0
            assert len(party._regions_by_id) == 0
            assert party._next_pool_id == 0
            assert party._next_op_id == 0
            assert party._next_region_id == 0
            assert len(party._outputs) == 0
            assert party.state_manager is not old_manager
            assert party._is_active is True

    def test_y5_clear_pools_preserves_codon_table_and_config(self):
        with pp.Party() as party:
            codon_obj = party._codon_table
            config_obj = party._config
            pp.from_seq("ATGAAA").named("dna")
            party.clear_pools()
            assert party._codon_table is codon_obj
            assert party._config is config_obj

    def test_y6_set_genetic_code_updates_future_orf_ops(self):
        custom_code = {aa: list(codons) for aa, codons in STANDARD_GENETIC_CODE.items()}
        custom_code["J"] = custom_code.pop("M")

        with pp.Party() as party:
            old_table = party.codon_table
            party.set_genetic_code(custom_code)
            assert party.codon_table is not old_table
            assert party.codon_table.codon_to_aa["ATG"] == "J"

            dna = pp.from_seq("ATGAAA")
            mut = dna.mutagenize_orf(
                num_mutations=1,
                mutation_type="any_codon",
                mode="random",
                num_states=2,
            )
            assert mut.operation.codon_table is party.codon_table
            assert mut.operation.codon_table.codon_to_aa["ATG"] == "J"

    def test_y8_output_duplicate_name_silently_overwrites(self):
        with pp.Party() as party:
            p1 = pp.from_seq("AAAA").named("p1")
            p2 = pp.from_seq("TTTT").named("p2")
            party.output(p1, name="dup")
            party.output(p2, name="dup")
            assert party._outputs["dup"] is p2
            assert len(party._outputs) == 1

    def test_y9_register_region_consistency(self):
        with pp.Party() as party:
            r1 = party.register_region("my_region", seq_length=4)
            r2 = party.register_region("my_region", seq_length=4)
            assert r2 is r1
            with pytest.raises(ValueError, match="already registered"):
                party.register_region("my_region", seq_length=5)

    def test_y10_duplicate_pool_and_operation_names_raise(self):
        with pp.Party():
            p1 = pp.from_seq("AAAA")
            p2 = pp.from_seq("TTTT")
            p1.name = "dup_pool"
            with pytest.raises(ValueError, match="Pool name"):
                p2.name = "dup_pool"

            p1.operation.name = "dup_op"
            with pytest.raises(ValueError, match="Operation name"):
                p2.operation.name = "dup_op"

    def test_party_edge_cases_from_rule(self):
        custom_code = {aa: list(codons) for aa, codons in STANDARD_GENETIC_CODE.items()}
        custom_code["J"] = custom_code.pop("M")

        with pp.Party() as outer:
            outer_before = outer.codon_table.codon_to_aa["ATG"]
            outer.clear_pools()
            outer.clear_pools()  # idempotent double-clear
            with pp.Party() as inner:
                inner.set_genetic_code(custom_code)
                assert inner.codon_table.codon_to_aa["ATG"] == "J"
            assert outer.codon_table.codon_to_aa["ATG"] == outer_before


class TestC11aFilterMixinContracts:
    def test_f1_filter_gc_exact_half_only(self):
        with pp.Party():
            pool = pp.from_seqs(["ATGC", "AAAA", "GGGG"], mode="sequential")
            filtered = pool.filter_gc(min_gc=0.5, max_gc=0.5)
            assert _seq_rows(filtered) == ["ATGC", None, None]

    def test_f2_filter_gc_validation(self):
        with pp.Party():
            pool = pp.from_seq("ATGC")
            with pytest.raises(ValueError, match="cannot be greater"):
                pool.filter_gc(min_gc=0.6, max_gc=0.5)
            with pytest.raises(ValueError, match="min_gc"):
                pool.filter_gc(min_gc=-0.1, max_gc=1.0)
            with pytest.raises(ValueError, match="min_gc"):
                pool.filter_gc(min_gc=1.1, max_gc=1.0)

    def test_f3_filter_gc_accept_all(self):
        with pp.Party():
            pool = pp.from_seqs(["ATGC", "AAAA", "GGGG"], mode="sequential")
            filtered = pool.filter_gc(min_gc=0.0, max_gc=1.0)
            assert _seq_rows(filtered) == ["ATGC", "AAAA", "GGGG"]

    def test_f4_filter_homopolymer_boundary(self):
        with pp.Party():
            pool = pp.from_seqs(["AAAA", "AAAAA"], mode="sequential")
            filtered = pool.filter_homopolymer(max_length=4)
            assert _seq_rows(filtered) == ["AAAA", None]

    def test_f5_filter_homopolymer_validation(self):
        with pp.Party():
            pool = pp.from_seq("AAAA")
            with pytest.raises(ValueError, match="max_length"):
                pool.filter_homopolymer(max_length=0)
            with pytest.raises(ValueError, match="max_length"):
                pool.filter_homopolymer(max_length=-1)

    def test_f6_filter_complexity_boundary(self):
        with pp.Party():
            pool = pp.from_seqs(["AAAAAA", "ATATAT", "ACGTACGT"], mode="sequential")
            lo = pool.filter_complexity(min_complexity=0.0)
            hi = pool.filter_complexity(min_complexity=1.0)
            lo_rows = _seq_rows(lo)
            hi_rows = _seq_rows(hi)

            assert lo_rows.count(None) == 0
            assert hi_rows.count(None) >= 1
            assert hi_rows.count(None) >= lo_rows.count(None)

    def test_f7_filter_dust_boundary(self):
        with pp.Party():
            pool = pp.from_seqs(["AAAAAA", "ATATAT", "ACGTACGT"], mode="sequential")
            strict = pool.filter_dust(max_score=0.0)
            permissive = pool.filter_dust(max_score=999.0)
            strict_rows = _seq_rows(strict)
            permissive_rows = _seq_rows(permissive)

            assert permissive_rows.count(None) == 0
            assert strict_rows.count(None) >= 1

    def test_f8_filter_restriction_sites_enzyme(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGTGAATTCACGT", "ACGTACGTACGT"], mode="sequential")
            filtered = pool.filter_restriction_sites(enzymes=["EcoRI"])
            assert _seq_rows(filtered) == [None, "ACGTACGTACGT"]

    def test_f9_filter_restriction_sites_preset(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGTGGTCTCACGT", "ACGTACGTACGT"], mode="sequential")
            filtered = pool.filter_restriction_sites(enzymes=["golden_gate"])
            rows = _seq_rows(filtered)
            assert isinstance(filtered, type(pool))
            assert rows == [None, "ACGTACGTACGT"]

    def test_f10_filter_restriction_sites_validation(self):
        with pp.Party():
            pool = pp.from_seq("ACGT")
            with pytest.raises(ValueError, match="At least one"):
                pool.filter_restriction_sites()

    def test_f11_filter_restriction_sites_reverse_complement_toggle(self):
        with pp.Party():
            # BsaI site: GGTCTC; reverse complement: GAGACC.
            pool = pp.from_seqs(["ACGTGAGACCACGT", "ACGTACGTACGT"], mode="sequential")
            with_rc = pool.filter_restriction_sites(enzymes=["BsaI"], check_rc=True)
            no_rc = pool.filter_restriction_sites(enzymes=["BsaI"], check_rc=False)
            assert _seq_rows(with_rc) == [None, "ACGTACGTACGT"]
            assert _seq_rows(no_rc) == ["ACGTGAGACCACGT", "ACGTACGTACGT"]

    def test_f12_filter_methods_preserve_pool_type(self):
        with pp.Party():
            pool = pp.from_seqs(["ATGC", "AAAA", "GGGG"], mode="sequential")
            results = [
                pool.filter_gc(),
                pool.filter_homopolymer(max_length=4),
                pool.filter_complexity(min_complexity=0.0),
                pool.filter_dust(max_score=2.0),
                pool.filter_restriction_sites(enzymes=["EcoRI"]),
            ]
            assert all(isinstance(r, type(pool)) for r in results)

    def test_filter_adversarial_short_sequences_gc(self):
        with pp.Party():
            pool = pp.from_seqs(["A", "GC"], mode="sequential")
            filtered = pool.filter_gc(min_gc=0.5, max_gc=1.0)
            assert _seq_rows(filtered) == [None, "GC"]

    def test_filter_adversarial_overlapping_site_hits(self):
        with pp.Party():
            pool = pp.from_seqs(["AAAAA", "ACGTA"], mode="sequential")
            filtered = pool.filter_restriction_sites(sites=["AAA"])
            assert _seq_rows(filtered) == [None, "ACGTA"]

    def test_filter_adversarial_single_base_complexity(self):
        with pp.Party():
            pool = pp.from_seqs(["A", "ACGT"], mode="sequential")
            filtered = pool.filter_complexity(min_complexity=0.5)
            rows = _seq_rows(filtered)
            assert len(rows) == 2
            assert isinstance(filtered, type(pool))


class TestC11bOperators:
    def test_operator_delegation_add_matches_stack(self):
        with pp.Party():
            a = pp.from_seqs(["A", "B"], mode="sequential")
            b = pp.from_seqs(["X", "Y"], mode="sequential")
            via_op = (a + b).named("via_op")
            via_fn = pp.stack([a, b]).named("via_fn")
            assert via_op.generate_library(num_cycles=1)["seq"].tolist() == via_fn.generate_library(
                num_cycles=1
            )["seq"].tolist()

    def test_operator_delegation_mul_matches_repeat(self):
        with pp.Party():
            pool = pp.from_seqs(["A", "B"], mode="sequential")
            via_op = (pool * 3).named("via_op")
            via_fn = pp.repeat(pool, 3).named("via_fn")
            assert via_op.generate_library(num_cycles=1)["seq"].tolist() == via_fn.generate_library(
                num_cycles=1
            )["seq"].tolist()

    def test_operator_delegation_rmul_matches_mul(self):
        with pp.Party():
            pool = pp.from_seqs(["A", "B"], mode="sequential")
            left = (3 * pool).named("left")
            right = (pool * 3).named("right")
            assert left.generate_library(num_cycles=1)["seq"].tolist() == right.generate_library(
                num_cycles=1
            )["seq"].tolist()

    def test_operator_delegation_getitem_matches_state_slice(self):
        with pp.Party():
            pool = pp.from_seqs(["A", "B", "C", "D", "E"], mode="sequential")
            via_op = pool[2:5].named("via_op")
            via_fn = pp.state_slice(pool, slice(2, 5)).named("via_fn")
            assert via_op.generate_library(num_cycles=1)["seq"].tolist() == via_fn.generate_library(
                num_cycles=1
            )["seq"].tolist()

    def test_operator_edge_mul_zero_raises(self):
        with pp.Party():
            pool = pp.from_seq("AAAA")
            with pytest.raises(ValueError, match="times"):
                _ = pool * 0

    def test_operator_edge_mul_negative_raises(self):
        with pp.Party():
            pool = pp.from_seq("AAAA")
            with pytest.raises(ValueError, match="times"):
                _ = pool * -1

    def test_operator_edge_large_index_raises_index_error(self):
        with pp.Party():
            pool = pp.from_seqs(["A", "B", "C", "D"], mode="sequential")
            with pytest.raises(IndexError, match="index 100 is out of range"):
                pool[100]

    def test_operator_edge_negative_one_returns_last_state(self):
        with pp.Party():
            pool = pp.from_seqs(["A", "B", "C", "D"], mode="sequential")
            last = pool[-1]
            assert last.generate_library(num_seqs=1)["seq"].tolist() == ["D"]

    def test_operator_api_preserves_pool_class(self):
        with pp.Party():
            a = pp.from_seq("AAAA")
            b = pp.from_seq("TTTT")
            assert type(a + b) is type(a)
            assert type(a * 2) is type(a)
            assert type(a[0:1]) is type(a)


class TestC11bCopyDeepcopy:
    def test_copy_delegation_keeps_operation_params_and_shared_parents(self):
        with pp.Party():
            root = pp.from_seqs(["AA", "TT"], mode="sequential")
            pool = root.upper(prefix="u")
            copied = pool.copy()

            assert copied.operation.mode == pool.operation.mode
            assert copied.operation.seq_length == pool.operation.seq_length
            assert copied.operation.parent_pools == pool.operation.parent_pools
            assert copied.operation is not pool.operation
            assert copied.generate_library(num_cycles=1)["seq"].tolist() == pool.generate_library(
                num_cycles=1
            )["seq"].tolist()
            # Current behavior: copy() on some fixed wrappers normalizes factory_name to "fixed".
            assert copied.operation.factory_name in {"fixed", pool.operation.factory_name}

    def test_deepcopy_delegation_creates_independent_dag(self):
        with pp.Party():
            root = pp.from_seqs(["AA", "TT"], mode="sequential")
            pool = root.upper(prefix="u")
            deep = pool.deepcopy(name="deep")

            assert deep.operation is not pool.operation
            assert deep.operation.parent_pools[0] is not pool.operation.parent_pools[0]
            assert deep.generate_library(num_cycles=1)["seq"].tolist() == pool.generate_library(
                num_cycles=1
            )["seq"].tolist()

    def test_copy_edge_region_tags_preserved(self):
        """FIXED (F3): copy/deepcopy now preserve regions."""
        with pp.Party():
            pool = pp.from_seq("AA<r>TT</r>CC")
            copied = pool.copy()
            deep = pool.deepcopy()
            assert sorted(r.name for r in pool.regions) == ["r"]
            assert sorted(r.name for r in copied.regions) == ["r"]
            assert sorted(r.name for r in deep.regions) == ["r"]

    def test_copy_edge_cards_preserved_in_copy_and_deepcopy(self):
        with pp.Party():
            pool = pp.from_seqs(["AA", "TT"], mode="sequential", cards=["seq_index"])
            copied = pool.copy()
            deep = pool.deepcopy()

            assert any(c.endswith(".seq_index") for c in copied.generate_library(num_cycles=1).columns)
            assert any(c.endswith(".seq_index") for c in deep.generate_library(num_cycles=1).columns)

    def test_copy_edge_mutating_original_after_deepcopy_does_not_affect_copy(self):
        with pp.Party():
            pool = pp.from_seqs(["AA", "TT"], mode="sequential")
            deep = pool.deepcopy(name="deep")
            pool.operation.seqs[0] = "GG"

            assert pool.generate_library(num_cycles=1)["seq"].tolist()[0] == "GG"
            assert deep.generate_library(num_cycles=1)["seq"].tolist()[0] == "AA"

    def test_copy_api_preserves_pool_class(self):
        with pp.Party():
            pool = pp.from_seq("AAAA")
            assert type(pool.copy()) is type(pool)
            assert type(pool.deepcopy()) is type(pool)

    def test_copy_api_name_behavior(self):
        """FIXED (F4): deepcopy() default name now uses '.deepcopy' suffix."""
        with pp.Party():
            pool = pp.from_seq("AAAA")
            copied_named = pool.copy(name="my_copy")
            copied_default = pool.copy()
            deep_named = pool.deepcopy(name="my_deep")
            deep_default = pool.deepcopy()

            assert copied_named.name == "my_copy"
            assert copied_default.name.endswith(".copy")
            assert deep_named.name == "my_deep"
            assert deep_default.name.endswith(".deepcopy")


class TestC11bRuntimeConfigToggles:
    def test_toggle_styles_effect_on_print_library_ansi_output(self):
        with pp.Party():
            pool = pp.from_seq("ACGT", style="red")
            pp.toggle_styles(True)
            out_on = _capture_print(
                pool.print_library,
                num_seqs=1,
                show_header=False,
                show_state=False,
                show_name=False,
            )
            pp.toggle_styles(False)
            out_off = _capture_print(
                pool.print_library,
                num_seqs=1,
                show_header=False,
                show_state=False,
                show_name=False,
            )

            assert "\x1b[" in out_on
            assert "\x1b[" not in out_off

    def test_toggle_cards_effect_on_generate_library_columns(self):
        with pp.Party():
            pool = pp.from_seqs(["AA", "TT"], mode="sequential", cards=["seq_index"])
            pp.toggle_cards(True)
            cols_on = pool.generate_library(num_cycles=1).columns.tolist()
            pp.toggle_cards(False)
            cols_off = pool.generate_library(num_cycles=1).columns.tolist()

            assert any(c.endswith(".seq_index") for c in cols_on)
            assert not any(c.endswith(".seq_index") for c in cols_off)

    def test_set_progress_mode_no_crash(self):
        with pp.Party() as party:
            pp.set_progress_mode("auto")
            assert party._config.progress_mode == "auto"

    def test_toggle_without_explicit_with_party_uses_default_active_party(self):
        party = pp.get_active_party()
        assert party is not None
        original = party._config.suppress_cards
        try:
            pp.toggle_cards(False)
            assert party._config.suppress_cards is True
            pp.toggle_cards(True)
            assert party._config.suppress_cards is False
        finally:
            party._config.suppress_cards = original

    def test_toggle_on_off_on_restores_state(self):
        with pp.Party() as party:
            pp.toggle_styles(True)
            assert party._config.suppress_styles is False
            pp.toggle_styles(False)
            assert party._config.suppress_styles is True
            pp.toggle_styles(True)
            assert party._config.suppress_styles is False

    def test_toggle_api_signatures_bool_with_default_true(self):
        sig_styles = inspect.signature(pp.toggle_styles)
        sig_cards = inspect.signature(pp.toggle_cards)
        sig_progress = inspect.signature(pp.set_progress_mode)

        assert sig_styles.parameters["on"].annotation is bool
        assert sig_cards.parameters["on"].annotation is bool
        assert sig_progress.parameters["mode"].annotation is str
        assert sig_styles.parameters["on"].default is True
        assert sig_cards.parameters["on"].default is True
        assert sig_progress.parameters["mode"].default == "auto"


class TestC11bTextViz:
    def test_viz_smoke_single_pool_print_graph_contains_pool_name(self, capsys):
        with pp.Party() as party:
            pp.from_seq("AAAA").named("single")

        party.print_graph()
        out = capsys.readouterr().out
        assert "single" in out

    def test_viz_smoke_multi_level_dag_no_crash(self, capsys):
        with pp.Party() as party:
            root = pp.from_seqs(["AA", "TT"], mode="sequential").named("root")
            mid = root.upper().named("mid")
            _leaf = (mid * 2).named("leaf")

        party.print_graph()
        out = capsys.readouterr().out
        assert "root" in out and "mid" in out and "leaf" in out

    def test_viz_edge_empty_party_no_crash(self, capsys):
        with pp.Party() as party:
            pass
        party.print_graph()
        out = capsys.readouterr().out
        assert "(no pools registered)" in out

    def test_viz_api_all_styles_run_without_error(self, capsys):
        with pp.Party() as party:
            pp.from_seq("AAAA").named("p")

        for style in ("clean", "minimal", "repr"):
            party.print_graph(style=style)
            out = capsys.readouterr().out
            assert len(out.strip()) > 0
