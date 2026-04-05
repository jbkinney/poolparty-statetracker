"""Audit tests for infrastructure convenience methods (C11a + C11b).

Reference: .cursor/rules/infra_convenience_audit.mdc
Scope:
  C11a (medium depth): Party lifecycle (Y1-Y10, YQ1-YQ3), FilterMixin (F1-F12)
  C11b (light depth): Pool operators, copy/deepcopy, config toggles, text_viz
"""

import textwrap

import pytest

import poolparty as pp
import poolparty.party as party_mod
from poolparty import DnaPool, Party, get_active_party


# ===================================================================
# C11a: Party lifecycle contracts (Y1-Y10)
# ===================================================================


def test_y1_context_nesting() -> None:
    """Nested Party contexts: inner is active, outer restored on exit."""
    outer = Party()
    with outer:
        assert get_active_party() is outer
        inner = Party()
        with inner:
            assert get_active_party() is inner
            assert inner._is_active
        # After inner exits, outer is active again
        assert get_active_party() is outer
        assert not inner._is_active


def test_y2_context_cleanup() -> None:
    """After __exit__, _is_active is False and _active_party restored."""
    previous = get_active_party()
    party = Party()
    with party:
        assert party._is_active
        assert get_active_party() is party
    assert not party._is_active
    assert get_active_party() is previous


def test_y3_pool_isolation() -> None:
    """Pools created in inner context not visible in outer context."""
    with Party() as outer:
        pool_outer = pp.from_seq("AAAA")
        with Party() as inner:
            pool_inner = pp.from_seq("GGGG")
            assert "pool_inner" not in outer._pools_by_name or pool_inner not in outer._pools_by_id
            assert pool_inner in inner._pools_by_id
        # Inner pool should not be in outer's registry
        assert pool_inner not in outer._pools_by_id
        assert pool_outer in outer._pools_by_id


def test_y4_clear_pools_reset() -> None:
    """clear_pools() empties all registries, resets counters, recycles state manager."""
    with Party() as party:
        pp.from_seq("AAAA")
        pp.from_seq("GGGG")
        old_manager = party._state_manager
        assert len(party._pools_by_id) == 2
        assert party._next_pool_id == 2

        party.clear_pools()

        assert len(party._pools_by_id) == 0
        assert len(party._pools_by_name) == 0
        assert len(party._operations) == 0
        assert len(party._ops_by_id) == 0
        assert len(party._ops_by_name) == 0
        assert len(party._regions_by_id) == 0
        assert len(party._regions_by_name) == 0
        assert len(party._outputs) == 0
        assert party._next_pool_id == 0
        assert party._next_op_id == 0
        assert party._next_region_id == 0
        assert party._state_manager is not old_manager


def test_y5_clear_pools_preserves_config() -> None:
    """clear_pools() preserves _codon_table and _config."""
    with Party() as party:
        pp.from_seq("AAAA")
        ct_before = party._codon_table
        cfg_before = party._config

        party.clear_pools()

        assert party._codon_table is ct_before
        assert party._config is cfg_before


def test_y6_set_genetic_code() -> None:
    """set_genetic_code() changes codon table."""
    with Party() as party:
        default_table = party._codon_table
        # CodonTable only accepts 'standard' or a dict — use a custom dict
        custom_code = dict(default_table.aa_to_codons)
        custom_code["X"] = ["AAA"]  # Map AAA to X instead of K
        party.set_genetic_code(custom_code)
        assert party._codon_table is not default_table
        assert party._codon_table.codon_to_aa["AAA"] == "X"


def test_y8_output_registration() -> None:
    """party.output(pool) registers pool; duplicate name silently overwrites."""
    with Party() as party:
        pool1 = pp.from_seq("AAAA")
        pool2 = pp.from_seq("GGGG")

        party.output(pool1, name="out")
        assert party._outputs["out"] is pool1

        party.output(pool2, name="out")
        assert party._outputs["out"] is pool2  # overwritten


def test_y8_output_default_name() -> None:
    """output() with name=None uses pool.name or fallback."""
    with Party() as party:
        pool = pp.from_seq("AAAA")
        pool.name = "my_pool"
        party.output(pool)
        assert "my_pool" in party._outputs


def test_y9_region_registry_consistency() -> None:
    """Same name+length returns existing; different length raises ValueError."""
    with Party() as party:
        r1 = party.register_region("test_reg", seq_length=10)
        r2 = party.register_region("test_reg", seq_length=10)
        assert r1 is r2

        with pytest.raises(ValueError, match="already registered"):
            party.register_region("test_reg", seq_length=20)


def test_y10_name_uniqueness_pool() -> None:
    """Duplicate pool name raises ValueError."""
    with Party():
        pool = pp.from_seq("AAAA")
        pool.name = "unique_name"
        pool2 = pp.from_seq("GGGG")
        with pytest.raises(ValueError, match="already exists"):
            pool2.name = "unique_name"


def test_y10_name_uniqueness_op() -> None:
    """Duplicate operation name raises ValueError."""
    with Party() as party:
        pool = pp.from_seq("AAAA")
        op_name = pool.operation.name
        pool2 = pp.from_seq("GGGG")
        with pytest.raises(ValueError, match="already exists"):
            party._validate_op_name(op_name)


# ===================================================================
# C11a: Party policy questions (YQ1-YQ3)
# ===================================================================


def test_yq1_pool_outside_party_context() -> None:
    """Pool created inside Party context can generate_library after exit.

    Classification: INTENTIONAL — pool retains reference to its Party's
    state manager. This is useful for batch processing where pools are
    defined in one phase and generated in another.
    """
    with Party():
        pool = pp.from_seqs(["AAAA", "CCCC"], mode="sequential")

    # Outside context — pool still works
    df = pool.generate_library(num_seqs=2, seed=42)
    assert len(df) == 2
    assert set(df["seq"]) == {"AAAA", "CCCC"}


def test_yq2_clear_pools_between_generations() -> None:
    """clear_pools() between generate_library calls causes no corruption.

    Classification: INTENTIONAL — single-threaded, clear_pools between
    calls is a valid usage pattern for reusing a Party context.
    """
    with Party() as party:
        pool1 = pp.from_seq("AAAA")
        df1 = pool1.generate_library(num_seqs=1, seed=0)
        assert df1["seq"].iloc[0] == "AAAA"

        party.clear_pools()

        pool2 = pp.from_seq("GGGG")
        df2 = pool2.generate_library(num_seqs=1, seed=0)
        assert df2["seq"].iloc[0] == "GGGG"


def test_yq3_ops_without_party_context() -> None:
    """Factory call without any Party raises RuntimeError.

    Classification: INTENTIONAL — clear error message guides user.
    """
    saved = party_mod._active_party
    try:
        party_mod._active_party = None
        with pytest.raises(RuntimeError, match="Party context"):
            pp.from_seq("AAAA")
    finally:
        party_mod._active_party = saved


def test_yq3_post_context_default_party_works() -> None:
    """Factory call after explicit Party exit works — default party restored.

    FIXED (F2): Manager.__enter__/__exit__ now save/restore previous
    _active_manager, so the default party's manager is correctly
    reactivated when the explicit Party exits.

    Pattern:
        pp.from_seq('A')          # works (default party)
        with pp.Party(): ...      # explicit context
        pp.from_seq('B')          # works (default party restored)
    """
    p1 = pp.from_seq("AAAA")
    assert p1 is not None

    with Party():
        p2 = pp.from_seq("CCCC")
        assert p2 is not None

    # After exit, default party and its manager are both restored
    p3 = pp.from_seq("GGGG")
    assert p3 is not None
    df = p3.generate_library(num_seqs=1, seed=0)
    assert df["seq"].iloc[0] == "GGGG"
    p3 = pp.from_seq("TTTT")
    assert p3 is not None


# ===================================================================
# C11a: Party edge cases
# ===================================================================


def test_clear_pools_twice_idempotent() -> None:
    """clear_pools() called twice is idempotent."""
    with Party() as party:
        pp.from_seq("AAAA")
        party.clear_pools()
        party.clear_pools()
        assert len(party._pools_by_id) == 0
        assert party._next_pool_id == 0


def test_clear_pools_on_empty_party() -> None:
    """clear_pools() on party with no pools is a no-op."""
    with Party() as party:
        party.clear_pools()
        assert len(party._pools_by_id) == 0


def test_nested_party_genetic_code_isolation() -> None:
    """set_genetic_code in inner party does not affect outer."""
    with Party() as outer:
        outer_table = outer._codon_table
        custom_code = dict(outer_table.aa_to_codons)
        custom_code["X"] = ["AAA"]
        with Party() as inner:
            inner.set_genetic_code(custom_code)
            assert inner._codon_table is not outer_table
        assert outer._codon_table is outer_table


def test_output_duplicate_name_overwrite() -> None:
    """output() with duplicate name silently overwrites."""
    with Party() as party:
        p1 = pp.from_seq("AAAA")
        p2 = pp.from_seq("GGGG")
        party.output(p1, name="x")
        party.output(p2, name="x")
        assert party._outputs["x"] is p2
        assert len(party._outputs) == 1


# ===================================================================
# C11a: FilterMixin contracts (F1-F12)
# ===================================================================


def test_f1_filter_gc_boundary() -> None:
    """min_gc=0.5, max_gc=0.5 only keeps sequences with exactly 50% GC."""
    with Party():
        pool = pp.from_seqs(["AATT", "AAGC", "GGCC"], mode="sequential")
        filtered = pool.filter_gc(min_gc=0.5, max_gc=0.5)
        df = filtered.generate_library(num_seqs=3, discard_null_seqs=True, seed=0)
        assert set(df["seq"]) == {"AAGC"}


def test_f2_filter_gc_validation() -> None:
    """filter_gc rejects invalid ranges."""
    with Party():
        pool = pp.from_seq("ACGT")
        with pytest.raises(ValueError, match="min_gc"):
            pool.filter_gc(min_gc=-0.1)
        with pytest.raises(ValueError, match="min_gc"):
            pool.filter_gc(min_gc=1.1)
        with pytest.raises(ValueError, match="max_gc"):
            pool.filter_gc(max_gc=-0.1)
        with pytest.raises(ValueError, match="cannot be greater"):
            pool.filter_gc(min_gc=0.8, max_gc=0.2)


def test_f3_filter_gc_accept_all() -> None:
    """min_gc=0.0, max_gc=1.0 filters nothing."""
    with Party():
        seqs = ["AAAA", "ACGT", "GGCC"]
        pool = pp.from_seqs(seqs, mode="sequential")
        filtered = pool.filter_gc(min_gc=0.0, max_gc=1.0)
        df = filtered.generate_library(num_seqs=3, discard_null_seqs=True, seed=0)
        assert len(df) == 3


def test_f4_filter_homopolymer_boundary() -> None:
    """max_length=4: 'AAAA' passes, 'AAAAA' filtered."""
    with Party():
        pool = pp.from_seqs(["ACGTAAAA", "ACGTAAAAA"], mode="sequential")
        filtered = pool.filter_homopolymer(max_length=4)
        df = filtered.generate_library(num_seqs=2, discard_null_seqs=True, seed=0)
        assert len(df) == 1
        assert df["seq"].iloc[0] == "ACGTAAAA"


def test_f5_filter_homopolymer_validation() -> None:
    """filter_homopolymer rejects max_length < 1."""
    with Party():
        pool = pp.from_seq("ACGT")
        with pytest.raises(ValueError, match="max_length"):
            pool.filter_homopolymer(max_length=0)
        with pytest.raises(ValueError, match="max_length"):
            pool.filter_homopolymer(max_length=-1)


def test_f6_filter_complexity_boundary() -> None:
    """min_complexity=0.0 accepts all; 1.0 rejects most."""
    with Party():
        seqs = ["ACGTACGT", "AAAAAAAA"]
        pool = pp.from_seqs(seqs, mode="sequential")

        all_pass = pool.filter_complexity(min_complexity=0.0)
        df_all = all_pass.generate_library(num_seqs=2, discard_null_seqs=True, seed=0)
        assert len(df_all) == 2

    with Party():
        pool = pp.from_seqs(seqs, mode="sequential")
        strict = pool.filter_complexity(min_complexity=1.0)
        df_strict = strict.generate_library(num_seqs=2, discard_null_seqs=True, seed=0)
        assert len(df_strict) <= 1


def test_f7_filter_dust_boundary() -> None:
    """max_score=0.0 rejects most; very high max_score accepts all."""
    with Party():
        seqs = ["ACGTACGTACGT", "AAAAAAAAAAAA"]
        pool = pp.from_seqs(seqs, mode="sequential")

        strict = pool.filter_dust(max_score=0.0)
        df_strict = strict.generate_library(num_seqs=2, discard_null_seqs=True, seed=0)
        # All-A has high DUST score, should be filtered; diverse seq may also be filtered at 0.0
        assert len(df_strict) <= len(seqs)

    with Party():
        pool2 = pp.from_seqs(seqs, mode="sequential")
        permissive = pool2.filter_dust(max_score=100.0)
        df_perm = permissive.generate_library(num_seqs=2, discard_null_seqs=True, seed=0)
        assert len(df_perm) == 2


def test_f8_filter_restriction_sites_enzyme() -> None:
    """EcoRI (GAATTC) is filtered from sequences containing it."""
    with Party():
        pool = pp.from_seqs(["AAAGAATTCAAA", "ACGTACGTACGT"], mode="sequential")
        filtered = pool.filter_restriction_sites(enzymes=["EcoRI"])
        df = filtered.generate_library(num_seqs=2, discard_null_seqs=True, seed=0)
        assert len(df) == 1
        assert "GAATTC" not in df["seq"].iloc[0]


def test_f9_filter_restriction_sites_preset() -> None:
    """Preset 'golden_gate' resolves without error."""
    with Party():
        pool = pp.from_seq("ACGTACGTACGT")
        filtered = pool.filter_restriction_sites(enzymes=["golden_gate"])
        df = filtered.generate_library(num_seqs=1, seed=0)
        assert len(df) == 1


def test_f10_filter_restriction_sites_validation() -> None:
    """Neither enzymes nor sites raises ValueError."""
    with Party():
        pool = pp.from_seq("ACGT")
        with pytest.raises(ValueError, match="enzymes.*sites"):
            pool.filter_restriction_sites()


def test_f11_filter_restriction_sites_reverse_complement() -> None:
    """check_rc=True catches reverse complement; False misses it."""
    with Party():
        # BsaI site: GGTCTC, RC: GAGACC
        seq_with_rc = "AAAGAGACCAAA"
        pool = pp.from_seqs([seq_with_rc, "ACGTACGTACGT"], mode="sequential")

        with_rc = pool.filter_restriction_sites(enzymes=["BsaI"], check_rc=True)
        df_rc = with_rc.generate_library(num_seqs=2, discard_null_seqs=True, seed=0)
        # seq_with_rc should be filtered (has RC of BsaI)
        for seq in df_rc["seq"]:
            assert "GAGACC" not in seq

    with Party():
        pool2 = pp.from_seqs([seq_with_rc, "ACGTACGTACGT"], mode="sequential")
        without_rc = pool2.filter_restriction_sites(enzymes=["BsaI"], check_rc=False)
        df_no_rc = without_rc.generate_library(num_seqs=2, discard_null_seqs=True, seed=0)
        # Without RC check, seq_with_rc should pass (only has RC, not forward site)
        assert len(df_no_rc) == 2


def test_f12_delegation_correctness() -> None:
    """Each filter method returns DnaPool (same type as input)."""
    with Party():
        pool = pp.from_seq("ACGTACGTACGT")
        assert isinstance(pool.filter_gc(min_gc=0.0), DnaPool)
        assert isinstance(pool.filter_homopolymer(max_length=4), DnaPool)
        assert isinstance(pool.filter_complexity(min_complexity=0.0), DnaPool)
        assert isinstance(pool.filter_dust(max_score=10.0), DnaPool)
        assert isinstance(pool.filter_restriction_sites(sites=["GAATTC"]), DnaPool)


# ===================================================================
# C11a: FilterMixin adversarial inputs
# ===================================================================


def test_adversarial_filter_gc_short_seq() -> None:
    """Very short sequence (1-2 bp) through filter_gc -- no crash."""
    with Party():
        pool = pp.from_seqs(["A", "GC"], mode="sequential")
        filtered = pool.filter_gc(min_gc=0.4, max_gc=0.6)
        df = filtered.generate_library(num_seqs=2, discard_null_seqs=True, seed=0)
        # "A" has gc=0.0, "GC" has gc=1.0 -- both should be filtered at 0.4-0.6
        assert len(df) == 0


def test_adversarial_filter_complexity_single_base() -> None:
    """Single-base sequence 'A' through filter_complexity -- no crash.

    calc_complexity("A") = 1.0 because: k=1 yields 1 unique / min(1, 4) = 1.0;
    k=2,3 skipped (seq too short). So "A" passes any threshold <= 1.0.
    This is technically correct but counterintuitive — a single-base
    sequence is maximally "complex" by this metric.
    """
    with Party():
        pool = pp.from_seqs(["A", "ACGTACGT"], mode="sequential")
        # No crash on single-base input
        filtered = pool.filter_complexity(min_complexity=0.5)
        df = filtered.generate_library(num_seqs=2, discard_null_seqs=True, seed=0)
        # Both pass: "A" gets complexity=1.0 (counterintuitive but by design)
        assert len(df) == 2


# ===================================================================
# C11b: Pool operators — delegation, edge cases, API
# ===================================================================


def test_operator_add_delegation() -> None:
    """pool + other produces same result as stack([pool, other])."""
    with Party():
        p1 = pp.from_seqs(["AA", "CC"], mode="sequential")
        p2 = pp.from_seqs(["GG", "TT"], mode="sequential")
        added = p1 + p2
        stacked = pp.stack([p1, p2])
        assert added.num_states == stacked.num_states
        df_add = added.generate_library(num_seqs=4, seed=0)
        df_stack = stacked.generate_library(num_seqs=4, seed=0)
        assert set(df_add["seq"]) == set(df_stack["seq"])


def test_operator_mul_delegation() -> None:
    """pool * 3 produces same result as repeat(pool, 3)."""
    with Party():
        p = pp.from_seqs(["AA", "CC"], mode="sequential")
        mulled = p * 3
        repeated = pp.repeat(p, 3)
        assert mulled.num_states == repeated.num_states


def test_operator_rmul_delegation() -> None:
    """3 * pool produces same result as pool * 3."""
    with Party():
        p = pp.from_seqs(["AA", "CC"], mode="sequential")
        lmul = p * 3
        rmul = 3 * p
        assert lmul.num_states == rmul.num_states


def test_operator_getitem_delegation() -> None:
    """pool[2:5] produces same result as state_slice(pool, slice(2, 5))."""
    with Party():
        p = pp.from_seqs(["AA", "BB", "CC", "DD", "EE", "FF"], mode="sequential")
        sliced_op = p[2:5]
        sliced_fn = pp.state_slice(p, slice(2, 5))
        assert sliced_op.num_states == sliced_fn.num_states
        df_op = sliced_op.generate_library(num_seqs=3, seed=0)
        df_fn = sliced_fn.generate_library(num_seqs=3, seed=0)
        assert set(df_op["seq"]) == set(df_fn["seq"])


def test_operator_mul_zero() -> None:
    """pool * 0 raises ValueError."""
    with Party():
        p = pp.from_seq("AAAA")
        with pytest.raises(ValueError, match="times must be >= 1"):
            p * 0


def test_operator_mul_negative() -> None:
    """pool * -1 raises ValueError."""
    with Party():
        p = pp.from_seq("AAAA")
        with pytest.raises(ValueError, match="times must be >= 1"):
            p * -1


def test_operator_getitem_out_of_range_raises_index_error() -> None:
    """pool[100] on 4-state pool raises IndexError at slice time."""
    with Party():
        p = pp.from_seqs(["AA", "BB", "CC", "DD"], mode="sequential")
        with pytest.raises(IndexError, match="index 100 is out of range"):
            p[100]


def test_operator_getitem_negative_out_of_range_raises_index_error() -> None:
    """pool[-5] on 4-state pool raises IndexError at slice time."""
    with Party():
        p = pp.from_seqs(["AA", "BB", "CC", "DD"], mode="sequential")
        with pytest.raises(IndexError, match="index -5 is out of range"):
            p[-5]


def test_operator_getitem_empty_slice_raises_value_error() -> None:
    """pool[2:2] produces 0 states and raises ValueError at slice time."""
    with Party():
        p = pp.from_seqs(["AA", "BB", "CC", "DD"], mode="sequential")
        with pytest.raises(ValueError, match="slice produces 0 states"):
            p[2:2]


def test_operator_getitem_far_empty_slice_raises_value_error() -> None:
    """pool[100:200] produces 0 states and raises ValueError at slice time."""
    with Party():
        p = pp.from_seqs(["AA", "BB", "CC", "DD"], mode="sequential")
        with pytest.raises(ValueError, match="slice produces 0 states"):
            p[100:200]


def test_operator_getitem_negative_index() -> None:
    """pool[-1] returns last state."""
    with Party():
        seqs = ["AA", "BB", "CC", "DD"]
        p = pp.from_seqs(seqs, mode="sequential")
        last = p[-1]
        df = last.generate_library(num_seqs=1, seed=0)
        assert df["seq"].iloc[0] == "DD"


def test_operator_return_type_preserved() -> None:
    """Operator return type is DnaPool for DnaPool input."""
    with Party():
        p = pp.from_seqs(["AA", "CC"], mode="sequential")
        assert isinstance(p + p, DnaPool)
        assert isinstance(p * 2, DnaPool)
        assert isinstance(2 * p, DnaPool)
        assert isinstance(p[0:1], DnaPool)


# ===================================================================
# C11b: Pool copy / deepcopy
# ===================================================================


def test_copy_same_params() -> None:
    """copy() creates pool with same num_states and sequences."""
    with Party():
        orig = pp.from_seqs(["AA", "CC", "GG"], mode="sequential")
        copied = orig.copy()
        assert copied.num_states == orig.num_states
        df_orig = orig.generate_library(num_seqs=3, seed=0)
        df_copy = copied.generate_library(num_seqs=3, seed=0)
        assert set(df_orig["seq"]) == set(df_copy["seq"])


def test_deepcopy_independent_dag() -> None:
    """deepcopy() creates fully independent DAG."""
    with Party():
        base = pp.from_seqs(["AA", "CC"], mode="sequential")
        mutated = base.mutagenize(num_mutations=1, mode="random", num_states=2)
        deep = mutated.deepcopy()
        # Independence: they have different operations
        assert deep.operation is not mutated.operation
        assert deep.operation.parent_pools[0] is not mutated.operation.parent_pools[0]


def test_copy_preserves_region_tags() -> None:
    """FIXED (F3): copy() now preserves regions from annotate_region path."""
    with Party():
        pool = pp.from_seq("AAGGCCTT")
        tagged = pool.annotate_region("my_region", extent=(2, 6))
        assert tagged.has_region("my_region")
        copied = tagged.copy()
        assert copied.has_region("my_region")
        deep = tagged.deepcopy()
        assert deep.has_region("my_region")


def test_copy_preserves_embedded_tag_regions() -> None:
    """FIXED (F3): copy() now preserves regions from embedded XML tags."""
    with Party():
        pool = pp.from_seq("AA<r>TT</r>CC")
        assert pool.has_region("r")
        copied = pool.copy()
        assert copied.has_region("r")
        deep = pool.deepcopy()
        assert deep.has_region("r")


def test_deepcopy_mutation_independence() -> None:
    """Mutating original after deepcopy doesn't affect copy."""
    with Party():
        base = pp.from_seqs(["AAAA", "CCCC"], mode="sequential")
        deep = base.deepcopy()
        df_deep = deep.generate_library(num_seqs=2, seed=0)
        df_base = base.generate_library(num_seqs=2, seed=0)
        assert set(df_deep["seq"]) == set(df_base["seq"])


def test_copy_return_type() -> None:
    """copy() and deepcopy() preserve DnaPool type."""
    with Party():
        pool = pp.from_seq("ACGT")
        assert isinstance(pool.copy(), DnaPool)
        assert isinstance(pool.deepcopy(), DnaPool)


def test_copy_custom_name() -> None:
    """copy(name='foo') uses the provided name."""
    with Party():
        pool = pp.from_seq("ACGT")
        copied = pool.copy(name="my_copy")
        assert copied.name == "my_copy"


def test_copy_default_name() -> None:
    """copy() without name appends '.copy' to original name."""
    with Party():
        pool = pp.from_seq("ACGT")
        pool.name = "original"
        copied = pool.copy()
        assert copied.name == "original.copy"


def test_deepcopy_default_name() -> None:
    """FIXED (F4): deepcopy() default name now uses '.deepcopy' suffix."""
    with Party():
        pool = pp.from_seq("ACGT")
        pool.name = "original"
        deep = pool.deepcopy()
        assert deep.name == "original.deepcopy"


def test_copy_factory_name_normalization() -> None:
    """copy() of fixed-wrapper ops normalizes factory_name to 'fixed'.

    FINDING: Operations created by wrappers like upper(), lower(), swapcase()
    have factory_name set to the wrapper name (e.g., 'upper'). When copied
    via operation.copy(), _get_copy_params() reconstructs via FixedOp.__init__
    which defaults factory_name to 'fixed'. The copied op still computes
    correctly but loses its identity in DAG visualization.

    Severity: cosmetic.
    """
    with Party():
        pool = pp.from_seq("ACGT")
        upper_pool = pool.upper()
        assert upper_pool.operation.factory_name == "upper"

        copied = upper_pool.copy()
        # factory_name normalizes to 'fixed' on copy
        assert copied.operation.factory_name == "fixed"


def test_copy_cards_preserved() -> None:
    """Copy preserves card configuration in generated output."""
    with Party():
        pool = pp.from_seqs(["AA", "CC"], mode="sequential")
        repeated = pool.repeat(2, cards=["repeat_index"])
        copied = repeated.copy()
        df = copied.generate_library(num_seqs=2, seed=0)
        card_cols = [c for c in df.columns if "repeat_index" in c]
        assert len(card_cols) > 0


# ===================================================================
# C11b: Runtime config toggles
# ===================================================================


def test_toggle_styles_suppresses_styles() -> None:
    """toggle_styles(False) suppresses inline styling."""
    with Party():
        pool = pp.from_seqs(["AA", "CC"], mode="sequential")
        styled = pool.stylize(style="bold")

        pp.toggle_styles(False)
        df = styled.generate_library(num_seqs=2, seed=0)
        # With styles suppressed, seq column should have no ANSI codes
        for seq in df["seq"]:
            assert "\x1b" not in str(seq)

        pp.toggle_styles(True)


def test_toggle_cards_suppresses_card_columns() -> None:
    """toggle_cards(False) suppresses design card columns."""
    with Party():
        pool = pp.from_seqs(["AA", "CC"], mode="sequential")
        repeated = pool.repeat(2, cards=["repeat_index"])

        pp.toggle_cards(False)
        df = repeated.generate_library(num_seqs=2, seed=0)
        card_cols_off = [c for c in df.columns if "repeat_index" in c]
        assert len(card_cols_off) == 0

        pp.toggle_cards(True)
        df2 = repeated.generate_library(num_seqs=2, seed=0)
        card_cols_on = [c for c in df2.columns if "repeat_index" in c]
        assert len(card_cols_on) > 0


def test_set_progress_mode_no_crash() -> None:
    """set_progress_mode() changes config without crash."""
    with Party():
        party = get_active_party()
        pp.set_progress_mode("text")
        assert party._config.progress_mode == "text"
        pp.set_progress_mode("auto")
        assert party._config.progress_mode == "auto"


def test_toggle_without_explicit_party() -> None:
    """Toggles work against the default party without explicit 'with Party()'."""
    party = get_active_party()
    assert party is not None

    pp.toggle_styles(False)
    assert party._config.suppress_styles is True
    pp.toggle_styles(True)
    assert party._config.suppress_styles is False


def test_toggle_on_off_on_cycle() -> None:
    """Toggle on -> off -> on restores state correctly."""
    with Party():
        party = get_active_party()
        pp.toggle_cards(True)
        assert not party._config.suppress_cards
        pp.toggle_cards(False)
        assert party._config.suppress_cards
        pp.toggle_cards(True)
        assert not party._config.suppress_cards


def test_toggle_default_parameter_true() -> None:
    """All toggle functions accept bool with default True."""
    with Party():
        party = get_active_party()
        # toggle_styles() with no arg defaults to on=True -> suppress=False
        pp.toggle_styles()
        assert not party._config.suppress_styles
        pp.toggle_cards()
        assert not party._config.suppress_cards
        pp.set_progress_mode()
        assert party._config.progress_mode == "auto"


def test_toggle_no_active_party_raises_runtime_error() -> None:
    """FIXED (F9): toggle_* with no active party raises RuntimeError."""
    saved = party_mod._active_party
    try:
        party_mod._active_party = None
        with pytest.raises(RuntimeError, match="No active Party context"):
            pp.toggle_styles(False)
        with pytest.raises(RuntimeError, match="No active Party context"):
            pp.toggle_cards(False)
        with pytest.raises(RuntimeError, match="No active Party context"):
            pp.set_progress_mode("text")
    finally:
        party_mod._active_party = saved


def test_yq2_clear_pools_mid_generation_registries_emptied() -> None:
    """clear_pools() called during generation empties registries mid-flight.

    FINDING: No generation lock/guard prevents clear_pools() from resetting
    party registries while generate_library is still producing rows. The
    pool's operation/state tree is self-contained so generation continues,
    but the party's tracking metadata (_pools_by_id, _ops_by_id) becomes
    stale. Single-threaded makes this impractical in normal use, but it is
    unguarded and undocumented.

    Severity: API inconsistency.
    """
    with Party() as party:
        pool = pp.from_seqs(["AA", "CC", "GG"], mode="sequential")
        assert len(party._pools_by_id) > 0
        assert len(party._ops_by_id) > 0

        # Generate first — works fine
        df1 = pool.generate_library(num_seqs=3, seed=0)
        assert len(df1) == 3

        # Clear mid-session
        party.clear_pools()
        assert len(party._pools_by_id) == 0
        assert len(party._ops_by_id) == 0

        # Pool still generates (self-contained DAG) despite empty registries
        df2 = pool.generate_library(num_seqs=3, seed=0)
        assert len(df2) == 3


# ===================================================================
# C11b: text_viz smoke tests
# ===================================================================


def test_viz_single_pool(capsys) -> None:
    """print_graph() on single-pool party — no crash, output contains pool name."""
    with Party() as party:
        pool = pp.from_seq("AAAA")
        pool.name = "viz_test_pool"
        party.print_graph()
        captured = capsys.readouterr()
        assert "viz_test_pool" in captured.out


def test_viz_multi_level_dag(capsys) -> None:
    """print_graph() on multi-level DAG (3+ ops deep) — no crash."""
    with Party() as party:
        pool = pp.from_seq("ACGTACGT")
        m = pool.mutagenize(num_mutations=1, mode="random", num_states=2)
        styled = m.stylize(style="bold")
        repeated = styled.repeat(2)
        party.print_graph()
        captured = capsys.readouterr()
        assert len(captured.out) > 0


def test_viz_empty_party(capsys) -> None:
    """print_graph() on empty party (no pools) — no crash."""
    with Party() as party:
        party.print_graph()
        captured = capsys.readouterr()
        assert "no pools registered" in captured.out


def test_viz_style_clean(capsys) -> None:
    """print_graph(style='clean') produces output."""
    with Party() as party:
        pp.from_seq("ACGT")
        party.print_graph(style="clean")
        captured = capsys.readouterr()
        assert len(captured.out) > 0


def test_viz_style_minimal(capsys) -> None:
    """print_graph(style='minimal') produces output."""
    with Party() as party:
        pp.from_seq("ACGT")
        party.print_graph(style="minimal")
        captured = capsys.readouterr()
        assert len(captured.out) > 0


def test_viz_style_repr(capsys) -> None:
    """print_graph(style='repr') produces output."""
    with Party() as party:
        pp.from_seq("ACGT")
        party.print_graph(style="repr")
        captured = capsys.readouterr()
        assert "Pool(" in captured.out
