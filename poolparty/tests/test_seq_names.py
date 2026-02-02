"""Integration tests for sequence name construction through DAG execution."""

import re

import pandas as pd

import poolparty as pp


def test_simple_operation_naming():
    """Test that a simple operation generates correct names."""
    with pp.Party():
        pool = pp.from_seq("ACGTACGT").mutagenize(num_mutations=1, mode="sequential", prefix="mut")

    df = pool.generate_library(num_cycles=1)

    # Check that names follow the pattern: mut_00, mut_01, mut_02, ... (zero-padded)
    for idx, name in enumerate(df["name"]):
        # Match mut_ followed by zero-padded number
        pattern = rf"^mut_0*{idx}$"
        assert re.match(pattern, name), (
            f"Expected name matching 'mut_{{zero-padded idx}}', got '{name}'"
        )

    # Verify no duplicate segments
    for name in df["name"]:
        parts = name.split(".")
        # Check for redundant consecutive segments
        for i in range(len(parts) - 1):
            assert parts[i] != parts[i + 1], f"Name has duplicate consecutive segments: {name}"


def test_operation_with_region_naming():
    """Test that operations with region parameter generate correct names."""
    with pp.Party():
        pool = pp.from_seq("AAA<reg>CCCCCC</reg>GGG").mutagenize(
            region="reg", num_mutations=1, mode="sequential", prefix="mut"
        )

    df = pool.generate_library(num_cycles=1)

    # Names should be mut_X format (NOT mut_X.mut_X or similar)
    for name in df["name"]:
        # Should start with mut_
        assert name.startswith("mut_"), f"Expected name to start with 'mut_', got '{name}'"

        # Should have no dots (single segment only)
        assert "." not in name, f"Expected single segment name, got '{name}'"

        # Ensure no duplicate segments like mut_0.mut_0
        assert "mut_" not in name[4:], f"Name appears to have duplicate 'mut_' segment: {name}"


def test_chained_operations_naming():
    """Test name construction through chained operations."""
    with pp.Party():
        pool = (
            pp.from_seq("ACGT<bc/>TGCA")
            .mutagenize(num_mutations=1, mode="random", num_states=5, prefix="mut")
            .insert_kmers(region="bc", length=3, mode="sequential", prefix="bc")
        )

    df = pool.generate_library(num_cycles=1, seed=42)

    # Names should follow pattern: mut_X.bc_Y
    for name in df["name"]:
        parts = name.split(".")
        assert len(parts) == 2, f"Expected 2 name parts, got {len(parts)}: {name}"
        assert parts[0].startswith("mut_"), f"First part should start with 'mut_': {name}"
        assert parts[1].startswith("bc_"), f"Second part should start with 'bc_': {name}"

        # Check no duplicates
        assert parts[0] != parts[1], f"Name has duplicate parts: {name}"


def test_stack_then_insert_kmers_naming():
    """Test the specific pattern from the MPRA example: stack -> insert_kmers."""
    with pp.Party():
        # Create two simple pools
        pool1 = pp.from_seq("AAAA<bc/>TTTT").mutagenize(
            num_mutations=1, mode="sequential", prefix="mut1"
        )
        pool2 = pp.from_seq("CCCC<bc/>GGGG").mutagenize(
            num_mutations=1, mode="sequential", prefix="mut2"
        )

        # Stack them and insert barcodes
        stacked = pp.stack([pool1, pool2]).insert_kmers(
            region="bc", length=2, mode="sequential", prefix="bc"
        )

    df = stacked.generate_library(num_cycles=1)

    # Names should be like: mut1_X.bc_Y or mut2_X.bc_Y
    for name in df["name"]:
        parts = name.split(".")
        assert len(parts) == 2, f"Expected 2 parts, got {len(parts)}: {name}"

        # First part should be mut1_X or mut2_X
        assert parts[0].startswith("mut1_") or parts[0].startswith("mut2_"), (
            f"First part should start with 'mut1_' or 'mut2_': {name}"
        )

        # Second part should be bc_Y
        assert parts[1].startswith("bc_"), f"Second part should start with 'bc_': {name}"

        # Check no duplicate segments (e.g., mut1_0.mut1_0.bc_0)
        name_str = str(name)
        # Split and check for any repeated segments
        segments = name_str.split(".")
        for i in range(len(segments) - 1):
            assert segments[i] != segments[i + 1], (
                f"Name has duplicate consecutive segments: {name}"
            )


def test_no_name_duplication_complex_pipeline():
    """Test complex pipeline similar to MPRA example doesn't produce duplicate names."""
    with pp.Party():
        template = pp.from_seq("AA<cre>CCCCCC</cre>GG<bc/>TT")

        mutated = template.mutagenize(
            region="cre", mutation_rate=0.1, mode="random", num_states=3, prefix="mut"
        )

        deleted = template.deletion_scan(
            region="cre", deletion_length=2, mode="sequential", prefix="del"
        )

        combined = pp.stack([mutated, deleted]).insert_kmers(
            region="bc", length=2, mode="sequential", prefix="bc"
        )

    df = combined.generate_library(num_cycles=1, seed=42)

    # Check all names for any duplicate segments
    for name in df["name"]:
        name_str = str(name)
        segments = name_str.split(".")

        # Check for consecutive duplicates
        for i in range(len(segments) - 1):
            assert segments[i] != segments[i + 1], (
                f"Found duplicate consecutive segments in: {name}"
            )

        # Check for any repeated segment at all (stricter test)
        # e.g., "mut_0.bc_0.mut_0" would fail this
        segment_set = set()
        for seg in segments:
            # Check if we've seen this exact segment before
            base_segment = seg.split("_")[0] if "_" in seg else seg
            if base_segment in ["mut", "del", "bc"]:
                # For operation names, the full segment should be unique
                assert seg not in segment_set, f"Found repeated segment '{seg}' in: {name}"
                segment_set.add(seg)


def test_repeat_naming():
    """Test that repeat generates correct names."""
    with pp.Party():
        pool = (
            pp.from_seq("ACGT")
            .mutagenize(num_mutations=1, mode="sequential", prefix="mut")
            .repeat(3, prefix="rep")
        )

    df = pool.generate_library(num_cycles=1)

    # Names should be like: mut_X.rep_Y
    for name in df["name"]:
        parts = name.split(".")
        assert len(parts) == 2, f"Expected 2 parts: {name}"
        assert parts[0].startswith("mut_"), f"First part should be 'mut_X': {name}"
        assert parts[1].startswith("rep_"), f"Second part should be 'rep_Y': {name}"

        # No duplicates
        assert parts[0] != parts[1]


def test_insertion_scan_naming():
    """Test insertion_scan generates correct names without duplication."""
    with pp.Party():
        sites = pp.from_seqs(["AAA", "TTT"], mode="sequential", prefix="site")
        pool = pp.from_seq("GGG<reg>CCCCCC</reg>GGG").insertion_scan(
            region="reg",
            ins_pool=sites,
            positions=[0, 3],
            mode="sequential",
            prefix="ins",
            prefix_position="pos",
            prefix_insert="site",
        )

    df = pool.generate_library(num_cycles=1)

    # Check names don't have duplicate segments
    for name in df["name"]:
        segments = name.split(".")
        # Should have pattern like: ins_X.pos_Y.site_Z (or similar)
        # Main check: no segment appears twice
        for i in range(len(segments) - 1):
            assert segments[i] != segments[i + 1], f"Duplicate consecutive segments in: {name}"


def test_shuffle_scan_naming():
    """Test shuffle_scan generates correct names."""
    with pp.Party():
        pool = pp.from_seq("AAA<reg>CCCCCC</reg>GGG").shuffle_scan(
            region="reg",
            shuffle_length=3,
            positions=[0, 3],
            mode="sequential",
            prefix="shuf",
            prefix_position="pos",
            prefix_shuffle="s",
        )

    df = pool.generate_library(num_cycles=1, seed=42)

    # Check no duplicate segments
    for name in df["name"]:
        segments = name.split(".")
        for i in range(len(segments) - 1):
            assert segments[i] != segments[i + 1], f"Duplicate consecutive segments in: {name}"


def test_stack_inactive_branch_prefixes_excluded():
    """Test that stacked pools don't leak prefixes from inactive branches.

    This is a regression test for a bug where operations with mode='random'
    that had a state but were inactive (state.value=None) would incorrectly
    use global_state for naming instead of returning no contribution.

    For example, with pools A (prefix='alpha') and B (prefix='beta') stacked:
    - States 0-N from pool A should have names like: alpha_0.bc_0, alpha_1.bc_1
    - States N+1-M from pool B should have names like: beta_0.bc_N+1, beta_1.bc_N+2

    The bug caused names like: alpha_0.beta_0.bc_0 (beta leaking into alpha rows)
    """
    with pp.Party():
        # Create two pools with distinct prefixes
        pool_alpha = pp.from_seq("AAAA<bc/>TTTT").mutagenize(
            num_mutations=1, mode="random", num_states=3, prefix="alpha"
        )
        pool_beta = pp.from_seq("CCCC<bc/>GGGG").mutagenize(
            num_mutations=1, mode="random", num_states=3, prefix="beta"
        )

        # Stack and add barcodes
        stacked = pp.stack([pool_alpha, pool_beta]).insert_kmers(
            region="bc", length=2, mode="random", prefix="bc"
        )

    df = stacked.generate_library(num_cycles=1, seed=42)

    # Total states = 3 (alpha) + 3 (beta) = 6
    assert len(df) == 6, f"Expected 6 rows, got {len(df)}"

    # Check each row's name only contains prefixes from its branch
    for idx, name in enumerate(df["name"]):
        if idx < 3:
            # First 3 states are from pool_alpha
            assert "alpha_" in name, f"Row {idx} should contain 'alpha_': {name}"
            assert "beta_" not in name, f"Row {idx} should NOT contain 'beta_': {name}"
        else:
            # Last 3 states are from pool_beta
            assert "beta_" in name, f"Row {idx} should contain 'beta_': {name}"
            assert "alpha_" not in name, f"Row {idx} should NOT contain 'alpha_': {name}"

        # All rows should have barcode
        assert "bc_" in name, f"Row {idx} should contain 'bc_': {name}"


def test_stack_multiple_branches_name_isolation():
    """Test that names are isolated to their respective branches with 5 pools.

    Regression test ensuring inactive branch prefixes never contaminate names.
    """
    with pp.Party():
        # Create 5 pools with distinct prefixes using mutagenize (mode='random')
        # All using the same base sequence with bc region
        pool_a = pp.from_seq("AA<bc/>TT").mutagenize(
            num_mutations=1, mode="random", num_states=2, prefix="poolA"
        )
        pool_b = pp.from_seq("CC<bc/>GG").mutagenize(
            num_mutations=1, mode="random", num_states=2, prefix="poolB"
        )
        pool_c = pp.from_seq("GG<bc/>CC").mutagenize(
            num_mutations=1, mode="random", num_states=2, prefix="poolC"
        )
        pool_d = pp.from_seq("TT<bc/>AA").mutagenize(
            num_mutations=1, mode="random", num_states=2, prefix="poolD"
        )
        pool_e = pp.from_seq("AC<bc/>GT").mutagenize(
            num_mutations=1, mode="random", num_states=2, prefix="poolE"
        )

        # Stack all and add barcodes
        stacked = pp.stack([pool_a, pool_b, pool_c, pool_d, pool_e]).insert_kmers(
            region="bc", length=2, mode="random", prefix="bc"
        )

    df = stacked.generate_library(num_cycles=1, seed=42)

    # Should have 10 states total (2 per pool)
    assert len(df) == 10, f"Expected 10 rows, got {len(df)}"

    # Define which prefixes belong to which rows
    branch_prefixes = {
        (0, 1): "poolA",
        (2, 3): "poolB",
        (4, 5): "poolC",
        (6, 7): "poolD",
        (8, 9): "poolE",
    }
    all_prefixes = ["poolA", "poolB", "poolC", "poolD", "poolE"]

    for idx, name in enumerate(df["name"]):
        # Find which branch this row belongs to
        expected_prefix = None
        for (start, end), prefix in branch_prefixes.items():
            if start <= idx <= end:
                expected_prefix = prefix
                break

        assert expected_prefix is not None, f"Row {idx} doesn't belong to any branch"

        # Check the expected prefix is present
        assert expected_prefix in name, f"Row {idx} should contain '{expected_prefix}': {name}"

        # Check NO OTHER branch prefixes are present
        for prefix in all_prefixes:
            if prefix != expected_prefix:
                assert prefix not in name, (
                    f"Row {idx} should NOT contain '{prefix}' (belongs to {expected_prefix}): {name}"
                )

        # Barcode should be present
        assert "bc_" in name, f"Row {idx} should contain 'bc_': {name}"


def test_fixed_operation_prefix():
    """Test that fixed operations can have a prefix that adds a constant label."""
    with pp.Party():
        # Fixed operation with prefix
        pool = pp.from_seq("ACGT", prefix="bg").upper(prefix="up")

    df = pool.generate_library(num_cycles=1)

    # Names should be "bg.up" - just constant labels, no indices
    assert len(df) == 1
    name = df.loc[0, "name"]
    assert "bg" in name, f"Name should contain 'bg': {name}"
    assert "up" in name, f"Name should contain 'up': {name}"


def test_fixed_and_variable_ops_prefix():
    """Test prefix on both fixed and variable operations."""
    with pp.Party():
        pool = (
            pp.from_seq("ACGTACGT", prefix="bg")
            .mutagenize(num_mutations=1, mode="sequential", prefix="mut")
            .upper(prefix="upper")
        )

    df = pool.generate_library(num_cycles=1)

    # Names should be like: bg.mut_00.upper, bg.mut_01.upper, ... (zero-padded)
    for idx, name in enumerate(df["name"]):
        assert "bg" in name, f"Name should contain 'bg': {name}"
        # Match mut_ followed by zero-padded number
        pattern = rf"mut_0*{idx}(?:\D|$)"
        assert re.search(pattern, name), f"Name should contain 'mut_' with index {idx}: {name}"
        assert "upper" in name, f"Name should contain 'upper': {name}"


def test_filter_with_prefix():
    """Test that filter operation can have a prefix."""
    with pp.Party():
        pool = pp.from_seqs(["AAAA", "CCCC", "GGGG"], mode="sequential", prefix="seq").filter(
            lambda s: s.startswith("A"), prefix="filtered"
        )

    df = pool.generate_library(num_seqs=3)

    # First row passes filter
    assert df.loc[0, "seq"] == "AAAA"
    name = df.loc[0, "name"]
    assert "seq_0" in name, f"Name should contain 'seq_0': {name}"
    assert "filtered" in name, f"Name should contain 'filtered': {name}"

    # Filtered rows have None/nan name
    assert pd.isna(df.loc[1, "name"])
    assert pd.isna(df.loc[2, "name"])
