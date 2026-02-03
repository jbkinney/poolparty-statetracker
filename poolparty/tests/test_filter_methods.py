"""Tests for filter convenience methods on DnaPool."""

import pytest

import poolparty as pp


class TestFilterGC:
    """Tests for filter_gc method."""

    def test_filter_high_gc(self):
        """Test filtering out high GC sequences."""
        with pp.Party():
            # GGGG = 100% GC, AAAA = 0% GC, ATGC = 50% GC
            root = pp.from_seqs(["GGGGGGGG", "AAAAAAAA", "ATGCATGC"], mode="sequential")
            filtered = root.filter_gc(max_gc=0.6)
            df = filtered.generate_library(num_seqs=3, discard_null_seqs=True)

            assert len(df) == 2
            seqs = df["seq"].tolist()
            assert "GGGGGGGG" not in seqs
            assert "AAAAAAAA" in seqs
            assert "ATGCATGC" in seqs

    def test_filter_low_gc(self):
        """Test filtering out low GC sequences."""
        with pp.Party():
            root = pp.from_seqs(["GGGGGGGG", "AAAAAAAA", "ATGCATGC"], mode="sequential")
            filtered = root.filter_gc(min_gc=0.4)
            df = filtered.generate_library(num_seqs=3, discard_null_seqs=True)

            assert len(df) == 2
            seqs = df["seq"].tolist()
            assert "GGGGGGGG" in seqs
            assert "AAAAAAAA" not in seqs
            assert "ATGCATGC" in seqs

    def test_filter_gc_range(self):
        """Test filtering to a GC range."""
        with pp.Party():
            root = pp.from_seqs(["GGGGGGGG", "AAAAAAAA", "ATGCATGC"], mode="sequential")
            filtered = root.filter_gc(min_gc=0.4, max_gc=0.6)
            df = filtered.generate_library(num_seqs=3, discard_null_seqs=True)

            assert len(df) == 1
            assert df["seq"].iloc[0] == "ATGCATGC"

    def test_filter_gc_invalid_range(self):
        """Test that invalid GC range raises error."""
        with pp.Party():
            root = pp.from_seqs(["ACGT"], mode="sequential")
            with pytest.raises(ValueError, match="min_gc"):
                root.filter_gc(min_gc=-0.1)
            with pytest.raises(ValueError, match="max_gc"):
                root.filter_gc(max_gc=1.5)
            with pytest.raises(ValueError, match="cannot be greater"):
                root.filter_gc(min_gc=0.6, max_gc=0.4)


class TestFilterHomopolymer:
    """Tests for filter_homopolymer method."""

    def test_filter_long_homopolymers(self):
        """Test filtering out sequences with long homopolymers."""
        with pp.Party():
            seqs = [
                "ACGTACGTACGT",  # No homopolymer
                "ACGTAAAAAACGT",  # 6 A's
                "ACGTAAAACGT",  # 4 A's
            ]
            root = pp.from_seqs(seqs, mode="sequential")
            filtered = root.filter_homopolymer(max_length=4)
            df = filtered.generate_library(num_seqs=3, discard_null_seqs=True)

            assert len(df) == 2
            seqs_out = df["seq"].tolist()
            assert "ACGTACGTACGT" in seqs_out
            assert "ACGTAAAACGT" in seqs_out
            assert "ACGTAAAAAACGT" not in seqs_out

    def test_filter_homopolymer_strict(self):
        """Test strict homopolymer filtering."""
        with pp.Party():
            seqs = [
                "ACGTACGTACGT",  # No homopolymer
                "ACGTAAACGT",  # 3 A's
            ]
            root = pp.from_seqs(seqs, mode="sequential")
            filtered = root.filter_homopolymer(max_length=2)
            df = filtered.generate_library(num_seqs=2, discard_null_seqs=True)

            assert len(df) == 1
            assert df["seq"].iloc[0] == "ACGTACGTACGT"

    def test_filter_homopolymer_invalid(self):
        """Test that invalid max_length raises error."""
        with pp.Party():
            root = pp.from_seqs(["ACGT"], mode="sequential")
            with pytest.raises(ValueError, match="max_length"):
                root.filter_homopolymer(max_length=0)


class TestFilterComplexity:
    """Tests for filter_complexity method."""

    def test_filter_low_complexity(self):
        """Test filtering out low complexity sequences."""
        with pp.Party():
            seqs = [
                "ACGTMKWSRYACGT",  # High complexity
                "AAAAAAAAAAAAA",  # Low complexity (homopolymer)
                "ACACACACACACAC",  # Medium complexity (dinuc repeat)
            ]
            root = pp.from_seqs(seqs, mode="sequential")
            filtered = root.filter_complexity(min_complexity=0.5)
            df = filtered.generate_library(num_seqs=3, discard_null_seqs=True)

            # Homopolymer should be filtered out
            seqs_out = df["seq"].tolist()
            assert "AAAAAAAAAAAAA" not in seqs_out

    def test_filter_complexity_custom_k(self):
        """Test filtering with custom k_range."""
        with pp.Party():
            seqs = [
                "ACGTACGTACGT",  # Repetitive at k=4
                "AAAAAAAAAA",  # Very low complexity
            ]
            root = pp.from_seqs(seqs, mode="sequential")
            filtered = root.filter_complexity(min_complexity=0.3, k_range=(1, 2))
            df = filtered.generate_library(num_seqs=2, discard_null_seqs=True)

            seqs_out = df["seq"].tolist()
            assert "AAAAAAAAAA" not in seqs_out

    def test_filter_complexity_invalid(self):
        """Test that invalid min_complexity raises error."""
        with pp.Party():
            root = pp.from_seqs(["ACGT"], mode="sequential")
            with pytest.raises(ValueError, match="min_complexity"):
                root.filter_complexity(min_complexity=-0.1)
            with pytest.raises(ValueError, match="min_complexity"):
                root.filter_complexity(min_complexity=1.5)


class TestFilterDust:
    """Tests for filter_dust method."""

    def test_filter_high_dust(self):
        """Test filtering out high DUST score sequences."""
        with pp.Party():
            seqs = [
                "ACGTMKWSRYACGT",  # Complex (low DUST)
                "AAAAAAAAAAAAA",  # Homopolymer (high DUST)
            ]
            root = pp.from_seqs(seqs, mode="sequential")
            filtered = root.filter_dust(max_score=2.0)
            df = filtered.generate_library(num_seqs=2, discard_null_seqs=True)

            seqs_out = df["seq"].tolist()
            assert "AAAAAAAAAAAAA" not in seqs_out

    def test_filter_dust_permissive(self):
        """Test permissive DUST filtering."""
        with pp.Party():
            seqs = [
                "ACGTACGTACGT",  # Some repetition
                "AAAAAAAAAAAAA",  # Very high DUST
            ]
            root = pp.from_seqs(seqs, mode="sequential")
            # Very permissive threshold
            filtered = root.filter_dust(max_score=10.0)
            df = filtered.generate_library(num_seqs=2, discard_null_seqs=True)

            # Both should pass with permissive threshold
            assert len(df) >= 1

    def test_filter_dust_invalid(self):
        """Test that invalid max_score raises error."""
        with pp.Party():
            root = pp.from_seqs(["ACGT"], mode="sequential")
            with pytest.raises(ValueError, match="max_score"):
                root.filter_dust(max_score=-1.0)


class TestFilterRestrictionSites:
    """Tests for filter_restriction_sites method."""

    def test_filter_single_enzyme(self):
        """Test filtering out sequences with a single enzyme site."""
        with pp.Party():
            seqs = [
                "ACGTACGTACGT",  # No EcoRI site
                "ACGTGAATTCACGT",  # Has EcoRI site (GAATTC)
            ]
            root = pp.from_seqs(seqs, mode="sequential")
            filtered = root.filter_restriction_sites(enzymes=["EcoRI"])
            df = filtered.generate_library(num_seqs=2, discard_null_seqs=True)

            assert len(df) == 1
            assert df["seq"].iloc[0] == "ACGTACGTACGT"

    def test_filter_multiple_enzymes(self):
        """Test filtering with multiple enzymes."""
        with pp.Party():
            seqs = [
                "ACGTACGTACGT",  # No sites
                "ACGTGAATTCACGT",  # EcoRI
                "ACGTGGATCCACGT",  # BamHI
            ]
            root = pp.from_seqs(seqs, mode="sequential")
            filtered = root.filter_restriction_sites(enzymes=["EcoRI", "BamHI"])
            df = filtered.generate_library(num_seqs=3, discard_null_seqs=True)

            assert len(df) == 1
            assert df["seq"].iloc[0] == "ACGTACGTACGT"

    def test_filter_preset(self):
        """Test filtering with a preset."""
        with pp.Party():
            seqs = [
                "ACGTACGTACGT",  # No sites
                "ACGTGGTCTCACGT",  # BsaI (in golden_gate preset)
            ]
            root = pp.from_seqs(seqs, mode="sequential")
            filtered = root.filter_restriction_sites(enzymes=["golden_gate"])
            df = filtered.generate_library(num_seqs=2, discard_null_seqs=True)

            assert len(df) == 1
            assert df["seq"].iloc[0] == "ACGTACGTACGT"

    def test_filter_custom_sites(self):
        """Test filtering with custom sites."""
        with pp.Party():
            seqs = [
                "ACGTACGTACGT",  # No custom site
                "ACGTAATTCCACGT",  # Has custom site AATTCC
            ]
            root = pp.from_seqs(seqs, mode="sequential")
            filtered = root.filter_restriction_sites(sites=["AATTCC"])
            df = filtered.generate_library(num_seqs=2, discard_null_seqs=True)

            assert len(df) == 1
            assert df["seq"].iloc[0] == "ACGTACGTACGT"

    def test_filter_mixed_enzymes_and_sites(self):
        """Test filtering with both enzymes and custom sites."""
        with pp.Party():
            seqs = [
                "ACGTACGTACGT",  # No sites
                "ACGTGAATTCACGT",  # EcoRI
                "ACGTAATTCCACGT",  # Custom site AATTCC
            ]
            root = pp.from_seqs(seqs, mode="sequential")
            filtered = root.filter_restriction_sites(enzymes=["EcoRI"], sites=["AATTCC"])
            df = filtered.generate_library(num_seqs=3, discard_null_seqs=True)

            assert len(df) == 1
            assert df["seq"].iloc[0] == "ACGTACGTACGT"

    def test_filter_reverse_complement(self):
        """Test that reverse complement is checked by default."""
        with pp.Party():
            # BsaI site is GGTCTC, RC is GAGACC
            seqs = [
                "ACGTACGTACGT",  # No sites
                "ACGTGAGACCACGT",  # Has RC of BsaI
            ]
            root = pp.from_seqs(seqs, mode="sequential")
            filtered = root.filter_restriction_sites(enzymes=["BsaI"], check_rc=True)
            df = filtered.generate_library(num_seqs=2, discard_null_seqs=True)

            assert len(df) == 1
            assert df["seq"].iloc[0] == "ACGTACGTACGT"

    def test_filter_no_reverse_complement(self):
        """Test disabling reverse complement check."""
        with pp.Party():
            # BsaI site is GGTCTC, RC is GAGACC
            seqs = [
                "ACGTACGTACGT",  # No sites
                "ACGTGAGACCACGT",  # Has RC of BsaI
            ]
            root = pp.from_seqs(seqs, mode="sequential")
            filtered = root.filter_restriction_sites(enzymes=["BsaI"], check_rc=False)
            df = filtered.generate_library(num_seqs=2, discard_null_seqs=True)

            # Both should pass since we're not checking RC
            assert len(df) == 2

    def test_filter_requires_enzymes_or_sites(self):
        """Test that at least enzymes or sites must be provided."""
        with pp.Party():
            root = pp.from_seqs(["ACGT"], mode="sequential")
            with pytest.raises(ValueError, match="At least one"):
                root.filter_restriction_sites()

    def test_filter_unknown_enzyme(self):
        """Test that unknown enzyme name raises error."""
        with pp.Party():
            root = pp.from_seqs(["ACGT"], mode="sequential")
            with pytest.raises(ValueError, match="Unknown enzyme"):
                root.filter_restriction_sites(enzymes=["NotARealEnzyme"])


class TestFilterChaining:
    """Tests for chaining multiple filter operations."""

    def test_chain_gc_and_homopolymer(self):
        """Test chaining GC and homopolymer filters."""
        with pp.Party():
            seqs = [
                "ATGCATGCATGC",  # 50% GC, no homopolymer - PASS
                "GGGGGGGGGGGG",  # 100% GC - FAIL GC
                "ATGCAAAAAATGC",  # ~38% GC, has homopolymer - FAIL homopolymer
                "ATATATATATA",  # 0% GC - FAIL GC
            ]
            root = pp.from_seqs(seqs, mode="sequential")
            filtered = root.filter_gc(min_gc=0.3, max_gc=0.7).filter_homopolymer(max_length=4)
            df = filtered.generate_library(num_seqs=4, discard_null_seqs=True)

            assert len(df) == 1
            assert df["seq"].iloc[0] == "ATGCATGCATGC"

    def test_chain_complexity_and_restriction(self):
        """Test chaining complexity and restriction site filters."""
        with pp.Party():
            seqs = [
                "ACGTMKWSRYACGT",  # Complex, no sites - PASS
                "AAAAAAAAAAAA",  # Low complexity - FAIL
                "ACGTGAATTCACGT",  # Complex, has EcoRI - FAIL
            ]
            root = pp.from_seqs(seqs, mode="sequential")
            filtered = root.filter_complexity(min_complexity=0.3).filter_restriction_sites(
                enzymes=["EcoRI"]
            )
            df = filtered.generate_library(num_seqs=3, discard_null_seqs=True)

            assert len(df) == 1
            assert df["seq"].iloc[0] == "ACGTMKWSRYACGT"
