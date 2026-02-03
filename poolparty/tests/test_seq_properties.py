"""Tests for sequence property calculation functions."""

import pytest

from poolparty.utils.seq_properties import (
    _expand_iupac,
    calc_complexity,
    calc_dust,
    calc_gc,
    get_sites_for_enzymes,
    has_homopolymer,
    has_restriction_site,
)


class TestCalcGC:
    """Tests for calc_gc function."""

    def test_all_gc(self):
        """Test sequence with 100% GC content."""
        assert calc_gc("GGGGCCCC") == 1.0
        assert calc_gc("GCGCGCGC") == 1.0

    def test_no_gc(self):
        """Test sequence with 0% GC content."""
        assert calc_gc("AAAATTTT") == 0.0
        assert calc_gc("ATATATAT") == 0.0

    def test_half_gc(self):
        """Test sequence with 50% GC content."""
        assert calc_gc("ATGC") == 0.5
        assert calc_gc("AATTGGCC") == 0.5

    def test_case_insensitive(self):
        """Test that GC calculation is case-insensitive."""
        assert calc_gc("atgc") == 0.5
        assert calc_gc("AtGc") == 0.5

    def test_ignores_non_dna(self):
        """Test that non-DNA characters are ignored."""
        assert calc_gc("AT-GC") == 0.5
        assert calc_gc("AT.GC") == 0.5
        assert calc_gc("AT GC") == 0.5

    def test_empty_sequence(self):
        """Test empty sequence returns 0.0."""
        assert calc_gc("") == 0.0

    def test_no_valid_bases(self):
        """Test sequence with no valid bases returns 0.0."""
        assert calc_gc("---") == 0.0
        assert calc_gc("NNN") == 0.0


class TestCalcComplexity:
    """Tests for calc_complexity function."""

    def test_homopolymer_low_complexity(self):
        """Test that homopolymers have low complexity."""
        complexity = calc_complexity("AAAAAAAAAA")
        assert complexity < 0.2

    def test_random_high_complexity(self):
        """Test that random-looking sequences have high complexity."""
        complexity = calc_complexity("ACGTMKWSRY")
        assert complexity > 0.5

    def test_repeat_medium_complexity(self):
        """Test that simple repeats have medium complexity."""
        complexity = calc_complexity("ACGTACGTACGT")
        assert 0.3 < complexity < 0.8

    def test_short_sequence(self):
        """Test that very short sequences return high complexity."""
        assert calc_complexity("AC") == 1.0
        assert calc_complexity("A") == 1.0

    def test_custom_k_range(self):
        """Test custom k_range parameter."""
        # Homopolymer with just k=1 should be very low (1 unique / 4 possible = 0.25)
        complexity_k1 = calc_complexity("AAAAAAAAAA", k_range=(1,))
        assert complexity_k1 < 0.3

    def test_case_insensitive(self):
        """Test that complexity is case-insensitive."""
        assert calc_complexity("ACGT") == calc_complexity("acgt")


class TestCalcDust:
    """Tests for calc_dust function."""

    def test_homopolymer_high_dust(self):
        """Test that homopolymers have high DUST scores."""
        dust = calc_dust("AAAAAAAAAAAAAAAAA")  # 17 A's for dust > 5.0
        assert dust > 5.0

    def test_complex_low_dust(self):
        """Test that complex sequences have low DUST scores."""
        dust = calc_dust("ACGTACGTTGCA")
        assert dust < 2.0

    def test_short_sequence(self):
        """Test short sequences."""
        assert calc_dust("AA") == 0.0
        assert calc_dust("A") == 0.0
        assert calc_dust("") == 0.0

    def test_triplet_repeat(self):
        """Test triplet repeats have elevated DUST scores."""
        dust = calc_dust("ATGATGATGATG")
        assert dust > 1.0


class TestHasHomopolymer:
    """Tests for has_homopolymer function."""

    def test_detects_homopolymer(self):
        """Test detection of homopolymer runs."""
        assert has_homopolymer("ACGTAAAAAACGT", 4) is True  # 6 A's, > 4
        assert has_homopolymer("ACGTAAAACGT", 4) is False  # 4 A's, not > 4

    def test_no_homopolymer(self):
        """Test sequences without long homopolymers."""
        assert has_homopolymer("ACGTACGTACGT", 4) is False

    def test_case_insensitive(self):
        """Test case insensitivity."""
        assert has_homopolymer("ACGTaaaaaACGT", 4) is True

    def test_max_length_boundary(self):
        """Test boundary conditions for max_length."""
        seq = "ACGTAAAACGT"  # 4 A's
        assert has_homopolymer(seq, 3) is True  # > 3
        assert has_homopolymer(seq, 4) is False  # not > 4
        assert has_homopolymer(seq, 5) is False  # not > 5

    def test_invalid_max_length(self):
        """Test that invalid max_length raises error."""
        with pytest.raises(ValueError):
            has_homopolymer("ACGT", 0)


class TestExpandIupac:
    """Tests for _expand_iupac helper function."""

    def test_no_ambiguity(self):
        """Test sequence without ambiguity codes."""
        assert _expand_iupac("GAATTC") == ["GAATTC"]

    def test_single_r(self):
        """Test R ambiguity code (A/G)."""
        result = _expand_iupac("RAT")
        assert set(result) == {"AAT", "GAT"}

    def test_single_y(self):
        """Test Y ambiguity code (C/T)."""
        result = _expand_iupac("YAT")
        assert set(result) == {"CAT", "TAT"}

    def test_multiple_ambiguities(self):
        """Test multiple ambiguity codes."""
        result = _expand_iupac("RY")
        assert set(result) == {"AC", "AT", "GC", "GT"}

    def test_n_code(self):
        """Test N ambiguity code (any base)."""
        result = _expand_iupac("AN")
        assert len(result) == 4
        assert set(result) == {"AA", "AC", "AG", "AT"}


class TestHasRestrictionSite:
    """Tests for has_restriction_site function."""

    def test_finds_ecori(self):
        """Test detection of EcoRI site."""
        assert has_restriction_site("ACGTGAATTCACGT", ["GAATTC"]) is True

    def test_no_site(self):
        """Test sequence without the site."""
        assert has_restriction_site("ACGTACGTACGT", ["GAATTC"]) is False

    def test_multiple_sites(self):
        """Test with multiple sites to check."""
        assert has_restriction_site("ACGTGAATTCACGT", ["GGATCC", "GAATTC"]) is True
        assert has_restriction_site("ACGTGAATTCACGT", ["GGATCC", "AAGCTT"]) is False

    def test_iupac_site(self):
        """Test site with IUPAC ambiguity codes."""
        # BstYI recognizes RGATCY (R=A/G, Y=C/T)
        assert has_restriction_site("ACGTAGATCTACGT", ["RGATCY"]) is True
        assert has_restriction_site("ACGTGGATCCACGT", ["RGATCY"]) is True
        assert has_restriction_site("ACGTCGATCGACGT", ["RGATCY"]) is False

    def test_reverse_complement_check(self):
        """Test reverse complement checking."""
        # BsaI site is GGTCTC, RC is GAGACC
        assert has_restriction_site("ACGTGAGACCACGT", ["GGTCTC"], check_rc=True) is True
        assert has_restriction_site("ACGTGAGACCACGT", ["GGTCTC"], check_rc=False) is False

    def test_case_insensitive(self):
        """Test case insensitivity."""
        assert has_restriction_site("acgtgaattcacgt", ["GAATTC"]) is True
        assert has_restriction_site("ACGTGAATTCACGT", ["gaattc"]) is True


class TestGetSitesForEnzymes:
    """Tests for get_sites_for_enzymes function."""

    def test_single_enzyme(self):
        """Test getting site for a single enzyme."""
        sites = get_sites_for_enzymes(enzymes=["EcoRI"])
        assert sites == ["GAATTC"]

    def test_multiple_enzymes(self):
        """Test getting sites for multiple enzymes."""
        sites = get_sites_for_enzymes(enzymes=["EcoRI", "BamHI"])
        assert "GAATTC" in sites
        assert "GGATCC" in sites

    def test_preset(self):
        """Test using a preset name."""
        sites = get_sites_for_enzymes(enzymes=["golden_gate"])
        assert len(sites) > 0
        # BsaI should be in golden_gate preset
        assert "GGTCTC" in sites

    def test_custom_sites(self):
        """Test adding custom sites."""
        sites = get_sites_for_enzymes(sites=["CUSTOM", "SITES"])
        assert "CUSTOM" in sites
        assert "SITES" in sites

    def test_mixed(self):
        """Test mixing enzymes and custom sites."""
        sites = get_sites_for_enzymes(enzymes=["EcoRI"], sites=["CUSTOM"])
        assert "GAATTC" in sites
        assert "CUSTOM" in sites

    def test_unknown_enzyme_raises(self):
        """Test that unknown enzyme name raises error."""
        with pytest.raises(ValueError, match="Unknown enzyme"):
            get_sites_for_enzymes(enzymes=["NotARealEnzyme"])

    def test_case_insensitive_enzyme(self):
        """Test case-insensitive enzyme lookup."""
        sites = get_sites_for_enzymes(enzymes=["ecori"])
        assert "GAATTC" in sites

    def test_deduplication(self):
        """Test that duplicate sites are removed."""
        sites = get_sites_for_enzymes(enzymes=["EcoRI"], sites=["GAATTC"])
        assert sites.count("GAATTC") == 1
