"""Tests for the stats readout."""

import json
from importlib import import_module

import pytest

import poolparty as pp
from poolparty.utils.seq_properties import (
    calc_dust,
    calc_gc,
    has_homopolymer,
    has_restriction_site,
    longest_homopolymer,
)
from poolparty.utils.stats_utils import _stats_from_seqs

# pp.stats is the function, so the module it lives in has to be asked for by
# name, exactly as for poolparty.generate_library.
stats_module = import_module("poolparty.stats")

# Three of these five pass a GC <= 0.5 filter.
MIXED_GC = ["AAAATTTT", "GGGGCCCC", "AATTAATT", "GGCCGGCC", "ACGTACGT"]


def filtered_pool():
    """A five-state pool whose filter rejects two sequences."""
    return pp.from_seqs(MIXED_GC, mode="sequential").filter_gc(max_gc=0.5)


class TestStatsFromSeqs:
    """Tests for the counting layer, which needs no Pool."""

    def test_counts_duplicates(self):
        """A repeated sequence counts once as unique and once as a duplicate."""
        result = _stats_from_seqs(["ACGT", "ACGT", "TGCA"])

        assert result["num_generated_seqs"] == 3
        assert result["num_valid_seqs"] == 3
        assert result["num_unique_seqs"] == 2
        assert result["num_duplicate_seqs"] == 1
        assert result["frac_duplicate_seqs"] == pytest.approx(1 / 3)
        assert result["max_seq_copies"] == 2

    def test_duplicates_count_excess_copies(self):
        """A sequence appearing three times contributes two duplicates, not three."""
        result = _stats_from_seqs(["ACGT"] * 3)

        assert result["num_unique_seqs"] == 1
        assert result["num_duplicate_seqs"] == 2
        assert result["max_seq_copies"] == 3

    def test_nulls_are_reported_as_filtered_out(self):
        """None marks a sequence a filter rejected, and is not counted as valid."""
        result = _stats_from_seqs(["ACGT", None, "TGCA", None])

        assert result["num_generated_seqs"] == 4
        assert result["num_filtered_out_seqs"] == 2
        assert result["num_valid_seqs"] == 2

    def test_no_sequences_survive(self):
        """Everything filtered out gives zero counts, not a division by zero."""
        result = _stats_from_seqs([None, None])

        assert result["num_valid_seqs"] == 0
        assert result["frac_duplicate_seqs"] == 0.0
        assert result["max_seq_copies"] == 0
        assert "gc_mean" not in result
        assert "hamming_min" not in result

    def test_single_sequence_has_no_pairs(self):
        """One sequence has no pair to compare, so the distance keys are absent."""
        result = _stats_from_seqs(["ACGT"])

        assert result["num_valid_seqs"] == 1
        assert result["gc_mean"] == pytest.approx(0.5)
        assert "hamming_min" not in result

    def test_the_funnel_adds_up(self):
        """generated = filtered out + valid, and valid = unique + duplicates."""
        result = _stats_from_seqs(["ACGT", "ACGT", None, "TGCA", None, "GGCC"])

        assert (
            result["num_generated_seqs"]
            == result["num_filtered_out_seqs"] + result["num_valid_seqs"]
        )
        assert result["num_valid_seqs"] == result["num_unique_seqs"] + result["num_duplicate_seqs"]


class TestComposition:
    """Tests for the composition counts on real pools."""

    def test_filter_leaves_num_states_unchanged(self):
        """A filter replaces rejected sequences rather than removing them."""
        with pp.Party():
            stats = filtered_pool().stats(show_progress=False)

            assert stats["num_states"] == 5
            assert stats["num_generated_seqs"] == 5
            assert stats["num_filtered_out_seqs"] == 2
            assert stats["num_valid_seqs"] == 3
            assert stats["num_duplicate_seqs"] == 0

    def test_repeat_duplicates_by_design(self):
        """repeat asks for copies, so the copies show up as duplicates."""
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential").repeat(times=3)

            stats = pool.stats(show_progress=False)

            assert stats["num_generated_seqs"] == 6
            assert stats["num_unique_seqs"] == 2
            assert stats["num_duplicate_seqs"] == 4
            assert stats["max_seq_copies"] == 3

    def test_sampling_with_replacement_duplicates(self):
        """sample defaults to with_replacement, which can draw a state twice."""
        with pp.Party():
            pool = pp.sample(
                pp.from_seqs(["ACGT", "TGCA", "GGCC"], mode="sequential"),
                num_seqs=30,
                seed=0,
            )

            stats = pool.stats(show_progress=False)

            assert stats["num_generated_seqs"] == 30
            assert stats["num_unique_seqs"] <= 3
            assert stats["num_duplicate_seqs"] >= 27

    def test_random_draws_collide(self):
        """Random sampling revisits sequences, which num_states cannot show."""
        with pp.Party():
            pool = pp.from_seq("ACGTACGTAC").mutagenize(num_mutations=2, mode="random")

            stats = pool.stats(num_seqs=500, show_progress=False)

            # 405 distinct two-base mutants exist, so 500 draws must collide.
            assert stats["num_duplicate_seqs"] > 0
            assert stats["num_unique_seqs"] < 500


class TestDesignKind:
    """Tests telling a design with a fixed size from one without."""

    def test_sequential_design_is_closed(self):
        """A design built only from sequential operations has a fixed size."""
        with pp.Party():
            pool = pp.from_seq("ACGTACGTAC").mutagenize(num_mutations=1, mode="sequential")

            stats = pool.stats(show_progress=False)

            assert stats["open_ended"] is False
            assert stats["num_states"] == 30
            assert stats["frac_design_covered"] == pytest.approx(1.0)

    def test_random_with_num_states_is_closed(self):
        """Giving a random operation num_states fixes which sequences exist."""
        with pp.Party():
            pool = pp.from_seq("ACGTACGTAC").mutagenize(
                num_mutations=2, mode="random", num_states=50
            )

            stats = pool.stats(show_progress=False)

            assert stats["open_ended"] is False
            assert stats["num_states"] == 50
            assert stats["num_generated_seqs"] == 50

    def test_random_without_num_states_is_open_ended(self):
        """Without num_states the design has no total, so a count is required."""
        with pp.Party():
            pool = pp.from_seq("ACGTACGTAC").mutagenize(num_mutations=2, mode="random")

            with pytest.raises(ValueError, match="no total number of sequences"):
                pool.stats(show_progress=False)

            stats = pool.stats(num_seqs=100, show_progress=False)

            assert stats["open_ended"] is True
            assert stats["num_states"] is None
            assert stats["frac_design_covered"] is None

    def test_num_cycles_is_refused_when_there_are_no_passes_to_make(self):
        """A design with no fixed size has no state space to cycle through."""
        with pp.Party():
            pool = pp.from_seq("ACGTACGTAC").mutagenize(num_mutations=2, mode="random")

            with pytest.raises(ValueError, match="num_cycles does not apply"):
                pool.stats(num_cycles=1, show_progress=False)

    def test_open_ended_step_makes_the_whole_design_open_ended(self):
        """One unfixed random step is enough, even behind a sequential source."""
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential").mutagenize(
                num_mutations=1, mode="random"
            )

            assert pool.num_states == 2  # a floor, not a total

            stats = pool.stats(num_seqs=20, show_progress=False)

            assert stats["open_ended"] is True
            assert stats["num_states"] is None


class TestAutoLimit:
    """Tests for measuring a design without being told how much of it to read."""

    def test_small_design_is_measured_in_full(self):
        """A design under the limit needs no count."""
        with pp.Party():
            stats = pp.from_seqs(MIXED_GC, mode="sequential").stats(show_progress=False)

            assert stats["num_generated_seqs"] == 5

    def test_large_design_refuses_and_says_how_to_proceed(self):
        """A design over the limit is not enumerated by accident."""
        with pp.Party():
            pool = pp.from_seq("ACGT" * 50)
            for _ in range(3):
                pool = pool.mutagenize(num_mutations=1, mode="sequential")

            assert pool.num_states > 1_000_000

            with pytest.raises(ValueError, match="above the .* limit"):
                pool.stats(show_progress=False)

    def test_num_cycles_generates_that_many_passes(self):
        """num_cycles is a documented escape hatch, so it must actually work."""
        with pp.Party():
            stats = filtered_pool().stats(num_cycles=2, show_progress=False)

            assert stats["num_generated_seqs"] == 10
            assert stats["num_filtered_out_seqs"] == 4
            assert stats["num_valid_seqs"] == 6
            assert stats["num_unique_seqs"] == 3
            assert stats["num_duplicate_seqs"] == 3
            assert stats["frac_design_covered"] == pytest.approx(2.0)

    def test_an_explicit_count_is_never_capped(self):
        """Naming a count overrides the limit, on either argument."""
        with pp.Party():
            pool = pp.from_seq("ACGT" * 50)
            for _ in range(3):
                pool = pool.mutagenize(num_mutations=1, mode="sequential")

            stats = pool.stats(num_seqs=50, show_progress=False)

            assert stats["num_generated_seqs"] == 50
            assert stats["num_states"] == pool.num_states
            assert stats["frac_design_covered"] < 1e-6


class TestPairwiseHamming:
    """Tests for the pairwise distance statistics."""

    def test_exact_distances_on_a_hand_checked_pool(self):
        """Three known sequences give known distances: 1, 3 and 3."""
        with pp.Party():
            pool = pp.from_seqs(["AAAA", "AAAC", "ACGT"], mode="sequential")

            stats = pool.stats(show_progress=False)

            assert stats["hamming_exact"] is True
            assert stats["hamming_seqs_compared"] == 3
            assert stats["hamming_min"] == 1
            assert stats["hamming_max"] == 3
            assert stats["hamming_mean"] == pytest.approx(7 / 3)

    def test_duplicates_put_the_minimum_at_zero(self):
        """Two identical sequences differ nowhere, so an exact minimum is 0."""
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "ACGT", "TGCA"], mode="sequential")

            stats = pool.stats(show_progress=False)

            assert stats["num_duplicate_seqs"] == 1
            assert stats["hamming_min"] == 0

    def test_subsampling_is_reported_as_inexact(self):
        """Comparing fewer sequences than exist is flagged, not hidden."""
        with pp.Party():
            pool = pp.from_seq("ACGTACGTAC").mutagenize(num_mutations=1, mode="sequential")

            stats = pool.stats(max_hamming_seqs=5, show_progress=False)

            assert stats["hamming_exact"] is False
            assert stats["hamming_seqs_compared"] == 5

    def test_the_same_seed_gives_the_same_subsample(self):
        """A report is reproducible, including which sequences were compared."""
        with pp.Party():
            pool = pp.from_seq("ACGTACGTAC").mutagenize(num_mutations=1, mode="sequential")

            first = pool.stats(max_hamming_seqs=5, seed=7, show_progress=False)
            second = pool.stats(max_hamming_seqs=5, seed=7, show_progress=False)

            assert dict(first) == dict(second)

    def test_a_different_seed_gives_a_different_subsample(self):
        """The seed must actually reach the subsample, not just be accepted."""
        with pp.Party():
            pool = pp.from_seq("ACGTACGTACGTACGTACGT").mutagenize(
                num_mutations=2, mode="sequential"
            )

            first = pool.stats(max_hamming_seqs=20, seed=1, show_progress=False)
            second = pool.stats(max_hamming_seqs=20, seed=2, show_progress=False)

            assert first["hamming_mean"] != second["hamming_mean"]

    def test_every_pair_is_compared_across_block_boundaries(self):
        """The comparison is tiled, and the tiling must not change the answer.

        _HAMMING_BLOCK equals the default max_hamming_seqs, so the off-diagonal
        tiles are never reached at any default setting. Shrinking the block is
        the only way to exercise them.
        """
        with pp.Party():
            pool = pp.from_seqs(["AAAA", "AAAC", "ACGT"], mode="sequential")

            unblocked = pool.stats(show_progress=False)

            import poolparty.utils.stats_utils as stats_utils

            original = stats_utils._HAMMING_BLOCK
            stats_utils._HAMMING_BLOCK = 2
            try:
                blocked = pool.stats(show_progress=False)
            finally:
                stats_utils._HAMMING_BLOCK = original

            assert dict(blocked) == dict(unblocked)
            assert blocked["hamming_min"] == 1
            assert blocked["hamming_max"] == 3

    def test_distances_are_skipped_when_asked(self):
        """max_hamming_seqs=None omits the quadratic part entirely."""
        with pp.Party():
            stats = filtered_pool().stats(max_hamming_seqs=None, show_progress=False)

            assert not any(key.startswith("hamming") for key in stats)

    def test_a_large_comparison_warns_first(self, monkeypatch):
        """The caller hears about a slow comparison before it starts.

        The real threshold is 20,000 sequences, which is too slow to compare in
        a test, so it is lowered here rather than skipped.
        """
        monkeypatch.setattr(stats_module, "_HAMMING_WARN_ABOVE", 2)

        with pp.Party():
            pool = pp.from_seqs(MIXED_GC, mode="sequential")

            with pytest.warns(UserWarning, match="pairwise"):
                pool.stats(show_progress=False)


class TestReproducibility:
    """Tests that a report depends on the design and nothing else."""

    def _sampling_pool(self):
        return pp.from_seq("ACGTACGTACGTACGTACGT").mutagenize(num_mutations=2, mode="random")

    def test_repeated_calls_agree(self):
        """Two calls on the same pool must give the same numbers."""
        with pp.Party():
            pool = self._sampling_pool()

            first = pool.stats(num_seqs=200, show_progress=False)
            second = pool.stats(num_seqs=200, show_progress=False)

            assert dict(first) == dict(second)

    def test_the_pools_history_does_not_change_the_report(self):
        """Generating from a pool beforehand must not move its statistics."""
        with pp.Party():
            fresh = self._sampling_pool().stats(num_seqs=200, show_progress=False)

        with pp.Party():
            pool = self._sampling_pool()
            pool.generate_library(num_seqs=5, seed=99)

            assert dict(pool.stats(num_seqs=200, show_progress=False)) == dict(fresh)

    def test_a_seed_changes_which_sequences_are_drawn(self):
        """The seed must reach generation, not only the distance subsample."""
        with pp.Party():
            pool = self._sampling_pool()

            first = pool.stats(num_seqs=200, seed=1, show_progress=False)
            second = pool.stats(num_seqs=200, seed=2, show_progress=False)

            assert first["gc_mean"] != second["gc_mean"]

    def test_chunked_generation_matches_a_single_pass(self):
        """More sequences than one chunk must not disturb the traversal."""
        with pp.Party():
            pool = pp.from_seq("ACGT" * 10).mutagenize(num_mutations=1, mode="sequential")
            assert pool.num_states == 120

            stats = pool.stats(num_seqs=1500, show_progress=False)

            # 1500 rows over 120 states: every sequence, twelve times, then 60 more.
            assert stats["num_generated_seqs"] == 1500
            assert stats["num_unique_seqs"] == 120


class TestDnaStats:
    """Tests that the DNA statistics agree with the helpers they come from."""

    def test_gc_matches_calc_gc(self):
        """GC is calc_gc over the same sequences."""
        with pp.Party():
            stats = pp.from_seqs(MIXED_GC, mode="sequential").stats(show_progress=False)

            expected = [calc_gc(seq) for seq in MIXED_GC]
            assert stats["gc_min"] == pytest.approx(min(expected))
            assert stats["gc_max"] == pytest.approx(max(expected))
            assert stats["gc_mean"] == pytest.approx(sum(expected) / len(expected))

    def test_dust_matches_calc_dust(self):
        """Repetitiveness is calc_dust over the same sequences."""
        with pp.Party():
            stats = pp.from_seqs(MIXED_GC, mode="sequential").stats(show_progress=False)

            expected = [calc_dust(seq) for seq in MIXED_GC]
            assert stats["dust_max"] == pytest.approx(max(expected))
            assert stats["dust_mean"] == pytest.approx(sum(expected) / len(expected))

    def test_homopolymers_match_the_helpers(self):
        """The longest run and the fraction over the limit agree with the helpers."""
        with pp.Party():
            stats = pp.from_seqs(MIXED_GC, mode="sequential").stats(
                max_homopolymer_run=3, show_progress=False
            )

            assert stats["longest_homopolymer"] == max(longest_homopolymer(seq) for seq in MIXED_GC)
            over = sum(has_homopolymer(seq, 3) for seq in MIXED_GC)
            assert stats["frac_seqs_with_long_homopolymer"] == pytest.approx(over / len(MIXED_GC))

    def test_fractions_are_over_the_surviving_sequences(self):
        """Every fraction divides by num_valid_seqs, not by the rows generated.

        The pool is chosen so the two denominators differ: four rows, two
        survivors, one of them a duplicate.
        """
        with pp.Party():
            pool = pp.from_seqs(
                ["AAAATTTT", "GGGGCCCC", "AAAATTTT", "GGCCGGCC"], mode="sequential"
            ).filter_gc(max_gc=0.5)

            stats = pool.stats(max_homopolymer_run=3, sites=["AAAA"], show_progress=False)

            assert stats["num_generated_seqs"] == 4
            assert stats["num_valid_seqs"] == 2
            assert stats["num_duplicate_seqs"] == 1
            assert stats["frac_duplicate_seqs"] == pytest.approx(0.5)
            assert stats["frac_seqs_with_long_homopolymer"] == pytest.approx(1.0)
            assert stats["frac_seqs_with_restriction_site"] == pytest.approx(1.0)

    def test_the_homopolymer_limit_can_be_dropped(self):
        """max_homopolymer_run=None keeps the longest run but drops the fraction."""
        with pp.Party():
            stats = pp.from_seqs(MIXED_GC, mode="sequential").stats(
                max_homopolymer_run=None, show_progress=False
            )

            assert "longest_homopolymer" in stats
            assert "frac_seqs_with_long_homopolymer" not in stats


class TestRestrictionSites:
    """Tests for the opt-in restriction-site statistic."""

    def test_absent_unless_asked_for(self):
        """No enzymes and no sites means no restriction statistic."""
        with pp.Party():
            stats = pp.from_seqs(MIXED_GC, mode="sequential").stats(show_progress=False)

            assert "frac_seqs_with_restriction_site" not in stats

    def test_named_enzyme_matches_the_helper(self):
        """An enzyme name resolves to its site and agrees with the helper."""
        seqs = ["ACGTGAATTCACGT", "ACGTACGTACGTAC"]
        with pp.Party():
            stats = pp.from_seqs(seqs, mode="sequential").stats(
                enzymes=["EcoRI"], show_progress=False
            )

            expected = sum(has_restriction_site(seq, ["GAATTC"]) for seq in seqs)
            assert stats["frac_seqs_with_restriction_site"] == pytest.approx(expected / len(seqs))

    def test_a_preset_covers_every_enzyme_in_it(self):
        """A preset name expands to the sites of all its enzymes."""
        # BsmBI (CGTCTC) is in the golden_gate preset but not named here.
        with pp.Party():
            stats = pp.from_seqs(["ACGTCGTCTCACGT"], mode="sequential").stats(
                enzymes=["golden_gate"], show_progress=False
            )

            assert stats["frac_seqs_with_restriction_site"] == pytest.approx(1.0)

    def test_the_reverse_strand_is_searched_too(self):
        """A site on the other strand cuts just as well, so it counts."""
        # GAGACC is the reverse complement of BsaI's GGTCTC.
        with pp.Party():
            stats = pp.from_seqs(["ACGTGAGACCACGT"], mode="sequential").stats(
                sites=["GGTCTC"], show_progress=False
            )

            assert stats["frac_seqs_with_restriction_site"] == pytest.approx(1.0)


class TestReport:
    """Tests for the returned object and the report it prints."""

    def test_behaves_as_a_plain_dict(self):
        """The result is a dict: subscriptable, copyable and JSON-serialisable."""
        with pp.Party():
            stats = filtered_pool().stats(show_progress=False)

            assert isinstance(stats, dict)
            assert json.loads(json.dumps(stats))["num_unique_seqs"] == 3

    def test_prints_the_sections_it_has(self):
        """Printing gives the report, with a heading per group of statistics."""
        with pp.Party():
            report = repr(filtered_pool().stats(show_progress=False))

            for heading in ("Composition", "Length", "GC content", "Homopolymer runs"):
                assert heading in report
            assert "unique sequences" in report

    def test_omits_sections_with_nothing_to_show(self):
        """A statistic that was not computed leaves no empty heading behind."""
        with pp.Party():
            report = repr(filtered_pool().stats(max_hamming_seqs=None, show_progress=False))

            assert "Restriction sites" not in report
            assert "Pairwise distance" not in report

    def test_an_open_ended_design_is_labelled_as_such(self):
        """The report says the duplicate count depends on how much was drawn."""
        with pp.Party():
            pool = pp.from_seq("ACGTACGTAC").mutagenize(num_mutations=2, mode="random")

            report = repr(pool.stats(num_seqs=100, show_progress=False))

            assert "unbounded" in report
            assert "not a property of the design" in report

    def test_a_subsampled_distance_is_labelled_as_such(self):
        """The report warns that a sampled minimum is only an upper bound."""
        with pp.Party():
            pool = pp.from_seq("ACGTACGTAC").mutagenize(num_mutations=1, mode="sequential")

            report = repr(pool.stats(max_hamming_seqs=5, show_progress=False))

            assert "upper bound" in report

    def test_exported_at_the_top_level(self):
        """pp.stats(pool) and pool.stats() are both available."""
        assert "stats" in pp.__all__
        with pp.Party():
            pool = pp.from_seqs(MIXED_GC, mode="sequential")

            assert dict(pp.stats(pool, show_progress=False)) == dict(
                pool.stats(show_progress=False)
            )


class TestKeys:
    """Tests for the exact set of keys a report contains."""

    def test_the_key_set_is_stable(self):
        """A default report has exactly these keys, and none of the cut ones."""
        with pp.Party():
            stats = filtered_pool().stats(show_progress=False)

            assert set(stats) == {
                "num_states",
                "open_ended",
                "num_generated_seqs",
                "frac_design_covered",
                "num_filtered_out_seqs",
                "num_valid_seqs",
                "num_unique_seqs",
                "num_duplicate_seqs",
                "frac_duplicate_seqs",
                "max_seq_copies",
                "length_min",
                "length_max",
                "gc_min",
                "gc_mean",
                "gc_max",
                "longest_homopolymer",
                "frac_seqs_with_long_homopolymer",
                "dust_mean",
                "dust_max",
                "hamming_exact",
                "hamming_seqs_compared",
                "hamming_min",
                "hamming_mean",
                "hamming_max",
            }

    def test_the_deliberately_omitted_keys_stay_omitted(self):
        """These were dropped as arithmetic derivable from the rest."""
        with pp.Party():
            stats = filtered_pool().stats(show_progress=False)

            for key in ("length_variable", "hamming_num_pairs", "hamming_sd", "dust_min"):
                assert key not in stats


class TestSequenceInput:
    """Tests for describing sequences that did not come from a pool."""

    def test_sequences_are_described_directly(self):
        """A list of sequences needs no pool and reports no design size."""
        stats = pp.stats(["ACGT", "ACGT", "TGCA"])

        assert stats["num_states"] is None
        assert stats["open_ended"] is False
        assert stats["frac_design_covered"] is None
        assert stats["num_unique_seqs"] == 2

    def test_a_bare_string_is_refused(self):
        """pp.stats(seq) would otherwise measure one character per sequence."""
        with pytest.raises(TypeError, match="single string"):
            pp.stats("ACGTACGT")

    def test_a_count_alongside_sequences_is_refused(self):
        """num_seqs has no meaning when the sequences are already in hand."""
        with pytest.raises(ValueError, match="apply to a pool"):
            pp.stats(["ACGT", "TGCA"], num_seqs=1)


class TestArgumentValidation:
    """Tests for the argument checks."""

    def test_both_counts_is_refused(self):
        """num_seqs and num_cycles are mutually exclusive, as in to_df."""
        with pp.Party():
            with pytest.raises(ValueError, match="only one of num_seqs or num_cycles"):
                filtered_pool().stats(num_seqs=2, num_cycles=1, show_progress=False)

    @pytest.mark.parametrize("kwargs", [{"num_seqs": 0}, {"num_cycles": -1}])
    def test_counts_must_be_positive(self, kwargs):
        """A count of zero or less asks for nothing."""
        with pp.Party():
            with pytest.raises(ValueError, match="must be positive"):
                filtered_pool().stats(show_progress=False, **kwargs)

    def test_a_pair_needs_two_sequences(self):
        """max_hamming_seqs below 2 could not form a pair."""
        with pp.Party():
            with pytest.raises(ValueError, match="at least 2"):
                filtered_pool().stats(max_hamming_seqs=1, show_progress=False)

    def test_the_homopolymer_limit_must_be_at_least_one(self):
        """A run is at least one base long."""
        with pp.Party():
            with pytest.raises(ValueError, match="at least 1"):
                filtered_pool().stats(max_homopolymer_run=0, show_progress=False)

    def test_protein_pools_are_not_supported(self):
        """Most of the statistics are DNA-specific, so a ProteinPool is refused."""
        with pp.Party():
            protein = pp.from_seq("ATGAAATAG").translate()

            with pytest.raises(TypeError, match="supports DnaPool"):
                pp.stats(protein)


class TestEdgeCases:
    """Tests for pools the statistics cannot fully describe."""

    def test_distances_need_equal_lengths(self):
        """Hamming distance is undefined between different lengths, so it is omitted."""
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "ACGTACGT"], mode="sequential")

            stats = pool.stats(show_progress=False)

            assert stats["length_min"] == 4
            assert stats["length_max"] == 8
            assert "hamming_min" not in stats

    def test_everything_filtered_out(self):
        """A filter that rejects every sequence gives counts, not an error."""
        with pp.Party():
            pool = pp.from_seqs(["GGGGCCCC", "GGCCGGCC"], mode="sequential").filter_gc(max_gc=0.5)

            stats = pool.stats(show_progress=False)

            assert stats["num_generated_seqs"] == 2
            assert stats["num_filtered_out_seqs"] == 2
            assert stats["num_valid_seqs"] == 0
            assert "gc_mean" not in stats

    def test_case_does_not_make_a_sequence_distinct(self):
        """Case is presentation in PoolParty, so it does not change the molecule."""
        with pp.Party():
            pool = pp.from_seqs(["acgtacgt", "ACGTACGT", "AcGtAcGt"], mode="sequential")

            stats = pool.stats(show_progress=False)

            assert stats["num_valid_seqs"] == 3
            assert stats["num_unique_seqs"] == 1
            assert stats["num_duplicate_seqs"] == 2

    def test_a_progress_bar_does_not_break_anything(self):
        """show_progress defaults to True, so the default path must be exercised."""
        with pp.Party():
            stats = pp.from_seqs(MIXED_GC, mode="sequential").stats(show_progress=True)

            assert stats["num_generated_seqs"] == 5

    def test_region_tags_are_not_counted(self):
        """Tags describe the design, not the molecule, so they are stripped."""
        with pp.Party():
            tagged = pp.from_seq("ACGT<bc>TTTT</bc>ACGT")
            plain = pp.from_seq("ACGTTTTTACGT")

            tagged_stats = tagged.stats(show_progress=False)

            assert tagged_stats["length_min"] == plain.stats(show_progress=False)["length_min"]
            assert tagged_stats["gc_mean"] == pytest.approx(
                plain.stats(show_progress=False)["gc_mean"]
            )


class TestNeverMutates:
    """Tests that a readout is only a readout."""

    def test_the_pool_is_unchanged(self):
        """Design size and parents survive the call."""
        with pp.Party():
            pool = filtered_pool()

            pool.stats(show_progress=False)

            assert pool.num_states == 5
            assert len(pool.parents) == 1

    def test_the_generation_cursor_is_put_back(self):
        """A readout must not change which sequence comes out next.

        Without this, calling stats() mid-pipeline silently shifts every
        subsequent generate_library on the same pool.
        """
        with pp.Party():
            pool = pp.from_seqs(MIXED_GC, mode="sequential")
            pool.generate_library(num_seqs=2, seed=99)
            cursor, master_seed = pool._current_state, pool._master_seed

            pool.stats(num_seqs=3, show_progress=False)

            assert (pool._current_state, pool._master_seed) == (cursor, master_seed)
