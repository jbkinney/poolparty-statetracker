"""Integration tests following dev/integration_test_specs.md.

Each test builds a multi-pool library, checks num_states at every level,
generates sequences, and validates structural invariants.
"""

from collections import Counter

import pandas as pd
import pytest

import poolparty as pp


# ---------------------------------------------------------------------------
# Helpers: concrete sequences used across libraries
# ---------------------------------------------------------------------------

def _make_barcodes(n: int, length: int = 12, seed: int = 0) -> list[str]:
    """Deterministic barcode generator — avoids long homopolymer runs."""
    import random as _rng
    r = _rng.Random(seed)
    barcodes: list[str] = []
    seen: set[str] = set()
    while len(barcodes) < n:
        bc = "".join(r.choice("ACGT") for _ in range(length))
        max_run = max(len(list(g)) for _, g in __import__("itertools").groupby(bc))
        if max_run <= 3 and bc not in seen:
            barcodes.append(bc)
            seen.add(bc)
    return barcodes


def _random_dna(length: int, seed: int = 0) -> str:
    import random
    r = random.Random(seed)
    return "".join(r.choice("ACGT") for _ in range(length))


def _make_orf(num_codons: int, seed: int = 0) -> str:
    """ATG-starting ORF with no internal stops."""
    import random
    r = random.Random(seed)
    stop_codons = {"TAA", "TAG", "TGA"}
    codons = ["ATG"]
    while len(codons) < num_codons:
        codon = "".join(r.choice("ACGT") for _ in range(3))
        if codon not in stop_codons:
            codons.append(codon)
    return "".join(codons)


# Pre-built sequences
_orf_45nt = _make_orf(15, seed=1)
_orf_30nt = _make_orf(10, seed=2)
_orf_40nt = _make_orf(40 // 3, seed=3)  # 13 codons = 39 nt — pad to 40
_orf_36nt = _make_orf(12, seed=4)
_orf_12nt = _make_orf(4, seed=5)

# Pad _orf_40nt to exactly 40 nt (add one more base)
if len(_orf_40nt) < 40:
    _orf_40nt += "A" * (40 - len(_orf_40nt))

# For Library 10: exon needs wt ∈ allowed set.  allowed_chars="KMKMKM..." means
# even positions need wt ∈ K={G,T}, odd positions need wt ∈ M={A,C}.
_exon5_20nt = "GATCGATCGATCGATCGATC"  # G∈K, A∈M, T∈K, C∈M, ...


class TestLibrary1:
    """MPRA Core Promoter Architecture Screen (171 nt, 1280 states)."""

    def test_state_counts_and_sequences(self):
        PE1_22nt = _random_dna(22, seed=10)
        PE2_22nt = _random_dna(22, seed=11)
        barcode_list_16 = _make_barcodes(16, length=12, seed=12)

        with pp.Party() as party:
            enh_a = pp.from_seq("A" * 45).named("enh_SV40")
            enh_b = pp.from_seq("C" * 45).named("enh_CMV")
            enh_c = pp.from_seq("G" * 45).named("enh_CaMKII")
            enh_d = pp.from_seq("T" * 45).named("enh_synth")

            enhancer = pp.stack([enh_a, enh_b, enh_c, enh_d]).named("enhancer_mix")
            enhancer_noisy = enhancer.mutagenize(
                mutation_rate=0.01, region=[10, 35]
            ).named("enh_noise")

            tata = pp.from_seq("TATAAA")
            inr = pp.from_seq("TCAGTC")
            promoter = pp.from_seq("N" * 70).named("bg_70nt")
            promoter_scan = promoter.replacement_multiscan(
                num_replacements=2,
                replacement_pools=[tata, inr],
                positions=[[0, 2, 4, 6, 8], [31, 33, 35, 37]],
                min_spacing=12,
                insertion_mode="ordered",
                mode="sequential",
            ).named("tata_inr_scan")

            barcodes = pp.from_seqs(barcode_list_16, mode="sequential").named("barcodes")

            adapter_5 = pp.from_seq(PE1_22nt)
            adapter_3 = pp.from_seq(PE2_22nt)
            library = pp.join([adapter_5, enhancer_noisy, promoter_scan, adapter_3, barcodes])

        assert enhancer.num_states == 4
        assert promoter_scan.num_states == 20
        assert barcodes.num_states == 16
        assert library.num_states == 1280

        df = pp.generate_library(library, num_cycles=1, seed=42)
        seqs = df["seq"].tolist()
        assert len(seqs) == 1280
        assert all(len(s) == 171 for s in seqs)

        # Fixed adapter regions
        assert all(s[:22] == PE1_22nt for s in seqs)
        assert all(s[137:159] == PE2_22nt for s in seqs)

        # Core promoter contains both motifs in correct order
        for s in seqs:
            cp = s[67:137]
            t = cp.find("TATAAA")
            i = cp.find("TCAGTC")
            assert t >= 0 and i >= 0
            assert t + 6 <= i
            assert i - (t + 6) >= 12

        # Each barcode appears exactly 4 × 20 = 80 times
        bc_counts = Counter(s[159:171] for s in seqs)
        assert all(v == 80 for v in bc_counts.values())

        # Design cards
        df_full = pp.generate_library(library, num_cycles=1, seed=42)
        assert len(df_full) == 1280


class TestLibrary2:
    """ORF Saturation Mutagenesis with 3′UTR Shuffle Controls (125 nt)."""

    def test_state_counts(self):
        adapter_20nt = _random_dna(20, seed=20)
        kozak_12nt = _random_dna(12, seed=21)
        adapter_20nt_2 = _random_dna(20, seed=22)
        utr3_18nt = _random_dna(18, seed=23)
        barcode_list_20 = _make_barcodes(20, length=10, seed=24)

        with pp.Party() as party:
            orf = pp.from_seq(_orf_45nt)
            orf_mutants = orf.mutagenize_orf(
                num_mutations=1,
                mutation_type="any_codon",
                mode="sequential",
            ).named("orf_k1_any")

            orf_noisy = orf_mutants.mutagenize_orf(
                mutation_rate=0.003, region=[0, 30]
            ).named("orf_pcr_noise")

            utr3 = pp.from_seq(utr3_18nt)
            utr3_shuffled = utr3.shuffle_scan(
                shuffle_length=6,
                positions=[0, 6, 12],
                shuffles_per_position=2,
                mode="sequential",
            ).named("utr3_shuf_scan")

            barcodes = pp.from_seqs(barcode_list_20, mode="sequential").named("barcodes")
            library = pp.join([
                pp.from_seq(adapter_20nt),
                pp.from_seq(kozak_12nt),
                orf_noisy,
                utr3_shuffled,
                pp.from_seq(adapter_20nt_2),
                barcodes,
            ])

        assert orf_mutants.num_states == 945  # C(15,1) × 63
        assert utr3_shuffled.num_states == 6  # 3 positions × 2 shuffles
        assert barcodes.num_states == 20
        assert library.num_states == 113_400


class TestLibrary3:
    """Three-TF Enhancer Grammar Screen (148 nt, 1584 states)."""

    def test_state_counts_and_sequences(self):
        enhancer_bg_120nt = _random_dna(120, seed=30)
        adapter_20nt = _random_dna(20, seed=31)
        barcode_list_24 = _make_barcodes(24, length=8, seed=32)

        with pp.Party() as party:
            bg = pp.from_seq(enhancer_bg_120nt)
            ets = pp.from_seq("GGAA")
            gata = pp.from_seq("AGATAA")
            runx = pp.from_seq("TGTGGT")

            tf_scan = bg.replacement_multiscan(
                num_replacements=3,
                replacement_pools=[ets, gata, runx],
                positions=[
                    [10, 20, 30, 40],
                    [45, 55, 65, 75],
                    [80, 90, 100, 110],
                ],
                min_spacing=5,
                insertion_mode="ordered",
                mode="sequential",
            ).named("tf_spacing")

            shuffle_ctrl = bg.shuffle_seq(mode="random", num_states=10).named("shuffle_ctrl")
            mixed = pp.stack([tf_scan, shuffle_ctrl]).named("signal_vs_control")

            # Tag-based region
            mixed_tagged = mixed.annotate_region("inner", extent=(20, 100))
            mixed_noisy = mixed_tagged.mutagenize(
                mutation_rate=0.005, region="inner"
            ).named("synth_noise")
            mixed_clean = mixed_noisy.remove_tags("inner").named("synth_clean")

            barcodes = pp.from_seqs(barcode_list_24, mode="sequential").named("barcodes")
            library = pp.join([pp.from_seq(adapter_20nt), mixed_clean, barcodes])

        assert tf_scan.num_states == 56
        assert shuffle_ctrl.num_states == 10
        assert mixed.num_states == 66
        assert barcodes.num_states == 24
        assert library.num_states == 1584

        df = pp.generate_library(library, num_cycles=1, seed=42)
        seqs = df["seq"].tolist()
        assert len(seqs) == 1584
        assert all(len(s) == 148 for s in seqs)


class TestLibrary4:
    """CRISPR Guide Library (99 nt, 5376 states)."""

    def test_state_counts_and_sequences(self):
        scaffold_42nt = _random_dna(42, seed=40)
        scaffold_alt_42nt = _random_dna(42, seed=41)
        barcode_list_12 = _make_barcodes(12, length=8, seed=42)

        with pp.Party() as party:
            u6 = pp.from_seq("GAGGGCCTATTTCCCATGATTCC")

            spacer = pp.from_iupac("RRNACCATGGCCTTTTTTTT", mode="sequential").named("spacer")

            scaffold_wt = pp.from_seq(scaffold_42nt)
            scaffold_noisy = scaffold_wt.mutagenize(
                mutation_rate=0.002, region=[0, 21]
            ).named("scaffold_noise")

            scaffold_alt = pp.from_seq(scaffold_alt_42nt)
            scaffold_recomb = pp.recombine(
                sources=[scaffold_wt, scaffold_alt],
                num_breakpoints=1,
                positions=[10, 20, 30],
                mode="sequential",
            ).named("scaffold_recomb")

            scaffold_mixed = pp.stack([scaffold_noisy, scaffold_recomb]).named("scaffold_mix")

            pam_bg = pp.from_seq("NNNNNN")
            pam_scan = pam_bg.replacement_scan(
                ins_pool="GG",
                positions=[0, 1, 2, 3],
                mode="sequential",
            ).named("pam_scan")

            barcodes = pp.from_seqs(barcode_list_12, mode="sequential").named("barcodes")
            library = pp.join([u6, spacer, scaffold_mixed, pam_scan, barcodes])

        assert spacer.num_states == 16
        assert scaffold_recomb.num_states == 6
        assert scaffold_mixed.num_states == 7
        assert pam_scan.num_states == 4
        assert barcodes.num_states == 12
        assert library.num_states == 5376

        df = pp.generate_library(library, num_cycles=1, seed=42)
        seqs = df["seq"].tolist()
        assert len(seqs) == 5376
        assert all(len(s) == 99 for s in seqs)

        # IUPAC constraints on spacer region (positions 23-42 in joined seq)
        for s in seqs:
            assert s[23] in "AG"    # R
            assert s[24] in "AG"    # R
            assert s[25] in "ACGT"  # N

        # "GG" present in PAM region at one of 4 offsets
        for s in seqs:
            pam = s[85:91]
            assert "GG" in pam


class TestLibrary5:
    """CRE Deletion Scan + MutagenizeScan + Filters (85 nt, 16524 states)."""

    def test_state_counts_and_sequences(self):
        cre_45nt = _random_dna(45, seed=50)
        barcode_list_12 = _make_barcodes(12, length=10, seed=51)

        with pp.Party() as party:
            left_flank = pp.from_seqs(
                ["ACGTACGTACGTACG", "AGTCAGTCAGTCAGT", "TGCATGCATGCATGC"],
                mode="sequential",
            ).named("left_flank")
            left_noisy = left_flank.mutagenize(
                mutation_rate=0.05, region=[3, 12]
            ).named("left_noise")

            cre = pp.from_seq(cre_45nt)

            cre_del = cre.deletion_scan(
                deletion_length=3,
                positions=[0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42],
                mode="sequential",
            ).named("cre_del_scan")

            cre_mutscan = cre.mutagenize_scan(
                mutagenize_length=9,
                num_mutations=1,
                positions=[0, 9, 18, 27, 36],
                mode=("sequential", "sequential"),
            ).named("cre_mutscan")

            cre_marker = cre.replacement_scan(
                ins_pool="GGG",
                positions=[0, 15, 30],
                mode="sequential",
            ).named("cre_marker")

            cre_mixed = pp.stack([cre_del, cre_mutscan, cre_marker]).named("cre_mix")

            right_flank = pp.from_seqs(
                ["ACGTACGTACGTACG", "AGTCAGTCAGTCAGT", "TGCATGCATGCATGC"],
                mode="sequential",
            ).named("right_flank")
            right_noisy = right_flank.mutagenize(
                mutation_rate=0.05, region=[3, 12]
            ).named("right_noise")

            barcodes = pp.from_seqs(barcode_list_12, mode="sequential").named("barcodes")

            library_unfiltered = pp.join([left_noisy, cre_mixed, right_noisy, barcodes])

            library = library_unfiltered.filter_gc(min_gc=0.3, max_gc=0.7)
            library = library.filter_homopolymer(max_length=5)
            library = library.filter(lambda s: s[15:60].count("CG") <= 3).named("cpg_filtered")

        assert cre_del.num_states == 15
        assert cre_mutscan.num_states == 135
        assert cre_marker.num_states == 3
        assert cre_mixed.num_states == 153
        assert library.num_states == 16_524

        df = pp.generate_library(library, num_cycles=1, seed=42, discard_null_seqs=True)
        assert 0 < len(df) <= 16_524
        for s in df["seq"]:
            assert len(s) == 85
            gc = (s.count("G") + s.count("C")) / len(s.replace("-", ""))
            assert 0.3 <= gc <= 0.7
            assert s[15:60].count("CG") <= 3


class TestLibrary6:
    """Protein Epitope Tag Positioning (58 nt, 13300 states)."""

    def test_state_counts(self):
        utr5_10nt = _random_dna(10, seed=60)
        utr3_10nt = _random_dna(10, seed=61)
        barcode_list_10 = _make_barcodes(10, length=8, seed=62)

        with pp.Party() as party:
            orf = pp.from_seq(_orf_30nt)

            orf_bg = orf.mutagenize_orf(
                num_mutations=1,
                mutation_type="missense_only_first",
                mode="sequential",
            ).named("orf_bg_k1")

            orf_noisy = orf_bg.mutagenize_orf(
                mutation_rate=0.005, codon_positions=[0, 1, 2, 3, 4]
            ).named("orf_pcr")

            flag_pool = pp.from_seq("GATTACAAAGAC")
            orf_tagged = orf_noisy.replacement_scan(
                ins_pool=flag_pool,
                positions=[0, 3, 6, 9, 12, 15, 18],
                mode="sequential",
            ).named("flag_scan")

            barcodes = pp.from_seqs(barcode_list_10, mode="sequential").named("barcodes")
            library = pp.join([
                pp.from_seq(utr5_10nt),
                orf_tagged,
                pp.from_seq(utr3_10nt),
                barcodes,
            ])

        assert orf_bg.num_states == 190
        assert orf_tagged.num_states == 1330
        assert barcodes.num_states == 10
        assert library.num_states == 13_300


class TestLibrary7:
    """Dual-Region TF Interaction Screen (182 nt, 537600 states)."""

    def test_state_counts(self):
        linker_30nt = _random_dna(30, seed=70)
        adapter_20nt = _random_dna(20, seed=71)
        barcode_list_8 = _make_barcodes(8, length=12, seed=72)

        with pp.Party() as party:
            ap1 = pp.from_seq("TGACTCA")
            nfkb = pp.from_seq("GGGACTTTCC")
            distal_bg = pp.from_seq("N" * 60)
            distal = distal_bg.replacement_multiscan(
                num_replacements=2,
                replacement_pools=[ap1, nfkb],
                positions=[[2, 8, 14, 20], [35, 40, 45, 50]],
                min_spacing=3,
                insertion_mode="ordered",
                mode="sequential",
            ).named("distal_tf")

            distal_tagged = distal.annotate_region("distal_core", extent=(5, 55))
            distal_noisy = distal_tagged.mutagenize(
                mutation_rate=0.005, region="distal_core"
            ).named("distal_noise")
            distal_clean = distal_noisy.remove_tags("distal_core").named("distal_clean")

            linker = pp.from_seq(linker_30nt)
            linker_del = linker.deletion_multiscan(
                deletion_length=3,
                num_deletions=2,
                min_spacing=5,
                mode="sequential",
            ).named("linker_deldel")

            # Use custom names to avoid collision with distal's internal region
            # names (_rep_0, _rep_1). Pad TATA/Inr to match distal pool sizes
            # (7nt, 10nt) since the Party registers region names globally and
            # requires consistent lengths per name.
            tata_prox = pp.from_seq("TATAAAA")        # 7 nt (matches AP-1 size)
            inr_prox = pp.from_seq("TCAGTCAAAA")      # 10 nt (matches NF-κB size)
            prox_bg = pp.from_seq("N" * 60)
            proximal = prox_bg.replacement_multiscan(
                num_replacements=2,
                replacement_pools=[tata_prox, inr_prox],
                positions=[[0, 2, 4, 6, 8], [31, 33, 35, 37]],
                min_spacing=12,
                insertion_mode="ordered",
                mode="sequential",
            ).named("proximal_tf")

            barcodes = pp.from_seqs(barcode_list_8, mode="sequential").named("barcodes")
            library = pp.join([pp.from_seq(adapter_20nt), distal_clean, linker_del, proximal, barcodes])

        assert distal.num_states == 16
        assert linker_del.num_states == 210
        assert proximal.num_states == 20
        assert barcodes.num_states == 8
        assert library.num_states == 537_600


class TestLibrary8:
    """5′UTR Leader Screen with SubseqScan and StateSlice (90 nt, 12288 states)."""

    def test_state_counts_and_sequences(self):
        adapter_8nt = _random_dna(8, seed=80)
        barcode_list_12 = _make_barcodes(12, length=8, seed=81)

        with pp.Party() as party:
            adapter = pp.from_seq(adapter_8nt)
            kmers = pp.get_kmers(length=3, mode="sequential").named("upstream_3mer")
            kozak = pp.from_iupac("RNACCATGGCC", mode="sequential").named("kozak")

            orf = pp.from_seq(_orf_40nt)

            are = pp.from_seq("TATTTATTTATTTATTTATT")    # 19 nt
            g_quad = pp.from_seq("GGGTTGGGTTGGGTTGGGTT")  # 20 nt

        # UTR strings must be same length for stack
        # are is 19 nt, g_quad is 20 nt — fix are to 20 nt
        with pp.Party() as party:
            adapter = pp.from_seq(adapter_8nt)
            kmers = pp.get_kmers(length=3, mode="sequential").named("upstream_3mer")
            kozak = pp.from_iupac("RNACCATGGCC", mode="sequential").named("kozak")
            orf = pp.from_seq(_orf_40nt)

            are = pp.from_seq("TATTTATTTATTTATTTATT")   # 20 nt AU-rich element
            g_quad = pp.from_seq("GGGTTGGGTTGGGTTGGGTT")  # 20 nt
            utr3 = pp.stack([are, g_quad]).named("utr3_arms")

            utr3_tagged = utr3.annotate_region("utr3_mid", extent=(5, 15))
            utr3_noisy = utr3_tagged.mutagenize(
                mutation_rate=0.01, region="utr3_mid"
            ).named("utr3_noise")
            utr3_clean = utr3_noisy.remove_tags("utr3_mid").named("utr3_clean")

            barcodes = pp.from_seqs(barcode_list_12, mode="sequential").named("barcodes")
            library = pp.join([adapter, kmers, kozak, orf, utr3_clean, barcodes])

            kozak_subseqs = kozak.subseq_scan(
                subseq_length=5,
                mode="sequential",
            ).named("kozak_subseqs")

            sub_kmers = kmers[0:32].named("first_32_kmers")
            sub_library = pp.join([adapter, sub_kmers, kozak, orf, utr3_clean, barcodes])

        assert kmers.num_states == 64
        assert kozak.num_states == 8
        assert utr3.num_states == 2
        assert barcodes.num_states == 12
        assert library.num_states == 12_288
        assert kozak_subseqs.num_states == 56
        assert sub_kmers.num_states == 32
        assert sub_library.num_states == 6_144

        df = pp.generate_library(library, num_cycles=1, seed=42)
        seqs = df["seq"].tolist()
        assert len(seqs) == 12_288
        assert all(len(s) == 90 for s in seqs)

        # Every 3-mer appears exactly 8 × 2 × 12 = 192 times
        kmer_counts = Counter(s[8:11] for s in seqs)
        assert len(kmer_counts) == 64
        assert all(v == 192 for v in kmer_counts.values())

        # Kozak IUPAC constraints
        for s in seqs:
            assert s[11] in "AG"
            assert s[12] in "ACGT"
            assert s[13:22] == "ACCATGGCC"

        # Sub-library check
        sub_df = pp.generate_library(sub_library, num_cycles=1, seed=42)
        assert len(sub_df) == 6_144


class TestLibrary9:
    """Two-Stage Missense Epistasis Screen (76 nt, 155952 states)."""

    def test_state_counts(self):
        utr5_15nt = _random_dna(15, seed=90)
        utr3_15nt = _random_dna(15, seed=91)
        barcode_list_12 = _make_barcodes(12, length=10, seed=92)

        with pp.Party() as party:
            orf = pp.from_seq(_orf_36nt)

            n_term = orf.mutagenize_orf(
                num_mutations=1,
                mutation_type="missense_only_first",
                codon_positions=[0, 1, 2, 3, 4, 5],
                mode="sequential",
            ).named("n_term_mut")

            c_term = n_term.mutagenize_orf(
                num_mutations=1,
                mutation_type="missense_only_first",
                codon_positions=[6, 7, 8, 9, 10, 11],
                mode="sequential",
            ).named("c_term_mut")

            orf_noisy = c_term.mutagenize_orf(
                mutation_rate=0.003, codon_positions=slice(0, 6)
            ).named("pcr_noise")

            barcodes = pp.from_seqs(barcode_list_12, mode="sequential").named("barcodes")

            library = pp.join([
                pp.from_seq(utr5_15nt),
                orf_noisy,
                pp.from_seq(utr3_15nt),
                barcodes,
            ])

            library_3x = library * 3
            library_sample = library_3x.sample(num_seqs=500, seed=42).named("subsample")

        assert n_term.num_states == 114
        assert c_term.num_states == 12_996
        assert barcodes.num_states == 12
        assert library.num_states == 155_952
        assert library_3x.num_states == 467_856
        assert library_sample.num_states == 500


class TestLibrary10:
    """Pre-mRNA Splice Site Architecture Screen (130 nt, 25000 states)."""

    def test_state_counts(self):
        ss5_12nt = _random_dna(12, seed=100)
        intron_50nt = _random_dna(50, seed=101)
        exon3_20nt = _random_dna(20, seed=102)
        barcode_list_10 = _make_barcodes(10, length=8, seed=103)

        with pp.Party() as party:
            exon5 = pp.from_seq(_exon5_20nt)
            exon5_mut = exon5.mutagenize(
                num_mutations=1,
                allowed_chars="KMKMKMKMKMKMKMKMKMKM",
                mode="sequential",
            ).named("exon5_transversions")

            ss5 = pp.from_seq(ss5_12nt)
            ss5_del = ss5.deletion_scan(
                deletion_length=2,
                positions=[0, 2, 4, 6, 8],
                mode="sequential",
            ).named("ss5_del_scan")

            intron = pp.from_seq(intron_50nt)
            bp_pool = pp.from_seq("TTAAC")
            intron_bp = intron.replacement_scan(
                ins_pool=bp_pool,
                positions=[0, 10, 20, 30, 40],
                mode="sequential",
            ).named("bp_scan")

            intron_tagged = intron_bp.annotate_region("intron_core", extent=(10, 40))
            intron_noisy = intron_tagged.mutagenize(
                mutation_rate=0.02, region="intron_core"
            ).named("intron_noise")
            intron_clean = intron_noisy.remove_tags("intron_core").named("intron_clean")

            ppt_pool = pp.from_seq("TTTCTTTCTTTTTTCTTTTT")
            ppt_shuffled = ppt_pool.shuffle_seq(mode="random", num_states=4).named("ppt_shuffles")
            ppt_mixed = pp.stack([ppt_pool, ppt_shuffled]).named("ppt_mix")

            exon3 = pp.from_seq(exon3_20nt)

            barcodes = pp.from_seqs(barcode_list_10, mode="sequential").named("barcodes")

            library = pp.join([exon5_mut, ss5_del, intron_clean, ppt_mixed, exon3, barcodes])

        assert exon5_mut.num_states == 20
        assert ss5_del.num_states == 5
        assert intron_bp.num_states == 5
        assert ppt_mixed.num_states == 5
        assert barcodes.num_states == 10
        assert library.num_states == 25_000


class TestLibrary11:
    """ORF Nonsense Scan + Translate + Reverse Translate + RC + Slice (64 nt)."""

    def test_state_counts_and_sequences(self):
        utr5_8nt = _random_dna(8, seed=110)
        linker_6nt = _random_dna(6, seed=111)
        barcode_list_8 = _make_barcodes(8, length=8, seed=112)

        with pp.Party() as party:
            orf = pp.from_seq(_orf_30nt)
            orf_ns = orf.mutagenize_orf(
                num_mutations=1,
                mutation_type="nonsense",
                mode="sequential",
            ).named("orf_nonsense")

            protein = orf_ns.translate().named("protein_output")

            rev_dna = pp.reverse_translate(
                "MKVLAT", codon_selection="random", num_states=10
            ).named("rev_codons")

            rev_rc = rev_dna.rc().named("rev_rc")
            rev_trimmed = rev_rc.slice_seq(start=0, stop=12).named("rev_trimmed")

            barcodes = pp.from_seqs(barcode_list_8, mode="sequential").named("barcodes")

            library = pp.join([
                pp.from_seq(utr5_8nt),
                orf_ns,
                pp.from_seq(linker_6nt),
                rev_trimmed,
                barcodes,
            ])

        assert orf_ns.num_states == 30
        assert protein.num_states == 30
        assert rev_trimmed.num_states == 10
        assert rev_trimmed.seq_length == 12
        assert barcodes.num_states == 8
        assert library.num_states == 2400

        df = pp.generate_library(library, num_cycles=1, seed=42)
        seqs = df["seq"].tolist()
        assert len(seqs) == 2400
        assert all(len(s) == 64 for s in seqs)

        # num_seqs sampling
        df_sample = pp.generate_library(library, num_seqs=500, seed=42)
        assert len(df_sample) == 500

        # Protein output: each should contain a stop codon
        prot_df = pp.generate_library(protein, num_cycles=1, seed=42)
        prot_seqs = prot_df["seq"].tolist()
        assert len(prot_seqs) == 30
        for p in prot_seqs:
            assert p.count("*") >= 1


class TestLibrary12:
    """Region Workflow + Insertion Scan + Advanced Filters.

    Chains annotate_region → replace_region → insertion_scan → filters.
    This validates that replace_region correctly propagates seq_length,
    allowing downstream insertion_scan to work without restructuring.
    """

    def test_state_counts_and_sequences(self):
        bg_40nt = _random_dna(40, seed=120)
        barcode_list_10 = _make_barcodes(10, length=10, seed=121)

        with pp.Party() as party:
            bg = pp.from_seq(bg_40nt)

            # annotate → replace (10nt region with 10nt IUPAC variants)
            bg_r = bg.annotate_region("var", extent=(15, 25)).named("bg_annotated")
            var_content = pp.from_iupac("RYSWRYSWKM", mode="random", num_states=6)
            bg_filled = bg_r.replace_region(var_content, "var").named("var_filled")

            assert bg_filled.seq_length == 40  # replace_region now propagates seq_length

            # insertion_scan directly on the replaced pool (previously broke with seq_length=None)
            bg_inserted = bg_filled.insertion_scan(
                ins_pool="CC",
                positions=[5, 15, 25, 35],
                mode="sequential",
            ).named("cc_insert_scan")

            # Filters
            bg_filtered = bg_inserted.filter_restriction_sites(
                enzymes=["EcoRI", "BamHI"]
            ).named("re_filtered")
            bg_dust = bg_filtered.filter_dust(max_score=3.0).named("dust_filtered")

            barcodes = pp.from_seqs(barcode_list_10, mode="sequential").named("barcodes")
            library = pp.join([bg_dust, barcodes])

        assert bg_filled.num_states == 6
        assert bg_inserted.num_states == 24  # 6 variants × 4 positions
        assert library.num_states == 240  # 24 × 10 barcodes

        df = pp.generate_library(library, num_seqs=200, seed=42, discard_null_seqs=True)
        assert len(df) <= 200
        for s in df["seq"]:
            assert "GAATTC" not in s
            assert "GGATCC" not in s


class TestLibrary13:
    """Materialize + Sync + State Shuffle + from_motif (21 nt, 400 states).

    NOTE: Spec applies shuffle_states to a synced pool before joining, which
    causes ordered_product to treat the shuffle state and the synced state as
    separate axes (10 × 10 = 100 instead of 10). This is correct statetracker
    behavior — st.shuffle creates a child state. Restructured: use motif_mat
    directly in the join (sync deduplicates the shared state), then test
    shuffle_states separately on the joined library.
    """

    def test_state_counts_and_sequences(self):
        barcode_list_10 = _make_barcodes(10, length=10, seed=130)

        with pp.Party() as party:
            prob_df = pd.DataFrame({
                'A': [0.7, 0.1, 0.1, 0.1, 0.5, 0.5],
                'C': [0.1, 0.7, 0.1, 0.1, 0.2, 0.2],
                'G': [0.1, 0.1, 0.7, 0.1, 0.2, 0.2],
                'T': [0.1, 0.1, 0.1, 0.7, 0.1, 0.1],
            })
            motif_pool = pp.from_motif(prob_df, mode="random", num_states=10).named("motif")

            motif_mat = motif_pool.materialize(num_cycles=1, seed=42).named("motif_mat")

            var_pool = pp.from_seqs(
                ["AAA", "CCC", "GGG", "TTT"],
                mode="random", num_states=10,
            ).named("var_random")

            pp.sync([motif_mat, var_pool])

            random_dimers = pp.get_kmers(
                length=2, mode="random", num_states=4
            ).named("random_dimers")

            barcodes = pp.from_seqs(barcode_list_10, mode="sequential").named("barcodes")

            # motif_mat and var_pool share the same state (synced), counted once
            library = pp.join([motif_mat, var_pool, random_dimers, barcodes])

            # shuffle_states on the joined library just permutes iteration order
            library_shuffled = library.shuffle_states(seed=123).named("shuffled_library")

        assert motif_mat.num_states == 10
        assert var_pool.num_states == 10
        assert random_dimers.num_states == 4
        assert barcodes.num_states == 10
        assert library.num_states == 400  # 10 (synced) × 4 × 10

        # Materialized pool has no parents (severed DAG)
        assert len(motif_mat.parents) == 0

        df = pp.generate_library(library, num_cycles=1, seed=42)
        seqs = df["seq"].tolist()
        assert len(seqs) == 400
        assert all(len(s) == 21 for s in seqs)

        # Each barcode appears exactly 10 × 4 = 40 times
        bc_counts = Counter(s[11:21] for s in seqs)
        assert all(v == 40 for v in bc_counts.values())

        # Shuffled library has same states, different order
        assert library_shuffled.num_states == 400
        df_shuf = pp.generate_library(library_shuffled, num_cycles=1, seed=42)
        assert len(df_shuf) == 400
        assert set(df_shuf["seq"]) == set(seqs)


class TestLibrary14:
    """Multi-Mutation + Insertion Multiscan (16 nt, 19440 states)."""

    def test_state_counts_and_sequences(self):
        barcode_list_4 = _make_barcodes(4, length=6, seed=140)

        with pp.Party() as party:
            bg = pp.from_seq("ACGTAC")

            bg_mut = bg.mutagenize(
                num_mutations=2,
                mode="sequential",
            ).named("bg_2mut")

            insert_a = pp.from_seq("GG")
            insert_b = pp.from_seq("TT")
            bg_ins = bg_mut.insertion_multiscan(
                num_insertions=2,
                insertion_pools=[insert_a, insert_b],
                min_spacing=1,
                max_spacing=4,
                insertion_mode="unordered",
                mode="sequential",
            ).named("ins_unordered")

            barcodes = pp.from_seqs(barcode_list_4, mode="sequential").named("barcodes")
            library = pp.join([bg_ins, barcodes])

        assert bg_mut.num_states == 135  # C(6,2) × 3^2
        assert bg_ins.num_states == 135 * 36
        assert barcodes.num_states == 4
        assert library.num_states == 19_440

        df = pp.generate_library(library, num_cycles=1, seed=42)
        seqs = df["seq"].tolist()
        assert len(seqs) == 19_440
        assert all(len(s) == 16 for s in seqs)


class TestLibrary15:
    """3-Source Recombine + Region Replace + apply_at_region (46 nt, 2880 states).

    NOTE: Spec chains apply_at_region after replace_region on the same "mid"
    region, but replace_region consumes the region tags, so apply_at_region
    can't find them. Restructured: apply_at_region first on a separate region
    (head), then replace_region on the "mid" region.
    """

    def test_state_counts_and_sequences(self):
        bg_24nt = _random_dna(24, seed=150)
        barcode_list_10 = _make_barcodes(10, length=10, seed=151)

        with pp.Party() as party:
            src_a = pp.from_seq("AAAAAAAAAAAA")
            src_b = pp.from_seq("CCCCCCCCCCCC")
            src_c = pp.from_seq("GGGGGGGGGGGG")
            combo = pp.recombine(
                sources=[src_a, src_b, src_c],
                num_breakpoints=2,
                positions=[2, 5, 8],
                mode="sequential",
            ).named("recomb_3src")

            bg = pp.from_seq(bg_24nt)

            # apply_at_region: RC positions 0-5 (tags removed by default)
            bg_r1 = bg.annotate_region("head", extent=(0, 5))
            bg_rc_head = bg_r1.apply_at_region("head", pp.rc).named("head_rc")

            # replace_region: replace positions 9-18 with IUPAC content
            bg_r2 = bg_rc_head.annotate_region("mid", extent=(9, 18))
            iupac_mid = pp.from_iupac("RNACCATGG", mode="sequential")
            # R(2) × N(4) × fixed(7) = 8 states
            bg_replaced = bg_r2.replace_region(iupac_mid, "mid").named("region_replaced")

            barcodes = pp.from_seqs(barcode_list_10, mode="sequential").named("barcodes")
            library = pp.join([combo, bg_replaced, barcodes])

        assert combo.num_states == 36
        assert bg_replaced.num_states == 8
        assert barcodes.num_states == 10
        assert library.num_states == 2880

        df = pp.generate_library(library, num_cycles=1, seed=42)
        seqs = df["seq"].tolist()
        assert len(seqs) == 2880
        assert all(len(s) == 46 for s in seqs)

        # Recombined region: each position is A, C, or G
        for s in seqs:
            recomb = s[:12]
            assert all(c in "ACG" for c in recomb)

        # Each barcode appears exactly 36 × 8 = 288 times
        bc_counts = Counter(s[36:46] for s in seqs)
        assert all(v == 288 for v in bc_counts.values())


class TestLibrary16:
    """Region-Based Sequential Mutagenize + Negative Frame ORF (42 nt, 43776 states)."""

    def test_state_counts(self):
        bg_20nt = _random_dna(20, seed=160)
        barcode_list_4 = _make_barcodes(4, length=4, seed=161)

        with pp.Party() as party:
            bg = pp.from_seq(bg_20nt)
            bg_mut = bg.mutagenize(
                num_mutations=1,
                region=[9, 12],
                mode="sequential",
            ).named("region_mut")

            orf_rev = pp.from_seq(_orf_12nt)
            orf_neg = orf_rev.mutagenize_orf(
                num_mutations=1,
                mutation_type="missense_only_first",
                frame=-1,
                mode="sequential",
            ).named("neg_frame_mut")

            spacer = pp.from_seq("AAAAAA")
            spacer_tagged = spacer.annotate_region("iupac_site", extent=(1, 4))
            iupac_in_bg = pp.from_iupac(
                "RNS",
                pool=spacer_tagged,
                region="iupac_site",
                mode="sequential",
            ).named("iupac_in_bg")
            iupac_clean = iupac_in_bg.remove_tags("iupac_site").named("iupac_clean")

            barcodes = pp.from_seqs(barcode_list_4, mode="sequential").named("barcodes")

            library_pre = pp.join([bg_mut, orf_neg, iupac_clean, barcodes])

            library = library_pre.filter(
                lambda s: s.count("AA") <= 3
            ).named("aa_filtered")

        assert bg_mut.num_states == 9
        assert orf_neg.num_states == 76
        assert iupac_clean.num_states == 16
        assert barcodes.num_states == 4
        assert library.num_states == 43_776


class TestLibrary17:
    """Random-Mode Scans + Reverse Translate (variable len, 590 states)."""

    def test_state_counts_and_sequences(self):
        bg_30nt = _random_dna(30, seed=170)
        bg2_24nt = _random_dna(24, seed=171)
        barcode_list_10 = _make_barcodes(10, length=10, seed=172)

        with pp.Party() as party:
            bg = pp.from_seq(bg_30nt)
            bg_del = bg.deletion_scan(
                deletion_length=3,
                mode="random",
                num_states=5,
            ).named("random_del_scan")

            bg2 = pp.from_seq(bg2_24nt)
            bg2_mutscan = bg2.mutagenize_scan(
                mutagenize_length=6,
                num_mutations=1,
                mode=("random", "sequential"),
                num_states=(3, None),  # None = auto-compute sequential mutagenize states
            ).named("random_pos_mutscan")

            rev_first = pp.reverse_translate(
                "MKVLA", codon_selection="first"
            ).named("rev_first")

            barcodes = pp.from_seqs(barcode_list_10, mode="sequential").named("barcodes")

            scan_mixed = pp.stack([bg_del, bg2_mutscan]).named("scan_arms")

            library = pp.join([scan_mixed, rev_first, barcodes])

        assert bg_del.num_states == 5
        assert bg2_mutscan.num_states == 54
        assert scan_mixed.num_states == 59
        assert rev_first.num_states == 1
        assert barcodes.num_states == 10
        assert library.num_states == 590

        # num_seqs sampling
        df_sample = pp.generate_library(library, num_seqs=200, seed=42)
        assert len(df_sample) == 200

        # num_cycles=2
        df_2x = pp.generate_library(library, num_cycles=2, seed=42)
        assert len(df_2x) == 590 * 2

        # Deterministic reverse translate: same output every time
        rev_df = pp.generate_library(rev_first, num_cycles=1, seed=42)
        rev_seqs = rev_df["seq"].tolist()
        assert len(rev_seqs) == 1
        rev_df2 = pp.generate_library(rev_first, num_cycles=1, seed=99)
        assert rev_df2["seq"].tolist() == rev_seqs
