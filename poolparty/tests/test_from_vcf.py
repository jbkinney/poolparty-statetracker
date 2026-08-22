"""Tests for the FromVcf operation."""

import textwrap
import warnings

import pytest

import poolparty as pp
from poolparty.base_ops.from_vcf import FromVcfOp, from_vcf

# A 400 bp reference on two contigs. Position 51 is 'C' on chr1 (1-based), which
# every SNV test below substitutes. The contig is deliberately longer than the
# default max_allele_length of 100, so an over-cap allele can be placed on it -
# with a 100 bp contig that default could not be exercised at all.
CHR1 = "AAAACCCCGG" * 5 + "CTTTTGGGGA" + "TTTTAAAACC" * 4 + "ACGTTGCAAT" * 30
CHR2 = "GGGGTTTTAA" * 10


def write_reference(tmp_path):
    """Write a two-contig FASTA and return its path."""
    path = tmp_path / "ref.fa"
    path.write_text(f">chr1\n{CHR1}\n>chr2\n{CHR2}\n")
    return str(path)


def write_vcf(tmp_path, body, name="v.vcf", chrom_style="chr"):
    """Write a minimal but well-formed VCF and return its path."""
    header = textwrap.dedent(
        """\
        ##fileformat=VCFv4.2
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
        """
    )
    if chrom_style == "bare":
        body = body.replace("chr1", "1").replace("chr2", "2")
    path = tmp_path / name
    path.write_text(header + body)
    return str(path)


def rows(pool):
    """Materialize a pool to a list of (name, seq) pairs in state order."""
    df = pool.to_df(num_cycles=1)
    return list(zip(df["name"], df["seq"]))


class TestWindowConstruction:
    """The emitted sequence for each allele."""

    def test_snv_window_is_flanks_plus_allele(self, tmp_path):
        """A SNV window carries the alt base flanked by reference sequence."""
        ref = write_reference(tmp_path)
        vcf = write_vcf(tmp_path, "chr1\t51\t.\tC\tG\t.\t.\t.\n")
        with pp.Party():
            got = dict(rows(from_vcf(vcf, ref, 5, 5)))
        # positions 46-50 and 52-56, 1-based, around the substituted base
        assert got["chr1_51_C"] == CHR1[45:50] + "C" + CHR1[51:56]
        assert got["chr1_51_C_G"] == CHR1[45:50] + "G" + CHR1[51:56]

    def test_deletion_shortens_alt_only(self, tmp_path):
        """A deletion leaves the ref window intact and shortens the alt window."""
        ref = write_reference(tmp_path)
        # REF spans positions 51-53; ALT keeps only the anchor base.
        vcf = write_vcf(tmp_path, f"chr1\t51\t.\t{CHR1[50:53]}\t{CHR1[50]}\t.\t.\t.\n")
        with pp.Party():
            got = dict(rows(from_vcf(vcf, ref, 4, 4)))
        ref_seq = next(v for k, v in got.items() if k.count("_") == 2)
        alt_seq = next(v for k, v in got.items() if k.count("_") == 3)
        assert len(ref_seq) == 4 + 3 + 4
        assert len(alt_seq) == 4 + 1 + 4

    def test_insertion_lengthens_alt_only(self, tmp_path):
        """An insertion leaves the ref window intact and lengthens the alt window."""
        ref = write_reference(tmp_path)
        vcf = write_vcf(tmp_path, f"chr1\t51\t.\t{CHR1[50]}\t{CHR1[50]}TTT\t.\t.\t.\n")
        with pp.Party():
            got = dict(rows(from_vcf(vcf, ref, 4, 4)))
        assert len(got["chr1_51_C"]) == 4 + 1 + 4
        assert len(got[f"chr1_51_C_C{'T' * 3}"]) == 4 + 4 + 4

    def test_asymmetric_flanks(self, tmp_path):
        """flank_left and flank_right are honoured independently."""
        ref = write_reference(tmp_path)
        vcf = write_vcf(tmp_path, "chr1\t51\t.\tC\tG\t.\t.\t.\n")
        with pp.Party():
            seq = dict(rows(from_vcf(vcf, ref, 10, 2, alleles="alt")))["chr1_51_C_G"]
        assert seq == CHR1[40:50] + "G" + CHR1[51:53]


class TestAlleleSelection:
    """The alleles= argument and reference de-duplication."""

    def test_one_reference_per_site_not_per_record(self, tmp_path):
        """Three records at one position share a single reference window."""
        ref = write_reference(tmp_path)
        vcf = write_vcf(
            tmp_path,
            "chr1\t51\t.\tC\tG\t.\t.\t.\nchr1\t51\t.\tC\tT\t.\t.\t.\nchr1\t51\t.\tC\tA\t.\t.\t.\n",
        )
        with pp.Party():
            pool = from_vcf(vcf, ref, 5, 5)
            names = [n for n, _ in rows(pool)]
        assert pool.num_states == 4
        assert names.count("chr1_51_C") == 1
        assert sorted(n for n in names if n != "chr1_51_C") == [
            "chr1_51_C_A",
            "chr1_51_C_G",
            "chr1_51_C_T",
        ]

    def test_multiallelic_row_splits_per_alt(self, tmp_path):
        """A comma-separated ALT yields one alt window per allele."""
        ref = write_reference(tmp_path)
        vcf = write_vcf(tmp_path, "chr1\t51\t.\tC\tG,T\t.\t.\t.\n")
        with pp.Party():
            names = sorted(n for n, _ in rows(from_vcf(vcf, ref, 3, 3)))
        assert names == ["chr1_51_C", "chr1_51_C_G", "chr1_51_C_T"]

    @pytest.mark.parametrize("alleles,expected", [("alt", 2), ("ref", 1), ("both", 3)])
    def test_state_counts(self, tmp_path, alleles, expected):
        """Two records at one site give M alts, one ref, and their sum."""
        ref = write_reference(tmp_path)
        vcf = write_vcf(tmp_path, "chr1\t51\t.\tC\tG\t.\t.\t.\nchr1\t51\t.\tC\tT\t.\t.\t.\n")
        with pp.Party():
            assert from_vcf(vcf, ref, 3, 3, alleles=alleles).num_states == expected


class TestDesignCards:
    """Design card contents."""

    def test_core_card_values(self, tmp_path):
        """Cards report VCF coordinates and a half-open window."""
        ref = write_reference(tmp_path)
        vcf = write_vcf(tmp_path, "chr1\t51\trs9\tC\tG\t.\tPASS\tAF=0.25\n")
        with pp.Party():
            df = from_vcf(
                vcf,
                ref,
                5,
                5,
                cards={
                    "chrom": "chrom",
                    "pos": "pos",
                    "allele": "allele",
                    "alt": "alt",
                    "variant_type": "vtype",
                    "variant_id": "vid",
                    "filter": "filt",
                    "window_start": "ws",
                    "window_stop": "wt",
                },
            ).to_df(num_cycles=1)
        alt = df[df["allele"] == "alt"].iloc[0]
        assert (alt["chrom"], alt["pos"], alt["alt"]) == ("chr1", 51, "G")
        assert alt["vtype"] == "snv"
        assert (alt["vid"], alt["filt"]) == ("rs9", "PASS")
        # pos is 1-based; the window is 0-based half-open
        assert (alt["ws"], alt["wt"]) == (45, 56)
        assert alt["wt"] - alt["ws"] == len(alt["seq"])

    def test_alt_is_null_on_reference_rows(self, tmp_path):
        """A reference row carries no alternate allele."""
        ref = write_reference(tmp_path)
        vcf = write_vcf(tmp_path, "chr1\t51\t.\tC\tG\t.\t.\t.\n")
        with pp.Party():
            df = from_vcf(vcf, ref, 3, 3, cards={"allele": "allele", "alt": "alt"}).to_df(
                num_cycles=1
            )
        assert df.loc[df["allele"] == "ref", "alt"].isna().all()
        assert df.loc[df["allele"] == "alt", "alt"].tolist() == ["G"]

    def test_variant_type_recovers_a_uniform_length_pool(self, tmp_path):
        """Filtering on variant_type isolates the SNVs, which share a length."""
        ref = write_reference(tmp_path)
        vcf = write_vcf(
            tmp_path,
            f"chr1\t51\t.\tC\tG\t.\t.\t.\n"
            f"chr1\t55\t.\t{CHR1[54]}\tA\t.\t.\t.\n"
            f"chr1\t61\t.\t{CHR1[60]}\t{CHR1[60]}TT\t.\t.\t.\n",
        )
        with pp.Party():
            df = from_vcf(vcf, ref, 4, 4, alleles="alt", cards={"variant_type": "vtype"}).to_df(
                num_cycles=1
            )
        assert set(df["vtype"]) == {"snv", "insertion"}
        snvs = df[df["vtype"] == "snv"]
        assert len(snvs) == 2 and snvs["seq"].str.len().nunique() == 1

    def test_info_fields_are_exposed_and_decoded(self, tmp_path):
        """Requested INFO keys become info_ cards, percent-decoded."""
        ref = write_reference(tmp_path)
        vcf = write_vcf(tmp_path, "chr1\t51\t.\tC\tG\t.\t.\tAF=0.5;NOTE=a%3Bb\n")
        with pp.Party():
            df = from_vcf(
                vcf,
                ref,
                3,
                3,
                alleles="alt",
                info_fields=["AF", "NOTE", "ABSENT"],
                cards={"info_AF": "af", "info_NOTE": "note", "info_ABSENT": "gone"},
            ).to_df(num_cycles=1)
        assert df.iloc[0]["af"] == "0.5"
        assert df.iloc[0]["note"] == "a;b"
        assert df["gone"].isna().all()


class TestSkipsAndFailures:
    """Records that cannot be represented, and inputs that must fail."""

    def test_reference_mismatch_is_skipped_with_a_warning(self, tmp_path):
        """A stray record whose REF disagrees with the FASTA is skipped, not trusted."""
        ref = write_reference(tmp_path)
        good = "".join(f"chr1\t{p}\t.\t{CHR1[p - 1]}\tA\t.\t.\t.\n" for p in range(61, 71))
        vcf = write_vcf(
            tmp_path,
            "chr1\t51\t.\tA\tG\t.\t.\t.\n" + good,  # position 51 is C, not A
        )
        with pp.Party(), pytest.warns(UserWarning, match="ref mismatch"):
            names = [n for n, _ in rows(from_vcf(vcf, ref, 3, 3, alleles="alt"))]
        assert "chr1_51_A_G" not in names
        assert len(names) == 10

    def test_widespread_mismatch_raises_rather_than_returning_a_pool(self, tmp_path):
        """A build mismatch must not yield a plausible, systematically wrong library."""
        ref = write_reference(tmp_path)
        # Enough records for a rate to be meaningful; below the floor a wholly
        # mismatched file still fails, through "no usable records".
        vcf = write_vcf(
            tmp_path,
            "".join(f"chr1\t{p}\t.\tA\tG\t.\t.\t.\n" for p in range(31, 76)),
        )
        with pp.Party(), pytest.raises(ValueError, match="compatible reference sequences"):
            from_vcf(vcf, ref, 3, 3, alleles="alt")

    def test_case_insensitive_reference_check(self, tmp_path):
        """A soft-masked reference still matches an uppercase VCF REF."""
        path = tmp_path / "soft.fa"
        path.write_text(f">chr1\n{CHR1.lower()}\n")
        vcf = write_vcf(tmp_path, "chr1\t51\t.\tC\tG\t.\t.\t.\n")
        with pp.Party():
            assert from_vcf(vcf, str(path), 3, 3, alleles="alt").num_states == 1

    def test_non_dna_alts_are_skipped(self, tmp_path):
        """Symbolic, missing and spanning-deletion alleles are skipped."""
        ref = write_reference(tmp_path)
        vcf = write_vcf(
            tmp_path,
            "chr1\t51\t.\tC\t.\t.\t.\t.\n"
            "chr1\t52\t.\t" + CHR1[51] + "\t<DEL>\t.\t.\t.\n"
            "chr1\t53\t.\t" + CHR1[52] + "\t*\t.\t.\t.\n"
            "chr1\t61\t.\t" + CHR1[60] + "\tA\t.\t.\t.\n",
        )
        with pp.Party(), pytest.warns(UserWarning, match="non dna alt"):
            names = [n for n, _ in rows(from_vcf(vcf, ref, 3, 3, alleles="alt"))]
        assert names == [f"chr1_61_{CHR1[60]}_A"]

    def test_max_allele_length_caps_the_window(self, tmp_path):
        """A long allele is skipped rather than silently widening the window."""
        ref = write_reference(tmp_path)
        long_ins = CHR1[50] + "T" * 40
        vcf = write_vcf(
            tmp_path,
            f"chr1\t51\t.\t{CHR1[50]}\t{long_ins}\t.\t.\t.\nchr1\t61\t.\t{CHR1[60]}\tA\t.\t.\t.\n",
        )
        with pp.Party(), pytest.warns(UserWarning, match="allele too long"):
            pool = from_vcf(vcf, ref, 3, 3, alleles="alt", max_allele_length=10)
            assert [n for n, _ in rows(pool)] == [f"chr1_61_{CHR1[60]}_A"]

    def test_window_off_contig_end_is_skipped(self, tmp_path):
        """A window that would run past a contig boundary is skipped."""
        ref = write_reference(tmp_path)
        vcf = write_vcf(
            tmp_path,
            "chr1\t2\t.\t" + CHR1[1] + "\tG\t.\t.\t.\n"  # needs 20 bases to the left
            "chr1\t51\t.\tC\tG\t.\t.\t.\n",
        )
        with pp.Party(), pytest.warns(UserWarning, match="off contig"):
            names = [n for n, _ in rows(from_vcf(vcf, ref, 20, 20, alleles="alt"))]
        assert names == ["chr1_51_C_G"]

    def test_contig_absent_from_reference_is_skipped(self, tmp_path):
        """A record on a contig the FASTA lacks does not abort the run."""
        ref = write_reference(tmp_path)
        vcf = write_vcf(
            tmp_path,
            "chrZ\t51\t.\tC\tG\t.\t.\t.\nchr1\t51\t.\tC\tG\t.\t.\t.\n",
        )
        with pp.Party(), pytest.warns(UserWarning, match="contig absent"):
            names = [n for n, _ in rows(from_vcf(vcf, ref, 3, 3, alleles="alt"))]
        assert names == ["chr1_51_C_G"]

    def test_chr_prefix_is_normalised(self, tmp_path):
        """A bare-numbered VCF works against a chr-prefixed reference."""
        ref = write_reference(tmp_path)
        vcf = write_vcf(tmp_path, "chr1\t51\t.\tC\tG\t.\t.\t.\n", chrom_style="bare")
        with pp.Party():
            names = [n for n, _ in rows(from_vcf(vcf, ref, 3, 3, alleles="alt"))]
        assert names == ["1_51_C_G"]  # the name keeps the VCF's spelling

    def test_no_usable_records_raises(self, tmp_path):
        """An empty result is an error, not an empty pool."""
        ref = write_reference(tmp_path)
        vcf = write_vcf(tmp_path, "chrZ\t51\t.\tC\tG\t.\t.\t.\n")
        with pp.Party(), pytest.warns(UserWarning):
            with pytest.raises(ValueError, match="No usable records"):
                from_vcf(vcf, ref, 3, 3)

    def test_malformed_line_raises_with_line_number(self, tmp_path):
        """A line with too few tab-separated fields names itself."""
        ref = write_reference(tmp_path)
        vcf = write_vcf(tmp_path, "chr1 51 . C G . . .\n")  # space-delimited
        with pp.Party(), pytest.raises(ValueError, match="line 3.*found 1"):
            from_vcf(vcf, ref, 3, 3)


class TestGzipAndComposition:
    """Gzip input, and behaviour as a pool in the wider package."""

    def test_reads_gzipped_vcf(self, tmp_path):
        """A .gz VCF is read without an index."""
        import gzip as gz

        ref = write_reference(tmp_path)
        path = tmp_path / "v.vcf.gz"
        with gz.open(path, "wt") as fh:
            fh.write(
                "##fileformat=VCFv4.2\n"
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
                "chr1\t51\t.\tC\tG\t.\t.\t.\n"
            )
        with pp.Party():
            assert from_vcf(str(path), ref, 3, 3, alleles="alt").num_states == 1

    @pytest.mark.parametrize("gzipped", [False, True], ids=["plain", "gzipped"])
    def test_byte_order_mark_is_tolerated(self, tmp_path, gzipped):
        """Windows tools write a BOM ahead of ##fileformat; it is not a data line."""
        import gzip as gz

        ref = write_reference(tmp_path)
        body = (
            "##fileformat=VCFv4.2\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
            "chr1\t51\t.\tC\tG\t.\t.\t.\n"
        )
        if gzipped:
            path = tmp_path / "bom.vcf.gz"
            with gz.open(path, "wt", encoding="utf-8-sig") as fh:
                fh.write(body)
        else:
            path = tmp_path / "bom.vcf"
            path.write_text(body, encoding="utf-8-sig")
        raw = gz.decompress(path.read_bytes()) if gzipped else path.read_bytes()
        assert raw[:3] == b"\xef\xbb\xbf"
        with pp.Party():
            names = [n for n, _ in rows(from_vcf(str(path), ref, 3, 3, alleles="alt"))]
        assert names == ["chr1_51_C_G"]

    def test_snv_only_pool_has_a_defined_seq_length(self, tmp_path):
        """Uniform windows keep seq_length, so length-dependent ops remain usable."""
        ref = write_reference(tmp_path)
        vcf = write_vcf(
            tmp_path,
            f"chr1\t51\t.\tC\tG\t.\t.\t.\nchr1\t61\t.\t{CHR1[60]}\tA\t.\t.\t.\n",
        )
        with pp.Party():
            pool = from_vcf(vcf, ref, 4, 4, alleles="alt")
            assert pool.seq_length == 9
            assert pp.subseq_scan(pool, 4).seq_length == 4

    def test_indel_pool_has_no_seq_length(self, tmp_path):
        """Mixed-length windows leave seq_length undefined, as documented."""
        ref = write_reference(tmp_path)
        vcf = write_vcf(
            tmp_path,
            f"chr1\t51\t.\tC\tG\t.\t.\t.\nchr1\t61\t.\t{CHR1[60]}\t{CHR1[60]}TT\t.\t.\t.\n",
        )
        with pp.Party():
            assert from_vcf(vcf, ref, 4, 4, alleles="alt").seq_length is None

    def test_prefix_is_folded_into_the_name(self, tmp_path):
        """A prefix is applied, and its absence leaves the bare name."""
        ref = write_reference(tmp_path)
        vcf = write_vcf(tmp_path, "chr1\t51\t.\tC\tG\t.\t.\t.\n")
        with pp.Party():
            bare = [n for n, _ in rows(from_vcf(vcf, ref, 3, 3, alleles="alt"))]
            with_prefix = [n for n, _ in rows(from_vcf(vcf, ref, 3, 3, alleles="alt", prefix="v"))]
        assert bare == ["chr1_51_C_G"]
        assert with_prefix == ["v_chr1_51_C_G"]

    def test_operation_type_and_copy(self, tmp_path):
        """The factory builds a FromVcfOp that survives a copy."""
        ref = write_reference(tmp_path)
        vcf = write_vcf(tmp_path, "chr1\t51\t.\tC\tG\t.\t.\t.\n")
        with pp.Party():
            pool = from_vcf(vcf, ref, 3, 3, alleles="alt")
            assert isinstance(pool.operation, FromVcfOp)
            assert pool.operation.copy().num_states == pool.num_states


class TestEdgeCases:
    """Classification, validation and coordinate handling at the edges."""

    def test_reference_row_carries_no_variant_metadata(self, tmp_path):
        """A shared reference row must not inherit one arbitrary variant's identity."""
        ref = write_reference(tmp_path)
        # An insertion listed first, then a SNV, at the same site.
        vcf = write_vcf(
            tmp_path,
            f"chr1\t51\trsINS\tC\tC{'T' * 3}\t.\tLOWQUAL\tCLNSIG=Benign\n"
            "chr1\t51\trsSNV\tC\tG\t.\tPASS\tCLNSIG=Pathogenic\n",
        )
        with pp.Party():
            df = from_vcf(
                vcf,
                ref,
                4,
                4,
                info_fields=["CLNSIG"],
                cards={
                    "allele": "allele",
                    "alt": "alt",
                    "variant_type": "vt",
                    "variant_id": "vid",
                    "filter": "filt",
                    "info_CLNSIG": "sig",
                },
            ).to_df(num_cycles=1)
        refrow = df[df["allele"] == "ref"]
        assert len(refrow) == 1
        for col in ("alt", "vt", "vid", "filt", "sig"):
            assert refrow[col].isna().all(), f"{col} leaked onto the reference row"
        # the alternate rows keep their own values
        assert set(df.loc[df["allele"] == "alt", "vid"]) == {"rsINS", "rsSNV"}

    def test_equal_length_multibase_is_a_substitution(self, tmp_path):
        """Two bases for two different bases deletes nothing."""
        ref = write_reference(tmp_path)
        two = CHR1[50:52]
        vcf = write_vcf(tmp_path, f"chr1\t51\t.\t{two}\tTT\t.\t.\t.\n")
        with pp.Party():
            df = from_vcf(vcf, ref, 4, 4, alleles="alt", cards={"variant_type": "vt"}).to_df(
                num_cycles=1
            )
        assert df.iloc[0]["vt"] == "substitution"
        assert len(df.iloc[0]["seq"]) == 4 + 2 + 4

    def test_variant_types_filter_gives_a_uniform_pool(self, tmp_path):
        """Restricting to length-preserving types restores seq_length."""
        ref = write_reference(tmp_path)
        vcf = write_vcf(
            tmp_path,
            f"chr1\t51\t.\tC\tG\t.\t.\t.\nchr1\t55\t.\t{CHR1[54]}\t{CHR1[54]}TT\t.\t.\t.\n",
        )
        with pp.Party():
            assert from_vcf(vcf, ref, 4, 4, alleles="alt").seq_length is None
            snv_only = from_vcf(vcf, ref, 4, 4, alleles="alt", variant_types=["snv"])
            assert snv_only.seq_length == 9
            assert pp.subseq_scan(snv_only, 4).seq_length == 4

    def test_variant_types_rejects_unknown_values(self, tmp_path):
        """A typo in variant_types names the valid set."""
        ref = write_reference(tmp_path)
        vcf = write_vcf(tmp_path, "chr1\t51\t.\tC\tG\t.\t.\t.\n")
        with pp.Party(), pytest.raises(ValueError, match="substitution"):
            from_vcf(vcf, ref, 3, 3, variant_types=["snp"])

    def test_breakend_and_ambiguous_alts_are_skipped(self, tmp_path):
        """Only ACGT alleles build a window; N and breakends do not."""
        ref = write_reference(tmp_path)
        vcf = write_vcf(
            tmp_path,
            "chr1\t51\t.\tC\tC]chr2:99]\t.\t.\t.\n"
            f"chr1\t55\t.\t{CHR1[54]}\tN\t.\t.\t.\n"
            f"chr1\t61\t.\t{CHR1[60]}\tR\t.\t.\t.\n"
            f"chr1\t65\t.\t{CHR1[64]}\tA\t.\t.\t.\n",
        )
        with pp.Party(), pytest.warns(UserWarning, match="non dna alt"):
            names = [n for n, _ in rows(from_vcf(vcf, ref, 3, 3, alleles="alt"))]
        assert names == [f"chr1_65_{CHR1[64]}_A"]

    def test_reference_window_survives_a_skipped_alt(self, tmp_path):
        """An unusable alternate allele must not remove a valid reference window."""
        ref = write_reference(tmp_path)
        vcf = write_vcf(tmp_path, "chr1\t51\t.\tC\t*\t.\t.\t.\n")
        # Under alleles="ref" with no type filter the alternates are never
        # inspected, so nothing is reported as skipped - the record contributed
        # exactly the window that was asked for.
        with pp.Party(), warnings.catch_warnings():
            warnings.simplefilter("error")
            names = [n for n, _ in rows(from_vcf(vcf, ref, 3, 3, alleles="ref"))]
        assert names == ["chr1_51_C"]
        # ...but with alleles="both" the unusable alternate is reported, and the
        # site still yields its reference window.
        with pp.Party(), pytest.warns(UserWarning, match="non dna alt"):
            both = [n for n, _ in rows(from_vcf(vcf, ref, 3, 3, alleles="both"))]
        assert both == ["chr1_51_C"]

    def test_negative_flanks_are_rejected(self, tmp_path):
        """A negative flank would invert the window."""
        ref = write_reference(tmp_path)
        vcf = write_vcf(tmp_path, "chr1\t51\t.\tC\tG\t.\t.\t.\n")
        with pp.Party(), pytest.raises(ValueError, match="flanks must be >= 0"):
            from_vcf(vcf, ref, -3, 5)

    def test_long_ref_is_capped_by_default(self, tmp_path):
        """The default cap fires on a long REF, not only a long ALT."""
        ref = write_reference(tmp_path)
        long_ref = CHR1[50:95]  # 45 bases
        vcf = write_vcf(tmp_path, f"chr1\t51\t.\t{long_ref}\tC\t.\t.\t.\n")
        with pp.Party():
            with pytest.warns(UserWarning, match="allele too long"):
                with pytest.raises(ValueError, match="No usable records"):
                    from_vcf(vcf, ref, 2, 2, alleles="alt", max_allele_length=10)
            # raising the cap admits it
            pool = from_vcf(vcf, ref, 2, 2, alleles="alt", max_allele_length=100)
            assert pool.num_states == 1

    def test_mixed_contig_spelling_keeps_names_self_consistent(self, tmp_path):
        """Both spellings resolve to one contig, but each keeps its own name.

        A reference name must remain a strict prefix of its variants' names, which
        is how a reader tells them apart without design cards. Sharing one
        reference window across spellings would orphan the other spelling's
        variants, so the site is keyed on the VCF's own text.
        """
        ref = write_reference(tmp_path)
        vcf = write_vcf(tmp_path, "chr1\t51\t.\tC\tG\t.\t.\t.\n1\t51\t.\tC\tT\t.\t.\t.\n")
        with pp.Party():
            df = from_vcf(vcf, ref, 3, 3, cards={"allele": "allele"}).to_df(num_cycles=1)
        refs = set(df.loc[df["allele"] == "ref", "name"])
        assert refs == {"chr1_51_C", "1_51_C"}
        assert df.loc[df["allele"] == "ref", "seq"].nunique() == 1  # same window
        for alt_name in df.loc[df["allele"] == "alt", "name"]:
            assert any(alt_name.startswith(r + "_") for r in refs)

    def test_two_refs_at_one_position_stay_separate(self, tmp_path):
        """Records sharing a position but differing in REF are different sites."""
        ref = write_reference(tmp_path)
        vcf = write_vcf(
            tmp_path,
            f"chr1\t51\t.\t{CHR1[50]}\tG\t.\t.\t.\n"
            f"chr1\t51\t.\t{CHR1[50:53]}\t{CHR1[50]}\t.\t.\t.\n",
        )
        with pp.Party():
            df = from_vcf(vcf, ref, 3, 3, cards={"allele": "allele"}).to_df(num_cycles=1)
        refs = df[df["allele"] == "ref"]
        assert len(refs) == 2 and refs["seq"].nunique() == 2

    def test_window_at_the_right_contig_edge_is_skipped(self, tmp_path):
        """The right-hand bound is checked, not only the left."""
        ref = write_reference(tmp_path)
        last = len(CHR1)
        vcf = write_vcf(
            tmp_path,
            f"chr1\t{last}\t.\t{CHR1[-1]}\tA\t.\t.\t.\nchr1\t51\t.\tC\tG\t.\t.\t.\n",
        )
        with pp.Party(), pytest.warns(UserWarning, match="off contig"):
            names = [n for n, _ in rows(from_vcf(vcf, ref, 3, 3, alleles="alt"))]
        assert names == ["chr1_51_C_G"]

    def test_blank_and_malformed_lines(self, tmp_path):
        """A blank line is tolerated; a non-integer POS names its line."""
        ref = write_reference(tmp_path)
        with pp.Party():
            ok = write_vcf(tmp_path, "chr1\t51\t.\tC\tG\t.\t.\t.\n\n", name="blank.vcf")
            assert from_vcf(ok, ref, 3, 3, alleles="alt").num_states == 1
            bad = write_vcf(tmp_path, "chr1\tNA\t.\tC\tG\t.\t.\t.\n", name="pos.vcf")
            with pytest.raises(ValueError, match="line 3: POS is not an integer"):
                from_vcf(bad, ref, 3, 3)

    def test_indel_window_content_is_anchored_on_pos(self, tmp_path):
        """The deleted bases are those after POS, and the flanks are exact."""
        ref = write_reference(tmp_path)
        # delete positions 52-53, keeping the anchor at 51
        deleted = CHR1[50:53]
        vcf = write_vcf(tmp_path, f"chr1\t51\t.\t{deleted}\t{CHR1[50]}\t.\t.\t.\n")
        with pp.Party():
            got = dict(rows(from_vcf(vcf, ref, 4, 4)))
        assert got[f"chr1_51_{deleted}"] == CHR1[46:50] + deleted + CHR1[53:57]
        assert got[f"chr1_51_{deleted}_{CHR1[50]}"] == CHR1[46:50] + CHR1[50] + CHR1[53:57]

    def test_skip_counts_are_exposed_on_the_operation(self, tmp_path):
        """Counts survive the warning, so a caller can inspect them."""
        ref = write_reference(tmp_path)
        vcf = write_vcf(
            tmp_path,
            f"chr1\t51\t.\tC\t*\t.\t.\t.\nchr1\t61\t.\t{CHR1[60]}\tA\t.\t.\t.\n",
        )
        with pp.Party(), pytest.warns(UserWarning):
            pool = from_vcf(vcf, ref, 3, 3, alleles="alt")
        assert pool.operation.skipped["non_dna_alt"] == 1
        assert pool.operation.skipped["ref_mismatch"] == 0


class TestGuardsAtTheirDefaults:
    """Guards exercised through the default arguments, not explicit ones.

    Passing a value explicitly leaves the default itself unpinned, so each guard
    is asserted here with the argument omitted.
    """

    def test_long_ref_capped_without_passing_the_argument(self, tmp_path):
        """The default cap gates the reference window, not only the alternate."""
        ref = write_reference(tmp_path)
        long_ref = CHR1[50:95]  # 45 bases, but the default cap is 100
        vcf = write_vcf(
            tmp_path,
            f"chr1\t51\t.\t{long_ref}\tC\t.\t.\t.\nchr1\t20\t.\t{CHR1[19]}\tA\t.\t.\t.\n",
        )
        with pp.Party():
            # 45 bases is under the default, so both windows appear
            assert from_vcf(vcf, ref, 2, 2).num_states == 4
            # tightened below it, the long record contributes nothing at all -
            # neither its alternate nor its reference window
            with pytest.warns(UserWarning, match="allele too long"):
                pool = from_vcf(vcf, ref, 2, 2, max_allele_length=10)
            assert [n for n, _ in rows(pool)] == [
                f"chr1_20_{CHR1[19]}",
                f"chr1_20_{CHR1[19]}_A",
            ]

    def test_variant_types_gates_the_reference_window_at_default_alleles(self, tmp_path):
        """Filtering by type must not leave the excluded types' reference windows."""
        ref = write_reference(tmp_path)
        vcf = write_vcf(
            tmp_path,
            f"chr1\t51\t.\tC\tG\t.\t.\t.\nchr1\t61\t.\t{CHR1[60:63]}\t{CHR1[60]}\t.\t.\t.\n",
        )
        with pp.Party():
            # alleles defaults to "both" - the deletion's reference window must go
            pool = from_vcf(vcf, ref, 4, 4, variant_types=["snv"])
            assert [n for n, _ in rows(pool)] == ["chr1_51_C", "chr1_51_C_G"]
            assert pool.seq_length == 9

    def test_non_dna_ref_is_skipped(self, tmp_path):
        """An N in the reference is unusable even though it matches the FASTA."""
        path = tmp_path / "gap.fa"
        path.write_text(">chr1\n" + "ACGT" * 5 + "NNN" + "ACGT" * 5 + "\n")
        vcf = write_vcf(
            tmp_path,
            "chr1\t21\t.\tNNN\tACG\t.\t.\t.\n"  # REF is the gap itself
            "chr1\t5\t.\tA\tG\t.\t.\t.\n",
        )
        with pp.Party(), pytest.warns(UserWarning, match="non dna ref"):
            names = [n for n, _ in rows(from_vcf(vcf, str(path), 2, 2, alleles="alt"))]
        assert names == ["chr1_5_A_G"]

    def test_gap_in_the_flanks_is_skipped(self, tmp_path):
        """An assembly gap in the flanks is as unusable as one in the allele."""
        path = tmp_path / "gap.fa"
        path.write_text(">chr1\n" + "ACGT" * 5 + "N" + "ACGT" * 5 + "\n")
        vcf = write_vcf(
            tmp_path,
            "chr1\t19\t.\tG\tA\t.\t.\t.\n"  # 2 bases from the N
            "chr1\t5\t.\tA\tG\t.\t.\t.\n",
        )
        with pp.Party(), pytest.warns(UserWarning, match="gap in window"):
            names = [n for n, _ in rows(from_vcf(vcf, str(path), 3, 3, alleles="alt"))]
        assert names == ["chr1_5_A_G"]

    def test_windows_are_uppercased(self, tmp_path):
        """A soft-masked reference must not mark the variant position by case."""
        path = tmp_path / "soft.fa"
        path.write_text(">chr1\n" + CHR1.lower() + "\n")
        vcf = write_vcf(tmp_path, "chr1\t51\t.\tC\tG\t.\t.\t.\n")
        with pp.Party():
            got = dict(rows(from_vcf(vcf, str(path), 4, 4)))
        assert got["chr1_51_C"] == got["chr1_51_C"].upper()
        assert got["chr1_51_C_G"] == got["chr1_51_C_G"].upper()
        # the pair differs at one position, and only in base
        pair = list(zip(got["chr1_51_C"], got["chr1_51_C_G"]))
        assert sum(a != b for a, b in pair) == 1

    def test_chrm_and_mt_name_the_same_contig(self, tmp_path):
        """A UCSC-style chrM VCF resolves against an Ensembl MT reference."""
        path = tmp_path / "mt.fa"
        path.write_text(">MT\n" + CHR1 + "\n")
        vcf = write_vcf(tmp_path, "chrM\t51\t.\tC\tG\t.\t.\t.\n")
        with pp.Party():
            names = [n for n, _ in rows(from_vcf(vcf, str(path), 3, 3, alleles="alt"))]
        assert names == ["chrM_51_C_G"]  # the VCF's spelling is kept

    def test_variant_type_exclusions_are_not_warned_about(self, tmp_path):
        """A filter the caller asked for is not a skip to report."""
        ref = write_reference(tmp_path)
        vcf = write_vcf(
            tmp_path,
            f"chr1\t51\t.\tC\tG\t.\t.\t.\nchr1\t61\t.\t{CHR1[60:63]}\t{CHR1[60]}\t.\t.\t.\n",
        )
        with pp.Party(), warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning fails the test
            pool = from_vcf(vcf, ref, 4, 4, alleles="alt", variant_types=["snv"])
        assert pool.operation.skipped["variant_type"] == 1  # counted, not warned

    def test_top_level_export(self):
        """Both the factory and its Operation are reachable as pp.*."""
        assert callable(pp.from_vcf)
        assert pp.FromVcfOp is FromVcfOp

    def test_copy_preserves_cards_and_rows(self, tmp_path):
        """copy() must carry the windows and every card key, not just the count."""
        ref = write_reference(tmp_path)
        vcf = write_vcf(tmp_path, "chr1\t51\t.\tC\tG\t.\tPASS\tAF=0.5\n")
        with pp.Party():
            pool = from_vcf(vcf, ref, 3, 3, alleles="alt", info_fields=["AF"], prefix="p")
            clone = pool.operation.copy()
        assert clone.rows == pool.operation.rows
        assert clone.design_card_keys == pool.operation.design_card_keys
        assert "info_AF" in clone.design_card_keys
        assert clone.prefix == "p"

    def test_crlf_vcf_parses(self, tmp_path):
        """A CRLF-terminated VCF is a real input shape and must parse cleanly."""
        ref = write_reference(tmp_path)
        path = tmp_path / "crlf.vcf"
        path.write_bytes(
            b"##fileformat=VCFv4.2\r\n"
            b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\r\n"
            b"chr1\t51\t.\tC\tG\t.\tPASS\tAF=0.5\r\n"
        )
        with pp.Party():
            df = from_vcf(
                str(path),
                ref,
                3,
                3,
                alleles="alt",
                info_fields=["AF"],
                cards={"info_AF": "af", "filter": "filt"},
            ).to_df(num_cycles=1)
        assert df.iloc[0]["af"] == "0.5"
        assert df.iloc[0]["filt"] == "PASS"


class TestGuardTable:
    """One test per row of the guard table in ``_read_windows``.

    Each asserts both the outcome and which reason it was counted under, since
    the reason is how a caller diagnoses their input.
    """

    def test_gap_at_the_variant_position_is_a_gap_not_a_mismatch(self, tmp_path):
        """An N where the variant sits is an assembly gap, not a wrong build."""
        path = tmp_path / "gap.fa"
        path.write_text(">chr1\n" + "ACGT" * 10 + "N" + "ACGT" * 10 + "\n")
        vcf = write_vcf(tmp_path, "chr1\t41\t.\tA\tG\t.\t.\t.\n")
        with pp.Party(), pytest.warns(UserWarning, match="gap in window"):
            with pytest.raises(ValueError, match="No usable records"):
                from_vcf(vcf, str(path), 3, 3, alleles="alt")

    def test_gap_in_the_left_flank_is_caught(self, tmp_path):
        """Both flanks are checked, not only the right one."""
        path = tmp_path / "gap.fa"
        path.write_text(">chr1\n" + "ACGT" * 5 + "N" + "ACGT" * 10 + "\n")
        vcf = write_vcf(
            tmp_path,
            "chr1\t24\t.\tG\tA\t.\t.\t.\n"  # N is 3 bases to its left
            "chr1\t50\t.\tA\tG\t.\t.\t.\n",
        )
        with pp.Party(), pytest.warns(UserWarning, match="gap in window"):
            names = [n for n, _ in rows(from_vcf(vcf, str(path), 4, 4, alleles="alt"))]
        assert names == ["chr1_50_A_G"]

    def test_iupac_ref_is_not_charged_to_the_mismatch_rate(self, tmp_path):
        """A REF that is neither ACGT nor equal to the FASTA is not a build error."""
        ref = write_reference(tmp_path)
        vcf = write_vcf(
            tmp_path,
            "chr1\t51\t.\tR\tG\t.\t.\t.\n"  # R disagrees with the FASTA's C
            f"chr1\t61\t.\t{CHR1[60]}\tA\t.\t.\t.\n",
        )
        with pp.Party(), pytest.warns(UserWarning, match="non dna ref"):
            pool = from_vcf(vcf, ref, 3, 3, alleles="alt")
        assert pool.operation.skipped["non_dna_ref"] == 1
        assert pool.operation.skipped["ref_mismatch"] == 0

    def test_empty_ref_is_rejected(self, tmp_path):
        """An empty REF field would otherwise pass every check."""
        ref = write_reference(tmp_path)
        vcf = write_vcf(tmp_path, "chr1\t51\t.\t\tG\t.\t.\t.\n")
        with pp.Party(), pytest.warns(UserWarning, match="non dna ref"):
            with pytest.raises(ValueError, match="No usable records"):
                from_vcf(vcf, ref, 3, 3)

    def test_over_cap_allele_skipped_at_the_default(self, tmp_path):
        """The default cap of 100 is exercised as a default, not passed in."""
        ref = write_reference(tmp_path)
        long_alt = CHR1[50] + "T" * 150  # 151 bases, over the default
        vcf = write_vcf(
            tmp_path,
            f"chr1\t51\t.\t{CHR1[50]}\t{long_alt}\t.\t.\t.\nchr1\t61\t.\t{CHR1[60]}\tA\t.\t.\t.\n",
        )
        with pp.Party(), pytest.warns(UserWarning, match="allele too long"):
            names = [n for n, _ in rows(from_vcf(vcf, ref, 3, 3, alleles="alt"))]
        assert names == [f"chr1_61_{CHR1[60]}_A"]

    def test_mismatch_rate_below_the_limit_returns_a_pool(self, tmp_path):
        """The limit is a rate, so a minority of bad records is survivable."""
        ref = write_reference(tmp_path)
        good = "".join(f"chr1\t{p}\t.\t{CHR1[p - 1]}\tA\t.\t.\t.\n" for p in range(61, 91))
        vcf = write_vcf(tmp_path, "chr1\t51\t.\tA\tG\t.\t.\t.\n" + good)
        with pp.Party(), pytest.warns(UserWarning, match="ref mismatch"):
            pool = from_vcf(vcf, ref, 3, 3, alleles="alt")  # 1/31 = 3%
        assert pool.num_states == 30

    def test_mismatch_rate_applies_at_any_file_size(self, tmp_path):
        """No record-count floor: a two-record wrong build still fails."""
        ref = write_reference(tmp_path)
        vcf = write_vcf(
            tmp_path,
            f"chr1\t51\t.\tA\tG\t.\t.\t.\nchr1\t61\t.\t{CHR1[60]}\tA\t.\t.\t.\n",
        )
        with pp.Party(), pytest.raises(ValueError, match="compatible reference sequences"):
            from_vcf(vcf, ref, 3, 3, alleles="alt")  # 1/2 = 50%

    def test_unrepresentable_alt_under_a_type_filter_drops_the_record(self, tmp_path):
        """A filtered site contributes nothing, whatever made its alleles unusable."""
        ref = write_reference(tmp_path)
        vcf = write_vcf(
            tmp_path,
            f"chr1\t51\t.\t{CHR1[50:53]}\t*\t.\t.\t.\n"  # symbolic, not a type
            f"chr1\t61\t.\t{CHR1[60]}\tA\t.\t.\t.\n",
        )
        with pp.Party(), pytest.warns(UserWarning, match="non dna alt"):
            pool = from_vcf(vcf, ref, 4, 4, variant_types=["snv"])
        # no orphan reference window for the filtered site, so the pool is uniform
        assert [n for n, _ in rows(pool)] == [
            f"chr1_61_{CHR1[60]}",
            f"chr1_61_{CHR1[60]}_A",
        ]
        assert pool.seq_length == 9

    def test_bare_string_arguments_are_rejected(self, tmp_path):
        """A str satisfies Sequence[str], so it must be rejected explicitly."""
        ref = write_reference(tmp_path)
        vcf = write_vcf(tmp_path, "chr1\t51\t.\tC\tG\t.\t.\t.\n")
        with pp.Party():
            for kwargs in ({"variant_types": "snv"}, {"info_fields": "AF"}):
                with pytest.raises(ValueError, match="not a bare string"):
                    from_vcf(vcf, ref, 3, 3, **kwargs)

    def test_meaningless_filter_values_are_rejected(self, tmp_path):
        """An empty or non-positive filter is a mistake, not a request for nothing."""
        ref = write_reference(tmp_path)
        vcf = write_vcf(tmp_path, "chr1\t51\t.\tC\tG\t.\t.\t.\n")
        with pp.Party():
            with pytest.raises(ValueError, match="at least one type"):
                from_vcf(vcf, ref, 3, 3, variant_types=[])
            with pytest.raises(ValueError, match="must be >= 1"):
                from_vcf(vcf, ref, 3, 3, max_allele_length=0)

    def test_ref_card_and_name_are_uppercased(self, tmp_path):
        """A lowercase VCF REF must not leak case into the card or the name."""
        ref = write_reference(tmp_path)
        vcf = write_vcf(tmp_path, "chr1\t51\t.\tc\tg\t.\t.\t.\n")
        with pp.Party():
            df = from_vcf(vcf, ref, 3, 3, cards={"ref": "ref", "alt": "alt"}).to_df(num_cycles=1)
        assert set(df["name"]) == {"chr1_51_C", "chr1_51_C_G"}
        assert set(df["ref"]) == {"C"}
        assert set(df["alt"].dropna()) == {"G"}
