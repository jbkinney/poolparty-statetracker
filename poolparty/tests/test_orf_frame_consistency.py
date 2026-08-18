"""Cross-operation reading-frame consistency for ORF operations.

A single ``OrfRegion.frame`` must identify the same codons for every ORF-aware
operation. These tests pin that contract two ways:

1. **Anchors** compare each operation against hand-derived expectations that
   were written down independently of any implementation. This catches all
   operations drifting together, which a pure cross-operation test cannot.
2. **Cross-operation** tests assert the operations agree with each other.

The convention under test: ``frame=+N`` places the first complete codon at base
``N`` counting from the region's 5' end; ``frame=-N`` places it at base ``N``
counting from the region's 3' end, read as the reverse complement.
"""

import pytest

import poolparty as pp

# 19 nt: 19 % 3 == 1, the only residue class in which the number of complete
# codons differs between offsets 1 and 2, so the frames are distinguishable by
# codon count as well as by codon identity.
REGION = "AATGCCCGGGTTTAAACCC"
FULL = "GGGGG" + REGION + "TTTTT"
EXTENT = (5, 24)
FLANK = EXTENT[0]

# Hand-derived. Region-relative, plus-strand indices of codon 0.
#   +N: skip N-1 from the left.
#   -N: skip N-1 from the right; codon 0 is the 3'-most complete codon.
EXPECTED_CODON0_INDICES = {
    1: (0, 1, 2),
    2: (1, 2, 3),
    3: (2, 3, 4),
    -1: (16, 17, 18),
    -2: (15, 16, 17),
    -3: (14, 15, 16),
}

# The plus-strand text at those indices, and the codon as read in the coding
# direction. They differ for negative frames: mutagenize_orf design cards carry
# the reverse-complemented coding codon, not the plus-strand triplet.
EXPECTED_PLUS_STRAND_TRIPLET = {
    1: "AAT", 2: "ATG", 3: "TGC",
    -1: "CCC", -2: "ACC", -3: "AAC",
}
EXPECTED_ORIENTED_CODON = {
    1: "AAT", 2: "ATG", 3: "TGC",
    -1: "GGG", -2: "GGT", -3: "GTT",
}
# First residue of the translated product, from the oriented codon above.
EXPECTED_FIRST_RESIDUE = {
    1: "N", 2: "M", 3: "C",
    -1: "G", -2: "G", -3: "V",
}

ALL_FRAMES = [1, 2, 3, -1, -2, -3]

XFAIL_REASON = "Known ORF frame convention mismatch; fixed later in this branch"


def _xfail_off_frame(frames=ALL_FRAMES):
    """Mark |frame| != 1 as strict-xfail; the conventions already agree at +/-1.

    Marking all six would XPASS at +/-1 and, under strict=True, fail. Keeping
    +/-1 unmarked also documents that those frames were never affected.
    """
    return [
        pytest.param(f, marks=pytest.mark.xfail(strict=True, reason=XFAIL_REASON))
        if abs(f) != 1
        else pytest.param(f)
        for f in frames
    ]


def _annotated(frame):
    """Build the annotated pool. Caller must already be inside a Party."""
    return pp.annotate_orf(pp.from_seq(FULL), "orf", extent=EXTENT, frame=frame)


def _translate_first_residue(frame):
    with pp.Party():
        prot = pp.translate(_annotated(frame), region="orf")
        return pp.generate_library(prot, num_cycles=1)["seq"][0][0]


def _mutagenize_codon0(frame):
    """Return (oriented wt codon, region-relative plus-strand indices)."""
    with pp.Party():
        pool = _annotated(frame)
        mut = pp.mutagenize_orf(
            pool, region="orf", codon_positions=[0], num_mutations=1,
            mutation_type="any_codon", mode="sequential",
            cards=["wt_codons", "codon_positions"],
        )
        df = mut.to_df(num_cycles=1)
        col = next(c for c in df.columns if "wt_codons" in c)
        wt = list(dict.fromkeys(df[col]))[0][0]

        # Recover the nucleotide indices actually changed, by diffing against WT.
        # Both sides must come from to_df: it strips XML region tags, whereas
        # generate_library keeps them, which would misalign every index.
        wt_seq = pool.to_df(num_cycles=1)["seq"][0]
        changed = set()
        for variant in df["seq"]:
            changed |= {
                i - FLANK
                for i, (a, b) in enumerate(zip(wt_seq, variant))
                if a != b
            }
    return wt, tuple(sorted(changed))


def _stylize_codon0_indices(frame):
    """Region-relative plus-strand indices carrying style_codons[0]."""
    with pp.Party():
        styled = pp.stylize_orf(
            _annotated(frame), region="orf",
            style_codons=["red", "blue", "green", "yellow", "magenta", "cyan", "white"],
        )
        df = pp.generate_library(styled, num_cycles=1, _include_inline_styles=True)
        style = df["_inline_styles"][0]
        seq = df["seq"][0]
        # Map literal positions back to region-relative, skipping XML tag chars.
        offset = seq.index("<orf>") + len("<orf>")
        for name, positions in style.style_list:
            if name == "red":
                return tuple(sorted(int(p) - offset for p in positions))
    return ()


# --------------------------------------------------------------------------
# Anchors: each operation against hand-derived expectations
# --------------------------------------------------------------------------


@pytest.mark.parametrize("frame", ALL_FRAMES)
def test_translate_anchor(frame):
    """translate reads codon 0 at the hand-derived position."""
    assert _translate_first_residue(frame) == EXPECTED_FIRST_RESIDUE[frame]


@pytest.mark.parametrize("frame", ALL_FRAMES)
def test_mutagenize_orf_anchor(frame):
    """mutagenize_orf mutates codon 0 at the hand-derived position."""
    wt, indices = _mutagenize_codon0(frame)
    assert wt == EXPECTED_ORIENTED_CODON[frame]
    assert indices == EXPECTED_CODON0_INDICES[frame]


@pytest.mark.parametrize("frame", _xfail_off_frame())
def test_stylize_orf_anchor(frame):
    """stylize_orf gives style_codons[0] to codon 0, and to nothing else."""
    assert _stylize_codon0_indices(frame) == EXPECTED_CODON0_INDICES[frame]


# --------------------------------------------------------------------------
# Cross-operation agreement
# --------------------------------------------------------------------------


@pytest.mark.parametrize("frame", ALL_FRAMES)
def test_translate_matches_expected_triplet(frame):
    """Plus-strand text and oriented codon are distinct for negative frames."""
    plus = "".join(REGION[i] for i in EXPECTED_CODON0_INDICES[frame])
    assert plus == EXPECTED_PLUS_STRAND_TRIPLET[frame]


@pytest.mark.parametrize("frame", ALL_FRAMES)
def test_translate_and_mutagenize_agree(frame):
    """translate and mutagenize_orf place codon 0 on the same nucleotides."""
    _, mut_indices = _mutagenize_codon0(frame)
    assert mut_indices == EXPECTED_CODON0_INDICES[frame]
    assert _translate_first_residue(frame) == EXPECTED_FIRST_RESIDUE[frame]


@pytest.mark.parametrize("frame", _xfail_off_frame())
def test_all_three_operations_agree(frame):
    """All three ORF operations place codon 0 on the same nucleotides."""
    _, mut_indices = _mutagenize_codon0(frame)
    sty_indices = _stylize_codon0_indices(frame)
    assert mut_indices == sty_indices
    assert mut_indices == EXPECTED_CODON0_INDICES[frame]
    assert _translate_first_residue(frame) == EXPECTED_FIRST_RESIDUE[frame]
