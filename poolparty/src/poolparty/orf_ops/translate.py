"""Translate operation - convert DNA sequences to protein sequences."""

import numpy as np

from ..codon_table import CodonTable
from ..operation import Operation
from ..party import get_active_party
from ..pool import Pool
from ..region import VALID_FRAMES, OrfRegion
from ..types import NullSeq, Optional, Real, RegionType, Seq, Union, beartype, is_null_seq
from ..utils.dna_utils import reverse_complement
from ..utils.protein_seq import ProteinSeq
from ..utils.style_utils import SeqStyle


def _resolve_frame(region: RegionType, frame: Optional[int]) -> int:
    """Resolve the frame value, looking up from OrfRegion if needed.

    Backward compatibility: defaults to frame=1 when region is None or an interval.
    When region is a named OrfRegion, uses the stored frame.
    When region is a named plain Region, raises an error (must specify frame).
    """
    if frame is not None:
        if frame not in VALID_FRAMES:
            raise ValueError(f"frame must be one of {sorted(VALID_FRAMES)}, got {frame}")
        return frame

    # frame is None - try to get from OrfRegion or use default
    if region is None or not isinstance(region, str):
        return 1

    # region is a string (region name) - look it up
    party = get_active_party()
    if party is None:
        raise RuntimeError("No active Party context.")

    if not party.has_region(region):
        return 1

    registered_region = party.get_region(region)
    if isinstance(registered_region, OrfRegion):
        return registered_region.frame
    else:
        raise ValueError(
            f"Region '{region}' is a plain Region, not an OrfRegion. "
            f"frame must be specified explicitly, or use annotate_orf() to "
            f"upgrade the region to an OrfRegion with a frame."
        )


def _get_shared_styles(seq_style: SeqStyle, positions: list[int]) -> list[str]:
    """Return style specs applied to ALL given positions."""
    if seq_style is None:
        return []
    shared = []
    for style_spec, style_positions in seq_style.style_list:
        pos_set = set(style_positions.tolist())
        if all(p in pos_set for p in positions):
            shared.append(style_spec)
    return shared


@beartype
def translate(
    pool: Union[Pool, str],
    region: RegionType = None,
    *,
    frame: Optional[int] = None,
    include_stop: bool = True,
    preserve_codon_styles: bool = True,
    genetic_code: Union[str, dict] = "standard",
    iter_order: Optional[Real] = None,
    prefix: Optional[str] = None,
):
    """Translate DNA sequence to protein.

    Parameters
    ----------
    pool : Union[Pool, str]
        Parent pool or sequence string to translate.
    region : RegionType, default=None
        Region to translate. Can be region name or [start, stop].
        If None, translates the entire sequence.
    frame : Optional[int], default=None
        Reading frame: +1, +2, +3, -1, -2, -3.
        If None and region is an OrfRegion, uses its frame; otherwise +1.
    include_stop : bool, default=True
        Whether to include stop codon (*) in output.
    preserve_codon_styles : bool, default=True
        If True, propagate styles to amino acids when all 3 nucleotides
        of a codon share the same style.
    genetic_code : Union[str, dict], default="standard"
        Genetic code to use for translation.
    iter_order : Optional[float], default=None
        Iteration order priority for the Operation.
    prefix : Optional[str], default=None
        Prefix for sequence names in the resulting Pool.

    Returns
    -------
    ProteinPool
        A Pool containing translated protein sequences.

    Raises
    ------
    ValueError
        If frame is invalid, or if region is a named plain Region
        without an explicit frame.
    """
    from ..fixed_ops.from_seq import from_seq
    from ..protein_pool import ProteinPool

    pool = from_seq(pool) if isinstance(pool, str) else pool

    # Resolve frame
    resolved_frame = _resolve_frame(region, frame)

    op = TranslateOp(
        parent_pool=pool,
        region=region,
        frame=resolved_frame,
        include_stop=include_stop,
        preserve_codon_styles=preserve_codon_styles,
        genetic_code=genetic_code,
        iter_order=iter_order,
        prefix=prefix,
    )
    return ProteinPool(operation=op)


class TranslateOp(Operation):
    """Translate DNA to protein sequence."""

    factory_name = "translate"
    design_card_keys = []

    def __init__(
        self,
        parent_pool,
        region: RegionType = None,
        frame: int = 1,
        include_stop: bool = True,
        preserve_codon_styles: bool = True,
        genetic_code: Union[str, dict] = "standard",
        iter_order: Optional[Real] = None,
        prefix: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        self._frame = frame
        self._include_stop = include_stop
        self._preserve_codon_styles = preserve_codon_styles
        self._genetic_code = genetic_code
        self._translate_region = (
            region  # Store region separately (don't use base class region handling)
        )
        self.codon_table = CodonTable(genetic_code)

        # Calculate output sequence length if possible
        parent_seq_length = parent_pool.seq_length
        if parent_seq_length is not None and region is None and include_stop:
            frame_offset = abs(frame) - 1
            num_codons = (parent_seq_length - frame_offset) // 3
            out_length = num_codons
        else:
            out_length = None

        # Note: Don't pass region to parent - we handle region extraction ourselves
        # (we don't want the base class to reassemble DNA prefix/suffix around our protein)
        super().__init__(
            parent_pools=[parent_pool],
            num_states=1,
            mode="fixed",
            seq_length=out_length,
            name=name,
            iter_order=iter_order,
            prefix=prefix,
            region=None,  # Don't use base class region handling
        )

    def _get_copy_params(self) -> dict:
        params = super()._get_copy_params()
        params["region"] = self._translate_region
        return params

    def _compute_core(
        self,
        parents: list[Seq],
        rng: np.random.Generator | None = None,
    ) -> tuple[ProteinSeq, dict]:
        """Translate DNA sequence to protein."""
        parent_seq = parents[0]

        # Handle NullSeq
        if is_null_seq(parent_seq):
            return NullSeq(), {}

        # Extract region if specified
        if self._translate_region is not None:
            from ..utils.region_context import RegionContext

            ctx = RegionContext.from_sequence(parent_seq, self._translate_region, remove_tags=False)
            _, parent_seq, _ = ctx.split_parent_seq(parent_seq)

        # Get molecular positions (DNA only, no gaps/tags)
        mol_length = parent_seq.molecular_length
        if mol_length == 0:
            return ProteinSeq.empty(), {}

        frame = self._frame
        is_reverse = frame < 0

        # Calculate frame offset (frame +1 = skip 0, frame +2 = skip 1, frame +3 = skip 2)
        frame_offset = abs(frame) - 1

        # Validate no IUPAC ambiguity codes in region (codon table requires ACGT only)
        # Strip tags to get actual sequence content
        from ..utils.parsing_utils import strip_all_tags

        clean_content = strip_all_tags(parent_seq.string)
        iupac_ambiguity = set("RYSWKMBDHVNryswkmbdhvn")
        invalid_chars = set(clean_content) & iupac_ambiguity
        if invalid_chars:
            raise ValueError(
                f"translate() cannot handle IUPAC ambiguity codes: {sorted(invalid_chars)}. "
                "Region must contain only A, C, G, T."
            )

        # Check if we have enough bases for at least one codon
        if mol_length < frame_offset + 3:
            return ProteinSeq.empty(), {}

        # Number of complete codons
        num_codons = (mol_length - frame_offset) // 3

        aa_chars = []
        aa_styles: list[tuple[str, int]] = []  # (style_spec, aa_position)

        for codon_idx in range(num_codons):
            # Get molecular positions for this codon
            if is_reverse:
                # For reverse frames, read from end
                mol_start = mol_length - frame_offset - (codon_idx + 1) * 3
            else:
                mol_start = frame_offset + codon_idx * 3

            mol_positions = [mol_start, mol_start + 1, mol_start + 2]

            # Convert to literal positions
            lit_positions = [parent_seq.molecular_to_literal(p) for p in mol_positions]

            # Extract codon string
            codon = "".join(parent_seq.string[p] for p in lit_positions).upper()

            # Handle reverse frame: reverse-complement the codon
            if is_reverse:
                codon = reverse_complement(codon)

            # Translate codon
            aa = self.codon_table.codon_to_aa.get(codon, "?")

            # Skip stop codon if not including stops
            if aa == "*" and not self._include_stop:
                continue

            aa_chars.append(aa)
            aa_idx = len(aa_chars) - 1

            # Check for shared styles if preserve_codon_styles is enabled
            if self._preserve_codon_styles and parent_seq.style is not None:
                shared_styles = _get_shared_styles(parent_seq.style, lit_positions)
                for style_spec in shared_styles:
                    aa_styles.append((style_spec, aa_idx))

        # Build protein string
        protein_string = "".join(aa_chars)

        # Build protein style
        if aa_styles and len(protein_string) > 0:
            # Group by style spec
            style_dict: dict[str, list[int]] = {}
            for style_spec, pos in aa_styles:
                if style_spec not in style_dict:
                    style_dict[style_spec] = []
                style_dict[style_spec].append(pos)

            # Create SeqStyle
            protein_style = SeqStyle.empty(len(protein_string))
            for style_spec, positions in style_dict.items():
                protein_style = protein_style.add_style(style_spec, np.array(positions))
        else:
            protein_style = SeqStyle.empty(len(protein_string)) if protein_string else None

        return ProteinSeq.from_string(protein_string, protein_style), {}
