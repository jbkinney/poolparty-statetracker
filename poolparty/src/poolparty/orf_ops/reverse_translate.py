"""Reverse translate operation - convert protein sequences to DNA sequences."""

import numpy as np

from ..codon_table import CodonTable
from ..operation import Operation
from ..pool import Pool
from ..types import Integral, Literal, NullSeq, Optional, Real, RegionType, Seq, Union, beartype, is_null_seq
from ..utils.dna_seq import DnaSeq
from ..utils.protein_seq import ProteinSeq
from ..utils.style_utils import SeqStyle


class _FromProteinSeqOp(Operation):
    """Simple operation that yields a fixed protein sequence."""

    factory_name = "from_protein_seq"
    design_card_keys = []

    def __init__(self, protein_string: str) -> None:
        self._protein_string = protein_string
        super().__init__(
            parent_pools=[],
            num_states=1,
            mode="fixed",
            seq_length=len(protein_string),
            name=None,
            iter_order=None,
            prefix=None,
            region=None,
        )

    def _compute_core(
        self,
        parents: list[Seq],
        rng: np.random.Generator | None = None,
    ) -> tuple[ProteinSeq, dict]:
        """Return the fixed protein sequence."""
        return ProteinSeq.from_string(self._protein_string), {}


@beartype
def reverse_translate(
    pool: Union[Pool, str],
    region: RegionType = None,
    *,
    codon_selection: Literal["first", "random"] = "first",
    num_states: Optional[Integral] = None,
    genetic_code: Union[str, dict] = "standard",
    iter_order: Optional[Real] = None,
    prefix: Optional[str] = None,
):
    """Reverse translate protein sequence to DNA.

    Parameters
    ----------
    pool : Union[Pool, str]
        Parent pool or protein sequence string to reverse translate.
    region : RegionType, default=None
        Region to reverse translate. Can be region name or [start, stop].
        If None, reverse translates the entire sequence.
    codon_selection : Literal["first", "random"], default="first"
        How to select codons for each amino acid:
        - "first": Use the first codon in the codon table list, which is the
          most frequent when using the built-in standard genetic code (deterministic, mode="fixed")
        - "random": Randomly select from synonymous codons (stochastic, mode="random")
    num_states : Optional[int], default=None
        Number of states to generate. Only relevant when codon_selection="random".
        If None with "random", generates sequences on-the-fly.
    genetic_code : Union[str, dict], default="standard"
        Genetic code to use for reverse translation.
    iter_order : Optional[float], default=None
        Iteration order priority for the Operation.
    prefix : Optional[str], default=None
        Prefix for sequence names in the resulting Pool.

    Returns
    -------
    DnaPool
        A Pool containing reverse-translated DNA sequences.

    Raises
    ------
    TypeError
        If pool is not a ProteinPool (after string conversion).
    """
    from ..dna_pool import DnaPool
    from ..protein_pool import ProteinPool

    # Convert string to ProteinPool if needed
    if isinstance(pool, str):
        pool = ProteinPool(operation=_FromProteinSeqOp(pool))

    if not isinstance(pool, ProteinPool):
        raise TypeError(
            f"reverse_translate requires a ProteinPool input, got {type(pool).__name__}. "
            "DNA sequences cannot be reverse-translated."
        )

    # Determine mode based on codon_selection
    if codon_selection == "first":
        mode = "fixed"
        effective_num_states = 1
    else:
        mode = "random"
        # num_states stays as provided (or None for pure random mode)
        effective_num_states = num_states

    op = ReverseTranslateOp(
        parent_pool=pool,
        region=region,
        codon_selection=codon_selection,
        num_states=effective_num_states,
        mode=mode,
        genetic_code=genetic_code,
        iter_order=iter_order,
        prefix=prefix,
    )
    return DnaPool(operation=op)


class ReverseTranslateOp(Operation):
    """Reverse translate protein to DNA sequence."""

    factory_name = "reverse_translate"
    design_card_keys = []

    def __init__(
        self,
        parent_pool,
        region: RegionType = None,
        codon_selection: Literal["first", "random"] = "first",
        num_states: Optional[Integral] = None,
        mode: Literal["fixed", "random"] = "fixed",
        genetic_code: Union[str, dict] = "standard",
        iter_order: Optional[Real] = None,
        prefix: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        self._codon_selection = codon_selection
        self._genetic_code = genetic_code
        self._reverse_translate_region = region
        self.codon_table = CodonTable(genetic_code)

        # Calculate output sequence length if possible
        parent_seq_length = parent_pool.seq_length
        if parent_seq_length is not None and region is None:
            out_length = parent_seq_length * 3  # Each AA becomes 3 nucleotides
        else:
            out_length = None

        # Determine num_states based on mode
        if mode == "fixed":
            effective_num_states = 1
        else:
            effective_num_states = num_states

        super().__init__(
            parent_pools=[parent_pool],
            num_states=effective_num_states,
            mode=mode,
            seq_length=out_length,
            name=name,
            iter_order=iter_order,
            prefix=prefix,
            region=None,  # Don't use base class region handling
        )

    def _get_copy_params(self) -> dict:
        params = super()._get_copy_params()
        params["region"] = self._reverse_translate_region
        return params

    def _compute_core(
        self,
        parents: list[Seq],
        rng: np.random.Generator | None = None,
    ) -> tuple[DnaSeq, dict]:
        """Reverse translate protein sequence to DNA."""
        parent_seq = parents[0]

        # Handle NullSeq
        if is_null_seq(parent_seq):
            return NullSeq(), {}

        # Extract region if specified
        if self._reverse_translate_region is not None:
            from ..utils.region_context import RegionContext

            ctx = RegionContext.from_sequence(
                parent_seq, self._reverse_translate_region, remove_tags=False
            )
            _, parent_seq, _ = ctx.split_parent_seq(parent_seq)

        # Get molecular positions (amino acids only, no gaps/tags)
        mol_length = parent_seq.molecular_length
        if mol_length == 0:
            return DnaSeq.empty(), {}

        codons = []
        dna_styles: list[tuple[str, list[int]]] = []  # (style_spec, dna_positions)

        for aa_idx in range(mol_length):
            # Get literal position for this amino acid
            lit_pos = parent_seq.molecular_to_literal(aa_idx)
            aa = parent_seq.string[lit_pos].upper()

            # Look up codons for this amino acid
            if aa not in self.codon_table.aa_to_codons:
                raise ValueError(
                    f"Invalid amino acid '{aa}' at position {aa_idx}. "
                    f"Valid amino acids: {sorted(self.codon_table.aa_to_codons.keys())}"
                )

            codon_options = self.codon_table.aa_to_codons[aa]

            # Select codon based on selection strategy
            if self._codon_selection == "first":
                codon = codon_options[0]
            else:
                # random selection
                if rng is None:
                    rng = np.random.default_rng()
                codon = codon_options[rng.integers(0, len(codon_options))]

            codons.append(codon)

            # Propagate styles from amino acid to all 3 nucleotides
            if parent_seq.style is not None:
                dna_start = aa_idx * 3
                dna_positions = [dna_start, dna_start + 1, dna_start + 2]

                for style_spec, style_positions in parent_seq.style.style_list:
                    if lit_pos in style_positions.tolist():
                        dna_styles.append((style_spec, dna_positions))

        # Build DNA string
        dna_string = "".join(codons)

        # Build DNA style
        if dna_styles and len(dna_string) > 0:
            # Group by style spec
            style_dict: dict[str, list[int]] = {}
            for style_spec, positions in dna_styles:
                if style_spec not in style_dict:
                    style_dict[style_spec] = []
                style_dict[style_spec].extend(positions)

            # Create SeqStyle
            dna_style = SeqStyle.empty(len(dna_string))
            for style_spec, positions in style_dict.items():
                # Remove duplicates and sort
                unique_positions = sorted(set(positions))
                dna_style = dna_style.add_style(style_spec, np.array(unique_positions))
        else:
            dna_style = SeqStyle.empty(len(dna_string)) if dna_string else None

        return DnaSeq.from_string(dna_string, dna_style), {}
