"""MutagenizeOrf operation - apply codon-level mutations to ORF sequences."""

from itertools import combinations
from math import comb
from numbers import Integral, Real

import numpy as np

from ..codon_table import UNIFORM_MUTATION_TYPES, VALID_MUTATION_TYPES
from ..dna_pool import DnaPool
from ..operation import Operation
from ..party import get_active_party
from ..pool import Pool
from ..region import VALID_FRAMES, OrfRegion
from ..types import CardsType, ModeType, Optional, RegionType, Seq, Sequence, Union, beartype
from ..utils.dna_seq import DnaSeq
from ..utils.dna_utils import reverse_complement
from ..utils.parsing_utils import find_all_regions


def _resolve_frame(region: RegionType, frame: Optional[int]) -> int:
    """Resolve the frame value, looking up from OrfRegion if needed.

    Backward compatibility: defaults to frame=1 when region is None or an interval.
    When region is a named OrfRegion, uses the stored frame.
    When region is a named plain Region, raises an error (must specify frame).
    """
    # If frame is explicitly provided, validate and use it
    if frame is not None:
        if frame not in VALID_FRAMES:
            raise ValueError(f"frame must be one of {sorted(VALID_FRAMES)}, got {frame}")
        return frame

    # frame is None - try to get from OrfRegion or use default
    if region is None or not isinstance(region, str):
        # Backward compatibility: default to frame=1 for non-named regions
        return 1

    # region is a string (region name) - look it up
    party = get_active_party()
    if party is None:
        raise RuntimeError("No active Party context.")

    if not party.has_region(region):
        # Region doesn't exist yet - use default frame=1
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


@beartype
def mutagenize_orf(
    pool: Union[Pool, str],
    region: RegionType = None,
    *,
    num_mutations: Optional[Integral] = None,
    mutation_rate: Optional[Real] = None,
    mutation_type: str = "missense_only_first",
    codon_positions: Union[Sequence[Integral], slice, None] = None,
    style: Optional[str] = None,
    frame: Optional[int] = None,
    prefix: Optional[str] = None,
    mode: ModeType = "random",
    num_states: Optional[Integral] = None,
    iter_order: Optional[Real] = None,
    cards: CardsType = None,
) -> Pool:
    """
    Apply codon-level mutations to an ORF sequence. Requires active Party context.

    Parameters
    ----------
    pool : Union[Pool, str]
        Parent pool or sequence string to mutate.
    region : RegionType, default=None
        Region to mutate. Can be marker name (e.g., "orf") or [start, stop].
        If None, mutates the entire sequence.
    num_mutations : Optional[Integral], default=None
        Fixed number of codon mutations (mutually exclusive with mutation_rate).
    mutation_rate : Optional[Real], default=None
        Per-codon mutation probability (mutually exclusive with num_mutations).
    mutation_type : str, default='missense_only_first'
        Type of mutation: 'any_codon', 'nonsynonymous_first', 'nonsynonymous_random',
        'missense_only_first', 'missense_only_random', 'synonymous', 'nonsense'.
    codon_positions : Union[Sequence[Integral], slice, None], default=None
        Eligible codon indices: None (all), list of indices, or slice.
    style : Optional[str], default=None
        Style to apply to mutated codon positions (e.g., 'red', 'bold').
    frame : Optional[int], default=None
        Reading frame and orientation. Valid values: +1, +2, +3, -1, -2, -3.
        Positive values indicate left-to-right orientation (5'->3'),
        negative values indicate right-to-left orientation (3'->5').
        The absolute value indicates the frame of the boundary base (1-indexed).
        If None and region is a named OrfRegion, uses the OrfRegion's frame.
    prefix : Optional[str], default=None
        Prefix for sequence names in the resulting Pool.
    mode : ModeType, default='random'
        Selection mode: 'random' or 'sequential'. Sequential requires ``num_mutations``
        (not ``mutation_rate``) and a uniform ``mutation_type`` ('any_codon',
        'nonsynonymous_first', 'missense_only_first', or 'nonsense').
    num_states : Optional[Integral], default=None
        Number of states. In sequential mode, overrides the computed count
        (cycling if greater, clipping if less). In random mode, if None
        defaults to 1 (pure random sampling).
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.
    cards : list[str] or dict, optional
        Design card keys to include. Available keys: ``'codon_positions'``,
        ``'wt_codons'``, ``'mut_codons'``, ``'wt_aas'``, ``'mut_aas'``.

    Returns
    -------
    Pool
        A Pool that generates codon-mutated sequences.

    Raises
    ------
    ValueError
        If frame is None and region is a named plain Region (not OrfRegion),
        if mutation_rate is used with sequential mode, if mutation_type is
        non-uniform with sequential mode, or if num_mutations exceeds
        eligible codons.
    """
    from ..fixed_ops.from_seq import from_seq

    pool = from_seq(pool) if isinstance(pool, str) else pool

    # Resolve frame (may look up from OrfRegion)
    resolved_frame = _resolve_frame(region, frame)

    op = MutagenizeOrfOp(
        parent_pool=pool,
        region=region,
        num_mutations=num_mutations,
        mutation_rate=mutation_rate,
        mutation_type=mutation_type,
        codon_positions=codon_positions,
        style=style,
        frame=resolved_frame,
        prefix=prefix,
        mode=mode,
        num_states=num_states,
        name=None,
        iter_order=iter_order,
        cards=cards,
    )
    return DnaPool(operation=op)


class MutagenizeOrfOp(Operation):
    """Apply codon-level mutations to an ORF sequence."""

    factory_name = "mutagenize_orf"
    design_card_keys = ["codon_positions", "wt_codons", "mut_codons", "wt_aas", "mut_aas"]

    def __init__(
        self,
        parent_pool: Pool,
        region: RegionType = None,
        num_mutations: Optional[Integral] = None,
        mutation_rate: Optional[Real] = None,
        mutation_type: str = "missense_only_first",
        codon_positions: Union[Sequence[Integral], slice, None] = None,
        style: Optional[str] = None,
        frame: int = 1,
        prefix: Optional[str] = None,
        mode: ModeType = "random",
        num_states: Optional[Integral] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
        cards: CardsType = None,
    ) -> None:
        """Initialize MutagenizeOrfOp."""
        party = get_active_party()
        if party is None:
            raise RuntimeError(
                "mutagenize_orf requires an active Party context. "
                "Use 'with pp.Party() as party:' to create one."
            )
        if frame not in VALID_FRAMES:
            raise ValueError(f"frame must be one of {sorted(VALID_FRAMES)}, got {frame}")
        if num_mutations is None and mutation_rate is None:
            raise ValueError("Either num_mutations or mutation_rate must be provided")
        if num_mutations is not None and mutation_rate is not None:
            raise ValueError("Only one of num_mutations or mutation_rate can be provided, not both")
        if num_mutations is not None and num_mutations < 1:
            raise ValueError(f"num_mutations must be >= 1, got {num_mutations}")
        if mutation_rate is not None:
            if mutation_rate < 0 or mutation_rate > 1:
                raise ValueError(f"mutation_rate must be between 0 and 1, got {mutation_rate}")
            if mode == "sequential":
                raise ValueError("mode='sequential' is not supported with mutation_rate")
        if mutation_type not in VALID_MUTATION_TYPES:
            raise ValueError(
                f"mutation_type must be one of {sorted(VALID_MUTATION_TYPES)}, got '{mutation_type}'"
            )
        if mode == "sequential" and mutation_type not in UNIFORM_MUTATION_TYPES:
            raise ValueError(
                f"mode='sequential' requires a uniform mutation type, got '{mutation_type}'"
            )

        self.num_mutations = num_mutations
        self.mutation_rate = mutation_rate
        self.mutation_type = mutation_type
        self._mode = mode
        self.codon_table = party.codon_table
        self._orf_region = region  # Store locally, will copy to _region after super().__init__
        self.style = style
        self.frame = frame
        self.reverse = frame < 0  # Derive reverse from frame sign
        # Calculate bases to skip to reach the first complete codon
        # frame=1: first base is position 1 in codon → skip 0 bases
        # frame=2: first base is position 2 in codon → skip 2 bases (partial has 2 bases)
        # frame=3: first base is position 3 in codon → skip 1 base (partial has 1 base)
        self.frame_offset = (4 - abs(frame)) % 3

        # Use effective seq_length (excluding tags)
        parent_seq_length = parent_pool.seq_length
        if parent_seq_length is None:
            raise ValueError("parent_pool must have a defined seq_length")

        # Validate and parse region - actual bounds determined at compute time for marker names
        self._validate_orf_region(region, parent_seq_length)
        self._seq_length = parent_seq_length

        # For interval regions, we can compute bounds now; for marker names, defer to compute time
        if region is None:
            self.orf_start = 0
            self.orf_end = parent_seq_length
        elif not isinstance(region, str):
            self.orf_start = int(region[0])
            self.orf_end = int(region[1])
        else:
            # Named region — resolve content length from registered region metadata
            region_obj = next(
                (r for r in parent_pool.regions if r.name == region), None
            )
            if region_obj is not None and region_obj.seq_length is not None:
                self.orf_start = 0
                self.orf_end = region_obj.seq_length
            elif mode == "sequential":
                raise ValueError(
                    f"Cannot determine geometry of named region '{region}' at "
                    f"init time for mode='sequential'. Either use from_seq() or "
                    f"annotate_orf() to create the parent (which register "
                    f"regions automatically), or use an interval region "
                    f"[start, stop]."
                )
            else:
                self.orf_start = 0
                self.orf_end = parent_seq_length

        # Calculate number of complete codons, accounting for frame offset
        # For positive frames: skip frame_offset bases at the start
        # For negative frames: skip frame_offset bases at the end
        orf_length = self.orf_end - self.orf_start
        effective_length = orf_length - self.frame_offset
        self.num_codons = effective_length // 3

        if codon_positions is None:
            self.eligible_positions = list(range(self.num_codons))
        elif isinstance(codon_positions, slice):
            start, stop, step = codon_positions.indices(self.num_codons)
            self.eligible_positions = list(range(start, stop, step))
        else:
            self.eligible_positions = list(codon_positions)
            for pos in self.eligible_positions:
                if pos < 0 or pos >= self.num_codons:
                    raise ValueError(
                        f"codon_positions value {pos} is out of range [0, {self.num_codons})"
                    )
            if len(self.eligible_positions) != len(set(self.eligible_positions)):
                raise ValueError("codon_positions must not contain duplicates")

        self.num_eligible = len(self.eligible_positions)
        if num_mutations is not None and num_mutations > self.num_eligible:
            raise ValueError(
                f"num_mutations ({num_mutations}) exceeds the number of "
                f"eligible codon positions ({self.num_eligible}). "
                f"The region spans {orf_length} bp with frame_offset="
                f"{self.frame_offset}, yielding {self.num_codons} complete "
                f"codon(s). Ensure the region covers at least "
                f"{num_mutations * 3 + self.frame_offset} bp."
            )

        self.uniform_num_alts = UNIFORM_MUTATION_TYPES.get(mutation_type)
        self._sequential_cache = None

        user_num_states = num_states
        natural_num_states = None

        match mode:
            case "sequential" if num_mutations is not None and self.uniform_num_alts is not None:
                natural_num_states = self._build_caches()
                num_states = user_num_states if user_num_states is not None else natural_num_states
            case "random":
                pass
            case _:
                num_states = 1

        super().__init__(
            parent_pools=[parent_pool],
            num_states=num_states,
            mode=mode,
            seq_length=self._seq_length,
            name=name,
            iter_order=iter_order,
            prefix=prefix,
            _natural_num_states=natural_num_states,
            cards=cards,
        )

    def _validate_orf_region(self, region: RegionType, seq_length: int) -> None:
        """Validate region parameter for ORF operations."""
        if region is None:
            return
        if isinstance(region, str):
            # Marker name - validated at compute time
            return
        # Interval [start, end]
        if len(region) != 2:
            raise ValueError(f"region must have exactly 2 elements, got {len(region)}")
        start, end = int(region[0]), int(region[1])
        if start < 0:
            raise ValueError(f"region start must be >= 0, got {start}")
        if end > seq_length:
            raise ValueError(f"region end ({end}) cannot exceed sequence length ({seq_length})")
        if start >= end:
            raise ValueError(f"region start ({start}) must be < end ({end})")

    def _get_molecular_region_bounds(self, seq_obj: Seq) -> tuple[int, int]:
        """Get the start/end positions of the region in molecular coordinates."""
        mol_length = seq_obj.molecular_length

        if self._orf_region is None:
            return (0, mol_length)

        # Handle [start, stop] interval - these are molecular coordinates
        if not isinstance(self._orf_region, str):
            return (int(self._orf_region[0]), int(self._orf_region[1]))

        # Handle region name - find the region and convert to molecular coordinates
        try:
            found_regions = find_all_regions(seq_obj.string)
        except ValueError:
            return (0, mol_length)

        for r in found_regions:
            if r.name == self._orf_region:
                # Convert content positions to molecular coordinates
                # content_start and content_end are literal positions
                mol_start = seq_obj.literal_to_molecular(r.content_start)
                mol_end_lit = r.content_end - 1  # Last char of content
                mol_end = seq_obj.literal_to_molecular(mol_end_lit)

                if mol_start is None or mol_end is None:
                    # Region contains non-molecular characters at boundaries
                    return (0, mol_length)

                return (mol_start, mol_end + 1)  # +1 to make it exclusive end

        # Region not found - use entire sequence
        return (0, mol_length)

    def _extract_codons_molecular(
        self, seq_obj: Seq, mol_start: int, mol_end: int, frame_offset: int
    ) -> list[str]:
        """Extract complete codons from a Seq object using molecular coordinates.

        Args:
            seq_obj: The Seq object.
            mol_start: Start position in molecular coordinates.
            mol_end: End position in molecular coordinates (exclusive).
            frame_offset: Number of bases to skip (0, 1, or 2).

        Returns:
            List of codon strings. For negative frames (reverse), codons are
            reverse-complemented so they can be looked up in the codon table.
        """
        codons = []
        orf_length = mol_end - mol_start
        effective_length = orf_length - frame_offset
        num_complete_codons = effective_length // 3

        if self.reverse:
            # For negative frames: skip frame_offset bases at the END
            # Read codons right-to-left, codon 0 is the rightmost complete codon
            codon_region_end = mol_end - frame_offset
            for codon_idx in range(num_complete_codons):
                codon_chars = []
                # Codon 0 starts at (codon_region_end - 3), codon 1 at (codon_region_end - 6), etc.
                codon_start = codon_region_end - (codon_idx + 1) * 3
                for j in range(3):
                    mol_pos = codon_start + j
                    lit_pos = seq_obj.molecular_to_literal(mol_pos)
                    codon_chars.append(seq_obj.string[lit_pos])
                # Reverse-complement for codon table lookup
                codon = "".join(codon_chars)
                codons.append(reverse_complement(codon))
        else:
            # For positive frames: skip frame_offset bases at the START
            # Read codons left-to-right
            codon_region_start = mol_start + frame_offset
            for codon_idx in range(num_complete_codons):
                codon_chars = []
                codon_start = codon_region_start + codon_idx * 3
                for j in range(3):
                    mol_pos = codon_start + j
                    lit_pos = seq_obj.molecular_to_literal(mol_pos)
                    codon_chars.append(seq_obj.string[lit_pos])
                codons.append("".join(codon_chars))

        return codons

    def _build_caches(self) -> int:
        """Build caches for sequential enumeration."""
        if self.num_mutations is None or self.uniform_num_alts is None:
            return 1
        num_combinations = comb(self.num_eligible, self.num_mutations)
        num_mut_patterns = self.uniform_num_alts**self.num_mutations
        cache = []
        for positions in combinations(self.eligible_positions, self.num_mutations):
            for mut_pattern in range(num_mut_patterns):
                mut_indices = []
                remaining = mut_pattern
                for _ in range(self.num_mutations):
                    mut_indices.append(remaining % self.uniform_num_alts)
                    remaining //= self.uniform_num_alts
                cache.append((positions, tuple(reversed(mut_indices))))
        self._sequential_cache = cache
        return num_combinations * num_mut_patterns

    def _random_mutation(
        self,
        codons: list[str],
        rng: np.random.Generator,
        eligible_positions: list[int],
    ) -> tuple[tuple, tuple, tuple, tuple, tuple]:
        """Generate random codon mutations."""
        num_eligible = len(eligible_positions)

        if self.num_mutations is not None:
            num_mut = self.num_mutations
        else:
            num_mut = rng.binomial(num_eligible, self.mutation_rate)
            if num_mut == 0:
                return tuple(), tuple(), tuple(), tuple(), tuple()

        if num_mut > num_eligible:
            num_mut = num_eligible
        pos_indices = rng.choice(num_eligible, size=num_mut, replace=False)
        positions = tuple(sorted(eligible_positions[i] for i in pos_indices))

        wt_codons, mut_codons, wt_aas, mut_aas = [], [], [], []
        for pos in positions:
            wt = codons[pos].upper()
            wt_codons.append(wt)
            wt_aas.append(self.codon_table.codon_to_aa.get(wt, "?"))
            alternatives = self.codon_table.get_mutations(wt, self.mutation_type)
            mut = alternatives[rng.integers(0, len(alternatives))] if alternatives else wt
            mut_codons.append(mut)
            mut_aas.append(self.codon_table.codon_to_aa.get(mut, "?"))

        return positions, tuple(wt_codons), tuple(mut_codons), tuple(wt_aas), tuple(mut_aas)

    def _compute_core(
        self,
        parents: list[Seq],
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[Seq, dict]:
        """Return mutated Seq and design card."""
        from ..utils.style_utils import styles_suppressed

        parent_seq = parents[0]

        # Get ORF bounds in molecular coordinates
        mol_start, mol_end = self._get_molecular_region_bounds(parent_seq)
        orf_length = mol_end - mol_start

        # For marker-based regions, recompute num_codons and eligible_positions
        # since we only know the actual bounds at compute time
        if isinstance(self._orf_region, str):
            effective_length = orf_length - self.frame_offset
            num_codons = effective_length // 3
            eligible_positions = list(range(num_codons))
        else:
            num_codons = self.num_codons
            eligible_positions = self.eligible_positions

        # Extract codons using molecular coordinates (with frame offset)
        codons = self._extract_codons_molecular(parent_seq, mol_start, mol_end, self.frame_offset)

        if self.mode == "random":
            if rng is None:
                raise RuntimeError(f"{self.mode.capitalize()} mode requires RNG")
            positions, wt_codons, mut_codons, wt_aas, mut_aas = self._random_mutation(
                codons, rng, eligible_positions
            )
        else:
            if self._sequential_cache is None:
                self._build_caches()
            # Use state 0 when inactive (state is None)
            state = self.state.value
            cache_idx = (0 if state is None else state) % len(self._sequential_cache)
            positions, mut_indices = self._sequential_cache[cache_idx]

            max_pos = max(positions) if positions else -1
            if max_pos >= len(codons):
                raise ValueError(
                    f"Sequential cache position {max_pos} exceeds actual codon "
                    f"count {len(codons)}. This indicates a geometry mismatch "
                    f"between init-time and compute-time ORF bounds for "
                    f"region '{self._orf_region}'."
                )

            wt_codons, mut_codons, wt_aas, mut_aas = [], [], [], []
            for pos, mut_idx in zip(positions, mut_indices):
                wt = codons[pos].upper()
                wt_codons.append(wt)
                wt_aas.append(self.codon_table.codon_to_aa.get(wt, "?"))
                alternatives = self.codon_table.get_mutations(wt, self.mutation_type)
                mut = alternatives[mut_idx] if alternatives else wt
                mut_codons.append(mut)
                mut_aas.append(self.codon_table.codon_to_aa.get(mut, "?"))
            wt_codons, mut_codons = tuple(wt_codons), tuple(mut_codons)
            wt_aas, mut_aas = tuple(wt_aas), tuple(mut_aas)

        # Apply mutations at molecular positions -> literal positions
        result_chars = list(parent_seq.string)
        for codon_pos, mut_codon in zip(positions, mut_codons):
            # Convert codon position to molecular position, accounting for frame_offset
            if self.reverse:
                # In reverse mode: skip frame_offset at the end, codon 0 is rightmost complete codon
                codon_region_end = mol_end - self.frame_offset
                mol_codon_start = codon_region_end - (codon_pos + 1) * 3
                # Reverse-complement the mutated codon before inserting
                insert_codon = reverse_complement(mut_codon)
            else:
                # In forward mode: skip frame_offset at the start
                codon_region_start = mol_start + self.frame_offset
                mol_codon_start = codon_region_start + codon_pos * 3
                insert_codon = mut_codon

            # Convert molecular positions to literal positions and apply mutation
            for i, mut_char in enumerate(insert_codon):
                mol_pos = mol_codon_start + i
                lit_pos = parent_seq.molecular_to_literal(mol_pos)
                result_chars[lit_pos] = mut_char

        result_string = "".join(result_chars)
        output_seq = DnaSeq(result_string, parent_seq.style)

        # Apply style to mutated positions if specified
        if self.style is not None and not styles_suppressed() and len(positions) > 0:
            # Convert mutated codon positions to literal positions for styling
            style_positions = []
            for codon_pos in positions:
                if self.reverse:
                    codon_region_end = mol_end - self.frame_offset
                    mol_codon_start = codon_region_end - (codon_pos + 1) * 3
                else:
                    codon_region_start = mol_start + self.frame_offset
                    mol_codon_start = codon_region_start + codon_pos * 3

                for i in range(3):
                    mol_pos = mol_codon_start + i
                    lit_pos = parent_seq.molecular_to_literal(mol_pos)
                    style_positions.append(lit_pos)

            output_seq = output_seq.add_style(self.style, np.array(style_positions, dtype=np.int64))

        return output_seq, {
            "codon_positions": positions,
            "wt_codons": wt_codons,
            "mut_codons": mut_codons,
            "wt_aas": wt_aas,
            "mut_aas": mut_aas,
        }

    def _get_copy_params(self) -> dict:
        """Return parameters needed to create a copy of this operation."""
        params = super()._get_copy_params()
        params["region"] = self._orf_region
        return params
