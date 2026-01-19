"""MutagenizeOrf operation - apply codon-level mutations to ORF sequences."""
from itertools import combinations
from math import comb
from numbers import Real, Integral
import re
from ..types import Union, ModeType, Optional, Sequence, beartype
from ..operation import Operation
from ..pool import Pool
from ..party import get_active_party
from ..orf_utils import validate_orf_extent
from ..marker_ops.parsing import TAG_PATTERN, strip_all_markers, find_all_markers
import numpy as np
from ..codon_table import UNIFORM_MUTATION_TYPES, VALID_MUTATION_TYPES


@beartype
def mutagenize_orf(
    pool: Union[Pool, str],
    num_mutations: Optional[Integral] = None,
    mutation_rate: Optional[Real] = None,
    mutation_type: str = 'missense_only_first',
    orf_extent: Optional[Sequence[Integral]] = None,
    codon_positions: Union[Sequence[Integral], slice, None] = None,
    mode: ModeType = 'random',
    num_hybrid_states: Optional[Integral] = None,
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None,
) -> Pool:
    """
    Apply codon-level mutations to an ORF sequence. Requires active Party context.

    Parameters
    ----------
    pool : Union[Pool, str]
        Parent pool or sequence string to mutate.
    num_mutations : Optional[Integral], default=None
        Fixed number of codon mutations (mutually exclusive with mutation_rate).
    mutation_rate : Optional[Real], default=None
        Per-codon mutation probability (mutually exclusive with num_mutations).
    mutation_type : str, default='missense_only_first'
        Type of mutation: 'any_codon', 'nonsynonymous_first', 'nonsynonymous_random',
        'missense_only_first', 'missense_only_random', 'synonymous', 'nonsense'.
    orf_extent : Optional[Sequence[Integral]], default=None
        ORF boundaries as (start, end) or None for entire sequence.
    codon_positions : Union[Sequence[Integral], slice, None], default=None
        Eligible codon indices: None (all), list of indices, or slice.
    mode : ModeType, default='random'
        Selection mode: 'random', 'sequential', or 'hybrid'.
    num_hybrid_states : Optional[Integral], default=None
        Required when mode='hybrid'.
    name : Optional[str], default=None
        Name for the resulting Pool.
    op_name : Optional[str], default=None
        Name for the underlying Operation.
    iter_order : Optional[Real], default=None
        Iteration order for the Pool.
    op_iter_order : Optional[Real], default=None
        Iteration order for the Operation.

    Returns
    -------
    Pool
        A Pool that generates codon-mutated sequences.
    """
    from ..fixed_ops.from_seq import from_seq
    pool = from_seq(pool) if isinstance(pool, str) else pool
    op = MutagenizeOrfOp(
        parent_pool=pool,
        num_mutations=num_mutations,
        mutation_rate=mutation_rate,
        mutation_type=mutation_type,
        orf_extent=orf_extent,
        codon_positions=codon_positions,
        mode=mode,
        num_hybrid_states=num_hybrid_states,
        name=op_name,
        iter_order=op_iter_order,
    )
    return Pool(operation=op, name=name, iter_order=iter_order)


@beartype
class MutagenizeOrfOp(Operation):
    """Apply codon-level mutations to an ORF sequence."""
    factory_name = "mutagenize_orf"
    design_card_keys = ['codon_positions', 'wt_codons', 'mut_codons', 'wt_aas', 'mut_aas']
    
    def __init__(
        self,
        parent_pool: Pool,
        num_mutations: Optional[Integral] = None,
        mutation_rate: Optional[Real] = None,
        mutation_type: str = 'missense_only_first',
        orf_extent: Optional[Sequence[Integral]] = None,
        codon_positions: Union[Sequence[Integral], slice, None] = None,
        mode: ModeType = 'random',
        num_hybrid_states: Optional[Integral] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
    ) -> None:
        """Initialize MutagenizeOrfOp."""
        party = get_active_party()
        if party is None:
            raise RuntimeError(
                "mutagenize_orf requires an active Party context. "
                "Use 'with pp.Party() as party:' to create one."
            )
        if num_mutations is None and mutation_rate is None:
            raise ValueError("Either num_mutations or mutation_rate must be provided")
        if num_mutations is not None and mutation_rate is not None:
            raise ValueError("Only one of num_mutations or mutation_rate can be provided, not both")
        if num_mutations is not None and num_mutations < 1:
            raise ValueError(f"num_mutations must be >= 1, got {num_mutations}")
        if mutation_rate is not None:
            if mutation_rate < 0 or mutation_rate > 1:
                raise ValueError(f"mutation_rate must be between 0 and 1, got {mutation_rate}")
            if mode == 'sequential':
                raise ValueError("mode='sequential' is not supported with mutation_rate")
        if mutation_type not in VALID_MUTATION_TYPES:
            raise ValueError(f"mutation_type must be one of {sorted(VALID_MUTATION_TYPES)}, got '{mutation_type}'")
        if mode == 'sequential' and mutation_type not in UNIFORM_MUTATION_TYPES:
            raise ValueError(f"mode='sequential' requires a uniform mutation type, got '{mutation_type}'")
        if mode == 'hybrid' and num_hybrid_states is None:
            raise ValueError("num_hybrid_states is required when mode='hybrid'")
        
        self.num_mutations = num_mutations
        self.mutation_rate = mutation_rate
        self.mutation_type = mutation_type
        self._mode = mode
        self.codon_table = party.codon_table
        
        # Use effective seq_length (excluding markers)
        parent_seq_length = parent_pool.seq_length
        if parent_seq_length is None:
            raise ValueError("parent_pool must have a defined seq_length")
        
        # orf_extent uses logical positions (in marker-free sequence)
        self.orf_start, self.orf_end, self.num_codons = validate_orf_extent(
            orf_extent, parent_seq_length
        )
        self._seq_length = parent_seq_length
        
        if codon_positions is None:
            self.eligible_positions = list(range(self.num_codons))
        elif isinstance(codon_positions, slice):
            start, stop, step = codon_positions.indices(self.num_codons)
            self.eligible_positions = list(range(start, stop, step))
        else:
            self.eligible_positions = list(codon_positions)
            for pos in self.eligible_positions:
                if pos < 0 or pos >= self.num_codons:
                    raise ValueError(f"codon_positions value {pos} is out of range [0, {self.num_codons})")
            if len(self.eligible_positions) != len(set(self.eligible_positions)):
                raise ValueError("codon_positions must not contain duplicates")
        
        self.num_eligible = len(self.eligible_positions)
        if num_mutations is not None and num_mutations > self.num_eligible:
            raise ValueError(f"num_mutations ({num_mutations}) exceeds eligible positions ({self.num_eligible})")
        
        self.uniform_num_alts = UNIFORM_MUTATION_TYPES.get(mutation_type)
        self._sequential_cache = None
        
        match mode:
            case 'sequential' if num_mutations is not None and self.uniform_num_alts is not None:
                num_states = self._build_caches()
            case 'hybrid':
                num_states = num_hybrid_states
            case _:
                num_states = 1
        
        super().__init__(
            parent_pools=[parent_pool],
            num_values=num_states,
            mode=mode,
            seq_length=self._seq_length,
            name=name,
            iter_order=iter_order,
        )
    
    def _build_caches(self) -> int:
        """Build caches for sequential enumeration."""
        if self.num_mutations is None or self.uniform_num_alts is None:
            return 1
        num_combinations = comb(self.num_eligible, self.num_mutations)
        num_mut_patterns = self.uniform_num_alts ** self.num_mutations
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
    
    def _strip_markers(self, seq: str) -> tuple[str, list[tuple[int, int, str, str]]]:
        """Strip markers from sequence and record their positions.
        
        Returns:
            (clean_seq, markers_info) where markers_info is a list of
            (clean_content_start, content_length, opening_tag, closing_tag) tuples.
            For self-closing markers, content_length is 0 and closing_tag is empty.
        """
        markers_info = []
        found_markers = find_all_markers(seq)
        
        # Calculate clean position for each marker
        tag_offset = 0  # Cumulative length of tags removed before this marker
        
        for marker in found_markers:
            # In clean sequence, content starts at marker.content_start - tag_offset
            clean_content_start = marker.content_start - tag_offset
            content_length = marker.content_end - marker.content_start
            
            # Extract opening and closing tags
            opening_tag = seq[marker.start:marker.content_start]
            closing_tag = seq[marker.content_end:marker.end]
            
            markers_info.append((clean_content_start, content_length, opening_tag, closing_tag))
            
            # Update offset: tags removed = opening_tag_len + closing_tag_len
            tag_offset += len(opening_tag) + len(closing_tag)
        
        # Remove all marker tags but keep content
        clean_seq = strip_all_markers(seq)
        return clean_seq, markers_info
    
    def _restore_markers(self, seq: str, markers_info: list[tuple[int, int, str, str]]) -> str:
        """Restore markers to their original positions in the sequence.
        
        Args:
            seq: The clean (mutated) sequence with marker content but no tags.
            markers_info: List of (clean_content_start, content_length, opening_tag, closing_tag).
        
        Returns:
            Sequence with marker tags restored around their content.
        """
        if not markers_info:
            return seq
        
        # Sort by clean position (should already be sorted, but ensure it)
        sorted_markers = sorted(markers_info, key=lambda x: x[0])
        
        # Build result by inserting tags, working from start to end
        result = seq
        offset = 0  # Track how much we've added
        
        for clean_content_start, content_length, opening_tag, closing_tag in sorted_markers:
            # Insert opening tag before content
            insert_pos = clean_content_start + offset
            result = result[:insert_pos] + opening_tag + result[insert_pos:]
            offset += len(opening_tag)
            
            # Insert closing tag after content (if not self-closing)
            if closing_tag:
                close_pos = insert_pos + len(opening_tag) + content_length
                result = result[:close_pos] + closing_tag + result[close_pos:]
                offset += len(closing_tag)
        
        return result
    
    def _extract_codons(self, seq: str) -> tuple[str, list[str], str]:
        """Extract (upstream, codons, downstream) from marker-free sequence.
        
        Uses logical positions (orf_start, orf_end) in the marker-free sequence.
        """
        upstream = seq[:self.orf_start]
        orf_seq = seq[self.orf_start:self.orf_end]
        downstream = seq[self.orf_end:]
        codons = [orf_seq[i:i+3] for i in range(0, len(orf_seq), 3)]
        return upstream, codons, downstream
    
    def _random_mutation(
        self,
        codons: list[str],
        rng: np.random.Generator,
    ) -> tuple[tuple, tuple, tuple, tuple, tuple]:
        """Generate random codon mutations."""
        if self.num_mutations is not None:
            num_mut = self.num_mutations
        else:
            num_mut = rng.binomial(self.num_eligible, self.mutation_rate)
            if num_mut == 0:
                return tuple(), tuple(), tuple(), tuple(), tuple()
        
        if num_mut > self.num_eligible:
            num_mut = self.num_eligible
        pos_indices = rng.choice(self.num_eligible, size=num_mut, replace=False)
        positions = tuple(sorted(self.eligible_positions[i] for i in pos_indices))
        
        wt_codons, mut_codons, wt_aas, mut_aas = [], [], [], []
        for pos in positions:
            wt = codons[pos].upper()
            wt_codons.append(wt)
            wt_aas.append(self.codon_table.codon_to_aa.get(wt, '?'))
            alternatives = self.codon_table.get_mutations(wt, self.mutation_type)
            mut = alternatives[rng.integers(0, len(alternatives))] if alternatives else wt
            mut_codons.append(mut)
            mut_aas.append(self.codon_table.codon_to_aa.get(mut, '?'))
        
        return positions, tuple(wt_codons), tuple(mut_codons), tuple(wt_aas), tuple(mut_aas)
    
    def compute_design_card(
        self,
        parent_seqs: list[str],
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        """Return design card with codon mutation details."""
        seq = parent_seqs[0]
        # Strip markers before extracting codons
        clean_seq, _ = self._strip_markers(seq)
        _, codons, _ = self._extract_codons(clean_seq)
        
        if self.mode in ('random', 'hybrid'):
            if rng is None:
                raise RuntimeError(f"{self.mode.capitalize()} mode requires RNG")
            positions, wt_codons, mut_codons, wt_aas, mut_aas = self._random_mutation(codons, rng)
        else:
            if self._sequential_cache is None:
                self._build_caches()
            state = self.state.value
            cache_idx = (0 if state is None else state) % len(self._sequential_cache)
            positions, mut_indices = self._sequential_cache[cache_idx]
            
            wt_codons, mut_codons, wt_aas, mut_aas = [], [], [], []
            for pos, mut_idx in zip(positions, mut_indices):
                wt = codons[pos].upper()
                wt_codons.append(wt)
                wt_aas.append(self.codon_table.codon_to_aa.get(wt, '?'))
                alternatives = self.codon_table.get_mutations(wt, self.mutation_type)
                mut = alternatives[mut_idx] if alternatives else wt
                mut_codons.append(mut)
                mut_aas.append(self.codon_table.codon_to_aa.get(mut, '?'))
            wt_codons, mut_codons = tuple(wt_codons), tuple(mut_codons)
            wt_aas, mut_aas = tuple(wt_aas), tuple(mut_aas)
        
        return {
            'codon_positions': positions,
            'wt_codons': wt_codons,
            'mut_codons': mut_codons,
            'wt_aas': wt_aas,
            'mut_aas': mut_aas,
        }
    
    def compute_seq_from_card(
        self,
        parent_seqs: list[str],
        card: dict,
    ) -> dict:
        """Apply mutations from design card to produce output sequence.
        
        Markers are stripped before codon processing and restored afterward.
        """
        seq = parent_seqs[0]
        # Strip markers and record their positions
        clean_seq, markers = self._strip_markers(seq)
        
        # Extract and mutate codons on clean sequence
        upstream, codons, downstream = self._extract_codons(clean_seq)
        mutated_codons = codons.copy()
        for pos, mut in zip(card['codon_positions'], card['mut_codons']):
            mutated_codons[pos] = mut
        mutated_clean_seq = upstream + ''.join(mutated_codons) + downstream
        
        # Restore markers at original positions
        result_seq = self._restore_markers(mutated_clean_seq, markers)
        return {'seq_0': result_seq}
    
    def _get_copy_params(self) -> dict:
        """Return parameters needed to create a copy of this operation."""
        return {
            'parent_pool': self.parent_pools[0],
            'num_mutations': self.num_mutations,
            'mutation_rate': self.mutation_rate,
            'mutation_type': self.mutation_type,
            'orf_extent': (self.orf_start, self.orf_end),
            'codon_positions': self.eligible_positions,
            'mode': self.mode,
            'num_hybrid_states': self.num_values if self.mode == 'hybrid' else None,
            'name': None,
            'iter_order': self.iter_order,
        }
