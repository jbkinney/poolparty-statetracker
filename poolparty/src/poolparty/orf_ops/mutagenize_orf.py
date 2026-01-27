"""MutagenizeOrf operation - apply codon-level mutations to ORF sequences."""
from itertools import combinations
from math import comb
from numbers import Real, Integral
import re
from ..types import Union, ModeType, Optional, Sequence, beartype, Seq
from ..operation import Operation
from ..pool import Pool
from ..party import get_active_party
from ..utils.orf_utils import validate_orf_extent
from ..utils.parsing_utils import TAG_PATTERN, strip_all_tags, find_all_regions
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
    num_states: Optional[Integral] = None,
    iter_order: Optional[Real] = None,
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
        Selection mode: 'random' or 'sequential'.
    num_states : Optional[Integral], default=None
        Number of states for random mode. If None, defaults to 1 (pure random sampling).
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.

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
        num_states=num_states,
        name=None,
        iter_order=iter_order,
    )
    return Pool(operation=op)


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
        num_states: Optional[Integral] = None,
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
        
        self.num_mutations = num_mutations
        self.mutation_rate = mutation_rate
        self.mutation_type = mutation_type
        self._mode = mode
        self.codon_table = party.codon_table
        
        # Use effective seq_length (excluding tags)
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
            case 'random':
                # num_states stays None for pure random mode
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
    
    def _strip_tags(self, seq: str) -> tuple[str, list[tuple[int, int, str, str]]]:
        """Strip tags from sequence and record their positions.
        
        Returns:
            (clean_seq, tags_info) where tags_info is a list of
            (clean_content_start, content_length, opening_tag, closing_tag) tuples.
            For self-closing tags, content_length is 0 and closing_tag is empty.
        """
        tags_info = []
        found_regions = find_all_regions(seq)
        
        # Calculate clean position for each marker
        tag_offset = 0  # Cumulative length of tags removed before this marker
        
        for region in found_regions:
            # In clean sequence, content starts at region.content_start - tag_offset
            clean_content_start = region.content_start - tag_offset
            content_length = region.content_end - region.content_start
            
            # Extract opening and closing tags
            opening_tag = seq[region.start:region.content_start]
            closing_tag = seq[region.content_end:region.end]
            
            tags_info.append((clean_content_start, content_length, opening_tag, closing_tag))
            
            # Update offset: tags removed = opening_tag_len + closing_tag_len
            tag_offset += len(opening_tag) + len(closing_tag)
        
        # Remove all tags but keep content
        clean_seq = strip_all_tags(seq)
        return clean_seq, tags_info
    
    def _restore_tags(self, seq: str, tags_info: list[tuple[int, int, str, str]]) -> str:
        """Restore tags to their original positions in the sequence.
        
        Args:
            seq: The clean (mutated) sequence with region content but no tags.
            tags_info: List of (clean_content_start, content_length, opening_tag, closing_tag).
        
        Returns:
            Sequence with tags restored around their content.
        """
        if not tags_info:
            return seq
        
        # Sort by clean position (should already be sorted, but ensure it)
        sorted_tags = sorted(tags_info, key=lambda x: x[0])
        
        # Build result by inserting tags, working from start to end
        result = seq
        offset = 0  # Track how much we've added
        
        for clean_content_start, content_length, opening_tag, closing_tag in sorted_tags:
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
    
    def compute(
        self,
        parents: list[Seq],
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[Seq, dict]:
        """Return mutated Seq and design card."""
        seq = parents[0].string
        # Strip tags and record their positions
        clean_seq, tags = self._strip_tags(seq)
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
        
        # Apply mutations to codons
        upstream, codons, downstream = self._extract_codons(clean_seq)
        mutated_codons = codons.copy()
        for pos, mut in zip(positions, mut_codons):
            mutated_codons[pos] = mut
        mutated_clean_seq = upstream + ''.join(mutated_codons) + downstream
        
        # Restore tags at original positions
        result_seq = self._restore_tags(mutated_clean_seq, tags)
        
        # Pass through parent styles (mutagenize_orf preserves sequence length)
        output_style = parents[0].style
        
        # Compute name
        name = self._default_name(parents)
        
        output_seq = Seq(result_seq, output_style, name)
        
        return output_seq, {
            'codon_positions': positions,
            'wt_codons': wt_codons,
            'mut_codons': mut_codons,
            'wt_aas': wt_aas,
            'mut_aas': mut_aas,
        }
    
    def _get_copy_params(self) -> dict:
        """Return parameters needed to create a copy of this operation."""
        params = super()._get_copy_params()
        # Build tuple from two separate attributes
        params['orf_extent'] = (self.orf_start, self.orf_end)
        return params
