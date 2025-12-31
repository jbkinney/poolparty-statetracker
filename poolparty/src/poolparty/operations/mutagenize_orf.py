"""MutagenizeOrf operation - apply codon-level mutations to ORF sequences."""
from itertools import combinations
from math import comb
from numbers import Real, Integral
from ..types import Union, ModeType, Optional, Sequence, beartype
from ..operation import Operation
from ..pool import Pool
from ..party import get_active_party
from ..codon_table import CodonTable
import numpy as np
from ..codon_table import UNIFORM_MUTATION_TYPES, VALID_MUTATION_TYPES


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
        orf_start: Integral = 0,
        orf_end: Optional[Integral] = None,
        codon_positions: Optional[Sequence[Integral]] = None,
        codon_start: Optional[Integral] = None,
        codon_end: Optional[Integral] = None,
        codon_step_size: Integral = 1,
        mode: ModeType = 'random',
        num_hybrid_states: Optional[Integral] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
    ) -> None:
        """Initialize MutagenizeOrfOp.
        
        Args:
            parent_pool: The parent pool to mutate.
            num_mutations: Fixed number of codon mutations to apply (mutually exclusive with mutation_rate).
            mutation_rate: Probability of mutation at each codon position (mutually exclusive with num_mutations).
            mutation_type: Type of codon mutation to apply.
            orf_start: Start index of ORF within sequence (0-based, inclusive).
            orf_end: End index of ORF within sequence (0-based, exclusive). Default: len(seq).
            codon_positions: Explicit codon indices eligible for mutation (overrides start/end/step_size).
            codon_start: Starting codon index for eligible range.
            codon_end: Ending codon index for eligible range.
            codon_step_size: Step between eligible codon positions.
            mode: 'random', 'sequential', or 'hybrid'. Sequential only for uniform mutation types.
            num_hybrid_states: Required when mode='hybrid'.
            name: Optional name for the operation.
            iter_order: Optional iteration order.
        
        Raises:
            RuntimeError: If called outside of a Party context.
        """
        # Get codon table from active Party context
        party = get_active_party()
        if party is None:
            raise RuntimeError(
                "mutagenize_orf requires an active Party context. "
                "Use 'with pp.Party() as party:' to create one."
            )
        # Validate mutually exclusive parameters
        if num_mutations is None and mutation_rate is None:
            raise ValueError("Either num_mutations or mutation_rate must be provided")
        if num_mutations is not None and mutation_rate is not None:
            raise ValueError("Only one of num_mutations or mutation_rate can be provided, not both")
        
        # Validate num_mutations
        if num_mutations is not None and num_mutations < 1:
            raise ValueError(f"num_mutations must be >= 1, got {num_mutations}")
        
        # Validate mutation_rate
        if mutation_rate is not None:
            if mutation_rate < 0 or mutation_rate > 1:
                raise ValueError(f"mutation_rate must be between 0 and 1, got {mutation_rate}")
            if mode == 'sequential':
                raise ValueError("mode='sequential' is not supported with mutation_rate (use num_mutations instead)")
        
        # Validate mutation_type
        if mutation_type not in VALID_MUTATION_TYPES:
            raise ValueError(
                f"mutation_type must be one of {sorted(VALID_MUTATION_TYPES)}, got '{mutation_type}'"
            )
        
        # Validate sequential mode with non-uniform mutation types
        if mode == 'sequential' and mutation_type not in UNIFORM_MUTATION_TYPES:
            raise ValueError(
                f"mode='sequential' requires a uniform mutation type "
                f"({sorted(UNIFORM_MUTATION_TYPES.keys())}), got '{mutation_type}'"
            )
        
        # Validate hybrid mode
        if mode == 'hybrid' and num_hybrid_states is None:
            raise ValueError("num_hybrid_states is required when mode='hybrid'")
        
        self.num_mutations = num_mutations
        self.mutation_rate = mutation_rate
        self.mutation_type = mutation_type
        self._mode = mode
        
        # Get codon table from Party context
        self.codon_table = party.codon_table
        
        # Store ORF boundary parameters
        self.orf_start = int(orf_start)
        self.orf_end = int(orf_end) if orf_end is not None else None
        
        # Get parent sequence length to validate ORF boundaries
        parent_seq_length = parent_pool.seq_length
        if parent_seq_length is None:
            raise ValueError("parent_pool must have a defined seq_length")
        
        # Default orf_end to sequence length
        if self.orf_end is None:
            self.orf_end = parent_seq_length
        
        # Validate ORF boundaries
        if self.orf_start < 0:
            raise ValueError(f"orf_start must be >= 0, got {self.orf_start}")
        if self.orf_end > parent_seq_length:
            raise ValueError(
                f"orf_end ({self.orf_end}) cannot exceed sequence length ({parent_seq_length})"
            )
        if self.orf_start >= self.orf_end:
            raise ValueError(
                f"orf_start ({self.orf_start}) must be < orf_end ({self.orf_end})"
            )
        
        # Validate ORF length is divisible by 3
        orf_length = self.orf_end - self.orf_start
        if orf_length % 3 != 0:
            raise ValueError(
                f"ORF region length must be divisible by 3, got {orf_length} "
                f"(orf_start={self.orf_start}, orf_end={self.orf_end})"
            )
        
        self.num_codons = orf_length // 3
        self._seq_length = parent_seq_length  # Output length same as input
        
        # Compute eligible codon positions
        if codon_positions is not None:
            # Explicit positions provided
            self.eligible_positions = list(codon_positions)
            # Validate positions
            for pos in self.eligible_positions:
                if pos < 0 or pos >= self.num_codons:
                    raise ValueError(
                        f"codon_positions value {pos} is out of range [0, {self.num_codons})"
                    )
            # Check for duplicates
            if len(self.eligible_positions) != len(set(self.eligible_positions)):
                raise ValueError("codon_positions must not contain duplicates")
        else:
            # Use codon_start/codon_end/codon_step_size
            c_start = int(codon_start) if codon_start is not None else 0
            c_end = int(codon_end) if codon_end is not None else self.num_codons
            c_step = int(codon_step_size)
            
            # Validate range
            if c_start < 0:
                raise ValueError(f"codon_start must be >= 0, got {c_start}")
            if c_end > self.num_codons:
                raise ValueError(
                    f"codon_end ({c_end}) cannot exceed number of codons ({self.num_codons})"
                )
            if c_start >= c_end:
                raise ValueError(f"codon_start ({c_start}) must be < codon_end ({c_end})")
            if c_step < 1:
                raise ValueError(f"codon_step_size must be >= 1, got {c_step}")
            
            self.eligible_positions = list(range(c_start, c_end, c_step))
        
        self.num_eligible = len(self.eligible_positions)
        
        # Validate num_mutations against eligible positions
        if num_mutations is not None and num_mutations > self.num_eligible:
            raise ValueError(
                f"num_mutations ({num_mutations}) exceeds number of eligible positions ({self.num_eligible})"
            )
        
        # For uniform mutation types, get the number of alternatives
        if mutation_type in UNIFORM_MUTATION_TYPES:
            self.uniform_num_alts = UNIFORM_MUTATION_TYPES[mutation_type]
        else:
            self.uniform_num_alts = None
        
        # Build caches for sequential mode
        self._sequential_cache = None
        
        # Determine num_states based on mode
        if mode == 'sequential':
            if num_mutations is not None and self.uniform_num_alts is not None:
                num_states = self._build_caches()
            else:
                num_states = 1
        elif mode == 'hybrid':
            num_states = num_hybrid_states
        else:
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
        total_states = num_combinations * num_mut_patterns
        
        # Build cache of (positions, mutation_indices) for each state
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
        return total_states
    
    def _extract_codons(self, seq: str) -> tuple[str, list[str], str]:
        """Extract ORF codons and flanking regions from sequence.
        
        Returns:
            Tuple of (upstream_flank, list of codons, downstream_flank)
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
        """Generate random codon mutations.
        
        Returns:
            Tuple of (positions, wt_codons, mut_codons, wt_aas, mut_aas)
        """
        # Determine number of mutations
        if self.num_mutations is not None:
            num_mut = self.num_mutations
        else:
            # Use binomial distribution based on mutation_rate
            num_mut = rng.binomial(self.num_eligible, self.mutation_rate)
            if num_mut == 0:
                return tuple(), tuple(), tuple(), tuple(), tuple()
        
        # Choose random positions from eligible positions
        if num_mut > self.num_eligible:
            num_mut = self.num_eligible
        pos_indices = rng.choice(self.num_eligible, size=num_mut, replace=False)
        positions = tuple(sorted(self.eligible_positions[i] for i in pos_indices))
        
        wt_codons = []
        mut_codons = []
        wt_aas = []
        mut_aas = []
        
        for pos in positions:
            wt = codons[pos].upper()
            wt_codons.append(wt)
            wt_aas.append(self.codon_table.codon_to_aa.get(wt, '?'))
            
            # Get available mutations
            alternatives = self.codon_table.get_mutations(wt, self.mutation_type)
            if alternatives:
                mut_idx = rng.integers(0, len(alternatives))
                mut = alternatives[mut_idx]
            else:
                mut = wt  # No mutation available
            mut_codons.append(mut)
            mut_aas.append(self.codon_table.codon_to_aa.get(mut, '?'))
        
        return positions, tuple(wt_codons), tuple(mut_codons), tuple(wt_aas), tuple(mut_aas)
    
    def compute_design_card(
        self,
        parent_seqs: list[str],
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        """Return design card with codon mutation positions and characters."""
        seq = parent_seqs[0]
        upstream, codons, downstream = self._extract_codons(seq)
        
        if self.mode in ('random', 'hybrid'):
            if rng is None:
                raise RuntimeError(f"{self.mode.capitalize()} mode requires RNG - use Party.generate(seed=...)")
            positions, wt_codons, mut_codons, wt_aas, mut_aas = self._random_mutation(codons, rng)
        else:
            # Sequential mode
            if self._sequential_cache is None:
                self._build_caches()
            
            state = self.counter.state
            state = 0 if state is None else state
            cache_idx = state % len(self._sequential_cache)
            positions, mut_indices = self._sequential_cache[cache_idx]
            
            wt_codons = []
            mut_codons = []
            wt_aas = []
            mut_aas = []
            
            for pos, mut_idx in zip(positions, mut_indices):
                wt = codons[pos].upper()
                wt_codons.append(wt)
                wt_aas.append(self.codon_table.codon_to_aa.get(wt, '?'))
                
                alternatives = self.codon_table.get_mutations(wt, self.mutation_type)
                mut = alternatives[mut_idx] if alternatives else wt
                mut_codons.append(mut)
                mut_aas.append(self.codon_table.codon_to_aa.get(mut, '?'))
            
            wt_codons = tuple(wt_codons)
            mut_codons = tuple(mut_codons)
            wt_aas = tuple(wt_aas)
            mut_aas = tuple(mut_aas)
        
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
        """Apply codon mutations to the parent sequence based on design card."""
        seq = parent_seqs[0]
        upstream, codons, downstream = self._extract_codons(seq)
        
        positions = card['codon_positions']
        mut_codons = card['mut_codons']
        
        # Apply mutations
        mutated_codons = codons.copy()
        for pos, mut in zip(positions, mut_codons):
            mutated_codons[pos] = mut
        
        # Reassemble sequence
        mutated_orf = ''.join(mutated_codons)
        result_seq = upstream + mutated_orf + downstream
        
        return {'seq_0': result_seq}
    
    def _get_copy_params(self) -> dict:
        """Return parameters needed to create a copy of this operation."""
        return {
            'parent_pool': self.parent_pools[0],
            'num_mutations': self.num_mutations,
            'mutation_rate': self.mutation_rate,
            'mutation_type': self.mutation_type,
            'orf_start': self.orf_start,
            'orf_end': self.orf_end,
            'codon_positions': self.eligible_positions if hasattr(self, 'eligible_positions') else None,
            'mode': self.mode,
            'num_hybrid_states': self.num_states if self.mode == 'hybrid' else None,
            'name': None,
            'iter_order': self.iter_order,
        }


@beartype
def mutagenize_orf(
    pool: Union[Pool, str],
    num_mutations: Optional[Integral] = None,
    mutation_rate: Optional[Real] = None,
    mutation_type: str = 'missense_only_first',
    orf_start: Integral = 0,
    orf_end: Optional[Integral] = None,
    codon_positions: Optional[Sequence[Integral]] = None,
    codon_start: Optional[Integral] = None,
    codon_end: Optional[Integral] = None,
    codon_step_size: Integral = 1,
    mode: ModeType = 'random',
    num_hybrid_states: Optional[Integral] = None,
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None,
) -> Pool:
    """Create a Pool that applies codon-level mutations to an ORF sequence.
    
    Must be called within a Party context. The genetic code is set via the Party
    constructor or Party.set_genetic_code() method.
    
    Supports two mutation modes (exactly one must be specified):
    - num_mutations: Apply exactly this many codon mutations to each sequence.
      Supports 'random', 'sequential' (for uniform types), and 'hybrid' modes.
    - mutation_rate: Apply a random number of mutations based on a binomial distribution.
      Only supports 'random' and 'hybrid' modes.
    
    Args:
        pool: Parent pool or sequence string to mutate.
        num_mutations: Fixed number of codon mutations to apply (mutually exclusive with mutation_rate).
        mutation_rate: Probability of mutation at each codon position (mutually exclusive with num_mutations).
        mutation_type: Type of codon mutation. One of:
            - 'any_codon': Any of the 63 other codons (uniform)
            - 'nonsynonymous_first': Different AA/stop, first codon (uniform)
            - 'nonsynonymous_random': Different AA/stop, random codon (non-uniform)
            - 'missense_only_first': Different AA, first codon, NO stop (uniform) [default]
            - 'missense_only_random': Different AA, random codon, NO stop (non-uniform)
            - 'synonymous': Synonymous codons only (non-uniform)
            - 'nonsense': Stop codons only (uniform)
        orf_start: Start index of ORF within sequence (0-based, inclusive). Default: 0.
        orf_end: End index of ORF within sequence (0-based, exclusive). Default: len(seq).
        codon_positions: Explicit list of codon indices eligible for mutation (overrides start/end/step_size).
        codon_start: Starting codon index for eligible range. Default: 0.
        codon_end: Ending codon index for eligible range. Default: num_codons.
        codon_step_size: Step between eligible codon positions. Default: 1.
        mode: 'random', 'sequential', or 'hybrid'. Sequential only for uniform mutation types.
        num_hybrid_states: Required when mode='hybrid'.
        name: Optional name for the output pool.
        op_name: Optional name for the operation.
        iter_order: Optional iteration order for the pool.
        op_iter_order: Optional iteration order for the operation.
    
    Returns:
        A Pool that generates codon-mutated sequences.
    
    Raises:
        RuntimeError: If called outside of a Party context.
    
    Examples:
        # Apply exactly 2 missense mutations
        >>> with pp.Party() as party:
        ...     mutants = mutagenize_orf('ATGAAATTT', num_mutations=2)
        
        # Apply mutations with 10% rate per codon
        >>> with pp.Party() as party:
        ...     mutants = mutagenize_orf('ATGAAATTT', mutation_rate=0.1)
        
        # Enumerate all single nonsense mutations
        >>> with pp.Party() as party:
        ...     mutants = mutagenize_orf('ATGAAATTT', num_mutations=1, mutation_type='nonsense', mode='sequential')
        
        # Mutate only codons 1 and 2 (0-indexed)
        >>> with pp.Party() as party:
        ...     mutants = mutagenize_orf('ATGAAATTTTTT', num_mutations=1, codon_positions=[1, 2])
    """
    from .from_seq import from_seq
    pool = from_seq(pool) if isinstance(pool, str) else pool
    op = MutagenizeOrfOp(
        parent_pool=pool,
        num_mutations=num_mutations,
        mutation_rate=mutation_rate,
        mutation_type=mutation_type,
        orf_start=orf_start,
        orf_end=orf_end,
        codon_positions=codon_positions,
        codon_start=codon_start,
        codon_end=codon_end,
        codon_step_size=codon_step_size,
        mode=mode,
        num_hybrid_states=num_hybrid_states,
        name=op_name,
        iter_order=op_iter_order,
    )
    result_pool = Pool(operation=op, name=name, iter_order=iter_order)
    return result_pool
