"""Mutagenize operation - apply mutations to a sequence."""

from functools import lru_cache
from itertools import combinations
from math import comb, prod

import numpy as np

from ..operation import Operation
from ..party import get_active_party
from ..pool import Pool
from ..types import CardsType, Integral, ModeType, Optional, Real, RegionType, Seq, Union, beartype
from ..utils import dna_utils
from ..utils.dna_seq import DnaSeq


@beartype
def mutagenize(
    pool: Union[Pool, str],
    region: RegionType = None,
    num_mutations: Optional[Integral] = None,
    mutation_rate: Optional[Real] = None,
    allowed_chars: Optional[str] = None,
    style: Optional[str] = None,
    prefix: Optional[str] = None,
    mode: ModeType = "random",
    num_states: Optional[Integral] = None,
    iter_order: Optional[Real] = None,
    _remove_tags: bool = False,
    cards: CardsType = None,
    _factory_name: Optional[str] = "mutagenize",
) -> Pool:
    """
    Create a Pool that applies mutations to a sequence.

    Parameters
    ----------
    pool : Union[Pool, str]
        Parent pool or sequence string to mutate.
    region : Union[str, Sequence[Integral], None], default=None
        Region to mutagenize. Can be a marker name (str), explicit interval [start, stop],
        or None to mutagenize entire sequence. Positions are region-relative.
    num_mutations : Optional[Integral], default=None
        Fixed number of mutations to apply (mutually exclusive with mutation_rate).
    mutation_rate : Optional[Real], default=None
        Probability of mutation at each position (mutually exclusive with num_mutations).
    allowed_chars : Optional[str], default=None
        IUPAC string of same length as sequence, specifying allowed bases at each position.
        Each character is an IUPAC code (A, C, G, T, R, Y, S, W, K, M, B, D, H, V, N).
        Positions where only the wild-type is allowed are treated as non-mutable.
    style : Optional[str], default=None
        Style to apply to mutated positions (e.g., 'red', 'blue bold').
    prefix : Optional[str], default=None
        Prefix for sequence names in the resulting Pool.
    mode : ModeType, default='random'
        Selection mode: 'random' or 'sequential'. Sequential only available with num_mutations.
    num_states : Optional[int], default=None
        Number of states. In sequential mode, overrides the computed count
        (cycling if greater, clipping if less). In random mode, if None
        defaults to 1 (pure random sampling).
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.
    cards : list[str] or dict, optional
        Design card keys to include. Available keys: ``'positions'``,
        ``'wt_chars'``, ``'mut_chars'``.

    Returns
    -------
    Pool
        A Pool that generates mutated sequences.
    """

    from ..fixed_ops.from_seq import from_seq

    pool = (
        from_seq(pool, _factory_name=f"{_factory_name}(from_seq)")
        if isinstance(pool, str)
        else pool
    )
    op = MutagenizeOp(
        pool=pool,
        num_mutations=num_mutations,
        mutation_rate=mutation_rate,
        allowed_chars=allowed_chars,
        region=region,
        style=style,
        prefix=prefix,
        mode=mode,
        num_states=num_states,
        iter_order=iter_order,
        _remove_tags=_remove_tags,
        cards=cards,
        _factory_name=_factory_name,
    )
    # Preserve the pool type from the input
    pool_class = type(pool)
    result_pool = pool_class(operation=op)
    return result_pool


class MutagenizeOp(Operation):
    """Apply mutations to a parent sequence or a specified region within it.

    Supports two mutation modes:
    - num_mutations: Apply exactly this many mutations to each sequence
    - mutation_rate: Apply a random number of mutations based on a binomial distribution

    Exactly one of num_mutations or mutation_rate must be provided.
    Sequential mode is only available when num_mutations is specified.
    """

    factory_name = "mutagenize"
    design_card_keys = ["positions", "wt_chars", "mut_chars"]

    def __init__(
        self,
        pool: Pool,
        num_mutations: Optional[Integral] = None,
        mutation_rate: Optional[Real] = None,
        allowed_chars: Optional[str] = None,
        region: RegionType = None,
        style: Optional[str] = None,
        prefix: Optional[str] = None,
        mode: ModeType = "random",
        num_states: Optional[Integral] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
        _remove_tags: bool = False,
        cards: CardsType = None,
        _factory_name: Optional[str] = "mutagenize",
    ) -> None:
        # Set factory name if provided
        if _factory_name is not None:
            self.factory_name = _factory_name

        # Get alphabet from active Party context
        party = get_active_party()
        if party is None:
            raise RuntimeError(
                "mutagenize requires an active Party context. "
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
            if mode == "sequential":
                raise ValueError(
                    "mode='sequential' is not supported with mutation_rate (use num_mutations instead)"
                )

        self.num_mutations = num_mutations
        self.mutation_rate = mutation_rate
        self.allowed_chars = allowed_chars
        self._style = style
        self.alpha_size = len(dna_utils.BASES)
        self._mode = mode

        # Validate and process allowed_chars if provided
        self._allowed_bases_per_pos = None  # Will be set if allowed_chars is provided
        self._mutation_counts_from_allowed = None  # Pre-computed mutation counts
        if allowed_chars is not None:
            invalid_chars = set()
            allowed_bases_per_pos = []
            mutation_counts = []
            for char in allowed_chars.upper():
                if char in dna_utils.IGNORE_CHARS:
                    continue  # Skip ignore chars (gaps, separators)
                if char not in dna_utils.IUPAC_TO_DNA:
                    invalid_chars.add(char)
                else:
                    bases = set(dna_utils.IUPAC_TO_DNA[char])
                    allowed_bases_per_pos.append(bases)
                    mutation_counts.append(len(bases) - 1)  # -1 for the wt
            if invalid_chars:
                raise ValueError(
                    f"allowed_chars contains invalid IUPAC character(s): {sorted(invalid_chars)}. "
                    f"Valid IUPAC characters are: {sorted(set(dna_utils.IUPAC_TO_DNA.keys()) - set('acgtryswkmbdhvn'))} "
                    f"(plus ignore characters: {sorted(dna_utils.IGNORE_CHARS)})"
                )
            self._allowed_bases_per_pos = allowed_bases_per_pos
            self._mutation_counts_from_allowed = mutation_counts

        # Build mutation map: (wt_char, index) -> mut_char
        self._mutation_map = {}
        for wt in dna_utils.VALID_CHARS:
            for i, mut in enumerate(dna_utils.get_mutations(wt)):
                self._mutation_map[(wt, i)] = mut

        self._seq_length = pool.seq_length
        self._sequential_cache = None
        self._num_mutable_positions = None  # Actual mutable positions, set on first use

        # Store user-provided num_states for potential override
        user_num_states = num_states
        natural_num_states = None

        # Determine num_states based on mode
        if mode == "sequential":
            # Sequential mode only available with num_mutations
            effective_length = self._seq_length
            # If region is specified, use the region's length instead of full sequence
            if isinstance(region, str):
                try:
                    region_obj = party.get_region_by_name(region)
                    if region_obj.seq_length is not None:
                        effective_length = region_obj.seq_length
                except ValueError:
                    pass  # Region not found, fall back to full sequence length
            elif region is not None:
                effective_length = int(region[1]) - int(region[0])

            # If allowed_chars is provided, use its length and pre-computed mutation counts
            if self._mutation_counts_from_allowed is not None:
                effective_length = len(self._mutation_counts_from_allowed)
                # Filter to positions with at least 1 mutation option
                mutable_counts = [c for c in self._mutation_counts_from_allowed if c > 0]
                num_mutable = len(mutable_counts)
                if num_mutable < num_mutations:
                    raise ValueError(
                        f"{num_mutations=} exceeds mutable positions={num_mutable}. "
                        f"Cannot apply {num_mutations} mutations."
                    )
                natural_num_states = self._build_caches(num_mutable, mutable_counts)
            elif effective_length is not None:
                if effective_length < num_mutations:
                    raise ValueError(
                        f"{num_mutations=} exceeds sequence length={effective_length}. "
                        f"Cannot apply {num_mutations} mutations to a sequence of length {effective_length}."
                    )
                natural_num_states = self._build_caches(effective_length)
            else:
                raise ValueError(
                    "mode='sequential' requires a parent pool with known "
                    "seq_length. The parent pool has seq_length=None (e.g., "
                    "from_seqs with variable-length sequences). Use "
                    "mode='random' instead, or ensure all parent sequences "
                    "have the same length."
                )
            # Use user-provided num_states if given, else natural count
            num_states = user_num_states if user_num_states is not None else natural_num_states
        elif mode == "random":
            # num_states stays as provided (or None for pure random mode)
            pass
        else:
            num_states = 1

        super().__init__(
            parent_pools=[pool],
            num_states=num_states,
            mode=mode,
            seq_length=self._seq_length,
            name=name,
            iter_order=iter_order,
            prefix=prefix,
            region=region,
            remove_tags=_remove_tags,
            _natural_num_states=natural_num_states,
            cards=cards,
        )

        # Create LRU-cached version for position data computation
        self._cached_get_positions = lru_cache(maxsize=8)(self._compute_positions_data)

    def _compute_positions_data(self, seq: str):
        """Compute and return position data (cached via _cached_get_positions)."""
        valid_char_positions = self._get_molecular_positions(seq)
        mutable_positions, mutation_options = self._get_position_mutations(
            seq, valid_char_positions
        )
        # Pre-convert to numpy for faster mutation application
        seq_bytes = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)

        # Pre-compute numpy arrays for vectorized mutation
        num_mutable = len(mutable_positions)
        if num_mutable > 0:
            # Convert positions to numpy arrays
            mutable_positions_arr = np.array(mutable_positions, dtype=np.intp)
            valid_char_positions_arr = np.array(valid_char_positions, dtype=np.intp)

            # Build raw_positions lookup: raw_pos = valid_char_positions[mutable_positions[i]]
            raw_positions_arr = valid_char_positions_arr[mutable_positions_arr]

            # Pre-extract WT characters as bytes at mutable positions
            wt_bytes_arr = seq_bytes[raw_positions_arr]

            # Build mutation options as 2D numpy array (num_positions x max_options)
            # Each row contains byte values of valid mutations, padded with 0
            mutation_counts = [len(opts) for opts in mutation_options]
            max_options = max(mutation_counts) if mutation_counts else 0
            mutation_options_arr = np.zeros((num_mutable, max_options), dtype=np.uint8)
            mutation_counts_arr = np.array(mutation_counts, dtype=np.intp)
            for i, opts in enumerate(mutation_options):
                for j, char in enumerate(opts):
                    mutation_options_arr[i, j] = ord(char)
        else:
            valid_char_positions_arr = (
                np.array(valid_char_positions, dtype=np.intp)
                if valid_char_positions
                else np.array([], dtype=np.intp)
            )
            mutable_positions_arr = np.array([], dtype=np.intp)
            raw_positions_arr = np.array([], dtype=np.intp)
            wt_bytes_arr = np.array([], dtype=np.uint8)
            mutation_options_arr = np.zeros((0, 0), dtype=np.uint8)
            mutation_counts_arr = np.array([], dtype=np.intp)

        return (
            valid_char_positions,  # Keep for compatibility
            mutable_positions,  # Keep for compatibility
            mutation_options,  # Keep for compatibility
            seq_bytes,  # Keep for _apply_mutations_numpy
            # Cached numpy arrays for vectorized operations:
            valid_char_positions_arr,
            mutable_positions_arr,
            raw_positions_arr,
            wt_bytes_arr,
            mutation_options_arr,
            mutation_counts_arr,
        )

    def _build_caches(self, num_positions: int, mutation_counts: Optional[list[int]] = None) -> int:
        """Build caches for sequential enumeration.

        Parameters
        ----------
        num_positions : int
            Number of mutable positions (valid alphabet characters only).
        mutation_counts : Optional[list[int]]
            Number of valid mutations per position. If None, uses uniform alpha_size-1.
        """
        if num_positions < self.num_mutations:
            raise ValueError(
                f"num_mutations={self.num_mutations} exceeds mutable positions={num_positions}. "
                f"Cannot apply {self.num_mutations} mutations."
            )

        if mutation_counts is None:
            # Uniform case: each position has alpha_size - 1 mutations
            alpha_minus_1 = self.alpha_size - 1
            num_combinations = comb(num_positions, self.num_mutations) * (
                alpha_minus_1**self.num_mutations
            )
            cache = []
            for positions in combinations(range(num_positions), self.num_mutations):
                num_mut_patterns = alpha_minus_1**self.num_mutations
                for mut_pattern in range(num_mut_patterns):
                    mut_indices = []
                    remaining = mut_pattern
                    for _ in range(self.num_mutations):
                        mut_indices.append(remaining % alpha_minus_1)
                        remaining //= alpha_minus_1
                    cache.append((positions, tuple(reversed(mut_indices))))
        else:
            # Non-uniform case: each position has different number of mutations
            cache = []
            for positions in combinations(range(num_positions), self.num_mutations):
                counts_for_positions = [mutation_counts[p] for p in positions]
                num_mut_patterns = prod(counts_for_positions)
                for mut_pattern in range(num_mut_patterns):
                    mut_indices = []
                    remaining = mut_pattern
                    for count in counts_for_positions:
                        mut_indices.append(remaining % count)
                        remaining //= count
                    cache.append((positions, tuple(mut_indices)))
            num_combinations = len(cache)

        self._sequential_cache = cache
        self._num_mutable_positions = num_positions
        self._mutation_counts = tuple(mutation_counts) if mutation_counts else None
        return num_combinations

    def _get_position_mutations(
        self, seq: str, valid_char_positions: list[int]
    ) -> tuple[list[int], list[list[str]]]:
        """Get mutable positions and their valid mutation options.

        Returns a tuple of (mutable_logical_positions, mutation_options_per_position).
        Positions where wt is the only allowed char are excluded.

        When allowed_chars is set, also validates that the input sequence has
        allowed characters at each position.
        """
        mutable_positions = []
        mutation_options = []

        for logical_pos, raw_pos in enumerate(valid_char_positions):
            wt = seq[raw_pos]
            wt_upper = wt.upper()

            if self._allowed_bases_per_pos is not None:
                # Validate length
                if len(self._allowed_bases_per_pos) != len(valid_char_positions):
                    raise ValueError(
                        f"allowed_chars length ({len(self._allowed_bases_per_pos)}) must match "
                        f"sequence length ({len(valid_char_positions)})"
                    )
                # Get pre-computed allowed bases at this position
                allowed_bases_upper = self._allowed_bases_per_pos[logical_pos]

                # Validate that wt is in the allowed set
                if wt_upper not in allowed_bases_upper:
                    raise ValueError(
                        f"Sequence character '{wt}' at position {logical_pos} is not in "
                        f"allowed_chars '{self.allowed_chars[logical_pos]}' (allowed: {sorted(allowed_bases_upper)})"
                    )

                # Get mutation targets (allowed bases minus wt), preserving case
                if wt.islower():
                    valid_muts = [b.lower() for b in sorted(allowed_bases_upper) if b != wt_upper]
                else:
                    valid_muts = [b for b in sorted(allowed_bases_upper) if b != wt_upper]
            else:
                # No restriction: all non-wt bases are valid
                valid_muts = dna_utils.MUTATIONS_DICT[wt]

            if valid_muts:
                mutable_positions.append(logical_pos)
                mutation_options.append(valid_muts)

        return mutable_positions, mutation_options

    def _random_mutation(
        self,
        rng: np.random.Generator,
        num_mutable: int,
        mutable_positions_arr: np.ndarray,
        wt_bytes_arr: np.ndarray,
        mutation_options_arr: np.ndarray,
        mutation_counts_arr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate random mutation positions and characters (vectorized).

        Returns numpy arrays: (positions, wt_bytes, mut_bytes)
        """
        if num_mutable == 0:
            return (
                np.array([], dtype=np.intp),
                np.array([], dtype=np.uint8),
                np.array([], dtype=np.uint8),
            )

        # Determine number of mutations
        if self.num_mutations is not None:
            num_mut = min(self.num_mutations, num_mutable)
        else:
            num_mut = rng.binomial(num_mutable, self.mutation_rate)
            if num_mut == 0:
                return (
                    np.array([], dtype=np.intp),
                    np.array([], dtype=np.uint8),
                    np.array([], dtype=np.uint8),
                )

        # Choose random position indices
        chosen_indices = rng.choice(num_mutable, size=num_mut, replace=False)

        # Vectorized lookups
        positions = mutable_positions_arr[chosen_indices]
        wt_bytes = wt_bytes_arr[chosen_indices]

        # Vectorized mutation selection
        # For each chosen position, pick a random index from 0 to mutation_count-1
        counts = mutation_counts_arr[chosen_indices]
        mut_indices = (rng.random(num_mut) * counts).astype(np.intp)
        mut_bytes = mutation_options_arr[chosen_indices, mut_indices]

        # Sort by position for consistent output (needed for design cards)
        sort_order = np.argsort(positions)
        return positions[sort_order], wt_bytes[sort_order], mut_bytes[sort_order]

    def _apply_mutations_numpy(
        self,
        seq_bytes: np.ndarray,
        positions: np.ndarray,
        mut_bytes: np.ndarray,
        valid_char_positions_arr: np.ndarray,
    ) -> str:
        """Apply mutations using NumPy for better performance on long sequences."""
        if len(positions) == 0:
            return seq_bytes.tobytes().decode("ascii")

        # Copy the cached array (fast for numpy)
        seq_arr = seq_bytes.copy()

        # Compute raw positions (vectorized)
        raw_positions = valid_char_positions_arr[positions]

        # Apply all mutations at once (vectorized)
        seq_arr[raw_positions] = mut_bytes

        # Convert back to string
        return seq_arr.tobytes().decode("ascii")

    def _compute_core(
        self,
        parents: list[Seq],
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[Seq, dict]:
        """Return mutated Seq and design card.

        Note: Region handling is done by base class compute() method.
        parents[0] is the region content when region is specified.
        """
        seq = parents[0].string

        # Validate no IUPAC ambiguity codes in region (mutations require ACGT only)
        # Strip tags to get actual sequence content (e.g., "<bc/>" should not trigger error)
        from ..utils.parsing_utils import strip_all_tags

        clean_content = strip_all_tags(seq)
        iupac_ambiguity = set("RYSWKMBDHVNryswkmbdhvn")
        invalid_chars = set(clean_content) & iupac_ambiguity
        if invalid_chars:
            raise ValueError(
                f"mutagenize() cannot mutate IUPAC ambiguity codes: {sorted(invalid_chars)}. "
                "Region must contain only A, C, G, T."
            )

        # Use cached position computation (includes pre-converted numpy arrays)
        (
            valid_char_positions,
            mutable_positions,
            mutation_options,
            seq_bytes,
            valid_char_positions_arr,
            mutable_positions_arr,
            raw_positions_arr,
            wt_bytes_arr,
            mutation_options_arr,
            mutation_counts_arr,
        ) = self._cached_get_positions(seq)
        num_mutable = len(mutable_positions)

        if self.num_mutations is not None and self.num_mutations > num_mutable:
            raise ValueError(
                f"Cannot apply {self.num_mutations} mutations: only {num_mutable} mutable positions"
            )

        if self.mode == "random":
            if rng is None:
                raise RuntimeError(f"{self.mode} mode requires RNG - use Party.generate(seed=...)")
            positions, wt_bytes, mut_bytes = self._random_mutation(
                rng,
                num_mutable,
                mutable_positions_arr,
                wt_bytes_arr,
                mutation_options_arr,
                mutation_counts_arr,
            )
        else:
            # Sequential mode — cache is always pre-built at init time
            # (seq_length=None + sequential is rejected at init)

            # Use state 0 when inactive (state is None)
            state = self.state.value
            state = 0 if state is None else state
            rel_positions, mut_indices = self._sequential_cache[state % len(self._sequential_cache)]

            # Map relative positions back to logical positions (as numpy arrays)
            positions = mutable_positions_arr[np.array(rel_positions, dtype=np.intp)]
            wt_bytes = wt_bytes_arr[np.array(rel_positions, dtype=np.intp)]
            mut_bytes = np.array(
                [
                    ord(mutation_options[rel_pos][mut_idx])
                    for rel_pos, mut_idx in zip(rel_positions, mut_indices)
                ],
                dtype=np.uint8,
            )

        # Apply mutations to sequence using NumPy for performance
        result_seq = self._apply_mutations_numpy(
            seq_bytes, positions, mut_bytes, valid_char_positions_arr
        )

        # Build output styles: pass through parent styles (mutagenize preserves length)
        # and add mutation style if _style is set
        output_style = parents[0].style

        if output_style is not None and self._style is not None and len(positions) > 0:
            # Convert logical positions to raw positions for styling (vectorized)
            raw_positions = valid_char_positions_arr[positions].astype(np.int64)
            output_style = output_style.add_style(self._style, raw_positions)

        output_seq = DnaSeq(result_seq, output_style)

        # Only convert bytes to chars for design cards (if not suppressed)
        if self._party.suppress_cards:
            return output_seq, {}

        return output_seq, {
            "positions": tuple(positions.tolist()),
            "wt_chars": tuple(chr(b) for b in wt_bytes),
            "mut_chars": tuple(chr(b) for b in mut_bytes),
        }
