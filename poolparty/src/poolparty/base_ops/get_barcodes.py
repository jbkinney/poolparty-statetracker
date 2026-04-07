"""GetBarcodes operation - generate DNA barcodes with distance and quality constraints."""

from numbers import Real
from typing import Literal, Union

import numpy as np

from ..dna_pool import DnaPool
from ..operation import Operation
from ..types import CardsType, Optional, Seq, Sequence, beartype
from ..utils.dna_seq import DnaSeq


_INT_TO_BASE = np.array(list("ACGT"))


def _hamming_distance(s1: str, s2: str) -> int:
    """Hamming distance between two equal-length strings."""
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def _edit_distance(s1: str, s2: str) -> int:
    """Levenshtein distance with O(min(m,n)) space."""
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    m, n = len(s1), len(s2)
    prev_row = list(range(m + 1))
    curr_row = [0] * (m + 1)
    for j in range(1, n + 1):
        curr_row[0] = j
        for i in range(1, m + 1):
            if s1[i - 1] == s2[j - 1]:
                curr_row[i] = prev_row[i - 1]
            else:
                curr_row[i] = 1 + min(prev_row[i], curr_row[i - 1], prev_row[i - 1])
        prev_row, curr_row = curr_row, prev_row
    return prev_row[m]


def _gc_filter_batch(candidates: np.ndarray, gc_range: tuple[float, float]) -> np.ndarray:
    """Boolean mask for candidates within GC range. Encoding: C=1, G=2."""
    gc_count = ((candidates == 1) | (candidates == 2)).sum(axis=1)
    gc_frac = gc_count / candidates.shape[1]
    return (gc_frac >= gc_range[0]) & (gc_frac <= gc_range[1])


def _homopolymer_filter_batch(candidates: np.ndarray, max_homopolymer: int) -> np.ndarray:
    """Boolean mask: True where no homopolymer run exceeds max_homopolymer."""
    n, L = candidates.shape
    if max_homopolymer < 1:
        return np.zeros(n, dtype=bool)
    if max_homopolymer >= L:
        return np.ones(n, dtype=bool)
    # same[i,j] is True when base j equals base j+1
    same = candidates[:, :-1] == candidates[:, 1:]
    window = max_homopolymer
    if window > same.shape[1]:
        return np.ones(n, dtype=bool)
    # Sliding-window sum via cumulative sum detects runs > max_homopolymer
    cumsum = np.zeros((n, same.shape[1] + 1), dtype=np.int32)
    cumsum[:, 1:] = np.cumsum(same, axis=1)
    window_sums = cumsum[:, window:] - cumsum[:, :-window]
    return ~(window_sums == window).any(axis=1)


def _generate_candidate_batch(
    rng: np.random.Generator,
    batch_size: int,
    length: int,
    gc_range: Optional[tuple[float, float]],
    max_homopolymer: Optional[int],
) -> np.ndarray:
    """Generate a batch of uint8 candidates, pre-filtered by GC and homopolymer."""
    candidates = rng.integers(0, 4, size=(batch_size, length), dtype=np.uint8)
    if gc_range is not None:
        candidates = candidates[_gc_filter_batch(candidates, gc_range)]
    if max_homopolymer is not None and len(candidates) > 0:
        candidates = candidates[_homopolymer_filter_batch(candidates, max_homopolymer)]
    return candidates


def _raise_insufficient(n_found: int, max_attempts: int, num_barcodes: int) -> None:
    raise ValueError(
        f"Could only generate {n_found} barcodes satisfying constraints "
        f"within {max_attempts} attempts (requested {num_barcodes}). "
        "Try relaxing constraints (lower min_edit_distance, wider gc_range, etc.) "
        "or increasing max_attempts."
    )


def _generate_barcodes(
    num_barcodes: int,
    lengths: list[int],
    length_proportions: Optional[list[float]],
    min_edit_distance: Optional[int],
    min_hamming_distance: Optional[int],
    max_homopolymer: Optional[int],
    gc_range: Optional[tuple[float, float]],
    avoid_sequences: list[str],
    avoid_min_distance: Optional[int],
    seed: Optional[int],
    max_attempts: int,
) -> list[str]:
    """Generate barcodes using vectorized batch generation with greedy selection."""
    if len(lengths) == 1:
        return _generate_fixed_length(
            num_barcodes, lengths[0], min_edit_distance, min_hamming_distance,
            max_homopolymer, gc_range, avoid_sequences, avoid_min_distance,
            seed, max_attempts,
        )
    return _generate_variable_length(
        num_barcodes, lengths, length_proportions, min_edit_distance,
        min_hamming_distance, max_homopolymer, gc_range, avoid_sequences,
        avoid_min_distance, seed, max_attempts,
    )


def _generate_fixed_length(
    num_barcodes: int,
    length: int,
    min_edit_distance: Optional[int],
    min_hamming_distance: Optional[int],
    max_homopolymer: Optional[int],
    gc_range: Optional[tuple[float, float]],
    avoid_sequences: list[str],
    avoid_min_distance: Optional[int],
    seed: Optional[int],
    max_attempts: int,
) -> list[str]:
    """Vectorized generation for fixed-length barcodes.

    Optimisation tiers (each subsumes the previous):
    1. min_distance <= 1  → set-based uniqueness, no pairwise check
    2. min_distance >= 2  → vectorized numpy hamming against accepted array
    3. min_edit >= 3      → hamming pre-filter (edit <= hamming rejects fast)
                            + full Levenshtein for survivors
    """
    rng = np.random.default_rng(seed)

    eff_edit = min_edit_distance or 0
    eff_hamming = min_hamming_distance or 0

    # For same-length strings, edit_distance <= hamming_distance.
    # edit=0 iff hamming=0; edit=1 iff hamming=1 (substitution only).
    # So for D <= 2: min_edit=D <=> min_hamming=D on same-length strings.
    # For D >= 3: full edit verification needed after hamming pre-filter.
    min_hamming_eff = max(eff_hamming, eff_edit)
    need_full_edit = min_edit_distance is not None and eff_edit >= 3

    accepted = np.empty((num_barcodes, length), dtype=np.uint8)
    accepted_set: set[bytes] = set()
    n_accepted = 0

    batch_size = max(num_barcodes * 4, 50_000)
    total_attempts = 0

    while n_accepted < num_barcodes and total_attempts < max_attempts:
        current_batch = min(batch_size, max_attempts - total_attempts)
        candidates = _generate_candidate_batch(
            rng, current_batch, length, gc_range, max_homopolymer
        )
        total_attempts += current_batch

        for i in range(len(candidates)):
            if n_accepted >= num_barcodes:
                break

            cand = candidates[i]
            key = cand.tobytes()

            if key in accepted_set:
                continue

            # Vectorized hamming distance against all accepted barcodes
            if n_accepted > 0 and min_hamming_eff > 1:
                hamming_dists = (accepted[:n_accepted] != cand).sum(axis=1)
                if hamming_dists.min() < min_hamming_eff:
                    continue

                if need_full_edit:
                    cand_str = "".join(_INT_TO_BASE[cand])
                    fail = False
                    for j in range(n_accepted):
                        if _edit_distance(cand_str, "".join(_INT_TO_BASE[accepted[j]])) < eff_edit:
                            fail = True
                            break
                    if fail:
                        continue

            if avoid_sequences and avoid_min_distance is not None:
                cand_str = "".join(_INT_TO_BASE[cand])
                if any(
                    _edit_distance(cand_str, av) < avoid_min_distance
                    for av in avoid_sequences
                ):
                    continue

            accepted_set.add(key)
            accepted[n_accepted] = cand
            n_accepted += 1

    if n_accepted < num_barcodes:
        _raise_insufficient(n_accepted, max_attempts, num_barcodes)

    chars = _INT_TO_BASE[accepted[:num_barcodes]]
    return ["".join(row) for row in chars]


def _generate_variable_length(
    num_barcodes: int,
    lengths: list[int],
    length_proportions: Optional[list[float]],
    min_edit_distance: Optional[int],
    min_hamming_distance: Optional[int],
    max_homopolymer: Optional[int],
    gc_range: Optional[tuple[float, float]],
    avoid_sequences: list[str],
    avoid_min_distance: Optional[int],
    seed: Optional[int],
    max_attempts: int,
) -> list[str]:
    """Generation for variable-length barcodes with batch candidate pools."""
    rng = np.random.default_rng(seed)

    if length_proportions is not None:
        length_quotas: dict[int, int] = {}
        remaining = num_barcodes
        for i, L in enumerate(lengths[:-1]):
            quota = round(length_proportions[i] * num_barcodes)
            length_quotas[L] = quota
            remaining -= quota
        length_quotas[lengths[-1]] = remaining
    else:
        base_quota = num_barcodes // len(lengths)
        remainder = num_barcodes % len(lengths)
        length_quotas = {}
        for i, L in enumerate(lengths):
            length_quotas[L] = base_quota + (1 if i < remainder else 0)

    length_counts = {L: 0 for L in lengths}

    pool_batch = max(num_barcodes * 2, 20_000)
    candidate_pools: dict[int, np.ndarray] = {}
    pool_idx: dict[int, int] = {}
    for L in lengths:
        candidate_pools[L] = _generate_candidate_batch(
            rng, pool_batch, L, gc_range, max_homopolymer
        )
        pool_idx[L] = 0

    accepted: list[str] = []
    accepted_set: set[str] = set()
    total_attempts = 0

    while len(accepted) < num_barcodes and total_attempts < max_attempts:
        available = [L for L in lengths if length_counts[L] < length_quotas[L]]
        if not available:
            break
        chosen_length = available[int(rng.integers(len(available)))]

        pool = candidate_pools[chosen_length]
        idx = pool_idx[chosen_length]

        if idx >= len(pool):
            pool = _generate_candidate_batch(
                rng, pool_batch, chosen_length, gc_range, max_homopolymer
            )
            candidate_pools[chosen_length] = pool
            pool_idx[chosen_length] = 0
            idx = 0
            if len(pool) == 0:
                total_attempts += pool_batch
                continue

        cand_arr = pool[idx]
        pool_idx[chosen_length] = idx + 1
        total_attempts += 1

        candidate = "".join(_INT_TO_BASE[cand_arr])

        if candidate in accepted_set:
            continue

        if avoid_sequences and avoid_min_distance is not None:
            if any(
                _edit_distance(candidate, av) < avoid_min_distance
                for av in avoid_sequences
            ):
                continue

        valid = True
        for existing in accepted:
            if min_edit_distance is not None:
                if _edit_distance(candidate, existing) < min_edit_distance:
                    valid = False
                    break
            if min_hamming_distance is not None and len(candidate) == len(existing):
                if _hamming_distance(candidate, existing) < min_hamming_distance:
                    valid = False
                    break
        if not valid:
            continue

        accepted.append(candidate)
        accepted_set.add(candidate)
        length_counts[chosen_length] += 1

    if len(accepted) < num_barcodes:
        _raise_insufficient(len(accepted), max_attempts, num_barcodes)

    return accepted


class GetBarcodesOp(Operation):
    """Generate constrained DNA barcodes via greedy random algorithm.

    All barcodes are pre-generated at construction time and stored.
    The resulting pool is a severed DAG leaf with sequential mode.
    """

    factory_name = "get_barcodes"
    design_card_keys: Sequence[str] = ["barcode_index", "barcode"]

    def __init__(
        self,
        num_barcodes: int,
        length: Union[int, list[int]],
        length_proportions: Optional[list[float]] = None,
        min_edit_distance: Optional[int] = None,
        min_hamming_distance: Optional[int] = None,
        gc_range: Optional[tuple[float, float]] = None,
        max_homopolymer: Optional[int] = None,
        avoid_sequences: Optional[list[str]] = None,
        avoid_min_distance: Optional[int] = None,
        padding_char: str = "-",
        padding_side: Literal["left", "right"] = "right",
        seed: Optional[int] = None,
        max_attempts: int = 100_000,
        style: Optional[str] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
        prefix: Optional[str] = None,
        cards: CardsType = None,
    ) -> None:
        from ..party import get_active_party

        party = get_active_party()
        if party is None:
            raise RuntimeError(
                "get_barcodes requires an active Party context. "
                "Use 'with pp.Party() as party:' to create one."
            )

        if not isinstance(num_barcodes, int) or num_barcodes <= 0:
            raise ValueError(f"num_barcodes must be a positive integer, got {num_barcodes}")

        lengths = [length] if isinstance(length, int) else list(length)
        if not lengths:
            raise ValueError("length must be a non-empty int or list of ints")
        for L in lengths:
            if not isinstance(L, int) or L <= 0:
                raise ValueError(f"All lengths must be positive integers, got {L}")

        is_variable = len(lengths) > 1

        if is_variable and min_hamming_distance is not None:
            raise ValueError(
                "min_hamming_distance cannot be used with variable-length barcodes. "
                "Use min_edit_distance instead."
            )

        if length_proportions is not None:
            if len(length_proportions) != len(lengths):
                raise ValueError(
                    f"length_proportions length ({len(length_proportions)}) must match "
                    f"length list length ({len(lengths)})"
                )
            if any(p <= 0 for p in length_proportions):
                raise ValueError("All length_proportions values must be positive")
            total = sum(length_proportions)
            length_proportions = [p / total for p in length_proportions]

        if gc_range is not None:
            if len(gc_range) != 2:
                raise ValueError("gc_range must be a tuple of (min_gc, max_gc)")
            min_gc, max_gc = gc_range
            if not (0 <= min_gc <= 1 and 0 <= max_gc <= 1):
                raise ValueError(f"gc_range values must be in [0, 1], got {gc_range}")
            if min_gc > max_gc:
                raise ValueError(f"gc_range min ({min_gc}) cannot exceed max ({max_gc})")

        if avoid_sequences is not None and avoid_min_distance is None:
            raise ValueError("avoid_min_distance is required when avoid_sequences is provided")

        self._style = style
        self._padding_char = padding_char
        self._padding_side = padding_side
        max_length = max(lengths)

        raw_barcodes = _generate_barcodes(
            num_barcodes=num_barcodes,
            lengths=lengths,
            length_proportions=length_proportions,
            min_edit_distance=min_edit_distance,
            min_hamming_distance=min_hamming_distance,
            max_homopolymer=max_homopolymer,
            gc_range=gc_range,
            avoid_sequences=avoid_sequences or [],
            avoid_min_distance=avoid_min_distance,
            seed=seed,
            max_attempts=max_attempts,
        )

        self._barcode_strings: list[str] = [
            self._pad(bc, max_length) for bc in raw_barcodes
        ]
        self._current_idx: int = 0

        seq_length = max_length

        super().__init__(
            parent_pools=[],
            num_states=num_barcodes,
            mode="sequential",
            seq_length=seq_length,
            name=name,
            iter_order=iter_order,
            prefix=prefix,
            cards=cards,
        )

    def _pad(self, barcode: str, max_length: int) -> str:
        if len(barcode) >= max_length:
            return barcode
        padding = self._padding_char * (max_length - len(barcode))
        if self._padding_side == "right":
            return barcode + padding
        return padding + barcode

    def _compute_core(
        self,
        parents: list[Seq],
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[Seq, dict]:
        state = self.state.value
        idx = (0 if state is None else state) % len(self._barcode_strings)
        self._current_idx = idx
        barcode = self._barcode_strings[idx]

        from ..utils.style_utils import SeqStyle, styles_suppressed

        if styles_suppressed():
            output_seq = DnaSeq(barcode, None)
        else:
            output_style = SeqStyle.full(len(barcode), self._style)
            output_seq = DnaSeq(barcode, output_style)

        return output_seq, {
            "barcode_index": idx,
            "barcode": barcode,
        }

    def compute_name_contributions(self, global_state=None, max_global_state=None) -> list[str]:
        if not self.state.is_active:
            return []
        return super().compute_name_contributions(global_state, max_global_state)


@beartype
def get_barcodes(
    num_barcodes: int,
    length: Union[int, list[int]],
    length_proportions: Optional[list[float]] = None,
    min_edit_distance: Optional[int] = None,
    min_hamming_distance: Optional[int] = None,
    gc_range: Optional[tuple[float, float]] = None,
    max_homopolymer: Optional[int] = None,
    avoid_sequences: Optional[list[str]] = None,
    avoid_min_distance: Optional[int] = None,
    padding_char: str = "-",
    padding_side: Literal["left", "right"] = "right",
    seed: Optional[int] = None,
    max_attempts: int = 100_000,
    style: Optional[str] = None,
    name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    prefix: Optional[str] = None,
    cards: CardsType = None,
) -> DnaPool:
    """Generate DNA barcodes satisfying distance and quality constraints.

    Pre-generates all barcodes at construction time using a greedy random
    algorithm. The resulting pool has num_states equal to num_barcodes,
    sequential mode, and a severed DAG (no parent references).

    Parameters
    ----------
    num_barcodes : int
        Number of barcodes to generate.
    length : int or list of int
        Barcode length. If a list, generates variable-length barcodes
        padded to max length.
    length_proportions : list of float, optional
        Distribution across lengths for variable-length barcodes.
        Must match length list. Values normalized to sum to 1.
        If None, equal distribution. Ignored if length is a single int.
    min_edit_distance : int, optional
        Minimum Levenshtein distance between any two barcodes.
    min_hamming_distance : int, optional
        Minimum Hamming distance. Only for fixed-length barcodes.
    gc_range : tuple of (float, float), optional
        (min_gc, max_gc) as fractions in [0, 1].
    max_homopolymer : int, optional
        Maximum consecutive identical bases allowed.
    avoid_sequences : list of str, optional
        External sequences to maintain distance from (e.g., adapters).
    avoid_min_distance : int, optional
        Minimum edit distance from avoid_sequences. Required if
        avoid_sequences is provided.
    padding_char : str, default '-'
        Character for padding variable-length barcodes.
    padding_side : 'left' or 'right', default 'right'
        Which side to pad shorter barcodes.
    seed : int, optional
        Random seed for reproducible generation.
    max_attempts : int, default 100000
        Maximum candidate attempts before raising an error.
    style : str, optional
        Inline style for barcode sequences.
    name : str, optional
        Operation name.
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.
    prefix : str, optional
        Name prefix.
    cards : CardsType, optional
        Design card control. Available keys: ``'barcode_index'``,
        ``'barcode'``.

    Returns
    -------
    DnaPool
        A pool of generated barcodes with num_states == num_barcodes.

    Raises
    ------
    ValueError
        If constraints cannot be satisfied within max_attempts.
    RuntimeError
        If called outside a Party context.

    Examples
    --------
    >>> barcodes = pp.get_barcodes(num_barcodes=100, length=8,
    ...     min_edit_distance=3, gc_range=(0.3, 0.6), seed=42)
    >>> barcodes.num_states
    100
    """
    op = GetBarcodesOp(
        num_barcodes=num_barcodes,
        length=length,
        length_proportions=length_proportions,
        min_edit_distance=min_edit_distance,
        min_hamming_distance=min_hamming_distance,
        gc_range=gc_range,
        max_homopolymer=max_homopolymer,
        avoid_sequences=avoid_sequences,
        avoid_min_distance=avoid_min_distance,
        padding_char=padding_char,
        padding_side=padding_side,
        seed=seed,
        max_attempts=max_attempts,
        style=style,
        name=name,
        iter_order=iter_order,
        prefix=prefix,
        cards=cards,
    )
    return DnaPool(operation=op)
