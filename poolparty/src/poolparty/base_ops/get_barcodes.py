"""GetBarcodes operation - generate DNA barcodes with distance and quality constraints."""

import random
from numbers import Real
from typing import Literal, Union

import numpy as np

from ..dna_pool import DnaPool
from ..operation import Operation
from ..types import CardsType, Optional, Pool_type, Seq, Sequence, beartype
from ..utils.dna_seq import DnaSeq


ALPHABET = ["A", "C", "G", "T"]


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


def _check_homopolymer(seq: str, max_homopolymer: int) -> bool:
    """True if no homopolymer run exceeds max_homopolymer."""
    if len(seq) <= max_homopolymer:
        return True
    run_length = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            run_length += 1
            if run_length > max_homopolymer:
                return False
        else:
            run_length = 1
    return True


def _check_gc_content(seq: str, min_gc: float, max_gc: float) -> bool:
    """True if GC content is within [min_gc, max_gc]."""
    if not seq:
        return True
    gc_count = sum(1 for base in seq if base in ("G", "C"))
    gc_frac = gc_count / len(seq)
    return min_gc <= gc_frac <= max_gc


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
    """Generate barcodes using greedy random algorithm."""
    rng = random.Random(seed)
    accepted: list[str] = []

    if len(lengths) > 1 and length_proportions is not None:
        length_quotas = {}
        remaining = num_barcodes
        for i, L in enumerate(lengths[:-1]):
            quota = round(length_proportions[i] * num_barcodes)
            length_quotas[L] = quota
            remaining -= quota
        length_quotas[lengths[-1]] = remaining
    elif len(lengths) > 1:
        base_quota = num_barcodes // len(lengths)
        remainder = num_barcodes % len(lengths)
        length_quotas = {}
        for i, L in enumerate(lengths):
            length_quotas[L] = base_quota + (1 if i < remainder else 0)
    else:
        length_quotas = None

    length_counts = {L: 0 for L in lengths} if length_quotas else None

    attempts = 0
    while len(accepted) < num_barcodes and attempts < max_attempts:
        attempts += 1

        if length_quotas is not None:
            available = [L for L in lengths if length_counts[L] < length_quotas[L]]
            if not available:
                break
            chosen_length = rng.choice(available)
        else:
            chosen_length = lengths[0]

        candidate = "".join(rng.choice(ALPHABET) for _ in range(chosen_length))

        if max_homopolymer is not None and not _check_homopolymer(candidate, max_homopolymer):
            continue
        if gc_range is not None and not _check_gc_content(candidate, gc_range[0], gc_range[1]):
            continue

        if avoid_sequences and avoid_min_distance is not None:
            if any(_edit_distance(candidate, av) < avoid_min_distance for av in avoid_sequences):
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
        if length_counts is not None:
            length_counts[chosen_length] += 1

    if len(accepted) < num_barcodes:
        raise ValueError(
            f"Could only generate {len(accepted)} barcodes satisfying constraints "
            f"within {max_attempts} attempts (requested {num_barcodes}). "
            "Try relaxing constraints (lower min_edit_distance, wider gc_range, etc.) "
            "or increasing max_attempts."
        )

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
