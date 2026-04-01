"""Materialize operation - freeze a pool's sequences into a new pool with fixed states."""

from numbers import Real

import numpy as np
import pandas as pd

from ..dna_pool import DnaPool
from ..operation import Operation
from ..types import CardsType, Optional, Pool_type, Seq, Sequence, beartype
from ..utils import dna_utils
from ..utils.dna_seq import DnaSeq


class MaterializeOp(Operation):
    """Materialize sequences from a source pool into a fixed state pool.

    Generates sequences from the source pool during initialization and stores
    them for later retrieval. The resulting pool has no parent references
    (severed DAG) and a well-defined num_states equal to the number of
    materialized sequences.
    """

    factory_name = "materialize"
    design_card_keys: Sequence[str] = ["seq_index", "seq_name"]

    def __init__(
        self,
        source_pool: Pool_type,
        num_seqs: Optional[int] = None,
        num_cycles: Optional[int] = None,
        seed: Optional[int] = None,
        discard_null_seqs: bool = True,
        max_iterations: Optional[int] = None,
        min_acceptance_rate: Optional[float] = None,
        attempts_per_rate_assessment: int = 100,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
        prefix: Optional[str] = None,
        cards: CardsType = None,
    ) -> None:
        """Initialize MaterializeOp by generating sequences from source pool."""
        from ..party import get_active_party

        party = get_active_party()
        if party is None:
            raise RuntimeError(
                "materialize requires an active Party context. "
                "Use 'with pp.Party() as party:' to create one."
            )

        # Validate num_seqs/num_cycles arguments
        if num_seqs is not None and num_cycles is not None:
            raise ValueError("Cannot specify both num_seqs and num_cycles")

        # Generate sequences from the source pool
        if num_seqs is not None:
            # num_seqs mode: use generate_library's discard_null_seqs directly
            df = source_pool.generate_library(
                num_seqs=num_seqs,
                seed=seed,
                discard_null_seqs=discard_null_seqs,
                max_iterations=max_iterations,
                min_acceptance_rate=min_acceptance_rate,
                attempts_per_rate_assessment=attempts_per_rate_assessment,
            )
        else:
            # num_cycles mode: generate all sequences, then filter ourselves
            cycles = num_cycles if num_cycles is not None else 1
            df = source_pool.generate_library(
                num_cycles=cycles,
                seed=seed,
                discard_null_seqs=False,  # Get all, filter below
            )
            # Filter out null sequences if requested
            if discard_null_seqs:
                df = df[df["seq"].notna() & (df["seq"] != "")].reset_index(drop=True)

        # Store the materialized sequences and names
        self._seqs: list[Seq] = []
        self._names: list[str] = []

        for _, row in df.iterrows():
            seq_str = row["seq"]
            # Use pd.notna() for cross-platform compatibility (None vs nan)
            if pd.notna(seq_str):
                self._seqs.append(DnaSeq.from_string(seq_str))
                self._names.append(row["name"] if pd.notna(row["name"]) else "")
            else:
                # Handle None/nan sequences (when discard_null_seqs=False)
                from ..types import NullSeq

                self._seqs.append(NullSeq())
                self._names.append("")

        # Track whether explicit names exist
        self._names_explicit = any(n for n in self._names)

        # Store current index for name computation
        self._current_idx: int = 0

        if len(self._seqs) == 0:
            raise ValueError(
                "No sequences were materialized. Check that the source pool "
                "has valid sequences or adjust filter criteria."
            )

        # Determine sequence length (None if variable)
        lengths = [dna_utils.get_length_without_tags(s.string) for s in self._seqs if s.string]
        seq_length = lengths[0] if lengths and all(L == lengths[0] for L in lengths) else None

        # Initialize with NO parents (severed DAG)
        super().__init__(
            parent_pools=[],  # Severed DAG - no parent references
            num_states=len(self._seqs),
            mode="sequential",
            seq_length=seq_length,
            name=name,
            iter_order=iter_order,
            prefix=prefix,
            cards=cards,
        )

    def _compute_core(
        self,
        parents: list[Seq],
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[Seq, dict]:
        """Return stored Seq and design card for current state."""
        # Get index from state (cycling if needed)
        state = self.state.value
        idx = (0 if state is None else state) % len(self._seqs)

        # Store for name computation
        self._current_idx = idx

        output_seq = self._seqs[idx]

        return output_seq, {
            "seq_index": idx,
            "seq_name": self._names[idx],
        }

    def compute_name_contributions(self, global_state=None, max_global_state=None) -> list[str]:
        """Compute name contributions - use stored names or prefix pattern."""
        # Check if state is inactive
        if not self.state.is_active:
            return []
        if self._names_explicit and self._names[self._current_idx]:
            # Use stored name for current index
            return [self._names[self._current_idx]]
        # Otherwise use default prefix logic from base class
        return super().compute_name_contributions(global_state, max_global_state)


@beartype
def materialize(
    pool: Pool_type,
    num_seqs: Optional[int] = None,
    num_cycles: Optional[int] = None,
    seed: Optional[int] = None,
    discard_null_seqs: bool = True,
    max_iterations: Optional[int] = None,
    min_acceptance_rate: Optional[float] = None,
    attempts_per_rate_assessment: int = 100,
    name: Optional[str] = None,
    prefix: Optional[str] = None,
    cards: CardsType = None,
) -> DnaPool:
    """Materialize a pool's sequences into a new pool with fixed states.

    Generates sequences from the input pool and creates a new pool that stores
    them. The resulting pool has a well-defined num_states and no parent
    references (severed DAG), making it a leaf node in any future computation.

    Parameters
    ----------
    pool : Pool_type
        Source pool to materialize sequences from.
    num_seqs : Optional[int], default=None
        Number of sequences to generate and store. Mutually exclusive with
        num_cycles.
    num_cycles : Optional[int], default=None
        Number of complete cycles through the source pool's state space.
        If both num_seqs and num_cycles are None, defaults to 1 cycle.
        Mutually exclusive with num_seqs.
    seed : Optional[int], default=None
        Random seed for reproducible generation.
    discard_null_seqs : bool, default=True
        If True, filtered/null sequences are excluded. If False, they are
        included as NullSeq objects.
    max_iterations : Optional[int], default=None
        Maximum iterations before stopping (only used with num_seqs).
    min_acceptance_rate : Optional[float], default=None
        Minimum fraction of sequences that must pass filters.
    attempts_per_rate_assessment : int, default=100
        Iterations between acceptance rate checks.
    name : Optional[str], default=None
        Optional name for the operation.
    prefix : Optional[str], default=None
        Prefix for auto-generated names.

    Returns
    -------
    Pool_type
        A new Pool containing the materialized sequences with a fixed number
        of states equal to the number of stored sequences.

    Examples
    --------
    >>> # Basic materialization with num_seqs
    >>> pool = pp.from_seqs(["AAA", "CCC", "GGG"])
    >>> materialized = pool.materialize(num_seqs=3, seed=42)
    >>> materialized.num_states
    3

    >>> # Materialize one full cycle (default)
    >>> filtered = pool.filter(some_predicate).materialize()
    >>> filtered.num_states  # Number of sequences that passed the filter

    >>> # Filtering + materialization with explicit num_seqs
    >>> barcodes = pp.get_kmers(10).filter(some_predicate).materialize(num_seqs=50, seed=42)
    >>> barcodes.num_states  # Exactly 50 valid barcodes
    50
    """
    op = MaterializeOp(
        source_pool=pool,
        num_seqs=num_seqs,
        num_cycles=num_cycles,
        seed=seed,
        discard_null_seqs=discard_null_seqs,
        max_iterations=max_iterations,
        min_acceptance_rate=min_acceptance_rate,
        attempts_per_rate_assessment=attempts_per_rate_assessment,
        name=name,
        prefix=prefix,
        cards=cards,
    )
    return DnaPool(operation=op)
