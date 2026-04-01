"""GetKmers operation - generate DNA k-mers."""

from numbers import Real

import numpy as np

from ..dna_pool import DnaPool
from ..operation import Operation
from ..party import get_active_party
from ..pool import Pool
from ..types import (
    CardsType,
    Integral,
    Literal,
    ModeType,
    Optional,
    Pool_type,
    RegionType,
    Seq,
    Union,
    beartype,
)
from ..utils import dna_utils
from ..utils.dna_seq import DnaSeq


@beartype
def get_kmers(
    length: Integral,
    pool: Optional[Union[Pool, str]] = None,
    region: RegionType = None,
    style: Optional[str] = None,
    case: Literal["lower", "upper"] = "upper",
    prefix: Optional[str] = None,
    mode: ModeType = "random",
    num_states: Optional[Integral] = None,
    iter_order: Optional[Real] = None,
    cards: CardsType = None,
) -> DnaPool:
    """Create a Pool that generates DNA k-mers (all possible sequences of length k).

    Must be called within a Party context.

    Parameters
    ----------
    pool : Optional[Union[Pool, str]], default=None
        Pool or sequence. If provided with region, generated k-mer
        replaces the region content.
    region : RegionType, default=None
        Region to replace in pool. Can be a marker name or [start, stop] interval.
        Required if pool is provided.
    length : int
        Length of k-mers to generate.
    case : Literal['lower', 'upper'], default='upper'
        Case of output k-mers: 'upper' for uppercase, 'lower' for lowercase.
    prefix : Optional[str], default=None
        Prefix for sequence names in the resulting Pool.
    mode : ModeType, default='random'
        Sequence selection mode: 'sequential' or 'random'.
    num_states : Optional[int], default=None
        Number of states. In sequential mode, overrides the computed count
        (cycling if greater, clipping if less). In random mode, if None
        defaults to 1 (pure random sampling).
    style : Optional[str], default=None
        Style to apply to generated k-mers (e.g., 'red', 'blue bold').
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.
    cards : list[str] or dict, optional
        Design card keys to include. Available keys: ``'kmer_index'``, ``'kmer'``.

    Returns
    -------
    Pool_type
        A Pool whose states yield DNA k-mers of the specified length.

    Raises
    ------
    RuntimeError
        If called outside of a Party context.
    ValueError
        If pool is provided without region.
    """
    from ..fixed_ops.from_seq import from_seq

    pool_obj = from_seq(pool) if isinstance(pool, str) else pool
    op = GetKmersOp(
        length,
        pool=pool_obj,
        region=region,
        style=style,
        case=case,
        prefix=prefix,
        mode=mode,
        num_states=num_states,
        name=None,
        iter_order=iter_order,
        cards=cards,
    )
    pool = DnaPool(operation=op)
    return pool


class GetKmersOp(Operation):
    """Generate DNA k-mers."""

    factory_name = "get_kmers"
    design_card_keys = ["kmer_index", "kmer"]

    def __init__(
        self,
        length: int,
        pool: Optional[Pool] = None,
        region: RegionType = None,
        style: Optional[str] = None,
        case: Literal["lower", "upper"] = "upper",
        prefix: Optional[str] = None,
        mode: ModeType = "random",
        num_states: Optional[Integral] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
        cards: CardsType = None,
    ) -> None:
        """Initialize GetKmersOp."""
        party = get_active_party()
        if party is None:
            raise RuntimeError(
                "get_kmers requires an active Party context. "
                "Use 'with pp.Party() as party:' to create one."
            )

        # Validate bg_pool/region combination
        if pool is not None and region is None:
            raise ValueError(
                "region is required when pool is provided. "
                "Specify which region of pool to replace with the generated k-mer."
            )

        if length < 1:
            raise ValueError(f"length must be >= 1, got {length}")

        self._style = style

        self.length = length
        self.case = case
        self.alpha_size = len(dna_utils.BASES)
        self._total_kmers = self.alpha_size**length

        # Determine num_states based on mode
        natural_num_states = None
        if mode == "sequential":
            natural_num_states = self._total_kmers
            if num_states is None:
                num_states = natural_num_states
        elif mode == "random":
            # num_states stays as provided (or None for pure random mode)
            pass
        else:
            num_states = 1

        parent_pools = [pool] if pool is not None else []

        # Compute seq_length: kmer length when standalone, adjusted when replacing region
        if pool is None:
            seq_length = length
        else:
            parent_seq_length = pool.seq_length
            # Determine region length
            if isinstance(region, str):
                # Region name - get length from registered region
                try:
                    region_obj = party.get_region_by_name(region)
                    region_length = region_obj.seq_length
                except (ValueError, KeyError):
                    region_length = None
            else:
                # Interval [start, stop]
                region_length = region[1] - region[0] if region is not None else None

            # Compute output seq_length (None if any component is None)
            if parent_seq_length is None or region_length is None:
                seq_length = None
            else:
                seq_length = parent_seq_length - region_length + length

        super().__init__(
            parent_pools=parent_pools,
            num_states=num_states,
            mode=mode,
            seq_length=seq_length,
            name=name,
            iter_order=iter_order,
            prefix=prefix,
            region=region,
            _natural_num_states=natural_num_states,
            cards=cards,
        )

    def _value_to_kmer(self, value: int) -> str:
        """Convert a value index to a k-mer string."""
        result = []
        remaining = value
        for _ in range(self.length):
            result.append(dna_utils.BASES[remaining % self.alpha_size])
            remaining //= self.alpha_size
        return "".join(reversed(result))

    def _random_kmer(self, rng: np.random.Generator) -> str:
        """Generate a random k-mer."""
        indices = rng.integers(0, self.alpha_size, size=self.length)
        return "".join(dna_utils.BASES[i] for i in indices)

    def _compute_core(
        self,
        parents: list[Seq],
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[Seq, dict]:
        """Return Seq and design card."""
        if self.mode == "random":
            if rng is None:
                raise RuntimeError(
                    f"{self.mode.capitalize()} mode requires RNG - use Party.generate(seed=...)"
                )
            kmer = self._random_kmer(rng)
            kmer_index = None
        else:
            # Use state 0 when inactive (state is None)
            idx = self.state.value
            idx = 0 if idx is None else idx
            # Apply modulo with total k-mers for cycling
            idx = idx % self._total_kmers
            kmer = self._value_to_kmer(idx)
            kmer_index = idx

        # Apply case transformation
        if self.case == "lower":
            kmer = kmer.lower()

        # Apply style to all positions if specified
        from ..utils.style_utils import SeqStyle, styles_suppressed

        if styles_suppressed():
            output_seq = DnaSeq(kmer, None)
        else:
            output_style = SeqStyle.full(len(kmer), self._style)
            output_seq = DnaSeq(kmer, output_style)

        return output_seq, {
            "kmer_index": kmer_index,
            "kmer": kmer,
        }
