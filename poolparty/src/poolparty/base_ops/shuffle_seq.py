"""SeqShuffle operation - shuffle characters within a sequence region."""

from numbers import Real

import numpy as np

from ..operation import Operation
from ..pool import Pool
from ..types import CardsType, Integral, Literal, ModeType, Optional, Pool_type, RegionType, Seq, Union, beartype
from ..utils.dna_seq import DnaSeq


@beartype
def shuffle_seq(
    pool: Union[Pool_type, str],
    region: RegionType = None,
    shuffle_type: Literal["mono", "dinuc"] = "mono",
    prefix: Optional[str] = None,
    mode: ModeType = "random",
    num_states: Optional[Integral] = None,
    iter_order: Optional[Real] = None,
    _remove_tags: bool = False,
    style: Optional[str] = None,
    cards: CardsType = None,
    _factory_name: Optional[str] = None,
) -> Pool:
    """
    Create a Pool that shuffles characters within a specified region.

    Parameters
    ----------
    pool : Pool_type
        Parent pool or sequence to shuffle.
    region : RegionType, default=None
        Region to shuffle. Can be a marker name (str), explicit interval [start, stop],
        or None to shuffle entire sequence.
    shuffle_type : Literal["mono", "dinuc"], default="mono"
        Type of shuffle to perform:
        - ``"mono"``: random permutation preserving mononucleotide composition.
        - ``"dinuc"``: Euler-path shuffle preserving dinucleotide frequencies.
          The first and last characters are always fixed (mathematical constraint
          of the Euler path algorithm).
    mode : ModeType, default='random'
        Shuffle mode: 'random'. Sequential is not supported.
    num_states : Optional[int], default=None
        Number of states for random mode. If None, defaults to 1 (pure random sampling).
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.
    prefix : Optional[str], default=None
        Prefix for sequence names in the resulting Pool.
    style : Optional[str], default=None
        Style to apply to shuffled characters (e.g., 'red', 'blue bold').
    cards : list[str] or dict, optional
        Design card keys to include. Available keys: ``'permutation'``.

    Returns
    -------
    Pool
        A Pool that yields shuffled sequences.
    """
    from ..fixed_ops.from_seq import from_seq

    pool_obj = from_seq(pool) if isinstance(pool, str) else pool
    op = SeqShuffleOp(
        parent_pool=pool_obj,
        region=region,
        shuffle_type=shuffle_type,
        prefix=prefix,
        mode=mode,
        num_states=num_states,
        name=None,
        iter_order=iter_order,
        _remove_tags=_remove_tags,
        style=style,
        cards=cards,
        _factory_name=_factory_name,
    )
    # Preserve the pool type from the input
    pool_class = type(pool_obj)
    result_pool = pool_class(operation=op)
    return result_pool


class SeqShuffleOp(Operation):
    """Randomly shuffle characters within a region of the parent sequence."""

    factory_name = "shuffle_seq"
    design_card_keys = ["permutation"]

    def __init__(
        self,
        parent_pool: Pool,
        region: RegionType = None,
        shuffle_type: Literal["mono", "dinuc"] = "mono",
        spacer_str: str = "",
        prefix: Optional[str] = None,
        mode: ModeType = "random",
        num_states: Optional[Integral] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
        _remove_tags: bool = False,
        style: Optional[str] = None,
        cards: CardsType = None,
        _factory_name: Optional[str] = None,
    ) -> None:
        """Initialize SeqShuffleOp."""

        if mode != "random":
            raise ValueError(
                f"mode={mode!r} is not supported for shuffle_seq. Only mode='random' is allowed."
            )

        # Set factory_name if provided
        if _factory_name is not None:
            self.factory_name = _factory_name

        self._shuffle_type = shuffle_type
        self._style = style

        # Determine num_states
        if mode == "random":
            # num_states stays None for pure random mode
            pass
        else:
            num_states = 1
        super().__init__(
            parent_pools=[parent_pool],
            num_states=num_states,
            mode=mode,
            seq_length=parent_pool.seq_length,
            name=name,
            iter_order=iter_order,
            prefix=prefix,
            region=region,
            remove_tags=_remove_tags,
            cards=cards,
        )

    def _compute_core(
        self,
        parents: list[Seq],
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[Seq, dict]:
        """Return shuffled Seq and design card.

        Note: Region handling is done by base class compute() method.
        parents[0] is the region content when region is specified.
        """
        if self.mode == "random":
            if rng is None:
                raise RuntimeError(
                    f"{self.mode.capitalize()} mode requires RNG - use Party.generate(seed=...)"
                )
        else:
            raise RuntimeError(f"Unsupported mode {self.mode!r}")

        seq = parents[0].string

        # Cache molecular positions, position array, and sequence array for repeated sequences
        if not hasattr(self, "_mol_pos_cache"):
            self._mol_pos_cache = {}
            self._pos_arr_cache = {}
            self._seq_arr_cache = {}

        if seq not in self._mol_pos_cache:
            self._mol_pos_cache[seq] = self._get_molecular_positions(seq)
            self._pos_arr_cache[seq] = np.array(self._mol_pos_cache[seq], dtype=np.intp)
            self._seq_arr_cache[seq] = np.array(list(seq), dtype="U1")
        molecular_positions = self._mol_pos_cache[seq]
        pos_arr = self._pos_arr_cache[seq]
        num_molecular = len(molecular_positions)

        if num_molecular == 0:
            permutation = tuple()
            shuffled_seq = seq
        elif self._shuffle_type == "dinuc":
            from collections import defaultdict

            from ..utils.shuffle_utils import dinucleotide_shuffle

            mol_str = "".join(self._seq_arr_cache[seq][pos_arr])
            shuffled_mol = dinucleotide_shuffle(mol_str, rng)

            seq_arr = self._seq_arr_cache[seq].copy()
            seq_arr[pos_arr] = list(shuffled_mol)
            shuffled_seq = "".join(seq_arr)

            # Compute permutation: greedily match each output char
            # to the first available input position of that character
            char_positions: dict[str, list[int]] = defaultdict(list)
            for i, c in enumerate(mol_str):
                char_positions[c].append(i)
            order = np.array(
                [char_positions[c].pop(0) for c in shuffled_mol], dtype=np.intp
            )
            permutation = tuple(np.argsort(order).tolist())
        else:
            order = rng.permutation(num_molecular)
            # Use argsort to compute inverse permutation (vectorized)
            permutation = tuple(np.argsort(order).tolist())

            # Vectorized character shuffling using cached numpy arrays
            seq_arr = self._seq_arr_cache[seq].copy()
            molecular_chars = seq_arr[pos_arr]
            seq_arr[pos_arr] = molecular_chars[order]
            shuffled_seq = "".join(seq_arr)

        # Pass through parent styles and add styling to shuffled characters if requested
        output_style = parents[0].style
        if output_style is not None and self._style and molecular_positions:
            output_style = output_style.add_style(
                self._style, np.array(molecular_positions, dtype=np.int64)
            )

        output_seq = DnaSeq(shuffled_seq, output_style)

        return output_seq, {
            "permutation": permutation,
        }
