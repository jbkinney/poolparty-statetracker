"""Score operation - evaluate a function on the sequence and record the result."""

import numpy as np

from ..operation import Operation
from ..pool import Pool
from ..types import Any, Callable, CardsType, Optional, Pool_type, RegionType, Seq, Sequence, Union, beartype
from ..utils.parsing_utils import strip_all_tags


@beartype
def score(
    pool: Union[Pool_type, str],
    fn: Callable[[str], Any],
    card_key: str = "score",
    region: RegionType = None,
    prefix: Optional[str] = None,
    cards: CardsType = None,
    _factory_name: Optional[str] = "score",
) -> Pool:
    """
    Apply a scoring function to sequences and record the result as a design card.

    The sequence passes through unchanged. The function ``fn`` receives the
    tag-stripped molecular content (or the clean region content when ``region``
    is specified) and its return value is stored under the given ``card_key``.

    Compatible with existing utilities such as ``pp.calc_gc``, ``pp.calc_dust``,
    and ``pp.calc_complexity``.

    Parameters
    ----------
    pool : Union[Pool_type, str]
        Parent pool or sequence string.
    fn : Callable[[str], Any]
        Scoring function. Receives a clean (tag-free) sequence string and
        returns a scalar value to record.
    card_key : str, default='score'
        Design card column name for the result.
    region : RegionType, default=None
        Region to score. When specified, ``fn`` receives only the clean
        region content. Can be a marker name (str) or [start, stop].
    prefix : Optional[str], default=None
        Prefix for sequence names.
    cards : list[str] or dict, optional
        Design card keys to include. The available key is the ``card_key``
        value (default ``'score'``).

    Returns
    -------
    Pool
        A Pool with the same sequences, plus a design card recording
        the score when ``cards`` is specified.
    """
    from .from_seq import from_seq

    pool = (
        from_seq(pool, _factory_name=f"{_factory_name}(from_seq)")
        if isinstance(pool, str)
        else pool
    )
    op = ScoreOp(
        parent_pool=pool,
        fn=fn,
        card_key=card_key,
        region=region,
        prefix=prefix,
        cards=cards,
        _factory_name=_factory_name,
    )
    pool_class = type(pool)
    return pool_class(operation=op)


class ScoreOp(Operation):
    """Passthrough operation that records a user-computed score as a design card."""

    factory_name = "score"

    def __init__(
        self,
        parent_pool: Pool,
        fn: Callable[[str], Any],
        card_key: str = "score",
        region: RegionType = None,
        prefix: Optional[str] = None,
        cards: CardsType = None,
        _factory_name: Optional[str] = "score",
    ) -> None:
        if _factory_name is not None:
            self.factory_name = _factory_name

        self._fn = fn
        self._card_key = card_key
        self.design_card_keys = [card_key]

        super().__init__(
            parent_pools=[parent_pool],
            num_states=1,
            mode="fixed",
            seq_length=parent_pool.seq_length,
            prefix=prefix,
            region=region,
            cards=cards,
        )

    def _compute_core(
        self,
        parents: list[Seq],
        rng: np.random.Generator | None = None,
    ) -> tuple[Seq, dict]:
        """Evaluate fn on the clean sequence and return unchanged Seq with score card."""
        parent_seq = parents[0]
        seq_str = parent_seq.string
        clean_seq = strip_all_tags(seq_str) if '<' in seq_str else seq_str
        result = self._fn(clean_seq)
        return parent_seq, {self._card_key: result}
