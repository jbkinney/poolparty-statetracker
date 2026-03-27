"""AddPrefix operation - add a prefix to sequence names without modifying sequences."""

from numbers import Real

from ..operation import Operation
from ..pool import Pool
from ..types import Optional, Seq, beartype


@beartype
def add_prefix(
    pool: Pool,
    prefix: str,
    iter_order: Optional[Real] = None,
) -> Pool:
    """
    Add a prefix to sequence names without modifying the sequences.

    This operation passes sequences through unchanged while contributing
    a prefix label to the sequence name.

    Parameters
    ----------
    pool : Pool
        Input pool.
    prefix : str
        Prefix to add to sequence names.
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.

    Returns
    -------
    Pool
        A Pool with the same sequences but updated names.

    Examples
    --------
    >>> with pp.Party():
    ...     pool = pp.from_seq("ACGT").mutagenize(prefix="mut").add_prefix("final")
    ...     # Names will be: mut_0.final, mut_1.final, ...
    """
    op = AddPrefixOp(pool, prefix=prefix, iter_order=iter_order)
    pool_class = type(pool)
    return pool_class(operation=op)


class AddPrefixOp(Operation):
    """Add a prefix to sequence names without modifying sequences."""

    factory_name = "add_prefix"
    design_card_keys = []

    def __init__(
        self,
        parent_pool: Pool,
        prefix: str,
        iter_order: Optional[Real] = None,
    ) -> None:
        """Initialize AddPrefixOp."""
        super().__init__(
            parent_pools=[parent_pool],
            mode="fixed",
            seq_length=parent_pool.seq_length,
            prefix=prefix,
            iter_order=iter_order,
        )

    def _compute_core(self, parents: list[Seq], rng=None) -> tuple[Seq, dict]:
        """Pass through the parent sequence unchanged."""
        return parents[0], {}
