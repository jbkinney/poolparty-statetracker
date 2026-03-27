"""Passthrough operation - pass sequence unchanged, optionally contributing custom names."""

from numbers import Real

from ..operation import Operation
from ..pool import Pool
from ..types import Callable, Optional, beartype


@beartype
def passthrough(
    pool: Pool,
    _name_fn: Optional[Callable[[], list[str]]] = None,
    iter_order: Optional[Real] = None,
    _factory_name: Optional[str] = None,
) -> Pool:
    """Pass through sequence unchanged, optionally contributing custom names.

    Parameters
    ----------
    pool : Pool
        Input pool.
    _name_fn : Optional[Callable], default=None
        Function that returns list of name elements for custom naming.
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.
    _factory_name : Optional[str], default=None
        Sets default name of the resulting operation.

    Returns
    -------
    Pool
        A Pool containing the same sequences as input.
    """
    op = PassthroughOp(pool, _name_fn=_name_fn, iter_order=iter_order, _factory_name=_factory_name)
    pool_class = type(pool)
    return pool_class(operation=op)


class PassthroughOp(Operation):
    """Pass through sequence unchanged, optionally contributing custom names."""

    factory_name = "passthrough"
    design_card_keys = []

    def __init__(
        self,
        parent_pool: Pool,
        _name_fn: Optional[Callable[[], list[str]]] = None,
        iter_order: Optional[Real] = None,
        _factory_name: Optional[str] = None,
    ) -> None:
        if _factory_name is not None:
            self.factory_name = _factory_name
        self._name_fn = _name_fn
        super().__init__(
            parent_pools=[parent_pool],
            mode="fixed",
            seq_length=parent_pool.seq_length,
            iter_order=iter_order,
        )

    def _compute_core(self, parents, rng=None):
        """Pass through the parent sequence unchanged."""
        return parents[0], {}

    def compute_name_contributions(self, global_state=None, max_global_state=None) -> list[str]:
        """Return custom name contributions if _name_fn is set."""
        if self._name_fn is not None:
            return self._name_fn()
        return super().compute_name_contributions(global_state, max_global_state)
