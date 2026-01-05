"""Fixed operation - create a pool from a fixed transformation of parent sequences."""
from numbers import Real
from ..types import Pool_type, Union, Optional, Sequence, Callable, beartype
from ..operation import Operation
from ..pool import Pool


@beartype
def fixed_operation(
    parents: Sequence[Union[Pool_type, str]],
    seq_from_seqs_fn: Callable[[list[str]], str],
    seq_length_from_pools_fn: Callable[[Sequence[Pool_type]], Optional[int]],
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None,
    _factory_name: Optional[str] = None,
) -> Pool:
    """
    Create a Pool from a fixed transformation of parent sequences.

    Parameters
    ----------
    parents : Sequence[Union[Pool_type, str]]
        Parent pools or strings (strings are auto-converted to pools via from_seq).
    seq_from_seqs_fn : Callable[[list[str]], str]
        Function that takes parent sequences and returns the output sequence.
    seq_length_from_pools_fn : Callable[[Sequence[Pool_type]], Optional[int]]
        Function that takes parent pools and returns the output sequence length
        (or None if variable/unknown).
    name : Optional[str], default=None
        Name for the resulting Pool.
    op_name : Optional[str], default=None
        Name for the underlying Operation.
    iter_order : Optional[Real], default=None
        Iteration order priority for the resulting Pool.
    op_iter_order : Optional[Real], default=None
        Iteration order priority for the underlying Operation.
    _factory_name: Optional[str], default=None
        Overrides FactoryOp.factory_name in setting the default operation name.

    Returns
    -------
    Pool
        A Pool yielding sequences computed from parent sequences.
    """
    from .from_seq import from_seq
    parent_pools = [from_seq(p) if isinstance(p, str) else p for p in parents]
    op = FixedOp(
        parent_pools=parent_pools,
        seq_from_seqs_fn=seq_from_seqs_fn,
        seq_length_from_pools_fn=seq_length_from_pools_fn,
        name=op_name,
        iter_order=op_iter_order,
        _factory_name=_factory_name,
    )
    pool = Pool(operation=op, name=name, iter_order=iter_order)
    return pool


@beartype
class FixedOp(Operation):
    """Fixed operation that applies a user-defined function to parent sequences."""
    factory_name = "fixed"
    design_card_keys: Sequence[str] = []

    def __init__(
        self,
        parent_pools: Sequence[Pool_type],
        seq_from_seqs_fn: Callable[[list[str]], str],
        seq_length_from_pools_fn: Callable[[Sequence[Pool_type]], Optional[int]],
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
        _factory_name: Optional[str] = None,
    ) -> None:
        """Initialize FixedOp."""
        self.seq_from_seqs_fn = seq_from_seqs_fn
        self.seq_length_from_pools_fn = seq_length_from_pools_fn
        seq_length = seq_length_from_pools_fn(parent_pools)
        if _factory_name is not None:
            self.factory_name = _factory_name
        super().__init__(
            parent_pools=list(parent_pools),
            num_states=1,
            mode='fixed',
            seq_length=seq_length,
            name=name,
            iter_order=iter_order,
        )

    def compute_design_card(self, parent_seqs: list[str], rng=None) -> dict:
        """Return empty design card (no design decisions)."""
        return {}

    def compute_seq_from_card(self, parent_seqs: list[str], card: dict) -> dict:
        """Compute output sequence using the user-defined function."""
        seq = self.seq_from_seqs_fn(parent_seqs)
        return {'seq_0': seq}

    def _get_copy_params(self) -> dict:
        """Return parameters needed to create a copy of this operation."""
        return {
            'parent_pools': self.parent_pools,
            'seq_from_seqs_fn': self.seq_from_seqs_fn,
            'seq_length_from_pools_fn': self.seq_length_from_pools_fn,
            'name': None,
            'iter_order': self.iter_order,
        }
