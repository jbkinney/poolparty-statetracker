"""Fixed operation - create a pool from a fixed transformation of parent sequences."""
from numbers import Real, Integral
from ..types import Pool_type, Union, Optional, Sequence, Callable, RegionType, beartype
from ..operation import Operation
from ..pool import Pool


@beartype
def fixed_operation(
    parent_pools: Sequence[Union[Pool_type, str]],
    seq_from_seqs_fn: Callable[[list[str]], str],
    seq_length_from_pool_lengths_fn: Callable[[Sequence[Union[int, None]]], Union[int, None]],
    region: RegionType = None,
    remove_marker: Optional[bool] = None,
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
        Function that takes sequences and returns the output sequence.
        When region is specified, the first sequence is the region content only.
    seq_length_from_pool_lengths_fn : Callable[[Sequence[Union[int, None]]], Union[int, None]]
        Function that takes pool lengths and returns output sequence length.
        First element is region length (if region specified) or first pool's seq_length.
    region : RegionType, default=None
        Region to apply transformation to. Can be marker name (str), [start, stop], or None.
    remove_marker : Optional[bool], default=None
        If True and region is a marker name, remove the marker tags from output.
        If None, uses Party default ('remove_marker').
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
    op = FixedOp(
        parent_pools=parent_pools,
        seq_from_seqs_fn=seq_from_seqs_fn,
        seq_length_from_pool_lengths_fn=seq_length_from_pool_lengths_fn,
        region=region,
        remove_marker=remove_marker,
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
        parent_pools: Sequence[Union[Pool_type, str]],
        seq_from_seqs_fn: Callable[[list[str]], str],
        seq_length_from_pool_lengths_fn: Callable[[Sequence[Union[int, None]]], Union[int, None]],
        region: RegionType = None,
        remove_marker: Optional[bool] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
        _factory_name: Optional[str] = None,
    ) -> None:
        """Initialize FixedOp."""
        from ..party import get_active_party
        from .from_seq import from_seq
        
        parent_pools = [from_seq(p) if isinstance(p, str) else p for p in parent_pools]
        
        self.seq_from_seqs_fn = seq_from_seqs_fn
        self._seq_length_from_pool_lengths_fn = seq_length_from_pool_lengths_fn
        self._region = region
        
        # Validate region parameter
        Operation._validate_region(region)
        
        # Resolve remove_marker from party default if None
        party = get_active_party()
        if remove_marker is None:
            self._remove_marker = party.get_default('remove_marker', True) if party else True
        else:
            self._remove_marker = remove_marker
        
        # Compute seq_length from pool_lengths
        if region is not None:
            if isinstance(region, str):
                # Marker name - look up registered marker's seq_length
                marker = party.get_marker(region)
                region_length = marker.seq_length
            else:
                # Explicit [start, stop] interval
                region_length = int(region[1]) - int(region[0])
            pool_lengths = [region_length] + [p.seq_length for p in parent_pools[1:]]
        else:
            pool_lengths = [p.seq_length for p in parent_pools]
        
        seq_length = seq_length_from_pool_lengths_fn(pool_lengths)
        
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
        if self._region is not None:
            # Extract region parts
            prefix, region_content, suffix = self._extract_region_parts(
                parent_seqs[0], self._region
            )
            
            # Apply transformation to region content
            # First sequence passed to fn is region content, rest are other parent seqs
            transformed = self.seq_from_seqs_fn([region_content] + parent_seqs[1:])
            
            # Handle marker removal
            if self._remove_marker and isinstance(self._region, str):
                # Get clean prefix/suffix without marker tags
                from ..marker_ops.parsing import parse_marker
                clean_prefix, _, clean_suffix, _ = parse_marker(parent_seqs[0], self._region)
                result = clean_prefix + transformed + clean_suffix
            else:
                result = prefix + transformed + suffix
        else:
            result = self.seq_from_seqs_fn(parent_seqs)
        
        return {'seq_0': result}

    def _get_copy_params(self) -> dict:
        """Return parameters needed to create a copy of this operation."""
        return {
            'parent_pools': self.parent_pools,
            'seq_from_seqs_fn': self.seq_from_seqs_fn,
            'seq_length_from_pool_lengths_fn': self._seq_length_from_pool_lengths_fn,
            'region': self._region,
            'remove_marker': self._remove_marker,
            'name': None,
            'iter_order': self.iter_order,
        }
