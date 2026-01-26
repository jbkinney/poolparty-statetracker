"""Fixed operation - create a pool from a fixed transformation of parent sequences."""
from numbers import Real, Integral
from ..types import Pool_type, Union, Optional, Sequence, Callable, RegionType, beartype, StyleList
from ..operation import Operation
from ..pool import Pool


@beartype
def fixed_operation(
    parent_pools: Sequence[Union[Pool_type, str]],
    seq_from_seqs_fn: Callable[[list[str]], str],
    seq_length_from_pool_lengths_fn: Callable[[Sequence[Union[int, None]]], Union[int, None]],
    region: RegionType = None,
    remove_tags: Optional[bool] = None,
    iter_order: Optional[Real] = None,
    _factory_name: Optional[str] = None,
    _pass_through_styles: bool = True,
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
    remove_tags : Optional[bool], default=None
        If True and region is a marker name, remove the marker tags from output.
        If None, uses Party default ('remove_tags').
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.
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
        remove_tags=remove_tags,
        name=None,
        iter_order=iter_order,
        _factory_name=_factory_name,
        _pass_through_styles=_pass_through_styles,
    )
    pool = Pool(operation=op)
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
        remove_tags: Optional[bool] = None,
        spacer_str: str = '',
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
        _factory_name: Optional[str] = None,
        _pass_through_styles: bool = True,
    ) -> None:
        """Initialize FixedOp."""
        from ..party import get_active_party
        from .from_seq import from_seq
        
        parent_pools = [from_seq(p) if isinstance(p, str) else p for p in parent_pools]
        
        self.seq_from_seqs_fn = seq_from_seqs_fn
        self._seq_length_from_pool_lengths_fn = seq_length_from_pool_lengths_fn
        self._pass_through_styles = _pass_through_styles
        
        # Compute seq_length from pool_lengths
        party = get_active_party()
        if region is not None:
            if isinstance(region, str):
                # Region name - look up registered region's seq_length
                region_obj = party.get_region(region)
                region_length = region_obj.seq_length
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
            num_values=1,
            mode='fixed',
            seq_length=seq_length,
            name=name,
            iter_order=iter_order,
            region=region,
            remove_tags=remove_tags,
        )

    def compute(
        self,
        parent_seqs: list[str],
        rng=None,
        parent_styles: list[StyleList] | None = None,
    ) -> dict:
        """Compute output sequence using the user-defined function.
        
        Note: Region handling is done by the base class wrapper methods.
        parent_seqs[0] is the region content when region is specified.
        """
        result = self.seq_from_seqs_fn(parent_seqs)
        # Pass through parent styles only if _pass_through_styles is True
        # When doing content replacement (e.g., from_seq with region), styles
        # from the original content should not apply to the new content
        output_styles: StyleList = []
        if self._pass_through_styles and parent_styles and len(parent_styles) > 0:
            output_styles.extend(parent_styles[0])
        return {'seq': result, 'style': output_styles}

    def _get_copy_params(self) -> dict:
        """Return parameters needed to create a copy of this operation."""
        return {
            'parent_pools': self.parent_pools,
            'seq_from_seqs_fn': self.seq_from_seqs_fn,
            'seq_length_from_pool_lengths_fn': self._seq_length_from_pool_lengths_fn,
            'region': self._region,
            'remove_tags': self._remove_tags,
            'name': None,
            'iter_order': self.iter_order,
            '_pass_through_styles': self._pass_through_styles,
        }
