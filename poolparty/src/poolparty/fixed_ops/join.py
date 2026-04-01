"""Join operation - join multiple sequences together."""

from numbers import Real
from typing import TypeVar

from ..pool import Pool
from ..types import Optional, Pool_type, Sequence, Union, beartype

T = TypeVar("T", bound=Pool)
from ..utils.style_utils import SeqStyle


def _make_join_style_combiner(spacer_str: str):
    """Create a style combiner function for join that handles spacers.

    Returns a function that combines parent styles with proper position offsets,
    inserting empty SeqStyle objects for spacer strings between parents.
    """

    def combine_styles(parent_styles, result_length: int):
        """Combine parent styles with spacer handling."""
        # If any parent has None style (styles suppressed), return None
        if any(s is None for s in parent_styles):
            return None

        # Build list of styles including spacers
        styles_with_spacers = []
        spacer_len = len(spacer_str)

        for i, parent_style in enumerate(parent_styles):
            if i > 0 and spacer_len > 0:
                # Insert empty style for spacer between parents
                styles_with_spacers.append(SeqStyle.empty(spacer_len))
            styles_with_spacers.append(parent_style)

        # Join all styles with proper position offsets
        return SeqStyle.join(styles_with_spacers)

    return combine_styles


@beartype
def join(
    pools: Sequence[Union[T, str]],
    spacer_str: str = "",
    iter_order: Optional[Real] = None,
    prefix: Optional[str] = None,
    style: Optional[str] = None,
    _factory_name: Optional[str] = None,
) -> T:
    """
    Concatenate multiple Pools or string sequences into a single Pool.

    Parameters
    ----------
    pools : Sequence[Union[Pool_type, str]]
        List of Pool objects and/or strings to be joined in order.
        Any provided string is automatically converted to a constant Pool.
    spacer_str : str, default=''
        String to insert between joined sequences.
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.
    prefix : Optional[str], default=None
        Prefix for sequence names in the resulting Pool.
    style : Optional[str], default=None
        Style to apply to the resulting concatenated sequences (e.g., 'red', 'blue bold').

    Returns
    -------
    Pool_type
        A Pool whose states yield joined sequences from the specified inputs.
    """
    from .fixed import fixed_operation

    if len(pools) == 0:
        raise ValueError("join requires at least one pool.")

    def seq_length_from_pool_lengths_fn(lengths: Sequence[Optional[int]]) -> Optional[int]:
        if all(L is not None for L in lengths):
            n_spacers = max(0, len(lengths) - 1)
            return sum(lengths) + len(spacer_str) * n_spacers
        return None

    # Create style combiner that properly joins styles from all parents
    style_combiner = _make_join_style_combiner(spacer_str)

    result_pool = fixed_operation(
        parent_pools=pools,
        seq_from_seqs_fn=lambda seqs: spacer_str.join(seqs),
        seq_length_from_pool_lengths_fn=seq_length_from_pool_lengths_fn,
        iter_order=iter_order,
        prefix=prefix,
        _factory_name=_factory_name if _factory_name is not None else "join",
        _style_combiner_fn=style_combiner,
    )

    # Apply style if specified
    if style is not None:
        from .stylize import stylize

        result_pool = stylize(result_pool, style=style)

    return result_pool
