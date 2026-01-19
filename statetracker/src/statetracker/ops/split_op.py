"""split_state - Split a state into multiple states."""
from ..imports import beartype, Sequence, Optional, Real, Integral, Union, State_type
from .slice_op import SliceOp


@beartype
def split(state: State_type, split_spec: Union[Integral, Sequence[Real]], names: Optional[Sequence[str]] = None):
    """
    Split a State into multiple sub-States according to equal or proportional partitioning.

    Parameters
    ----------
    state : State_type
        The State object to be split.
    split_spec : Union[Integral, Sequence[Real]]
        If an integer N, split into N roughly equal parts. If a sequence of proportions, split according to these proportions (sequence length = number of parts).
    names : Optional[Sequence[str]], default=None
        Optional sequence of names for each resulting sub-State. If provided, must match the number of parts.

    Returns
    -------
    Tuple[State_type, ...]
        Tuple of new State objects corresponding to each partition of the original values.
    """
    from ..state import State
    num_values = state.num_values
    if isinstance(split_spec, Integral):
        if split_spec < 2:
            raise ValueError(f"split_spec must be >= 2, got {split_spec}")
        sizes = _compute_equal_sizes(num_values, split_spec)
    else:
        if len(split_spec) < 2:
            raise ValueError(f"split_spec sequence must have length >= 2, got {len(split_spec)}")
        if not all(p > 0 for p in split_spec):
            raise ValueError("All proportions must be positive numbers")
        sizes = _compute_proportional_sizes(num_values, split_spec)
    num_parts = len(sizes)
    if names is not None:
        if len(names) != num_parts:
            raise ValueError(f"names has length {len(names)}, but split produces {num_parts} parts")
    result = []
    start = 0
    for i, size in enumerate(sizes):
        stop = start + size
        new_state = State(_parents=(state,), _op=SliceOp(start, stop, 1))
        if names is not None:
            new_state.name = names[i]
        result.append(new_state)
        start = stop
    return tuple(result)


def _compute_equal_sizes(num_values, num_parts):
    """Compute sizes for equal splitting."""
    if num_values < num_parts:
        raise ValueError(
            f"Cannot split {num_values} values into {num_parts} parts "
            f"(each part must have at least 1 value)"
        )
    base_size = num_values // num_parts
    remainder = num_values % num_parts
    sizes = []
    for i in range(num_parts):
        if i < remainder:
            sizes.append(base_size + 1)
        else:
            sizes.append(base_size)
    return sizes


def _compute_proportional_sizes(num_values, proportions):
    """Compute sizes based on proportions."""
    num_parts = len(proportions)
    if num_values < num_parts:
        raise ValueError(
            f"Cannot split {num_values} values into {num_parts} parts "
            f"(each part must have at least 1 value)"
        )
    total_proportion = sum(proportions)
    raw_sizes = [(p / total_proportion) * num_values for p in proportions]
    sizes = [round(s) for s in raw_sizes]
    for i in range(len(sizes)):
        if sizes[i] < 1:
            sizes[i] = 1
    current_total = sum(sizes)
    diff = num_values - current_total
    if diff != 0:
        remainders = [(raw_sizes[i] - sizes[i], i) for i in range(len(sizes))]
        if diff > 0:
            remainders.sort(reverse=True)
            for j in range(diff):
                idx = remainders[j % len(remainders)][1]
                sizes[idx] += 1
        else:
            remainders.sort()
            removed = 0
            for j in range(abs(diff)):
                for _, idx in remainders:
                    if sizes[idx] > 1:
                        sizes[idx] -= 1
                        removed += 1
                        break
                if removed > j:
                    continue
    return sizes
