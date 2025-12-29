"""
split_counter - Split a counter into multiple counters.
"""
from numbers import Real
from collections.abc import Sequence

from .slice_coop import SliceCoOp


def split_counter(counter, split_spec, names=None):
    """Split a counter into multiple counters.
    
    Creates multiple counters that together cover all states of the parent
    counter. Each resulting counter represents a contiguous slice of the
    parent's states.
    
    Args:
        counter: The Counter to split.
        split_spec: Either:
            - int (>= 2): Number of parts to split into (as evenly as possible)
            - Sequence[float]: Relative proportions for each part (length >= 2)
        names: Optional Sequence[str] of names for resulting counters.
            Must match the number of parts if provided.
    
    Returns:
        Tuple of Counter objects.
    
    Examples:
        A = Counter(10)
        
        # Split into 3 equal parts: sizes [4, 3, 3]
        parts = split_counter(A, 3)
        
        # Split by proportions 1:2:1 -> sizes [3, 5, 2]
        parts = split_counter(A, (1.0, 2.0, 1.0))
        
        # With names
        left, right = split_counter(A, 2, names=['left', 'right'])
    
    Raises:
        TypeError: If counter is not a Counter.
        ValueError: If split_spec is invalid or names has wrong length.
    """
    from ..counter import Counter
    
    if not isinstance(counter, Counter):
        raise TypeError(f"Expected Counter, got {type(counter)}")
    
    num_states = counter.num_states
    
    # Determine the sizes for each part
    if isinstance(split_spec, int):
        # Split into N equal parts
        if split_spec < 2:
            raise ValueError(f"split_spec must be >= 2, got {split_spec}")
        sizes = _compute_equal_sizes(num_states, split_spec)
    elif isinstance(split_spec, Sequence) and not isinstance(split_spec, str):
        # Split by proportions
        if len(split_spec) < 2:
            raise ValueError(f"split_spec sequence must have length >= 2, got {len(split_spec)}")
        if not all(isinstance(p, Real) and p > 0 for p in split_spec):
            raise ValueError("All proportions must be positive numbers")
        sizes = _compute_proportional_sizes(num_states, split_spec)
    else:
        raise TypeError(f"split_spec must be int or Sequence[float], got {type(split_spec)}")
    
    num_parts = len(sizes)
    
    # Validate names
    if names is not None:
        if len(names) != num_parts:
            raise ValueError(f"names has length {len(names)}, but split produces {num_parts} parts")
    
    # Create counters using SliceCoOp
    result = []
    start = 0
    for i, size in enumerate(sizes):
        stop = start + size
        new_counter = Counter(_parents=(counter,), _op=SliceCoOp(start, stop, 1))
        if names is not None:
            new_counter.name = names[i]
        result.append(new_counter)
        start = stop
    
    return tuple(result)


def _compute_equal_sizes(num_states, num_parts):
    """Compute sizes for equal splitting.
    
    Distributes states as evenly as possible. Larger parts come first.
    E.g., 10 states into 3 parts -> [4, 3, 3]
    """
    if num_states < num_parts:
        raise ValueError(
            f"Cannot split {num_states} states into {num_parts} parts "
            f"(each part must have at least 1 state)"
        )
    
    base_size = num_states // num_parts
    remainder = num_states % num_parts
    
    # First 'remainder' parts get base_size + 1, rest get base_size
    sizes = []
    for i in range(num_parts):
        if i < remainder:
            sizes.append(base_size + 1)
        else:
            sizes.append(base_size)
    
    return sizes


def _compute_proportional_sizes(num_states, proportions):
    """Compute sizes based on proportions.
    
    Scales proportions to sum to num_states, rounds, then adjusts
    to ensure exact total.
    """
    num_parts = len(proportions)
    
    if num_states < num_parts:
        raise ValueError(
            f"Cannot split {num_states} states into {num_parts} parts "
            f"(each part must have at least 1 state)"
        )
    
    total_proportion = sum(proportions)
    
    # Compute raw (floating point) sizes
    raw_sizes = [(p / total_proportion) * num_states for p in proportions]
    
    # Round to integers
    sizes = [round(s) for s in raw_sizes]
    
    # Ensure each part has at least 1 state
    for i in range(len(sizes)):
        if sizes[i] < 1:
            sizes[i] = 1
    
    # Adjust to match total
    current_total = sum(sizes)
    diff = num_states - current_total
    
    if diff != 0:
        # Compute fractional remainders (after rounding)
        remainders = [(raw_sizes[i] - sizes[i], i) for i in range(len(sizes))]
        
        if diff > 0:
            # Need to add states - prioritize parts with largest positive remainders
            remainders.sort(reverse=True)
            for j in range(diff):
                idx = remainders[j % len(remainders)][1]
                sizes[idx] += 1
        else:
            # Need to remove states - prioritize parts with largest negative remainders
            # But ensure we don't go below 1
            remainders.sort()
            removed = 0
            for j in range(abs(diff)):
                # Find next part that can be reduced
                for _, idx in remainders:
                    if sizes[idx] > 1:
                        sizes[idx] -= 1
                        removed += 1
                        break
                if removed > j:
                    continue
    
    return sizes

