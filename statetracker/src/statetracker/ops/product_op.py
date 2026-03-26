"""ProductOp - Cartesian product of N states."""

from ..imports import Literal, Optional, Sequence, State_type, beartype, math
from ..operation import Operation

# Module-level flag for product ordering mode
_product_order_mode: str = "first_state_slowest"


def set_product_order_mode(mode: Literal["first_state_fastest", "first_state_slowest"]) -> None:
    """Set the global ordering mode for ordered_product()."""
    global _product_order_mode
    if mode not in ("first_state_fastest", "first_state_slowest"):
        raise ValueError(
            f"mode must be 'first_state_fastest' or 'first_state_slowest', got {mode!r}"
        )
    _product_order_mode = mode


def get_product_order_mode() -> str:
    """Get the current global ordering mode for ordered_product()."""
    return _product_order_mode


def _collect_product_bases(state: State_type) -> list:
    """
    Recursively collect base states, flattening through ProductOp only.

    This enables proper deduplication when the same state appears at different
    levels of a nested product hierarchy (diamond pattern).

    Parameters
    ----------
    state : State_type
        A state to collect bases from.

    Returns
    -------
    list
        List of base states. For ProductOp states, this recursively collects
        from parents. For leaf states or non-product operations, returns [state].
    """
    if not state._parents:
        # Leaf state (no parents)
        return [state]
    elif isinstance(state._op, ProductOp):
        # Product state - recurse into parents
        bases = []
        for parent in state._parents:
            bases.extend(_collect_product_bases(parent))
        return bases
    else:
        # Other operation (stack, slice, etc.) - treat as atomic
        return [state]


def ordered_product(states: Sequence[State_type], name: Optional[str] = None):
    """
    Create a product State from the provided states, removing duplicates and
    automatically imposing an order based on state.iter_order and state.id.

    This function recursively flattens nested product states before deduplication,
    which handles diamond patterns where the same state appears both as a direct
    parent and as an ancestor through another parent.

    Parameters
    ----------
    states : Sequence[State_type]
        Sequence of parent States to combine into the product. Duplicates are removed
        and order is determined by (iter_order, id).
    name : Optional[str], default=None
        Name for the resulting product State.

    Returns
    -------
    State_type
        A State representing the ordered, uniquified cartesian product of the input states.

    Notes
    -----
    Product is associative, so nested products are flattened:
    ``ordered_product([A*B, C, D*A])`` becomes ``ordered_product([A, B, C, D])``

    Non-product operations (like stack, slice) are NOT flattened:
    ``ordered_product([stack(A,B), C])`` keeps ``stack(A,B)`` as an atomic unit.
    """
    from ..state import State

    if len(states) == 0:
        return State(1, name=name)

    # Recursively collect bases, flattening through ProductOp only
    base_states = []
    for s in states:
        base_states.extend(_collect_product_bases(s))

    # Deduplicate by sync group: synced states represent a single
    # dimension of variation and must not multiply in the product.
    seen_groups: set = set()
    unique_states = []
    for s in base_states:
        gid = id(s._synced_group)
        if gid not in seen_groups:
            seen_groups.add(gid)
            unique_states.append(s)
    id_sign = -1 if _product_order_mode == "first_state_slowest" else 1
    ordered_states = sorted(unique_states, key=lambda s: (s._iter_order, id_sign * s._id))

    return State(_parents=ordered_states, _op=ProductOp(), name=name)


@beartype
def product(states: Sequence[State_type], name: Optional[str] = None):
    """
    Create a State representing the cartesian product of the provided States.

    Parameters
    ----------
    states : Sequence[State_type]
        Sequence of parent States to combine into a product State. No duplicates allowed.
    name : Optional[str], default=None
        Optional name for the resulting product State.

    Returns
    -------
    State_type
        A State whose values index the cartesian product of the input states' values.
    """
    from ..state import State

    if len(states) != len(set(states)):
        raise ValueError("product() does not allow duplicate states")
    if len(states) == 0:
        result = State(1, name=name)
    else:
        result = State(_parents=states, _op=ProductOp(), name=name)
    return result


@beartype
class ProductOp(Operation):
    """Cartesian product of N states."""

    def compute_num_states(self, parent_num_values: Sequence):
        # If ALL parents are fixed (None), result is fixed (None)
        # Otherwise, treat None as 1 in the product
        non_none = [n for n in parent_num_values if n is not None]
        if not non_none:
            return None  # All fixed -> fixed
        return math.prod(non_none)

    def decompose(self, value, parent_num_values):
        if value is None:
            return tuple(None for _ in parent_num_values)
        result = []
        for n in parent_num_values:
            if n is None:
                # Fixed state: always 0 when active
                result.append(0)
            else:
                result.append(value % n)
                value //= n
        return tuple(result)
