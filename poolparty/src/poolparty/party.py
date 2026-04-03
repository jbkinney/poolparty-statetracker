"""Party class - context manager for building and executing sequence libraries."""

import logging

import statetracker as st

from .codon_table import CodonTable
from .region import OrfRegion, Region
from .types import Operation_type, Optional, Pool_type, Union, beartype
from .utils import dna_utils

logger = logging.getLogger(__name__)

_active_party: Optional["Party"] = None
_default_party: Optional["Party"] = None


@beartype
def configure_logging(
    level: str = "WARNING",
    format: str = "%(levelname)s - %(name)s - %(message)s",
    handler: Optional[logging.Handler] = None,
) -> None:
    """Configure logging for poolparty and statetracker.

    Parameters
    ----------
    level : str
        Logging level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
    format : str
        Log message format string.
    handler : Optional[logging.Handler]
        Custom handler (defaults to StreamHandler if None).
    """
    for logger_name in ("poolparty", "statetracker"):
        pkg_logger = logging.getLogger(logger_name)
        pkg_logger.setLevel(getattr(logging, level.upper()))

        if handler is None:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(logging.Formatter(format))
            handler_to_add = stream_handler
        else:
            handler_to_add = handler

        # Replace NullHandler with real handler
        pkg_logger.handlers.clear()
        pkg_logger.addHandler(handler_to_add)

        # Reset handler to None for next iteration if we created it
        if handler is None:
            handler = None


@beartype
def get_active_party() -> Optional["Party"]:
    """Get the currently active Party context, or None if not in a context."""
    return _active_party


@beartype
def init(
    genetic_code: Union[str, dict] = "standard",
    log_level: Optional[str] = None,
) -> "Party":
    """Initialize (or reset) the default Party, clearing all registered pools/operations/regions.

    Parameters
    ----------
    genetic_code : Union[str, dict]
        Genetic code to use for ORF operations.
    log_level : Optional[str]
        If provided, configure logging at this level ("DEBUG", "INFO", "WARNING", "ERROR").
    """
    global _active_party, _default_party
    # Configure logging if requested
    if log_level is not None:
        configure_logging(level=log_level)
    # Exit current default party if active
    if _default_party is not None and _default_party._is_active:
        _default_party._state_manager.__exit__(None, None, None)
        _default_party._is_active = False
    # Create new default party
    _default_party = Party(genetic_code=genetic_code)
    _default_party._state_manager.__enter__()
    _default_party._is_active = True
    _active_party = _default_party
    logger.info("Initialized default Party")
    return _default_party


def _init_default_party() -> None:
    """Initialize the default party on module import (called from __init__.py)."""
    global _default_party
    if _default_party is None:
        init()


@beartype
def clear_pools() -> None:
    """Clear all pools, operations, and regions from the active Party without resetting configuration or genetic code."""
    party = get_active_party()
    if party is None:
        raise RuntimeError("No active Party context.")
    party.clear_pools()


@beartype
def set_genetic_code(genetic_code: Union[str, dict]) -> None:
    """Set the genetic code on the active Party.

    Args:
        genetic_code: Either 'standard' or a dict mapping amino acids to codon lists,
            e.g. ``{"M": ["ATG"], "F": ["TTC", "TTT"], ...}``.
            For custom dicts, the order of codons in each list matters:
            operations using ``codon_selection="first"`` or ``mutation_type="*_first"``
            will pick the first codon in each list.

    Raises:
        RuntimeError: If no active Party context exists.
    """
    party = get_active_party()
    if party is None:
        raise RuntimeError("No active Party context.")
    party.set_genetic_code(genetic_code)


@beartype
class Party:
    """Context manager for building and executing sequence libraries."""

    def __init__(
        self,
        genetic_code: Union[str, dict] = "standard",
    ) -> None:
        self._operations: list = []
        self._outputs: dict[str, Pool_type] = {}
        self._is_active: bool = False
        self._previous_party: Optional[Party] = None
        self._state_manager: st.Manager = st.Manager()
        self._next_pool_id: int = 0
        self._next_op_id: int = 0
        self._next_region_id: int = 0
        # Track pools and operations by ID (list) and name (dict)
        self._pools_by_id: list[Pool_type] = []
        self._ops_by_id: list[Operation_type] = []
        self._pools_by_name: dict[str, Pool_type] = {}
        self._ops_by_name: dict[str, Operation_type] = {}
        # Track regions by ID (list) and name (dict)
        self._regions_by_id: list[Region] = []
        self._regions_by_name: dict[str, Region] = {}
        # Build codon table for ORF operations
        self._codon_table: CodonTable = CodonTable(genetic_code)
        # Configuration for library output
        from .config import Config

        self._config: Config = Config()

    def _get_next_pool_id(self) -> int:
        """Get the next unique pool ID."""
        id_ = self._next_pool_id
        self._next_pool_id += 1
        return id_

    def _get_next_op_id(self) -> int:
        """Get the next unique operation ID."""
        id_ = self._next_op_id
        self._next_op_id += 1
        return id_

    @property
    def state_manager(self) -> st.Manager:
        """Access the statetracker Manager for debugging state iteration."""
        return self._state_manager

    @property
    def codon_table(self) -> CodonTable:
        """Access the CodonTable for ORF operations."""
        return self._codon_table

    @property
    def suppress_styles(self) -> bool:
        """Return True if inline styles are suppressed."""
        return self._config.suppress_styles

    @property
    def suppress_cards(self) -> bool:
        """Return True if design cards are suppressed."""
        return self._config.suppress_cards

    def set_genetic_code(self, genetic_code: Union[str, dict]) -> None:
        """Set or change the genetic code used for ORF operations."""
        self._codon_table = CodonTable(genetic_code)

    def get_effective_seq_length(self, seq: str) -> int:
        """Get effective sequence length (DNA characters only, excluding markers)."""
        return dna_utils.get_seq_length(seq)

    def get_length_without_tags(self, seq: str) -> int:
        """Get sequence length excluding only region tags (includes all chars)."""
        return dna_utils.get_length_without_tags(seq)

    def get_molecular_positions(self, seq: str) -> list[int]:
        """Get raw string positions of valid DNA characters, excluding marker interiors."""
        return dna_utils.get_molecular_positions(seq)

    def __enter__(self) -> "Party":
        """Enter the Party context, saving any previous active party."""
        global _active_party
        # Save previous party to restore on exit
        self._previous_party = _active_party
        _active_party = self
        self._is_active = True
        self._state_manager.__enter__()
        logger.info("Entered Party context")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the Party context, restoring the previous party."""
        global _active_party
        self._state_manager.__exit__(exc_type, exc_val, exc_tb)
        self._is_active = False
        # Restore previous party (could be default or another explicit party)
        _active_party = self._previous_party
        self._previous_party = None
        logger.info("Exited Party context")

    def _validate_pool_name(self, name: str, pool: Optional[Pool_type] = None) -> str:
        """Validate that a pool name is unique."""
        existing = self._pools_by_name.get(name)
        if existing is not None and existing is not pool:
            raise ValueError(f"Pool name '{name}' already exists")
        return name

    def _validate_op_name(self, name: str, op: Optional[Operation_type] = None) -> str:
        """Validate that an operation name is unique."""
        existing = self._ops_by_name.get(name)
        if existing is not None and existing is not op:
            raise ValueError(f"Operation name '{name}' already exists")
        return name

    def _register_pool(self, pool: Pool_type) -> None:
        """Register a pool with this party."""
        self._pools_by_id.append(pool)
        self._pools_by_name[pool.name] = pool
        logger.debug(
            "Registered pool id=%s name=%s num_states=%s", pool._id, pool.name, pool.num_states
        )

    def _update_pool_name(self, pool: Pool_type, old_name: str, new_name: str) -> None:
        """Update a pool's name in the tracking dict."""
        if old_name in self._pools_by_name:
            del self._pools_by_name[old_name]
        self._pools_by_name[new_name] = pool

    def _register_operation(self, operation: Operation_type) -> None:
        """Register an operation with this party."""
        if operation not in self._operations:
            self._operations.append(operation)
        self._ops_by_id.append(operation)
        self._ops_by_name[operation.name] = operation
        logger.debug(
            "Registered operation id=%s name=%s mode=%s",
            operation._id,
            operation.name,
            operation.mode,
        )

    def _update_op_name(self, op: Operation_type, old_name: str, new_name: str) -> None:
        """Update an operation's name in the tracking dict."""
        if old_name in self._ops_by_name:
            del self._ops_by_name[old_name]
        self._ops_by_name[new_name] = op

    def get_pool_by_id(self, id_: int) -> Pool_type:
        """Get a pool by its ID."""
        return self._pools_by_id[id_]

    def get_pool_by_name(self, name: str) -> Pool_type:
        """Get a pool by its name."""
        return self._pools_by_name[name]

    def get_op_by_id(self, id_: int) -> Operation_type:
        """Get an operation by its ID."""
        return self._ops_by_id[id_]

    def get_op_by_name(self, name: str) -> Operation_type:
        """Get an operation by its name."""
        return self._ops_by_name[name]

    def _get_next_region_id(self) -> int:
        """Get the next unique region ID."""
        id_ = self._next_region_id
        self._next_region_id += 1
        return id_

    def register_region(self, name: str, seq_length: Optional[int]) -> Region:
        """
        Register a region with this party.

        If a region with the same name already exists:
        - If it has the same seq_length, return the existing region
        - If it has a different seq_length, raise ValueError

        Parameters
        ----------
        name : str
            The region name.
        seq_length : Optional[int]
            The expected content length (None for variable, 0 for zero-length).

        Returns
        -------
        Region
            The registered region (existing or newly created).

        Raises
        ------
        ValueError
            If a region with the same name but different seq_length exists.
        """
        existing = self._regions_by_name.get(name)
        if existing is not None:
            if existing.seq_length == seq_length:
                return existing
            else:
                # Format lengths for error message
                existing_len = (
                    "variable" if existing.seq_length is None else str(existing.seq_length)
                )
                new_len = "variable" if seq_length is None else str(seq_length)
                hint = ""
                if name.startswith("_rep_") or name.startswith("_ins_"):
                    hint = (
                        " This commonly happens when calling insertion_multiscan "
                        "or replacement_multiscan multiple times with different-"
                        "sized pools in the same Party. Use the `names=` parameter "
                        "to assign unique region names to each call."
                    )
                raise ValueError(
                    f"Region '{name}' already registered with seq_length={existing_len}, "
                    f"cannot re-register with seq_length={new_len}. "
                    f"Region lengths must be consistent within a Party.{hint}"
                )

        # Create and register new region
        region = Region(name=name, seq_length=seq_length, _id=self._get_next_region_id())
        self._regions_by_id.append(region)
        self._regions_by_name[name] = region
        logger.debug("Registered region id=%s name=%s seq_length=%s", region._id, name, seq_length)
        return region

    def register_orf_region(
        self, name: str, seq_length: Optional[int], frame: int = 1
    ) -> OrfRegion:
        """
        Register an ORF region with this party.

        If a region with the same name already exists:
        - If it's an OrfRegion with same seq_length and frame, return it
        - Otherwise raise ValueError

        Parameters
        ----------
        name : str
            The region name.
        seq_length : Optional[int]
            The expected content length (None for variable, 0 for zero-length).
        frame : int
            Reading frame (+1, +2, +3, -1, -2, -3). Default +1.

        Returns
        -------
        OrfRegion
            The registered ORF region.
        """
        existing = self._regions_by_name.get(name)
        if existing is not None:
            if isinstance(existing, OrfRegion):
                if existing.seq_length == seq_length and existing.frame == frame:
                    return existing
                else:
                    raise ValueError(
                        f"OrfRegion '{name}' already registered with different attributes. "
                        f"Existing: seq_length={existing.seq_length}, frame={existing.frame}. "
                        f"Requested: seq_length={seq_length}, frame={frame}."
                    )
            else:
                raise ValueError(
                    f"Region '{name}' already exists as a plain Region. "
                    f"Use upgrade_to_orf_region() to convert it to an OrfRegion."
                )

        # Create and register new ORF region
        orf_region = OrfRegion(
            name=name, seq_length=seq_length, _id=self._get_next_region_id(), frame=frame
        )
        self._regions_by_id.append(orf_region)
        self._regions_by_name[name] = orf_region
        logger.debug(
            "Registered ORF region id=%s name=%s seq_length=%s frame=%s",
            orf_region._id,
            name,
            seq_length,
            frame,
        )
        return orf_region

    def upgrade_to_orf_region(self, name: str, frame: int = 1) -> OrfRegion:
        """
        Upgrade an existing plain Region to an OrfRegion.

        Only valid if the existing region is a plain Region (not already an OrfRegion).

        Parameters
        ----------
        name : str
            The name of the existing region to upgrade.
        frame : int
            Reading frame for the ORF (+1, +2, +3, -1, -2, -3). Default +1.

        Returns
        -------
        OrfRegion
            The upgraded ORF region.

        Raises
        ------
        ValueError
            If region doesn't exist or is already an OrfRegion.
        """
        existing = self._regions_by_name.get(name)
        if existing is None:
            raise ValueError(f"Region '{name}' not found. Cannot upgrade non-existent region.")

        if isinstance(existing, OrfRegion):
            raise ValueError(
                f"Region '{name}' is already an OrfRegion with frame={existing.frame}. "
                f"Cannot change frame of an existing OrfRegion."
            )

        # Create new OrfRegion with same name/seq_length but new frame
        # Keep the same _id for consistency
        orf_region = OrfRegion(
            name=existing.name,
            seq_length=existing.seq_length,
            _id=existing._id,
            frame=frame,
        )

        # Replace in registry (by name only, _regions_by_id keeps original order)
        self._regions_by_name[name] = orf_region
        # Update in the id list as well
        for i, r in enumerate(self._regions_by_id):
            if r._id == existing._id:
                self._regions_by_id[i] = orf_region
                break

        logger.debug("Upgraded region '%s' to OrfRegion with frame=%s", name, frame)
        return orf_region

    def get_region_by_id(self, id_: int) -> Region:
        """Get a region by its ID."""
        if id_ < 0 or id_ >= len(self._regions_by_id):
            raise ValueError(f"No region with ID {id_}")
        return self._regions_by_id[id_]

    def get_region_by_name(self, name: str) -> Region:
        """Get a region by its name."""
        region = self._regions_by_name.get(name)
        if region is None:
            available = list(self._regions_by_name.keys())
            if available:
                raise ValueError(f"Region '{name}' not found. Available: {available}")
            else:
                raise ValueError(f"Region '{name}' not found. No regions registered.")
        return region

    def get_region(self, name: str) -> Region:
        """Get a registered region by name. Alias for get_region_by_name."""
        return self.get_region_by_name(name)

    def has_region(self, name: str) -> bool:
        """Check if a region with the given name is registered."""
        return name in self._regions_by_name

    def clear_pools(self) -> None:
        """Clear all pools, operations, and regions without resetting configuration or genetic code.

        Unlike init(), this preserves:
        - Configuration settings (_config)
        - Genetic code settings (_codon_table)
        """
        # Clear pool tracking
        self._pools_by_id.clear()
        self._pools_by_name.clear()
        # Clear operation tracking
        self._operations.clear()
        self._ops_by_id.clear()
        self._ops_by_name.clear()
        # Clear region tracking
        self._regions_by_id.clear()
        self._regions_by_name.clear()
        # Reset ID counters
        self._next_pool_id = 0
        self._next_op_id = 0
        self._next_region_id = 0
        # Clear outputs
        self._outputs.clear()
        # Reset counter manager to clear counter state
        if self._is_active:
            self._state_manager.__exit__(None, None, None)
            self._state_manager = st.Manager()
            self._state_manager.__enter__()
        else:
            self._state_manager = st.Manager()

    def output(self, pool: Pool_type, name: Optional[str] = None) -> None:
        """Mark a pool as an output of this library."""
        if name is None:
            name = pool.name or f"output_{len(self._outputs)}"
        self._outputs[name] = pool

    def __repr__(self) -> str:
        return f"Party(outputs={list(self._outputs.keys())})"

    def print_graph(self, style: str = "clean") -> None:
        """Print an ASCII tree visualization of the Pool-Operation computation graph.

        Shows pools (places) with parentheses and operations (transitions) with brackets,
        similar to a Petri net diagram. Root pools (not consumed by other operations)
        are printed first, with their upstream DAGs.

        Args:
            style: Display style - 'clean' (default), 'minimal', or 'repr'.
                - 'clean': Shows names with key attributes
                    Pool: (name) pool: n=num_states
                    Op: [name] op: factory_name, mode, n=num_states
                - 'minimal': Shows just names
                    Pool: (name)
                    Op: [name]
                - 'repr': Shows full repr() of each object
        """
        from .text_viz import print_pool_graph

        print_pool_graph(self._pools_by_id, self._ops_by_id, style=style)


def cards_suppressed() -> bool:
    """Return True if design cards are suppressed in the active party."""
    party = get_active_party()
    return party.suppress_cards if party else False
