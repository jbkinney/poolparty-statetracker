"""Pool class for poolparty."""

import logging

logger = logging.getLogger(__name__)

from typing import Literal

import pandas as pd

import statetracker as st

from .pool_mixins import (
    BaseOpsMixin,
    FixedOpsMixin,
    RegionOpsMixin,
    ScanOpsMixin,
    StateOpsMixin,
)
from .region import Region
from .types import Integral, Operation_type, Optional, Pool_type, Real, Sequence, Union, beartype


@beartype
class Pool(BaseOpsMixin, ScanOpsMixin, FixedOpsMixin, StateOpsMixin, RegionOpsMixin):
    """A node in the computation DAG."""

    def __init__(
        self,
        operation: Operation_type,
        name: Optional[str] = None,
        state: Optional[st.State] = None,
        iter_order: Optional[Real] = None,
        regions: Optional[set[Region]] = None,
    ) -> None:
        """Initialize Pool and build its state."""
        from .party import get_active_party
        from .region import Region

        party = get_active_party()
        if party is None:
            raise RuntimeError(
                "Pools must be created inside a Party context. Use: with pp.Party() as party: ..."
            )
        self._party = party
        self._id = party._get_next_pool_id()
        self.operation = operation
        if state is not None:
            self.state = state
        else:
            self.state: st.State | None = operation.build_pool_counter(operation.parent_pools)
        if iter_order is not None and self.state is not None:
            self.state.iter_order = iter_order
        self._name: str = ""
        self.name = name if name is not None else f"pool[{self._id}]"

        # Track regions: inherit from parents if not explicitly provided
        if regions is not None:
            self._regions: set[Region] = set(regions)
        else:
            # Inherit regions from all parent pools
            self._regions = set()
            for parent in operation.parent_pools:
                if hasattr(parent, "_regions"):
                    self._regions.update(parent._regions)

        # Register pool with party after name is set
        party._register_pool(self)
        logger.debug(
            "Created pool id=%s name=%s seq_length=%s num_states=%s",
            self._id,
            self._name,
            self.seq_length,
            self.num_states,
        )

    @property
    def iter_order(self) -> Real:
        """Iteration order for this pool."""
        if self.state is None:
            return 0
        return self.state.iter_order

    @iter_order.setter
    def iter_order(self, value: Real) -> None:
        """Set iteration order for this pool."""
        if self.state is not None:
            self.state.iter_order = value

    @property
    def name(self) -> str:
        """Name of this pool."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Set pool name and update state name.

        Validates name uniqueness with the Party before accepting.

        Raises:
            ValueError: If the name is already used by another pool.
        """
        # Validate name with party (excludes self for renaming case)
        self._party._validate_pool_name(value, self)
        old_name = self._name
        self._name = value
        # When pool.state is the same as operation.state (source operations),
        # preserve operation state name if operation has explicit name (not default)
        # Otherwise, use pool state name
        if self.state is not None:
            if self.state is self.operation.state:
                # Check if operation has explicit name (not default like "op[0]:from_seqs")
                op_name = self.operation.name
                is_default_op_name = op_name.startswith("op[") and "]:" in op_name
                if not is_default_op_name:
                    # Operation has explicit name, preserve it
                    # State name should already be set to operation name
                    pass
                else:
                    # Operation has default name, use pool name
                    self.state.name = f"{value}.state"
            else:
                # Different states, set pool state name normally
                self.state.name = f"{value}.state"
        # Update party's name tracking if this is a rename (not initial set)
        if old_name:
            self._party._update_pool_name(self, old_name, value)

    @property
    def num_states(self) -> int | None:
        """Number of states for this pool."""
        if self.state is None:
            return None
        return self.state.num_values

    @property
    def parents(self) -> list:
        """Get parent pools from the operation."""
        return self.operation.parent_pools

    @property
    def seq_length(self) -> Optional[int]:
        """Sequence length (None if variable)."""
        return self.operation.seq_length

    @property
    def regions(self) -> set[Region]:
        """Set of Region objects present in this pool's sequences."""
        return self._regions

    def has_region(self, name: str) -> bool:
        """Check if a region with the given name is present in this pool."""
        return any(r.name == name for r in self._regions)

    def add_region(self, region: Region) -> None:
        """Add a region to this pool's region set."""
        self._regions.add(region)

    def _untrack_region(self, name: str) -> None:
        """Remove a region from this pool's region set by name."""
        self._regions = {r for r in self._regions if r.name != name}

    #########################################################################
    # Counter-based operators
    #########################################################################

    def __add__(self, other: Pool_type) -> Pool_type:
        """Stack two pools (union of states via sum_counters)."""
        from .state_ops.stack import stack

        return stack([self, other])

    def __mul__(self, n: int) -> Pool_type:
        """Repeat this pool n times (repeat states)."""
        from .state_ops.repeat import repeat

        return repeat(self, n)

    def __rmul__(self, n: int) -> Pool_type:
        """Repeat this pool n times (repeat states)."""
        return self.__mul__(n)

    def __getitem__(self, key: Union[int, slice]) -> Pool_type:
        """Slice this pool's states (not sequences)."""
        from .state_ops.state_slice import state_slice

        return state_slice(self, key)

    def __repr__(self) -> str:
        num_states_str = "None" if self.num_states is None else str(self.num_states)
        return f"Pool(id={self._id}, name={self.name!r}, op={self.operation.name!r}, num_states={num_states_str})"

    def named(self, name: str, op_name: Optional[str] = None) -> Pool_type:
        """Set the name of this pool and its operation, return self for chaining."""
        self.name = name
        # self.operation.name = op_name if op_name is not None else name + '.op'
        return self

    def copy(self, name: Optional[str] = None) -> Pool_type:
        """Create a copy of this pool with a copied operation.

        The copied operation references the same parent_pools, so the copy
        represents a parallel branch in the computation graph that shares
        the same upstream DAG.

        Must be called within an active Party context.

        Args:
            name: Optional name for the copied pool. If None, uses
                self.name + '.copy' as the default.

        Returns:
            A new Pool backed by a copied Operation.
        """
        new_op = self.operation.copy()
        new_pool = Pool(operation=new_op)
        if name is not None:
            new_pool.name = name
        else:
            new_pool.name = self.name + ".copy"
        return new_pool

    def deepcopy(self, name: Optional[str] = None) -> Pool_type:
        """Create a deep copy of this pool, recursively copying the entire upstream DAG.

        Unlike copy(), this creates independent copies of all upstream pools
        and operations, resulting in a fully independent computation DAG.

        Must be called within an active Party context.

        Args:
            name: Optional name for the copied pool. If None, uses
                self.name + '.copy' as the default.

        Returns:
            A new Pool backed by a recursively copied Operation.
        """
        new_op = self.operation.deepcopy()
        new_pool = Pool(operation=new_op, name=name)
        return new_pool

    #########################################################################
    # Generation
    #########################################################################

    def generate_library(
        self,
        num_cycles: int = 1,
        num_seqs: Optional[int] = None,
        seed: Optional[int] = None,
        init_state: Optional[int] = None,
        seqs_only: bool = False,
        report_design_cards: bool = False,
        aux_pools: Sequence[Pool_type] = (),
        pools_to_report: Union[str, Sequence[Pool_type]] = "all",
        organize_columns_by: Literal["pool", "type"] = "type",
        _include_inline_styles: bool = False,
        discard_null_seqs: bool = False,
        max_iterations: Optional[int] = None,
        min_acceptance_rate: Optional[float] = None,
        attempts_per_rate_assessment: int = 100,
        # Deprecated parameters (for backwards compatibility)
        report_seq: Optional[bool] = None,
        report_pool_seqs: Optional[bool] = None,
        report_pool_states: Optional[bool] = None,
        report_op_states: Optional[bool] = None,
        report_op_keys: Optional[bool] = None,
    ) -> Union[pd.DataFrame, list[str]]:
        from .generate_library import generate_library

        # Handle deprecated parameters by temporarily modifying config
        party = self._party
        if any(
            arg is not None
            for arg in [
                report_seq,
                report_pool_seqs,
                report_pool_states,
                report_op_states,
                report_op_keys,
            ]
        ):
            # Save original config values
            orig_show_seq = party._config.show_seq
            orig_show_pool_seqs = party._config.show_pool_seqs
            orig_show_pool_states = party._config.show_pool_states
            orig_show_op_states = party._config.show_op_states
            orig_design_cards = party._config._design_cards.copy()

            # Apply deprecated parameters
            if report_seq is not None:
                party._config.show_seq = report_seq
            if report_pool_seqs is not None:
                party._config.show_pool_seqs = report_pool_seqs
            if report_pool_states is not None:
                party._config.show_pool_states = report_pool_states
            if report_op_states is not None:
                party._config.show_op_states = report_op_states
            if report_op_keys is not None and not report_op_keys:
                # Set all design cards to empty to filter out all keys
                party._config._design_cards = {
                    "from_seqs": set(),
                    "from_iupac": set(),
                    "from_motif": set(),
                    "get_kmers": set(),
                    "mutagenize": set(),
                    "recombine": set(),
                    "shuffle_seq": set(),
                    "mutagenize_orf": set(),
                    "region_scan": set(),
                    "region_multiscan": set(),
                    "repeat": set(),
                    "stack": set(),
                }

            try:
                return generate_library(
                    pool=self,
                    num_cycles=num_cycles,
                    num_seqs=num_seqs,
                    seed=seed,
                    init_state=init_state,
                    seqs_only=seqs_only,
                    report_design_cards=report_design_cards,
                    aux_pools=aux_pools,
                    pools_to_report=pools_to_report,
                    organize_columns_by=organize_columns_by,
                    _include_inline_styles=_include_inline_styles,
                    discard_null_seqs=discard_null_seqs,
                    max_iterations=max_iterations,
                    min_acceptance_rate=min_acceptance_rate,
                    attempts_per_rate_assessment=attempts_per_rate_assessment,
                )
            finally:
                # Restore original config values
                party._config.show_seq = orig_show_seq
                party._config.show_pool_seqs = orig_show_pool_seqs
                party._config.show_pool_states = orig_show_pool_states
                party._config.show_op_states = orig_show_op_states
                party._config._design_cards = orig_design_cards
        else:
            return generate_library(
                pool=self,
                num_cycles=num_cycles,
                num_seqs=num_seqs,
                seed=seed,
                init_state=init_state,
                seqs_only=seqs_only,
                report_design_cards=report_design_cards,
                aux_pools=aux_pools,
                pools_to_report=pools_to_report,
                organize_columns_by=organize_columns_by,
                _include_inline_styles=_include_inline_styles,
                discard_null_seqs=discard_null_seqs,
                max_iterations=max_iterations,
                min_acceptance_rate=min_acceptance_rate,
                attempts_per_rate_assessment=attempts_per_rate_assessment,
            )

    def print_library(
        self,
        num_seqs: Optional[Integral] = None,
        num_cycles: Optional[Integral] = None,
        show_header: bool = True,
        show_state: bool = True,
        show_name: bool = True,
        show_seq: bool = True,
        pad_names: bool = True,
        seed: Optional[Integral] = None,
        discard_null_seqs: bool = False,
        max_iterations: Optional[int] = None,
        min_acceptance_rate: Optional[float] = None,
        attempts_per_rate_assessment: int = 100,
    ) -> Pool_type:
        """Print preview sequences from this pool; returns self for chaining.

        Args:
            num_seqs: Number of sequences to generate.
            num_cycles: Number of complete iterations through all states.
            show_header: Whether to show the pool header line.
            show_state: Whether to show the state column.
            show_name: Whether to show the name column.
            show_seq: Whether to show the seq column.
            pad_names: Whether to pad names to align sequences.
            seed: Random seed for reproducibility.
            discard_null_seqs: If True, only show valid (non-null) sequences.
            max_iterations: Maximum iterations before stopping.
            min_acceptance_rate: Minimum fraction of sequences that must pass.
            attempts_per_rate_assessment: Iterations between acceptance rate checks.
        """
        # Build kwargs for generate_library, only including num_cycles when needed
        gen_kwargs = {
            "seqs_only": False,
            "report_design_cards": True,
            "init_state": 0,
            "seed": seed,
            "_include_inline_styles": True,
            "discard_null_seqs": discard_null_seqs,
            "max_iterations": max_iterations,
            "min_acceptance_rate": min_acceptance_rate,
            "attempts_per_rate_assessment": attempts_per_rate_assessment,
        }
        if num_seqs is not None:
            gen_kwargs["num_seqs"] = num_seqs
        else:
            gen_kwargs["num_cycles"] = num_cycles if num_cycles is not None else 1
        df = self.generate_library(**gen_kwargs)
        has_name = show_name and "name" in df.columns and df["name"].notna().any()
        max_name_len = df["name"].str.len().max() if has_name and pad_names else 0

        if show_header:
            num_states_str = "None" if self.num_states is None else str(self.num_states)
            print(f"{self.name}: seq_length={self.seq_length}, num_states={num_states_str}")
            # Build header columns
            header_parts = []
            if show_state:
                header_parts.append("state")
            if has_name:
                header_parts.append(f"{'name':<{max_name_len}}" if pad_names else "name")
            if show_seq:
                header_parts.append("seq")
            if header_parts:
                print("  ".join(header_parts))

        state_col = f"{self.name}.state"
        for _, row in df.iterrows():
            # Build row columns
            row_parts = []
            if show_state:
                row_parts.append(f"{row[state_col]:5d}")
            if has_name:
                name = row['name'] if row['name'] is not None else ""
                if pad_names:
                    row_parts.append(f"{name:<{max_name_len}}")
                else:
                    row_parts.append(f"{name}")
            if show_seq:
                seq = row["seq"]
                # Handle None (filtered) sequences
                if seq is None:
                    row_parts.append("None")
                else:
                    from .utils.style_utils import SeqStyle

                    # Get per-sequence inline styles (from operation style parameters)
                    inline_styles = row.get("_inline_styles", SeqStyle.empty(0))
                    # Apply inline styles if present
                    if inline_styles is not None:
                        seq = inline_styles.apply(seq)
                    row_parts.append(seq)
            print("  ".join(row_parts))
        print("")
        return self  # For chaining

    #########################################################################
    # Tree visualization
    #########################################################################

    def print_dag(self, style: str = "clean", show_pools: bool = True) -> Pool_type:
        """Print the ASCII tree visualization rooted at this pool."""
        from .text_viz import print_pool_tree

        print_pool_tree(self, style=style, show_pools=show_pools)
        return self  # For chaining

    #########################################################################
    # Operation methods provided by mixins:
    # - BaseOpsMixin: mutagenize, shuffle_seq, insert_from_iupac,
    #                 insert_from_motif, insert_kmers
    # - ScanOpsMixin: mutagenize_scan, deletion_scan, insertion_scan,
    #                 replacement_scan, shuffle_scan
    # - FixedOpsMixin: rc, swapcase, upper, lower, clear_gaps,
    #                  clear_annotation, stylize
    # - StateOpsMixin: repeat_states, sample_states, shuffle_states,
    #                  slice_states
    # - RegionOpsMixin: apply_at_region, insert_tags, remove_tags,
    #                   replace_region, clear_tags
    #########################################################################
