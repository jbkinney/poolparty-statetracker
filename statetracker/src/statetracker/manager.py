"""Manager - Context manager for handling State objects."""

import logging

logger = logging.getLogger(__name__)

try:
    from IPython.display import display
except ImportError:
    display = print
import pandas as pd

from .utils import clean_df_int_columns


class Manager:
    """Context manager for handling State objects."""

    _active_manager = None

    def __init__(self):
        """Initialize empty state manager."""
        self._states = []
        self._next_id = 0

    def __enter__(self):
        """Enter context and set as active manager, saving any previous."""
        self._previous_manager = Manager._active_manager
        Manager._active_manager = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore previous active manager."""
        Manager._active_manager = self._previous_manager
        self._previous_manager = None
        return False

    def register(self, state):
        """Register a state with this manager."""
        state._id = self._next_id
        self._next_id += 1
        if state._name is None:
            state._name = f"State[{state._id}]"
        self._states.append(state)
        logger.debug(
            "Registered state id=%s name=%s num_values=%s", state._id, state._name, state.num_values
        )

    def get_ancestors(self, state, visited=None):
        """Recursively collect all ancestor states and sync group peers."""
        if visited is None:
            visited = []
            logger.debug("Getting ancestors for state id=%s name=%s", state._id, state._name)
        if state in visited:
            return visited
        visited.append(state)
        # Include sync group peers and recurse into their parents too
        for peer in state._synced_group:
            if peer not in visited and peer is not state:
                self.get_ancestors(peer, visited)
        # Then recurse into computational parents
        for parent in state._parents:
            self.get_ancestors(parent, visited)
        return visited

    def clear_all_values(self):
        """Directly clear all states and sync groups to None (bypasses setter)."""
        logger.debug("Clearing all values for %d states", len(self._states))
        seen_groups = set()
        for state in self._states:
            state._value = None
            group = state._synced_group
            if id(group) not in seen_groups:
                group._value = None
                seen_groups.add(id(group))

    def inactivate_all(self, states=None):
        """Set states to inactive value (None)."""
        targets = states if states is not None else self._states
        for state in targets:
            state.value = None

    def reset_all(self, states=None):
        """Reset states to value 0."""
        targets = states if states is not None else self._states
        for state in targets:
            state.reset()

    def get_all_names(self):
        """Return list of names of all registered states."""
        return [s.name for s in self._states]

    def get_by_name(self, name):
        """Return state by name."""
        for state in self._states:
            if state.name == name:
                return state
        raise KeyError(f"No state with name '{name}' found")

    def get_iteration_df(self, iter_state, states=None):
        """Return DataFrame showing value of states as iter_state is iterated."""
        targets = states if states is not None else self._states
        col_names = [f"{s.name}" for s in targets]
        self.inactivate_all()
        iter_state.reset()
        rows = []
        for _ in iter_state:
            row = [s.value for s in targets]
            rows.append(row)
        self.inactivate_all()
        df = pd.DataFrame(rows, columns=col_names)
        df.index.name = f"{iter_state.name}"
        if iter_state.name in df.columns:
            df = df.drop(columns=[iter_state.name])
        df = clean_df_int_columns(df)
        return df

    def print_graph(self, style: str = "clean"):
        """Print an ASCII tree visualization of the state dependency graph.

        Args:
            style: Display style - 'clean' (default), 'minimal', or 'repr'.
        """
        from .text_viz import print_graph

        print_graph(self._states, style=style)
