"""CounterManager for handling Counter objects."""
from IPython.display import display
import pandas as pd

from ..utils import clean_df_int_columns


class CounterManager:
    """Context manager for handling Counter objects."""
    _active_manager = None
    
    def __init__(self):
        """Initialize empty counter manager."""
        self._counters = []
        self._next_id = 0
    
    def __enter__(self):
        """Enter context and set as active manager."""
        CounterManager._active_manager = self
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and clear active manager."""
        CounterManager._active_manager = None
        return False
    
    def register(self, counter):
        """Register a counter with this manager."""
        counter._id = self._next_id
        self._next_id += 1
        if counter._name is None:
            counter._name = f'id_{counter._id}'
        self._counters.append(counter)
    
    def _get_ancestors(self, counter, visited=None):
        """Recursively collect all ancestor counters in the computation graph."""
        if visited is None:
            visited = set()
        counter_id = counter._id
        if counter_id in visited:
            return visited
        visited.add(counter_id)
        for parent in counter._parents:
            self._get_ancestors(parent, visited)
        return visited
    
    def inactivate_all(self, counters=None):
        """Set counters to inactive state (None)."""
        targets = counters if counters is not None else self._counters
        for counter in targets:
            counter.state = None
    
    def reset_all(self, counters=None):
        """Reset counters to state 0."""
        targets = counters if counters is not None else self._counters
        for counter in targets:
            counter.reset()
    
    def get_all_names(self):
        """Return list of names of all registered counters."""
        return [c.name for c in self._counters]
    
    def get_by_name(self, name):
        """Return counter by name."""
        for counter in self._counters:
            if counter.name == name:
                return counter
        raise KeyError(f"No counter with name '{name}' found")
    
    def test_iteration(self, iter_counter, counters=None, display_df=True, return_df=True):
        """Return DataFrame showing state of counters as iter_counter is iterated."""
        targets = counters if counters is not None else self._counters
        col_names = [f"{ctr.name}.state" for ctr in targets]
        self.inactivate_all()
        iter_counter.reset()
        rows = []
        for _ in iter_counter:
            row = [ctr.state for ctr in targets]
            rows.append(row)
        self.inactivate_all()
        df = pd.DataFrame(rows, columns=col_names)
        df.index.name = f"{iter_counter.name}.state"
        df = clean_df_int_columns(df)
        if display_df:
            display(df)
        if return_df:
            return df

    def print_graph(self, style: str = 'clean'):
        """Print an ASCII tree visualization of the counter dependency graph.
        
        Args:
            style: Display style - 'clean' (default), 'minimal', or 'repr'.
        """
        from ..text_viz import print_counter_graph
        print_counter_graph(self._counters, style=style)
