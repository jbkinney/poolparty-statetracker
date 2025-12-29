"""Manager - Context manager for handling Counter objects."""
from IPython.display import display
import pandas as pd

from .utils import clean_df_int_columns


class Manager:
    """Context manager for handling Counter objects."""
    _active_manager = None
    
    def __init__(self):
        """Initialize empty counter manager."""
        self._counters = []
        self._next_id = 0
    
    def __enter__(self):
        """Enter context and set as active manager."""
        Manager._active_manager = self
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and clear active manager."""
        Manager._active_manager = None
        return False
    
    def register(self, counter):
        """Register a counter with this manager."""
        counter._id = self._next_id
        self._next_id += 1
        if counter._name is None:
            counter._name = f'Counter[{counter._id}]'
        self._counters.append(counter)
    
    def get_ancestors(self, counter, visited=None):
        """Recursively collect all ancestor counters in the computation graph."""
        if visited is None:
            visited = []
        if counter in visited:
            return visited
        visited.append(counter)
        for parent in counter._parents:
            self.get_ancestors(parent, visited)
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
    
    def get_iteration_df(self, iter_counter, counters=None):
        """Return DataFrame showing state of counters as iter_counter is iterated."""
        targets = counters if counters is not None else self._counters
        col_names = [f"{ctr.name}" for ctr in targets]
        self.inactivate_all()
        iter_counter.reset()
        rows = []
        for _ in iter_counter:
            row = [ctr.state for ctr in targets]
            rows.append(row)
        self.inactivate_all()
        df = pd.DataFrame(rows, columns=col_names)
        df.index.name = f"{iter_counter.name}"
        if iter_counter.name in df.columns:
            df = df.drop(columns=[iter_counter.name])
        df = clean_df_int_columns(df)
        return df

    def print_graph(self, style: str = 'clean'):
        """Print an ASCII tree visualization of the counter dependency graph.
        
        Args:
            style: Display style - 'clean' (default), 'minimal', or 'repr'.
        """
        from .text_viz import print_graph
        print_graph(self._counters, style=style)
