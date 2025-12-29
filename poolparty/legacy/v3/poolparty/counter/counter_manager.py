"""
CounterManager for handling Counter objects.

Provides a context manager for registering and managing counters.
"""
import pandas as pd
from IPython.display import display


class CounterManager:
    """
    Context manager for handling Counter objects.
    
    Counters created within the context auto-register with the manager.
    
    Usage:
        with CounterManager() as mgr:
            A = Counter(2, name='A')
            B = Counter(3, name='B')
            C = A * B
            C.name = 'C'
            
            mgr.get_counter_names()       # ['A', 'B', 'C']
            mgr.get_counter('A')          # returns A
            mgr.reset_counters()          # all to state=0
            mgr.inactivate_counters([A])  # just A to state=-1
            
            df = mgr.get_iteration_df(C)  # DataFrame with states
    """
    
    _active_manager = None  # Class variable to track active context
    
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
        # Assign ID and name first (for use in error messages)
        counter._id = self._next_id
        self._next_id += 1
        if counter._name is None:
            counter._name = f'id_{counter._id}'
        
        # Validate tree structure - no counter should appear multiple times
        counts_dict = {}
        self._count_ancestors(counter, counts_dict)
        
        for counter_id, count in counts_dict.items():
            if count > 1:
                # counter._id is the index in self._counters
                dup_counter = self._counters[counter_id]
                raise ValueError(
                    f"Counter '{dup_counter.name}' appears {count} times in the "
                    f"computation graph for '{counter.name}'. Graph must be a tree."
                )
        
        self._counters.append(counter)
    
    def _count_ancestors(self, counter, counts_dict):
        """Recursively count occurrences of each counter in the computation graph."""
        counter_id = counter._id
        counts_dict[counter_id] = counts_dict.get(counter_id, 0) + 1
        
        for parent in counter._parents:
            self._count_ancestors(parent, counts_dict)
    
    def inactivate_all(self, counters=None):
        """
        Set counters to inactive state (-1).
        
        Parameters:
            counters: List of Counter objects to inactivate. 
                      If None, applies to all registered counters.
        """
        targets = counters if counters is not None else self._counters
        for counter in targets:
            counter.inactive()
    
    def reset_all(self, counters=None):
        """
        Reset counters to state 0.
        
        Parameters:
            counters: List of Counter objects to reset.
                      If None, applies to all registered counters.
        """
        targets = counters if counters is not None else self._counters
        for counter in targets:
            counter.reset()
    
    def get_all_names(self):
        """Return list of names of all registered counters."""
        return [c.name for c in self._counters]
    
    def get_by_name(self, name):
        """
        Return counter by name.
        
        Parameters:
            name: Name of the counter to retrieve.
            
        Returns:
            Counter object with matching name.
            
        Raises:
            KeyError: If no counter with the given name is found.
        """
        for counter in self._counters:
            if counter.name == name:
                return counter
        raise KeyError(f"No counter with name '{name}' found")
    
    def test_iteration(self, iter_counter, counters=None, display_df=True, return_df=True):
        """
        Return DataFrame showing state of counters as iter_counter is iterated.
        
        Parameters:
            iter_counter: The Counter to iterate over.
            counters: List of counters to show states for.
                      If None, uses all registered counters.
        
        Returns:
            pandas DataFrame with columns '{name}.state' for each counter,
            indexed by the iter_counter's state.
        """
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
        if display_df:
            display(df)
        if return_df:
            return df

    def print_graph(self):
        """
        Print an ASCII tree visualization of the counter dependency graph.
        
        Shows all root counters (counters that are not parents of any other counter)
        and their parent trees.
        """
        # Find root counters (those that are not parents of any other counter)
        parent_ids = set()
        for counter in self._counters:
            for parent in counter._parents:
                parent_ids.add(parent._id)
        
        roots = [c for c in self._counters if c._id not in parent_ids]
        
        if not roots:
            print("(no counters registered)")
            return
        
        # Print each root tree
        for i, root in enumerate(roots):
            root.print_tree()
            if i < len(roots) - 1:
                print()  # Blank line between separate trees

