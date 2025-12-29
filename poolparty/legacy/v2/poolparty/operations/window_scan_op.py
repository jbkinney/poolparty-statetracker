"""WindowScan operation - scan windows across a sequence and apply transforms."""

import inspect
import numpy as np
import pandas as pd

from ..types import Union, Optional, ModeType, Sequence, Callable, beartype
from ..operation import Operation
from ..pool import Pool


#########################################################
# SlotOp Class
#########################################################

@beartype
class SlotOp(Operation):
    """Mutable slot that provides batch content to child operations."""
    design_card_keys = ['seq', 'slot_index']
    
    def __init__(
        self,
        mode: ModeType = 'fixed',
        name: Optional[str] = None,
        design_card_keys: Optional[Sequence[str]] = None,
    ):
        self._contents: list[str] = []
        super().__init__(
            parent_pools=[],
            num_states=1,
            mode=mode,
            seq_length=None,
            name=name or 'slot',
            design_card_keys=design_card_keys,
        )
    
    def set_contents(self, contents: Sequence[str]) -> None:
        """Inject batch of contents for processing."""
        self._contents = list(contents)
        self.num_states = len(contents) if contents else 1
        if contents:
            self.seq_length = len(contents[0])
    
    def compute_results(
        self, 
        input_strings_lists: Sequence[Sequence[str]], 
        sequential_states: Sequence[int],
    ) -> None:
        rows = []
        for i, state in enumerate(sequential_states):
            idx = i % len(self._contents) if self._contents else 0
            rows.append({
                'seq': self._contents[idx] if self._contents else '',
                'slot_index': idx,
            })
        self._results_df = pd.DataFrame(rows)
    
    def compute_results_row(
        self, 
        input_strings: Sequence[str], 
        sequential_state: int,
    ) -> dict:
        idx = sequential_state % len(self._contents) if self._contents else 0
        return {
            'seq': self._contents[idx] if self._contents else '',
            'slot_index': idx,
        }


#########################################################
# WindowScanOp Class
#########################################################

@beartype
class WindowScanOp(Operation):
    """Scan windows across a background sequence and apply transforms."""
    design_card_keys = ['position', 'window_size', 'transform_idx', 
                        'original_window', 'transformed_window', 'seq']
    
    #########################################################
    # Constructor
    #########################################################
    
    def __init__(
        self,
        parent: Pool,
        window_size: int,
        transform: Union[Callable[[str], str], Callable[[Pool], Pool]],
        positions: Sequence[int],
        position_probs: Optional[Sequence[float]],
        num_transforms_per_window: int,
        transform_seed: Optional[int],
        mode: ModeType,
        name: Optional[str],
        design_card_keys: Optional[Sequence[str]],
    ):
        self.window_size = window_size
        self.positions = list(positions)
        self.position_probs = position_probs
        self.num_transforms_per_window = num_transforms_per_window
        self.transform_seed = transform_seed
        self.transform = transform
        
        # Detect transform type
        self._is_pool_transform = self._detect_pool_transform(transform)
        
        if self._is_pool_transform:
            self._slot_op = SlotOp()
            self._slot_pool = Pool(self._slot_op)
            self._slot_op.set_contents(['N' * window_size])
            self._transform_pool = transform(self._slot_pool)
            self._string_transform = None
            
            if self._transform_pool.sequential_ops:
                self._num_transform_states = self._transform_pool.num_sequential_states
                self._transform_is_sequential = True
            else:
                self._num_transform_states = num_transforms_per_window
                self._transform_is_sequential = False
        else:
            self._slot_op = None
            self._slot_pool = None
            self._transform_pool = None
            self._string_transform = transform
            self._num_transform_states = num_transforms_per_window
            self._transform_is_sequential = False
        
        num_states = len(positions) * self._num_transform_states
        
        super().__init__(
            parent_pools=[parent],
            num_states=num_states,
            mode=mode,
            seq_length=None,
            name=name or 'window_scan',
            design_card_keys=design_card_keys,
        )
    
    #########################################################
    # Helper methods
    #########################################################
    
    def _detect_pool_transform(self, transform) -> bool:
        """Detect if transform is Pool-based or string-based."""
        # Try type hints first
        try:
            sig = inspect.signature(transform)
            params = list(sig.parameters.values())
            if params:
                ann_str = str(params[0].annotation)
                if 'Pool' in ann_str:
                    return True
                if 'str' in ann_str:
                    return False
        except (ValueError, TypeError):
            pass
        
        # Test with string - if works, it's a string transform
        try:
            result = transform('ACGT')
            if isinstance(result, str):
                return False
        except (TypeError, AttributeError):
            pass
        
        # Test with Pool
        try:
            dummy_slot_op = SlotOp()
            dummy_slot = Pool(dummy_slot_op)
            dummy_slot_op.set_contents(['ACGT'])
            result = transform(dummy_slot)
            return isinstance(result, Pool)
        except:
            return False
    
    #########################################################
    # Results computation
    #########################################################
    
    def compute_results(
        self, 
        input_strings_lists: Sequence[Sequence[str]], 
        sequential_states: Sequence[int],
    ) -> None:
        num_seqs = len(sequential_states)
        background_seqs = input_strings_lists[0] if input_strings_lists else [''] * num_seqs
        
        # Decompose states into (position_idx, transform_idx) pairs
        indices = []
        for state in sequential_states:
            idx = int(state) % self.num_states
            pos_idx = idx // self._num_transform_states
            transform_idx = idx % self._num_transform_states
            indices.append((pos_idx, transform_idx))
        
        # Extract windows
        windows = []
        positions_used = []
        for i, (bg_seq, (pos_idx, _)) in enumerate(zip(background_seqs, indices)):
            pos = self.positions[pos_idx]
            positions_used.append(pos)
            windows.append(bg_seq[pos:pos + self.window_size])
        
        # Apply transforms
        if self._is_pool_transform:
            self._slot_op.set_contents(windows)
            transform_indices = [idx[1] for idx in indices]
            
            if self._transform_is_sequential:
                self._transform_pool.set_all_sequential_op_states(0, num_seqs)
                for op in self._transform_pool.sequential_ops:
                    op.states = np.array(transform_indices) % op.num_states
            else:
                if self.transform_seed is not None:
                    seed = hash((transform_indices[0], self.transform_seed)) & 0x7FFFFFFF
                else:
                    seed = transform_indices[0] if transform_indices else 0
                self._transform_pool.set_random_op_seeds(seed)
            
            transformed_seqs = self._transform_pool.operation.run(num_seqs)
            transform_cards_df = self._transform_pool._collate_results()
        else:
            transformed_seqs = [self._string_transform(w) for w in windows]
            transform_cards_df = None
        
        # Build results
        rows = []
        for i, (bg_seq, (pos_idx, transform_idx)) in enumerate(zip(background_seqs, indices)):
            pos = positions_used[i]
            window = windows[i]
            transformed = transformed_seqs[i]
            seq = bg_seq[:pos] + transformed + bg_seq[pos + self.window_size:]
            
            row = {
                'seq': seq,
                'position': pos,
                'window_size': self.window_size,
                'transform_idx': transform_idx,
                'original_window': window,
                'transformed_window': transformed,
            }
            
            if transform_cards_df is not None and len(transform_cards_df) > i:
                for col in transform_cards_df.columns:
                    if col != 'seq':
                        row[f'transform.{col}'] = transform_cards_df.iloc[i][col]
            
            rows.append(row)
        
        self._results_df = pd.DataFrame(rows)
        
        if transform_cards_df is not None:
            for col in transform_cards_df.columns:
                if col != 'seq':
                    key = f'transform.{col}'
                    if key not in self.active_design_card_keys:
                        self.active_design_card_keys.append(key)
    
    def compute_results_row(
        self, 
        input_strings: Sequence[str], 
        sequential_state: int,
    ) -> dict:
        bg_seq = input_strings[0]
        
        idx = int(sequential_state) % self.num_states
        pos_idx = idx // self._num_transform_states
        transform_idx = idx % self._num_transform_states
        pos = self.positions[pos_idx]
        window = bg_seq[pos:pos + self.window_size]
        
        if self._is_pool_transform:
            self._slot_op.set_contents([window])
            if self._transform_is_sequential:
                self._transform_pool.set_state(transform_idx)
            else:
                if self.transform_seed is not None:
                    seed = hash((transform_idx, self.transform_seed)) & 0x7FFFFFFF
                else:
                    seed = transform_idx
                self._transform_pool.set_random_op_seeds(seed)
            transformed = self._transform_pool.seq
        else:
            transformed = self._string_transform(window)
        
        seq = bg_seq[:pos] + transformed + bg_seq[pos + self.window_size:]
        
        return {
            'seq': seq,
            'position': pos,
            'window_size': self.window_size,
            'transform_idx': transform_idx,
            'original_window': window,
            'transformed_window': transformed,
        }


#########################################################
# Factory functions
#########################################################

@beartype
def window_scan(
    background: Union[Pool, str],
    window_size: int,
    transform: Union[Callable[[str], str], Callable[[Pool], Pool]],
    start: Optional[int] = None,
    end: Optional[int] = None,
    step_size: Optional[int] = None,
    positions: Optional[Sequence[int]] = None,
    position_probs: Optional[Sequence[float]] = None,
    num_transforms_per_window: int = 1,
    transform_seed: Optional[int] = None,
    mode: ModeType = 'random',
    name: str = 'window_scan',
    design_card_keys: Optional[Sequence[str]] = None,
) -> Pool:
    """Scan windows across a background sequence and apply transforms.
    
    Args:
        background: Background sequence (string or Pool) to scan across
        window_size: Size of window to extract (0 = insert mode)
        transform: Callable[[str], str] or Callable[[Pool], Pool]
        start/end/step_size: Range-based position interface
        positions/position_probs: Position-based interface (mutually exclusive)
        num_transforms_per_window: Samples per position (for random transforms)
        transform_seed: Seed for reproducible transforms
        mode: 'random' or 'sequential'
    """
    from .from_seqs_op import from_seqs
    
    if isinstance(background, str):
        parent = from_seqs([background], design_card_keys=[])
    else:
        parent = background
    
    # Validate position interface
    range_provided = any(p is not None for p in [start, end, step_size])
    if range_provided and positions is not None:
        raise ValueError("Cannot specify both range-based and position-based parameters.")
    
    # Compute positions
    seq_len = parent.seq_length
    if positions is not None:
        computed_positions = list(positions)
    else:
        start_val = start if start is not None else 0
        end_val = end if end is not None else seq_len
        step_val = step_size if step_size is not None else 1
        
        if window_size == 0:
            computed_positions = list(range(start_val, min(end_val, seq_len) + 1, step_val))
        else:
            computed_positions = list(range(start_val, min(end_val, seq_len) - window_size + 1, step_val))
    
    if len(computed_positions) == 0:
        raise ValueError("No valid positions for given parameters")
    
    # Validate positions
    for pos in computed_positions:
        if window_size == 0:
            if pos < 0 or pos > seq_len:
                raise ValueError(f"Insert position {pos} out of bounds [0, {seq_len}]")
        else:
            if pos < 0 or pos + window_size > seq_len:
                raise ValueError(f"Window at position {pos} exceeds sequence length")
    
    return Pool(
        operation=WindowScanOp(
            parent=parent,
            window_size=window_size,
            transform=transform,
            positions=computed_positions,
            position_probs=position_probs,
            num_transforms_per_window=num_transforms_per_window,
            transform_seed=transform_seed,
            mode=mode,
            name=name,
            design_card_keys=design_card_keys,
        )
    )


@beartype
def shuffle_scan(
    background: Union[Pool, str],
    shuffle_size: int,
    num_shuffles: int = 1,
    start: Optional[int] = None,
    end: Optional[int] = None,
    step_size: Optional[int] = None,
    positions: Optional[Sequence[int]] = None,
    position_probs: Optional[Sequence[float]] = None,
    seed: Optional[int] = None,
    mode: ModeType = 'random',
    name: str = 'shuffle_scan',
    design_card_keys: Optional[Sequence[str]] = None,
) -> Pool:
    """Scan windows and shuffle characters within each window."""
    rng = np.random.default_rng(seed)
    
    def shuffle_transform(window: str) -> str:
        chars = list(window)
        rng.shuffle(chars)
        return ''.join(chars)
    
    return window_scan(
        background=background,
        window_size=shuffle_size,
        transform=shuffle_transform,
        start=start, end=end, step_size=step_size,
        positions=positions, position_probs=position_probs,
        num_transforms_per_window=num_shuffles,
        transform_seed=seed,
        mode=mode, name=name,
        design_card_keys=design_card_keys,
    )


@beartype
def deletion_scan(
    background: Union[Pool, str],
    deletion_size: int,
    mark_deletions: bool = True,
    start: Optional[int] = None,
    end: Optional[int] = None,
    step_size: Optional[int] = None,
    positions: Optional[Sequence[int]] = None,
    position_probs: Optional[Sequence[float]] = None,
    mode: ModeType = 'random',
    name: str = 'deletion_scan',
    design_card_keys: Optional[Sequence[str]] = None,
) -> Pool:
    """Scan windows and delete (mark with '-' or remove) each window."""
    transform = (lambda w: '-' * len(w)) if mark_deletions else (lambda w: '')
    
    return window_scan(
        background=background,
        window_size=deletion_size,
        transform=transform,
        start=start, end=end, step_size=step_size,
        positions=positions, position_probs=position_probs,
        num_transforms_per_window=1,
        mode=mode, name=name,
        design_card_keys=design_card_keys,
    )


@beartype
def insertion_scan(
    background: Union[Pool, str],
    insert: str,
    overwrite: bool = True,
    start: Optional[int] = None,
    end: Optional[int] = None,
    step_size: Optional[int] = None,
    positions: Optional[Sequence[int]] = None,
    position_probs: Optional[Sequence[float]] = None,
    mode: ModeType = 'random',
    name: str = 'insertion_scan',
    design_card_keys: Optional[Sequence[str]] = None,
) -> Pool:
    """Scan positions and insert a sequence (overwrite or insert mode)."""
    window_size = len(insert) if overwrite else 0
    
    return window_scan(
        background=background,
        window_size=window_size,
        transform=lambda w: insert,
        start=start, end=end, step_size=step_size,
        positions=positions, position_probs=position_probs,
        num_transforms_per_window=1,
        mode=mode, name=name,
        design_card_keys=design_card_keys,
    )
